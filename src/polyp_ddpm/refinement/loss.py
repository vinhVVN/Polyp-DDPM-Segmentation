"""Multi-objective refinement loss functions: epsilon MSE, RGB restoration, LPIPS, boundary, and prior."""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn.functional as F

from .color import rgb_to_lab, tissue_support
from .degradation import protection_regions


def masked_mean(error: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Tính trung bình có trọng số trên các chiều không gian (C, H, W)."""
    return (error * weight).sum((1, 2, 3)) / (
        weight.sum((1, 2, 3)) * error.shape[1]
    ).clamp_min(1.0)


def region_weights(clean_rgb: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Phân tách trọng số vùng polyp (fg) và vùng nền nội soi (bg)."""
    mask = ((condition[:, :1] + 1.0) / 2.0).clamp(0.0, 1.0)
    valid = tissue_support(clean_rgb).detach()
    # Loại trừ viền đen ngoài khung nhìn (FOV transition)
    valid = (F.avg_pool2d(valid, 5, 1, 2) > 0.99).float()
    return mask * valid, (1.0 - mask) * valid


def balanced_error(error: torch.Tensor, fg: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
    """Cân bằng tổn thất giữa vùng polyp tiền cảnh và vùng nền."""
    a, b = masked_mean(error, fg), masked_mean(error, bg)
    va = (fg.sum((1, 2, 3)) > 0).float()
    vb = (bg.sum((1, 2, 3)) > 0).float()
    return ((a * va + b * vb) / (va + vb).clamp_min(1.0)).mean()


def region_gradient(a: torch.Tensor, b: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    """Đo lường sự sai khác gradient không gian trong vùng quan tâm (ROI)."""
    terms = []
    for dim in [2, 3]:
        err = (a.diff(dim=dim) - b.diff(dim=dim)).abs()
        if dim == 2:
            w = region[:, :, :-1] * region[:, :, 1:]
        else:
            w = region[:, :, :, :-1] * region[:, :, :, 1:]
        terms.append(masked_mean(err, w))
    return sum(terms)


def roi_perceptual(per: Any, a: torch.Tensor, b: torch.Tensor, region: torch.Tensor, side: int = 128) -> torch.Tensor:
    """Tính LPIPS perceptual loss trên patch có mật độ vùng quan tâm cao nhất."""
    if per is None:
        return a.sum() * 0.0
    side = min(side, a.shape[-1], a.shape[-2])
    losses = []
    for i in range(len(a)):
        coords = torch.nonzero(region[i, 0] > 0.5, as_tuple=False)
        if not len(coords):
            continue
        chosen = torch.linspace(
            0, len(coords) - 1, min(8, len(coords)), device=a.device
        ).long()
        options = []
        for point in coords[chosen]:
            y = max(0, min(a.shape[-2] - side, int(point[0]) - side // 2))
            x = max(0, min(a.shape[-1] - side, int(point[1]) - side // 2))
            score = float(region[i, :, y : y + side, x : x + side].mean())
            options.append((score, y, x))
        _, y, x = max(options)
        
        # Hỗ trợ cả AlexNet LPIPS trực tiếp hoặc callable
        pred_patch = a[i : i + 1, :, y : y + side, x : x + side]
        real_patch = b[i : i + 1, :, y : y + side, x : x + side]
        if hasattr(per, "loss"):
            losses.append(per.loss(pred_patch, real_patch))
        else:
            losses.append(per(pred_patch, real_patch))
    return torch.stack(losses).mean() if losses else a.sum() * 0.0


def background_loss(
    diff: Any,
    eps: torch.Tensor,
    noisy: torch.Tensor,
    clean: torch.Tensor,
    target: torch.Tensor,
    t: torch.Tensor,
    condition: torch.Tensor,
    per: Any,
    cfg: Dict[str, Any],
    teacher_eps: Optional[torch.Tensor] = None,
    pure: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tổn thất nền v2: cân bằng epsilon, gradient, perceptual LPIPS và RGB l1."""
    prediction = diff.predict_start_from_noise(noisy, t, eps).clamp(-1.0, 1.0)
    fg, bg = region_weights((clean + 1.0) / 2.0, condition)
    err = (eps - target).abs()
    epsilon = 0.5 * err.mean() + 0.5 * balanced_error(err, fg, bg)
    eligible = t <= cfg.get("aux_t_max", 120)

    grad = prediction.sum() * 0.0
    patch = grad
    rgb = grad
    if eligible.any():
        pred, real = prediction[eligible], clean[eligible]
        f, b = fg[eligible], bg[eligible]
        pl = rgb_to_lab((pred + 1.0) / 2.0)[:, :1]
        rl = rgb_to_lab((real + 1.0) / 2.0)[:, :1]
        grad = (
            0.5 * region_gradient(pl, rl, f).mean()
            + 0.5 * region_gradient(pl, rl, b).mean()
        )
        patch = 0.5 * roi_perceptual(per, pred, real, f) + 0.5 * roi_perceptual(
            per, pred, real, b
        )
        rgb = balanced_error((pred - real).abs(), f, b)

    prior = prediction.sum() * 0.0
    if teacher_eps is not None and pure is not None and pure.any():
        prior = (eps[pure] - teacher_eps[pure]).abs().mean()

    total = (
        epsilon
        + cfg.get("gradient_weight", 0.05) * grad
        + cfg.get("lpips_weight", 0.1) * patch
        + cfg.get("rgb_weight", 0.2) * rgb
        + cfg.get("teacher_weight", 0.0) * prior
    )
    return total, torch.stack([epsilon, grad, patch, prior, rgb])


def continuation_loss(
    diff: Any,
    eps: torch.Tensor,
    noisy: torch.Tensor,
    clean: torch.Tensor,
    target: torch.Tensor,
    t: torch.Tensor,
    condition: torch.Tensor,
    per: Any,
    cfg: Dict[str, Any],
    teacher_eps: Optional[torch.Tensor] = None,
    pure: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tổn thất tiếp tục huấn luyện refiner (continuation loss đến 49k updates).
    
    Tích hợp thêm:
    - Bảo vệ dải biên polyp (boundary band)
    - Prior matching từ mô hình giáo viên v2 đã hội tụ
    - Giám sát độ dốc viền (boundary gradient supervision).
    """
    total, parts = background_loss(
        diff, eps, noisy, clean, target, t, condition, per, cfg, None, None
    )
    mask, band, _ = protection_regions(condition, cfg)
    valid = tissue_support((clean + 1.0) / 2.0).detach()
    band = band * valid

    prior = eps.sum() * 0.0
    if teacher_eps is not None and pure is not None and pure.any():
        weights = 1.0 + cfg.get("prior_fg_boost", 0.5) * mask + cfg.get("prior_band_boost", 1.0) * band
        prior = masked_mean((eps[pure] - teacher_eps[pure]).abs(), weights[pure]).mean()

    pred = diff.predict_start_from_noise(noisy, t, eps).clamp(-1.0, 1.0)
    eligible = t <= cfg.get("aux_t_max", 120)
    boundary_rgb = eps.sum() * 0.0
    boundary_grad = boundary_rgb

    if eligible.any():
        a, b, w = pred[eligible], clean[eligible], band[eligible]
        boundary_rgb = masked_mean((a - b).abs(), w).mean()
        al = rgb_to_lab((a + 1.0) / 2.0)[:, :1]
        bl = rgb_to_lab((b + 1.0) / 2.0)[:, :1]
        boundary_grad = region_gradient(al, bl, w).mean()

    total = (
        total
        + cfg.get("teacher_weight", 0.1) * prior
        + cfg.get("boundary_rgb_weight", 0.15) * boundary_rgb
        + cfg.get("boundary_gradient_weight", 0.05) * boundary_grad
    )
    parts = torch.cat(
        [parts[:3], prior[None], parts[4:], boundary_rgb[None], boundary_grad[None]]
    )
    return total, parts
