"""Degradation modeling, noise injection, and spatial protection regions."""

from typing import Dict, Any, Tuple
import torch
import torch.nn.functional as F

from .color import tissue_support
from ..models.diffusion import extract


def degrade_real(clean: torch.Tensor, condition: torch.Tensor, seed: int) -> torch.Tensor:
    """Mô hình làm suy giảm ảnh thực có kiểm soát (paired blur + grain simulation).
    
    Tạo ra các biến dạng nhẹ về độ sắc nét (blur) và nhiễu vi thể (grain/speckle),
    tập trung vào vùng nền và mô nâng đỡ (tissue support), giữ nguyên cấu trúc hình học.
    Đầu vào: clean [-1, 1], condition [-1, 1]. Trả về: degraded [-1, 1].
    """
    gen = torch.Generator(device=clean.device).manual_seed(int(seed))
    result = []
    rgb = (clean + 1.0) / 2.0
    mask = ((condition[:, :1] + 1.0) / 2.0).clamp(0.0, 1.0)

    for i in range(len(clean)):
        sigma = 0.5 + 1.0 * float(torch.rand((), device=clean.device, generator=gen))
        axis = torch.arange(-3, 4, device=clean.device, dtype=clean.dtype)
        kernel = torch.exp(-axis.square() / (2.0 * sigma * sigma))
        kernel /= kernel.sum()
        k = (kernel[:, None] * kernel[None, :])[None, None].repeat(3, 1, 1, 1)

        blur = F.conv2d(
            F.pad(rgb[i : i + 1], (3, 3, 3, 3), mode="reflect"), k, groups=3
        )
        gaussian = torch.randn(rgb[i : i + 1].shape, device=clean.device, generator=gen)
        grain = (
            0.007 + 0.018 * float(torch.rand((), device=clean.device, generator=gen))
        ) * gaussian
        speckle = (
            0.025
            * torch.randn((1, 1, *rgb.shape[-2:]), device=clean.device, generator=gen)
            * rgb[i : i + 1]
        )
        mix = (1.0 - 0.65 * mask[i : i + 1]) * tissue_support(rgb[i : i + 1])
        degraded = rgb[i : i + 1] + mix * (blur + grain + speckle - rgb[i : i + 1])
        # Bảo toàn các điểm ảnh đen thực sự (genuine black pixels)
        degraded = torch.where(
            rgb[i : i + 1].amax(1, keepdim=True) < 0.025, rgb[i : i + 1], degraded
        )
        result.append(degraded.clamp(0.0, 1.0))

    return torch.cat(result) * 2.0 - 1.0


def protection_regions(condition: torch.Tensor, cfg: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tạo mặt nạ vùng bảo vệ (protection regions): mask, dải biên (band) và vùng mềm (soft)."""
    mask = ((condition[:, :1] + 1.0) / 2.0).clamp(0.0, 1.0)
    radius = cfg.get("guard_radius", 4)
    dilated = F.max_pool2d(mask, 2 * radius + 1, 1, radius)
    erode_radius = max(1, radius // 2)
    eroded = -F.max_pool2d(-mask, 2 * erode_radius + 1, 1, erode_radius)
    band = (dilated - eroded).clamp(0.0, 1.0)
    soft = F.avg_pool2d(dilated, 9, 1, 4)
    return mask, band, soft


def mixed_training_state(
    diff: Any,
    clean: torch.Tensor,
    condition: torch.Tensor,
    t: torch.Tensor,
    cfg: Dict[str, Any],
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tạo trạng thái huấn luyện hỗn hợp (mixed training state).
    
    Phối trộn giữa nhánh khuếch tán chuẩn (clean + Gaussian noise) và nhánh phục hồi
    (corrupted real image + noise), giúp refiner học cách tinh chỉnh cấu trúc lỗi.
    """
    gen = torch.Generator(device=clean.device).manual_seed(int(seed))
    repair_prob = cfg.get("restoration_probability", 0.5)
    repair = (
        torch.rand((len(clean), 1, 1, 1), device=clean.device, generator=gen) < repair_prob
    )
    repair = repair & (t >= cfg.get("restoration_t_min", 20))[:, None, None, None]

    corrupted = degrade_real(clean, condition, seed + 1)
    base = torch.where(repair, corrupted, clean)
    noise = torch.randn(clean.shape, device=clean.device, generator=gen)
    noisy = diff.q_sample(base, t, noise)

    alpha = extract(diff.sqrt_alphas_cumprod, t, clean.shape)
    sigma = extract(diff.sqrt_one_minus_alphas_cumprod, t, clean.shape).clamp_min(1e-5)
    effective_target = (noisy - alpha * clean) / sigma
    return noisy, effective_target, repair.flatten()
