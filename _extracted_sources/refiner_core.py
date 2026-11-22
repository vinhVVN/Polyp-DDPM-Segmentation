"""Mask-conditioned partial DDPM refinement. New research adaptation; no quality claim."""

import os, json, hashlib, math, random, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from models import GaussianDiffusion, create_model, extract
from color_base import (
    rgb_to_lab,
    lab_to_rgb,
    lab_moments,
    tissue_support,
    gradient_match,
)


def canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)


def save_pt(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def load_pt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_for(sid, replicate=0):
    return int(
        hashlib.sha256((sid + ":" + str(replicate)).encode()).hexdigest()[:8], 16
    )


def png(path, size, mask=False):
    with Image.open(path) as im:
        a = np.array(
            im.convert("L" if mask else "RGB").resize(
                (size, size),
                Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR,
            ),
            copy=True,
        )
    return (
        torch.from_numpy((a > 127).astype(np.float32))[None]
        if mask
        else torch.from_numpy(a).permute(2, 0, 1).float() / 255
    )


def save_png(x, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = x.detach().cpu().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
    tmp = path.with_name(path.stem + ".tmp.png")
    Image.fromarray(arr).save(tmp)
    os.replace(tmp, path)


class TinyDenoiser(nn.Module):
    """Explicit CPU test fixture only; never used for production outputs."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 8, 3, padding=1), nn.SiLU(), nn.Conv2d(8, 3, 3, padding=1)
        )

    def forward(self, x, t):
        return self.conv(x)


def factory(cfg, device):
    net = (
        TinyDenoiser()
        if cfg.get("cpu_fixture", False)
        else create_model(256, 64, 2, in_channels=6, out_channels=3)
    )
    return GaussianDiffusion(
        net, image_size=cfg["size"], timesteps=250, loss_type="l1"
    ).to(device)


class Perceptual:
    def __init__(self, cfg, device):
        self.fixture = cfg.get("cpu_fixture", False)
        if not self.fixture:
            import lpips

            self.model = (
                lpips.LPIPS(net="alex", verbose=False)
                .to(device)
                .eval()
                .requires_grad_(False)
            )

    def loss(self, a, b):
        return (a - b).abs().mean() if self.fixture else self.model(a, b).mean()

    @torch.no_grad()
    def features(self, x, mask):
        # Low-level pretrained AlexNet regional feature moments, descriptive only.
        fmaps = (
            [x, F.avg_pool2d(x, 2)]
            if self.fixture
            else list(self.model.net(self.model.scaling_layer(x)))[:2]
        )
        keep = tissue_support((x + 1) / 2)
        vectors = []
        for reg in [mask, 1 - mask]:
            parts = []
            for f in fmaps:
                w = F.interpolate(reg * keep, size=f.shape[-2:], mode="area")
                n = w.sum((2, 3)).clamp_min(1)
                mu = (f * w).sum((2, 3)) / n
                sd = (
                    ((f - mu[:, :, None, None]) ** 2 * w).sum((2, 3)) / n + 1e-6
                ).sqrt()
                parts.extend([mu, sd])
            vectors.append(torch.cat(parts, 1))
        return torch.stack(vectors, 1).cpu().numpy()


def low_pass(x, side=32):
    # Smooth approximate low-pass; avoid block boundaries from nearest upsampling.
    small = F.interpolate(
        x, size=(side, side), mode="bilinear", align_corners=False, antialias=True
    )
    return F.interpolate(small, size=x.shape[-2:], mode="bilinear", align_corners=False)


def chroma_lock(image, reference, strength=0.75):
    """Adjust a,b only before RGB gamut conversion; L change after gamut is measured."""
    a = rgb_to_lab(image)
    b = rgb_to_lab(reference)
    ma, mb = lab_moments(image), lab_moments(reference)
    scale = (mb[:, 4:6] / ma[:, 4:6].clamp_min(0.015)).clamp(0.5, 2)
    target = (a[:, 1:] - ma[:, 1:3, None, None]) * scale[:, :, None, None] + mb[
        :, 1:3, None, None
    ]
    strength_map = strength * ((a[:, :1] - 0.02) / 0.1).clamp(0, 1)
    corrected = torch.cat(
        [a[:, :1], a[:, 1:] * (1 - strength_map) + target * strength_map], 1
    )
    out = lab_to_rgb(corrected)
    preserve = (reference.amax(1, keepdim=True) <= 0.025) | (
        reference.amin(1, keepdim=True) >= 0.995
    )
    out = torch.where(preserve, reference, out)
    keep_output = (image.amax(1, keepdim=True) <= 0.025) | (
        image.amin(1, keepdim=True) >= 0.995
    )
    return torch.where(keep_output, image, out)


@torch.no_grad()
def partial_reverse(
    diff, reference, condition, steps, seed, low_side=32, anchor=0.5, lock=True
):
    """Steps=60 uses original indices59..0. Noise-consistent ILVR-style coarse anchor."""
    if not 1 <= steps <= diff.num_timesteps:
        raise ValueError("Invalid reverse steps")
    device = reference.device
    gen = torch.Generator(device=device).manual_seed(int(seed))
    clean = reference * 2 - 1
    noise = torch.randn(clean.shape, device=device, generator=gen)
    ts = torch.full((len(clean),), steps - 1, device=device, dtype=torch.long)
    x = diff.q_sample(clean, ts, noise=noise)
    for t in reversed(range(steps)):
        ts = torch.full((len(x),), t, device=device, dtype=torch.long)
        eps = diff.denoise_fn(torch.cat([x, condition], 1), ts)
        x0 = diff.predict_start_from_noise(x, ts, eps).clamp(-1, 1)
        mean, _, logvar = diff.q_posterior(x0, x, ts)
        z = torch.randn(x.shape, device=device, generator=gen)
        proposal = mean + (t > 0) * (0.5 * logvar).exp() * z
        # Anchor at t-1, not by inserting a clean image into a noisy state.
        if t:
            a = diff.alphas_cumprod[t - 1]
            ref_state = a.sqrt() * clean + (1 - a).sqrt() * noise
        else:
            ref_state = clean
        x = proposal + anchor * (
            low_pass(ref_state, low_side) - low_pass(proposal, low_side)
        )
    out = ((x + 1) / 2).clamp(0, 1)
    return chroma_lock(out, reference) if lock else out


def paired_loss(diff, eps, noisy, clean, noise, t, perceptual, cfg):
    pred = diff.predict_start_from_noise(noisy, t, eps).clamp(-1, 1)
    l_eps = (eps - noise).abs().mean()
    l_grad = gradient_match(
        rgb_to_lab((pred + 1) / 2)[:, :1], rgb_to_lab((clean + 1) / 2)[:, :1]
    )
    l_per = perceptual.loss(pred, clean)
    total = l_eps + cfg["gradient_weight"] * l_grad + cfg["lpips_weight"] * l_per
    return total, torch.stack([l_eps, l_grad, l_per])


def sample_ids(ids, positions, seed):
    n = len(ids)
    memo = {}
    out = []
    for pos in positions:
        epoch, index = divmod(int(pos), n)
        if epoch not in memo:
            memo[epoch] = np.random.default_rng(seed + epoch).permutation(n)
        out.append(ids[int(memo[epoch][index])])
    return out


def training_batch(cfg, records, step, rank, world, micro):
    b = cfg["per_gpu_batch"]
    acc = cfg["accumulation"]
    total = b * world * acc
    positions = [step * total + micro * b * world + rank * b + j for j in range(b)]
    chosen = sample_ids(cfg["fit_ids"], positions, cfg["seed"] + 9)
    images = []
    masks = []
    for sid, pos in zip(chosen, positions):
        r = records[sid]
        x = png(r["image"], cfg["size"])
        m = png(r["mask"], cfg["size"], True)
        rng = np.random.default_rng(cfg["seed"] + pos)
        if rng.random() < 0.5:
            x = x.flip(-1)
            m = m.flip(-1)
        k = int(rng.integers(4))
        x = torch.rot90(x, k, [-2, -1])
        m = torch.rot90(m, k, [-2, -1])
        images.append(x)
        masks.append(m.repeat(3, 1, 1))
    return torch.stack(images) * 2 - 1, torch.stack(masks) * 2 - 1, chosen


def descriptive_metrics(x, source, mask):
    lab, base = rgb_to_lab(x), rgb_to_lab(source)
    support = tissue_support(source)
    color = (lab_moments(x)[:, :3] - lab_moments(source)[:, :3]).abs().mean()
    drift = (
        (
            low_pass(x, 32 if x.shape[-1] >= 32 else 4)
            - low_pass(source, 32 if x.shape[-1] >= 32 else 4)
        )
        .abs()
        .mean()
    )
    luma = (lab[:, :1] - base[:, :1]).abs().mean()
    grey = lab[:, :1]
    lap = F.conv2d(
        grey, grey.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])[None, None], padding=1
    ).abs()
    interior = F.avg_pool2d(support, 9, 1, 4) > 0.999
    values = lap[interior]
    return {
        "color_moment_shift": float(color),
        "coarse_rgb_drift": float(drift),
        "L_change": float(luma),
        "laplacian_median": float(values.median()) if values.numel() else 0.0,
        "laplacian_p95": float(torch.quantile(values, 0.95)) if values.numel() else 0.0,
    }


def masked_mean(error, weight):
    return (error * weight).sum((1, 2, 3)) / (
        weight.sum((1, 2, 3)) * error.shape[1]
    ).clamp_min(1)


def region_weights(clean_rgb, condition):
    mask = ((condition[:, :1] + 1) / 2).clamp(0, 1)
    valid = tissue_support(clean_rgb).detach()
    # Keep ROI losses off the FOV transition; epsilon still supervises the full image.
    valid = (F.avg_pool2d(valid, 5, 1, 2) > 0.99).float()
    return mask * valid, (1 - mask) * valid


def balanced_error(error, fg, bg):
    a, b = masked_mean(error, fg), masked_mean(error, bg)
    va = (fg.sum((1, 2, 3)) > 0).float()
    vb = (bg.sum((1, 2, 3)) > 0).float()
    return ((a * va + b * vb) / (va + vb).clamp_min(1)).mean()


def region_gradient(a, b, region):
    terms = []
    for dim in [2, 3]:
        err = (a.diff(dim=dim) - b.diff(dim=dim)).abs()
        if dim == 2:
            w = region[:, :, :-1] * region[:, :, 1:]
        else:
            w = region[:, :, :, :-1] * region[:, :, :, 1:]
        terms.append(masked_mean(err, w))
    return sum(terms)


def roi_perceptual(per, a, b, region, side=128):
    # Aligned paired patches; select a center whose patch contains the most ROI support.
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
        losses.append(
            per.loss(
                a[i : i + 1, :, y : y + side, x : x + side],
                b[i : i + 1, :, y : y + side, x : x + side],
            )
        )
    return torch.stack(losses).mean() if losses else a.sum() * 0


def degrade_real(clean, condition, seed):
    """Paired blur+grain simulation, especially background. No geometry warp or color shift.

    This small original degradation model is inspired by restoration training, not BSRGAN reproduction.
    """
    gen = torch.Generator(device=clean.device).manual_seed(int(seed))
    result = []
    rgb = (clean + 1) / 2
    mask = ((condition[:, :1] + 1) / 2).clamp(0, 1)
    for i in range(len(clean)):
        sigma = 0.5 + 1.0 * float(torch.rand((), device=clean.device, generator=gen))
        axis = torch.arange(-3, 4, device=clean.device, dtype=clean.dtype)
        kernel = torch.exp(-axis.square() / (2 * sigma * sigma))
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
        mix = (1 - 0.65 * mask[i : i + 1]) * tissue_support(rgb[i : i + 1])
        # Preserve genuine black pixels; no claim that simulated artifacts exactly match DDPM outputs.
        degraded = rgb[i : i + 1] + mix * (blur + grain + speckle - rgb[i : i + 1])
        degraded = torch.where(
            rgb[i : i + 1].amax(1, keepdim=True) < 0.025, rgb[i : i + 1], degraded
        )
        result.append(degraded.clamp(0, 1))
    return torch.cat(result) * 2 - 1


def mixed_training_state(diff, clean, condition, t, cfg, seed):
    gen = torch.Generator(device=clean.device).manual_seed(int(seed))
    repair = (
        torch.rand((len(clean), 1, 1, 1), device=clean.device, generator=gen)
        < cfg["restoration_probability"]
    )
    repair = repair & (t >= cfg["restoration_t_min"])[:, None, None, None]
    corrupted = degrade_real(clean, condition, seed + 1)
    base = torch.where(repair, corrupted, clean)
    noise = torch.randn(clean.shape, device=clean.device, generator=gen)
    noisy = diff.q_sample(base, t, noise)
    # A corrupted-real branch must NOT incorrectly use plain Gaussian noise as target.
    alpha = extract(diff.sqrt_alphas_cumprod, t, clean.shape)
    sigma = extract(diff.sqrt_one_minus_alphas_cumprod, t, clean.shape).clamp_min(1e-5)
    effective_target = (noisy - alpha * clean) / sigma
    return noisy, effective_target, repair.flatten()


def background_loss(
    diff, eps, noisy, clean, target, t, condition, per, cfg, teacher_eps=None, pure=None
):
    prediction = diff.predict_start_from_noise(noisy, t, eps).clamp(-1, 1)
    fg, bg = region_weights((clean + 1) / 2, condition)
    err = (eps - target).abs()
    epsilon = 0.5 * err.mean() + 0.5 * balanced_error(err, fg, bg)
    eligible = t <= cfg["aux_t_max"]
    grad = prediction.sum() * 0
    patch = grad
    rgb = grad
    if eligible.any():
        pred, real = prediction[eligible], clean[eligible]
        f, b = fg[eligible], bg[eligible]
        pl, rl = rgb_to_lab((pred + 1) / 2)[:, :1], rgb_to_lab((real + 1) / 2)[:, :1]
        grad = (
            0.5 * region_gradient(pl, rl, f).mean()
            + 0.5 * region_gradient(pl, rl, b).mean()
        )
        patch = 0.5 * roi_perceptual(per, pred, real, f) + 0.5 * roi_perceptual(
            per, pred, real, b
        )
        rgb = balanced_error((pred - real).abs(), f, b)
    prior = prediction.sum() * 0
    if teacher_eps is not None and pure is not None and pure.any():
        prior = (eps[pure] - teacher_eps[pure]).abs().mean()
    total = (
        epsilon
        + cfg["gradient_weight"] * grad
        + cfg["lpips_weight"] * patch
        + cfg["rgb_weight"] * rgb
        + cfg["teacher_weight"] * prior
    )
    return total, torch.stack([epsilon, grad, patch, prior, rgb])


def palette_chroma_only(image, reference, strength=0.25):
    # Use only a,b means from an unrelated train palette; never its spatial black/white mask.
    lab = rgb_to_lab(image)
    a = lab_moments(image)
    b = lab_moments(reference)
    shift = (b[:, 1:3] - a[:, 1:3]).clamp(-0.08, 0.08)[:, :, None, None] * strength
    fade = ((lab[:, :1] - 0.02) / 0.1).clamp(0, 1)
    out = lab_to_rgb(torch.cat([lab[:, :1], lab[:, 1:] + fade * shift], 1))
    keep = (image.amax(1, keepdim=True) < 0.025) | (image.amin(1, keepdim=True) > 0.995)
    return torch.where(keep, image, out)


@torch.no_grad()
def background_reverse(diff, reference, condition, steps, seed, cfg, palette=None):
    """Higher-noise background resynthesis with early, spatially selective coarse anchors."""
    if not 1 <= steps <= diff.num_timesteps:
        raise ValueError("Invalid reverse steps")
    device = reference.device
    clean = reference * 2 - 1
    gen = torch.Generator(device=device).manual_seed(int(seed))
    noise = torch.randn(clean.shape, device=device, generator=gen)
    ts = torch.full((len(clean),), steps - 1, device=device, dtype=torch.long)
    x = diff.q_sample(clean, ts, noise)
    soft = F.avg_pool2d(((condition[:, :1] + 1) / 2).clamp(0, 1), 15, 1, 7)
    for t in reversed(range(steps)):
        ts = torch.full((len(x),), t, device=device, dtype=torch.long)
        eps = diff.denoise_fn(torch.cat([x, condition], 1), ts)
        x0 = diff.predict_start_from_noise(x, ts, eps).clamp(-1, 1)
        mean, _, logvar = diff.q_posterior(x0, x, ts)
        z = torch.randn(x.shape, device=device, generator=gen)
        x = mean + (t > 0) * (0.5 * logvar).exp() * z
        # Last detail steps are completely free of source reinjection.
        fade = max(
            0.0,
            (t - cfg["release_last_steps"])
            / max(steps - 1 - cfg["release_last_steps"], 1),
        )
        if fade:
            alpha = diff.alphas_cumprod[t - 1] if t else x.new_tensor(1.0)
            ref = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
            fg = low_pass(ref, 32 if reference.shape[-1] >= 32 else 4) - low_pass(
                x, 32 if reference.shape[-1] >= 32 else 4
            )
            bg = low_pass(ref, cfg["low_side"]) - low_pass(x, cfg["low_side"])
            x = x + fade * (
                cfg["foreground_anchor"] * soft * fg
                + cfg["background_anchor"] * (1 - soft) * bg
            )
    out = ((x + 1) / 2).clamp(0, 1)
    return palette_chroma_only(
        out, reference if palette is None else palette, cfg["chroma_strength"]
    )


# --- v2 continuation: spatial protection and aligned boundary supervision ---
v2_reverse = background_reverse
v2_background_loss = background_loss


def protection_regions(condition, cfg):
    mask = ((condition[:, :1] + 1) / 2).clamp(0, 1)
    radius = cfg["guard_radius"]
    dilated = F.max_pool2d(mask, 2 * radius + 1, 1, radius)
    erode_radius = max(1, radius // 2)
    eroded = -F.max_pool2d(-mask, 2 * erode_radius + 1, 1, erode_radius)
    band = (dilated - eroded).clamp(0, 1)
    soft = F.avg_pool2d(dilated, 9, 1, 4)
    return mask, band, soft


def continuation_loss(
    diff, eps, noisy, clean, target, t, condition, per, cfg, teacher_eps=None, pure=None
):
    # Teacher is the frozen selected v2 refiner, not a new author initialization.
    total, parts = v2_background_loss(
        diff, eps, noisy, clean, target, t, condition, per, cfg, None, None
    )
    mask, band, _ = protection_regions(condition, cfg)
    valid = tissue_support((clean + 1) / 2).detach()
    band = band * valid
    prior = eps.sum() * 0
    if teacher_eps is not None and pure is not None and pure.any():
        weights = 1 + cfg["prior_fg_boost"] * mask + cfg["prior_band_boost"] * band
        prior = masked_mean((eps[pure] - teacher_eps[pure]).abs(), weights[pure]).mean()
    pred = diff.predict_start_from_noise(noisy, t, eps).clamp(-1, 1)
    eligible = t <= cfg["aux_t_max"]
    boundary_rgb = eps.sum() * 0
    boundary_grad = boundary_rgb
    if eligible.any():
        a, b, w = pred[eligible], clean[eligible], band[eligible]
        boundary_rgb = masked_mean((a - b).abs(), w).mean()
        al = rgb_to_lab((a + 1) / 2)[:, :1]
        bl = rgb_to_lab((b + 1) / 2)[:, :1]
        boundary_grad = region_gradient(al, bl, w).mean()
    total = (
        total
        + cfg["teacher_weight"] * prior
        + cfg["boundary_rgb_weight"] * boundary_rgb
        + cfg["boundary_gradient_weight"] * boundary_grad
    )
    parts = torch.cat(
        [parts[:3], prior[None], parts[4:], boundary_rgb[None], boundary_grad[None]]
    )
    return total, parts


@torch.no_grad()
def protected_reverse(diff, reference, condition, steps, seed, cfg, palette=None):
    # Protect only coarse structure near the polyp; do not copy real-reference pixels.
    if not 1 <= steps <= diff.num_timesteps:
        raise ValueError("Invalid reverse steps")
    clean = reference * 2 - 1
    gen = torch.Generator(device=reference.device).manual_seed(int(seed))
    noise = torch.randn(clean.shape, device=clean.device, generator=gen)
    ts = torch.full((len(clean),), steps - 1, device=clean.device, dtype=torch.long)
    x = diff.q_sample(clean, ts, noise)
    mask, band, protected = protection_regions(condition, cfg)
    side = 32 if reference.shape[-1] >= 32 else 4
    for t in reversed(range(steps)):
        ts = torch.full((len(x),), t, device=x.device, dtype=torch.long)
        eps = diff.denoise_fn(torch.cat([x, condition], 1), ts)
        x0 = diff.predict_start_from_noise(x, ts, eps).clamp(-1, 1)
        mean, _, logvar = diff.q_posterior(x0, x, ts)
        z = torch.randn(x.shape, device=x.device, generator=gen)
        x = mean + (t > 0) * (0.5 * logvar).exp() * z
        fade = max(
            0.0,
            (t - cfg["release_last_steps"])
            / max(steps - 1 - cfg["release_last_steps"], 1),
        )
        alpha = diff.alphas_cumprod[t - 1] if t else x.new_tensor(1.0)
        ref = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        coarse_delta = low_pass(ref, side) - low_pass(x, side)
        bg_delta = low_pass(ref, cfg["low_side"]) - low_pass(x, cfg["low_side"])
        fg_strength = (
            fade * cfg["foreground_anchor"] + (1 - fade) * cfg["foreground_floor"]
        )
        guard_strength = (1 - fade) * cfg["boundary_floor"]
        weight = (fg_strength * mask + guard_strength * band).clamp(0, 1)
        # Background remains fully released in the last detail steps, as in v2.
        x = (
            x
            + weight * coarse_delta
            + fade * cfg["background_anchor"] * (1 - protected) * bg_delta
        )
    out = ((x + 1) / 2).clamp(0, 1)
    return palette_chroma_only(
        out, reference if palette is None else palette, cfg["chroma_strength"]
    )


def background_reverse(diff, reference, condition, steps, seed, cfg, palette=None):
    return protected_reverse(diff, reference, condition, steps, seed, cfg, palette)


def state_fingerprint(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
