"""Trainable color calibration of own DDPM output. No spatial resampling of output pixels."""

import math
import torch
from torch import nn
import torch.nn.functional as F


def rgb_to_lab(x):
    x = x.clamp(0, 1)
    linear = torch.where(
        x > 0.04045, ((x + 0.055) / 1.055).clamp_min(1e-8).pow(2.4), x / 12.92
    )
    matrix = x.new_tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = torch.einsum("ij,bjhw->bihw", matrix, linear)
    xyz = xyz / x.new_tensor([0.95047, 1.0, 1.08883])[None, :, None, None]
    delta = 6 / 29
    f = torch.where(
        xyz > delta**3, xyz.clamp_min(1e-8).pow(1 / 3), xyz / (3 * delta**2) + 4 / 29
    )
    return torch.stack(
        [
            (116 * f[:, 1] - 16) / 100,
            500 * (f[:, 0] - f[:, 1]) / 128,
            200 * (f[:, 1] - f[:, 2]) / 128,
        ],
        dim=1,
    )


def lab_to_rgb(lab, clip=True):
    fy = (lab[:, 0] * 100 + 16) / 116
    fx, fz = fy + lab[:, 1] * 128 / 500, fy - lab[:, 2] * 128 / 200
    f = torch.stack([fx, fy, fz], dim=1)
    delta = 6 / 29
    xyz = torch.where(f > delta, f**3, 3 * delta**2 * (f - 4 / 29))
    xyz = xyz * lab.new_tensor([0.95047, 1.0, 1.08883])[None, :, None, None]
    matrix = lab.new_tensor(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    )
    linear = torch.einsum("ij,bjhw->bihw", matrix, xyz)
    x = torch.where(
        linear > 0.0031308,
        1.055 * linear.clamp_min(1e-8).pow(1 / 2.4) - 0.055,
        12.92 * linear,
    )
    return x.clamp(0, 1) if clip else x


def tissue_support(x):
    # Exclude black FOV and nearly pure white specular pixels from color statistics.
    keep = (x.amax(1, keepdim=True) > 0.06) & (x.amin(1, keepdim=True) < 0.98)
    fallback = torch.ones_like(keep)
    return torch.where(keep.sum((2, 3), keepdim=True) >= 16, keep, fallback).float()


def lab_moments(x, support=None):
    lab = rgb_to_lab(x)
    w = tissue_support(x).detach() if support is None else support.detach()
    n = w.sum((2, 3)).clamp_min(1)
    mu = (lab * w).sum((2, 3)) / n
    variance = ((lab - mu[:, :, None, None]) ** 2 * w).sum((2, 3)) / n
    std = (variance + 1e-6).sqrt()
    return torch.cat([mu, std], 1)


def palette_projection(x, palette, gain_limit=3.0):
    """Global diagonal Lab color transfer, an explicit first/second moment constraint.

    It has no learned geometry. RGB gamut clipping means final moments need not
    exactly match the target; actual post-gamut errors are always measured.
    """
    support = tissue_support(x)
    source = lab_moments(x, support)
    lab = rgb_to_lab(x)
    floor = x.new_tensor([0.035, 0.015, 0.015])[None]
    src_std = torch.maximum(source[:, 3:], floor)
    target_std = torch.maximum(palette[:, 3:], floor)
    scale = (target_std / src_std).clamp(1 / gain_limit, gain_limit)
    corrected = (lab - source[:, :3, None, None]) * scale[:, :, None, None] + palette[
        :, :3, None, None
    ]
    # Preserve deep shadows continuously; no threshold-based hole cutting/FOV segmentation.
    lightness = torch.minimum(corrected[:, :1], lab[:, :1].clamp_min(0) * 3.0)
    corrected = torch.cat([lightness, corrected[:, 1:]], dim=1)
    raw = lab_to_rgb(corrected, clip=False)
    result = raw.clamp(0, 1)
    # Keep existing black borders and pure white highlights at the same locations.
    preserve = (x.amax(1, keepdim=True) <= 0.025) | (x.amin(1, keepdim=True) >= 0.995)
    result = torch.where(preserve, x, result)
    penalty_mask = (~preserve).float()
    excess = ((F.relu(-raw) + F.relu(raw - 1)) * penalty_mask).sum() / (
        penalty_mask.sum() * 3
    ).clamp_min(1)
    return result, excess


def identity_lut(dim, device=None):
    axis = torch.linspace(0, 1, dim, device=device)
    blue, green, red = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack([red, green, blue])


def apply_lut(x, lut):
    # 5-D grid_sample: D=B, H=G, W=R. Grid coordinates are (R,G,B).
    grid = x.permute(0, 2, 3, 1).unsqueeze(1) * 2 - 1
    return F.grid_sample(
        lut, grid, mode="bilinear", padding_mode="border", align_corners=True
    ).squeeze(2)


class PaletteLUT(nn.Module):
    """Small image/palette-conditioned LUT model, inspired by image-adaptive 3DLUT.

    Original implementation using native PyTorch. This is an adaptation, not
    the published authors' architecture or released weights.
    """

    def __init__(self, dim=17, bases=3):
        super().__init__()
        self.dim, self.bases = dim, bases
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 48, 3, 2, 1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.predictor = nn.Sequential(
            nn.Linear(48 + 6, 48), nn.SiLU(), nn.Linear(48, bases)
        )
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)
        values = torch.randn(bases, 3, dim, dim, dim) * 0.002
        values -= values.mean(0, keepdim=True)
        self.residuals = nn.Parameter(values)
        self.register_buffer("identity", identity_lut(dim))

    def forward(self, x, palette, project=True):
        features = self.encoder(
            F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        ).flatten(1)
        weights = self.predictor(torch.cat([features, palette], 1)).softmax(1)
        tables = self.identity[None] + 0.75 * self.residuals.tanh()
        lut = torch.einsum("bm,mcijk->bcijk", weights, tables)
        raw = apply_lut(x, lut)
        clip_penalty = (F.relu(-raw) + F.relu(raw - 1)).mean()
        rendered = raw.clamp(0, 1)
        if project:
            rendered, gamut = palette_projection(rendered, palette)
            clip_penalty = clip_penalty + gamut
            baseline, _ = palette_projection(x, palette)
            # Learned increment is bounded to avoid large new chromatic speckles.
            rendered = baseline + 0.04 * torch.tanh((rendered - baseline) / 0.04)
            rendered = rendered.clamp(0, 1)
            preserve = (x.amax(1, keepdim=True) <= 0.025) | (
                x.amin(1, keepdim=True) >= 0.995
            )
            rendered = torch.where(preserve, x, rendered)
        return rendered, clip_penalty

    def regularization(self):
        tables = self.identity[None] + 0.75 * self.residuals.tanh()
        tv = sum((self.residuals.diff(dim=axis) ** 2).mean() for axis in [2, 3, 4])
        # Monotonicity in the corresponding color axis; cross-channel mixing is allowed.
        mono = F.relu(-tables[:, 0].diff(dim=3)).mean()
        mono += F.relu(-tables[:, 1].diff(dim=2)).mean()
        mono += F.relu(-tables[:, 2].diff(dim=1)).mean()
        return tv, mono


def color_corruption(clean, seed, severe=True):
    """Paired supervision: invertible logit color/exposure shifts, no spatial warp."""
    g = torch.Generator(device=clean.device).manual_seed(int(seed))
    b = len(clean)
    bias = torch.randn(b, 3, 1, 1, device=clean.device, generator=g) * (
        1.0 if severe else 0.25
    )
    exposure = torch.randn(b, 1, 1, 1, device=clean.device, generator=g) * 0.7
    gamma = (
        (0.25 * torch.randn(b, 1, 1, 1, device=clean.device, generator=g))
        .exp()
        .clamp(0.65, 1.55)
    )
    logit = torch.logit(clean.clamp(0.005, 0.995))
    noisy = torch.sigmoid(gamma * logit + bias + exposure)
    black = clean.amax(1, keepdim=True) <= 0.025
    return torch.where(black, clean, noisy)


def gradient_match(a, b):
    # Aligned paired target gradients, not an energy statistic that rewards speckles.
    return (a.diff(dim=2) - b.diff(dim=2)).abs().mean() + (
        a.diff(dim=3) - b.diff(dim=3)
    ).abs().mean()


def normalized_luma(x):
    l = rgb_to_lab(x)[:, :1]
    mu, std = l.mean((2, 3), keepdim=True), l.std((2, 3), keepdim=True).clamp_min(0.05)
    return (l - mu) / std


def palette_swd(x, y, seed=123, pixels=256, directions=16):
    """Spatially unpaired Lab sliced Wasserstein loss; never pixel L1 to author."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    a, b = rgb_to_lab(x).flatten(2), rgb_to_lab(y).flatten(2)
    valid_a, valid_b = (
        tissue_support(x).flatten(2).bool(),
        tissue_support(y).flatten(2).bool(),
    )
    theta = torch.randn(3, directions, device=x.device, generator=g)
    theta = theta / theta.norm(dim=0, keepdim=True).clamp_min(1e-6)
    losses = []
    for i in range(len(x)):
        ia = torch.nonzero(valid_a[i, 0], as_tuple=True)[0]
        ib = torch.nonzero(valid_b[i, 0], as_tuple=True)[0]
        n = min(pixels, len(ia), len(ib))
        ia = ia[torch.linspace(0, len(ia) - 1, n, device=x.device).long()]
        ib = ib[torch.linspace(0, len(ib) - 1, n, device=x.device).long()]
        aa, bb = theta.T @ a[i, :, ia], theta.T @ b[i, :, ib]
        losses.append((aa.sort(-1).values - bb.sort(-1).values).abs().mean())
    return torch.stack(losses).mean()


def delta_e76(a, b):
    difference = (rgb_to_lab(a) - rgb_to_lab(b)) * a.new_tensor([100, 128, 128])[
        None, :, None, None
    ]
    return difference.square().sum(1).clamp_min(1e-10).sqrt().mean()


def restoration_metrics(candidate, target):
    mse = (candidate - target).square().mean().clamp_min(1e-10)
    return {
        "deltaE76": float(delta_e76(candidate, target)),
        "PSNR": float(-10 * torch.log10(mse)),
        "gradient_L1": float(gradient_match(candidate, target)),
    }
