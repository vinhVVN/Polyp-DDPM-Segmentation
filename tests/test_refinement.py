"""Smoke tests for PaletteLUT, color transforms, and spatial degradation."""

import sys
from pathlib import Path
import torch

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from polyp_ddpm.refinement.color import rgb_to_lab, lab_to_rgb, PaletteLUT, tissue_support, lab_moments
from polyp_ddpm.refinement.degradation import degrade_real, protection_regions


def test_color_space_inversion():
    x = torch.rand(2, 3, 32, 32).clamp(0.01, 0.99)
    lab = rgb_to_lab(x)
    rec = lab_to_rgb(lab)
    diff = (x - rec).abs().max()
    assert diff < 1e-2, f"Sai số chuyển đổi không gian màu Lab-RGB quá lớn: {diff}"
    print("test_color_space_inversion PASSED!")


def test_palette_lut():
    lut = PaletteLUT(dim=17, bases=3)
    x = torch.rand(2, 3, 32, 32)
    # palette vector represents 6 Lab moments (mean and std of L, a, b)
    palette_vec = lab_moments(torch.rand(2, 3, 32, 32))
    out, gamut = lut(x, palette_vec, project=True)
    assert out.shape == x.shape
    assert gamut >= 0.0
    print("test_palette_lut PASSED!")


def test_degradation_and_protection():
    clean = torch.rand(2, 3, 64, 64) * 2.0 - 1.0
    condition = torch.ones(2, 3, 64, 64)
    degraded = degrade_real(clean, condition, seed=42)
    assert degraded.shape == clean.shape
    assert degraded.min() >= -1.0 and degraded.max() <= 1.0

    cfg = {"guard_radius": 4}
    mask, band, soft = protection_regions(condition, cfg)
    assert mask.shape == (2, 1, 64, 64)
    assert band.shape == (2, 1, 64, 64)
    print("test_degradation_and_protection PASSED!")


if __name__ == "__main__":
    test_color_space_inversion()
    test_palette_lut()
    test_degradation_and_protection()
