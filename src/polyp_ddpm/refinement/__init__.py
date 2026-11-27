"""Refinement modules: color calibration, tissue/background degradation, and continuation losses."""

from .color import (
    rgb_to_lab,
    lab_to_rgb,
    tissue_support,
    lab_moments,
    palette_projection,
    identity_lut,
    apply_lut,
    PaletteLUT,
    palette_swd,
    color_objective,
)
from .degradation import (
    degrade_real,
    protection_regions,
    mixed_training_state,
)
from .loss import continuation_loss

# Alias thuận tiện
swd_loss = palette_swd

__all__ = [
    "rgb_to_lab",
    "lab_to_rgb",
    "tissue_support",
    "lab_moments",
    "palette_projection",
    "identity_lut",
    "apply_lut",
    "PaletteLUT",
    "palette_swd",
    "swd_loss",
    "color_objective",
    "degrade_real",
    "protection_regions",
    "mixed_training_state",
    "continuation_loss",
]
