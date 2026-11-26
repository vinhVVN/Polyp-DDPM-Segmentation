"""Verify that all modules and subpackages import cleanly."""

import sys
from pathlib import Path

# Add src to sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def test_imports():
    import polyp_ddpm
    from polyp_ddpm import config, common, checkpointing
    from polyp_ddpm.data import dataset_maps, make_split, tensor_pair
    from polyp_ddpm.models import UNetModel, create_model, GaussianDiffusion
    from polyp_ddpm.refinement import (
        rgb_to_lab,
        lab_to_rgb,
        PaletteLUT,
        degrade_real,
        protection_regions,
        continuation_loss,
    )
    from polyp_ddpm.generation import full_reverse, protected_reverse, ensure_final_bank
    from polyp_ddpm.training import DDPMTrainer, RefinerTrainer
    from polyp_ddpm.evaluation import kernel_mmd, binary_metrics, create_segmentation_model
    from polyp_ddpm.visualization import show_inline, plot_loss_curve, show_image_mask_pairs

    assert polyp_ddpm.__version__ == "1.0.0"
    print("test_imports PASSED!")


if __name__ == "__main__":
    test_imports()
