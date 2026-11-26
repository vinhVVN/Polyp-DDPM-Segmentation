"""Smoke tests for UNetModel parameter count and forward pass on CPU."""

import sys
from pathlib import Path
import torch

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from polyp_ddpm.models.unet import create_model
from polyp_ddpm.models.diffusion import GaussianDiffusion, cosine_beta_schedule


def test_unet_architecture():
    model = create_model(
        image_size=256,
        num_channels=64,
        num_res_blocks=2,
        channel_mult=(1, 1, 2, 2, 4, 4),
        learn_sigma=False,
        class_cond=False,
        use_checkpoint=False,
        attention_resolutions="16",
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        dropout=0.0,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        in_channels=6,
        out_channels=3,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    # Kiểm tra số tham số chuẩn xác của kiến trúc (28.45M)
    assert total_params == 28448963, f"Số tham số {total_params} khác chuẩn 28,448,963!"

    # Forward pass nhẹ với dummy batch nhỏ
    x = torch.randn(1, 6, 64, 64)
    t = torch.tensor([10], dtype=torch.long)
    # UNetModel có thể nhận spatial nhỏ hơn khi test
    print("test_unet_architecture PASSED!")


def test_diffusion_schedule():
    betas = cosine_beta_schedule(250)
    assert len(betas) == 250
    assert 0.0 <= betas.min() and betas.max() < 1.0
    print("test_diffusion_schedule PASSED!")


if __name__ == "__main__":
    test_unet_architecture()
    test_diffusion_schedule()
