"""Tests for dataset pairing, deterministic split, and tensor contracts."""

import sys
from pathlib import Path
import numpy as np
import torch

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from polyp_ddpm.data.splits import make_split, verify_split
from polyp_ddpm.data.preprocessing import verify_tensor_contracts


def test_split_reproducibility():
    # Giả lập 1000 ID Kvasir
    fake_images = {f"kvasir_{i:04d}": f"/path/to/img_{i:04d}.jpg" for i in range(1000)}
    fake_masks = {f"kvasir_{i:04d}": f"/path/to/mask_{i:04d}.jpg" for i in range(1000)}

    split1 = make_split(fake_images, fake_masks, seed=42)
    split2 = make_split(fake_images, fake_masks, seed=42)

    assert split1["train"] == split2["train"], "Split không tất định với cùng seed!"
    assert split1["test"] == split2["test"], "Split test không tất định!"
    assert verify_split(split1), "Tính toàn vẹn của split bị sai!"
    print("test_split_reproducibility PASSED!")


def test_tensor_contracts():
    # Giả lập tensor RGB [0, 1] và mask {0, 1}
    x = torch.rand(3, 256, 256)
    y = (torch.rand(1, 256, 256) > 0.5).float()

    df = verify_tensor_contracts(x, y)
    assert len(df) == 3
    assert df.loc[0, "Tensor"].startswith("generator image")
    print("test_tensor_contracts PASSED!")


if __name__ == "__main__":
    test_split_reproducibility()
    test_tensor_contracts()
