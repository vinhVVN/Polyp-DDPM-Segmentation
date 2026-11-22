"""Image-mask preprocessing, synchronized augmentation, and tensor contract verification."""

from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
import pandas as pd


def tensor_pair(
    record: Dict[str, Any],
    size: int = 256,
    seed: Optional[int] = None,
    synthetic: Optional[Union[str, Path]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tải và tiền xử lý đồng bộ một cặp ảnh-mask.
    
    Quy ước chuẩn hóa:
    - Ảnh RGB: Bilinear resize về (size, size), chuẩn hoá [-1.0, 1.0] cho mô hình DDPM,
      hoặc [0.0, 1.0] cho downstream segmenter.
    - Mask nhị phân: Nearest resize về (size, size), ngưỡng nhị phân > 127.
      Trả về 1 kênh {0.0, 1.0} hoặc lặp 3 kênh {-1.0, +1.0} tuỳ theo mục đích.
    """
    img_path = synthetic or record["image"]
    with Image.open(img_path) as handle:
        image = handle.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)

    mask_path = record["mask"]
    with Image.open(mask_path) as handle:
        mask = handle.convert("L").resize((size, size), Image.Resampling.NEAREST)

    # Augmentation ngẫu nhiên có kiểm soát hạt giống (seed)
    if seed is not None:
        rng = random.Random(int(seed))
        if rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        angle = rng.uniform(-45.0, 45.0)
        image = image.rotate(angle, Image.Resampling.BILINEAR)
        mask = mask.rotate(angle, Image.Resampling.NEAREST)

    # Chuyển đổi sang PyTorch Tensor:
    # x: [0, 1] float tensor (3, H, W)
    x = torch.from_numpy(np.asarray(image).copy().transpose(2, 0, 1)).float() / 255.0
    # y: {0, 1} float tensor (1, H, W)
    y = torch.from_numpy((np.asarray(mask) > 127).astype(np.float32).copy())[None]

    return x, y


def normalize_segmenter(x: torch.Tensor) -> torch.Tensor:
    """Chuẩn hoá tensor ảnh [0, 1] theo phân phối ImageNet (dùng cho bộ phân đoạn downstream)."""
    mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (x - mean) / std


def verify_tensor_contracts(x: torch.Tensor, y: torch.Tensor) -> pd.DataFrame:
    """Kiểm tra và trả về bảng đối soát ràng buộc tensor (tensor contracts).
    
    Bảo đảm:
    - DDPM Generator image: (3, 256, 256) thuộc [-1, 1]
    - DDPM Condition mask: (3, 256, 256) thuộc {-1, +1}
    - Segmentation target: (1, 256, 256) thuộc {0, 1}
    """
    # DDPM formats
    ddpm_img = x * 2.0 - 1.0
    ddpm_mask = y.repeat(1, 3, 1, 1) * 2.0 - 1.0

    contracts = [
        {
            "Tensor": "generator image (DDPM)",
            "Shape": str(tuple(ddpm_img.shape[1:]) if ddpm_img.ndim == 4 else tuple(ddpm_img.shape)),
            "Min": float(ddpm_img.min()),
            "Max": float(ddpm_img.max()),
            "Kỳ vọng": "[-1.0, 1.0]",
        },
        {
            "Tensor": "condition mask (DDPM)",
            "Shape": str(tuple(ddpm_mask.shape[1:]) if ddpm_mask.ndim == 4 else tuple(ddpm_mask.shape)),
            "Min": float(ddpm_mask.min()),
            "Max": float(ddpm_mask.max()),
            "Kỳ vọng": "{-1.0, +1.0}",
        },
        {
            "Tensor": "segmentation target (SMP)",
            "Shape": str(tuple(y.shape[1:]) if y.ndim == 4 else tuple(y.shape)),
            "Min": float(y.min()),
            "Max": float(y.max()),
            "Kỳ vọng": "{0.0, 1.0}",
        },
    ]
    return pd.DataFrame(contracts)
