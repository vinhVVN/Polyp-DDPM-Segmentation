"""Kvasir-SEG dataset pair resolution, file mapping, and sample streaming."""

import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union, Any
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .preprocessing import tensor_pair


def dataset_maps(
    image_dir: Optional[Union[str, Path]],
    mask_dir: Optional[Union[str, Path]],
    input_root: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Quét và đối chiếu 1.000 cặp ảnh-mask Kvasir-SEG theo stem ID trùng khớp."""
    def maps(path: Path) -> Dict[str, str]:
        result = {}
        for p in Path(path).iterdir():
            if p.is_file() and p.suffix.lower() in [
                ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
            ]:
                if p.stem in result:
                    raise ValueError(f"Trùng lặp ID trong thư mục {path}: {p.stem}")
                result[p.stem] = str(p)
        return result

    if bool(image_dir) != bool(mask_dir):
        raise ValueError("Cần cung cấp cả image_dir và mask_dir, hoặc để cả hai là None để tự tìm")

    candidates: List[Tuple[Path, Path]] = []
    if image_dir and mask_dir:
        candidates = [(Path(image_dir), Path(mask_dir))]
    elif input_root:
        candidates = [
            (p, p.parent / "masks")
            for p in Path(input_root).rglob("images")
            if "kvasir" in str(p).lower() and (p.parent / "masks").is_dir()
        ]

    good = []
    for a, b in candidates:
        if not a.exists() or not b.exists():
            continue
        im, ma = maps(a), maps(b)
        # Kỳ vọng đúng 1000 cặp ảnh nội soi Kvasir-SEG
        if len(set(im) & set(ma)) == 1000 and set(im) == set(ma):
            good.append((im, ma))

    if len(good) != 1:
        raise ValueError(
            f"Kỳ vọng tìm thấy duy nhất 1 bộ dữ liệu Kvasir-SEG (1000 cặp ảnh–mask). "
            f"Tìm thấy: {len(good)} bộ thỏa mãn. Vui lòng kiểm tra lại PATHS."
        )
    return good[0]


def stream_ids(ids: List[str], positions: List[int], seed: int) -> List[str]:
    """Sinh danh sách ID mẫu tương ứng với các vị trí update step toàn cục.
    
    Đảm bảo tính tất định qua nhiều epoch và worker mà không cần lưu toàn bộ mảng ID.
    """
    cache: Dict[int, np.ndarray] = {}
    chosen: List[str] = []
    n = len(ids)
    for position in positions:
        epoch, index = divmod(int(position), n)
        if epoch not in cache:
            cache[epoch] = np.random.default_rng(seed + epoch).permutation(n)
        chosen.append(ids[int(cache[epoch][index])])
    return chosen


class KvasirDataset(Dataset):
    """PyTorch Dataset cho Kvasir-SEG, hỗ trợ cả ảnh thật và ảnh sinh (synthetic)."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        size: int = 256,
        is_train: bool = True,
        seed: Optional[int] = None,
    ):
        self.records = records
        self.size = size
        self.is_train = is_train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        sample_seed = (self.seed + idx) if (self.is_train and self.seed is not None) else None
        return tensor_pair(rec, size=self.size, seed=sample_seed)
