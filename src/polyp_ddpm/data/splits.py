"""Deterministic train/test split generation and verification for Kvasir-SEG."""

from typing import Dict, List, Any
import numpy as np


def make_split(images: Dict[str, str], masks: Dict[str, str], seed: int = 42) -> Dict[str, Any]:
    """Tạo phân chia Train 900 / Test 100 tất định theo hạt giống ngẫu nhiên (seed 42).
    
    Quy trình:
    1. Sắp xếp danh sách stem ID giao nhau (1000 ID).
    2. Dùng np.random.default_rng(seed).permutation để xáo trộn.
    3. Lấy 900 mẫu đầu cho train, 100 mẫu còn lại cho test độc lập.
    """
    ids = np.asarray(sorted(set(images) & set(masks)))
    if len(ids) != 1000 or set(images) != set(masks):
        raise ValueError(f"Kỳ vọng đúng 1000 ID trùng khớp giữa ảnh và mask. Thực tế: {len(ids)}")

    order = np.random.default_rng(seed).permutation(len(ids))
    train = ids[order[:900]].tolist()
    test = ids[order[900:]].tolist()

    assert not (set(train) & set(test)), "Tập train và test có phần tử giao nhau!"
    assert len(set(train + test)) == 1000, "Tổng số mẫu train và test khác 1000!"

    return {
        "dataset": "Kvasir-SEG",
        "seed": int(seed),
        "train_count": 900,
        "test_count": 100,
        "method": "sorted matched IDs -> default_rng(seed).permutation -> first900/rest100",
        "train": train,
        "test": test,
    }


def verify_split(split: Dict[str, Any], expected_train: int = 900, expected_test: int = 100) -> bool:
    """Kiểm tra tính toàn vẹn của split."""
    train_set = set(split.get("train", []))
    test_set = set(split.get("test", []))
    if len(train_set) != expected_train:
        return False
    if len(test_set) != expected_test:
        return False
    if bool(train_set & test_set):
        return False
    return True
