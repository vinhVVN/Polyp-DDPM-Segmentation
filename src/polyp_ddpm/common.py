"""Shared utility primitives, hashing, RNG control, and system telemetry."""

import os
import sys
import json
import time
import math
import random
import hashlib
import gc
import contextlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch


def canonical(value: Any) -> str:
    """Tạo mã băm SHA-256 chuẩn hoá từ JSON object (sắp xếp key)."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def digest(path: Union[str, Path]) -> str:
    """Tính mã băm SHA-256 của file nhị phân trên đĩa."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Union[str, Path], value: Any) -> None:
    """Ghi JSON nguyên tử hoặc chuẩn hoá."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + f".tmp_{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(p)


def load_pt(path: Union[str, Path], map_location: str = "cpu") -> Any:
    """Nạp checkpoint PyTorch an toàn."""
    return torch.load(path, map_location=map_location)


def save_pt(path: Union[str, Path], payload: Any) -> None:
    """Lưu checkpoint PyTorch nguyên tử."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + f".tmp_{os.getpid()}")
    torch.save(payload, temp)
    temp.replace(p)


def seed_all(seed: int) -> None:
    """Thiết lập seed đồng bộ trên Python, NumPy và PyTorch."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def seed_for(base_seed: int, step: int, rank: int = 0) -> int:
    """Sinh seed tất định cho từng update step và GPU worker."""
    return int((int(base_seed) * 1000003 + int(step) * 97 + int(rank)) % (2**31 - 1))


def rng_state(device: torch.device) -> Dict[str, Any]:
    """Lưu lại trạng thái RNG hiện tại để khôi phục."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None
        ),
    }


def restore_rng(value: Dict[str, Any], device: torch.device) -> None:
    """Khôi phục chính xác trạng thái RNG đã lưu."""
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if device.type == "cuda" and value["cuda"] is not None:
        torch.cuda.set_rng_state(value["cuda"], device)


@contextlib.contextmanager
def fixed_seed(seed: int, device: torch.device):
    """Context manager chạy một khối tính toán với seed tạm thời mà không làm lệch chuỗi RNG chính."""
    state = rng_state(device)
    seed_all(int(seed))
    try:
        yield
    finally:
        restore_rng(state, device)


def save_png(path: Union[str, Path], arr: np.ndarray) -> None:
    """Lưu mảng numpy (H, W, C) hoặc (H, W) thành file PNG."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(p)


def png(arr: np.ndarray) -> bytes:
    """Chuyển mảng ảnh thành bytes định dạng PNG (cho hiển thị hoặc lưu trữ)."""
    from io import BytesIO
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    bio = BytesIO()
    Image.fromarray(arr).save(bio, format="PNG")
    return bio.getvalue()


def resource_stop(cfg: Dict[str, Any], device: Optional[torch.device] = None, synchronize: bool = False) -> bool:
    """Kiểm tra áp lực bộ nhớ RAM và cgroup để dừng trước khi kernel bị OOM killer."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        total_rss = proc.memory_info().rss + sum(
            c.memory_info().rss for c in proc.children(recursive=True) if c.is_running()
        )
        total_rss_gib = total_rss / (1024 ** 3)
        free_mem_gib = psutil.virtual_memory().available / (1024 ** 3)
        rss_limit = cfg.get("host_rss_limit_gib", 20.0)
        free_min = cfg.get("host_free_min_gib", 3.0)
        stop = (total_rss_gib > rss_limit) or (free_mem_gib < free_min)
    except Exception:
        stop = False

    if synchronize and torch.distributed.is_initialized() and device is not None:
        flag = torch.tensor(int(stop), device=device)
        torch.distributed.all_reduce(flag)
        stop = bool(flag.item())
    return bool(stop)


def telemetry(cfg: Dict[str, Any], phase: str, rank: int, device: torch.device, **extra) -> None:
    """Ghi lại thông số bộ nhớ GPU và tiến trình hệ thống."""
    row = {
        "phase": phase,
        "rank": rank,
        "device": str(device),
        "pid": os.getpid(),
        "timestamp": time.time(),
        **extra,
    }
    if device.type == "cuda":
        row.update(
            name=torch.cuda.get_device_name(device),
            allocated_GiB=torch.cuda.memory_allocated(device) / (1024 ** 3),
            peak_GiB=torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        )
    out_dir = Path(cfg.get("output_dir", cfg.get("out", "."))) / "execution"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"{phase}_rank{rank}.json", row)
