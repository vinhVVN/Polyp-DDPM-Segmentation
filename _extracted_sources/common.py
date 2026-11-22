"""Shared primitives for the report master. No service or repository calls."""

import os, sys, json, time, math, random, hashlib, gc, shutil, contextlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from refiner_core import (
    canonical,
    digest,
    write_json,
    save_pt,
    load_pt,
    png,
    save_png,
    factory,
    seed_for,
    seed_all,
    state_fingerprint,
)


def rng_state(device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None
        ),
    }


def restore_rng(value, device):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(value["cuda"], device)


@contextlib.contextmanager
def fixed_seed(seed, device):
    value = rng_state(device)
    seed_all(int(seed))
    try:
        yield
    finally:
        restore_rng(value, device)


def tensor_pair(record, size, seed=None, synthetic=None):
    with Image.open(synthetic or record["image"]) as handle:
        image = handle.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    with Image.open(record["mask"]) as handle:
        mask = handle.convert("L").resize((size, size), Image.Resampling.NEAREST)
    if seed is not None:
        rng = random.Random(int(seed))
        if rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        angle = rng.uniform(-45, 45)
        image = image.rotate(angle, Image.Resampling.BILINEAR)
        mask = mask.rotate(angle, Image.Resampling.NEAREST)
    x = torch.from_numpy(np.asarray(image).copy().transpose(2, 0, 1)).float() / 255
    y = torch.from_numpy((np.asarray(mask) > 127).astype("float32").copy())[None]
    return x, y


def stream_ids(ids, positions, seed):
    cache = {}
    chosen = []
    for position in positions:
        epoch, index = divmod(int(position), len(ids))
        if epoch not in cache:
            cache[epoch] = np.random.default_rng(seed + epoch).permutation(len(ids))
        chosen.append(ids[int(cache[epoch][index])])
    return chosen


def runtime(cfg, training=False):
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if cfg.get("cpu_fixture", False):
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available() or local >= torch.cuda.device_count():
            raise RuntimeError("Missing assigned CUDA device")
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    if world != cfg["world_size"]:
        raise ValueError("World size differs from protocol")
    if training and world > 1:
        import datetime

        dist.init_process_group(
            "gloo" if device.type == "cpu" else "nccl",
            timeout=datetime.timedelta(minutes=10),
        )
    torch.set_num_threads(cfg.get("cpu_threads_per_rank", 2))
    seed_all(cfg["seed"] + rank)
    return rank, world, device


def telemetry(cfg, phase, rank, device, **extra):
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
            allocated_GiB=torch.cuda.memory_allocated(device) / 2**30,
            peak_GiB=torch.cuda.max_memory_allocated(device) / 2**30,
        )
    write_json(Path(cfg["out"]) / "execution" / f"{phase}_rank{rank}.json", row)
    print(json.dumps(row), flush=True)


def resource_stop(cfg, device=None, synchronize=False):
    import psutil
    import torch.distributed as dist

    total = 0.0
    try:
        parent = psutil.Process(os.getpid())
        total = parent.memory_info().rss + sum(
            p.memory_info().rss
            for p in parent.children(recursive=True)
            if p.is_running()
        )
    except psutil.Error:
        pass
    rss_stop = total / 2**30 > cfg.get(
        "host_rss_limit_gib", 20.0
    ) or psutil.virtual_memory().available / 2**30 < cfg.get("host_free_min_gib", 3.0)
    container_stop = False
    for used, limit in [
        ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
        (
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ),
    ]:
        try:
            a, b = int(Path(used).read_text()), int(Path(limit).read_text())
            if b < 2**60:
                container_stop = a / b > cfg.get("host_memory_fraction", 0.80)
                break
        except (OSError, ValueError):
            pass
    deadline_stop = time.time() > cfg["deadline"] - 180
    stop = rss_stop or container_stop or deadline_stop
    if stop:
        write_json(
            Path(cfg["out"]) / "execution" / f"stop_reason_{os.getpid()}.json",
            {
                "time": time.time(),
                "deadline": deadline_stop,
                "host_rss_or_free": bool(rss_stop),
                "cgroup_memory": bool(container_stop),
                "rss_gib": total / 2**30,
            },
        )
    if synchronize and dist.is_initialized():
        flag = torch.tensor(int(stop), device=device)
        dist.all_reduce(flag)
        stop = bool(flag.item())
    return bool(stop)


def assert_config(cfg, source_dir):
    if canonical(cfg["locked"]) != cfg["protocol"]:
        raise ValueError("Invalid master protocol")
    contract = cfg.get("execution_contract")
    if contract and canonical(contract["payload"]) != contract["hash"]:
        raise ValueError("Invalid execution compatibility contract")
    code = contract["payload"]["runtime_code"] if contract else cfg["locked"]["code"]
    if contract and contract["payload"]["compute_protocol"] != cfg["protocol"]:
        raise ValueError("Compatibility contract belongs to another experiment")
    for name, sha in code.items():
        if digest(Path(source_dir) / name) != sha:
            raise ValueError("Changed runtime source: " + name)
    for key, value in cfg["locked"]["settings"].items():
        if cfg[key] != value:
            raise ValueError("Changed locked setting: " + key)
    if cfg["author_sha256"] != cfg["locked"]["author_sha256"]:
        raise ValueError("Runtime author identity differs from protocol")
    if (
        cfg["split"] != cfg["locked"]["split"]
        or cfg["train_ids"] != cfg["split"]["train"]
        or cfg["test_ids"] != cfg["split"]["test"]
    ):
        raise ValueError("Runtime split differs from protocol")
    expected_records = {r["id"]: r for r in cfg["locked"]["data"]}
    if set(expected_records) != {r["id"] for r in cfg["records"]}:
        raise ValueError("Runtime data IDs differ from protocol")
    for record in cfg["records"]:
        for key in ["image_sha256", "mask_sha256"]:
            if record[key] != expected_records[record["id"]][key]:
                raise ValueError("Runtime data identity differs from protocol")
    for r in cfg["records"]:
        for key in ["image", "mask"]:
            if digest(r[key]) != r[key + "_sha256"]:
                raise ValueError("Changed data: " + r["id"])
    if digest(cfg["author_checkpoint"]) != cfg["author_sha256"]:
        raise ValueError("Author checkpoint changed")


@torch.no_grad()
def full_reverse(diff, condition, seed, trace=False):
    """One full 250-step trajectory with its own RNG scope; [0,1] RGB out."""
    diff.eval()
    device = condition.device
    frames = []
    with fixed_seed(seed, device):
        x = torch.randn(condition.shape, device=device)
        if trace:
            frames.append((x[0].cpu(), "initial Gaussian noise"))
        for timestep in reversed(range(diff.num_timesteps)):
            t = torch.full((len(x),), timestep, device=device, dtype=torch.long)
            x = diff.p_sample(x, t, condition_tensors=condition)
            if trace and timestep in [199, 149, 99, 49, 0]:
                frames.append((x[0].cpu(), f"after t={timestep}"))
    return (x + 1).div(2).clamp(0, 1), frames


def segmentation_model(cfg, device, pretrained=False):
    if cfg.get("cpu_fixture"):
        return nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 1)
        ).to(device)
    import segmentation_models_pytorch as smp

    return smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=cfg["encoder_weights"] if pretrained else None,
        in_channels=3,
        classes=1,
    ).to(device)


def normalize_segmenter(x):
    mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (x - mean) / std


def dice_loss(logits, y):
    p = logits.sigmoid()
    return 1 - (2 * (p * y).sum() + 1e-7) / (p.sum() + y.sum() + 1e-7)


def binary_metrics(prob, mask):
    from scipy.ndimage import binary_erosion, binary_dilation

    pred = prob >= 0.5
    gt = mask.astype(bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    a = pred ^ binary_erosion(pred)
    b = gt ^ binary_erosion(gt)
    if not a.any() or not b.any():
        bf = float(not a.any() and not b.any())
    else:
        precision = (a & binary_dilation(b, iterations=2)).sum() / a.sum()
        recall = (b & binary_dilation(a, iterations=2)).sum() / b.sum()
        bf = float(2 * precision * recall / (precision + recall + 1e-7))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": float((tp + 1e-7) / (tp + fp + fn + 1e-7)),
        "dice": float((2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)),
        "boundary_f1": bf,
        "precision": float((tp + 1e-7) / (tp + fp + 1e-7)),
        "recall": float((tp + 1e-7) / (tp + fn + 1e-7)),
    }
