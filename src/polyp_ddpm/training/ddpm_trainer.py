"""Baseline conditional DDPM trainer (10,000 updates, AdamW, EMA, MSE loss)."""

import os
import copy
import time
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from torch.optim import AdamW

from ..common import seed_for, resource_stop
from ..data.dataset import stream_ids
from ..data.preprocessing import tensor_pair
from ..checkpointing import CheckpointManager


class DDPMTrainer:
    """Bộ huấn luyện mô hình Conditional DDPM ban đầu."""

    def __init__(
        self,
        diff: torch.nn.Module,
        cfg: Dict[str, Any],
        device: torch.device,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):
        self.diff = diff
        self.cfg = cfg
        self.device = device
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(cfg.get("output_dir", "."))

        # Khởi tạo mô hình EMA (Exponential Moving Average)
        self.ema = copy.deepcopy(self.diff).to(self.device)
        for p in self.ema.parameters():
            p.requires_grad = False

        self.lr = cfg.get("own_lr", 1e-4)
        self.optimizer = AdamW(self.diff.parameters(), lr=self.lr, weight_decay=1e-4)
        self.history: List[Dict[str, Any]] = []

    def train_step(
        self,
        batch_x: torch.Tensor,
        batch_condition: torch.Tensor,
        step: int,
    ) -> float:
        """Một lượt lan truyền xuôi, tính loss và cập nhật trọng số optimizer."""
        self.diff.train()
        t = torch.randint(0, self.diff.num_timesteps, (len(batch_x),), device=self.device)
        losses = self.diff.training_losses(
            self.diff.denoise_fn,
            batch_x,
            t,
            condition_tensors=batch_condition,
        )
        loss = losses["loss"].mean()

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # Cập nhật EMA với hệ số decay 0.9999
        decay = self.cfg.get("ema_decay", 0.9999)
        with torch.no_grad():
            for p_ema, p_diff in zip(self.ema.parameters(), self.diff.parameters()):
                p_ema.data.mul_(decay).add_(p_diff.data, alpha=1.0 - decay)

        return float(loss.item())

    def run(self, max_steps: Optional[int] = None) -> List[Dict[str, Any]]:
        """Vòng lặp huấn luyện chính đến mốc max_steps (mặc định 10.000)."""
        target_steps = max_steps or self.cfg.get("own_steps", 10000)
        train_ids = self.cfg.get("train_ids", self.cfg.get("split", {}).get("train", []))
        rec_map = {r["id"]: r for r in self.cfg.get("records", [])}
        batch_size = self.cfg.get("own_per_gpu_batch", 4)
        size = self.cfg.get("size", 256)
        base_seed = self.cfg.get("seed", 42)

        start_step = len(self.history)
        for step in range(start_step + 1, target_steps + 1):
            if resource_stop(self.cfg, self.device):
                print(f"[Resource Guard] Dừng an toàn tại step {step} do áp lực bộ nhớ.")
                break

            # Lấy batch mẫu tất định
            pos = [step * batch_size + i for i in range(batch_size)]
            batch_ids = stream_ids(train_ids, pos, base_seed)

            xs, ys = [], []
            for idx, sid in enumerate(batch_ids):
                sample_seed = seed_for(base_seed, step, idx)
                x, y = tensor_pair(rec_map[sid], size=size, seed=sample_seed)
                xs.append(x * 2.0 - 1.0)
                ys.append(y.repeat(3, 1, 1) * 2.0 - 1.0)

            bx = torch.stack(xs).to(self.device)
            by = torch.stack(ys).to(self.device)

            loss_val = self.train_step(bx, by, step)

            if step % self.cfg.get("log_every", 50) == 0:
                record = {
                    "step": step,
                    "loss": loss_val,
                    "lr": float(self.optimizer.param_groups[0]["lr"]),
                    "timestamp": time.time(),
                }
                self.history.append(record)

            if step % self.cfg.get("save_every", 1000) == 0 or step == target_steps:
                self.checkpoint_manager.save_checkpoint_own(
                    self.diff, self.ema, self.optimizer, step, self.history, self.cfg
                )

        return self.history


def train_own_ddpm(cfg: Dict[str, Any], device: torch.device, diff: torch.nn.Module) -> List[Dict[str, Any]]:
    """Hàm wrapper tiện lợi để chạy huấn luyện DDPM 10k."""
    trainer = DDPMTrainer(diff, cfg, device)
    return trainer.run()
