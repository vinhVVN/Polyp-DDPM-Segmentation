"""Refiner continuation trainer up to 49,000 updates with milestone snapshotting."""

import os
import copy
import time
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from torch.optim import AdamW

from ..common import seed_for, resource_stop, write_json
from ..data.dataset import stream_ids
from ..data.preprocessing import tensor_pair
from ..refinement.degradation import mixed_training_state
from ..refinement.loss import continuation_loss
from ..checkpointing import CheckpointManager


class RefinerTrainer:
    """Bộ huấn luyện và tinh chỉnh refiner đến 49k updates."""

    def __init__(
        self,
        diff: torch.nn.Module,
        cfg: Dict[str, Any],
        device: torch.device,
        perceptual_net: Optional[Any] = None,
        teacher_diff: Optional[torch.nn.Module] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):
        self.diff = diff
        self.cfg = cfg
        self.device = device
        self.perceptual_net = perceptual_net
        self.teacher_diff = teacher_diff
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(cfg.get("output_dir", "."))

        self.lr = cfg.get("continuation_lr", 5e-7)
        self.optimizer = AdamW(self.diff.parameters(), lr=self.lr, weight_decay=1e-4)
        self.history: List[Dict[str, Any]] = []
        self.best: Dict[str, Any] = {"step": 0, "val_loss": float("inf")}

    def train_step(
        self,
        clean: torch.Tensor,
        condition: torch.Tensor,
        step: int,
    ) -> Dict[str, float]:
        """Một bước cập nhật refiner với trạng thái hỗn hợp và đa hàm mất mát."""
        self.diff.train()
        base_seed = self.cfg.get("seed", 42)
        step_seed = seed_for(base_seed, step, 0)

        t = torch.randint(0, self.diff.num_timesteps, (len(clean),), device=self.device)
        noisy, target, repair_flags = mixed_training_state(
            self.diff, clean, condition, t, self.cfg, step_seed
        )

        eps = self.diff.denoise_fn(torch.cat([noisy, condition], dim=1), t)

        teacher_eps = None
        if self.teacher_diff is not None:
            with torch.no_grad():
                self.teacher_diff.eval()
                teacher_eps = self.teacher_diff.denoise_fn(torch.cat([noisy, condition], dim=1), t)

        pure = ~repair_flags
        loss, parts = continuation_loss(
            self.diff,
            eps,
            noisy,
            clean,
            target,
            t,
            condition,
            self.perceptual_net,
            self.cfg,
            teacher_eps=teacher_eps,
            pure=pure,
        )

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return {
            "loss": float(loss.item()),
            "eps_loss": float(parts[0].item()),
            "grad_loss": float(parts[1].item()),
            "lpips_loss": float(parts[2].item()),
        }

    def run(self, start_step: int = 3000, target_step: int = 49000) -> List[Dict[str, Any]]:
        """Vòng lặp tiếp tục refiner từ start_step đến target_step (49.000)."""
        train_ids = self.cfg.get("train_ids", self.cfg.get("split", {}).get("train", []))
        rec_map = {r["id"]: r for r in self.cfg.get("records", [])}
        batch_size = self.cfg.get("own_per_gpu_batch", 4)
        size = self.cfg.get("size", 256)
        base_seed = self.cfg.get("seed", 42)
        milestones = set(self.cfg.get("milestone_totals", [3500, 10000, 20000, 30000, 40000, 49000]))

        for step in range(start_step + 1, target_step + 1):
            if resource_stop(self.cfg, self.device):
                print(f"[Resource Guard] Dừng an toàn tại step {step} do áp lực RAM.")
                break

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

            metrics = self.train_step(bx, by, step)

            if step % self.cfg.get("log_every", 50) == 0:
                record = {
                    "step": step,
                    "total_refiner_updates": step,
                    **metrics,
                    "timestamp": time.time(),
                }
                self.history.append(record)

            if step in milestones or step % self.cfg.get("save_every", 1000) == 0:
                self.checkpoint_manager.save_checkpoint_refiner(
                    self.diff, self.optimizer, step, self.history, self.best, self.cfg
                )

        return self.history


def train_refiner_continuation(
    cfg: Dict[str, Any],
    device: torch.device,
    diff: torch.nn.Module,
    start_step: int = 3000,
    target_step: int = 49000,
) -> List[Dict[str, Any]]:
    """Hàm wrapper tiện lợi để chạy tiếp tục huấn luyện refiner."""
    trainer = RefinerTrainer(diff, cfg, device)
    return trainer.run(start_step=start_step, target_step=target_step)
