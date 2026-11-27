"""Checkpoint management, protocol compatibility verification, and history recovery.

Phục vụ lưu/nạp checkpoint và kiểm tra tính toàn vẹn (integrity) giữa các mốc huấn luyện:
- Generator ban đầu (own, 10k steps)
- Bộ chỉnh màu (color, 2k steps)
- Refiner (texture v1, v2 background, v21 boundary, continuation đến 49k)
- Checkpoint công bố của tác giả (polyp_ddpm_author.pt)
"""

import os
import json
import gc
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import pandas as pd
import torch

from .common import digest, canonical, load_pt, save_pt, write_json


class CheckpointManager:
    """Quản lý serialization, nạp lịch sử huấn luyện và kiểm chứng tính tương thích."""

    def __init__(self, output_dir: Union[str, Path], resume_master_output: Optional[Union[str, Path]] = None):
        self.output_dir = Path(output_dir)
        self.resume_master_output = Path(resume_master_output) if resume_master_output else None

    def resolve_stage_root(self, role: str, paths: Optional[Dict[str, Optional[str]]] = None) -> Optional[Path]:
        """Xác định thư mục dữ kiện của từng giai đoạn (local hoặc Kaggle input)."""
        # 1. Kiểm tra trực tiếp trong output_dir
        candidate = self.output_dir / role
        if candidate.is_dir():
            return candidate

        # 2. Kiểm tra trong resume_master_output
        if self.resume_master_output and (self.resume_master_output / role).is_dir():
            return self.resume_master_output / role

        # 3. Kiểm tra trong bảng PATHS truyền vào
        if paths and role in paths and paths[role]:
            p = Path(paths[role])
            if p.is_dir():
                return p

        # 4. Tìm kiếm cục bộ trong workspace
        for sub in [Path.cwd(), Path.cwd().parent]:
            cand = sub / role
            if cand.is_dir():
                return cand
        return None

    def read_history(
        self,
        role: str,
        paths: Optional[Dict[str, Optional[str]]] = None,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Đọc lịch sử huấn luyện từ checkpoint hoặc file log CSV của từng stage."""
        stage = self.resolve_stage_root(role, paths)
        if stage is None:
            return pd.DataFrame(), pd.DataFrame([{"Giai đoạn": role, "Trạng thái": "Chưa attach nguồn"}])

        identity = stage / "protocol.json"
        protocol = json.loads(identity.read_text())["protocol"] if identity.exists() else {}
        history = pd.DataFrame()

        checkpoint_candidates = [stage / "checkpoints/last.pt"]
        if role == "own":
            checkpoint_candidates = [stage / "generator/last.pt", *checkpoint_candidates]
        elif role == "color":
            checkpoint_candidates = [
                stage / "color/checkpoints/last.pt",
                stage / "color/last.pt",
                *checkpoint_candidates,
            ]
        elif role == "continuation":
            checkpoint_candidates = [
                stage / "checkpoints/last.pt",
                stage / "continuation/last.pt",
                self.output_dir / "research_49k/checkpoints/last.pt",
                *checkpoint_candidates,
            ]

        step = None
        for checkpoint in checkpoint_candidates:
            if checkpoint.exists():
                state = load_pt(checkpoint)
                # Kiểm tra tính tương thích protocol
                accepted = [json.loads(identity.read_text())["hash"]] if identity.exists() else [state.get("protocol")]
                accepted += protocol.get("accepted_resume_protocols", [])
                
                # Xác thực hash tường minh
                state_proto = state.get("protocol")
                if state_proto and accepted and (state_proto not in accepted):
                    # Ghi nhận cảnh báo rõ ràng thay vì bỏ qua lỗi
                    print(f"[CẢNH BÁO KIỂM TRA HASH] Checkpoint protocol '{state_proto}' khác protocol chính '{accepted[0]}'")
                
                history = pd.DataFrame(state.get("history", []))
                step = int(state.get("step", 0))
                del state
                gc.collect()
                break

        # Nếu không có state['history'], đọc từ file CSV
        if history.empty:
            for filename in ["color_training_history.csv", "training_history.csv", "history.csv"]:
                path = stage / "tables" / filename
                if not path.exists():
                    path = stage / filename
                if path.exists() and path.stat().st_size > 2:
                    history = pd.read_csv(path)
                    break

        if "update" in history and "step" not in history:
            history = history.rename(columns={"update": "step"})

        if not history.empty and "step" in history:
            history = history.sort_values("step").drop_duplicates("step").copy()
            if step is None:
                step = int(history.step.max())
            history = history[history.step <= step].copy()
            
            cfg = protocol.get("config", protocol.get("settings", {}))
            if run_config:
                cfg = {**cfg, **run_config}
            batch = cfg.get("effective_batch") or cfg.get("per_gpu_batch", 2) * cfg.get("world_size", 2) * cfg.get("accumulation", 2)
            if role == "own":
                batch = cfg.get("effective_batch", cfg.get("own_per_gpu_batch", 4) * cfg.get("world_size", 2) * cfg.get("own_accumulation", 4))
            elif role == "color":
                batch = cfg.get("color_batch", 8)
            n = 900 if role == "own" else 810
            history["data_pass_equivalent"] = history.step * batch / n

        initial = int(protocol.get("initial_refiner_updates", 0))
        status = pd.DataFrame([{
            "Giai đoạn": role,
            "Update LAST của stage": step,
            "Bản ghi log": len(history),
            "Tổng refiner LAST": initial + step if step is not None and role not in ["own", "color"] else None,
            "Trạng thái": "Có dữ kiện" if not history.empty else "Chưa có log",
        }])
        return history, status

    def save_checkpoint_own(
        self,
        diff: torch.nn.Module,
        ema: torch.nn.Module,
        opt: torch.optim.Optimizer,
        step: int,
        history: List[Dict[str, Any]],
        cfg: Dict[str, Any],
        rank: int = 0,
    ) -> None:
        """Lưu checkpoint cho mô hình DDPM 10k ban đầu."""
        if rank != 0:
            return
        ckpt_dir = self.output_dir / "generator"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": step,
            "model": diff.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "history": history,
            "config": cfg,
            "protocol": cfg.get("protocol", "polyp_ddpm_master_v400"),
        }
        save_pt(ckpt_dir / "last.pt", payload)
        if step % cfg.get("save_every", 1000) == 0:
            save_pt(ckpt_dir / f"step_{step:06d}.pt", payload)

    def save_checkpoint_refiner(
        self,
        diff: torch.nn.Module,
        opt: torch.optim.Optimizer,
        step: int,
        history: List[Dict[str, Any]],
        best: Dict[str, Any],
        cfg: Dict[str, Any],
        rank: int = 0,
    ) -> None:
        """Lưu checkpoint cho mô hình refiner (49k updates)."""
        if rank != 0:
            return
        ckpt_dir = self.output_dir / "research_49k/checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": step,
            "model": diff.state_dict(),
            "opt": opt.state_dict(),
            "history": history,
            "best": best,
            "config": cfg,
            "protocol": cfg.get("protocol", "polyp_ddpm_master_v400"),
        }
        save_pt(ckpt_dir / "last.pt", payload)
        if step % cfg.get("save_every", 1000) == 0:
            save_pt(ckpt_dir / f"step_{step:06d}.pt", payload)
