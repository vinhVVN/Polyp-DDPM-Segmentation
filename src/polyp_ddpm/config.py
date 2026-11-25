"""Configuration schemas and default settings for Polyp-DDPM.

Bảo toàn toàn bộ tham số thí nghiệm, seed, cấu hình mạng và đường dẫn lịch sử
theo đúng thiết kế thực nghiệm đã công bố.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import os
import copy


@dataclass
class MasterConfig:
    """Cấu hình tổng cho pipeline nghiên cứu Polyp-DDPM."""
    action: str = "PRESENT_ONLY"  # "PRESENT_ONLY", "REPORT_AND_EXTEND", "REBUILD"
    input_root: str = "/kaggle/input"
    output_dir: str = "/kaggle/working/polyp_ddpm_report_master_v400"
    resume_master_output: Optional[str] = (
        "/kaggle/input/notebooks/trumbomhang/polypddpm-master-full/polyp_ddpm_report_master_v400"
    )
    session_hours: float = 9.0
    install_missing: bool = True

    # Đường dẫn dữ liệu và checkpoints
    paths: Dict[str, Optional[str]] = field(default_factory=lambda: {
        "images": "/kaggle/input/datasets/debeshjha1/kvasirseg/Kvasir-SEG/Kvasir-SEG/images",
        "masks": "/kaggle/input/datasets/debeshjha1/kvasirseg/Kvasir-SEG/Kvasir-SEG/masks",
        "own": "/kaggle/input/notebooks/thnhvinhs/polyp-ddpm-master-end-to-end-2/polyp_ddpm_report_master_recovery_v130",
        "color": "/kaggle/input/notebooks/thnhvinhs/polyp-ddpm-master-end-to-end-3/polyp_ddpm_trained_color_repair_v141",
        "texture_v1": "/kaggle/input/notebooks/thnhvinhs/polyp-ddpm-texture-refinement/polyp_ddpm_texture_t4x2_v100",
        "v2": "/kaggle/input/notebooks/thnhvinhs/polyp-ddpm-restoration/polyp_ddpm_background_repair_v200",
        "v21": "/kaggle/input/notebooks/thnhvinhs/polypddpm-v2-continuation-boundary-protected/polyp_ddpm_v2_continuation_v210",
        "author": "/kaggle/input/datasets/thnhvinhs/polyp-ddpm-author/polyp_ddpm_author.pt",
    })

    # Siêu tham số huấn luyện và mô hình
    settings: Dict[str, Any] = field(default_factory=lambda: {
        "seed": 42,
        "split_seed": 42,
        "size": 256,
        "run_additional_refiner": True,
        "continuation_from": "last",
        "continuation_lr": 5e-07,
        "own_steps": 10000,
        "own_lr": 0.0001,
        "own_per_gpu_batch": 4,
        "own_accumulation": 4,
        "color_steps": 2000,
        "color_lr": 0.0002,
        "color_batch": 8,
        "color_train_size": 128,
        "lut_dim": 17,
        "color_split_seed": 14042,
        "color_val_images": 90,
        "color_cache_train": 32,
        "color_cache_val": 8,
        "color_validation_n": 16,
        "color_validate_every": 200,
        "color_save_every": 100,
        "rebuild_refiner_steps": 1000,
        "demo_cases": 4,
        "demo_replicates": 3,
        "quality_n": 100,
        "fid_enabled": False,
        "seg_steps": 2000,
        "seg_batch": 8,
        "seg_lr": 0.0001,
        "encoder_weights": "imagenet",
        "eval_batch": 8,
        "prediction_cases": 6,
        "save_all_predictions": True,
        "save_every": 100,
        "cpu_threads_per_rank": 2,
        "disable_p2p": True,
        "host_rss_limit_gib": 20.0,
        "host_free_min_gib": 3.0,
        "host_memory_fraction": 0.8,
        "refiner_target_total": 49000,
        "milestone_totals": [3500, 10000, 20000, 30000, 40000, 49000],
    })

    def resolve_paths(self, local_base: Optional[Path] = None):
        """Tự động kiểm tra và giải quyết đường dẫn cục bộ nếu không chạy trên Kaggle."""
        base = local_base or Path.cwd()
        resolved = copy.deepcopy(self.paths)
        for k, v in resolved.items():
            if v and not Path(v).exists():
                # Tìm kiếm tương đối trong workspace
                candidate = base / Path(v).name
                if candidate.exists():
                    resolved[k] = str(candidate)
        return resolved


@dataclass
class EvaluationConfig:
    """Cấu hình đánh giá downstream segmentation và FID/KID/IS."""
    output_dir: str = "/kaggle/working/polyp_ddpm_evaluation_v8"
    input_root: str = "/kaggle/input"
    master_root: Optional[str] = None
    real_images: Optional[str] = None
    real_masks: Optional[str] = None
    resume_evaluation: Optional[str] = None

    architectures: List[str] = field(default_factory=lambda: ["unetpp", "fpn", "deeplabv3plus"])
    seeds: List[int] = field(default_factory=lambda: [42])
    train_epochs: int = 100
    batch_size: int = 8
    amp: bool = False
    max_workers: int = 2

    preprocessing_mode: str = "reference_code"
    checkpoint_policy: str = "best_kvasir_validation"

    external: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "HyperKvasir": {"images": None, "masks": None, "expected_n": 1000, "mask_prefix": ""},
        "ETIS": {"images": None, "masks": None, "expected_n": 196, "mask_prefix": ""},
    })

    baselines: Dict[str, Dict[str, Optional[str]]] = field(default_factory=lambda: {
        "LDM": {"pairs_csv": None, "quality_images": None},
        "SinGAN-Seg": {"pairs_csv": None, "quality_images": None},
    })

    repaired_pairs_csv: Optional[str] = None
    repaired_label: str = "Checkpoint chỉnh texture và nền"
