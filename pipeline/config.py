# app_stremlit/pipeline/config.py
from __future__ import annotations
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Tuple, Dict, Optional


def _ts() -> str:
    # Windows-safe timestamp (no ":"), e.g., 2025-08-08_00-35-21
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def _normalize_bbox(bbox: Iterable[float]) -> Tuple[float, float, float, float]:
    west, south, east, north = map(float, bbox)
    if east < west:
        west, east = east, west
    if north < south:
        south, north = north, south
    return (west, south, east, north)


@dataclass
class PipelineConfig:
    # IO roots
    data_dir: str = "data"                         # root where all outputs go
    out_dir: str = ""                              # concrete run folder (auto if empty)
    checkpoints_dir: str = "checkpoints"           # where models live

    # Region of interest
    bbox: Tuple[float, float, float, float] = (-90.6, 14.58, -90.5, 14.66)

    # Device / thresholds
    device: str = "cuda"
    box_threshold: float = 0.24
    text_threshold: float = 0.24

    # Imagery
    tms_source: str = "satellite"                  # customize (e.g., "Esri.WorldImagery")
    zoom: int = 16

    # Checkpoints (names you actually have on disk)
    sam2_checkpoint: Optional[str] = None          # e.g., "sam2_hiera_large.pt" or "sam2_hiera_l.pt"
    sam_checkpoint: Optional[str] = None           # e.g., "sam_vit_h_4b8939.pth"
    mobile_sam_checkpoint: Optional[str] = None    # e.g., "mobile_sam.pt"

    # Resolved absolute paths (filled in after init)
    resolved_ckpts: Dict[str, Optional[str]] = field(default_factory=dict)

    def ensure_dirs(self) -> None:
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoints_dir).mkdir(parents=True, exist_ok=True)

    def resolve_checkpoints(self) -> None:
        """
        Resolve checkpoints by filename presence under checkpoints_dir.
        Keys: 'sam2', 'sam', 'mobile_sam'.
        Preference for SAM2: 'sam2_hiera_large.pt' > 'sam2_hiera_l.pt'.
        """
        ckproot = Path(self.checkpoints_dir)

        # SAM2
        cands_sam2 = []
        if self.sam2_checkpoint:
            cands_sam2.append(self.sam2_checkpoint)
        cands_sam2 += ["sam2_hiera_large.pt", "sam2_hiera_l.pt"]

        sam2_path = None
        for name in cands_sam2:
            p = ckproot / name
            if p.exists() and p.is_file() and p.suffix.lower() == ".pt":
                sam2_path = str(p)
                break

        # Classic SAM
        sam_path = None
        if self.sam_checkpoint:
            p = ckproot / self.sam_checkpoint
            if p.exists() and p.is_file():
                sam_path = str(p)
        else:
            p = ckproot / "sam_vit_h_4b8939.pth"
            if p.exists() and p.is_file():
                sam_path = str(p)

        # Mobile SAM
        mobile_path = None
        if self.mobile_sam_checkpoint:
            p = ckproot / self.mobile_sam_checkpoint
            if p.exists() and p.is_file():
                mobile_path = str(p)
        else:
            p = ckproot / "mobile_sam.pt"
            if p.exists() and p.is_file():
                mobile_path = str(p)

        self.resolved_ckpts = {
            "sam2": sam2_path,
            "sam": sam_path,
            "mobile_sam": mobile_path,
        }

    def to_dict(self) -> Dict:
        return asdict(self)


def load_config(
    *,
    bbox: Iterable[float],
    out_dir: Optional[str] = None,
    data_dir: str = "data",
    checkpoints_dir: str = "checkpoints",
    device: str = "cuda",
    box_threshold: float = 0.24,
    text_threshold: float = 0.24,
    tms_source: str = "satellite",
    zoom: int = 16,
    sam2_checkpoint: Optional[str] = None,
    sam_checkpoint: Optional[str] = None,
    mobile_sam_checkpoint: Optional[str] = None,
    # ---- Backward-compat alias ----
    model_dir: Optional[str] = None,
) -> PipelineConfig:
    """
    NOTE: 'model_dir' is a deprecated alias for 'checkpoints_dir'.
    If both are provided, 'checkpoints_dir' wins.
    """
    # Backward-compat: if old callers pass model_dir, use it unless checkpoints_dir was explicitly set differently
    if model_dir and checkpoints_dir == "checkpoints":
        checkpoints_dir = model_dir

    bbox_n = _normalize_bbox(bbox)

    # Default run folder under data/runs/<timestamp>
    if not out_dir:
        out_root = Path(data_dir) / "runs"
        out_dir = str(out_root / _ts())

    cfg = PipelineConfig(
        data_dir=data_dir,
        out_dir=out_dir,
        checkpoints_dir=checkpoints_dir,
        bbox=bbox_n,
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        tms_source=tms_source,
        zoom=zoom,
        sam2_checkpoint=sam2_checkpoint,
        sam_checkpoint=sam_checkpoint,
        mobile_sam_checkpoint=mobile_sam_checkpoint,
    )

    cfg.ensure_dirs()
    cfg.resolve_checkpoints()
    return cfg
