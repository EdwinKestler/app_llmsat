from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Any
import yaml

@dataclass
class PipelineConfig:
    """Configuration options for the processing pipeline."""
    bbox: tuple[float, float, float, float]
    zoom: int = 18
    out_dir: str = "output"
    model_dir: str = "checkpoints"
    sam2_checkpoint: str = "sam2_hiera_l.pt"
    box_threshold: float = 0.24
    text_threshold: float = 0.24

def load_config(path: Optional[str] = None, **overrides: Any) -> PipelineConfig:
    """Load a :class:`PipelineConfig` from a YAML file."""
    data: dict[str, Any] = {}
    if path:
        with open(path) as f:
            file_data = yaml.safe_load(f) or {}
            if not isinstance(file_data, dict):
                raise TypeError("YAML configuration must be a mapping")
            data.update(file_data)
    data.update(overrides)
    return PipelineConfig(**data)