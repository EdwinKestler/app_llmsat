from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Any
import yaml


@dataclass
class PipelineConfig:
    """Configuration options for the processing pipeline."""

    bbox: Iterable[float] = (0.0, 0.0, 0.0, 0.0)
    zoom: int = 18
    out_dir: str = "data"
    model_dir: str = "models"
    box_threshold: float = 0.24
    text_threshold: float = 0.24
    sam2_checkpoint: str = "sam2_hiera_l.pt"


def load_config(path: Optional[str] = None, **overrides: Any) -> PipelineConfig:
    """Load a :class:`PipelineConfig` from a YAML file.

    Parameters
    ----------
    path: str, optional
        Path to a YAML file with configuration values. Missing values
        fall back to defaults from :class:`PipelineConfig`.
    **overrides: Any
        Additional keyword arguments used to override both defaults and
        values loaded from ``path``. This is useful for CLI or
        Streamlit-based overrides.
    """
    data: dict[str, Any] = {}
    if path:
        with open(path) as f:
            file_data = yaml.safe_load(f) or {}
            if not isinstance(file_data, dict):
                raise TypeError("YAML configuration must be a mapping")
            data.update(file_data)
    data.update(overrides)
    return PipelineConfig(**data)
