from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Any
import yaml


@dataclass
class PipelineConfig:
    """Configuration options for the processing pipeline."""

    from pipeline import PipelineConfig

config = PipelineConfig(
    bbox=(-74.01, 40.70, -73.99, 40.72),  # west, south, east, north (EPSG:4326)
    zoom=18,
    out_dir="output",
    model_dir="checkpoints",
    sam2_checkpoint="sam2_hiera_l.pt",
    box_threshold=0.24,
    text_threshold=0.24,
)


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
