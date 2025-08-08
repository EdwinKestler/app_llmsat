from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple
import yaml


@dataclass
class PipelineConfig:
    """Configuration options for the processing pipeline.

    Attributes
    ----------
    bbox:
        Bounding box in ``(west, south, east, north)`` order expressed in
        EPSG:4326 coordinates.
    zoom:
        Web mercator zoom level used when downloading imagery tiles.
    out_dir:
        Directory where all generated artefacts are written.
    model_dir:
        Directory containing model checkpoints.
    sam2_checkpoint:
        File name of the SAM2 checkpoint located inside ``model_dir``.
    box_threshold / text_threshold:
        Detection thresholds passed through to the segmentation models.
    """

    bbox: Tuple[float, float, float, float] = (-74.01, 40.70, -73.99, 40.72)
    zoom: int = 18
    out_dir: str = "output"
    model_dir: str = "checkpoints"
    sam2_checkpoint: str = "sam2_hiera_l.pt"
    box_threshold: float = 0.24
    text_threshold: float = 0.24


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
