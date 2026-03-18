"""Stub LangSAM implementation."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin


class LangSAM:
    def predict(self, image: str, text_prompt: str, box_threshold: float = 0.24, text_threshold: float = 0.24):
        """Placeholder predict method doing nothing."""
        pass

    def show_anns(self, cmap: str, add_boxes: bool, alpha: float, blend: bool, output: str):
        transform = from_origin(0, 0, 1, 1)
        profile = {
            "driver": "GTiff",
            "height": 1,
            "width": 1,
            "count": 1,
            "dtype": "uint8",
            "transform": transform,
        }
        with rasterio.open(output, "w", **profile) as dst:
            dst.write(np.zeros((1, 1, 1), dtype=np.uint8))
