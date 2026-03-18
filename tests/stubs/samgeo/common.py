"""Stub for samgeo.common — provides tms_to_geotiff."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin


def tms_to_geotiff(output: str, bbox, zoom: int, source: str = "Satellite", overwrite: bool = True, **kwargs):
    """Create a dummy GeoTIFF file representing downloaded imagery."""
    transform = from_origin(0, 0, 1, 1)
    profile = {
        "driver": "GTiff",
        "height": 1,
        "width": 1,
        "count": 3,
        "dtype": "uint8",
        "transform": transform,
    }
    with rasterio.open(output, "w", **profile) as dst:
        data = np.zeros((3, 1, 1), dtype=np.uint8)
        dst.write(data)
