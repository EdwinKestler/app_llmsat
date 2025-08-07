"""Minimal stub of the ``samgeo`` package for testing purposes."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin


def tms_to_geotiff(output: str, bbox, zoom: int, source: str = "Satellite", overwrite: bool = True):
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


class SamGeo:
    """Placeholder class mimicking the API of the real ``SamGeo``."""

    def __init__(self, model_type: str | None = None, checkpoint: str | None = None, device: str | None = None, sam_kwargs=None):
        self.model_type = model_type
        self.checkpoint = checkpoint

    def generate(self, source: str, output: str, **kwargs):
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

    def tiff_to_vector(self, mask: str, out_vector: str):
        import geopandas as gpd
        from shapely.geometry import Polygon

        gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs="EPSG:4326")
        gdf.to_file(out_vector, driver="GPKG")
