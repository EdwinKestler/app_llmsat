"""Minimal stub of the ``samgeo`` package for testing purposes."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin


class SamGeo3:
    """Placeholder class mimicking the API of the real ``SamGeo3``."""

    def __init__(self, model_id: str = "facebook/sam3", **kwargs):
        self.model_id = model_id
        self._masks = None

    def set_image(self, image, **kwargs):
        self._image = image

    def generate_masks(self, prompt: str, **kwargs):
        self._masks = True
        return [{"segmentation": np.zeros((1, 1), dtype=np.uint8)}]

    def save_masks(self, output: str = None, **kwargs):
        if output is None:
            return
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

    def raster_to_vector(self, raster: str, vector: str, **kwargs):
        import geopandas as gpd
        from shapely.geometry import Polygon

        gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs="EPSG:4326")
        gdf.to_file(vector, driver="GPKG")
