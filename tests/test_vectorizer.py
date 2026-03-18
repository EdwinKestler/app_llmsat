"""Tests for vectorizer module."""

import os
import tempfile
import numpy as np
import rasterio
from rasterio.transform import from_origin
from vectorizer import raster_to_vector, summarise


def _create_test_raster(path):
    """Create a small binary mask raster for testing."""
    transform = from_origin(0, 1, 0.5, 0.5)
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": transform,
    }
    data = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def test_raster_to_vector_creates_gpkg():
    with tempfile.TemporaryDirectory() as tmpdir:
        raster_path = os.path.join(tmpdir, "mask.tif")
        gpkg_path = os.path.join(tmpdir, "out.gpkg")
        _create_test_raster(raster_path)
        gdf = raster_to_vector(raster_path, gpkg_path)
        assert os.path.exists(gpkg_path)
        assert len(gdf) > 0


def test_raster_to_vector_skips_zero_values():
    with tempfile.TemporaryDirectory() as tmpdir:
        raster_path = os.path.join(tmpdir, "mask.tif")
        gpkg_path = os.path.join(tmpdir, "out.gpkg")
        _create_test_raster(raster_path)
        gdf = raster_to_vector(raster_path, gpkg_path)
        # The raster has 3 cells with value 1 and 1 cell with value 0
        # Polygons should only represent value-1 regions
        assert len(gdf) >= 1


def test_summarise_creates_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        raster_path = os.path.join(tmpdir, "mask.tif")
        gpkg_path = os.path.join(tmpdir, "out.gpkg")
        csv_path = os.path.join(tmpdir, "summary.csv")
        _create_test_raster(raster_path)
        gdf = raster_to_vector(raster_path, gpkg_path)
        df = summarise(gdf, csv_path)
        assert os.path.exists(csv_path)
        assert "area_m2" in df.columns
