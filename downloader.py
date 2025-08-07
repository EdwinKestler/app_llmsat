"""Utilities for downloading imagery."""

from __future__ import annotations

import os
from samgeo import tms_to_geotiff
from pipeline.config import PipelineConfig


def download_imagery(config: PipelineConfig, *, source: str = "Satellite") -> str:
    """Download basemap imagery within a bounding box as a GeoTIFF.

    Parameters
    ----------
    config: PipelineConfig
        Configuration with bounding box, zoom level and output directory.
    source: str, optional
        Basemap source passed to :func:`samgeo.tms_to_geotiff`.

    Returns
    -------
    str
        Path to the downloaded GeoTIFF.
    """
    os.makedirs(config.out_dir, exist_ok=True)
    image_path = os.path.join(config.out_dir, "s2harm_rgb_saa.tif")
    tms_to_geotiff(
        output=image_path,
        bbox=config.bbox,
        zoom=config.zoom,
        source=source,
        overwrite=True,
        crs="EPSG:4326"  # Explicitly set CRS
    )
    return image_path
