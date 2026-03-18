"""Utilities for downloading imagery."""

from __future__ import annotations

import os
from samgeo.common import tms_to_geotiff
from pipeline.config import PipelineConfig


def download_imagery(
    config: PipelineConfig,
    *,
    source: str = "Satellite",
    overwrite: bool = False,
) -> str:
    """Download basemap imagery within a bounding box as a GeoTIFF.

    If the output file already exists and *overwrite* is ``False`` the
    download is skipped and the existing path is returned.

    Parameters
    ----------
    config: PipelineConfig
        Configuration with bounding box, zoom level and output directory.
    source: str, optional
        Basemap source passed to :func:`samgeo.tms_to_geotiff`.
    overwrite: bool, optional
        Re-download even if the file already exists (default ``False``).

    Returns
    -------
    str
        Path to the downloaded GeoTIFF.
    """
    os.makedirs(config.out_dir, exist_ok=True)
    image_path = os.path.join(config.out_dir, "s2harm_rgb_saa.tif")
    if not overwrite and os.path.exists(image_path):
        return image_path
    tms_to_geotiff(
        output=image_path,
        bbox=config.bbox,
        zoom=config.zoom,
        source=source,
        overwrite=True,
        crs="EPSG:4326",
    )
    return image_path
