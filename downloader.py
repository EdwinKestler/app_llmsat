"""Utilities for downloading imagery."""

from __future__ import annotations

import os
from typing import Sequence
from samgeo import tms_to_geotiff


def download_imagery(bbox: Sequence[float], zoom: int, out_dir: str, *, source: str = "Satellite") -> str:
    """Download basemap imagery within a bounding box as a GeoTIFF.

    Parameters
    ----------
    bbox: Sequence[float]
        [minx, miny, maxx, maxy] in geographic coordinates.
    zoom: int
        Web map zoom level to request.
    out_dir: str
        Directory for the downloaded image.
    source: str, optional
        Basemap source passed to :func:`samgeo.tms_to_geotiff`.

    Returns
    -------
    str
        Path to the downloaded GeoTIFF.
    """
    os.makedirs(out_dir, exist_ok=True)
    image_path = os.path.join(out_dir, "image.tif")
    tms_to_geotiff(output=image_path, bbox=bbox, zoom=zoom, source=source, overwrite=True)
    return image_path
