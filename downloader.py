# app_stremlit/downloader.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

try:
    from samgeo import tms_to_geotiff  # real
    _USING_STUB = False
except Exception:
    from samgeo_stub import tms_to_geotiff  # fallback
    _USING_STUB = True


def download_imagery(
    out_dir: str,
    bbox: tuple[float, float, float, float],
    *,
    tms_source: str = "satellite",
    zoom: int = 16,
    filename: str = "imagery.tif",
    overwrite: bool = True,
) -> str:
    """
    Downloads (or stubs) TMS imagery into a GeoTIFF at out_dir/filename.
    """
    out_path = Path(out_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # The real samgeo.tms_to_geotiff accepts (source, bbox, zoom, output)
    tms_to_geotiff(
        source=tms_source,
        bbox=bbox,
        zoom=zoom,
        output=str(out_path),
        overwrite=overwrite,
    )
    return str(out_path)
