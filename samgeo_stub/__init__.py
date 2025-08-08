# app_stremlit/samgeo_stub/__init__.py
"""
Local stub for selected samgeo API calls so code can run without the real library.
"""

from pathlib import Path
from typing import Optional

def tms_to_geotiff(
    source: str,
    bbox: tuple[float, float, float, float],
    zoom: int,
    output: str,
    overwrite: bool = True,
    **kwargs,
) -> str:
    """
    Minimal placeholder for samgeo.tms_to_geotiff.
    Writes an empty file so downstream steps don't crash.
    Replace with real imagery download if needed.
    """
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_bytes(b"")  # placeholder
    print(f"[samgeo_stub] tms_to_geotiff called with source={source}, bbox={bbox}, zoom={zoom}")
    print(f"[samgeo_stub] Created placeholder GeoTIFF at {output}")
    return output

# Keep LangSAM fallback import so Segmenter still works without real samgeo
from .text_sam import LangSAM  # noqa: E402
