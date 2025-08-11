# app_stremlit/downloader.py
from __future__ import annotations
from pathlib import Path
from typing import Sequence

try:
    from samgeo import tms_to_geotiff  # real
    _USING_STUB = False
except Exception:
    from samgeo_stub import tms_to_geotiff  # fallback
    _USING_STUB = True


def _normalize_bbox_to_list(bbox: Sequence[float]) -> list[float]:
    if bbox is None or len(bbox) != 4:
        raise ValueError("bbox must be a sequence of 4 numbers: [xmin, ymin, xmax, ymax]")
    xmin, ymin, xmax, ymax = map(float, bbox)
    # normalize order in case user flips coords
    xmin, xmax = min(xmin, xmax), max(xmin, xmax)
    ymin, ymax = min(ymin, ymax), max(ymin, ymax)
    return [xmin, ymin, xmax, ymax]


def download_imagery(
    out_dir: str,
    bbox: Sequence[float],
    *,
    tms_source: str = "satellite",
    zoom: int = 16,
    filename: str = "imagery.tif",
    overwrite: bool = True,
) -> str:
    """
    Downloads (or stubs) TMS imagery into a GeoTIFF at out_dir/filename.
    Ensures bbox is a LIST in [xmin, ymin, xmax, ymax] order for samgeo.
    """
    out_path = Path(out_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bbox_list = _normalize_bbox_to_list(bbox)

    tms_to_geotiff(
        source=tms_source,
        bbox=bbox_list,   # <-- samgeo requires list
        zoom=zoom,
        output=str(out_path),
        overwrite=overwrite,
    )
    return str(out_path)
