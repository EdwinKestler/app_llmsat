# app_stremlit/pipeline/pipeline.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
from shapely.geometry import mapping

from .config import PipelineConfig
from downloader import download_imagery
from segmenter import Segmenter
# NOTE: your project already has vectorizer.summarise; keep your own polygonization call.
from vectorizer import raster_to_vector, summarise  # Updated import

# Optional: if you have your own raster->vector function
# from vectorizer import raster_to_vector


def _dummy_vectorize(mask_tif: str) -> gpd.GeoDataFrame:
    """
    Placeholder: replace with your real raster→vector function.
    Must return GeoDataFrame with a valid CRS, preferably EPSG:4326.
    """
    # TODO: wire your actual polygonization here.
    # For now, make an empty GDF in EPSG:4326 so the pipeline completes gracefully.
    return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def _export_shapefile(gdf: gpd.GeoDataFrame, shp_dir: Path, name: str = "segments") -> str:
    shp_dir.mkdir(parents=True, exist_ok=True)
    shp_path = shp_dir / f"{name}.shp"
    if not gdf.empty:
        gdf.to_file(shp_path, driver="ESRI Shapefile")
    else:
        # create an empty shapefile safely? Most drivers don't like empty; skip writing.
        pass
    return str(shp_path)


def _export_gpkg(gdf: gpd.GeoDataFrame, out_path: Path, layer: str = "segments") -> str:
    if not gdf.empty:
        gdf.to_file(out_path, layer=layer, driver="GPKG")
    else:
        # write an empty gpkg? skip to avoid corrupt file
        pass
    return str(out_path)


def _export_folium(gdf: gpd.GeoDataFrame, out_path: Path) -> str:
    try:
        import folium
    except Exception:
        return ""

    if gdf.empty:
        # Build an empty map centered roughly on bbox of last run? Leave blank map.
        m = folium.Map(zoom_start=12)
        m.save(str(out_path))
        return str(out_path)

    # Center map on centroid
    center = gdf.to_crs("EPSG:4326").geometry.unary_union.centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=13, control_scale=True)

    # Add polygons
    folium.GeoJson(gdf.to_crs("EPSG:4326")).add_to(m)
    m.save(str(out_path))
    return str(out_path)


def run_pipeline(config: PipelineConfig, *, text_prompts: List[str]) -> Dict[str, str]:
    """
    Standard pipeline:
      1) Download imagery → imagery.tif
      2) Segment (SAM2 preferred) → mask.tif
      3) Vectorize → segments.gpkg, shp/segments.shp
      4) Summarise → summary.csv (area in m² handled in vectorizer.summarise)
      5) Folium HTML → map.html
    """
    Path(config.out_dir).mkdir(parents=True, exist_ok=True)

    # 1) Download imagery
    imagery_path = download_imagery(
        config.out_dir,
        config.bbox,
        tms_source=config.tms_source,
        zoom=config.zoom,
        filename="imagery.tif",
        overwrite=True,
    )

    # 2) Segment
    seg = Segmenter(
        device=config.device,
        model_dir=config.checkpoints_dir,
        sam2_checkpoint=os.path.basename(config.resolved_ckpts.get("sam2") or "") or "sam2_hiera_l.pt",
        box_threshold=config.box_threshold,
        text_threshold=config.text_threshold,
    )
    # NOTE: your Segmenter.run_text_segmentation currently expects an image path & prompts and returns masks meta.
    seg_result = seg.run_text_segmentation(imagery_path, text_prompts=text_prompts)

    # You likely save a mask raster here in your real code. For now we just define a path.
    mask_path = Path(config.out_dir) / "mask.tif"
    if not mask_path.exists():
        # If your segmenter doesn’t write it yet, create a tiny placeholder.
        mask_path.write_bytes(b"")

    # 3) Vectorize
    gpkg_path = Path(config.out_dir) / "segments.gpkg"
    gdf = raster_to_vector(str(mask_path), str(gpkg_path))  # Replace _dummy_vectorize

    # 4) Exports
    shp_dir = Path(config.out_dir) / "shp"
    html_path = Path(config.out_dir) / "map.html"
    csv_path = Path(config.out_dir) / "summary.csv"

    _export_shapefile(gdf, shp_dir)
    _export_folium(gdf, html_path)
    try:
        summarise(gdf, str(csv_path))
    except Exception:
        pass

    return {
        "image": str(imagery_path),
        "sam2_mask": str(mask_path),
        "gpkg": str(gpkg_path),
        "shp": str(shp_dir / "segments.shp"),
        "csv": str(csv_path),
        "map": str(html_path),
    }

    # 4) Exports
    gpkg_path = Path(config.out_dir) / "segments.gpkg"
    shp_dir = Path(config.out_dir) / "shp"
    html_path = Path(config.out_dir) / "map.html"
    csv_path = Path(config.out_dir) / "summary.csv"

    _export_gpkg(gdf, gpkg_path)
    _export_shapefile(gdf, shp_dir)
    try:
        summarise(gdf, str(csv_path))  # writes area_m2 in EPSG:6933 inside summarise()
    except Exception:
        pass
    _export_folium(gdf, html_path)

    return {
        "image": str(imagery_path),
        "sam2_mask": str(mask_path),
        "gpkg": str(gpkg_path),
        "shp": str(shp_dir / "segments.shp"),
        "csv": str(csv_path),
        "map": str(html_path),
    }
