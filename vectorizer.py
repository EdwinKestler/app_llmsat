"""Vectorisation, summary, and export helpers."""

from __future__ import annotations

import json
import os
from typing import Optional

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping


def raster_to_vector(raster_path: str, out_path: str) -> gpd.GeoDataFrame:
    """Convert a binary mask raster to vector polygons saved to ``out_path``.

    Parameters
    ----------
    raster_path: str
        Path to the mask raster.
    out_path: str
        Destination GeoPackage path.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame of polygons derived from the mask.
    """
    with rasterio.open(raster_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs

    geoms = []
    for geom, value in shapes(mask, transform=transform):
        if value == 0:
            continue
        geoms.append(shape(geom))

    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    gdf.to_file(out_path, driver="GPKG")
    return gdf


def summarise(gdf: gpd.GeoDataFrame, out_csv: str) -> pd.DataFrame:
    """Create a summary CSV of polygon areas in square metres."""
    if gdf.empty:
        df = pd.DataFrame(columns=["area_m2"])
        df.to_csv(out_csv, index=False)
        return gdf
    summary = gdf.copy()
    centroid = summary.geometry.union_all().centroid
    zone = int((centroid.x + 180) / 6) + 1
    epsg = 32600 + zone if centroid.y >= 0 else 32700 + zone
    projected = summary.to_crs(f"EPSG:{epsg}")
    summary["area_m2"] = projected.geometry.area
    summary[["area_m2"]].to_csv(out_csv, index=False)
    return summary


def export_geojson(
    segment_dirs: dict[str, str],
    bbox: Optional[list[float]] = None,
) -> str:
    """Export all segments as a single GeoJSON FeatureCollection.

    Each feature has properties: ``segment``, ``area_m2``, ``instance_id``.
    Compatible with GitHub Gists (auto-renders map), QGIS, kepler.gl,
    Mapbox, Google Earth, and any GIS software.

    Parameters
    ----------
    segment_dirs : dict[str, str]
        Mapping of segment name to its output directory
        (e.g. ``{"tree": "output/tree", "building": "output/building"}``).
    bbox : list[float], optional
        ``[west, south, east, north]`` for metadata.

    Returns
    -------
    str
        GeoJSON string.
    """
    all_features = []

    for seg_name, seg_dir in segment_dirs.items():
        gpkg_path = os.path.join(seg_dir, "segments.gpkg")
        if not os.path.exists(gpkg_path):
            continue

        gdf = gpd.read_file(gpkg_path)
        if gdf.empty:
            continue

        # Compute area in UTM
        centroid = gdf.geometry.union_all().centroid
        zone = int((centroid.x + 180) / 6) + 1
        epsg = 32600 + zone if centroid.y >= 0 else 32700 + zone
        projected = gdf.to_crs(f"EPSG:{epsg}")
        areas = projected.geometry.area

        for idx, (_, row) in enumerate(gdf.iterrows()):
            feature = {
                "type": "Feature",
                "properties": {
                    "segment": seg_name,
                    "instance_id": idx + 1,
                    "area_m2": round(float(areas.iloc[idx]), 2),
                },
                "geometry": mapping(row.geometry),
            }
            all_features.append(feature)

    collection = {
        "type": "FeatureCollection",
        "features": all_features,
    }

    # Add metadata
    if bbox:
        collection["bbox"] = bbox
    collection["properties"] = {
        "generator": "LLMSat",
        "segments": list(segment_dirs.keys()),
        "total_features": len(all_features),
    }

    return json.dumps(collection, indent=2)


def export_per_segment_geojson(seg_name: str, seg_dir: str) -> str:
    """Export a single segment as GeoJSON.

    Returns GeoJSON string, or empty FeatureCollection if no data.
    """
    return export_geojson({seg_name: seg_dir})
