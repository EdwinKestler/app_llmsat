"""Query Google Open Buildings data for a bounding box.

Uses DuckDB to efficiently scan compressed CSV tiles from Google's
Open Buildings v3 dataset.  The tiles must be downloaded to a local
directory (default ``openbuildings/``).

Main entry point: :func:`query_buildings` returns a GeoDataFrame of
building polygons within a given bbox.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import duckdb
import geopandas as gpd
import pandas as pd
from shapely import wkt


# Default directory containing ``*_buildings.csv.gz`` tiles.
DEFAULT_TILE_DIR = os.path.join(os.path.dirname(__file__), "openbuildings")


def _find_tile_files(tile_dir: str) -> list[str]:
    """Return paths to all ``*_buildings.csv.gz`` files in *tile_dir*."""
    d = Path(tile_dir)
    if not d.is_dir():
        return []
    return sorted(str(p) for p in d.glob("*_buildings.csv.gz"))


def _find_relevant_tiles(tile_dir: str, bbox: list[float]) -> list[str]:
    """Return only tile files whose geographic extent overlaps *bbox*.

    Uses ``tiles.geojson`` index if available; falls back to all tiles.
    """
    import json

    geojson_path = Path(tile_dir) / "tiles.geojson"
    all_tiles = _find_tile_files(tile_dir)

    if not geojson_path.exists():
        return all_tiles

    west, south, east, north = bbox
    try:
        with open(geojson_path) as f:
            index = json.load(f)
    except (json.JSONDecodeError, OSError):
        return all_tiles

    relevant_ids = set()
    for feat in index.get("features", []):
        coords = feat.get("geometry", {}).get("coordinates", [[]])[0]
        if not coords:
            continue
        lngs = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        t_west, t_south, t_east, t_north = min(lngs), min(lats), max(lngs), max(lats)
        # Check overlap
        if t_east >= west and t_west <= east and t_north >= south and t_south <= north:
            tile_id = feat.get("properties", {}).get("tile_id", "")
            if tile_id:
                relevant_ids.add(tile_id)

    if not relevant_ids:
        return all_tiles  # Fallback: can't determine, scan all

    return [t for t in all_tiles if any(tid in Path(t).stem for tid in relevant_ids)]


def query_buildings(
    bbox: list[float],
    *,
    min_confidence: float = 0.7,
    tile_dir: str = DEFAULT_TILE_DIR,
    max_buildings: int = 10_000,
) -> gpd.GeoDataFrame:
    """Query Open Buildings for a bounding box.

    Parameters
    ----------
    bbox : list[float]
        ``[west, south, east, north]`` in EPSG:4326.
    min_confidence : float
        Minimum confidence score (0–1). Default 0.7.
    tile_dir : str
        Directory containing ``*_buildings.csv.gz`` tile files.
    max_buildings : int
        Safety limit on returned rows.

    Returns
    -------
    geopandas.GeoDataFrame
        Building polygons with columns: latitude, longitude,
        area_in_meters, confidence, geometry.
    """
    tile_files = _find_relevant_tiles(tile_dir, bbox)
    if not tile_files:
        return gpd.GeoDataFrame(columns=["geometry", "area_in_meters", "confidence"])

    # Validate bbox values are safe floats
    try:
        west, south, east, north = [float(x) for x in bbox]
        min_confidence = float(min_confidence)
        max_buildings = int(max_buildings)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid query parameters: {e}")

    import math
    if any(not math.isfinite(x) for x in [west, south, east, north, min_confidence]):
        raise ValueError("Query parameters must be finite numbers")
    if not (-180 <= west <= 180 and -180 <= east <= 180):
        raise ValueError("Longitude must be between -180 and 180")
    if not (-90 <= south <= 90 and -90 <= north <= 90):
        raise ValueError("Latitude must be between -90 and 90")

    con = duckdb.connect()

    # Validate tile file paths are within expected directory
    tile_dir_resolved = Path(tile_dir).resolve()
    for tf in tile_files:
        if not Path(tf).resolve().is_relative_to(tile_dir_resolved):
            raise ValueError(f"Tile path escapes directory: {tf}")

    # Use parameterized queries to prevent SQL injection
    queries = []
    for tf in tile_files:
        queries.append(f"""
            SELECT latitude, longitude, area_in_meters, confidence, geometry
            FROM read_csv_auto('{tf}', compression='gzip')
            WHERE longitude BETWEEN $west AND $east
              AND latitude BETWEEN $south AND $north
              AND confidence >= $min_conf
        """)

    sql = " UNION ALL ".join(queries) + " LIMIT $max_rows"
    params = {
        "west": west, "east": east,
        "south": south, "north": north,
        "min_conf": min_confidence,
        "max_rows": max_buildings,
    }
    df = con.execute(sql, params).fetchdf()
    con.close()

    if df.empty:
        return gpd.GeoDataFrame(columns=["geometry", "area_in_meters", "confidence"])

    # Parse WKT geometry
    df["geometry"] = df["geometry"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    return gdf


def buildings_to_boxes(
    gdf: gpd.GeoDataFrame,
) -> list[list[float]]:
    """Convert building polygons to bounding boxes for SAM3.

    Returns a list of ``[xmin, ymin, xmax, ymax]`` boxes in EPSG:4326.
    """
    boxes = []
    for geom in gdf.geometry:
        b = geom.bounds  # (minx, miny, maxx, maxy)
        boxes.append([b[0], b[1], b[2], b[3]])
    return boxes


def buildings_summary(gdf: gpd.GeoDataFrame) -> dict:
    """Return a summary dict for display."""
    if gdf.empty:
        return {"count": 0, "total_area_m2": 0, "avg_confidence": 0}
    return {
        "count": len(gdf),
        "total_area_m2": gdf["area_in_meters"].sum(),
        "avg_area_m2": gdf["area_in_meters"].mean(),
        "avg_confidence": gdf["confidence"].mean(),
        "min_confidence": gdf["confidence"].min(),
        "max_confidence": gdf["confidence"].max(),
    }
