"""Query OpenStreetMap road data for a bounding box via Overpass API.

Roads are fetched as linestrings, buffered to approximate road width,
and returned as a GeoDataFrame for rasterization.
"""

from __future__ import annotations

import requests
import geopandas as gpd
from shapely.geometry import LineString

# Road types to query and their approximate widths in metres.
ROAD_TYPES = {
    "motorway": 12,
    "trunk": 10,
    "primary": 8,
    "secondary": 7,
    "tertiary": 6,
    "residential": 5,
    "service": 3,
    "unclassified": 4,
    "motorway_link": 6,
    "trunk_link": 5,
    "primary_link": 5,
    "secondary_link": 4,
    "tertiary_link": 4,
    "living_street": 4,
    "pedestrian": 3,
    "track": 3,
    "footway": 2,
    "cycleway": 2,
    "path": 1.5,
}

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Road types included in the default query.
_DEFAULT_HIGHWAY_FILTER = (
    "motorway|trunk|primary|secondary|tertiary|residential|"
    "service|unclassified|motorway_link|trunk_link|primary_link|"
    "secondary_link|tertiary_link|living_street"
)


def query_roads(
    bbox: list[float],
    *,
    timeout: int = 60,
) -> gpd.GeoDataFrame:
    """Fetch road geometries from OpenStreetMap for a bounding box.

    Parameters
    ----------
    bbox : list[float]
        ``[west, south, east, north]`` in EPSG:4326.
    timeout : int
        Overpass API timeout in seconds.

    Returns
    -------
    geopandas.GeoDataFrame
        Roads with columns: highway (type), name, width_m, geometry.
    """
    west, south, east, north = bbox

    query = (
        f'[out:json][timeout:{timeout}];'
        f'way["highway"~"^({_DEFAULT_HIGHWAY_FILTER})$"]'
        f'({south},{west},{north},{east});'
        f'out geom;'
    )

    import time
    data = None
    for attempt in range(3):
        resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=timeout + 10)
        if resp.status_code == 200:
            data = resp.json()
            break
        if resp.status_code in (429, 504):
            time.sleep(5 * (attempt + 1))
            continue
        resp.raise_for_status()

    if data is None:
        return gpd.GeoDataFrame(columns=["highway", "name", "width_m", "geometry"])

    elements = data.get("elements", [])
    if not elements:
        return gpd.GeoDataFrame(columns=["highway", "name", "width_m", "geometry"])

    records = []
    for el in elements:
        geom_nodes = el.get("geometry", [])
        if len(geom_nodes) < 2:
            continue
        coords = [(n["lon"], n["lat"]) for n in geom_nodes]
        line = LineString(coords)
        tags = el.get("tags", {})
        highway_type = tags.get("highway", "unclassified")
        name = tags.get("name", "")
        width = ROAD_TYPES.get(highway_type, 4)

        records.append({
            "highway": highway_type,
            "name": name,
            "width_m": width,
            "geometry": line,
        })

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    return gdf


def buffer_roads(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Buffer road linestrings to polygons based on road width.

    Projects to UTM for accurate metre-based buffering, then
    reprojects back to EPSG:4326.
    """
    if gdf.empty:
        return gdf

    # Estimate UTM zone from centroid
    centroid = gdf.geometry.union_all().centroid
    zone = int((centroid.x + 180) / 6) + 1
    epsg = 32600 + zone if centroid.y >= 0 else 32700 + zone
    utm_crs = f"EPSG:{epsg}"

    projected = gdf.to_crs(utm_crs)
    # Buffer each road by half its width (width = total road width)
    projected["geometry"] = projected.apply(
        lambda row: row.geometry.buffer(row["width_m"] / 2, cap_style="flat"),
        axis=1,
    )

    return projected.to_crs("EPSG:4326")


def roads_summary(gdf: gpd.GeoDataFrame) -> dict:
    """Return a summary dict for display."""
    if gdf.empty:
        return {"count": 0, "types": {}}

    types = gdf["highway"].value_counts().to_dict()
    return {
        "count": len(gdf),
        "types": types,
    }
