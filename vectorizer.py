# app_stremlit/vectorizer.py
from __future__ import annotations
import geopandas as gpd
import pandas as pd
from pathlib import Path

def raster_to_vector(raster_path: str, out_gpkg: str, layer: str = "segments") -> gpd.GeoDataFrame:
    """
    IMPLEMENTATION NOTE:
    Keep your existing polygonization logic here (not shown to avoid duplication).
    This stub assumes you've already built a GeoDataFrame `gdf` in CRS of the raster.

    Replace this stub with your current implementation if present.
    """
    raise NotImplementedError("raster_to_vector() should be implemented in your codebase.")

def summarise(gdf: gpd.GeoDataFrame, out_csv: str) -> pd.DataFrame:
    """
    Create a summary CSV of polygon areas in square metres by reprojecting
    to a metre-based CRS before computing area.

    MODIFIED (~lines 25–55): reprojection to EPSG:6933 for area in m².
    """
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS; cannot compute area.")

    # Reproject to a world metre-based CRS for area calculations
    gdf_m = gdf.to_crs("EPSG:6933").copy()
    gdf_m["area_m2"] = gdf_m.geometry.area

    # Persist summary CSV
    df = gdf_m[["area_m2"]].copy()
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
