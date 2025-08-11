# app_stremlit/vectorizer.py
from __future__ import annotations
import geopandas as gpd
import pandas as pd
from pathlib import Path
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

def raster_to_vector(raster_path: str, out_gpkg: str, layer: str = "segments") -> gpd.GeoDataFrame:
    """
    Convert a raster mask to vector polygons and save to GeoPackage.
    """
    with rasterio.open(raster_path) as src:
        if src.count > 1:
            raise ValueError("Expected single-band raster")
        mask = src.read(1)  # Read the first band
        transform = src.transform
        crs = src.crs or "EPSG:4326"  # Default to EPSG:4326 if no CRS

        # Convert raster to shapes
        results = shapes(mask, transform=transform)
        geometries = [shape(geom) for geom, value in results if value != 0]  # Exclude background (0)

    gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs)
    if not gdf.empty:
        Path(out_gpkg).parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(out_gpkg, layer=layer, driver="GPKG")
    
    return gdf

def summarise(gdf: gpd.GeoDataFrame, out_csv: str) -> pd.DataFrame:
    """
    Create a summary CSV of polygon areas in square metres by reprojecting
    to a metre-based CRS before computing area.
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
