#!/usr/bin/env python3
"""Geospatial analysis pipeline for segmentation and interactive mapping using GEE, SAM, and LangSAM."""

import os
import argparse
import logging
import sys
import tempfile
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict
import requests
from dotenv import load_dotenv
from getpass import getpass
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import box
from samgeo import SamGeo, tms_to_geotiff
from samgeo.text_sam import LangSAM
import folium
from folium import FeatureGroup, LayerControl, GeoJson
from rasterio.features import shapes
import torch
import base64
import io
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import ee
from google.oauth2 import service_account
import geemap
from tqdm import tqdm

# === Configuration ===
CONFIG = {
    'key_path': 'gee_auth/analytics-416905-88fa0828f9a2.json',
    'project_id': 'analytics-416905',
    'data_dir': 'data',
    'output_dir': 'output',
    'checkpoint_dir': 'checkpoints',
    'min_area': 2,
    'bbox': [-90.015147, 14.916566, -90.010159, 14.919471],
    'zoom': 19,
    'prompt_thresholds': {
        'tree': {'box_threshold': 0.3, 'text_threshold': 0.24},
        'water': {'box_threshold': 0.24, 'text_threshold': 0.24},
        'building': {'box_threshold': 0.28, 'text_threshold': 0.25},
        'road': {'box_threshold': 0.24, 'text_threshold': 0.2}
    },
    'gee_export': {
        's2_date_range': (
            (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
            datetime.now().strftime('%Y-%m-%d')
        ),
        's2_cloud_percentage': 50,
        'dem_scale': 30,
        'lc_scale': 50,
        'precip_scale': 500
    }
}

# Load environment variables
load_dotenv()

# Initialize logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
def setup_logging(output_dir: str, timestamp: str) -> logging.Logger:
    """Configure logging to file and console with timestamped output."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, f'process_{timestamp}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Logging initialized to {log_file}")
        return logger
    except Exception as e:
        print(f"Error setting up logging: {e}")
        sys.exit(1)

logger = setup_logging(CONFIG['output_dir'], timestamp)

# === CDSE Authentication and Date Query ===
def get_cdse_access_token(username: str, password: str, client_id: str = "cdse-public", totp: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Obtain an access token from the CDSE OAuth2 endpoint."""
    token_file = os.path.join(CONFIG['output_dir'], 'cdse_token.json')
    
    if os.path.exists(token_file):
        try:
            with open(token_file, 'r') as f:
                token_data = json.load(f)
            if token_data.get('access_token'):
                logger.info("Using cached CDSE access token")
                return token_data
        except Exception as e:
            logger.warning(f"Error reading cached token: {e}")

    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "username": username,
        "password": password,
        "grant_type": "password",
        "client_id": client_id
    }
    if totp:
        data["totp"] = totp

    try:
        response = requests.post(url, headers=headers, data=data, timeout=10)
        response.raise_for_status()
        token_data = response.json()
        if "access_token" not in token_data:
            logger.error("No access token in response")
            return None
        logger.info("Successfully obtained CDSE access token")
        with open(token_file, 'w') as f:
            json.dump({"access_token": token_data["access_token"], "refresh_token": token_data.get("refresh_token")}, f)
        return {
            "access_token": token_data["access_token"],
            "refresh_token": token_data.get("refresh_token")
        }
    except requests.HTTPError as e:
        logger.error(f"HTTP error obtaining CDSE access token: {e.response.status_code} {e.response.text}")
        return None
    except requests.RequestException as e:
        logger.error(f"Error obtaining CDSE access token: {e}")
        return None

def get_latest_sentinel2_date(bbox: list, access_token: str, max_cloud_cover: int = 50) -> Optional[str]:
    """Query the CDSE Catalog API for the latest Sentinel-2 date."""
    cloud_limits = [max_cloud_cover, 80, 100]
    for cloud_limit in cloud_limits:
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            url = (
                "https://catalogue.dataspace.copernicus.eu/resto/api/collections/Sentinel2/search.json?"
                f"box={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
                f"&startDate={start_date}T00:00:00Z"
                f"&completionDate={end_date}T23:59:59Z"
                f"&cloudCover=[0,{cloud_limit}]"
                "&sortParam=startDate&sortOrder=descending&maxRecords=1"
            )
            headers = {"Authorization": f"Bearer {access_token}"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if not data.get('features'):
                logger.warning(f"No Sentinel-2 data found with cloud cover <= {cloud_limit}%.")
                continue
            latest_date = data['features'][0]['properties']['startDate'].split('T')[0]
            logger.info(f"Latest Sentinel-2 date found with cloud cover <= {cloud_limit}%: {latest_date}")
            return latest_date
        except requests.HTTPError as e:
            logger.error(f"HTTP error querying CDSE Catalog API: {e.response.status_code} {e.response.text}")
        except requests.RequestException as e:
            logger.error(f"Error querying CDSE Catalog API with cloud cover {cloud_limit}%: {e}")
    logger.warning("No Sentinel-2 data found after trying all cloud cover limits.")
    return None

def get_latest_sentinel2_date_fallback(bbox: list, max_cloud_cover: int = 50) -> Optional[str]:
    """Query GEE for the latest Sentinel-2 date as a fallback."""
    try:
        region = ee.Geometry.Rectangle(bbox)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(region) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_cover)) \
            .sort('system:time_start', False)
        latest_image = collection.first()
        if latest_image is None:
            logger.warning("No Sentinel-2 data found in GEE for the given bbox and time range.")
            return None
        latest_date = ee.Date(latest_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        logger.info(f"Latest Sentinel-2 date found in GEE: {latest_date}")
        return latest_date
    except ee.EEException as e:
        logger.error(f"GEE error querying Sentinel-2 date: {e}")
        return None
    except Exception as e:
        logger.error(f"Error querying GEE for latest Sentinel-2 date: {e}")
        return None

# === Utility Functions ===
def validate_bbox(bbox: list) -> list:
    """Validate a geographic bounding box."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must be a list/tuple of 4 values: [minx, miny, maxx, maxy]")
    minx, miny, maxx, maxy = bbox
    if not (-180 <= minx <= maxx <= 180 and -90 <= miny <= maxy <= 90):
        raise ValueError("bbox coordinates out of range: longitudes [-180, 180], latitudes [-90, 90]")
    if maxx - minx <= 0 or maxy - miny <= 0:
        raise ValueError("Invalid bbox: maxx must be > minx and maxy > miny")
    area = (maxx - minx) * (maxy - miny)
    if area < 0.0001:
        logger.warning(f"Small bbox area ({area:.6f} sq. degrees) may cause empty exports.")
    return bbox

def is_valid_file(filepath: str, min_size: int = 500) -> bool:
    """Check if a file exists and meets minimum size requirements."""
    if not filepath or not os.path.isfile(filepath):
        logger.warning(f"File {filepath} does not exist.")
        return False
    if os.path.getsize(filepath) < min_size:
        logger.warning(f"File {filepath} is too small (< {min_size} bytes).")
        return False
    return True

def setup_directories(config: dict) -> None:
    """Create necessary directories if they don't exist."""
    try:
        for dir_path in [config['data_dir'], config['output_dir'], config['checkpoint_dir']]:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Directory ensured: {dir_path}")
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
        raise

def get_utm_crs(bbox: list) -> str:
    """Determine the UTM CRS for a bounding box."""
    lon_center = (bbox[0] + bbox[2]) / 2
    lat_center = (bbox[1] + bbox[3]) / 2
    utm_zone = int((lon_center + 180) / 6) + 1
    hemisphere = 'north' if lat_center >= 0 else 'south'
    epsg_code = f"EPSG:{32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone}"
    return epsg_code

# === GEE Authentication and Exports ===
def authenticate_gee(key_path: str, project_id: str) -> None:
    """Authenticate with Google Earth Engine."""
    try:
        if not hasattr(ee.data, '_credentials') or not ee.data._credentials:
            credentials = service_account.Credentials.from_service_account_file(
                key_path, scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            ee.Initialize(credentials=credentials, project=project_id)
            logger.info(f"Earth Engine authenticated with project: {project_id}")
        else:
            logger.info("Earth Engine already authenticated.")
    except Exception as e:
        logger.error(f"Error authenticating GEE: {e}")
        raise

def export_gee_image(image: ee.Image, region: ee.Geometry, filename: str, scale: int = 30) -> str:
    """Export a GEE image to a local file."""
    try:
        geemap.ee_export_image(image.clip(region), filename=filename, scale=scale, region=region, file_per_band=False)
        if is_valid_file(filename):
            logger.info(f"Exported: {filename}")
            return filename
        return None
    except ee.EEException as e:
        logger.error(f"GEE error exporting {filename}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error exporting {filename}: {e}")
        return None

def export_sentinel2(bbox: list, timestamp: str, config: dict) -> str:
    """Export Sentinel-2 RGB image from GEE with dynamic date range."""
    region = ee.Geometry.Rectangle(bbox)
    try:
        username = os.getenv('CDSE_USERNAME')
        password = os.getenv('CDSE_PASSWORD')
        if not (username and password):
            logger.warning("CDSE credentials not found in environment variables. Prompting for input.")
            username = input("Enter CDSE username: ")
            password = getpass("Enter CDSE password: ")
            totp = getpass("Enter 2FA code (if required, else press Enter): ") or None
        else:
            totp = None

        if not (username and password):
            logger.warning("No valid credentials provided. Using fallback date range.")
            start_date = config['gee_export']['s2_date_range'][0]
            end_date = config['gee_export']['s2_date_range'][1]
        else:
            token_data = get_cdse_access_token(username, password, totp=totp)
            if not token_data and totp is None:
                logger.warning("First CDSE authentication attempt failed. Retrying with 2FA.")
                totp = getpass("Enter 2FA code: ") or None
                token_data = get_cdse_access_token(username, password, totp=totp)
            if not token_data:
                logger.warning("Failed to obtain CDSE access token. Trying GEE fallback.")
                latest_date = get_latest_sentinel2_date_fallback(bbox, config['gee_export']['s2_cloud_percentage'])
            else:
                latest_date = get_latest_sentinel2_date(bbox, token_data["access_token"],
                                                       config['gee_export']['s2_cloud_percentage'])

            if not latest_date:
                logger.warning("No valid Sentinel-2 date found. Using fallback date range.")
                start_date = config['gee_export']['s2_date_range'][0]
                end_date = config['gee_export']['s2_date_range'][1]
            else:
                end_date = latest_date
                start_date = (datetime.strptime(latest_date, '%Y-%m-%d') - timedelta(days=5)).strftime('%Y-%m-%d')

        logger.info(f"Querying Sentinel-2 for {start_date} to {end_date}")
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(region) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", config['gee_export']['s2_cloud_percentage'])) \
            .sort('system:time_start', False) \
            .first()
        if s2 is None:
            logger.warning("No valid Sentinel-2 image found for the given bbox and date range.")
            return None
        rgb = s2.select(['B4', 'B3', 'B2'])
        return export_gee_image(rgb, region, os.path.join(config['data_dir'], f"s2_rgb_{timestamp}.tif"), scale=10)
    except ee.EEException as e:
        logger.error(f"GEE error exporting Sentinel-2: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error exporting Sentinel-2: {e}")
        return None

def export_dem_and_slope(bbox: list, timestamp: str, config: dict) -> tuple:
    """Export DEM and slope from GEE."""
    region = ee.Geometry.Rectangle(bbox)
    try:
        dem_collection = ee.ImageCollection("COPERNICUS/DEM/GLO30") \
            .filterBounds(region)
        dem = dem_collection.mosaic()
        slope = ee.Terrain.slope(dem)
        dem_path = export_gee_image(dem, region, os.path.join(config['data_dir'], f"dem_glo30_{timestamp}.tif"),
                                    scale=config['gee_export']['dem_scale'])
        slope_path = export_gee_image(slope, region, os.path.join(config['data_dir'], f"slope_glo30_{timestamp}.tif"),
                                      scale=config['gee_export']['dem_scale'])
        return dem_path, slope_path
    except Exception as e:
        logger.error(f"Error exporting DEM/Slope: {e}")
        return None, None

def export_cgls_lc100(bbox: list, timestamp: str, config: dict) -> str:
    """Export CGLS land cover from GEE."""
    region = ee.Geometry.Rectangle(bbox)
    try:
        landcover = ee.Image("COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019") \
            .select("discrete_classification")
        return export_gee_image(landcover, region, os.path.join(config['data_dir'], f"cgls_lc100_{timestamp}.tif"),
                                scale=config['gee_export']['lc_scale'])
    except Exception as e:
        logger.error(f"Error exporting CGLS-LC100: {e}")
        return None

def export_era5_precipitation(bbox: list, timestamp: str, config: dict) -> str:
    """Export ERA5 precipitation from GEE."""
    region = ee.Geometry.Rectangle(bbox)
    try:
        current_date = datetime.now()
        start_date = f"{current_date.year}-01-01"
        end_date = current_date.strftime('%Y-%m-%d')
        era5 = ee.ImageCollection("ECMWF/ERA5/DAILY") \
            .filterDate(start_date, end_date) \
            .select("total_precipitation") \
            .mean()
        return export_gee_image(era5, region, os.path.join(config['data_dir'], f"era5_precipitation_{timestamp}.tif"),
                                scale=config['gee_export']['precip_scale'])
    except Exception as e:
        logger.error(f"Error exporting ERA5: {e}")
        return None

def export_gee_layers(bbox: list, timestamp: str, config: dict) -> tuple:
    """Export all GEE layers in parallel."""
    tasks = [
        (export_dem_and_slope, "DEM/Slope"),
        (export_cgls_lc100, "CGLS-LC100"),
        (export_era5_precipitation, "ERA5 Precipitation")
    ]
    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_task = {executor.submit(func, bbox, timestamp, config): name for func, name in tasks}
        for future in as_completed(future_to_task):
            name = future_to_task[future]
            try:
                results[name] = future.result()
            except Exception as e:
                logger.error(f"Error in {name}: {e}")
    dem_path, slope_path = results.get("DEM/Slope", (None, None))
    cgls_path = results.get("CGLS-LC100")
    precip_path = results.get("ERA5 Precipitation")
    return dem_path, slope_path, cgls_path, precip_path

# === Image Processing ===
def convert_tiff_to_uint8(input_path: str, output_path: str = None) -> str:
    """Convert a TIFF image to uint8 format."""
    if not is_valid_file(input_path):
        return None
    if output_path is None:
        output_path = input_path.replace('.tif', '_uint8.tif')
    try:
        with rasterio.open(input_path) as src:
            arr = src.read()
            arr_uint8 = np.empty_like(arr, dtype=np.uint8)
            for b in range(arr.shape[0]):
                band = arr[b]
                minval, maxval = np.min(band), np.max(band)
                scaled = np.zeros_like(band) if maxval <= minval else ((band - minval) / (maxval - minval)) * 255
                arr_uint8[b] = scaled.astype(np.uint8)
            profile = src.profile
            profile.update(dtype='uint8')
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(arr_uint8)
        logger.info(f"Image converted to uint8: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error converting {input_path} to uint8: {e}")
        return None

def tif_to_base64_image(tif_path: str, bands: list = [1, 2, 3]) -> tuple:
    """Convert a TIFF to a base64-encoded PNG for map display."""
    try:
        with rasterio.open(tif_path) as src:
            bounds = src.bounds
            image = src.read(bands)
            if image.shape[0] == 1:
                image = np.repeat(image, 3, axis=0)
            image = np.moveaxis(image, 0, -1)
            image = (image - image.min()) / (image.max() - image.min() + 1e-10) * 255
            image = image.astype(np.uint8)
        fig, ax = plt.subplots()
        ax.axis("off")
        plt.imshow(image)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return encoded, bounds
    except Exception as e:
        logger.error(f"Error converting TIFF to base64: {e}")
        return None, None

def enhance_segmentation_with_slope(image_path: str, slope_path: str) -> str:
    """Enhance an image by adding a slope band."""
    if not (is_valid_file(image_path) and is_valid_file(slope_path)):
        logger.warning("Missing image or slope, skipping slope enhancement.")
        return image_path
    try:
        with rasterio.open(image_path) as img_src, rasterio.open(slope_path) as slope_src:
            image = img_src.read()
            slope = slope_src.read(1)
            profile = img_src.profile
            if slope.shape != image.shape[1:]:
                slope = ndimage.zoom(slope, (image.shape[1] / slope.shape[0], image.shape[2] / slope.shape[1]), order=1)
            enhanced_image = np.concatenate([image, slope[np.newaxis, :, :]], axis=0)
        enhanced_path = image_path.replace(".tif", "_enhanced.tif")
        profile.update(count=enhanced_image.shape[0])
        with rasterio.open(enhanced_path, 'w', **profile) as dst:
            dst.write(enhanced_image)
        logger.info(f"Enhanced image with slope: {enhanced_path}")
        return enhanced_path
    except Exception as e:
        logger.error(f"Error enhancing image with slope: {e}")
        return image_path

def get_valid_base_image(bbox: list, timestamp: str, config: dict) -> str:
    """Get a valid base image (Sentinel-2 or Esri TMS fallback) in uint8 format."""
    logger.info("Searching for cloud-free Sentinel-2 image...")
    s2_path = export_sentinel2(bbox, timestamp, config)
    if s2_path and is_valid_file(s2_path):
        with rasterio.open(s2_path) as src:
            if src.dtypes[0] != 'uint8':
                return convert_tiff_to_uint8(s2_path)
        return s2_path

    logger.warning("Falling back to Esri World Imagery TMS.")
    tms_path = os.path.join(config['data_dir'], f"esri_image_{timestamp}.tif")
    try:
        tms_to_geotiff(output=tms_path, bbox=bbox, zoom=config['zoom'], source="Satellite", overwrite=True)
        if is_valid_file(tms_path):
            with rasterio.open(tms_path) as src:
                if src.dtypes[0] != 'uint8':
                    return convert_tiff_to_uint8(tms_path)
            logger.info(f"TMS base image created: {tms_path}")
            return tms_path
    except Exception as e:
        logger.error(f"Error generating TMS: {e}")
    logger.error("No valid base image obtained. Aborting.")
    return None

# === Segmentation ===
def process_text_segmentation(text_prompts: list, image_path: str, config: dict) -> dict:
    """Perform text-based segmentation using LangSAM."""
    if not is_valid_file(image_path):
        return {}
    lang_sam = LangSAM()
    prompt_outputs = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        for prompt in tqdm(text_prompts, desc="Segmenting prompts"):
            try:
                lang_sam.predict(
                    image=image_path,
                    text_prompt=prompt,
                    box_threshold=config['prompt_thresholds'][prompt]['box_threshold'],
                    text_threshold=config['prompt_thresholds'][prompt]['text_threshold']
                )
                mask_path = os.path.join(tmp_dir, f"mask_{prompt}.tif")
                lang_sam.show_anns(cmap="Greens", add_boxes=True, alpha=1, blend=False, output=mask_path)
                if not is_valid_file(mask_path):
                    continue
                with rasterio.open(mask_path) as src:
                    mask = src.read(1)
                    transform = src.transform
                    crs = src.crs
                geoms, props = [], []
                for geom, value in shapes(mask, transform=transform):
                    polygon = shape(geom)
                    if value == 0 or polygon.area < config['min_area']:
                        continue
                    geoms.append(polygon)
                    props.append({"type": prompt, "label": f"{prompt.capitalize()} Feature"})
                vector_out = os.path.join(config['output_dir'], f"segment_{prompt}_{timestamp}.gpkg")
                gdf = gpd.GeoDataFrame(props, geometry=geoms, crs=crs)
                gdf.to_file(vector_out, driver="GPKG")
                logger.info(f"Created {len(gdf)} records for {prompt}")
                prompt_outputs[prompt] = vector_out
            except Exception as e:
                logger.error(f"Error segmenting '{prompt}': {e}")
    return prompt_outputs

def run_sam_segmentation(image_path: str, config: dict) -> str:
    """Perform SAM-based segmentation."""
    if not is_valid_file(image_path):
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = os.path.join(config['checkpoint_dir'], "sam_vit_h_4b8939.pth")
    sam = SamGeo(model_type="vit_h", checkpoint=checkpoint_path, device=device)
    mask_path = os.path.join(config['output_dir'], f"sam_mask_{timestamp}.tif")
    try:
        sam.generate(source=image_path, output=mask_path, batch=True, foreground=True)
        if not is_valid_file(mask_path):
            return None
        with rasterio.open(mask_path) as src:
            mask_data = src.read(1)
            transform = src.transform
            crs = src.crs
        geoms, props = [], []
        for geom, value in shapes(mask_data, transform=transform):
            polygon = shape(geom)
            if value == 0 or polygon.area < config['min_area']:
                continue
            geoms.append(polygon)
            props.append({"type": "SAM_segment", "label": "SAM Segment"})
        vector_out = os.path.join(config['output_dir'], f"sam_segment_{timestamp}.gpkg")
        gdf = gpd.GeoDataFrame(props, geometry=geoms, crs=crs)
        gdf.to_file(vector_out, driver="GPKG")
        logger.info(f"Created {len(gdf)} records for SAM segmentation")
        return vector_out
    except Exception as e:
        logger.error(f"Error in SAM segmentation: {e}")
        return None

# === Metrics and Mapping ===
def calculate_segmentation_urbanization(building_vector: str, road_vector: str, total_area: float) -> float:
    """Calculate urbanization percentage based on building and road areas."""
    area_urban = 0.0
    for vector in [building_vector, road_vector]:
        if is_valid_file(vector):
            try:
                gdf = gpd.read_file(vector)
                area_urban += gdf.geometry.area.sum()
            except Exception as e:
                logger.error(f"Error reading {vector} for urbanization: {e}")
    return round((area_urban / total_area) * 100, 2) if total_area > 0 else 0.0

def calculate_vegetation_precipitation(tree_vector: str, precip_path: str, total_area: float) -> tuple:
    """Calculate vegetation percentage and average precipitation."""
    veg_percentage = avg_precip = 0
    if is_valid_file(tree_vector):
        try:
            gdf = gpd.read_file(tree_vector)
            veg_area = gdf.geometry.area.sum()
            veg_percentage = round((veg_area / total_area) * 100, 2)
        except Exception as e:
            logger.error(f"Error reading {tree_vector} for vegetation: {e}")
    if is_valid_file(precip_path):
        try:
            with rasterio.open(precip_path) as src:
                precip = src.read(1)
                precip = precip[np.isfinite(precip)]
                avg_precip = round(precip.mean() * 1000, 2) if precip.size > 0 else 0  # Convert m to mm
        except Exception as e:
            logger.error(f"Error reading {precip_path} for precipitation: {e}")
    else:
        logger.warning("Precipitation data unavailable; setting average precipitation to 0 mm.")
    return veg_percentage, avg_precip

def initialize_map(bbox: list, zoom: int) -> folium.Map:
    """Initialize a Folium map with Esri satellite tiles."""
    center = [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]
    m = folium.Map(location=center, zoom_start=zoom-1, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite"
    ).add_to(m)
    return m

def add_gee_layers(m: folium.Map, gee_layers: list) -> None:
    """Add GEE raster layers to the map."""
    for layer in tqdm(gee_layers, desc="Adding GEE layers"):
        if not is_valid_file(layer["path"]):
            continue
        try:
            encoded, bounds = tif_to_base64_image(layer["path"], layer["bands"])
            if encoded and bounds:
                folium.raster_layers.ImageOverlay(
                    image=f"data:image/png;base64,{encoded}",
                    bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                    opacity=layer["opacity"],
                    name=layer["name"]
                ).add_to(m)
        except Exception as e:
            logger.error(f"Error adding layer {layer['name']}: {e}")

def add_segmentation_layers(m: folium.Map, prompt_outputs: dict, sam_vector_out: str, layer_styles: dict) -> None:
    """Add segmentation vector layers to the map."""
    text_group = FeatureGroup(name="Text Segmentations", show=True)
    sam_group = FeatureGroup(name="SAM Segmentations", show=True)
    for prompt, vector_out in prompt_outputs.items():
        if not is_valid_file(vector_out):
            continue
        try:
            gdf = gpd.read_file(vector_out).to_crs(epsg=4326)
            GeoJson(
                gdf.__geo_interface__,
                name=f"{prompt.capitalize()} Segmentation",
                style_function=lambda feature, p=prompt: layer_styles[p],
                popup=folium.GeoJsonPopup(fields=["label"])
            ).add_to(text_group)
        except Exception as e:
            logger.error(f"Error adding segmentation layer {prompt}: {e}")
    if is_valid_file(sam_vector_out):
        try:
            gdf = gpd.read_file(sam_vector_out).to_crs(epsg=4326)
            GeoJson(
                gdf.__geo_interface__,
                name="SAM Segments",
                style_function=lambda feature: layer_styles["SAM_segment"],
                popup=folium.GeoJsonPopup(fields=["label"])
            ).add_to(sam_group)
        except Exception as e:
            logger.error(f"Error adding SAM segmentation layer: {e}")
    text_group.add_to(m)
    sam_group.add_to(m)

def add_metrics_panel(m: folium.Map, prompt_outputs: dict, precip_path: str, total_area: float) -> None:
    """Add an HTML panel with environmental metrics to the map."""
    seg_urban = calculate_segmentation_urbanization(
        prompt_outputs.get("building"), prompt_outputs.get("road"), total_area)
    veg_percentage, avg_precip = calculate_vegetation_precipitation(
        prompt_outputs.get("tree"), precip_path, total_area)
    metrics_html = f"""
    <div style="position: fixed; bottom: 40px; left: 40px; width: 240px; background-color: white; border: 2px solid #00A878; border-radius: 8px; padding: 12px; font-family: 'Inter', sans-serif; font-size: 14px; box-shadow: 2px 2px 8px rgba(0,0,0,0.2); z-index: 9999;">
      <b>Environmental Metrics</b><br>
      Urban (Seg): <span style="color:#00A878; font-weight:bold;">{seg_urban}%</span><br>
      Vegetation: <span style="color:#00A878; font-weight:bold;">{veg_percentage}%</span><br>
      Avg Precipitation: <span style="color:#00A878; font-weight:bold;">{avg_precip} mm</span><br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(metrics_html))

def create_interactive_map(bbox: list, zoom: int, prompt_outputs: dict, sam_vector_out: str, gee_layers: list) -> None:
    """Create and save an interactive Folium map with GEE and segmentation layers."""
    layer_styles = {
        "tree": {"color": "#00CC00", "weight": 2, "fillColor": "#00FF00", "fillOpacity": 0.2},
        "water": {"color": "#3399FF", "weight": 2, "fillColor": "#66B2FF", "fillOpacity": 0.2},
        "building": {"color": "#800080", "weight": 2, "fillColor": "#CC00CC", "fillOpacity": 0.2},
        "road": {"color": "#808080", "weight": 2, "fillColor": "#A9A9A9", "fillOpacity": 0.2},
        "SAM_segment": {"color": "#FF6600", "weight": 2, "fillColor": "#FF9933", "fillOpacity": 0.2}
    }
    # Calculate total area in square meters
    try:
        bbox_poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
        gdf_bbox = gpd.GeoDataFrame([{'geometry': bbox_poly}], crs="EPSG:4326")
        utm_crs = get_utm_crs(bbox)
        gdf_bbox = gdf_bbox.to_crs(utm_crs)
        total_area = gdf_bbox.geometry.area.iloc[0]  # Area in square meters
        logger.info(f"Total area calculated: {total_area:.2f} square meters")
    except Exception as e:
        logger.error(f"Error calculating total area: {e}")
        total_area = 1.0  # Fallback to avoid division by zero

    m = initialize_map(bbox, zoom)
    add_gee_layers(m, gee_layers)
    add_segmentation_layers(m, prompt_outputs, sam_vector_out, layer_styles)
    precip_path = next((l["path"] for l in gee_layers if l["name"].startswith("ERA5")), None)
    add_metrics_panel(m, prompt_outputs, precip_path, total_area)
    LayerControl(position="topright", collapsed=True).add_to(m)
    map_path = os.path.join(CONFIG['output_dir'], f"integrated_map_{timestamp}.html")
    m.save(map_path)
    logger.info(f"Map saved: {map_path}")

# === Main Workflow ===
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Geospatial analysis pipeline")
    parser.add_argument('--bbox', type=float, nargs=4, default=CONFIG['bbox'],
                        help="Bounding box: minx miny maxx maxy")
    parser.add_argument('--zoom', type=int, default=CONFIG['zoom'],
                        help="Zoom level for TMS")
    parser.add_argument('--output-dir', default=CONFIG['output_dir'],
                        help="Output directory")
    return parser.parse_args()

def main():
    """Main function to run the geospatial analysis pipeline."""
    args = parse_args()
    global timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config = CONFIG.copy()
    config['bbox'] = validate_bbox(args.bbox)
    config['zoom'] = args.zoom
    config['output_dir'] = args.output_dir

    logger.info("Starting geospatial analysis pipeline")
    setup_directories(config)
    authenticate_gee(config['key_path'], config['project_id'])

    dem_path, slope_path, cgls_path, precip_path = export_gee_layers(config['bbox'], timestamp, config)
    base_image_path = get_valid_base_image(config['bbox'], timestamp, config)
    if not base_image_path:
        logger.error("No valid base image obtained. Aborting.")
        return

    enhanced_image_path = enhance_segmentation_with_slope(base_image_path, slope_path)
    text_prompts = ["tree", "water", "building", "road"]
    prompt_outputs = process_text_segmentation(text_prompts, enhanced_image_path, config)
    sam_vector_out = run_sam_segmentation(enhanced_image_path, config)
    gee_layers = [
        {"name": "Base Image", "path": base_image_path, "bands": [1, 2, 3], "opacity": 0.75} if is_valid_file(base_image_path) else None,
        {"name": "DEM Copernicus", "path": dem_path, "bands": [1], "opacity": 0.5} if is_valid_file(dem_path) else None,
        {"name": "Slope Copernicus", "path": slope_path, "bands": [1], "opacity": 0.5} if is_valid_file(slope_path) else None,
        {"name": "CGLS-LC100", "path": cgls_path, "bands": [1], "opacity": 0.4} if is_valid_file(cgls_path) else None,
        {"name": "ERA5 Precipitation", "path": precip_path, "bands": [1], "opacity": 0.3} if is_valid_file(precip_path) else None
    ]
    gee_layers = [layer for layer in gee_layers if layer]
    create_interactive_map(config['bbox'], config['zoom'], prompt_outputs, sam_vector_out, gee_layers)
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()