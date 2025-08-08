import os
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import shape
from datetime import datetime
from samgeo import SamGeo, tms_to_geotiff
from samgeo.text_sam import LangSAM
import leafmap
import torch
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.features import shapes

# === CONFIGURACIÓN GENERAL ===
bbox = [-90.015147, 14.916566, -90.010159, 14.919471]
zoom = 19
output_dir = "output"
checkpoint_dir = "checkpoints"
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

def setup_directories():
    """Create output and checkpoint directories if they don't exist."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

def download_tms_image(bbox, zoom):
    """Download TMS image and save it to the output directory."""
    image_path = os.path.join(output_dir, f"esri_image_{timestamp}.tif")
    try:
        tms_to_geotiff(output=image_path, bbox=bbox, zoom=zoom, source="Satellite", overwrite=True)
        print(f"✅ Imagen TMS descargada: {image_path}")
        return image_path
    except Exception as e:
        print(f"❌ Error al descargar la imagen TMS: {e}")
        raise

def combine_text_masks(text_prompts, image_path, profile):
    """Generate and combine masks for multiple text prompts using LangSAM."""
    combined_mask = None
    lang_sam = LangSAM()  # Initialize once to avoid reloading the model

    print("🧠 Generando máscaras por texto:", text_prompts)
    for prompt in text_prompts:
        print(f"   📌 Procesando prompt: '{prompt}'")
        try:
            lang_sam.predict(image=image_path, text_prompt=prompt, box_threshold=0.24, text_threshold=0.24)
            temp_mask_path = os.path.join(output_dir, f"mask_{prompt}_{timestamp}.tif")
            lang_sam.show_anns(cmap="Greens", add_boxes=True, alpha=1, blend=False, output=temp_mask_path)

            with rasterio.open(temp_mask_path) as src:
                mask = src.read(1)
                if combined_mask is None:
                    combined_mask = mask
                    profile.update(dtype=rasterio.uint8, count=1)
                else:
                    combined_mask = np.logical_or(combined_mask, mask)
            os.remove(temp_mask_path)  # Clean up temporary file
        except Exception as e:
            print(f"❌ Error procesando prompt '{prompt}': {e}")
            raise

    multi_mask_path = os.path.join(output_dir, f"excluded_mask_{timestamp}.tif")
    with rasterio.open(multi_mask_path, 'w', **profile) as dst:
        dst.write(combined_mask.astype(rasterio.uint8), 1)
    print(f"✅ Máscara combinada de exclusión guardada: {multi_mask_path}")
    return combined_mask, multi_mask_path

def run_sam_segmentation(image_path):
    """Run automatic SAM segmentation on the input image."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = os.path.join(checkpoint_dir, "sam_vit_h_4b8939.pth")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SAM checkpoint not found at {checkpoint_path}. Please download it.")

    sam = SamGeo(
        model_type="vit_h",
        checkpoint=checkpoint_path,
        device=device,
    )
    filtered_mask_path = os.path.join(output_dir, f"sam_mask_all_{timestamp}.tif")
    try:
        sam.generate(source=image_path, output=filtered_mask_path, batch=True, foreground=True)
        print(f"✅ Segmentación SAM completada: {filtered_mask_path}")
        return filtered_mask_path
    except Exception as e:
        print(f"❌ Error en segmentación SAM: {e}")
        raise

def filter_masks(image_path, filtered_mask_path, multi_mask_path):
    """Filter SAM masks that intersect with the combined semantic mask."""
    print("🧼 Filtrando segmentos que coinciden con exclusión semántica...")
    try:
        with rasterio.open(image_path) as img_src, \
             rasterio.open(filtered_mask_path) as mask_src, \
             rasterio.open(multi_mask_path) as exclusion_mask_src:
            image_data = img_src.read([1, 2, 3])
            mask_data = mask_src.read(1)
            exclusion_mask = exclusion_mask_src.read(1)
            transform = mask_src.transform
            crs = mask_src.crs

        geoms = []
        props = []

        for geom, value in shapes(mask_data, transform=transform):
            if value == 0:
                continue

            polygon = shape(geom)
            mask_raster = rasterio.features.geometry_mask([polygon], image_data.shape[1:], transform=transform, invert=True)

            if np.any(exclusion_mask[mask_raster] > 0):
                continue

            geoms.append(polygon)
            props.append({})

        vector_out = os.path.join(output_dir, f"filtered_segment_{timestamp}.gpkg")
        gdf = gpd.GeoDataFrame(props, geometry=geoms, crs=crs)
        gdf.to_file(vector_out, driver="GPKG")
        print(f"✅ Segmentación final guardada en: {vector_out}")
        return vector_out, crs
    except Exception as e:
        print(f"❌ Error al filtrar máscaras: {e}")
        raise

def create_interactive_map(bbox, zoom, vector_out, style=None):
    """Create an interactive map using Leafmap and save it as HTML."""
    if style is None:
        style = {"color": "#3388ff", "weight": 2, "fillColor": "#ff7800", "fillOpacity": 0.4}

    html_path = os.path.join(output_dir, f"segmentacion_mapa_{timestamp}.html")
    try:
        m = leafmap.Map(center=[(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2], zoom=zoom)
        m.add_basemap("Esri.WorldImagery")
        m.add_vector(vector_out, layer_name="Segmentación sin árboles/agua", style=style)
        m.to_html(outfile=html_path)
        print(f"📁 Mapa HTML exportado: {html_path}")
        return m
    except Exception as e:
        print(f"❌ Error al crear el mapa interactivo: {e}")
        raise

def main():
    """
    Process a satellite image to segment and filter features using SAM and LangSAM.

    This script:
    1. Downloads a satellite image using TMS.
    2. Uses LangSAM to create masks for specified text prompts (e.g., 'tree', 'water').
    3. Performs automatic segmentation with SAMGeo.
    4. Filters out SAM segments that intersect with the LangSAM masks.
    5. Saves the filtered segments as a GeoPackage.
    6. Visualizes the results on an interactive Leafmap.

    Inputs:
    - bbox: Bounding box coordinates [minx, miny, maxx, maxy]
    - zoom: Zoom level for TMS download
    - output_dir: Directory to save outputs
    - checkpoint_dir: Directory containing SAM model checkpoint

    Outputs:
    - TIFF files for images and masks
    - GeoPackage file for filtered segments
    - HTML file for interactive map visualization
    """
    setup_directories()
    
    # Step 1: Download TMS image
    image_path = download_tms_image(bbox, zoom)

    # Step 2: Segment with LangSAM for text prompts
    text_prompts = ["tree", "water"]
    with rasterio.open(image_path) as src:
        profile = src.profile
    combined_mask, multi_mask_path = combine_text_masks(text_prompts, image_path, profile)

    # Step 3: Automatic SAM segmentation
    filtered_mask_path = run_sam_segmentation(image_path)

    # Step 4: Filter masks
    vector_out, crs = filter_masks(image_path, filtered_mask_path, multi_mask_path)

    # Step 5: Visualize with Leafmap
    print("🗺️ Mostrando mapa interactivo...")
    create_interactive_map(bbox, zoom, vector_out)
    print("✅ Proceso completado.")

if __name__ == "__main__":
    main()