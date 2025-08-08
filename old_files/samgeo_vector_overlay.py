import os
import torch
import rasterio
import hashlib
from datetime import datetime
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from samgeo import SamGeo, tms_to_geotiff, get_basemaps
import leafmap  # ✅ se usa leafmap, no Map de samgeo

# === CONFIGURACIÓN ===
bbox = [-90.015147, 14.916566, -90.010159, 14.919471]
zoom = 18
output_dir = "output"
cache_dir = "EsriCache"
checkpoint_path = 'checkpoints/sam_vit_h_4b8939.pth'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

def cache_name(bbox, zoom):
    key = f"{bbox}-{zoom}".encode()
    return hashlib.md5(key).hexdigest()

cache_file = os.path.join(cache_dir, f"{cache_name(bbox, zoom)}.tif")
image_georef = os.path.join(output_dir, f"esri_export_georef_{timestamp}.tif")
mask_path = os.path.join(output_dir, f"segment_{timestamp}.tif")
vector_path = os.path.join(output_dir, f"segment_{timestamp}.gpkg")

# === 1. Descarga imagen desde Esri ===
if not os.path.exists(cache_file):
    print("⏬ Descargando imagen...")
    tms_to_geotiff(output=cache_file, bbox=bbox, zoom=zoom, source='Satellite', overwrite=True)
else:
    print("📂 Usando imagen desde caché.")

# === 2. Georreferenciación ===
print("🌐 Aplicando transformación...")
with rasterio.open(cache_file) as src:
    data = src.read()
    count = src.count
    dtype = src.dtypes[0]
    height, width = src.height, src.width
    transform = from_bounds(*bbox, width, height)
    crs = CRS.from_epsg(4326)

    profile = src.profile.copy()
    profile.update({
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': count,
        'dtype': dtype,
        'crs': crs,
        'transform': transform
    })

    with rasterio.open(image_georef, 'w', **profile) as dst:
        dst.write(data)

sam_kwargs = {
    "points_per_side": 32,
    "pred_iou_thresh": 0.86,
    "stability_score_thresh": 0.92,
    "crop_n_layers": 1,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 100,
}

# === 3. Segmentación SAM ===
print("🧠 Segmentando...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sam = SamGeo(
    model_type='vit_h',
    checkpoint=checkpoint_path,
    device=device,
    sam_kwargs=sam_kwargs,
)

sam.generate(
    image_georef, 
    mask_path,
    batch=True,
    foreground=True,
    erosion_kernel=(3, 3),
    mask_multiplier=255
)

# === 4. Vectorización ===
print("📦 Exportando a GPKG...")
sam.tiff_to_gpkg(mask_path, vector_path, simplify_tolerance=None)
sam.show_masks(cmap="binary_r", show=True)
sam.show_anns(axis="off", opacity=1, output="annotations2.tif")


# === 5. Visualización ===
print("🗺️ Visualizando en mapa...")

# Configurar estilo de vector
style = {
    "color": "#3388ff",
    "weight": 2,
    "fillColor": "#7c4185",
    "fillOpacity": 0.5,
}

# Crear mapa con center automático
lat_center = (bbox[1] + bbox[3]) / 2
lon_center = (bbox[0] + bbox[2]) / 2
m = leafmap.Map(center=[lat_center, lon_center], zoom=zoom)

# Añadir imagen y vector
m.add_basemap("Esri.WorldImagery")
m.add_raster(image_georef, layer_name="GeoTIFF Segmentado")
m.add_vector(vector_path, layer_name="Segmentación SAM", style=style)

# Mostrar el mapa (en notebook o streamlit)
#m.show()  # si estás en Jupyter
#m.to_streamlit(height=700)  # si estás en Streamlit
m.to_html(f"output/mapa_segmentado_{timestamp}.html")