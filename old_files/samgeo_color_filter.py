import os
import numpy as np
import rasterio
import torch
import cv2
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import shape
from datetime import datetime
from samgeo import SamGeo, tms_to_geotiff
import leafmap
from sklearn.cluster import KMeans

# === CONFIGURACIÓN ===
bbox = [-90.015147, 14.916566, -90.010159, 14.919471]
zoom = 18
checkpoint_path = 'checkpoints/sam_vit_h_4b8939.pth'
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# === RUTAS DE ARCHIVOS ===
image_path = os.path.join(output_dir, f"esri_image_{timestamp}.tif")
mask_path = os.path.join(output_dir, f"sam_mask_all_{timestamp}.tif")
vector_out = os.path.join(output_dir, f"filtered_segment_{timestamp}.gpkg")

# === 1. Descargar imagen TMS ===
tms_to_geotiff(output=image_path, bbox=bbox, zoom=zoom, source="Satellite", overwrite=True)

# === 2. Segmentación Automática ===
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = SamGeo(
    model_type="vit_h",
    checkpoint=checkpoint_path,
    device=device,
)
sam.generate(source=image_path, output=mask_path, batch=True, foreground=True)

# === 3. Extraer máscaras individuales ===
print("🎨 Filtrando máscaras por color...")
with rasterio.open(image_path) as img_src, rasterio.open(mask_path) as mask_src:
    image_data = img_src.read([1, 2, 3])  # RGB
    mask_data = mask_src.read(1)
    transform = mask_src.transform
    crs = mask_src.crs

# === 4. Filtrar máscaras por color dominante ===

from rasterio.features import shapes

def get_dominant_rgb(image_patch):
    pixels = image_patch.reshape(-1, 3)
    pixels = pixels[~np.all(pixels == 0, axis=1)]  # quitar fondo negro
    if len(pixels) < 10:
        return [0, 0, 0]
    kmeans = KMeans(n_clusters=1).fit(pixels)
    return kmeans.cluster_centers_[0].astype(int)

def is_excluded(rgb):
    r, g, b = rgb
    if b > 34  and g < 54 and r < 31: return True  # Azul: agua
    if g > 106 and r < 34 and b < 41: return True  # Verde: vegetación
    if r > 150 and g > 140 and b < 107: return True    # Marrón: tierra
    return False

geoms = []
props = []

for geom, value in shapes(mask_data, transform=transform):
    if value == 0:
        continue

    polygon = shape(geom)
    mask_raster = rasterio.features.geometry_mask([polygon], image_data.shape[1:], transform=transform, invert=True)
    masked_img = np.transpose(image_data, (1, 2, 0))
    masked_pixels = masked_img[mask_raster]

    if masked_pixels.size == 0:
        continue

    avg_color = get_dominant_rgb(masked_pixels)
    if not is_excluded(avg_color):
        geoms.append(polygon)
        props.append({"r": int(avg_color[0]), "g": int(avg_color[1]), "b": int(avg_color[2])})

# === 5. Guardar GeoPackage ===
gdf = gpd.GeoDataFrame(props, geometry=geoms, crs=crs)
gdf.to_file(vector_out, driver="GPKG")
print(f"✅ Segmentación filtrada guardada en: {vector_out}")

# === 6. Visualizar ===
print("🗺️ Mostrando en Leafmap...")
m = leafmap.Map(center=[(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2], zoom=zoom)
m.add_basemap("Esri.WorldImagery")
m.add_vector(vector_out, layer_name="Segmentación filtrada", style={
    "color": "#3388ff",
    "weight": 2,
    "fillColor": "#ff7800",
    "fillOpacity": 0.4,
})

# === 7. Exportar como archivo HTML interactivo ===
html_path = os.path.join(output_dir, f"segmentacion_mapa_{timestamp}.html")
m.to_html(outfile=html_path)
print(f"📁 Mapa interactivo guardado en: {html_path}")