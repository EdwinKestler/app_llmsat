import os
import numpy as np
import rasterio
import cv2
import geopandas as gpd
from shapely.geometry import shape
from datetime import datetime
from samgeo import SamGeo, tms_to_geotiff
from samgeo.text_sam import LangSAM
import leafmap
from sklearn.cluster import KMeans
import torch
from rasterio.transform import from_bounds
from rasterio.crs import CRS

# === CONFIGURACIÓN ===
bbox = [-90.015147, 14.916566, -90.010159, 14.919471]
zoom = 18
checkpoint_path = 'checkpoints/sam_vit_h_4b8939.pth'
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# === RUTAS DE ARCHIVOS ===
image_path = os.path.join(output_dir, f"esri_image_{timestamp}.tif")
tree_mask_path = os.path.join(output_dir, f"tree_mask_{timestamp}.tif")
filtered_mask_path = os.path.join(output_dir, f"sam_mask_all_{timestamp}.tif")
vector_out = os.path.join(output_dir, f"filtered_segment_{timestamp}.gpkg")

# === 1. Descargar imagen TMS ===
tms_to_geotiff(output=image_path, bbox=bbox, zoom=zoom, source="Satellite", overwrite=True)

# === 2. Segmentación por texto para excluir "tree" ===
print("🌳 Ejecutando segmentación por texto: 'tree'")
text_prompt = "tree"
lang_sam = LangSAM()
lang_sam.predict(image=image_path, text_prompt=text_prompt, box_threshold=0.24, text_threshold=0.24)
lang_sam.show_anns(
    cmap="Greys_r", add_boxes=False, alpha=1, blend=False, output=tree_mask_path
)

# === 3. Segmentación Automática General con SAM ===
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = SamGeo(
    model_type="vit_h",
    checkpoint=checkpoint_path,
    device=device,
)
sam.generate(source=image_path, output=filtered_mask_path, batch=True, foreground=True)

# === 4. Filtrar máscaras automáticas excluyendo "tree" y colores ===
print("🎨 Filtrando máscaras automáticas...")
with rasterio.open(image_path) as img_src, rasterio.open(filtered_mask_path) as mask_src, rasterio.open(tree_mask_path) as tree_mask_src:
    image_data = img_src.read([1, 2, 3])
    mask_data = mask_src.read(1)
    tree_mask = tree_mask_src.read(1)
    transform = mask_src.transform
    crs = mask_src.crs

from rasterio.features import shapes

def get_dominant_rgb(image_patch):
    pixels = image_patch.reshape(-1, 3)
    pixels = pixels[~np.all(pixels == 0, axis=1)]
    if len(pixels) < 10:
        return [0, 0, 0]
    kmeans = KMeans(n_clusters=1).fit(pixels)
    return kmeans.cluster_centers_[0].astype(int)

def is_in_range(rgb, r_range, g_range, b_range):
    r, g, b = rgb
    return r_range[0] <= r <= r_range[1] and g_range[0] <= g <= g_range[1] and b_range[0] <= b <= b_range[1]

def is_excluded(rgb):
    if is_in_range(rgb, r_range=(0, 60), g_range=(0, 100), b_range=(35, 255)):
        return True
    if is_in_range(rgb, r_range=(0, 90), g_range=(100, 255), b_range=(0, 100)):
        return True
    if is_in_range(rgb, r_range=(120, 255), g_range=(100, 210), b_range=(50, 130)):
        return True
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

    # Excluir si intersecta con la máscara de "tree"
    intersect = np.any(tree_mask[mask_raster] > 0)
    if intersect:
        continue

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

# === 6. Visualización ===
print("🗌 Mostrando en Leafmap...")
m = leafmap.Map(center=[(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2], zoom=zoom)
m.add_basemap("Esri.WorldImagery")
m.add_vector(vector_out, layer_name="Segmentación filtrada", style={
    "color": "#3388ff",
    "weight": 2,
    "fillColor": "#ff7800",
    "fillOpacity": 0.4,
})

# === 7. Exportar como HTML interactivo ===
html_path = os.path.join(output_dir, f"segmentacion_mapa_{timestamp}.html")
m.to_html(outfile=html_path)
print(f"📁 Mapa interactivo guardado en: {html_path}")
