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
os.makedirs(output_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# === RUTAS ===
image_path = os.path.join(output_dir, f"esri_image_{timestamp}.tif")
multi_mask_path = os.path.join(output_dir, f"excluded_mask_{timestamp}.tif")
filtered_mask_path = os.path.join(output_dir, f"sam_mask_all_{timestamp}.tif")
vector_out = os.path.join(output_dir, f"filtered_segment_{timestamp}.gpkg")
html_path = os.path.join(output_dir, f"segmentacion_mapa_{timestamp}.html")

# === 1. Descargar imagen TMS ===
tms_to_geotiff(output=image_path, bbox=bbox, zoom=zoom, source="Satellite", overwrite=True)

# === 2. Segmentación por texto múltiple ===
text_prompts = ["tree", "water"]
combined_mask = None

print("🧠 Generando máscaras por texto:", text_prompts)
for prompt in text_prompts:
    print(f"   📌 Procesando prompt: '{prompt}'")
    lang_sam = LangSAM()
    lang_sam.predict(image=image_path, text_prompt=prompt, box_threshold=0.24, text_threshold=0.24)
    
    temp_mask_path = os.path.join(output_dir, f"mask_{prompt}_{timestamp}.tif")
    lang_sam.show_anns(cmap="Greens", add_boxes=True, alpha=1, blend=False, output=temp_mask_path)

    with rasterio.open(temp_mask_path) as src:
        mask = src.read(1)
        if combined_mask is None:
            combined_mask = mask
            profile = src.profile
        else:
            combined_mask = np.logical_or(combined_mask, mask)

# Guardar máscara combinada
profile.update(dtype=rasterio.uint8, count=1)
with rasterio.open(multi_mask_path, 'w', **profile) as dst:
    dst.write(combined_mask.astype(rasterio.uint8), 1)
print(f"✅ Máscara combinada de exclusión guardada: {multi_mask_path}")

# === 3. Segmentación Automática SAM ===
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = SamGeo(
    model_type="vit_h",
    checkpoint=os.path.join(checkpoint_dir, "sam_vit_h_4b8939.pth"),
    device=device,
)
sam.generate(source=image_path, output=filtered_mask_path, batch=True, foreground=True)

# === 4. Filtrar máscaras que intersectan con la combinada ===
print("🧼 Filtrando segmentos que coinciden con exclusión semántica...")
with rasterio.open(image_path) as img_src, rasterio.open(filtered_mask_path) as mask_src, rasterio.open(multi_mask_path) as exclusion_mask_src:
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

    # Excluir si intersecta con la máscara semántica
    if np.any(exclusion_mask[mask_raster] > 0):
        continue

    geoms.append(polygon)
    props.append({})  # puedes agregar atributos si lo deseas

# === 5. Guardar GeoPackage ===
gdf = gpd.GeoDataFrame(props, geometry=geoms, crs=crs)
gdf.to_file(vector_out, driver="GPKG")
print(f"✅ Segmentación final guardada en: {vector_out}")

# === 6. Visualización con Leafmap ===
print("🗺️ Mostrando mapa interactivo...")
m = leafmap.Map(center=[(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2], zoom=zoom)
m.add_basemap("Esri.WorldImagery")
m.add_vector(vector_out, layer_name="Segmentación sin árboles/agua", style={
    "color": "#3388ff",
    "weight": 2,
    "fillColor": "#ff7800",
    "fillOpacity": 0.4,
})
m.to_html(outfile=html_path)
print(f"📁 Mapa HTML exportado: {html_path}")
print("✅ Proceso completado.")