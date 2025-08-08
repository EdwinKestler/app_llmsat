import os
import torch
import rasterio
import hashlib
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from samgeo_stub import SamGeo, tms_to_geotiff

# === CONFIGURACIÓN ===
bbox = [-90.015147, 14.916566, -90.010159, 14.919471]  # [west, south, east, north]
zoom = 18
output_dir = "output"
cache_dir = "EsriCache"
checkpoint_path = "checkpoints/sam_vit_h_4b8939.pth"

# === Crear carpetas necesarias ===
os.makedirs(output_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)


# === Función para generar nombre hash del bbox y zoom (único por zona) ===
def cache_name(bbox, zoom):
    key = f"{bbox}-{zoom}".encode()
    return hashlib.md5(key).hexdigest()


# === Rutas ===
cache_file = os.path.join(cache_dir, f"{cache_name(bbox, zoom)}.tif")
image_georef = os.path.join(output_dir, "esri_export_georef.tif")
mask_path = os.path.join(output_dir, "segment.tif")
vector_path = os.path.join(output_dir, "segment.gpkg")

# === 1. Descarga o reutiliza desde caché ===
if not os.path.exists(cache_file):
    print("⏬ Imagen no está en caché. Descargando desde Esri...")
    tms_to_geotiff(
        output=cache_file, bbox=bbox, zoom=zoom, source="Satellite", overwrite=True
    )
else:
    print(f"📂 Usando imagen desde caché: {cache_file}")

# === 2. Georreferenciación ===
print("🌍 Aplicando transformación espacial...")
with rasterio.open(cache_file) as src:
    data = src.read()
    count = src.count
    dtype = src.dtypes[0]
    height, width = src.height, src.width
    transform = from_bounds(*bbox, width, height)
    crs = CRS.from_epsg(4326)

    profile = src.profile.copy()
    profile.update(
        {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": count,
            "dtype": dtype,
            "crs": crs,
            "transform": transform,
        }
    )

    with rasterio.open(image_georef, "w", **profile) as dst:
        dst.write(data)

# === 3. SAM Segmentación ===
print("🧠 Ejecutando SAM...")
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = SamGeo(
    model_type="vit_h",
    checkpoint=checkpoint_path,
    device=device,
    erosion_kernel=(3, 3),
    mask_multiplier=255,
    sam_kwargs=None,
)

sam.generate(image_georef, mask_path)

# === 4. Vectorización ===
print("📦 Exportando segmentación a GPKG...")
sam.tiff_to_gpkg(mask_path, vector_path, simplify_tolerance=None)

print("✅ Proceso completo finalizado:")
print(f" - GeoTIFF cacheado: {cache_file}")
print(f" - Segmentación: {mask_path}")
print(f" - Vectores: {vector_path}")
# El código anterior descarga una imagen de Esri, la georreferencia y aplica segmentación automática usando SAM-Geo.
