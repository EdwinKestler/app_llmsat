import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

# Definir el bounding box y tamaño de la imagen (ajústalo según tu imagen real)
bbox = [-90.02, 14.91, -90.00, 14.93]  # [minX, minY, maxX, maxY]
input_path = "data/esri_export.tif"
output_path = "data/esri_export_georef.tif"

with rasterio.open(input_path) as src:
    data = src.read()
    count = src.count
    dtype = src.dtypes[0]
    height, width = src.height, src.width

    transform = from_bounds(*bbox, width, height)
    crs = CRS.from_epsg(4326)  # WGS84

    profile = src.profile
    profile.update({
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': count,
        'dtype': dtype,
        'crs': crs,
        'transform': transform
    })

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data)

print("✅ Imagen georreferenciada guardada como:", output_path)
# El código anterior georreferencia una imagen TIFF utilizando un bounding box definido.
# Asegúrate de ajustar el bounding box y el tamaño de la imagen según tus necesidades.