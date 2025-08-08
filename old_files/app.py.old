import streamlit as st
import folium
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import os
import geopandas as gpd
from streamlit_folium import st_folium
from samgeo_utils import run_samgeo_on_tif


# === FUNCIÓN: Cargar TIFF y convertirlo a PNG base64 ===
def tif_to_base64_image(tif_path, bands=[1, 2, 3]):
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        image = src.read(bands)
        image = np.moveaxis(image, 0, -1)

        image = (image - image.min()) / (image.max() - image.min()) * 255
        image = image.astype(np.uint8)

    fig, ax = plt.subplots()
    ax.axis("off")
    plt.imshow(image)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return encoded, bounds


# === CONFIG STREAMLIT ===
st.set_page_config(layout="wide")
st.title("🌍 Visor de capas GeoTIFF con control de opacidad y segmentación SAM-Geo")

# === DEFINIR CAPAS ===
layer_options = {
    "Sentinel-2 RGB": ("data/s2harm_rgb_saa.tif", [1, 2, 3]),
    "Copernicus Urban RGB": ("data/urban_rgb_copernicus.tif", [1, 2, 3]),
    "NDBI Urban Mask": ("data/urban_mask_ndbi.tif", [1])
}

# === CENTRAR MAPA ===
# Usar Sentinel-2 como referencia para center
tif_path_for_center, bands_center = layer_options["Sentinel-2 RGB"]
_, bounds_center = tif_to_base64_image(tif_path_for_center, bands_center)
center = [(bounds_center.top + bounds_center.bottom)/2, (bounds_center.left + bounds_center.right)/2]

# === BLOQUE: Segmentación automática SAM-Geo ===
st.subheader("🧠 Segmentación automática con SAM-Geo")

tif_file = st.selectbox("Selecciona una imagen GeoTIFF para segmentar:", options=list(layer_options.values()), format_func=lambda x: os.path.basename(x[0]))
run_segmentation = st.button("Ejecutar SAM-Geo")

if run_segmentation:
    st.info(f"Procesando segmentación de {tif_file[0]}...")
    mask_tif, vector_gpkg = run_samgeo_on_tif(tif_file[0])
    st.success(f"✅ Segmentación completada: {mask_tif}")

    gdf = gpd.read_file(vector_gpkg)

    m = folium.Map(location=center, zoom_start=15)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="ESRI Satellite"
    ).add_to(m)

    folium.GeoJson(
        gdf,
        name="Máscara SAM-Geo",
        style_function=lambda x: {
            "fillColor": "#ff0000",
            "color": "#000000",
            "weight": 0.8,
            "fillOpacity": 0.4,
        }
    ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width=1400, height=600)

# === BLOQUE: Visualizador de capas con opacidad ===
st.subheader("🗺️ Capas disponibles con control de opacidad")

selected_layers = st.multiselect(
    "Selecciona capas a visualizar:",
    options=list(layer_options.keys()),
    default=list(layer_options.keys())
)

opacity_dict = {}
for layer in selected_layers:
    opacity = st.slider(f"Opacidad para '{layer}'", 0.0, 1.0, 0.7, 0.05)
    opacity_dict[layer] = opacity

# Crear nuevo mapa con capas seleccionadas
m = folium.Map(location=center, zoom_start=15)

# Capa base satélite ESRI
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="ESRI Satellite"
).add_to(m)

# Añadir cada capa seleccionada
for layer in selected_layers:
    path, bands = layer_options[layer]
    encoded, bounds = tif_to_base64_image(path, bands)
    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{encoded}",
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=opacity_dict[layer],
        name=layer
    ).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, width=1400, height=700)
# === FOOTER ===