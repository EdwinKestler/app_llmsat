import os
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import shape
from datetime import datetime
from samgeo import SamGeo, tms_to_geotiff
from samgeo.text_sam import LangSAM
import torch
from rasterio.features import shapes
import folium
from folium import LayerControl
from folium.features import GeoJson, GeoJsonTooltip
import pandas as pd

# === CONFIGURACIÓN GENERAL ===
bbox = [-90.015147, 14.916566, -90.010159, 14.919471]
zoom = 19
output_dir = os.path.abspath("output")
checkpoint_dir = os.path.abspath("checkpoints")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

MIN_AREA = 2  # Área mínima para considerar polígonos válidos

lang_checkpoint = os.path.join(checkpoint_dir, "mobile_sam.pt")
if not os.path.isfile(lang_checkpoint):
    print(f"❌ Checkpoint 'mobile_sam.pt' no se encontró en {lang_checkpoint}")
    raise FileNotFoundError("mobile_sam.pt is required for LangSAM to work.")
else:
    print(f"✅ Checkpoint encontrado: {lang_checkpoint}")

def setup_directories():
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

def download_tms_image(bbox, zoom):
    image_path = os.path.join(output_dir, f"esri_image_{timestamp}.tif")
    tms_to_geotiff(output=image_path, bbox=bbox, zoom=zoom, source="Satellite", overwrite=True)
    if os.path.getsize(image_path) < 10_000:
        raise ValueError("Archivo TMS descargado parece estar corrupto o vacío.")
    print(f"✅ Imagen TMS descargada: {image_path}")
    return image_path

def process_text_segmentation(text_prompts, image_path, profile):
    lang_sam = LangSAM()
    prompt_outputs = {}

    print("🧠 Generando máscaras por texto:", text_prompts)
    for prompt in text_prompts:
        print(f"   📌 Procesando prompt: '{prompt}'")
        lang_sam.predict(image=image_path, text_prompt=prompt, box_threshold=0.24, text_threshold=0.24)
        temp_mask_path = os.path.join(output_dir, f"mask_{prompt}_{timestamp}.tif")
        lang_sam.show_anns(cmap="Greens", add_boxes=True, alpha=1, blend=False, output=temp_mask_path)

        with rasterio.open(temp_mask_path) as src:
            mask = src.read(1)
            transform = src.transform
            crs = src.crs

        geoms, props = [], []
        for geom, value in shapes(mask, transform=transform):
            polygon = shape(geom)
            if value == 0 or polygon.area < MIN_AREA:
                continue
            geoms.append(polygon)
            props.append({"type": prompt, "label": f"{prompt.capitalize()} Feature"})

        vector_out = os.path.join(output_dir, f"segment_{prompt}_{timestamp}.gpkg")
        gdf = gpd.GeoDataFrame(props, geometry=geoms, crs=crs)
        gdf.to_file(vector_out, driver="GPKG")
        gdf.to_file(vector_out.replace('.gpkg', '.shp'), driver="ESRI Shapefile")
        print(f"✅ Segmentación para '{prompt}' guardada en: {vector_out}")

        prompt_outputs[prompt] = vector_out
        os.remove(temp_mask_path)

    return prompt_outputs

def run_sam_segmentation(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = os.path.join(checkpoint_dir, "sam_vit_h_4b8939.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SAM checkpoint not found at {checkpoint_path}.")

    sam = SamGeo(model_type="vit_h", checkpoint=checkpoint_path, device=device)
    filtered_mask_path = os.path.join(output_dir, f"sam_mask_all_{timestamp}.tif")
    sam.generate(source=image_path, output=filtered_mask_path, batch=True, foreground=True)
    print(f"✅ Segmentación SAM completada: {filtered_mask_path}")
    return filtered_mask_path

def save_sam_segments(filtered_mask_path):
    with rasterio.open(filtered_mask_path) as mask_src:
        mask_data = mask_src.read(1)
        transform = mask_src.transform
        crs = mask_src.crs

    geoms, props = [], []
    for geom, value in shapes(mask_data, transform=transform):
        polygon = shape(geom)
        if value == 0 or polygon.area < MIN_AREA:
            continue
        geoms.append(polygon)
        props.append({"type": "SAM_segment", "label": "SAM Segment"})

    vector_out = os.path.join(output_dir, f"sam_segment_{timestamp}.gpkg")
    gdf = gpd.GeoDataFrame(props, geometry=geoms, crs=crs)
    gdf.to_file(vector_out, driver="GPKG")
    gdf.to_file(vector_out.replace('.gpkg', '.shp'), driver="ESRI Shapefile")
    print(f"✅ Segmentación SAM final guardada en: {vector_out}")
    return vector_out

def calculate_urbanization_index(prompt_outputs):
    """Calcula el índice de urbanización como el porcentaje del área cubierta por edificios y caminos respecto al total segmentado."""
    area_urban = 0.0
    total_area = 0.0
    for cls, vector_path in prompt_outputs.items():
        gdf = gpd.read_file(vector_path)
        class_area = gdf.geometry.area.sum()
        total_area += class_area
        if cls in ["building", "road"]:
            area_urban += class_area
    return round((area_urban / total_area) * 100, 2) if total_area > 0 else 0.0


def create_interactive_map_folium(bbox, prompt_outputs, sam_vector_out, output_html):
    """
    Crea el mapa interactivo Folium y añade una caja flotante con el porcentaje de cada segmentación por texto.
    """
    center_lat = (bbox[1] + bbox[3]) / 2
    center_lon = (bbox[0] + bbox[2]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles='Esri.WorldImagery')

    layer_styles = {
        "tree": {"color": "#00CC00", "fillColor": "#00FF00"},
        "river": {"color": "#3399FF", "fillColor": "#66B2FF"},
        "building": {"color": "#8B4513", "fillColor": "#CD853F"},
        "bridge": {"color": "#555555", "fillColor": "#AAAAAA"},
        "SAM_segment": {"color": "#FF6600", "fillColor": "#FF9933"}
    }

    # Añade capas vectoriales segmentadas
    for cls, vector_path in prompt_outputs.items():
        gdf = gpd.read_file(vector_path).to_crs(epsg=4326)
        style = layer_styles.get(cls, {"color": "#000", "fillColor": "#888"})
        geojson = folium.GeoJson(
            gdf,
            name=f"{cls.capitalize()}",
            style_function=lambda x, style=style: {
                "color": style["color"],
                "weight": 1,
                "fillColor": style["fillColor"],
                "fillOpacity": 0.4
            },
            tooltip=GeoJsonTooltip(fields=["label"])
        )
        geojson.add_to(m)

    gdf_sam = gpd.read_file(sam_vector_out).to_crs(epsg=4326)
    folium.GeoJson(
        gdf_sam,
        name="SAM_segment",
        style_function=lambda x: {
            "color": "#FF6600",
            "weight": 1,
            "fillColor": "#FF9933",
            "fillOpacity": 0.2
        },
        tooltip=GeoJsonTooltip(fields=["label"])
    ).add_to(m)

    LayerControl(collapsed=False).add_to(m)

    # === Calcula los porcentajes de cobertura para cada clase segmentada por texto (excluye SAM_segment) ===
    class_areas = {}
    total_area = 0.0
    for cls, vector_path in prompt_outputs.items():
        gdf = gpd.read_file(vector_path)
        area = gdf.geometry.area.sum()
        class_areas[cls] = area
        total_area += area

    # Genera HTML de la tabla de porcentajes
    rows = ""
    for cls in ["tree", "river", "building", "bridge"]:
        pct = (class_areas.get(cls, 0) / total_area * 100) if total_area > 0 else 0.0
        color = layer_styles[cls]["fillColor"]
        rows += f"<tr><td style='padding:4px 8px;'><span style='display:inline-block;width:16px;height:16px;background:{color};margin-right:8px;border-radius:3px;'></span>{cls.capitalize()}</td><td style='padding:4px 8px;text-align:right;font-weight:bold;'>{pct:.2f}%</td></tr>"

    coverage_html = f"""
    <div style='position: fixed; bottom: 30px; left: 30px; min-width: 240px; background-color: white; border: 2px solid #00A878; border-radius: 8px; padding: 14px; font-family: "Inter", sans-serif; font-size: 15px; box-shadow: 2px 2px 10px rgba(0,0,0,0.15); z-index:9999;'>
      <b>Porcentaje de Cobertura</b>
      <table style='width:100%;margin-top:8px;'>
        {rows}
      </table>
    </div>
    """
    m.get_root().html.add_child(folium.Element(coverage_html))

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    m.save(output_html)
    print(f"🌍 Mapa Folium exportado a: {output_html}")

def calculate_class_coverage(prompt_outputs, sam_vector_out):
    print("📊 Calculando cobertura por clase...")
    total_area = 0.0
    class_areas = {}

    for prompt, vector_out in prompt_outputs.items():
        gdf = gpd.read_file(vector_out)
        area = gdf.geometry.area.sum()
        class_areas[prompt] = area
        total_area += area

    gdf_sam = gpd.read_file(sam_vector_out)
    sam_area = gdf_sam.geometry.area.sum()
    class_areas["SAM_segment"] = sam_area
    total_area += sam_area

    coverage_data = []
    for cls, area in class_areas.items():
        pct = (area / total_area) * 100 if total_area > 0 else 0
        coverage_data.append({"class": cls, "area": area, "percentage": pct})
        print(f"   🔹 {cls.capitalize()}: {pct:.2f}% cobertura")

    df = pd.DataFrame(coverage_data)
    csv_path = os.path.join(output_dir, f"coverage_summary_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"📄 Cobertura exportada a CSV: {csv_path}")
    
    

def main():
    setup_directories()
    image_path = download_tms_image(bbox, zoom)
    text_prompts = ["tree", "river", "building", "bridge"]
    with rasterio.open(image_path) as src:
        profile = src.profile
    prompt_outputs = process_text_segmentation(text_prompts, image_path, profile)
    filtered_mask_path = run_sam_segmentation(image_path)
    sam_vector_out = save_sam_segments(filtered_mask_path)
    create_interactive_map_folium(bbox, prompt_outputs, sam_vector_out,
                                  os.path.join(output_dir, "folium_with_satellite_and_all_layers.html"))
    calculate_class_coverage(prompt_outputs, sam_vector_out)
    print("✅ Proceso completado.")

if __name__ == "__main__":
    main()
