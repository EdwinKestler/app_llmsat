# LLMSat – Geospatial Segmentation & Analysis
# Copyright (C) 2026 Edwin Kestler
# Licensed under AGPL-3.0 — see LICENSE file for details.

"""LLMSat – Streamlit UI with three-step workflow + settings.

Tab 1 (Main):
  Step 1: Select area on interactive map  →  Load satellite imagery
  Step 2: Pick segment prompts via chips   →  Run SAM3 segmentation
  Step 3: Analyze results with AI vision   →  Display overlays, areas, charts

Tab 2 (Settings):
  API keys, model config, thresholds, paths — persisted to config.json
"""

import logging
import shutil
import streamlit as st
import os
from datetime import datetime
import io
import base64
import numpy as np
import altair as alt
import pandas as pd
import rasterio
import folium
from PIL import Image
from dotenv import load_dotenv
from streamlit_folium import st_folium
from matplotlib import cm

# Configure GeoDeep/LLMSat logging to terminal
logging.basicConfig(
    format="%(asctime)s  %(name)-28s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
# Suppress noisy libraries, keep our diagnostics
logging.getLogger("geodeep").setLevel(logging.WARNING)
logging.getLogger("llmsat.geodetector").setLevel(logging.DEBUG)

import config_manager as cfg
from downloader import download_imagery
from segmenter import run_text_segmentation, run_auto_segmentation
from vectorizer import raster_to_vector, summarise
from nl_query.openai_handler import fetch_segment_data
from pipeline.config import PipelineConfig
from pipeline.rfdetector import (
    is_available as rfdetr_available,
    available_models as rfdetr_models,
    run_detection as rfdetr_run_detection,
    result_to_geojson as rfdetr_to_geojson,
    pixel_bboxes_to_geo as rfdetr_bboxes_to_geo,
    RFDetectionResult,
)
from pipeline.geodetector import (
    is_available as geodeep_available,
    available_models as geodeep_models,
    run_detection as geodeep_run_detection,
    run_cpu_segmentation as geodeep_cpu_segment,
    has_cpu_fallback as geodeep_has_fallback,
    detection_result_to_geojson_features,
)

load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ── Config-driven constants ─────────────────────────────────────────

_cfg = cfg.load_config()

SEGMENT_COLORS = {
    k: tuple(v) for k, v in _cfg.get("segment_colors", cfg.DEFAULTS["segment_colors"]).items()
}

DEFAULT_CENTER = _cfg.get("default_center", cfg.DEFAULTS["default_center"])
DEFAULT_BBOX = _cfg.get("default_bbox", cfg.DEFAULTS["default_bbox"])

# ── Helpers ─────────────────────────────────────────────────────────


def _load_rgb(tif_path: str) -> np.ndarray:
    with rasterio.open(tif_path) as src:
        bands = src.read([1, 2, 3])  # single I/O call
    return np.moveaxis(bands, 0, -1)  # (3, H, W) → (H, W, 3)


def _load_mask(tif_path: str) -> np.ndarray:
    with rasterio.open(tif_path) as src:
        return src.read(1) > 0


def _overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple) -> np.ndarray:
    out = rgb.copy()
    alpha = color[3] / 255.0
    for c in range(3):
        out[:, :, c] = np.where(
            mask,
            np.clip(out[:, :, c] * (1 - alpha) + color[c] * alpha, 0, 255).astype(np.uint8),
            out[:, :, c],
        )
    return out


def _image_to_data_url(img_array: np.ndarray, max_side: int = 1024) -> str:
    pil = Image.fromarray(img_array)
    pil.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"






def _bbox_from_folium_bounds(bounds: dict) -> list[float]:
    """Convert folium bounds dict to [west, south, east, north]."""
    sw = bounds["_southWest"]
    ne = bounds["_northEast"]
    return [sw["lng"], sw["lat"], ne["lng"], ne["lat"]]


# ── Session state defaults ──────────────────────────────────────────

if "bbox" not in st.session_state:
    st.session_state.bbox = DEFAULT_BBOX
if "imagery_loaded" not in st.session_state:
    st.session_state.imagery_loaded = False
if "rgb" not in st.session_state:
    st.session_state.rgb = None
if "image_path" not in st.session_state:
    st.session_state.image_path = None
if "segmentation_done" not in st.session_state:
    st.session_state.segmentation_done = False
if "seg_results" not in st.session_state:
    st.session_state.seg_results = {}
if "rfdetr_results" not in st.session_state:
    st.session_state.rfdetr_results = {}
if "detection_results" not in st.session_state:
    st.session_state.detection_results = {}
if "detection_done" not in st.session_state:
    st.session_state.detection_done = False

# ── Page config ─────────────────────────────────────────────────────

st.set_page_config(page_title="LLMSat", layout="wide")

tab_main, tab_training, tab_settings = st.tabs(["Main", "Train Custom Detector", "Settings"])

# =====================================================================
# SETTINGS TAB
# =====================================================================

with tab_settings:
    st.header("Configuration")
    st.caption("Settings are saved to `config.json`. API keys are saved to `.env` (never committed to git).")

    settings_cfg = cfg.load_config()

    # ── API Keys (secrets — saved to .env, not config.json) ─────────
    st.subheader("API Keys")
    col_oai, col_hf = st.columns(2)
    with col_oai:
        oai_key = st.text_input(
            "OpenAI API Key",
            value=cfg.get_secret("OPENAI_API_KEY"),
            type="password",
            help="Used for NL query parsing and AI vision analysis.",
        )
    with col_hf:
        hf_token = st.text_input(
            "HuggingFace Token",
            value=cfg.get_secret("HF_TOKEN"),
            type="password",
            help="Required for SAM3 gated model access.",
        )

    # ── Model & Segmentation ────────────────────────────────────────
    st.subheader("Models & Segmentation")
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        openai_model = st.text_input("OpenAI Model", value=settings_cfg.get("openai_model", "gpt-5.4-nano"))
    with col_m2:
        conf_thresh = st.slider("SAM3 Confidence Threshold", 0.0, 1.0, settings_cfg.get("confidence_threshold", 0.5), 0.05)
    with col_m3:
        mask_thresh = st.slider("SAM3 Mask Threshold", 0.0, 1.0, settings_cfg.get("mask_threshold", 0.5), 0.05)

    # ── Advanced / Legacy Options ────────────────────────────────
    st.subheader("Advanced")

    show_geodeep = st.toggle(
        "Show GeoDeep legacy detection (CPU, low accuracy)",
        value=settings_cfg.get("show_geodeep_legacy", False),
        help=(
            "Enables GeoDeep ONNX models (YOLO v7/v9) in the main tab. "
            "These are CPU-only but have significantly lower accuracy than "
            "RF-DETR. Only use when no GPU is available."
        ),
    )

    cpu_fallback = st.toggle(
        "CPU-only mode (no GPU required)",
        value=settings_cfg.get("cpu_fallback", False),
        help=(
            "When enabled, tree/building/road segmentation uses lightweight "
            "GeoDeep ONNX models instead of SAM3. No GPU needed — runs on any machine. "
            "Accuracy will be lower than SAM3."
        ),
    )
    if cpu_fallback and not geodeep_available():
        st.warning("geodeep package is not installed. Run: `pip install geodeep`")

    hybrid_mode = False  # reserved for future RF-DETR → SAM3 hybrid

    # ── Data Sources ────────────────────────────────────────────────
    st.subheader("Data Sources")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        ob_dir = st.text_input("Open Buildings Directory", value=settings_cfg.get("open_buildings_dir", "openbuildings"))
        ob_min_conf = st.slider("Open Buildings Min Confidence", 0.5, 1.0, settings_cfg.get("open_buildings_min_confidence", 0.65), 0.05)
    with col_d2:
        osm_url = st.text_input("Overpass API URL", value=settings_cfg.get("osm_overpass_url", "https://overpass-api.de/api/interpreter"))
        et_root = st.text_input("ET Monitor Root (optional)", value=settings_cfg.get("et_monitor_root", ""),
                                help="Path to weatherunderground2csv project for data layer overlays.")

    # ── Defaults ────────────────────────────────────────────────────
    st.subheader("Map Defaults")
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        default_zoom = st.slider("Default Zoom", 14, 20, settings_cfg.get("default_zoom", 18))
    with col_b2:
        out_dir_setting = st.text_input("Output Directory", value=settings_cfg.get("output_dir", "output"))

    # ── Save button ─────────────────────────────────────────────────
    if st.button("Save Settings", type="primary"):
        # Validate inputs before saving
        _save_errors = []

        if osm_url:
            url_ok, url_err = cfg.validate_url(osm_url)
            if not url_ok:
                _save_errors.append(f"Overpass URL: {url_err}")

        if out_dir_setting:
            path_ok, path_err = cfg.validate_safe_path(out_dir_setting)
            if not path_ok:
                _save_errors.append(f"Output dir: {path_err}")

        if ob_dir:
            ob_ok, ob_err = cfg.validate_safe_path(ob_dir)
            if not ob_ok:
                _save_errors.append(f"Open Buildings dir: {ob_err}")

        if _save_errors:
            for err in _save_errors:
                st.error(err)
        else:
            # Save secrets to .env (with restricted permissions)
            if oai_key:
                cfg.set_secret("OPENAI_API_KEY", oai_key)
            if hf_token:
                cfg.set_secret("HF_TOKEN", hf_token)

            # Save config to config.json
            settings_cfg.update({
                "openai_model": openai_model,
                "confidence_threshold": conf_thresh,
                "mask_threshold": mask_thresh,
                "show_geodeep_legacy": show_geodeep,
                "cpu_fallback": cpu_fallback,
                "hybrid_mode": hybrid_mode,
                "open_buildings_dir": ob_dir,
                "open_buildings_min_confidence": ob_min_conf,
                "osm_overpass_url": osm_url,
                "et_monitor_root": et_root,
                "default_zoom": default_zoom,
                "output_dir": out_dir_setting,
            })
            cfg.save_config(settings_cfg)
            st.success("Settings saved!")
            st.rerun()

# =====================================================================
# MAIN TAB
# =====================================================================

with tab_main:
    st.title("🌍 LLMSat – Geospatial Segmentation")

    # ── Config values ──────────────────────────────────────────────
    out_dir = _cfg.get("output_dir", "output")
    zoom = _cfg.get("default_zoom", 18)

    with st.sidebar:
        st.image("https://img.icons8.com/color/96/satellite.png", width=48)
        st.markdown("### LLMSat")
        st.caption("Geospatial Segmentation & Analysis")
        st.markdown("---")
        if not cfg.get_secret("OPENAI_API_KEY"):
            st.warning("No OpenAI key — set in Settings tab")
        if not cfg.get_secret("HF_TOKEN"):
            st.warning("No HF token — set in Settings tab")
        st.markdown("---")
        _sources = (
            "**Data sources:**\n"
            "- SAM3 (Meta)\n"
            "- RF-DETR (Roboflow)\n"
            "- Google Open Buildings\n"
            "- OpenStreetMap\n"
            "- OpenAI Vision\n"
        )
        st.markdown(_sources)

    # =====================================================================
    # STEP 1 — Select Area & Load Imagery
    # =====================================================================

    st.header("Step 1: Select Area & Load Imagery")

    # ── Example presets ─────────────────────────────────────────────
    EXAMPLES = {
        "Guatemala City": [-90.5115, 14.6304, -90.5015, 14.6394],
        "Antigua Guatemala": [-90.7380, 14.5540, -90.7280, 14.5640],
        "Tikal (jungle)": [-89.6280, 17.2180, -89.6180, 17.2280],
        "Lake Atitlan": [-91.2150, 14.6850, -91.1950, 14.7050],
    }

    st.caption("Quick presets or draw your own area on the map.")
    preset_cols = st.columns(len(EXAMPLES) + 1)
    for i, (name, ex_bbox) in enumerate(EXAMPLES.items()):
        with preset_cols[i]:
            if st.button(name, key=f"preset_{name}", use_container_width=True):
                st.session_state.bbox = ex_bbox
                st.session_state.imagery_loaded = False
                st.session_state.segmentation_done = False
                st.rerun()

    # ── Map + controls ──────────────────────────────────────────────
    col_map, col_preview = st.columns([3, 2])

    with col_map:
        bbox = st.session_state.bbox
        center_lat = (bbox[1] + bbox[3]) / 2
        center_lng = (bbox[0] + bbox[2]) / 2

        m = folium.Map(location=[center_lat, center_lng], zoom_start=16, tiles="Esri.WorldImagery")
        folium.plugins.Draw(
            draw_options={
                "rectangle": {"shapeOptions": {"color": "#ff0000", "weight": 2}},
                "polygon": False, "polyline": False, "circle": False,
                "circlemarker": False, "marker": False,
            },
            edit_options={"edit": False},
        ).add_to(m)
        folium.Rectangle(
            bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
            color="#3388ff", weight=2, fill=True, fill_opacity=0.1,
        ).add_to(m)

        map_data = st_folium(m, height=380, width=None, returned_objects=["last_active_drawing"])

        # Update bbox from drawn rectangle
        if map_data and map_data.get("last_active_drawing"):
            drawing = map_data["last_active_drawing"]
            if drawing.get("geometry", {}).get("type") == "Polygon":
                coords = drawing["geometry"]["coordinates"][0]
                lngs = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                st.session_state.bbox = [min(lngs), min(lats), max(lngs), max(lats)]
                st.session_state.imagery_loaded = False
                st.session_state.segmentation_done = False

        # Compact input row: bbox + zoom + load button
        input_col1, input_col2, input_col3 = st.columns([5, 1, 2])
        with input_col1:
            bbox_str = st.text_input(
                "Bbox",
                value=", ".join(f"{v:.6f}" for v in st.session_state.bbox),
                label_visibility="collapsed",
                help="west, south, east, north (EPSG:4326)",
            )
            try:
                manual_bbox = [float(x.strip()) for x in bbox_str.split(",")]
                if manual_bbox != st.session_state.bbox:
                    ok, err = cfg.validate_bbox(manual_bbox)
                    if ok:
                        st.session_state.bbox = manual_bbox
                        st.session_state.imagery_loaded = False
                        st.session_state.segmentation_done = False
            except ValueError:
                pass
        with input_col2:
            zoom = st.selectbox("Zoom", list(range(14, 21)), index=zoom - 14, label_visibility="collapsed")
        with input_col3:
            load_btn = st.button("📡 Load Imagery", type="primary", use_container_width=True)

    with col_preview:
        bbox = st.session_state.bbox
        w = abs(bbox[2] - bbox[0]) * 111320 * np.cos(np.radians((bbox[1] + bbox[3]) / 2))
        h = abs(bbox[3] - bbox[1]) * 110540

        metric_cols = st.columns(2)
        with metric_cols[0]:
            st.metric("Width", f"{w:.0f} m")
        with metric_cols[1]:
            st.metric("Height", f"{h:.0f} m")

        if st.session_state.imagery_loaded and st.session_state.rgb is not None:
            st.image(st.session_state.rgb, caption="Loaded satellite imagery", width="stretch")
        else:
            st.info("Select an area and click **Load Imagery**.")

    # ── Process load ────────────────────────────────────────────────
    if load_btn:
        # Validate bbox
        bbox = st.session_state.bbox
        bbox_ok, bbox_err = cfg.validate_bbox(bbox)
        if not bbox_ok:
            st.error(f"Invalid bounding box: {bbox_err}")
            st.stop()

        # Validate output path is safe (within project)
        path_ok, path_err = cfg.validate_safe_path(out_dir)
        if not path_ok:
            st.error(f"Invalid output directory: {path_err}")
            st.stop()

        imagery_dir = os.path.join(out_dir, "_imagery")

        if os.path.isdir(imagery_dir):
            shutil.rmtree(imagery_dir)
        for entry in os.listdir(out_dir) if os.path.isdir(out_dir) else []:
            entry_path = os.path.join(out_dir, entry)
            if os.path.isdir(entry_path) and entry != "_imagery":
                shutil.rmtree(entry_path)

        config = PipelineConfig(bbox=bbox, zoom=zoom, out_dir=imagery_dir)
        status = st.status("Downloading satellite tiles...", expanded=True)
        with status:
            try:
                image_path = download_imagery(config=config, overwrite=True)
                rgb = _load_rgb(image_path)
                st.session_state.rgb = rgb
                st.session_state.image_path = image_path
                st.session_state.imagery_loaded = True
                st.session_state.segmentation_done = False
                st.session_state.seg_results = {}
                status.update(label="Imagery loaded!", state="complete")
                st.rerun()
            except Exception as e:
                status.update(label=f"Download failed: {e}", state="error")

    # =====================================================================
    # STEP 2 — Segment
    # =====================================================================

    if st.session_state.imagery_loaded:
        st.markdown("---")
        st.header("Step 2: Select Segments & Run SAM3")

        # Prompt chips + curated dropdown
        st.caption("Select what to detect. SAM3 works best with specific object terms.")
        seg_presets = {"🌳 Tree": "tree", "🏠 Building": "building", "💧 Water": "water", "🛣️ Road": "road"}

        # SAM3-compatible terms (tested to produce results)
        _SAM3_EXTRA_TERMS = [
            "car", "vehicle", "truck", "bus",
            "parking lot", "swimming pool", "tennis court",
            "solar panel", "roof", "fence",
            "grass", "field", "crop", "farmland",
            "river", "lake", "pond",
            "shadow", "bare soil", "rock",
            "bridge", "railway", "runway",
            "container", "construction site",
        ]

        chip_cols = st.columns(len(seg_presets) + 1)
        selected = []
        for i, (label, key) in enumerate(seg_presets.items()):
            with chip_cols[i]:
                if st.checkbox(label, value=(key in ["tree", "building"]), key=f"chip_{key}"):
                    selected.append(key)

        with chip_cols[len(seg_presets)]:
            extra = st.selectbox(
                "More",
                options=[""] + _SAM3_EXTRA_TERMS,
                index=0,
                label_visibility="collapsed",
                help="Additional SAM3-compatible segment types.",
            )
            if extra:
                selected.append(extra)

        # Color picker expander
        with st.expander("Customize segment colors"):
            color_cols = st.columns(max(len(selected), 1))
            for i, seg in enumerate(selected):
                with color_cols[i % len(color_cols)]:
                    current = SEGMENT_COLORS.get(seg, (255, 255, 0, 140))
                    hex_color = f"#{current[0]:02x}{current[1]:02x}{current[2]:02x}"
                    new_hex = st.color_picker(seg.capitalize(), hex_color, key=f"color_{seg}")
                    r_v = int(new_hex[1:3], 16)
                    g_v = int(new_hex[3:5], 16)
                    b_v = int(new_hex[5:7], 16)
                    SEGMENT_COLORS[seg] = (r_v, g_v, b_v, 140)

        # Show Open Buildings preview for building prompts
        if "building" in selected:
            try:
                from open_buildings import query_buildings, buildings_summary
                _ob_gdf = query_buildings(st.session_state.bbox, min_confidence=0.7, max_buildings=500)
                _ob_info = buildings_summary(_ob_gdf)
                if _ob_info["count"] > 0:
                    st.success(
                        f"Open Buildings: **{_ob_info['count']}** known buildings in this area "
                        f"(avg conf: {_ob_info['avg_confidence']:.2f}). "
                        f"These will be used as box prompts for SAM3."
                    )

                    # Preview: satellite image with building polygons overlaid
                    with st.expander("Preview: Open Buildings footprints", expanded=True):
                        preview_rgb = st.session_state.rgb.copy()

                        # We need to project polygon coords to pixel coords
                        # Read the geotransform from the imagery
                        img_path = st.session_state.image_path
                        with rasterio.open(img_path) as src:
                            transform = src.transform
                            img_h, img_w = src.height, src.width

                        from rasterio.transform import rowcol
                        from PIL import ImageDraw

                        pil_img = Image.fromarray(preview_rgb)
                        draw = ImageDraw.Draw(pil_img, "RGBA")

                        drawn = 0
                        for _, row in _ob_gdf.iterrows():
                            geom = row.geometry
                            if geom is None or geom.is_empty:
                                continue
                            # Get exterior ring coordinates
                            if geom.geom_type == "MultiPolygon":
                                polys = list(geom.geoms)
                            else:
                                polys = [geom]

                            for poly in polys:
                                coords = list(poly.exterior.coords)
                                pixel_coords = []
                                for lon, lat in coords:
                                    r, c = rowcol(transform, lon, lat)
                                    # Clamp to image bounds
                                    r = max(0, min(img_h - 1, int(r)))
                                    c = max(0, min(img_w - 1, int(c)))
                                    pixel_coords.append((c, r))  # PIL uses (x, y)

                                if len(pixel_coords) >= 3:
                                    conf = row["confidence"]
                                    # Color by confidence: red (low) → green (high)
                                    g = int(min(255, conf * 300))
                                    r_val = int(min(255, (1 - conf) * 300))
                                    draw.polygon(pixel_coords, fill=(r_val, g, 0, 60),
                                                 outline=(r_val, g, 0, 200))
                                    drawn += 1

                        preview_arr = np.array(pil_img)
                        col_prev, col_stats = st.columns([3, 1])
                        with col_prev:
                            st.image(preview_arr, caption=f"{drawn} building footprints from Google Open Buildings",
                                     width="stretch")
                        with col_stats:
                            st.metric("Buildings", f"{_ob_info['count']:,}")
                            st.metric("Total Area", f"{_ob_info['total_area_m2']:,.0f} m²")
                            st.metric("Avg Area", f"{_ob_info['avg_area_m2']:,.0f} m²")
                            st.metric("Avg Confidence", f"{_ob_info['avg_confidence']:.2f}")
                else:
                    st.info("No Open Buildings data for this area — using SAM3 text prompts only.")
            except Exception as e:
                st.info(f"Open Buildings preview unavailable — using SAM3 text prompts only. ({e})")

        # Show OSM road preview for road prompts
        if "road" in selected:
            try:
                from osm_roads import query_roads, buffer_roads, roads_summary
                _rd_gdf = query_roads(st.session_state.bbox)
                _rd_info = roads_summary(_rd_gdf)
                if _rd_info["count"] > 0:
                    st.success(
                        f"OpenStreetMap: **{_rd_info['count']}** road segments in this area. "
                        f"These will be rasterized directly as the road mask."
                    )

                    with st.expander("Preview: OSM road network", expanded=True):
                        preview_rgb = st.session_state.rgb.copy()
                        img_path = st.session_state.image_path

                        with rasterio.open(img_path) as src:
                            transform = src.transform
                            img_h, img_w = src.height, src.width

                        from rasterio.transform import rowcol
                        from PIL import ImageDraw

                        pil_img = Image.fromarray(preview_rgb)
                        draw = ImageDraw.Draw(pil_img, "RGBA")

                        # Road type colors
                        _road_colors = {
                            "primary": (255, 200, 0, 180),
                            "secondary": (255, 150, 0, 160),
                            "tertiary": (255, 100, 0, 140),
                            "residential": (200, 200, 200, 140),
                            "service": (150, 150, 150, 120),
                        }

                        drawn = 0
                        for _, row in _rd_gdf.iterrows():
                            geom = row.geometry
                            if geom is None or geom.is_empty:
                                continue
                            coords = list(geom.coords)
                            pixel_coords = []
                            for lon, lat in coords:
                                r, c = rowcol(transform, lon, lat)
                                r = max(0, min(img_h - 1, int(r)))
                                c = max(0, min(img_w - 1, int(c)))
                                pixel_coords.append((c, r))

                            if len(pixel_coords) >= 2:
                                color = _road_colors.get(row["highway"], (200, 200, 0, 140))
                                width_px = max(1, int(row["width_m"] * 0.8))
                                draw.line(pixel_coords, fill=color, width=width_px)
                                drawn += 1

                        preview_arr = np.array(pil_img)
                        col_prev, col_stats = st.columns([3, 1])
                        with col_prev:
                            st.image(preview_arr, caption=f"{drawn} road segments from OpenStreetMap",
                                     width="stretch")
                        with col_stats:
                            st.metric("Roads", f"{_rd_info['count']:,}")
                            for rtype, count in list(_rd_info["types"].items())[:5]:
                                st.caption(f"{rtype}: {count}")
                else:
                    st.info("No OSM road data for this area — using SAM3 text prompts only.")
            except Exception as e:
                st.info(f"OSM road preview unavailable — using SAM3 text prompts only. ({e})")

        # ── RF-DETR Object Detection (GPU, high accuracy) ──────────────
        selected_rfdetr = []
        selected_detections = []  # legacy GeoDeep, populated only if enabled

        if rfdetr_available():
            st.markdown("---")
            st.subheader("Object Detection (RF-DETR)")
            st.caption(
                "Transformer-based detection (ICLR 2026). 90 mAP on aerial imagery — "
                "no NMS, excellent at small objects. Runs on GPU."
            )

            _rf_models = rfdetr_models()
            rf_cols = st.columns(len(_rf_models))
            for i, (key, info) in enumerate(_rf_models.items()):
                with rf_cols[i]:
                    if st.checkbox(
                        f"{info['icon']} {info['label']}",
                        key=f"rf_{key}",
                        help=info["description"],
                    ):
                        selected_rfdetr.append(key)

            if selected_rfdetr:
                _rf_opt_cols = st.columns(3)
                with _rf_opt_cols[0]:
                    _rf_thresh = st.slider(
                        "Confidence threshold", 0.1, 0.9,
                        _cfg.get("rfdetr_threshold", 0.3), 0.05,
                        key="rf_thresh",
                    )
                with _rf_opt_cols[1]:
                    _rf_aerial = st.checkbox(
                        "Aerial classes only",
                        value=_cfg.get("rfdetr_aerial_only", True),
                        key="rf_aerial",
                        help="Filter to vehicles, boats, planes, people — skip indoor objects.",
                    )

        # ── GeoDeep legacy (opt-in from Settings) ─────────────────────
        if _cfg.get("show_geodeep_legacy", False) and geodeep_available():
            with st.expander("Legacy CPU detection (GeoDeep) — low accuracy", expanded=False):
                st.caption(
                    "YOLO v7/v9 ONNX models. CPU-only but significantly lower accuracy "
                    "than RF-DETR. Use only when no GPU is available."
                )
                _gd_models = geodeep_models()
                _gd_det_models = {k: v for k, v in _gd_models.items() if v["type"] == "detection"}
                _gd_seg_models = {k: v for k, v in _gd_models.items() if v["type"] == "segmentation"}

                gd_cols = st.columns(min(len(_gd_det_models) + len(_gd_seg_models), 5))
                idx = 0
                for key, info in {**_gd_det_models, **_gd_seg_models}.items():
                    with gd_cols[idx % len(gd_cols)]:
                        if st.checkbox(
                            f"{info['icon']} {info['label']}",
                            key=f"gd_{key}",
                            help=info["description"],
                        ):
                            selected_detections.append(key)
                    idx += 1

        if not selected and not selected_rfdetr and not selected_detections:
            st.warning("Select at least one segment type or detection model.")
        elif selected or selected_rfdetr or selected_detections:
            seg_col1, seg_col2 = st.columns([1, 3])
            with seg_col1:
                segment_btn = st.button("🔍 Run Segmentation", type="primary", width="stretch")

            if segment_btn:
                _shutil = shutil  # noqa: local alias

                bbox = st.session_state.bbox
                image_path = st.session_state.image_path
                seg_results = {}

                _use_cpu = _cfg.get("cpu_fallback", False)
                _status_label = "Running CPU segmentation (GeoDeep)..." if _use_cpu else "Running SAM3 segmentation..."
                status = st.status(_status_label, expanded=True)
                with status:
                    for seg in selected:
                        seg_out_dir = os.path.join(out_dir, seg)

                        # Clear stale outputs for this segment so nothing is reused
                        if os.path.isdir(seg_out_dir):
                            _shutil.rmtree(seg_out_dir)
                        os.makedirs(seg_out_dir, exist_ok=True)

                        # Copy current imagery into segment dir
                        seg_image = os.path.join(seg_out_dir, "s2harm_rgb_saa.tif")
                        _shutil.copy2(image_path, seg_image)

                        # Track which path was used (for skipping auto-seg)
                        _used_alt_path = False

                        # ── CPU fallback path ─────────────────────────
                        if _use_cpu and geodeep_has_fallback(seg):
                            st.write(f"Segmenting **{seg}** (CPU / GeoDeep)...")
                            try:
                                has_data, mask_path = geodeep_cpu_segment(
                                    seg_image, seg, seg_out_dir,
                                )
                                if has_data:
                                    with rasterio.open(mask_path) as src:
                                        n_pixels = np.count_nonzero(src.read(1))
                                    st.write(f"  ✅ **{seg}**: detected ({n_pixels:,} pixels) — CPU")
                                else:
                                    st.write(f"  ⚠️ **{seg}**: no objects found (CPU)")
                                _used_alt_path = True
                            except Exception as e:
                                has_data = False
                                mask_path = os.path.join(seg_out_dir, "langsam_mask.tif")
                                st.write(f"  ❌ **{seg}** CPU fallback failed: {e}")
                                _used_alt_path = True

                        # ── Standard SAM3 path ────────────────────────
                        if not _used_alt_path:
                            st.write(f"Segmenting **{seg}**...")
                            config = PipelineConfig(
                                bbox=bbox, zoom=zoom, out_dir=seg_out_dir,
                                box_threshold=_cfg.get("confidence_threshold", 0.5),
                                text_threshold=_cfg.get("mask_threshold", 0.5),
                            )
                            try:
                                mask_path = run_text_segmentation(seg_image, [seg], config)
                                with rasterio.open(mask_path) as src:
                                    mask_data = src.read(1)
                                has_data = mask_data.any()
                                n_pixels = np.count_nonzero(mask_data)

                                if has_data:
                                    st.write(f"  ✅ **{seg}**: detected ({n_pixels:,} pixels)")
                                else:
                                    st.write(f"  ⚠️ **{seg}**: no objects found")
                            except Exception as e:
                                has_data = False
                                st.write(f"  ❌ **{seg}**: error — {e}")

                        # Auto segmentation + vectorise
                        try:
                            if not _used_alt_path:
                                config = PipelineConfig(
                                    bbox=bbox, zoom=zoom, out_dir=seg_out_dir,
                                    box_threshold=_cfg.get("confidence_threshold", 0.5),
                                    text_threshold=_cfg.get("mask_threshold", 0.5),
                                )
                                auto_mask = run_auto_segmentation(seg_image, config)
                            gpkg_path = os.path.join(seg_out_dir, "segments.gpkg")
                            csv_path = os.path.join(seg_out_dir, "summary.csv")
                            gdf = raster_to_vector(mask_path, gpkg_path)
                            summarise(gdf, csv_path)
                        except Exception as e:
                            st.write(f"  ⚠️ Vectorisation skipped for {seg}: {e}")

                        seg_results[seg] = {"has_data": has_data, "out_dir": seg_out_dir}

                    st.session_state.seg_results = seg_results
                    st.session_state.segmentation_done = True
                    st.session_state.selected_segments = selected

                    # ── RF-DETR detection pass (primary) ──────────────
                    rfdetr_results = {}
                    if selected_rfdetr:
                        st.write("---")
                        st.write("**Running RF-DETR object detection (GPU)...**")
                        from pipeline.geodetector import _inspect_geotiff
                        _img_diag = _inspect_geotiff(image_path)
                        st.caption(
                            f"Input: {_img_diag['width']}x{_img_diag['height']} px  "
                            f"| {_img_diag['res_cm_x']:.1f}x{_img_diag['res_cm_y']:.1f} cm/px  "
                            f"| CRS: {_img_diag['crs']}"
                        )

                        for rf_key in selected_rfdetr:
                            rf_info = rfdetr_models().get(rf_key, {})
                            st.write(f"  Detecting **{rf_info.get('label', rf_key)}**...")
                            result = rfdetr_run_detection(
                                image_path,
                                rf_key,
                                threshold=_rf_thresh if '_rf_thresh' in dir() else 0.3,
                                aerial_only=_rf_aerial if '_rf_aerial' in dir() else True,
                            )
                            if result.error:
                                st.write(f"  ❌ {rf_info.get('label', rf_key)}: {result.error}")
                            elif result.count > 0:
                                from collections import Counter as _Ctr
                                _cls = _Ctr(result.classes)
                                _cls_str = ", ".join(f"{c}: {n}" for c, n in _cls.most_common())
                                _scores = result.scores
                                _avg = sum(_scores) / len(_scores) if _scores else 0
                                _mn = min(_scores) if _scores else 0
                                _mx = max(_scores) if _scores else 0
                                st.write(
                                    f"  ✅ **{rf_info.get('label', rf_key)}**: "
                                    f"**{result.count}** objects in {result.elapsed:.1f}s"
                                )
                                st.caption(
                                    f"  Classes: {_cls_str}  |  "
                                    f"Confidence: min={_mn:.3f}  avg={_avg:.3f}  max={_mx:.3f}"
                                )
                            else:
                                st.write(
                                    f"  ⚠️ {rf_info.get('label', rf_key)}: "
                                    f"no objects found ({result.elapsed:.1f}s)"
                                )
                            rfdetr_results[rf_key] = result

                    # ── GeoDeep legacy detection (only if enabled) ────
                    detection_results = {}
                    if selected_detections:
                        st.write("---")
                        st.write("**Running GeoDeep legacy detection (CPU)...**")
                        for det_key in selected_detections:
                            det_info = geodeep_models().get(det_key, {})
                            st.write(f"  Detecting **{det_info.get('label', det_key)}**...")
                            result = geodeep_run_detection(
                                image_path, det_key,
                                progress_callback=lambda text, pct: None,
                            )
                            if result.error:
                                st.write(f"  ❌ {det_info.get('label', det_key)}: {result.error}")
                            elif result.count > 0:
                                st.write(f"  ✅ **{det_info.get('label', det_key)}**: **{result.count}** objects")
                            else:
                                st.write(f"  ⚠️ {det_info.get('label', det_key)}: no objects found")
                            detection_results[det_key] = result

                    st.session_state.rfdetr_results = rfdetr_results
                    st.session_state.detection_results = detection_results
                    st.session_state.detection_done = bool(rfdetr_results) or bool(detection_results)

                    # Reset AI chat for new segmentation
                    st.session_state.vision_chat = []
                    st.session_state.vision_analysis_done = False
                    status.update(label="Segmentation complete!", state="complete")

                st.rerun()

    # =====================================================================
    # STEP 3 — Results & Analysis
    # =====================================================================

    if st.session_state.segmentation_done:
        st.markdown("---")
        st.header("Step 3: Results & Analysis")

        seg_results = st.session_state.seg_results
        segments = st.session_state.selected_segments

        # Load RGB from a segment directory (matches mask resolution)
        rgb = None
        for seg in segments:
            info = seg_results.get(seg, {})
            seg_dir = info.get("out_dir", os.path.join(out_dir, seg))
            seg_tif = os.path.join(seg_dir, "s2harm_rgb_saa.tif")
            if os.path.exists(seg_tif):
                rgb = _load_rgb(seg_tif)
                break
        if rgb is None:
            rgb = st.session_state.rgb  # fallback to preview imagery

        # ── Combined overlay ────────────────────────────────────────────

        st.subheader("Satellite Imagery & Segmentation")
        col_orig, col_overlay = st.columns(2)

        overlay = rgb.copy()
        mask_found = {}
        for seg in segments:
            info = seg_results.get(seg, {})
            seg_dir = info.get("out_dir", os.path.join(out_dir, seg))
            mask_path = os.path.join(seg_dir, "langsam_mask.tif")
            if os.path.exists(mask_path):
                mask = _load_mask(mask_path)
                has_data = bool(mask.any())
                mask_found[seg] = has_data
                if has_data:
                    color = SEGMENT_COLORS.get(seg, (255, 255, 0, 140))
                    overlay = _overlay_mask(overlay, mask, color)
            else:
                mask_found[seg] = False

        with col_orig:
            st.markdown("**Original Satellite Image**")
            st.image(rgb, width="stretch")
        with col_overlay:
            st.markdown("**Segmentation Overlay**")
            st.image(overlay, width="stretch")

        # Legend
        legend_items = []
        for seg in segments:
            c = SEGMENT_COLORS.get(seg, (255, 255, 0, 140))
            status = "detected" if mask_found.get(seg) else "none found"
            swatch = f"<span style='display:inline-block;width:14px;height:14px;background:rgb({c[0]},{c[1]},{c[2]});margin-right:6px;vertical-align:middle;border-radius:2px;'></span>"
            legend_items.append(f"{swatch} **{seg}** ({status})")
        st.markdown(" &nbsp;&nbsp;&nbsp; ".join(legend_items), unsafe_allow_html=True)

        # ── Per-segment detail ──────────────────────────────────────────

        st.subheader("Segment Details")

        for seg in segments:
            info = seg_results.get(seg, {})
            seg_dir = info.get("out_dir", os.path.join(out_dir, seg))
            mask_path = os.path.join(seg_dir, "langsam_mask.tif")
            unique_path = os.path.join(seg_dir, f"sam3_{seg}_masks.tif")
            scores_path = os.path.join(seg_dir, f"sam3_{seg}_scores.tif")

            with st.expander(f"**{seg.capitalize()}**" + (" ✅" if mask_found.get(seg) else " ⚠️"), expanded=mask_found.get(seg, False)):
                if not mask_found.get(seg):
                    st.info(f"No objects detected for '{seg}'. Try a different prompt (e.g. 'roof' instead of 'building').")
                    continue

                cols = st.columns(3)
                with cols[0]:
                    st.markdown("**Binary Overlay**")
                    mask = _load_mask(mask_path)
                    color = SEGMENT_COLORS.get(seg, (255, 255, 0, 140))
                    st.image(_overlay_mask(rgb.copy(), mask, color), width="stretch")

                with cols[1]:
                    st.markdown("**Instance Masks**")
                    if os.path.exists(unique_path):
                        with rasterio.open(unique_path) as src:
                            unique_data = src.read(1)
                        n_instances = len(np.unique(unique_data)) - 1
                        st.caption(f"{n_instances} instances detected")
                        if unique_data.max() > 0:
                            colored = (cm.tab20(unique_data % 20)[:, :, :3] * 255).astype(np.uint8)
                            colored[unique_data == 0] = rgb[unique_data == 0]
                            st.image(colored, width="stretch")
                    else:
                        st.info("Not available")

                with cols[2]:
                    st.markdown("**Confidence Scores**")
                    if os.path.exists(scores_path):
                        with rasterio.open(scores_path) as src:
                            scores_data = src.read(1).astype(float)
                        if scores_data.max() > 0:
                            norm = scores_data / scores_data.max()
                            colored_scores = (cm.coolwarm(norm)[:, :, :3] * 255).astype(np.uint8)
                            colored_scores[scores_data == 0] = rgb[scores_data == 0]
                            st.image(colored_scores, width="stretch")
                            st.caption(f"Range: {scores_data[scores_data > 0].min():.2f} – {scores_data.max():.2f}")
                    else:
                        st.info("Not available")

        # ── RF-DETR detection results (primary) ─────────────────────────

        rfdetr_results = st.session_state.get("rfdetr_results", {})
        detection_results = st.session_state.get("detection_results", {})

        def _draw_bbox_overlay(rgb_img, result_bboxes, result_scores, result_classes, is_pixel_coords=True):
            """Draw bounding boxes on image. Returns (image_array, class_counts)."""
            from PIL import ImageDraw
            _det_colors = [
                (0, 255, 0, 200), (255, 50, 50, 200), (50, 50, 255, 200),
                (255, 255, 0, 200), (255, 0, 255, 200), (0, 255, 255, 200),
                (255, 128, 0, 200), (128, 0, 255, 200),
            ]
            pil_img = Image.fromarray(rgb_img.copy())
            draw = ImageDraw.Draw(pil_img, "RGBA")
            img_h, img_w = rgb_img.shape[:2]

            unique_cls = sorted(set(result_classes))
            cls_cmap = {c: _det_colors[i % len(_det_colors)] for i, c in enumerate(unique_cls)}

            if is_pixel_coords:
                for bbox, score, cls in zip(result_bboxes, result_scores, result_classes):
                    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_w, x2), min(img_h, y2)
                    color = cls_cmap.get(cls, (0, 255, 0, 200))
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            else:
                from rasterio.transform import rowcol
                with rasterio.open(st.session_state.image_path) as src:
                    transform = src.transform
                for bbox, score, cls in zip(result_bboxes, result_scores, result_classes):
                    r1, c1 = rowcol(transform, bbox[0], bbox[3])
                    r2, c2 = rowcol(transform, bbox[2], bbox[1])
                    r1, c1 = max(0, min(img_h-1, int(r1))), max(0, min(img_w-1, int(c1)))
                    r2, c2 = max(0, min(img_h-1, int(r2))), max(0, min(img_w-1, int(c2)))
                    color = cls_cmap.get(cls, (0, 255, 0, 200))
                    draw.rectangle([c1, r1, c2, r2], outline=color, width=2)

            from collections import Counter
            return np.array(pil_img), Counter(result_classes)

        if rfdetr_results:
            st.subheader("Object Detection Results (RF-DETR)")

            # Summary metrics
            rf_metric_cols = st.columns(min(len(rfdetr_results), 4))
            for i, (rf_key, result) in enumerate(rfdetr_results.items()):
                with rf_metric_cols[i % len(rf_metric_cols)]:
                    if result.error:
                        st.metric(f"🎯 {result.label}", "Error")
                    else:
                        st.metric(f"🎯 {result.label}", f"{result.count}")

            for rf_key, result in rfdetr_results.items():
                if result.error or result.count == 0:
                    continue

                with st.expander(
                    f"**🎯 {result.label}** — {result.count} objects ({result.elapsed:.1f}s)",
                    expanded=True,
                ):
                    det_arr, cls_counts = _draw_bbox_overlay(
                        rgb, result.bboxes, result.scores, result.classes,
                        is_pixel_coords=True,
                    )
                    col_img, col_stats = st.columns([3, 1])
                    with col_img:
                        st.image(det_arr, caption=f"{result.count} RF-DETR detections", width="stretch")
                    with col_stats:
                        for cls_name, cnt in cls_counts.most_common():
                            st.caption(f"{cls_name}: {cnt}")
                        if result.scores:
                            avg_score = sum(result.scores) / len(result.scores)
                            st.metric("Avg confidence", f"{avg_score:.2f}")
                            st.metric("Inference", f"{result.elapsed:.2f}s")

        # ── GeoDeep legacy detection results ─────────────────────────────
        if detection_results:
            with st.expander("Legacy Detection Results (GeoDeep CPU)", expanded=False):
                for det_key, result in detection_results.items():
                    if result.error or result.count == 0:
                        continue
                    det_info = geodeep_models().get(det_key, {})
                    if hasattr(result, 'bboxes') and result.bboxes:
                        det_arr, cls_counts = _draw_bbox_overlay(
                            rgb, result.bboxes, result.scores, result.classes,
                            is_pixel_coords=False,
                        )
                        st.image(det_arr, caption=f"{result.count} GeoDeep detections ({result.label})")
                    elif hasattr(result, 'geojson') and result.geojson:
                        st.info(f"{result.count} polygon features ({result.label})")

        # ── Area summary ────────────────────────────────────────────────

        st.subheader("Area Summary")

        area_data = []
        for seg in segments:
            info = seg_results.get(seg, {})
            seg_dir = info.get("out_dir", os.path.join(out_dir, seg))
            try:
                _, area = fetch_segment_data(seg, seg_dir)
            except Exception:
                area = 0.0
            area_data.append({"segment": seg, "area_m2": area})

        df = pd.DataFrame(area_data)

        col_table, col_chart = st.columns([1, 2])
        with col_table:
            st.dataframe(df.style.format({"area_m2": "{:,.1f}"}), width="stretch")
            total = df["area_m2"].sum()
            st.metric("Total Area", f"{total:,.1f} m²")
        with col_chart:
            if not df.empty and total > 0:
                chart = (
                    alt.Chart(df)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X("segment:N", title="Segment"),
                        y=alt.Y("area_m2:Q", title="Area (m²)"),
                        color=alt.Color("segment:N", legend=None),
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart, width="stretch")

        # ── AI Vision Analysis & Chat ─────────────────────────────────

        if cfg.get_secret("OPENAI_API_KEY") and OpenAI is not None:
            st.subheader("AI Vision Analysis")

            if "vision_chat" not in st.session_state:
                st.session_state.vision_chat = []
            if "vision_analysis_done" not in st.session_state:
                st.session_state.vision_analysis_done = False

            # ── Collect ALL segment images + metadata (cached) ────────
            area_lines = [f"- {r['segment']}: {r['area_m2']:,.1f} m²" for _, r in df.iterrows()]
            area_context = "\n".join(area_lines) or "No area data."

            # Build vision images only once per segmentation run
            if "vision_images" not in st.session_state or not st.session_state.get("vision_analysis_done"):
                seg_detail_lines = []
                vision_images = []

                vision_images.append(("Original satellite imagery", _image_to_data_url(rgb)))
                vision_images.append(("Combined segmentation overlay", _image_to_data_url(overlay)))

                # Per-segment: binary overlay + instance mask + stats
                for seg in segments:
                    info = seg_results.get(seg, {})
                    seg_dir = info.get("out_dir", os.path.join(out_dir, seg))
                    mask_path = os.path.join(seg_dir, "langsam_mask.tif")
                    unique_path = os.path.join(seg_dir, f"sam3_{seg}_masks.tif")
                    scores_path = os.path.join(seg_dir, f"sam3_{seg}_scores.tif")

                    n_instances = 0
                    n_pixels = 0
                    avg_conf = 0.0

                    # Binary overlay per segment
                    if os.path.exists(mask_path) and mask_found.get(seg):
                        mask = _load_mask(mask_path)
                        n_pixels = int(mask.sum())
                        color = SEGMENT_COLORS.get(seg, (255, 255, 0, 140))
                        seg_overlay = _overlay_mask(rgb.copy(), mask, color)
                        vision_images.append((
                            f"{seg.capitalize()} binary overlay ({n_pixels:,} pixels)",
                            _image_to_data_url(seg_overlay),
                        ))

                    # Instance mask — count unique objects
                    if os.path.exists(unique_path):
                        with rasterio.open(unique_path) as src:
                            unique_data = src.read(1)
                        n_instances = len(np.unique(unique_data)) - 1  # exclude 0
                        if n_instances > 0 and unique_data.max() > 0:
                            colored = (cm.tab20(unique_data % 20)[:, :, :3] * 255).astype(np.uint8)
                            colored[unique_data == 0] = rgb[unique_data == 0]
                            vision_images.append((
                                f"{seg.capitalize()} instance masks ({n_instances} objects)",
                                _image_to_data_url(colored),
                            ))

                    # Confidence scores — compute average
                    if os.path.exists(scores_path):
                        with rasterio.open(scores_path) as src:
                            scores_data = src.read(1).astype(float)
                        valid = scores_data[scores_data > 0]
                        if len(valid) > 0:
                            avg_conf = float(valid.mean())

                    seg_detail_lines.append(
                        f"- **{seg}**: {n_instances} instances, {n_pixels:,} pixels, "
                        f"avg confidence {avg_conf:.2f}"
                    )

                # Cache in session state
                st.session_state.vision_images = vision_images
                st.session_state.seg_detail = "\n".join(seg_detail_lines)

            # Use cached images for all subsequent renders
            vision_images = st.session_state.get("vision_images", [])
            seg_detail = st.session_state.get("seg_detail", "")

            # ── System prompt with full context ─────────────────────
            # Build detection context for AI
            _det_lines = []
            if rfdetr_results:
                for _dk, _dr in rfdetr_results.items():
                    if _dr.error or _dr.count == 0:
                        continue
                    from collections import Counter as _Counter
                    _cls_counts = _Counter(_dr.classes)
                    _cls_str = ", ".join(f"{c}: {n}" for c, n in _cls_counts.most_common())
                    _det_lines.append(f"- RF-DETR {_dr.label}: {_dr.count} objects ({_cls_str})")
            if detection_results:
                for _dk, _dr in detection_results.items():
                    if hasattr(_dr, 'error') and _dr.error:
                        continue
                    _det_lines.append(f"- GeoDeep {_dr.label}: {_dr.count} objects")
            _det_context = ""
            if _det_lines:
                _det_context = "Object detection results:\n" + "\n".join(_det_lines) + "\n\n"

            _SYSTEM_PROMPT = (
                "You are a geospatial analyst. You have been given multiple satellite "
                "images of a specific area including the original imagery, a combined "
                "segmentation overlay, and individual per-segment binary overlays and "
                "instance masks.\n\n"
                f"Detected segments: {', '.join(segments)}\n\n"
                f"Area measurements:\n{area_context}\n\n"
                f"Per-segment instance counts:\n{seg_detail}\n\n"
                f"{_det_context}"
                f"Bounding box (EPSG:4326): {st.session_state.bbox}\n\n"
                "IMAGE LEGEND (in order):\n"
            )
            for i, (label, _) in enumerate(vision_images):
                _SYSTEM_PROMPT += f"  Image {i+1}: {label}\n"
            _SYSTEM_PROMPT += (
                "\nRULES:\n"
                "- ONLY answer questions about these specific images and this area.\n"
                "- If asked about anything unrelated, respond: "
                "'I can only analyze the satellite imagery shown here.'\n"
                "- Do NOT answer general knowledge questions, write code, or discuss "
                "topics outside geospatial analysis of this specific area.\n"
                "- Be concise and insightful. Reference what you see in the images.\n"
                "- Use the instance counts and area measurements when answering "
                "questions about quantities (e.g. 'how many buildings?').\n"
                "- Distinguish between segments — each has its own mask and stats."
            )

            # ── Build the initial user message with all images ──────
            def _build_image_message(prompt_text: str) -> dict:
                content = [{"type": "input_text", "text": prompt_text}]
                for label, url in vision_images:
                    content.append({"type": "input_image", "image_url": url, "detail": "high"})
                return {"role": "user", "content": content}

            # ── Initial analysis ────────────────────────────────────
            if not st.session_state.vision_analysis_done:
                st.caption(f"{len(vision_images)} images will be sent to the AI model for analysis.")
                if st.button("🤖 Run AI Analysis", type="secondary"):
                    client = OpenAI(api_key=cfg.get_secret("OPENAI_API_KEY"))
                    with st.spinner(f"Analysing {len(vision_images)} images..."):
                        try:
                            response = client.responses.create(
                                model=cfg.get("openai_model", "gpt-5.4-nano"),
                                input=[
                                    {"role": "system", "content": _SYSTEM_PROMPT},
                                    _build_image_message(
                                        "Analyze all the provided images. For each segment, "
                                        "report: how many instances were found, total area, "
                                        "and whether the segmentation looks accurate. "
                                        "Describe notable spatial patterns."
                                    ),
                                ],
                            )
                            st.session_state.vision_chat = [
                                {"role": "assistant", "content": response.output_text},
                            ]
                            st.session_state.vision_analysis_done = True
                            st.rerun()
                        except Exception as e:
                            st.warning("Vision analysis failed. Check your API key in Settings.")

            # ── Chat interface ──────────────────────────────────────
            if st.session_state.vision_analysis_done:
                for msg in st.session_state.vision_chat:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                user_input = st.chat_input("Ask about this imagery...")
                if user_input:
                    st.session_state.vision_chat.append({"role": "user", "content": user_input})
                    with st.chat_message("user"):
                        st.markdown(user_input)

                    client = OpenAI(api_key=cfg.get_secret("OPENAI_API_KEY"))

                    # First message always includes all images
                    api_messages = [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        _build_image_message("Here are all the segmentation images for reference."),
                    ]
                    for msg in st.session_state.vision_chat:
                        api_messages.append({"role": msg["role"], "content": msg["content"]})

                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            try:
                                response = client.responses.create(
                                    model=cfg.get("openai_model", "gpt-5.4-nano"),
                                    input=api_messages,
                                )
                                reply = response.output_text
                                st.markdown(reply)
                                st.session_state.vision_chat.append({"role": "assistant", "content": reply})
                            except Exception as e:
                                st.warning("Chat request failed. Check your API key and try again.")

                # Reset button
                if st.button("Clear Chat", type="secondary"):
                    st.session_state.vision_chat = []
                    st.session_state.vision_analysis_done = False
                    st.rerun()
        else:
            st.info("Set your OpenAI API key in the **Settings** tab to enable AI vision analysis.")

        # ── Download results ────────────────────────────────────────────

        st.subheader("Export")
        st.caption("GeoJSON files can be imported into QGIS, Google Earth, kepler.gl, or pasted into a GitHub Gist (auto-renders a map).")

        # Build segment dirs mapping for export
        _seg_dirs = {}
        for seg in segments:
            info = seg_results.get(seg, {})
            _seg_dirs[seg] = info.get("out_dir", os.path.join(out_dir, seg))

        export_cols = st.columns(4)
        with export_cols[0]:
            csv_data = df.to_csv(index=False)
            st.download_button("📄 CSV Summary", csv_data, "area_summary.csv", "text/csv", use_container_width=True)
        with export_cols[1]:
            overlay_pil = Image.fromarray(overlay)
            buf = io.BytesIO()
            overlay_pil.save(buf, format="PNG")
            st.download_button("🖼️ Overlay PNG", buf.getvalue(), "overlay.png", "image/png", use_container_width=True)
        with export_cols[2]:
            from vectorizer import export_geojson
            geojson_data = export_geojson(_seg_dirs, bbox=list(st.session_state.bbox))
            # Merge RF-DETR detection features into the GeoJSON export
            import json as _json
            _fc = _json.loads(geojson_data)
            if rfdetr_results:
                for _rf_result in rfdetr_results.values():
                    if _rf_result.count > 0:
                        _rf_geojson = rfdetr_to_geojson(_rf_result, st.session_state.image_path)
                        _rf_fc = _json.loads(_rf_geojson)
                        _fc["features"].extend(_rf_fc.get("features", []))
            # Merge legacy GeoDeep detection features
            if detection_results:
                for _det_result in detection_results.values():
                    _fc["features"].extend(detection_result_to_geojson_features(_det_result))
            _fc["properties"]["total_features"] = len(_fc["features"])
            geojson_data = _json.dumps(_fc, indent=2)
            st.download_button(
                "🌐 All Segments GeoJSON",
                geojson_data,
                "llmsat_segments.geojson",
                "application/geo+json",
                use_container_width=True,
            )
        with export_cols[3]:
            # Per-segment GeoJSON downloads in an expander
            from vectorizer import export_per_segment_geojson
            with st.popover("📦 Per-Segment GeoJSON"):
                for seg in segments:
                    seg_dir = _seg_dirs.get(seg, "")
                    seg_geojson = export_per_segment_geojson(seg, seg_dir)
                    st.download_button(
                        f"{seg.capitalize()}",
                        seg_geojson,
                        f"llmsat_{seg}.geojson",
                        "application/geo+json",
                        key=f"export_{seg}",
                        use_container_width=True,
                    )

    # =====================================================================
    # TRAINING TAB
    # =====================================================================

with tab_training:
    st.title("🎓 Train Custom Detector")
    st.caption(
        "Fine-tune RF-DETR on your own satellite annotations. "
        "Segment an area in the Main tab, review detections here, then train."
    )

    from pipeline.dataset_builder import (
        tile_geotiff, instance_masks_to_bboxes, rfdetr_bboxes_to_annotations,
        clip_annotations_to_tile, create_dataset, list_datasets,
        load_dataset_metadata, dataset_dir as get_dataset_dir,
    )
    from pipeline.annotator import (
        merge_detections, render_annotated_tile, annotations_to_summary,
    )
    from pipeline.trainer import (
        check_training_deps, train_rfdetr, register_custom_model,
        load_custom_models, list_all_detection_models,
    )

    # ── Step 1: Build Dataset ─────────────────────────────────────────

    st.header("Step 1: Build Dataset from Segmentation Results")

    # Check if we have segmentation results
    _has_seg = st.session_state.get("segmentation_done", False)
    _has_rfdetr = bool(st.session_state.get("rfdetr_results", {}))
    _image_path = st.session_state.get("image_path")

    if not _has_seg and not _has_rfdetr:
        st.info(
            "No segmentation results yet. Go to the **Main** tab, select an area, "
            "run segmentation and/or RF-DETR detection, then come back here."
        )
    else:
        st.success("Segmentation results available. You can build a training dataset.")

        # Tiling controls
        _tile_cols = st.columns(3)
        with _tile_cols[0]:
            _tile_size = st.selectbox("Tile size (px)", [320, 512, 640], index=2)
        with _tile_cols[1]:
            _tile_overlap = st.slider("Tile overlap", 0.0, 0.4, 0.1, 0.05)
        with _tile_cols[2]:
            _project_name = st.text_input(
                "Dataset name",
                value=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M')}",
            )

        # Collect annotations from all sources
        if st.button("Generate Tiles & Annotations", type="primary"):
            from datetime import datetime

            _status = st.status("Building dataset...", expanded=True)
            with _status:
                # Tile the imagery
                st.write("Tiling satellite imagery...")
                tiles = tile_geotiff(_image_path, tile_size=_tile_size, overlap=_tile_overlap)
                st.write(f"  Generated **{len(tiles)}** tiles ({_tile_size}x{_tile_size} px)")

                # Collect global annotations from all sources
                global_anns = []

                # From SAM3 instance masks
                seg_results = st.session_state.get("seg_results", {})
                for seg_name, seg_info in seg_results.items():
                    seg_dir = seg_info.get("out_dir", "")
                    mask_path = os.path.join(seg_dir, f"sam3_{seg_name}_masks.tif")
                    if os.path.exists(mask_path) and seg_info.get("has_data"):
                        anns = instance_masks_to_bboxes(mask_path, class_name=seg_name)
                        global_anns.extend(anns)
                        st.write(f"  SAM3 **{seg_name}**: {len(anns)} annotations")

                # From RF-DETR detections
                rfdetr_res = st.session_state.get("rfdetr_results", {})
                for rf_key, result in rfdetr_res.items():
                    if result.count > 0 and not result.error:
                        anns = rfdetr_bboxes_to_annotations(
                            result.bboxes, result.scores, result.classes,
                        )
                        global_anns.extend(anns)
                        st.write(f"  RF-DETR **{result.label}**: {len(anns)} annotations")

                # Merge and deduplicate
                st.write("Merging annotations...")
                merged = merge_detections(
                    sam3_annotations=[a for a in global_anns if a.get("source") == "sam3_mask"],
                    rfdetr_annotations=[a for a in global_anns if a.get("source") == "rfdetr"],
                )
                st.write(f"  **{len(merged)}** annotations after deduplication")

                # Clip annotations to each tile
                st.write("Assigning annotations to tiles...")
                tile_anns = []
                total_tile_anns = 0
                for tile in tiles:
                    clipped = clip_annotations_to_tile(
                        merged,
                        tile["x_off"], tile["y_off"],
                        _tile_size, _tile_size,
                    )
                    tile_anns.append(clipped)
                    total_tile_anns += len(clipped)
                st.write(f"  **{total_tile_anns}** tile-level annotations")

                # Determine class names
                class_names = sorted(set(a["class"] for a in merged))
                st.write(f"  Classes: {', '.join(class_names)}")

                # Estimate GSD
                _gsd = 0.0
                if _image_path:
                    with rasterio.open(_image_path) as src:
                        _gsd = abs(src.transform[0]) * 111320 * 100  # deg → cm

                # Store for review
                st.session_state._train_tiles = tiles
                st.session_state._train_tile_anns = tile_anns
                st.session_state._train_class_names = class_names
                st.session_state._train_merged = merged
                st.session_state._train_project_name = _project_name
                st.session_state._train_gsd = _gsd

                _status.update(label=f"Ready: {len(tiles)} tiles, {len(merged)} annotations", state="complete")

        # ── Annotation review ─────────────────────────────────────────
        if "_train_tiles" in st.session_state:
            tiles = st.session_state._train_tiles
            tile_anns = st.session_state._train_tile_anns
            class_names = st.session_state._train_class_names

            st.subheader("Review & Edit Annotations")
            st.caption(
                "Browse all tiles. Accept/reject each annotation, change classes, "
                "or add new bounding boxes manually."
            )

            # ── Summary metrics ───────────────────────────────────
            # Recount from tile_anns (reflects edits)
            _all_anns_flat = [a for ta in tile_anns for a in ta]
            _n_accepted = sum(1 for a in _all_anns_flat if a.get("accepted", True))
            _n_rejected = sum(1 for a in _all_anns_flat if not a.get("accepted", True))

            sum_cols = st.columns(5)
            with sum_cols[0]:
                st.metric("Tiles", len(tiles))
            with sum_cols[1]:
                st.metric("Total annotations", len(_all_anns_flat))
            with sum_cols[2]:
                st.metric("Accepted", _n_accepted)
            with sum_cols[3]:
                st.metric("Rejected", _n_rejected)
            with sum_cols[4]:
                st.metric("GSD", f"{st.session_state._train_gsd:.1f} cm/px")

            # Class breakdown
            _cls_counts = {}
            for a in _all_anns_flat:
                if a.get("accepted", True):
                    _cls_counts[a["class"]] = _cls_counts.get(a["class"], 0) + 1
            if _cls_counts:
                cls_df = pd.DataFrame([
                    {"class": k, "count": v} for k, v in _cls_counts.items()
                ])
                st.bar_chart(cls_df.set_index("class"))

            # ── Tile browser ──────────────────────────────────────
            st.subheader("Tile Browser")

            # Tile navigation
            _n_tiles = len(tiles)
            _tiles_per_page = 4
            _n_pages = max(1, (_n_tiles + _tiles_per_page - 1) // _tiles_per_page)

            nav_cols = st.columns([1, 3, 1])
            with nav_cols[1]:
                _page = st.slider(
                    "Page", 1, _n_pages, 1,
                    key="tile_page",
                    help=f"{_n_tiles} tiles, {_tiles_per_page} per page",
                )
            _start = (_page - 1) * _tiles_per_page
            _end = min(_start + _tiles_per_page, _n_tiles)

            # Show tiles for this page
            page_cols = st.columns(min(_tiles_per_page, _end - _start))
            for col_idx, tile_idx in enumerate(range(_start, _end)):
                with page_cols[col_idx]:
                    _tile = tiles[tile_idx]
                    _anns = tile_anns[tile_idx]
                    rendered = render_annotated_tile(
                        _tile["rgb"], _anns, class_names, show_rejected=True,
                    )
                    _n_acc = sum(1 for a in _anns if a.get("accepted", True))
                    st.image(
                        rendered,
                        caption=f"Tile {tile_idx} — {_n_acc}/{len(_anns)} accepted",
                        width="stretch",
                    )

            # ── Per-tile annotation editor ────────────────────────
            st.subheader("Edit Annotations")
            _edit_tile = st.selectbox(
                "Select tile to edit",
                range(_n_tiles),
                format_func=lambda i: f"Tile {i} ({len(tile_anns[i])} annotations)",
                key="edit_tile_idx",
            )

            _tile = tiles[_edit_tile]
            _anns = tile_anns[_edit_tile]

            # Show tile with current annotations
            edit_cols = st.columns([3, 2])
            with edit_cols[0]:
                rendered = render_annotated_tile(
                    _tile["rgb"], _anns, class_names, show_rejected=True,
                )
                st.image(rendered, caption=f"Tile {_edit_tile}", width="stretch")

            with edit_cols[1]:
                st.markdown(f"**{len(_anns)} annotations**")

                if not _anns:
                    st.info("No annotations on this tile.")
                else:
                    for ann_idx, ann in enumerate(_anns):
                        bx, by, bw, bh = ann["bbox"]
                        _key_prefix = f"t{_edit_tile}_a{ann_idx}"

                        with st.container(border=True):
                            # Row 1: accept toggle + class selector
                            r1c1, r1c2 = st.columns([1, 2])
                            with r1c1:
                                _accepted = st.checkbox(
                                    "Keep",
                                    value=ann.get("accepted", True),
                                    key=f"{_key_prefix}_acc",
                                )
                                ann["accepted"] = _accepted
                            with r1c2:
                                _cls_idx = class_names.index(ann["class"]) if ann["class"] in class_names else 0
                                _new_cls = st.selectbox(
                                    "Class",
                                    class_names,
                                    index=_cls_idx,
                                    key=f"{_key_prefix}_cls",
                                    label_visibility="collapsed",
                                )
                                ann["class"] = _new_cls

                            # Row 2: bbox coordinates (editable)
                            r2c1, r2c2, r2c3, r2c4 = st.columns(4)
                            with r2c1:
                                ann["bbox"][0] = st.number_input("x", value=float(bx), step=1.0, key=f"{_key_prefix}_x", label_visibility="collapsed")
                            with r2c2:
                                ann["bbox"][1] = st.number_input("y", value=float(by), step=1.0, key=f"{_key_prefix}_y", label_visibility="collapsed")
                            with r2c3:
                                ann["bbox"][2] = st.number_input("w", value=float(bw), step=1.0, min_value=1.0, key=f"{_key_prefix}_w", label_visibility="collapsed")
                            with r2c4:
                                ann["bbox"][3] = st.number_input("h", value=float(bh), step=1.0, min_value=1.0, key=f"{_key_prefix}_h", label_visibility="collapsed")

                            st.caption(f"source: {ann.get('source', '?')} | score: {ann.get('score', 0):.2f}")

                # ── Add new annotation ────────────────────────────
                st.markdown("---")
                st.markdown("**Add annotation**")
                add_cols = st.columns([2, 1, 1, 1, 1, 1])
                with add_cols[0]:
                    _add_cls = st.selectbox("Class", class_names, key=f"add_cls_{_edit_tile}")
                with add_cols[1]:
                    _add_x = st.number_input("x", value=10.0, step=1.0, key=f"add_x_{_edit_tile}")
                with add_cols[2]:
                    _add_y = st.number_input("y", value=10.0, step=1.0, key=f"add_y_{_edit_tile}")
                with add_cols[3]:
                    _add_w = st.number_input("w", value=40.0, step=1.0, min_value=1.0, key=f"add_w_{_edit_tile}")
                with add_cols[4]:
                    _add_h = st.number_input("h", value=40.0, step=1.0, min_value=1.0, key=f"add_h_{_edit_tile}")
                with add_cols[5]:
                    if st.button("Add", key=f"add_btn_{_edit_tile}", use_container_width=True):
                        tile_anns[_edit_tile].append({
                            "bbox": [_add_x, _add_y, _add_w, _add_h],
                            "class": _add_cls,
                            "score": 1.0,
                            "source": "manual",
                            "accepted": True,
                            "id": len(_anns) + 1,
                        })
                        # Also add to class_names if new
                        if _add_cls not in class_names:
                            class_names.append(_add_cls)
                            st.session_state._train_class_names = class_names
                        st.rerun()

            # ── Bulk actions ──────────────────────────────────────
            st.markdown("---")
            bulk_cols = st.columns(4)
            with bulk_cols[0]:
                if st.button("Accept all on this tile", key="bulk_accept"):
                    for a in tile_anns[_edit_tile]:
                        a["accepted"] = True
                    st.rerun()
            with bulk_cols[1]:
                if st.button("Reject all on this tile", key="bulk_reject"):
                    for a in tile_anns[_edit_tile]:
                        a["accepted"] = False
                    st.rerun()
            with bulk_cols[2]:
                if st.button("Delete rejected", key="bulk_delete"):
                    tile_anns[_edit_tile] = [a for a in tile_anns[_edit_tile] if a.get("accepted", True)]
                    st.rerun()
            with bulk_cols[3]:
                _new_class = st.text_input("Add class", key="new_class_input", placeholder="e.g. solar_panel")
                if _new_class and _new_class not in class_names:
                    class_names.append(_new_class)
                    st.session_state._train_class_names = class_names

            # ── Save dataset button ───────────────────────────────
            st.markdown("---")
            # Filter out rejected annotations before saving
            _save_tile_anns = [
                [a for a in ta if a.get("accepted", True)]
                for ta in tile_anns
            ]
            _total_save = sum(len(ta) for ta in _save_tile_anns)
            st.caption(f"Will save **{_total_save}** accepted annotations across **{len(tiles)}** tiles.")

            if st.button("💾 Save Dataset", type="primary"):
                _proj = st.session_state._train_project_name
                _bbox = st.session_state.get("bbox")
                dataset_path = create_dataset(
                    tiles=tiles,
                    tile_annotations=_save_tile_anns,
                    class_names=class_names,
                    project_name=_proj,
                    gsd_cm=st.session_state._train_gsd,
                    source_bbox=_bbox,
                )
                st.success(f"Dataset saved to `{dataset_path}`")
                st.rerun()

    # ── Existing datasets ─────────────────────────────────────────────

    st.markdown("---")
    st.subheader("Existing Datasets")
    _datasets = list_datasets()
    if _datasets:
        for ds in _datasets:
            with st.expander(f"**{ds['project_name']}** — {ds['n_train']}+{ds['n_val']} images, {ds.get('n_annotations_train', 0)}+{ds.get('n_annotations_val', 0)} annotations"):
                ds_cols = st.columns(4)
                with ds_cols[0]:
                    st.caption(f"Classes: {', '.join(ds.get('class_names', []))}")
                with ds_cols[1]:
                    st.caption(f"GSD: {ds.get('gsd_cm', 0):.1f} cm/px")
                with ds_cols[2]:
                    st.caption(f"Created: {ds.get('created', 'unknown')[:10]}")
    else:
        st.info("No datasets yet. Build one from segmentation results above.")

    # =====================================================================
    # Step 2-3: Configure & Train
    # =====================================================================

    st.header("Step 2: Train")

    _deps_ok, _deps_msg = check_training_deps()
    if not _deps_ok:
        st.warning(f"Training prerequisites: {_deps_msg}")
    else:
        st.success(_deps_msg)

    if _datasets:
        train_cols = st.columns(3)
        with train_cols[0]:
            _ds_names = [ds["project_name"] for ds in _datasets]
            _sel_dataset = st.selectbox("Dataset", _ds_names)
        with train_cols[1]:
            _base_model = st.selectbox("Base model", ["rfdetr_base", "rfdetr_large"], index=0)
        with train_cols[2]:
            _epochs = st.number_input("Epochs", 10, 500, 50, 10)

        adv_cols = st.columns(3)
        with adv_cols[0]:
            _batch = st.number_input("Batch size", 1, 32, 4)
        with adv_cols[1]:
            _grad_accum = st.number_input("Grad accumulation", 1, 16, 4)
        with adv_cols[2]:
            _lr = st.number_input("Learning rate", 1e-6, 1e-2, 1e-4, format="%.1e")

        _early_stop = st.checkbox("Early stopping", value=True)

        if st.button("🚀 Start Training", type="primary", disabled=not _deps_ok):
            _ds_path = get_dataset_dir(_sel_dataset)
            _out_dir = os.path.join("output", "training", _sel_dataset)

            status = st.status("Training RF-DETR...", expanded=True)
            with status:
                st.write(f"Dataset: **{_sel_dataset}**")
                st.write(f"Base model: **{_base_model}** | Epochs: **{_epochs}** | Batch: **{_batch}×{_grad_accum}**")

                result = train_rfdetr(
                    dataset_dir=_ds_path,
                    output_dir=_out_dir,
                    base_model=_base_model,
                    epochs=_epochs,
                    batch_size=_batch,
                    grad_accum_steps=_grad_accum,
                    lr=_lr,
                    early_stopping=_early_stop,
                )

                if result.get("error"):
                    st.error(f"Training failed: {result['error']}")
                    status.update(label="Training failed", state="error")
                else:
                    st.write(f"✅ Checkpoint saved: `{result['checkpoint_path']}`")
                    if result.get("best_map50"):
                        st.write(f"Best mAP@50: **{result['best_map50']:.3f}**")
                    st.write(f"Elapsed: {result['elapsed_seconds']:.0f}s")

                    # Auto-register
                    _ds_meta = load_dataset_metadata(_sel_dataset)
                    _cls_names = _ds_meta.get("class_names", []) if _ds_meta else []
                    register_custom_model(
                        name=_sel_dataset,
                        checkpoint_path=result["checkpoint_path"],
                        class_names=_cls_names,
                        base_model=_base_model,
                        gsd_cm=_ds_meta.get("gsd_cm", 0) if _ds_meta else 0,
                    )
                    st.success(f"Model **{_sel_dataset}** registered! It now appears in the detection dropdown.")
                    status.update(label="Training complete!", state="complete")

    # =====================================================================
    # Step 4: Manage Custom Models
    # =====================================================================

    st.header("Step 3: Custom Models")

    _custom = load_custom_models()
    if _custom:
        for name, info in _custom.items():
            with st.expander(f"🎓 **{name}** — {info.get('description', '')}"):
                st.caption(f"Checkpoint: `{info.get('checkpoint_path', 'unknown')}`")
                st.caption(f"Base: {info.get('base_model', 'unknown')} | Classes: {', '.join(info.get('class_names', []))}")
                st.caption(f"GSD: {info.get('gsd_cm', 0):.1f} cm/px")
                st.info("This model appears in the Main tab → Object Detection dropdown.")
    else:
        st.info("No custom models yet. Train one above and it will appear here.")

    # ── Footer ──────────────────────────────────────────────────────────

    st.markdown("---")
    st.markdown("Powered by SAM3, RF-DETR, DINOv3, and OpenAI Vision. Built by Edwin Kestler.")
