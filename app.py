"""LLMSat – Streamlit UI with three-step workflow + settings.

Tab 1 (Main):
  Step 1: Select area on interactive map  →  Load satellite imagery
  Step 2: Pick segment prompts via chips   →  Run SAM3 segmentation
  Step 3: Analyze results with AI vision   →  Display overlays, areas, charts

Tab 2 (Settings):
  API keys, model config, thresholds, paths — persisted to config.json
"""

import shutil
import streamlit as st
import os
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

import config_manager as cfg
from downloader import download_imagery
from segmenter import run_text_segmentation, run_auto_segmentation
from vectorizer import raster_to_vector, summarise
from nl_query.openai_handler import fetch_segment_data
from pipeline.config import PipelineConfig

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

# ── Page config ─────────────────────────────────────────────────────

st.set_page_config(page_title="LLMSat", layout="wide")

tab_main, tab_settings = st.tabs(["Main", "Settings"])

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
        # Save secrets to .env
        if oai_key:
            cfg.set_secret("OPENAI_API_KEY", oai_key)
        if hf_token:
            cfg.set_secret("HF_TOKEN", hf_token)

        # Save config to config.json
        settings_cfg.update({
            "openai_model": openai_model,
            "confidence_threshold": conf_thresh,
            "mask_threshold": mask_thresh,
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
        st.markdown(
            "**Data sources:**\n"
            "- SAM3 (Meta)\n"
            "- Google Open Buildings\n"
            "- OpenStreetMap\n"
            "- OpenAI Vision"
        )

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
                if len(manual_bbox) == 4 and manual_bbox != st.session_state.bbox:
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
        bbox = st.session_state.bbox
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

        if not selected:
            st.warning("Select at least one segment type.")
        else:
            seg_col1, seg_col2 = st.columns([1, 3])
            with seg_col1:
                segment_btn = st.button("🔍 Run Segmentation", type="primary", width="stretch")

            if segment_btn:
                _shutil = shutil  # noqa: local alias

                bbox = st.session_state.bbox
                image_path = st.session_state.image_path
                seg_results = {}

                status = st.status("Running SAM3 segmentation...", expanded=True)
                with status:
                    for seg in selected:
                        st.write(f"Segmenting **{seg}**...")

                        seg_out_dir = os.path.join(out_dir, seg)

                        # Clear stale outputs for this segment so nothing is reused
                        if os.path.isdir(seg_out_dir):
                            _shutil.rmtree(seg_out_dir)
                        os.makedirs(seg_out_dir, exist_ok=True)

                        # Copy current imagery into segment dir
                        seg_image = os.path.join(seg_out_dir, "s2harm_rgb_saa.tif")
                        _shutil.copy2(image_path, seg_image)

                        config = PipelineConfig(
                            bbox=bbox, zoom=zoom, out_dir=seg_out_dir,
                            box_threshold=_cfg.get("confidence_threshold", 0.5),
                            text_threshold=_cfg.get("mask_threshold", 0.5),
                        )

                        # Text-prompted segmentation
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
                has_data = mask.any()
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

            # 3. Per-segment: binary overlay + instance mask + stats
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
            else:
                seg_detail_lines = []

            # Use cached images for all subsequent renders
            vision_images = st.session_state.get("vision_images", [])
            seg_detail = st.session_state.get("seg_detail", "")

            # ── System prompt with full context ─────────────────────
            _SYSTEM_PROMPT = (
                "You are a geospatial analyst. You have been given multiple satellite "
                "images of a specific area including the original imagery, a combined "
                "segmentation overlay, and individual per-segment binary overlays and "
                "instance masks.\n\n"
                f"Detected segments: {', '.join(segments)}\n\n"
                f"Area measurements:\n{area_context}\n\n"
                f"Per-segment instance counts:\n{seg_detail}\n\n"
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
                            st.warning(f"Vision analysis failed: {e}")

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
                                st.warning(f"Chat failed: {e}")

                # Reset button
                if st.button("Clear Chat", type="secondary"):
                    st.session_state.vision_chat = []
                    st.session_state.vision_analysis_done = False
                    st.rerun()
        else:
            st.info("Set your OpenAI API key in the **Settings** tab to enable AI vision analysis.")

        # ── Download results ────────────────────────────────────────────

        st.subheader("Export")
        export_cols = st.columns(3)
        with export_cols[0]:
            csv_data = df.to_csv(index=False)
            st.download_button("📄 Download CSV", csv_data, "area_summary.csv", "text/csv", width="stretch")
        with export_cols[1]:
            overlay_pil = Image.fromarray(overlay)
            buf = io.BytesIO()
            overlay_pil.save(buf, format="PNG")
            st.download_button("🗺️ Download Overlay PNG", buf.getvalue(), "overlay.png", "image/png", width="stretch")

    # ── Footer ──────────────────────────────────────────────────────────
    
    st.markdown("---")
    st.markdown("Powered by SAM3, DINOv3, and OpenAI Vision. Built by Edwin Kestler.")
