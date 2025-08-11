# app_stremlit/app.py
import os
import streamlit as st
import altair as alt
import pandas as pd
import torch
from dotenv import load_dotenv

from pipeline.config import PipelineConfig, load_config
from pipeline.pipeline import run_pipeline
from nl_query.openai_handler import ask

load_dotenv()
st.set_page_config(layout="wide")
st.title("🌍 Geospatial Segmentation Query Tool")

# Sidebar controls
st.sidebar.header("Configuration")
out_dir = st.sidebar.text_input("Output directory", value="data")
device = st.sidebar.selectbox("Device", options=["cuda", "cpu"], index=0 if torch.cuda.is_available() else 1)
checkpoints_dir = st.sidebar.text_input("Checkpoints directory", value="checkpoints")  # renamed for clarity
sam2_checkpoint = st.sidebar.text_input("SAM2 checkpoint filename", value="sam2_hiera_l.pt")
box_threshold = st.sidebar.slider("Box threshold", min_value=0.0, max_value=1.0, value=0.24, step=0.01)
text_threshold = st.sidebar.slider("Text threshold", min_value=0.0, max_value=1.0, value=0.24, step=0.01)

st.sidebar.markdown("---")
st.sidebar.header("ROI (EPSG:4326)")
col1, col2 = st.sidebar.columns(2)
with col1:
    west = float(st.number_input("West (xmin, λ min)", value=-90.6))
    east = float(st.number_input("East (xmax, λ max)", value=-90.5))
with col2:
    south = float(st.number_input("South (ymin, φ min)", value=14.58))
    north = float(st.number_input("North (ymax, φ max)", value=14.66))

# --- NEW (~lines 55–70): normalize & coerce to list for samgeo ---
xmin = min(west, east)
xmax = max(west, east)
ymin = min(south, north)
ymax = max(south, north)
bbox_list = [xmin, ymin, xmax, ymax]  # samgeo requires list, not tuple

# CUDA status
if device == "cuda":
    st.info("CUDA selected.")
    if not torch.cuda.is_available():
        st.warning("CUDA was selected but torch.cuda.is_available() == False. Falling back to CPU at runtime.")

st.markdown("---")

tab1, tab2 = st.tabs(["🔎 Natural Language Query", "🧪 Run Pipeline (Manual)"])

with tab1:
    question = st.text_input("Ask for segments (e.g., 'water bodies and urban areas')", value="water and urban")
    use_altair = st.checkbox("Show Altair chart", value=True)
    if st.button("Run NL Query"):
        try:
            chart, df = ask(
                question,
                bbox_list,                 # <- pass list
                out_dir=out_dir,
                use_altair=use_altair,
                device=device,
                checkpoints_dir=checkpoints_dir,
                sam2_checkpoint=sam2_checkpoint,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            if chart is not None:
                st.altair_chart(chart, use_container_width=True)
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error: {e}")

with tab2:
    segments_raw = st.text_input("Text prompts (comma-separated)", value="water")
    segments = [s.strip() for s in segments_raw.split(",") if s.strip()]
    if st.button("Run Manual Pipeline"):
        try:
            cfg = load_config(
                bbox=bbox_list,             # <- pass list
                out_dir=out_dir,
                device=device,
                checkpoints_dir=checkpoints_dir,
                sam2_checkpoint=sam2_checkpoint,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            run_pipeline(cfg, text_prompts=segments)
            st.success("Pipeline run completed.")
            st.write(f"Output directory: `{out_dir}`")
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.caption("Powered by SAM2, LangSAM, and OpenAI. Ensure required models are in the checkpoints directory.")
