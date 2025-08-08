import streamlit as st
from pipeline import load_config, run_pipeline

st.title("Geospatial Processing Pipeline")

with st.sidebar:
    st.header("Parameters")
    west = st.number_input("West", value=-74.01)
    south = st.number_input("South", value=40.70)
    east = st.number_input("East", value=-73.99)
    north = st.number_input("North", value=40.72)
    out_dir = st.text_input("Output directory", "output")
    prompt_text = st.text_area("Prompts (one per line)")
    run_button = st.button("Run pipeline")

if run_button:
    bbox = (west, south, east, north)
    prompts = [p.strip() for p in prompt_text.splitlines() if p.strip()]
    config = load_config(bbox=bbox, out_dir=out_dir)
    outputs = run_pipeline(config, prompts)

    st.success("Pipeline complete")
    st.image(outputs["image"], caption="Downloaded imagery")
    st.image(outputs["semantic_mask"], caption="LangSAM mask")
    st.image(outputs["sam2_mask"], caption="SAM2 mask")
    st.write("Vector data:", outputs["gpkg"])
    st.write("Summary CSV:", outputs["csv"])
