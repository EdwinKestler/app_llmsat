import streamlit as st
import os
import altair as alt
import pandas as pd
from dotenv import load_dotenv

from nl_query.openai_handler import ask
from pipeline.config import PipelineConfig

# Load .env file
load_dotenv()

st.set_page_config(layout="wide")
st.title("🌍 Geospatial Segmentation Query Tool")

with st.sidebar:
    st.header("Configuration")
    out_dir = st.text_input("Output Directory", value="output")
    model_dir = st.text_input("Model Directory", value="checkpoints")
    sam2_checkpoint = st.text_input("SAM2 Checkpoint", value="sam2_hiera_l.pt")

st.subheader("Define Area and Query")
bbox_str = st.text_input(
    "Bounding Box (west, south, east, north in EPSG:4326)",
    value="-74.01, 40.70, -73.99, 40.72",
    help="Example: Manhattan area coordinates.",
)
question = st.text_area(
    "Natural Language Question",
    value="What is the total area of buildings and trees?",
    help="Ask about segments like water, trees, buildings, roads. E.g., 'How much area is covered by water and roads?'",
)

run_query = st.button("Run Query", type="primary")

if run_query:
    if not question:
        st.error("Please enter a question.")
    elif not bbox_str:
        st.error("Please enter a bounding box.")
    else:
        try:
            bbox = [float(coord.strip()) for coord in bbox_str.split(",")]
            if len(bbox) != 4:
                raise ValueError(
                    "Bounding box must have exactly 4 values: west, south, east, north."
                )

            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
            else:
                st.warning(
                    "No OpenAI API key found in .env file. Falling back to simple keyword parsing."
                )

            config = PipelineConfig(
                bbox=bbox,
                zoom=18,
                out_dir=out_dir,
                model_dir=model_dir,
                sam2_checkpoint=sam2_checkpoint,
                box_threshold=0.24,
                text_threshold=0.24,
            )

            with st.spinner(
                "Processing query... This may take a while if segmentation is needed."
            ):
                chart, df = ask(question, bbox, out_dir=out_dir)

            st.subheader("Results")
            st.dataframe(df.style.format({"area_m2": "{:.2f}"}))

            st.subheader("Visualization")
            st.altair_chart(
                chart.mark_bar()
                .encode(
                    x=alt.X("segment:N", title="Segment"),
                    y=alt.Y("area_m2:Q", title="Area (m²)"),
                    color="segment:N",
                )
                .properties(width=600, height=400),
                use_container_width=True,
            )

        except ValueError as ve:
            st.error(f"Invalid input: {str(ve)}")
        except FileNotFoundError as fe:
            st.error(
                f"Missing file: {str(fe)}. Ensure models are downloaded to the model directory."
            )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

st.markdown("---")
st.markdown(
    "Powered by SAM2, LangSAM, and OpenAI. Ensure required models are in the checkpoints directory."
)
