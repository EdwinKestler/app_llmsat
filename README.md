# LLMSat - Geospatial Segmentation Query Tool

Automated satellite imagery analysis pipeline that downloads imagery, segments it using SAM2 and LangSAM, vectorises the results, and lets users query segments via natural language (powered by OpenAI).

## Features

- **Satellite imagery download** — fetches tiles for a bounding box via TMS and exports a GeoTIFF.
- **Semantic segmentation (LangSAM)** — finds objects by text prompt (trees, water, buildings, roads).
- **General segmentation (SAM2)** — detects all visually distinct regions in the image.
- **Vectorisation** — converts raster masks to GeoPackage polygons with area summaries.
- **Natural language queries** — ask questions like *"What is the total area of buildings and trees?"* and get a chart + table.
- **Two interfaces** — Streamlit web UI (`app.py`) and a CLI (`cli.py`).

## Requirements

- Python 3.11+
- SAM2 checkpoint file in the `checkpoints/` directory

## Setup

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure your environment variables
cp .env.example .env   # then add your OPENAI_API_KEY
```

## Usage

### Streamlit UI

```bash
streamlit run app.py
```

Configure bounding box, model paths, and ask natural-language questions from the sidebar.

### CLI

```bash
python cli.py --bbox -74.01 40.70 -73.99 40.72 --prompt "trees" --prompt "buildings"
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--bbox` | *(required)* | West South East North in EPSG:4326 |
| `--out-dir` | `output` | Directory for all outputs |
| `--prompt` | *(repeatable)* | Text prompt(s) for LangSAM |

## Outputs

| File | Description |
|------|-------------|
| `output/s2harm_rgb_saa.tif` | Downloaded satellite imagery |
| `output/langsam_mask.tif` | Combined LangSAM semantic mask |
| `output/sam2_mask.tif` | SAM2 general segmentation mask |
| `output/segments.gpkg` | Vectorised polygons (GeoPackage) |
| `output/summary.csv` | Area summary per polygon |

## Configuration

Pipeline settings can be passed programmatically via `PipelineConfig` or loaded from a YAML file using `load_config()`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bbox` | — | Bounding box (west, south, east, north) |
| `zoom` | `18` | Tile zoom level |
| `out_dir` | `output` | Output directory |
| `model_dir` | `checkpoints` | Directory containing model weights |
| `sam2_checkpoint` | `sam2_hiera_l.pt` | SAM2 checkpoint filename |
| `box_threshold` | `0.24` | LangSAM box detection threshold |
| `text_threshold` | `0.24` | LangSAM text detection threshold |

## Author

Developed by Edwin Kestler for precision agriculture, urban monitoring, and environmental analysis.
