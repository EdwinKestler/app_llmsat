# LLMSat - Geospatial Segmentation & Analysis

Satellite imagery analysis tool that combines **SAM3** (Segment Anything 3), **Google Open Buildings**, and **OpenAI Vision** to segment, classify, and analyze geospatial features from satellite imagery.

## Features

- **Interactive 3-step workflow** — select area on a map, pick segments, analyze results.
- **SAM3 segmentation** — text-prompted segmentation ("tree", "building", "water") using Meta's SAM3 model.
- **Google Open Buildings integration** — 1.8B building footprints used as box prompts for SAM3, dramatically improving building detection.
- **AI vision analysis** — GPT-4o-mini analyzes the satellite imagery and segmentation overlay to provide natural language insights.
- **Vectorisation** — converts raster masks to GeoPackage polygons with UTM-projected area summaries.
- **Progressive UI** — each step shows results as they arrive instead of waiting behind a single spinner.

## Requirements

- Python 3.12+
- CUDA-compatible GPU with 8+ GB VRAM (24 GB recommended)
- PyTorch 2.7+
- HuggingFace access to `facebook/sam3` (gated model)

## Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and HF_TOKEN
```

### HuggingFace access

SAM3 is a gated model. You must:

1. Create a HuggingFace account and generate an access token.
2. Go to [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) and accept the license.
3. Add `HF_TOKEN=hf_...` to your `.env` file.

### Google Open Buildings (optional)

Download building tiles for your region from [Google Open Buildings](https://sites.research.google/gr/open-buildings/) and place the `*_buildings.csv.gz` files in `openbuildings/`. The app will automatically use them as box prompts for building segmentation.

For Central America / Guatemala:
- `859_buildings.csv.gz` (1.0 GB) — covers western Guatemala
- `8f7_buildings.csv.gz` (1.2 GB) — covers eastern Guatemala
- `tiles.geojson` — tile geometry index
- `score_thresholds_s2_level_4.csv` — confidence thresholds

## Usage

### Streamlit UI (recommended)

```bash
streamlit run app.py
```

The app follows a three-step workflow:

#### Step 1: Select Area & Load Imagery
- Draw a rectangle on the interactive Leaflet map or enter coordinates manually.
- Click **Load Imagery** to download satellite tiles for the selected area.
- Preview the satellite image and area dimensions.

#### Step 2: Select Segments & Run SAM3
- Pick segment types via chips: Tree, Building, Water, Road.
- Choose additional SAM3-compatible terms from a curated dropdown (car, solar panel, swimming pool, crop, etc.).
- Customize overlay colors per segment via the color picker expander.
- If Open Buildings data is available, a preview shows known building footprints overlaid on the satellite image.
- If OSM road data is available, a preview shows the road network before segmentation.
- Click **Run Segmentation** to generate masks with progressive status updates per segment.

#### Step 3: Results & Analysis
- Side-by-side satellite image and segmentation overlay with color legend.
- Expandable per-segment detail: binary overlay, instance masks, confidence heatmap.
- Area summary table and bar chart.
- **AI Vision Chat** — all per-segment images (binary overlays, instance masks, confidence scores) are sent to the AI model with instance counts and area data. Ask follow-up questions in a chat interface with guardrails that keep responses focused on the imagery.
- Export results as CSV or overlay PNG.

#### Settings Tab
- Configure API keys (OpenAI, HuggingFace) via password fields.
- Adjust SAM3 thresholds, OpenAI model, data source paths.
- All settings persisted to `config.json`. Secrets stored in `.env`.

### CLI

```bash
python cli.py --bbox -90.52 14.63 -90.50 14.64 --prompt "trees" --prompt "buildings"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--bbox` | *(required)* | West South East North in EPSG:4326 |
| `--out-dir` | `output` | Directory for all outputs |
| `--prompt` | *(repeatable)* | Text prompt(s) for SAM3 segmentation |

## Output Files

Each segment gets its own directory under `output/`:

```
output/
├── _imagery/
│   └── s2harm_rgb_saa.tif          # Downloaded satellite imagery
├── tree/
│   ├── s2harm_rgb_saa.tif          # Copy of imagery
│   ├── langsam_mask.tif            # Combined binary mask
│   ├── sam3_tree_masks.tif         # Per-instance unique masks
│   ├── sam3_tree_scores.tif        # Confidence scores per mask
│   ├── auto_mask.tif               # General auto-segmentation mask
│   ├── segments.gpkg               # Vectorised polygons
│   └── summary.csv                 # Area summary
└── building/
    └── ...                          # Same structure
```

## Architecture

```
User selects bbox on map
    │
    ▼
download_imagery()          TMS tiles → GeoTIFF
    │
    ├─── "building" prompt ──→ Open Buildings query (DuckDB)
    │                              → 2,400+ building bboxes
    │                              → sam3.generate_masks_by_boxes()
    │
    ├─── "tree" prompt ─────→ sam3.generate_masks(prompt="tree")
    │
    ▼
run_text_segmentation()     → langsam_mask.tif (binary)
                            → sam3_*_masks.tif (instances)
                            → sam3_*_scores.tif (confidence)
    │
    ▼
raster_to_vector()          → segments.gpkg + summary.csv
    │
    ▼
GPT-5.4-nano Vision         → multi-image chat analysis
    │
    ▼
Streamlit UI                → overlays, charts, export
```

## Configuration

Settings are managed through two files:

- **`config.json`** — all non-secret settings (thresholds, paths, defaults). Editable via the Settings tab in the app.
- **`.env`** — secrets only (API keys, tokens). Also editable via the Settings tab.

Neither file is committed to git.

### `config.json` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `default_bbox` | Guatemala City 1km | Default bounding box |
| `default_zoom` | `18` | Tile zoom level (~0.6 m/pixel) |
| `output_dir` | `output` | Output directory |
| `confidence_threshold` | `0.5` | SAM3 confidence threshold |
| `mask_threshold` | `0.5` | SAM3 mask threshold |
| `openai_model` | `gpt-5.4-nano` | OpenAI model for vision analysis |
| `open_buildings_dir` | `openbuildings` | Path to Open Buildings data |
| `open_buildings_min_confidence` | `0.65` | Min confidence for building footprints |
| `osm_overpass_url` | Overpass API | URL for OSM road queries |
| `et_monitor_root` | *(empty)* | Path to ET Monitor project (planned) |

### `.env` secrets

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for NL parsing and vision analysis |
| `HF_TOKEN` | HuggingFace token for SAM3 gated model access |

## Technology Stack

| Component | Technology |
|-----------|------------|
| Segmentation | SAM3 (Meta) via segment-geospatial |
| Building data | Google Open Buildings v3 (1.8B footprints) |
| Road data | OpenStreetMap via Overpass API |
| Vision analysis | OpenAI GPT-5.4-nano |
| Satellite tiles | Esri World Imagery via TMS |
| Spatial queries | DuckDB, GeoPandas, Rasterio |
| UI | Streamlit + Folium |

## Planned: ET Monitor Data Layers

Multi-layer environmental data overlay using satellite products from the companion [ET Monitor](https://github.com/EdwinKestler/weatherunderground2csv) project. See [PLAN_ET_LAYERS.md](PLAN_ET_LAYERS.md) for the full implementation plan.

| Layer | Source | Resolution |
|-------|--------|-----------|
| Evapotranspiration (AETI) | FAO WaPOR v3 | 300m |
| Vegetation Index (NDVI) | MODIS | 1km |
| Precipitation (CHIRPS) | UCSB CHC | 5.5km |
| Soil Moisture (SMAP) | NASA | 9km |
| Temperature (ERA5) | ECMWF | 9km |
| Weather Stations | Weather Underground | point (14 stations) |

## Author

Developed by Edwin Kestler for precision agriculture, urban monitoring, and environmental analysis.
