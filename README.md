# LLMSat - Geospatial Segmentation & Analysis

Satellite imagery analysis tool that combines **SAM3** (Segment Anything 3), **Google Open Buildings**, and **OpenAI Vision** to segment, classify, and analyze geospatial features from satellite imagery.

## Features

- **Interactive 3-step workflow** — select area on a map, pick segments, analyze results.
- **SAM3 segmentation** — text-prompted segmentation ("tree", "building", "water") using Meta's SAM3 model.
- **Google Open Buildings integration** — 1.8B building footprints used to generate precise building masks without SAM3.
- **OpenStreetMap roads** — road network fetched and rasterized directly from OSM.
- **AI vision chat** — multi-image analysis with GPT, ask follow-up questions about your segmentation results.
- **GeoJSON export** — export segmented polygons for use in QGIS, Google Earth, GitHub Gists, kepler.gl.
- **Progressive UI** — each step shows results as they arrive.

---

## Installation (Step by Step)

### Prerequisites

Before you start, make sure you have:

- **Python 3.12 or newer** — check with `python3 --version`
- **NVIDIA GPU with CUDA** — SAM3 needs a GPU with at least 8 GB VRAM (24 GB recommended)
- **NVIDIA drivers + CUDA toolkit** — check with `nvidia-smi`
- **GDAL system library** — needed for geospatial file handling
- **Git** — to clone the repository

### Step 1: Install system dependencies

**Ubuntu / Debian / WSL2:**
```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libgdal-dev gdal-bin git
```

**macOS (Homebrew):**
```bash
brew install python gdal git
```

### Step 2: Clone the repository

```bash
git clone https://github.com/EdwinKestler/app_llmsat.git
cd app_llmsat
```

### Step 3: Create a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows PowerShell
```

You should see `(.venv)` in your terminal prompt.

### Step 4: Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install PyTorch, SAM3, Streamlit, and all other dependencies. The download is large (~5 GB for PyTorch with CUDA). Be patient.

If GDAL fails to install, make sure the system library matches:
```bash
gdal-config --version          # e.g. 3.10.2
pip install GDAL==3.10.2       # must match system version
```

### Step 5: Get API keys

You need two API keys. Both can be set later via the app's **Settings** tab, or manually:

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```
OPENAI_API_KEY=sk-your-openai-key-here
HF_TOKEN=hf_your-huggingface-token-here
```

**Where to get them:**

| Key | Where to get it | What it's for |
|-----|-----------------|---------------|
| `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | AI vision analysis and NL query parsing |
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Downloading the SAM3 model |

### Step 6: Request access to SAM3 (required)

SAM3 is a **gated model** — you must accept Meta's license before the first run:

1. Go to [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
2. Log in with your HuggingFace account
3. Click **"Agree and access repository"**
4. Wait for approval (usually instant)

### Step 7: Download Google Open Buildings data (optional but recommended)

This dramatically improves building detection. Download the tiles for your region from [Google Open Buildings](https://sites.research.google/gr/open-buildings/).

For Guatemala / Central America, download these files and place them in the `openbuildings/` folder:

```bash
mkdir -p openbuildings
# Download from https://sites.research.google/gr/open-buildings/
# Place *_buildings.csv.gz files in openbuildings/
```

| File | Size | Covers |
|------|------|--------|
| `859_buildings.csv.gz` | 1.0 GB | Western Guatemala, southern Mexico |
| `8f7_buildings.csv.gz` | 1.2 GB | Eastern Guatemala, Honduras |
| `8f5_buildings.csv.gz` | 300 MB | Yucatan, Belize |
| `85f_buildings.csv.gz` | 572 MB | Central Mexico |

The `tiles.geojson` file (already included) tells the app which tile covers your area.

### Step 8: Run the app

```bash
streamlit run app.py
```

Open your browser to **http://localhost:8501**. The first run will download the SAM3 model (~850 MB) — this only happens once.

---

## How to Use the App

### Step 1: Select an area and load imagery

1. Click a **preset location** (Guatemala City, Antigua, Tikal, Lake Atitlan) or draw a rectangle on the map.
2. Adjust the **zoom level** (18 = ~0.6 m/pixel, good for buildings; 16 = ~2.4 m/pixel, good for large areas).
3. Click **Load Imagery** — satellite tiles will download and a preview appears.

### Step 2: Choose what to segment

1. Check the segment types you want: **Tree**, **Building**, **Water**, **Road**.
2. Optionally select an extra term from the dropdown (car, solar panel, swimming pool, etc.).
3. Customize overlay colors by expanding the color picker.
4. If Open Buildings data is available, you'll see a **preview** of known building footprints.
5. Click **Run Segmentation** — watch the progress per segment.

### Step 3: Explore results

1. **Original vs Overlay** — side-by-side comparison with color legend.
2. **Segment Details** — expand each segment to see binary overlay, instance masks, and confidence heatmap.
3. **Area Summary** — table and bar chart with area per segment in square metres.
4. **AI Vision Chat** — click **Run AI Analysis** for an automated report, then ask follow-up questions like:
   - "How many buildings are there?"
   - "Which area has the most tree coverage?"
   - "Does the road segmentation look accurate?"
5. **Export** — download results as:
   - **CSV** — area summary table
   - **PNG** — overlay image
   - **GeoJSON** — all segment polygons (open in QGIS, paste into a GitHub Gist for a map)
   - **Per-segment GeoJSON** — individual files per segment type

### Settings tab

Click the **Settings** tab at the top to:
- Enter or change your API keys (stored securely in `.env`, never in git)
- Change the OpenAI model
- Adjust SAM3 confidence and mask thresholds
- Set data source paths (Open Buildings directory, Overpass API URL)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'osgeo'` | Install GDAL: `sudo apt install libgdal-dev gdal-bin` then `pip install GDAL==$(gdal-config --version)` |
| `'BertModel' object has no attribute 'get_head_mask'` | Run `pip install "transformers<5"` |
| `401 Client Error` when loading SAM3 | Accept the license at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) and check your `HF_TOKEN` |
| `No objects detected for 'building'` | Install Open Buildings data in `openbuildings/` — SAM3 text prompts alone often miss buildings |
| App is slow on first run | Normal — SAM3 model (~850 MB) downloads once and is cached in `~/.cache/huggingface/` |
| `CUDA out of memory` | Reduce zoom level (fewer pixels) or use a smaller bounding box |
| Overpass API timeout (504) | Try again in 30 seconds — the public API has rate limits |

---

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
│   ├── segments.gpkg               # Vectorised polygons (GeoPackage)
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
    ├─── "building" ──→ Open Buildings → rasterize polygons (no SAM3)
    ├─── "road" ──────→ OSM Overpass API → buffer + rasterize (no SAM3)
    ├─── "tree" ──────→ SAM3 text prompt → generate_masks()
    │
    ▼
run_text_segmentation()     → langsam_mask.tif + instances + scores
    │
    ▼
raster_to_vector()          → segments.gpkg + summary.csv
    │
    ▼
export_geojson()            → llmsat_segments.geojson
    │
    ▼
GPT-5.4-nano Vision         → multi-image chat analysis
```

## Configuration

Settings are managed through two files:

- **`config.json`** — all non-secret settings. Editable via the Settings tab.
- **`.env`** — secrets only (API keys). Also editable via the Settings tab.

Neither file is committed to git.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `default_bbox` | Guatemala City 1km | Default bounding box |
| `default_zoom` | `18` | Tile zoom level (~0.6 m/pixel) |
| `output_dir` | `output` | Output directory |
| `confidence_threshold` | `0.5` | SAM3 confidence threshold |
| `mask_threshold` | `0.5` | SAM3 mask threshold |
| `openai_model` | `gpt-5.4-nano` | OpenAI model for vision analysis |
| `open_buildings_dir` | `openbuildings` | Path to Open Buildings data tiles |
| `osm_overpass_url` | Overpass API | URL for OSM road queries |

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
| Export | GeoJSON, GeoPackage, CSV, PNG |

## Roadmap

### Gradio + Modal.com Migration
Cloud deployment with serverless GPU. See [PLAN_GRADIO_MODAL.md](PLAN_GRADIO_MODAL.md).

### ET Monitor Data Layers
Multi-layer environmental data overlay. See [PLAN_ET_LAYERS.md](PLAN_ET_LAYERS.md).

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

You are free to:
- **Use** — for academic, research, educational, and personal projects
- **Study** — read and learn from the source code
- **Modify** — adapt the code for your own needs
- **Distribute** — share copies with others

Under these conditions:
- **Share alike** — modifications must be released under AGPL-3.0
- **Disclose source** — network deployment requires source code access for users
- **Attribution** — give credit and indicate changes

**For commercial licensing**, contact the author.

See [LICENSE](LICENSE) for the full text.

## Author

Developed by Edwin Kestler for precision agriculture, urban monitoring, and environmental analysis.
