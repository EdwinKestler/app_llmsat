# Project Map

## Directory Structure

```
app_llmsat/
├── app.py                  # Streamlit web UI — 3-step workflow
├── cli.py                  # Command-line interface
├── requirements.txt        # pip dependencies
├── .env                    # API keys: OPENAI_API_KEY, HF_TOKEN (not committed)
├── .env.example            # Template for .env
│
├── pipeline/               # Core processing pipeline
│   ├── __init__.py         # Lazy exports: PipelineConfig, load_config, run_pipeline
│   ├── config.py           # PipelineConfig dataclass + YAML loader
│   └── pipeline.py         # run_pipeline() — download → segment → vectorise
│
├── downloader.py           # download_imagery() — TMS tiles to GeoTIFF
├── segmenter.py            # SAM3 segmentation — text, box, and auto modes
├── vectorizer.py           # raster_to_vector() and summarise()
├── open_buildings.py       # Google Open Buildings query via DuckDB
├── osm_roads.py            # OpenStreetMap road query via Overpass API
│
├── et_layers/              # ET Monitor data integration (PLANNED)
│   ├── __init__.py         # Package exports
│   ├── config.py           # Layer definitions, ET_MONITOR_ROOT path
│   ├── raster_reader.py    # GeoTIFF load, crop, resample, colormap
│   ├── station_reader.py   # SQLite queries for WU weather stations
│   └── overlay.py          # Image blending, colorbars, Folium helpers
│
├── nl_query/               # Natural language query layer
│   ├── __init__.py         # Exports ask()
│   └── openai_handler.py   # NL parsing, segment lookup, area calculation
│
├── openbuildings/          # Google Open Buildings data tiles (not committed)
│   ├── *_buildings.csv.gz  # S2 level 4 building polygon tiles
│   ├── tiles.geojson       # Tile geometry index
│   └── score_thresholds_s2_level_4.csv
│
├── tests/                  # Test suite (pytest)
│   ├── conftest.py         # Prepends stubs/ to sys.path
│   ├── stubs/samgeo/       # Lightweight samgeo stubs (no GPU needed)
│   ├── test_config.py
│   ├── test_downloader.py
│   ├── test_nl_query.py
│   ├── test_segmenter.py
│   └── test_vectorizer.py
│
├── checkpoints/            # Model weights directory (not committed)
├── output/                 # Pipeline output (not committed)
└── old_files/              # Archived experimental scripts
```

## Data Flow

```
                    ┌──────────────────────────────────┐
                    │  Step 1: Select Area              │
                    │  Interactive Leaflet map           │
                    │  → bbox coordinates                │
                    └──────────────┬───────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────┐
                    │  download_imagery()               │
                    │  TMS tiles → s2harm_rgb_saa.tif   │
                    └──────────────┬───────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │  Step 2: Segment              │
                    ├──────────────────────────────┤
                    │                              │
              "building"                    "tree" / other
                    │                              │
                    ▼                              ▼
        ┌───────────────────┐       ┌──────────────────────┐
        │ Open Buildings    │       │ SAM3 text prompt      │
        │ rasterize polygons│       │ generate_masks()      │
        │ (no SAM3 needed)  │       │ → masks, scores       │
        └─────────┬─────────┘       └──────────┬───────────┘
                  │                             │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │ langsam_mask.tif (binary)     │
                  │ sam3_*_masks.tif (instances)   │
                  │ sam3_*_scores.tif (confidence) │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │ raster_to_vector()            │
                  │ → segments.gpkg + summary.csv │
                  └──────────────┬───────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │  Step 3: Analyze          │
                    ├──────────────────────────┤
                    │ Segmentation overlays     │
                    │ Instance masks + scores   │
                    │ Area table + chart         │
                    │ GPT-5.4-nano vision analysis│
                    │ ET Monitor data layers (PLANNED) │
                    │ CSV / PNG export           │
                    └──────────────────────────┘
```

## Segmentation Strategy

| Prompt | Data Source | Method |
|--------|-----------|--------|
| `building` | Google Open Buildings | Direct polygon rasterization (no SAM3) |
| `road` | OpenStreetMap (Overpass API) | Linestring buffer → rasterization (no SAM3) |
| `tree` | None | Text-prompted SAM3 (`generate_masks(prompt=...)`) |
| `water` | None | Text-prompted SAM3 |
| custom | None | Text-prompted SAM3 |
| auto | Bounding box | Box-prompted SAM3 (full area) |

## Key Classes and Functions

| Symbol | Location | Purpose |
|--------|----------|---------|
| `PipelineConfig` | `pipeline/config.py` | Dataclass: bbox, zoom, out_dir, thresholds |
| `load_config()` | `pipeline/config.py` | Build config from YAML + overrides |
| `run_pipeline()` | `pipeline/pipeline.py` | End-to-end: download → segment → vectorise |
| `download_imagery()` | `downloader.py` | Fetch TMS tiles as GeoTIFF (caches if exists) |
| `run_text_segmentation()` | `segmenter.py` | Text/data-driven masks (Open Buildings for buildings, OSM for roads, SAM3 for others) |
| `run_auto_segmentation()` | `segmenter.py` | SAM3 auto-segment entire bbox |
| `_rasterize_open_buildings()` | `segmenter.py` | Burn Open Buildings polygons directly into mask |
| `_rasterize_osm_roads()` | `segmenter.py` | Burn OSM road polygons directly into mask |
| `query_buildings()` | `open_buildings.py` | DuckDB query of Open Buildings tiles for bbox |
| `query_roads()` | `osm_roads.py` | Overpass API query for OSM roads in bbox |
| `buffer_roads()` | `osm_roads.py` | Buffer linestrings to road-width polygons |
| `raster_to_vector()` | `vectorizer.py` | Binary mask → GeoPackage polygons |
| `summarise()` | `vectorizer.py` | Compute UTM-projected areas, write CSV |
| `ask()` | `nl_query/openai_handler.py` | Parse question → run pipeline → return chart + df |
| `parse_user_text()` | `nl_query/openai_handler.py` | Extract segment keywords (OpenAI or regex) |
| `fetch_segment_data()` | `nl_query/openai_handler.py` | Load GeoPackage, compute total area |
| `_vision_analysis()` | `app.py` | Send images to GPT-5.4-nano for analysis |

## Entry Points

| Entry Point | Command | Description |
|-------------|---------|-------------|
| `app.py` | `streamlit run app.py` | 3-step web UI with interactive map |
| `cli.py` | `python cli.py --bbox ... --prompt ...` | Headless pipeline execution |
| `nl_query/openai_handler.py` | `python -m nl_query.openai_handler` | Standalone NL query CLI |

## Model Cache Locations

| Model | Cache Path | Size |
|-------|-----------|------|
| SAM3 | `~/.cache/huggingface/hub/models--facebook--sam3/` | ~850 MB |
| GroundingDINO | `~/.cache/huggingface/hub/models--ShilongLiu--GroundingDINO/` | ~940 MB |
| BERT tokenizer | `~/.cache/huggingface/hub/models--bert-base-uncased/` | ~440 MB |
| SAM v1 (legacy) | `~/.cache/torch/hub/checkpoints/sam_vit_h_4b8939.pth` | 2.4 GB |

All models are downloaded once on first run and cached permanently.
