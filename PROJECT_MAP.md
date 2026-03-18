# Project Map

## Directory Structure

```
app_llmsat/
├── app.py                  # Streamlit web UI entry point
├── cli.py                  # Command-line interface entry point
├── requirements.txt        # pip dependencies
├── .env                    # Environment variables (OPENAI_API_KEY) — not committed
│
├── pipeline/               # Core processing pipeline
│   ├── __init__.py         # Exports PipelineConfig, load_config, run_pipeline
│   ├── config.py           # PipelineConfig dataclass + YAML loader
│   └── pipeline.py         # run_pipeline() — orchestrates download → segment → vectorise
│
├── downloader.py           # download_imagery() — TMS tiles to GeoTIFF
├── segmenter.py            # run_langsam() and run_sam2() — raster mask generation
├── vectorizer.py           # raster_to_vector() and summarise() — mask to GeoPackage/CSV
│
├── nl_query/               # Natural language query layer
│   ├── __init__.py         # Exports ask()
│   └── openai_handler.py   # NL parsing (OpenAI or keyword fallback), segment lookup, charting
│
├── samgeo/                 # Local stubs for the samgeo package (used for testing)
│   ├── __init__.py         # Stub tms_to_geotiff() and SamGeo class
│   └── text_sam.py         # Stub LangSAM class
│
├── checkpoints/            # Model weights directory (not committed)
│
├── old_files/              # Archived experimental scripts (not part of active codebase)
└── EsriCache/              # Cached Esri imagery tiles (not committed)
```

## Data Flow

```
User Input (bbox + prompts)
        │
        ▼
┌─────────────────┐
│  download_imagery│  (downloader.py)
│  TMS → GeoTIFF  │
└────────┬────────┘
         │ image.tif
         ▼
┌─────────────────┐     ┌─────────────────┐
│   run_langsam   │     │    run_sam2      │
│  text → mask    │     │  auto-segment    │
└────────┬────────┘     └────────┬────────┘
         │ langsam_mask.tif      │ sam2_mask.tif
         │                       │
         └───────────┬───────────┘
                     ▼
          ┌─────────────────┐
          │ raster_to_vector │  (vectorizer.py)
          │ mask → polygons  │
          └────────┬────────┘
                   │ segments.gpkg + summary.csv
                   ▼
          ┌─────────────────┐
          │   NL Query /    │  (nl_query/openai_handler.py)
          │   Streamlit UI  │
          └─────────────────┘
```

## Entry Points

| Entry Point | Command | Description |
|-------------|---------|-------------|
| `app.py` | `streamlit run app.py` | Web UI with map config, NL questions, and charts |
| `cli.py` | `python cli.py --bbox ... --prompt ...` | Headless pipeline execution |
| `nl_query/openai_handler.py` | `python -m nl_query.openai_handler` | Standalone NL query CLI |

## Key Classes and Functions

| Symbol | Location | Purpose |
|--------|----------|---------|
| `PipelineConfig` | `pipeline/config.py` | Dataclass holding all pipeline parameters |
| `load_config()` | `pipeline/config.py` | Build config from YAML file and/or overrides |
| `run_pipeline()` | `pipeline/pipeline.py` | End-to-end: download → segment → vectorise |
| `download_imagery()` | `downloader.py` | Fetch TMS tiles as a single GeoTIFF |
| `run_langsam()` | `segmenter.py` | Text-guided segmentation via LangSAM |
| `run_sam2()` | `segmenter.py` | Automatic segmentation via SAM2 |
| `raster_to_vector()` | `vectorizer.py` | Binary mask raster → GeoPackage polygons |
| `summarise()` | `vectorizer.py` | Compute area stats and write CSV |
| `ask()` | `nl_query/openai_handler.py` | Parse question, run pipeline if needed, return chart + dataframe |
| `parse_user_text()` | `nl_query/openai_handler.py` | Extract segment keywords via OpenAI or regex fallback |
