# Plan: Gradio + Modal.com Migration

## Goal

Migrate LLMSat from Streamlit (local-only, no GPU cloud) to **Gradio** (better UX, chat-native) deployed on **Modal.com** (serverless GPU, scale-to-zero, pay-per-second).

## Why Migrate

| Problem with Streamlit | How Gradio + Modal fixes it |
|---|---|
| Full page reruns on every interaction | Gradio only updates affected components |
| No GPU cloud deployment | Modal provides A100/H100 on demand |
| No built-in chat interface | `gr.ChatInterface` is native |
| Session state management is fragile | Gradio state is component-based |
| No API endpoint | Every Gradio app gets a free REST API |
| Streamlit Cloud has no GPU | Modal has serverless GPU with scale-to-zero |

## Architecture

```
Modal.com (serverless)
│
├── GPU Function: segment_image()
│   ├── SAM3 model (loaded once, stays warm)
│   ├── run_text_segmentation()
│   └── run_auto_segmentation()
│
├── CPU Function: query_data()
│   ├── Open Buildings (DuckDB)
│   ├── OSM Roads (Overpass API)
│   └── raster_to_vector()
│
├── CPU Function: analyze_vision()
│   └── OpenAI GPT-5.4-nano (multi-image chat)
│
└── Web Endpoint: Gradio UI
    ├── Tab 1: Map & Imagery
    ├── Tab 2: Segmentation
    ├── Tab 3: Results & Chat
    └── Tab 4: Settings
```

### Key Design Decisions

1. **SAM3 runs on GPU function** — model loaded in container startup, stays warm between requests. Scale to zero when idle.
2. **Data queries run on CPU** — DuckDB, Overpass API don't need GPU. Cheaper.
3. **Gradio serves the UI** — runs as a Modal web endpoint. No separate hosting needed.
4. **Open Buildings data stored in Modal Volume** — persistent storage, shared across function invocations.
5. **Config/secrets via Modal Secrets** — no .env files in cloud.

---

## Module Mapping (Streamlit → Gradio)

### Modules that stay the same (framework-agnostic)
| Module | Changes needed |
|--------|---------------|
| `config_manager.py` | Minor: read Modal Secrets when deployed |
| `segmenter.py` | None |
| `open_buildings.py` | None |
| `osm_roads.py` | None |
| `vectorizer.py` | None |
| `downloader.py` | None |
| `pipeline/config.py` | None |
| `pipeline/pipeline.py` | None |
| `nl_query/openai_handler.py` | None |

### Modules that change
| Module | What changes |
|--------|-------------|
| `app.py` → `gradio_app.py` | Complete rewrite: Streamlit → Gradio |
| NEW: `modal_app.py` | Modal deployment config: functions, volumes, secrets |

---

## Gradio UI Layout

```python
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌍 LLMSat – Geospatial Segmentation")

    with gr.Tab("1. Area & Imagery"):
        with gr.Row():
            map_html = gr.HTML()           # Folium map (rendered as HTML)
            preview_img = gr.Image()       # Satellite preview
        with gr.Row():
            bbox_input = gr.Textbox()      # Bbox coordinates
            zoom_dropdown = gr.Dropdown()  # Zoom level
            load_btn = gr.Button("📡 Load Imagery")
        preset_btns = gr.Radio(["Guatemala City", "Antigua", "Tikal", "Lake Atitlan"])

    with gr.Tab("2. Segment"):
        with gr.Row():
            seg_checks = gr.CheckboxGroup(["tree", "building", "water", "road"])
            extra_dropdown = gr.Dropdown(SAM3_EXTRA_TERMS)
        with gr.Row():
            ob_preview = gr.Image()        # Open Buildings overlay
            osm_preview = gr.Image()       # OSM roads overlay
        segment_btn = gr.Button("🔍 Run Segmentation")
        progress = gr.Textbox()            # Progressive status

    with gr.Tab("3. Results"):
        with gr.Row():
            original_img = gr.Image()
            overlay_img = gr.Image()
        gallery = gr.Gallery()             # Per-segment masks
        area_table = gr.Dataframe()
        area_chart = gr.Plot()

        # AI Chat (native Gradio component)
        chatbot = gr.Chatbot()
        chat_input = gr.Textbox(placeholder="Ask about this imagery...")
        chat_btn = gr.Button("Send")

    with gr.Tab("4. Export"):
        geojson_file = gr.File(label="All Segments GeoJSON")
        csv_file = gr.File(label="Area Summary CSV")
        png_file = gr.File(label="Overlay PNG")
        per_seg_files = gr.File(label="Per-Segment GeoJSON", file_count="multiple")

    with gr.Tab("⚙️ Settings"):
        oai_key = gr.Textbox(label="OpenAI API Key", type="password")
        hf_token = gr.Textbox(label="HuggingFace Token", type="password")
        # ... thresholds, paths ...
        save_btn = gr.Button("Save")
```

---

## Modal Deployment

### `modal_app.py`

```python
import modal

app = modal.App("llmsat")

# Persistent storage for Open Buildings tiles
volume = modal.Volume.from_name("llmsat-data", create_if_missing=True)

# Docker image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgdal-dev", "gdal-bin")
    .pip_install(
        "segment-geospatial[samgeo3]",
        "transformers<5",
        "rasterio", "geopandas", "duckdb",
        "openai", "gradio", "folium",
    )
)

# GPU function for segmentation (A100, stays warm 5 min)
@app.function(
    image=image,
    gpu="A100",
    timeout=600,
    container_idle_timeout=300,  # Stay warm 5 min
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("llmsat-secrets")],
)
def segment_image(image_bytes, prompts, bbox, config):
    from segmenter import run_text_segmentation, run_auto_segmentation
    # ... process and return results ...

# CPU function for data queries
@app.function(
    image=image,
    volumes={"/data": volume},
)
def query_data(bbox, query_type):
    # Open Buildings, OSM Roads, vectorization
    ...

# CPU function for AI vision analysis
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("llmsat-secrets")],
)
def analyze_vision(images, segments, area_data, user_message):
    # OpenAI multi-image chat
    ...

# Gradio web endpoint
@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("llmsat-secrets")],
    allow_concurrent_inputs=10,
)
@modal.asgi_app()
def web():
    from gradio_app import create_app
    demo = create_app()
    return demo
```

### Modal Secrets Setup

```bash
modal secret create llmsat-secrets \
    OPENAI_API_KEY=sk-... \
    HF_TOKEN=hf_...
```

### Modal Volume Setup (Open Buildings data)

```bash
# Upload Open Buildings tiles to Modal volume
modal volume put llmsat-data openbuildings/ /openbuildings/
```

---

## Migration Phases

### Phase 1: Gradio UI (local, no Modal)
- Create `gradio_app.py` with same 3-step flow
- Reuse all existing modules unchanged
- Test locally: `python gradio_app.py`
- **Goal:** Feature parity with Streamlit version

### Phase 2: Modal Integration
- Create `modal_app.py` with GPU/CPU function split
- Move SAM3 to GPU function (A100)
- Move data queries to CPU function
- Upload Open Buildings data to Modal Volume
- Set up Modal Secrets
- **Goal:** Deploy to Modal, accessible via URL

### Phase 3: Optimize for Cloud
- Pre-warm SAM3 model in container startup
- Cache satellite tiles in Modal Volume
- Add concurrent request handling
- Add usage tracking / rate limiting
- **Goal:** Production-ready deployment

### Phase 4: API Endpoint
- Gradio automatically provides REST API
- Document API for programmatic access
- CI/CD: auto-deploy on git push
- **Goal:** API + UI accessible from anywhere

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Folium map in Gradio | Medium | Render as HTML, embed via `gr.HTML()` |
| Modal cold start (SAM3 load) | Medium | `container_idle_timeout=300` keeps warm |
| Open Buildings data size (3 GB) | Low | Modal Volume persists data across invocations |
| Gradio chat vs Streamlit chat | Low | `gr.ChatInterface` is more capable |
| Cost control | Medium | Scale-to-zero + budget alerts in Modal |
| File upload/download UX | Low | `gr.File` component handles this natively |

---

## Cost Estimate (Modal.com)

| Resource | Usage | Cost |
|----------|-------|------|
| A100 GPU (segmentation) | ~30s per query | ~$0.03 per query |
| CPU (data queries) | ~10s per query | ~$0.001 per query |
| CPU (Gradio UI) | Always on during use | ~$0.01/hour |
| Storage (Volume) | 3 GB Open Buildings | ~$0.15/month |
| **Total per query** | | **~$0.03** |
| **Monthly (100 queries/day)** | | **~$90/month** |

$30 free credits = ~1,000 segmentation queries to start.

---

## Files Summary

### New files
| File | Purpose |
|------|---------|
| `gradio_app.py` | Gradio UI (replaces app.py) |
| `modal_app.py` | Modal deployment config |

### Unchanged files
| File | Why unchanged |
|------|--------------|
| `segmenter.py` | Framework-agnostic |
| `open_buildings.py` | Framework-agnostic |
| `osm_roads.py` | Framework-agnostic |
| `vectorizer.py` | Framework-agnostic |
| `downloader.py` | Framework-agnostic |
| `config_manager.py` | Minor env var changes only |
| `pipeline/*` | Framework-agnostic |
| `nl_query/*` | Framework-agnostic |

### Kept but deprecated
| File | Status |
|------|--------|
| `app.py` | Kept for local/offline use, not deployed to Modal |
