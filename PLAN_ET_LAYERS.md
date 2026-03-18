# Plan: ET Monitor Data Layers Integration

## Goal

Add multi-layer geospatial data visualization to LLMSat by integrating satellite products and weather station data from the ET Monitor project. Users can toggle layers (AETI, NDVI, precipitation, soil moisture, temperature, weather stations) on top of the existing SAM3 segmentation results.

## Data Sources

| Layer | Product | Resolution | Source Path | Colormap | Units |
|-------|---------|-----------|-------------|----------|-------|
| Actual ET | FAO WaPOR AETI | 300m | `data/raw/wapor/aeti/` | YlGnBu | mm/month |
| NDVI | MODIS | 1km | `data/raw/ndvi/` | YlGn | 0-1 |
| Precipitation | CHIRPS | 5.5km | `data/raw/chirps/` | Blues | mm/month |
| Surface Soil Moisture | NASA SMAP | 9km | `data/raw/smap_surface/` | PuBu | m3/m3 |
| Root Zone Soil Moisture | NASA SMAP | 9km | `data/raw/smap_rootzone/` | PuBu | m3/m3 |
| Temperature | ERA5 | 9km | `data/raw/era5/` | RdYlBu_r | C |
| Weather Stations | Weather Underground | point | `et_monitor.db` | markers | multi |

All rasters are EPSG:4326 GeoTIFFs, monthly temporal resolution. Station data is daily/hourly in SQLite.

---

## Architecture

```
ET Monitor (read-only)                    LLMSat
/mnt/e/.../weatherunderground2csv/        /home/kestl/.../app_llmsat/
├── data/raw/wapor/aeti/*.tif    ←──┐
├── data/raw/ndvi/*.tif          ←──┤    et_layers/
├── data/raw/chirps/*.tif        ←──┤    ├── __init__.py
├── data/raw/smap_surface/*.tif  ←──┼──→ ├── config.py        (paths + layer defs)
├── data/raw/smap_rootzone/*.tif ←──┤    ├── raster_reader.py (load, crop, resample)
├── data/raw/era5/*.tif          ←──┤    ├── station_reader.py(SQLite queries)
└── et_monitor.db                ←──┘    └── overlay.py       (blend, colorbar, markers)
                                                    │
                                              app.py (Step 3)
```

**Key principle:** Never import from ET Monitor. Build a self-contained data access layer that reads files using a configurable root path (`ET_MONITOR_ROOT` env var).

---

## Implementation Phases

### Phase 1: Data Access Layer (`et_layers/` package)

**1.1 `et_layers/config.py`**
- `LAYER_DEFS` dict with layer metadata (subdir, colormap, vmin, vmax, units, label)
- Read `ET_MONITOR_ROOT` from `.env`, construct base paths
- Handle path split: WaPOR products under `data/raw/wapor/`, GEE products under `data/raw/`

**1.2 `et_layers/raster_reader.py`**
- `list_available_months(product)` → sorted list of YYYYMM strings
- `load_layer_for_bbox(product, yyyymm, bbox)` → `(ndarray, transform, meta)` or None
  - Uses `rasterio.windows.from_bounds()` to read only the bbox window
  - Returns None if bbox doesn't overlap raster extent
- `resample_to_target(data, src_transform, target_h, target_w, target_transform)` → ndarray
  - Uses `rasterio.warp.reproject()` with bilinear interpolation
- `apply_colormap(data, layer_key, opacity)` → RGBA uint8 array
  - Normalizes to vmin/vmax, applies matplotlib colormap, sets alpha

**1.3 `et_layers/station_reader.py`**
- Direct `sqlite3.connect()` with read-only mode
- `get_stations()` → DataFrame with stationID, lat, lon
- `get_latest_observations()` → DataFrame with latest tempAvg, humidityAvg, precipTotal, solarRadiationHigh per station
- `get_station_monthly(station_id, month)` → DataFrame

**1.4 `et_layers/overlay.py`**
- `blend_layer_on_rgb(rgb, rgba_layer)` → RGB with layer composited
- `make_colorbar(layer_key, width, height)` → PIL Image for legend display
- `make_folium_image_overlay(data, bounds, layer_key, opacity)` → folium.ImageOverlay
- `make_station_markers(stations_df)` → list of folium.Marker

### Phase 2: UI Integration (`app.py`)

**2.1 Session state** (new keys)
- `et_layers_enabled`: `{layer_key: bool}` — all False initially
- `et_opacity`: `{layer_key: float}` — all 0.6 initially
- `et_month`: selected YYYYMM string
- `et_stations_visible`: bool

**2.2 Step 1: Station markers on Folium map**
- Checkbox "Show Weather Stations" near the map
- When enabled, add circle markers for 14 WU stations with popup stats
- Different color from bbox selection rectangle

**2.3 Step 3: "Data Layers" panel** (insert after Segment Details, before Area Summary)
- Guard: check `ET_MONITOR_ROOT` is set, show info message if not
- Month selector: `st.select_slider` with available YYYYMM values
- Layer toggles: `st.columns(4)` with `st.toggle()` + `st.slider(opacity)` per layer
- For each enabled layer:
  1. `load_layer_for_bbox()` — crop to current bbox
  2. `resample_to_target()` — match satellite image dimensions
  3. `apply_colormap()` — colormapped RGBA
  4. `blend_layer_on_rgb()` — composite onto satellite RGB
  5. `st.image()` — display + colorbar legend
- WU station toggle: draw markers as circles on the composite image

**2.4 Enrich AI Vision Analysis**
- When layers are active, append layer context to the GPT prompt
- Example: "CHIRPS precipitation for this area in January 2020: 180mm"

**2.5 Extend Export**
- Download composited image with data layers as PNG
- Optional: export cropped raster values as CSV

### Phase 3: Polish

**3.1** Folium raster overlays in Step 1 map (interactive layer toggle)
**3.2** Error handling: bbox outside Guatemala, missing months, missing env var
**3.3** Performance: cache in session_state, `@st.cache_data` for stations, lazy-load
**3.4** Tests: synthetic GeoTIFFs, in-memory SQLite, overlay shape validation

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Cross-project path dependency** | High | Use `ET_MONITOR_ROOT` env var; entire panel hidden if not set |
| **Resolution mismatch** (0.6m vs 9km) | Medium | `rasterio.warp.reproject` with bilinear; coarse layers will appear as smooth gradients, which is correct for 9km data |
| **WSL2 I/O performance** (reading from /mnt/e/) | Medium | Cache loaded data in session_state; windowed reads to minimize I/O |
| **Bbox outside Guatemala** | Low | Check raster bounds before reading; warn if no overlap |
| **Missing months for some products** | Low | Union of available months; grey out products with no data for selected month |
| **Large composite images** | Low | All operations on cropped windows, not full rasters; 1km bbox = small arrays |
| **ET Monitor DB schema changes** | Low | Use explicit column names in queries, not `SELECT *` |

---

## Files Summary

### New files (5)
| File | Purpose |
|------|---------|
| `et_layers/__init__.py` | Package init, public API exports |
| `et_layers/config.py` | Layer definitions, path config from env |
| `et_layers/raster_reader.py` | GeoTIFF load, crop, resample, colormap |
| `et_layers/station_reader.py` | SQLite read-only queries for WU stations |
| `et_layers/overlay.py` | Image blending, colorbars, Folium helpers |

### Modified files (3)
| File | Changes |
|------|---------|
| `app.py` | Step 1 station markers, Step 3 data layers panel, enriched AI analysis, extended export |
| `.env.example` | Add `ET_MONITOR_ROOT` and `ET_MONITOR_DB` |
| `requirements.txt` | No new deps needed (already have rasterio, folium, matplotlib, pandas) |

### No changes to
- `segmenter.py`, `vectorizer.py`, `downloader.py`, `open_buildings.py`, `osm_roads.py`, `nl_query/`, `pipeline/`

---

## Estimated Effort

| Phase | Scope | Complexity |
|-------|-------|------------|
| Phase 1: Data access layer | 4 new files, ~300 lines | Medium — rasterio windowed reads + resampling |
| Phase 2: UI integration | 1 file modified, ~150 lines added | Medium — Streamlit layout + session state |
| Phase 3: Polish | Error handling + caching + tests | Low |

---

## Decision Points for Review

1. **Should layers be visible in Step 1 (Folium map) or only Step 3 (static image)?** — Plan includes both but Step 1 is lower priority.
2. **Should the month selector be global (one month for all layers) or per-layer?** — Plan uses global for simplicity. Per-layer adds complexity.
3. **Should we copy ET Monitor data into LLMSat's directory or always read from the external path?** — Plan reads from external path to avoid data duplication (2+ GB of GeoTIFFs).
4. **ERA5 has 6 bands — which band to show by default?** — Plan shows temperature (band 1). Could add a band selector.
