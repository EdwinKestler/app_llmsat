# app_stremlit/nl_query/openai_handler.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import geopandas as gpd
import pandas as pd

from pipeline.config import PipelineConfig, load_config
from pipeline.pipeline import run_pipeline

# ----------------------------
# Lightweight NL parsing
# ----------------------------
KNOWN_SEGMENTS = {
    "water": ["water", "river", "lake", "lagoon", "reservoir"],
    "urban": ["urban", "building", "settlement", "town", "city"],
    "vegetation": ["vegetation", "forest", "trees", "crop", "agriculture"],
}

def parse_user_text(text: str) -> List[str]:
    t = text.lower()
    found = []
    for key, vocab in KNOWN_SEGMENTS.items():
        if any(v in t for v in vocab):
            found.append(key)
    return list(dict.fromkeys(found))  # de-dup order


# ----------------------------
# File discovery aligned to pipeline outputs
# ----------------------------
def _segment_file(out_dir: str) -> Optional[Path]:
    """
    MODIFIED: pipeline writes a single 'segments.gpkg' per run/output folder.
    """
    p = Path(out_dir) / "segments.gpkg"
    return p if p.exists() else None


def fetch_segment_data(out_dir: str) -> Tuple[gpd.GeoDataFrame, float]:
    gpkg = _segment_file(out_dir)
    if gpkg is None:
        raise FileNotFoundError(f"No GeoPackage found in '{out_dir}'")
    gdf = gpd.read_file(gpkg)
    # Raw area here is unitless – you should prefer using vectorizer.summarise for m².
    area = gdf.geometry.area.sum()
    return gdf, float(area)


# ----------------------------
# Main entry for Streamlit
# ----------------------------
def ask(
    question: str,
    bbox: Iterable[float],
    *,
    out_dir: str = "data",
    use_altair: bool = True,
    device: str = "cuda",
    checkpoints_dir: str = "checkpoints",   # <— renamed, but we’ll still pass model_dir for legacy safety
    sam2_checkpoint: str = "sam2_hiera_l.pt",
    box_threshold: float = 0.24,
    text_threshold: float = 0.24,
    # legacy shim (won't be needed once app.py stops passing it)
    model_dir: Optional[str] = None,
):
    """
    MODIFIED: accept and propagate device/checkpoint/thresholds so CUDA is respected end-to-end.
    """
    segments = parse_user_text(question)
    if not segments:
        raise ValueError("No known segment types referenced in question")

    results = []
    for seg in segments:
        seg_out_dir = os.path.join(out_dir, seg)
        cfg: PipelineConfig = load_config(
            bbox=tuple(bbox),
            out_dir=seg_out_dir,
            device=device,
            checkpoints_dir=checkpoints_dir,
            sam2_checkpoint=sam2_checkpoint,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            # pass legacy param too in case someone else still uses it
            model_dir=model_dir,
        )
        if _segment_file(seg_out_dir) is None:
            run_pipeline(cfg, text_prompts=[seg])

        gdf, area = fetch_segment_data(seg_out_dir)
        results.append({"segment": seg, "area_units": "crs_units", "area": area})

    df = pd.DataFrame(results)

    if use_altair:
        import altair as alt

        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("segment:N", title="Segment"),
                y=alt.Y("area:Q", title="Area (crs units; see vectorizer.summarise for m²)"),
                tooltip=["segment", "area"]
            )
        )
        return chart, df

    return None, df
