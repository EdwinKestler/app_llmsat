# app_stremlit/tests/test_pipeline.py

import os
from pathlib import Path
import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from pipeline.config import PipelineConfig
from pipeline.pipeline import run_pipeline
from vectorizer import summarise
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.mark.skip(reason="Provide a tiny bbox & mocked downloader in CI; enable locally for end-to-end.")
def test_pipeline_smoke(tmp_path):
    cfg = PipelineConfig(
        bbox=(-74.01, 40.70, -73.99, 40.72),
        out_dir=str(tmp_path / "out"),
        device="cpu",
    )
    out = run_pipeline(cfg, text_prompts=["water"])
    assert os.path.exists(out["image"])
    assert os.path.exists(out["sam2_mask"])
    assert os.path.exists(out["gpkg"])

def test_summarise_area_is_m2(tmp_path):
    # Small square polygon around the equator-ish; unit test focuses on units, not exact value
    poly = Polygon([(-0.00009, 0.0), (0.0, 0.0), (0.0, 0.00009), (-0.00009, 0.00009)])
    gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
    csv_path = tmp_path / "summary.csv"
    df = summarise(gdf, str(csv_path))
    assert "area_m2" in df.columns
    # magnitude check (def not degrees^2)
    assert df["area_m2"].iloc[0] > 1.0
    assert df["area_m2"].iloc[0] < 1e8
