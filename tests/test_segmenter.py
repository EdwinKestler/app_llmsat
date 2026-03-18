"""Tests for segmenter module."""

import os
import tempfile
import pytest
from pipeline.config import PipelineConfig
from downloader import download_imagery
from segmenter import run_langsam, run_sam2


@pytest.fixture
def setup_pipeline():
    """Create a temp dir with a downloaded image and config."""
    tmpdir = tempfile.mkdtemp()
    cfg = PipelineConfig(
        bbox=(-74.01, 40.70, -73.99, 40.72),
        out_dir=tmpdir,
        model_dir=tmpdir,
    )
    image_path = download_imagery(cfg)
    yield cfg, image_path, tmpdir


def test_run_langsam_creates_mask(setup_pipeline):
    cfg, image_path, tmpdir = setup_pipeline
    mask_path = run_langsam(image_path, ["trees", "water"], cfg)
    assert os.path.exists(mask_path)
    assert mask_path.endswith(".tif")


def test_run_sam2_creates_mask(setup_pipeline):
    cfg, image_path, tmpdir = setup_pipeline
    mask_path = run_sam2(image_path, cfg)
    assert os.path.exists(mask_path)
    assert mask_path.endswith(".tif")
