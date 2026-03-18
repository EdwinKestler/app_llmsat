"""Tests for downloader module."""

import os
import tempfile
import pytest
from pipeline.config import PipelineConfig
from downloader import download_imagery


def test_download_imagery_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = PipelineConfig(bbox=(-74.01, 40.70, -73.99, 40.72), out_dir=tmpdir)
        path = download_imagery(cfg)
        assert os.path.exists(path)
        assert path.endswith(".tif")


def test_download_imagery_output_dir_created():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "nested", "dir")
        cfg = PipelineConfig(bbox=(0, 0, 1, 1), out_dir=out)
        path = download_imagery(cfg)
        assert os.path.isdir(out)
        assert os.path.exists(path)
