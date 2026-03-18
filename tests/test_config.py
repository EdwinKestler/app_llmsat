"""Tests for pipeline.config module."""

import os
import tempfile
import pytest
from pipeline.config import PipelineConfig, load_config


def test_pipeline_config_defaults():
    cfg = PipelineConfig(bbox=(-74.01, 40.70, -73.99, 40.72))
    assert cfg.zoom == 18
    assert cfg.out_dir == "output"
    assert cfg.model_dir == "checkpoints"
    assert cfg.sam2_checkpoint == "sam2_hiera_l.pt"
    assert cfg.box_threshold == 0.24
    assert cfg.text_threshold == 0.24


def test_pipeline_config_custom():
    cfg = PipelineConfig(
        bbox=(0, 0, 1, 1),
        zoom=15,
        out_dir="custom_out",
        box_threshold=0.5,
    )
    assert cfg.zoom == 15
    assert cfg.out_dir == "custom_out"
    assert cfg.box_threshold == 0.5


def test_load_config_with_overrides():
    cfg = load_config(bbox=(1, 2, 3, 4), out_dir="test_output")
    assert cfg.bbox == (1, 2, 3, 4)
    assert cfg.out_dir == "test_output"


def test_load_config_from_yaml():
    yaml_content = "bbox: [10, 20, 30, 40]\nzoom: 16\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        cfg = load_config(path=f.name)
    os.unlink(f.name)
    assert cfg.bbox == [10, 20, 30, 40]
    assert cfg.zoom == 16


def test_load_config_yaml_with_overrides():
    yaml_content = "bbox: [10, 20, 30, 40]\nzoom: 16\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        cfg = load_config(path=f.name, zoom=12)
    os.unlink(f.name)
    assert cfg.zoom == 12
