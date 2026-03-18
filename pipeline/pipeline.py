"""High level geospatial processing pipeline."""

from __future__ import annotations

import os
from typing import Iterable, Dict

from downloader import download_imagery
from segmenter import run_text_segmentation, run_auto_segmentation
from .config import PipelineConfig
from vectorizer import raster_to_vector, summarise


def run_pipeline(config: PipelineConfig, text_prompts: Iterable[str]) -> Dict[str, str]:
    """Execute the end-to-end processing pipeline.

    Steps
    -----
    1. Download imagery covering ``config.bbox`` at ``config.zoom``.
    2. Run SAM3 text-prompted segmentation for each prompt.
    3. Run SAM3 auto-segmentation over the bounding box.
    4. Vectorise the **text-prompted** mask and export GeoPackage/CSV.

    Returns
    -------
    dict
        Mapping of product names to file paths.
    """
    os.makedirs(config.out_dir, exist_ok=True)

    image_path = download_imagery(config=config)
    semantic_mask = run_text_segmentation(
        image_path=image_path, text_prompts=text_prompts, config=config,
    )
    auto_mask = run_auto_segmentation(image_path=image_path, config=config)

    # Vectorise the TEXT-PROMPTED mask (not the auto mask)
    gpkg_path = os.path.join(config.out_dir, "segments.gpkg")
    csv_path = os.path.join(config.out_dir, "summary.csv")
    gdf = raster_to_vector(semantic_mask, gpkg_path)
    summarise(gdf, csv_path)

    return {
        "image": image_path,
        "semantic_mask": semantic_mask,
        "auto_mask": auto_mask,
        "gpkg": gpkg_path,
        "csv": csv_path,
    }
