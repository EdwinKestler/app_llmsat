"""High level geospatial processing pipeline."""

from __future__ import annotations

import os
from typing import Iterable, Dict

from downloader import download_imagery
from segmenter import run_langsam, run_sam2
from .config import PipelineConfig
from vectorizer import raster_to_vector, summarise


def run_pipeline(config: PipelineConfig, text_prompts: Iterable[str]) -> Dict[str, str]:
    """Execute the end-to-end processing pipeline.

    Steps
    -----
    1. Download imagery covering ``config.bbox`` at ``config.zoom`` using ``tms_to_geotiff``.
    2. Run LangSAM to generate a semantic mask from ``text_prompts``.
    3. Run SAM2 to obtain a general segmentation mask.
    4. Vectorise the SAM2 mask and export GeoPackage/CSV summaries.

    Parameters
    ----------
    config: PipelineConfig
        Pipeline configuration with bounding box, directories, thresholds
        and model information.
    text_prompts: Iterable[str]
        Prompts for LangSAM.

    Returns
    -------
    dict
        Mapping of product names to file paths.
    """
    os.makedirs(config.out_dir, exist_ok=True)

    image_path = download_imagery(config=config)
    semantic_mask = run_langsam(image_path=image_path, text_prompts=text_prompts, config=config)
    sam2_mask = run_sam2(image_path=image_path, config=config)

    gpkg_path = os.path.join(config.out_dir, "segments.gpkg")
    csv_path = os.path.join(config.out_dir, "summary.csv")
    gdf = raster_to_vector(sam2_mask, gpkg_path)
    summarise(gdf, csv_path)

    return {
        "image": image_path,
        "semantic_mask": semantic_mask,
        "sam2_mask": sam2_mask,
        "gpkg": gpkg_path,
        "csv": csv_path,
    }
