"""Segmentation helpers using SAM3.

Entry points:

* ``run_text_segmentation`` – text-prompted masks (e.g. "tree", "building").
  For "building" prompts, if Open Buildings data is available, it uses
  known building bounding boxes to guide SAM3 for much better results.
* ``run_auto_segmentation`` – general segmentation over the bounding box.

Both share a single cached ``SamGeo3`` instance so the 848 M checkpoint
is loaded only once per Streamlit process.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import rasterio
from samgeo import SamGeo3
from pipeline.config import PipelineConfig

# Module-level cache — loaded once, reused across calls.
_sam3_instance: SamGeo3 | None = None


def _get_sam3() -> SamGeo3:
    """Return a cached SamGeo3 instance, creating it on first call."""
    global _sam3_instance
    if _sam3_instance is None:
        _sam3_instance = SamGeo3(
            backend="meta",
            device=None,
            checkpoint_path=None,
            load_from_HF=True,
        )
    return _sam3_instance


def _mask_has_data(path: str) -> bool:
    """Return True if a mask file exists AND contains non-zero pixels."""
    if not os.path.exists(path):
        return False
    with rasterio.open(path) as src:
        return src.read(1).any()


# ── Open Buildings guided segmentation ──────────────────────────────

# Prompts that can be enhanced with Open Buildings data.
_BUILDING_PROMPTS = {"building", "buildings", "house", "houses", "structure", "roof"}

# Max buildings to send as box prompts (SAM3 limit / memory).
_MAX_BOX_PROMPTS = 500


def _try_open_buildings_boxes(
    bbox: list[float],
    prompt: str,
) -> list[list[float]] | None:
    """Query Open Buildings for box prompts if available and relevant.

    Returns a list of [xmin, ymin, xmax, ymax] boxes or None.
    """
    if prompt not in _BUILDING_PROMPTS:
        return None

    try:
        from open_buildings import query_buildings, buildings_to_boxes
    except ImportError:
        return None

    gdf = query_buildings(bbox, min_confidence=0.7, max_buildings=_MAX_BOX_PROMPTS)
    if gdf.empty:
        return None

    return buildings_to_boxes(gdf)


def run_box_segmentation(
    image_path: str,
    boxes: list[list[float]],
    config: PipelineConfig,
    label: str = "building",
) -> str:
    """Segment using bounding box prompts from Open Buildings.

    Parameters
    ----------
    image_path : str
        Path to GeoTIFF image.
    boxes : list[list[float]]
        List of [xmin, ymin, xmax, ymax] in EPSG:4326.
    config : PipelineConfig
        Pipeline config with out_dir.
    label : str
        Label for output filenames.

    Returns
    -------
    str
        Path to binary mask GeoTIFF.
    """
    os.makedirs(config.out_dir, exist_ok=True)

    sam = _get_sam3()
    sam.set_image(image_path)

    # All boxes are positive (include) prompts
    box_labels = [True] * len(boxes)

    # Process in batches to avoid memory issues
    batch_size = 50
    with rasterio.open(image_path) as src:
        profile = src.profile

    combined = None
    for i in range(0, len(boxes), batch_size):
        batch = boxes[i : i + batch_size]
        batch_labels = box_labels[i : i + batch_size]

        sam.generate_masks_by_boxes(batch, batch_labels, box_crs="EPSG:4326")

        if not sam.masks:
            continue

        tmp = os.path.join(config.out_dir, f"_batch_{i}.tif")
        sam.save_masks(output=tmp, unique=False)
        with rasterio.open(tmp) as src:
            mask = src.read(1)
        combined = mask if combined is None else np.logical_or(combined, mask)
        os.remove(tmp)

    # Save unique instance mask + scores for the full result
    if combined is not None and combined.any():
        # Re-run on all boxes for the unique mask (for visualization)
        # Use text prompt as well for the unique/scores output
        sam.generate_masks(prompt=label)
        if sam.masks:
            unique_path = os.path.join(config.out_dir, f"sam3_{label}_masks.tif")
            scores_path = os.path.join(config.out_dir, f"sam3_{label}_scores.tif")
            sam.save_masks(output=unique_path, save_scores=scores_path, unique=True)

    mask_path = os.path.join(config.out_dir, "langsam_mask.tif")
    profile.update(dtype=rasterio.uint8, count=1)
    if combined is None:
        combined = np.zeros((profile["height"], profile["width"]), dtype=np.uint8)
    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(combined.astype(rasterio.uint8), 1)
    return mask_path


# ── text-prompted segmentation ──────────────────────────────────────

def run_text_segmentation(
    image_path: str,
    text_prompts: Iterable[str],
    config: PipelineConfig,
) -> str:
    """Generate a combined binary mask for the provided text prompts.

    For building-related prompts, attempts to use Open Buildings box
    prompts first (much better accuracy).  Falls back to SAM3 text
    prompts for other segment types or if Open Buildings data is
    unavailable.

    Returns the path to the combined mask GeoTIFF.
    """
    os.makedirs(config.out_dir, exist_ok=True)
    mask_path = os.path.join(config.out_dir, "langsam_mask.tif")

    if _mask_has_data(mask_path):
        return mask_path

    sam = _get_sam3()
    sam.set_image(image_path)

    with rasterio.open(image_path) as src:
        profile = src.profile

    combined = None
    for prompt in text_prompts:
        # Try Open Buildings guided segmentation for building prompts
        if config.bbox:
            boxes = _try_open_buildings_boxes(list(config.bbox), prompt)
            if boxes:
                box_mask_path = run_box_segmentation(
                    image_path, boxes, config, label=prompt,
                )
                with rasterio.open(box_mask_path) as src:
                    mask = src.read(1)
                if mask.any():
                    combined = mask if combined is None else np.logical_or(combined, mask)
                    continue  # Skip text prompt — boxes worked

        # Fallback: SAM3 text prompt
        sam.generate_masks(prompt=prompt)

        if not sam.masks:
            continue

        unique_path = os.path.join(config.out_dir, f"sam3_{prompt}_masks.tif")
        scores_path = os.path.join(config.out_dir, f"sam3_{prompt}_scores.tif")
        sam.save_masks(output=unique_path, save_scores=scores_path, unique=True)

        tmp_binary = os.path.join(config.out_dir, f"sam3_{prompt}_binary.tif")
        sam.save_masks(output=tmp_binary, unique=False)
        with rasterio.open(tmp_binary) as mask_src:
            mask = mask_src.read(1)
        combined = mask if combined is None else np.logical_or(combined, mask)
        os.remove(tmp_binary)

    profile.update(dtype=rasterio.uint8, count=1)
    if combined is None:
        combined = np.zeros((profile["height"], profile["width"]), dtype=np.uint8)
    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(combined.astype(rasterio.uint8), 1)
    return mask_path


# ── backwards-compatible aliases ────────────────────────────────────
run_langsam = run_text_segmentation


# ── auto segmentation ──────────────────────────────────────────────

def run_auto_segmentation(image_path: str, config: PipelineConfig) -> str:
    """Segment everything in the image using a bounding-box prompt."""
    os.makedirs(config.out_dir, exist_ok=True)
    mask_path = os.path.join(config.out_dir, "auto_mask.tif")

    if _mask_has_data(mask_path):
        return mask_path

    sam = _get_sam3()
    sam.set_image(image_path)

    if config.bbox:
        boxes = [list(config.bbox)]
        box_labels = [True]
        sam.generate_masks_by_boxes(boxes, box_labels, box_crs="EPSG:4326")
    else:
        sam.generate_masks(prompt="segment everything")

    if sam.masks:
        sam.save_masks(output=mask_path, unique=True)
    else:
        with rasterio.open(image_path) as src:
            profile = src.profile
        profile.update(dtype=rasterio.uint8, count=1)
        with rasterio.open(mask_path, "w", **profile) as dst:
            dst.write(np.zeros((profile["height"], profile["width"]), dtype=np.uint8), 1)

    return mask_path


# ── backwards-compatible aliases ────────────────────────────────────
run_sam2 = run_auto_segmentation
