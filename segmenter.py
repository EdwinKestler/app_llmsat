"""Segmentation helpers using SAM3 and Google Open Buildings.

Entry points:

* ``run_text_segmentation`` – text-prompted masks (e.g. "tree", "road").
  For "building" prompts, if Open Buildings data is available, polygons
  are rasterized directly into a mask (no SAM3 needed).
* ``run_auto_segmentation`` – general segmentation over the bounding box.

Both share a single cached ``SamGeo3`` instance so the 848 M checkpoint
is loaded only once per Streamlit process.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import rowcol
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


# ── Open Buildings rasterization ────────────────────────────────────

# Prompts that can use Open Buildings data directly.
_BUILDING_PROMPTS = {"building", "buildings", "house", "houses", "structure", "roof"}


def _rasterize_open_buildings(
    image_path: str,
    bbox: list[float],
    config: PipelineConfig,
    label: str = "building",
) -> str | None:
    """Rasterize Open Buildings polygons directly into a mask.

    Instead of sending individual boxes to SAM3, this burns all building
    footprints from Google Open Buildings into a single raster mask.
    Much faster and more complete than box-prompted SAM3.

    Returns the path to the mask, or None if Open Buildings data is
    unavailable or has no buildings for this area.
    """
    try:
        from open_buildings import query_buildings
    except ImportError:
        return None

    gdf = query_buildings(bbox, min_confidence=0.65, max_buildings=50_000)
    if gdf.empty:
        return None

    with rasterio.open(image_path) as src:
        profile = src.profile
        transform = src.transform
        height = src.height
        width = src.width

    # Rasterize all building polygons at once — each gets a unique ID
    shapes_with_values = [(geom, idx + 1) for idx, geom in enumerate(gdf.geometry) if geom is not None]

    if not shapes_with_values:
        return None

    # Unique instance mask (each building = different value)
    unique_mask = rasterize(
        shapes_with_values,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.int32,
    )

    # Binary mask
    binary_mask = (unique_mask > 0).astype(np.uint8)

    # Save binary mask as langsam_mask.tif
    mask_path = os.path.join(config.out_dir, "langsam_mask.tif")
    out_profile = profile.copy()
    out_profile.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(mask_path, "w", **out_profile) as dst:
        dst.write(binary_mask, 1)

    # Save unique instance mask for visualization
    unique_path = os.path.join(config.out_dir, f"sam3_{label}_masks.tif")
    inst_profile = profile.copy()
    inst_profile.update(dtype=rasterio.int32, count=1)
    with rasterio.open(unique_path, "w", **inst_profile) as dst:
        dst.write(unique_mask, 1)

    # Save confidence as a raster (burn confidence values per polygon)
    conf_values = gdf["confidence"].values
    shapes_with_conf = [
        (geom, float(conf))
        for geom, conf in zip(gdf.geometry, conf_values)
        if geom is not None
    ]
    conf_mask = rasterize(
        shapes_with_conf,
        out_shape=(height, width),
        transform=transform,
        fill=0.0,
        dtype=np.float32,
    )
    scores_path = os.path.join(config.out_dir, f"sam3_{label}_scores.tif")
    score_profile = profile.copy()
    score_profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(scores_path, "w", **score_profile) as dst:
        dst.write(conf_mask, 1)

    n_buildings = len(shapes_with_values)
    n_pixels = int(binary_mask.sum())
    print(f"Rasterized {n_buildings} Open Buildings footprints ({n_pixels:,} pixels)")

    return mask_path


# ── OSM road rasterization ──────────────────────────────────────────

_ROAD_PROMPTS = {"road", "roads", "street", "streets", "highway", "path"}


def _rasterize_osm_roads(
    image_path: str,
    bbox: list[float],
    config: PipelineConfig,
    label: str = "road",
) -> str | None:
    """Rasterize OpenStreetMap road polygons directly into a mask.

    Fetches roads via Overpass API, buffers linestrings to approximate
    road width, and burns them into a raster mask.  Much more complete
    than SAM3 text prompts for road detection.

    Returns the path to the mask, or None if the query fails or
    returns no roads.
    """
    try:
        from osm_roads import query_roads, buffer_roads
    except ImportError:
        return None

    try:
        gdf = query_roads(bbox)
    except Exception as e:
        print(f"OSM road query failed: {e}")
        return None

    if gdf.empty:
        return None

    # Buffer linestrings to polygons based on road type width
    gdf_buffered = buffer_roads(gdf)

    with rasterio.open(image_path) as src:
        profile = src.profile
        transform = src.transform
        height = src.height
        width = src.width

    shapes_with_values = [
        (geom, idx + 1)
        for idx, geom in enumerate(gdf_buffered.geometry)
        if geom is not None and not geom.is_empty
    ]

    if not shapes_with_values:
        return None

    # Unique instance mask
    unique_mask = rasterize(
        shapes_with_values,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.int32,
    )

    binary_mask = (unique_mask > 0).astype(np.uint8)

    # Save binary mask
    mask_path = os.path.join(config.out_dir, "langsam_mask.tif")
    out_profile = profile.copy()
    out_profile.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(mask_path, "w", **out_profile) as dst:
        dst.write(binary_mask, 1)

    # Save unique instance mask for visualization
    unique_path = os.path.join(config.out_dir, f"sam3_{label}_masks.tif")
    inst_profile = profile.copy()
    inst_profile.update(dtype=rasterio.int32, count=1)
    with rasterio.open(unique_path, "w", **inst_profile) as dst:
        dst.write(unique_mask, 1)

    # Save road type as "confidence" (use normalized width as proxy)
    max_width = max(r["width_m"] for _, r in gdf_buffered.iterrows()) if len(gdf_buffered) else 1
    shapes_with_width = [
        (geom, float(row["width_m"]) / max_width)
        for _, row in gdf_buffered.iterrows()
        for geom in [row.geometry]
        if geom is not None and not geom.is_empty
    ]
    width_mask = rasterize(
        shapes_with_width,
        out_shape=(height, width),
        transform=transform,
        fill=0.0,
        dtype=np.float32,
    )
    scores_path = os.path.join(config.out_dir, f"sam3_{label}_scores.tif")
    score_profile = profile.copy()
    score_profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(scores_path, "w", **score_profile) as dst:
        dst.write(width_mask, 1)

    n_roads = len(shapes_with_values)
    n_pixels = int(binary_mask.sum())
    print(f"Rasterized {n_roads} OSM road segments ({n_pixels:,} pixels)")

    return mask_path


# ── text-prompted segmentation ──────────────────────────────────────

def run_text_segmentation(
    image_path: str,
    text_prompts: Iterable[str],
    config: PipelineConfig,
) -> str:
    """Generate a combined binary mask for the provided text prompts.

    For building-related prompts, Open Buildings polygons are rasterized
    directly (no SAM3 needed).  For other prompts, SAM3 text-prompted
    segmentation is used.

    Returns the path to the combined mask GeoTIFF.
    """
    os.makedirs(config.out_dir, exist_ok=True)
    mask_path = os.path.join(config.out_dir, "langsam_mask.tif")

    if _mask_has_data(mask_path):
        return mask_path

    with rasterio.open(image_path) as src:
        profile = src.profile

    combined = None
    for prompt in text_prompts:

        # For building prompts: rasterize Open Buildings directly
        if prompt in _BUILDING_PROMPTS and config.bbox:
            ob_path = _rasterize_open_buildings(
                image_path, list(config.bbox), config, label=prompt,
            )
            if ob_path:
                with rasterio.open(ob_path) as src:
                    mask = src.read(1)
                if mask.any():
                    combined = mask if combined is None else np.logical_or(combined, mask)
                    continue

        # For road prompts: rasterize OpenStreetMap roads directly
        if prompt in _ROAD_PROMPTS and config.bbox:
            osm_path = _rasterize_osm_roads(
                image_path, list(config.bbox), config, label=prompt,
            )
            if osm_path:
                with rasterio.open(osm_path) as src:
                    mask = src.read(1)
                if mask.any():
                    combined = mask if combined is None else np.logical_or(combined, mask)
                    continue

        # Fallback: SAM3 text prompt
        sam = _get_sam3()
        sam.set_image(image_path)
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
