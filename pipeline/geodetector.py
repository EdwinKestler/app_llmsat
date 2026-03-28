"""GeoDeep integration — CPU-only object detection and segmentation.

Wraps the GeoDeep library to provide lightweight, GPU-free detection of
cars, trees, planes, buildings, roads, and multi-class aerial objects on
the same GeoTIFF imagery that the SAM3 pipeline uses.

All models are ONNX int8-quantized and downloaded on demand from
Hugging Face (~10-100 MB each).
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box, shape

logger = logging.getLogger("llmsat.geodetector")

try:
    from geodeep import run as geodeep_run
    from geodeep.models import list_models as geodeep_list_models

    GEODEEP_AVAILABLE = True
except ImportError:
    GEODEEP_AVAILABLE = False


# ── Diagnostics helper ──────────────────────────────────────────────

def _inspect_geotiff(path: str) -> dict:
    """Read GeoTIFF metadata and return a diagnostics dict."""
    with rasterio.open(path) as src:
        transform = src.transform
        res_x = abs(transform[0]) * 111_320  # degrees → metres (approx at equator)
        res_y = abs(transform[4]) * 110_540
        res_cm_x = res_x * 100
        res_cm_y = res_y * 100
        return {
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "dtype": str(src.dtypes[0]),
            "crs": str(src.crs),
            "transform_valid": transform[0] != 0 and transform[4] != 0,
            "res_cm_x": round(res_cm_x, 1),
            "res_cm_y": round(res_cm_y, 1),
            "bounds": src.bounds,
            "is_tiled": src.is_tiled,
        }


def _score_stats(scores: list[float]) -> str:
    """Return a compact string summarising a list of confidence scores."""
    if not scores:
        return "no scores"
    arr = np.array(scores)
    return (
        f"min={arr.min():.3f}  mean={arr.mean():.3f}  "
        f"max={arr.max():.3f}  std={arr.std():.3f}"
    )


# ── Model catalogue (subset relevant to satellite/aerial imagery) ────

DETECTION_MODELS = {
    "cars": {
        "model": "cars",
        "label": "Vehicles",
        "icon": "🚗",
        "type": "detection",
        "description": "Cars and vehicles (10 cm/px, YOLO v7)",
    },
    "trees_yolov9": {
        "model": "trees_yolov9",
        "label": "Trees (counted)",
        "icon": "🌲",
        "type": "detection",
        "description": "Individual tree crowns (10 cm/px, YOLO v9)",
    },
    "planes": {
        "model": "planes",
        "label": "Aircraft",
        "icon": "✈️",
        "type": "detection",
        "description": "Aircraft on runways/aprons (70 cm/px, YOLO v7)",
    },
    "aerovision": {
        "model": "aerovision",
        "label": "Multi-class (16)",
        "icon": "🔍",
        "type": "detection",
        "description": "Vehicles, boats, tanks, sports fields, bridges, cranes, helicopters (30 cm/px, YOLO v8)",
    },
    "utilities": {
        "model": "utilities",
        "label": "Utility markings",
        "icon": "🔧",
        "type": "detection",
        "description": "Gas, manhole, power, sewer, water markings (3 cm/px, YOLO v8)",
    },
    "buildings_geodeep": {
        "model": "buildings",
        "label": "Buildings (CPU)",
        "icon": "🏗️",
        "type": "segmentation",
        "description": "Building footprints via XUNet (50 cm/px, CPU-only)",
    },
    "roads_geodeep": {
        "model": "roads",
        "label": "Roads (CPU)",
        "icon": "🛤️",
        "type": "segmentation",
        "description": "Road segmentation via CNN (21 cm/px, CPU-only)",
    },
}


@dataclass
class DetectionResult:
    """Result from a single GeoDeep detection/segmentation run."""

    model_key: str
    label: str
    model_type: str  # "detection" or "segmentation"
    count: int = 0
    geojson: str = ""
    bboxes: list = field(default_factory=list)
    scores: list = field(default_factory=list)
    classes: list = field(default_factory=list)
    mask: Optional[np.ndarray] = None
    error: Optional[str] = None


def is_available() -> bool:
    """Return True if the geodeep package is importable."""
    return GEODEEP_AVAILABLE


def available_models() -> dict:
    """Return the detection model catalogue."""
    return DETECTION_MODELS


def run_detection(
    geotiff_path: str,
    model_key: str,
    conf_threshold: float | None = None,
    progress_callback=None,
) -> DetectionResult:
    """Run a GeoDeep model on a GeoTIFF and return structured results.

    Parameters
    ----------
    geotiff_path : str
        Path to the input GeoTIFF (RGB, georeferenced).
    model_key : str
        Key from ``DETECTION_MODELS`` (e.g. "cars", "aerovision").
    conf_threshold : float, optional
        Override detection confidence threshold.
    progress_callback : callable, optional
        ``fn(text, percent)`` for progress reporting.

    Returns
    -------
    DetectionResult
    """
    if not GEODEEP_AVAILABLE:
        return DetectionResult(
            model_key=model_key,
            label=DETECTION_MODELS.get(model_key, {}).get("label", model_key),
            model_type="unknown",
            error="geodeep package is not installed",
        )

    info = DETECTION_MODELS.get(model_key)
    if info is None:
        return DetectionResult(
            model_key=model_key,
            label=model_key,
            model_type="unknown",
            error=f"Unknown model key: {model_key}",
        )

    model_name = info["model"]
    model_type = info["type"]

    # ── Pre-flight diagnostics ────────────────────────────────────
    diag = _inspect_geotiff(geotiff_path)
    logger.info(
        "[GeoDeep] run_detection  model=%s  type=%s  image=%dx%d  "
        "bands=%d  dtype=%s  crs=%s  res=%.1f×%.1f cm/px  tiled=%s",
        model_name, model_type, diag["width"], diag["height"],
        diag["bands"], diag["dtype"], diag["crs"],
        diag["res_cm_x"], diag["res_cm_y"], diag["is_tiled"],
    )
    if not diag["transform_valid"]:
        logger.warning(
            "[GeoDeep] GeoTIFF transform looks invalid — GeoDeep will "
            "estimate resolution, which may reduce accuracy."
        )
    if not diag["is_tiled"]:
        logger.info(
            "[GeoDeep] GeoTIFF is not internally tiled — I/O may be slower."
        )

    t0 = time.time()

    try:
        if model_type == "detection":
            geojson_str = geodeep_run(
                geotiff_path,
                model_name,
                output_type="geojson",
                conf_threshold=conf_threshold,
                progress_callback=progress_callback,
            )
            elapsed = time.time() - t0
            fc = json.loads(geojson_str)
            features = fc.get("features", [])

            # Extract bboxes/scores/classes from GeoJSON for overlay drawing
            bboxes_out = []
            scores_out = []
            classes_out = []
            for feat in features:
                props = feat.get("properties", {})
                scores_out.append(props.get("score", 0.0))
                classes_out.append(props.get("class", "unknown"))
                # Extract bbox from polygon coordinates
                coords = feat["geometry"]["coordinates"][0]
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                bboxes_out.append([min(lons), min(lats), max(lons), max(lats)])

            # ── Post-detection diagnostics ────────────────────────
            cls_counts = Counter(classes_out)
            logger.info(
                "[GeoDeep] detection complete  model=%s  objects=%d  "
                "elapsed=%.1fs  classes=%s",
                model_name, len(features), elapsed,
                dict(cls_counts),
            )
            logger.info(
                "[GeoDeep] confidence stats: %s",
                _score_stats(scores_out),
            )
            # Log per-class score breakdown
            for cls_name in sorted(cls_counts):
                cls_scores = [s for s, c in zip(scores_out, classes_out) if c == cls_name]
                logger.info(
                    "[GeoDeep]   class %-15s  count=%-4d  scores: %s",
                    cls_name, cls_counts[cls_name], _score_stats(cls_scores),
                )

            return DetectionResult(
                model_key=model_key,
                label=info["label"],
                model_type="detection",
                count=len(features),
                geojson=geojson_str,
                bboxes=bboxes_out,
                scores=scores_out,
                classes=classes_out,
            )

        else:  # segmentation
            geojson_str = geodeep_run(
                geotiff_path,
                model_name,
                output_type="geojson",
                conf_threshold=conf_threshold,
                progress_callback=progress_callback,
            )
            elapsed = time.time() - t0
            fc = json.loads(geojson_str)
            features = fc.get("features", [])

            logger.info(
                "[GeoDeep] segmentation complete  model=%s  polygons=%d  "
                "elapsed=%.1fs",
                model_name, len(features), elapsed,
            )

            return DetectionResult(
                model_key=model_key,
                label=info["label"],
                model_type="segmentation",
                count=len(features),
                geojson=geojson_str,
            )

    except Exception as e:
        logger.error(
            "[GeoDeep] run_detection FAILED  model=%s  error=%s",
            model_name, e,
        )
        return DetectionResult(
            model_key=model_key,
            label=info["label"],
            model_type=model_type,
            error=str(e),
        )


# ── CPU fallback mapping ─────────────────────────────────────────
# Maps SAM3 segment prompts to GeoDeep models for CPU fallback mode.
CPU_FALLBACK_MAP = {
    "tree":     {"model": "trees_yolov9", "type": "detection"},
    "building": {"model": "buildings",    "type": "segmentation"},
    "road":     {"model": "roads",        "type": "segmentation"},
}


def has_cpu_fallback(segment_prompt: str) -> bool:
    """Return True if a SAM3 prompt can be handled by a GeoDeep CPU model."""
    return GEODEEP_AVAILABLE and segment_prompt in CPU_FALLBACK_MAP


def run_cpu_segmentation(
    geotiff_path: str,
    segment_prompt: str,
    out_dir: str,
) -> tuple[bool, str]:
    """Run a GeoDeep model as CPU fallback for a SAM3 segment prompt.

    Writes output files compatible with the LLMSat pipeline:
    - ``langsam_mask.tif`` — binary mask (uint8)
    - ``sam3_{prompt}_masks.tif`` — instance mask (int32)
    - ``sam3_{prompt}_scores.tif`` — confidence raster (float32)
    - ``segments.gpkg`` — vectorised polygons

    Parameters
    ----------
    geotiff_path : str
        Path to the input GeoTIFF (the satellite imagery).
    segment_prompt : str
        The segment name (e.g. "tree", "building", "road").
    out_dir : str
        Output directory for this segment.

    Returns
    -------
    (has_data, mask_path)
        Whether the mask contains any detections, and the path to the
        binary mask file.
    """
    if not GEODEEP_AVAILABLE:
        return False, ""

    info = CPU_FALLBACK_MAP.get(segment_prompt)
    if info is None:
        return False, ""

    model_name = info["model"]
    model_type = info["type"]

    os.makedirs(out_dir, exist_ok=True)
    mask_path = os.path.join(out_dir, "langsam_mask.tif")
    unique_path = os.path.join(out_dir, f"sam3_{segment_prompt}_masks.tif")
    scores_path = os.path.join(out_dir, f"sam3_{segment_prompt}_scores.tif")

    # ── Pre-flight diagnostics ────────────────────────────────────
    diag = _inspect_geotiff(geotiff_path)
    logger.info(
        "[GeoDeep] run_cpu_segmentation  prompt=%s  model=%s  type=%s  "
        "image=%dx%d  res=%.1f×%.1f cm/px  crs=%s",
        segment_prompt, model_name, model_type,
        diag["width"], diag["height"],
        diag["res_cm_x"], diag["res_cm_y"], diag["crs"],
    )

    with rasterio.open(geotiff_path) as src:
        profile = src.profile
        transform = src.transform
        height = src.height
        width = src.width

    t0 = time.time()

    if model_type == "segmentation":
        # GeoDeep segmentation returns a raw mask (numpy uint8 array)
        raw_mask = geodeep_run(
            geotiff_path, model_name, output_type="raw",
        )
        elapsed = time.time() - t0

        if raw_mask is None or not raw_mask.any():
            logger.info(
                "[GeoDeep] cpu_seg  model=%s  result=EMPTY  elapsed=%.1fs",
                model_name, elapsed,
            )
            _write_empty_masks(profile, height, width, mask_path, unique_path, scores_path)
            return False, mask_path

        logger.info(
            "[GeoDeep] cpu_seg  model=%s  raw_mask=%s  unique_classes=%s  "
            "elapsed=%.1fs",
            model_name, raw_mask.shape,
            sorted(np.unique(raw_mask).tolist()), elapsed,
        )

        # Raw mask may be smaller than imagery due to model resolution
        if raw_mask.shape != (height, width):
            logger.info(
                "[GeoDeep] resizing mask %s → %s (model res ≠ image res)",
                raw_mask.shape, (height, width),
            )
            from PIL import Image as _PILImage
            resized = _PILImage.fromarray(raw_mask).resize(
                (width, height), _PILImage.NEAREST,
            )
            raw_mask = np.array(resized)

        binary = (raw_mask > 0).astype(np.uint8)
        instance = raw_mask.astype(np.int32)
        max_val = float(raw_mask.max()) if raw_mask.max() > 0 else 1.0
        scores = (raw_mask.astype(np.float32) / max_val)
        scores[raw_mask == 0] = 0.0

    elif model_type == "detection":
        bboxes, det_scores, det_classes = geodeep_run(
            geotiff_path, model_name, output_type="bsc",
        )
        elapsed = time.time() - t0

        if not len(bboxes):
            logger.info(
                "[GeoDeep] cpu_seg(det)  model=%s  result=EMPTY  elapsed=%.1fs",
                model_name, elapsed,
            )
            _write_empty_masks(profile, height, width, mask_path, unique_path, scores_path)
            return False, mask_path

        det_score_list = [float(s) for s in det_scores]
        cls_counts = Counter(c[1] if isinstance(c, tuple) else str(c) for c in det_classes)
        logger.info(
            "[GeoDeep] cpu_seg(det)  model=%s  detections=%d  "
            "elapsed=%.1fs  classes=%s",
            model_name, len(bboxes), elapsed, dict(cls_counts),
        )
        logger.info(
            "[GeoDeep] confidence stats: %s", _score_stats(det_score_list),
        )

        # Rasterise bounding boxes into masks
        binary = np.zeros((height, width), dtype=np.uint8)
        instance = np.zeros((height, width), dtype=np.int32)
        score_raster = np.zeros((height, width), dtype=np.float32)
        n_clipped = 0

        for idx, (bbox_px, score) in enumerate(zip(bboxes, det_scores)):
            x1, y1, x2, y2 = [int(round(v)) for v in bbox_px]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                n_clipped += 1
                continue
            binary[y1:y2, x1:x2] = 1
            instance[y1:y2, x1:x2] = idx + 1
            score_raster[y1:y2, x1:x2] = np.maximum(
                score_raster[y1:y2, x1:x2], float(score),
            )

        if n_clipped:
            logger.warning(
                "[GeoDeep] %d/%d bboxes clipped to zero area (outside image bounds)",
                n_clipped, len(bboxes),
            )

        scores = score_raster
    else:
        return False, ""

    # Write binary mask
    out_profile = profile.copy()
    out_profile.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(mask_path, "w", **out_profile) as dst:
        dst.write(binary, 1)

    # Write instance mask
    inst_profile = profile.copy()
    inst_profile.update(dtype=rasterio.int32, count=1)
    with rasterio.open(unique_path, "w", **inst_profile) as dst:
        dst.write(instance, 1)

    # Write scores
    score_profile = profile.copy()
    score_profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(scores_path, "w", **score_profile) as dst:
        dst.write(scores, 1)

    has_data = bool(binary.any())
    n_pixels = int(binary.sum())
    n_instances = len(np.unique(instance)) - 1  # exclude 0
    logger.info(
        "[GeoDeep] cpu_seg DONE  prompt=%s  has_data=%s  "
        "instances=%d  pixels=%s  coverage=%.2f%%",
        segment_prompt, has_data, n_instances, f"{n_pixels:,}",
        (n_pixels / (height * width)) * 100,
    )
    return has_data, mask_path


def _write_empty_masks(profile, height, width, mask_path, unique_path, scores_path):
    """Write zero-filled mask files for a segment with no detections."""
    out_profile = profile.copy()
    out_profile.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(mask_path, "w", **out_profile) as dst:
        dst.write(np.zeros((height, width), dtype=np.uint8), 1)

    inst_profile = profile.copy()
    inst_profile.update(dtype=rasterio.int32, count=1)
    with rasterio.open(unique_path, "w", **inst_profile) as dst:
        dst.write(np.zeros((height, width), dtype=np.int32), 1)

    score_profile = profile.copy()
    score_profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(scores_path, "w", **score_profile) as dst:
        dst.write(np.zeros((height, width), dtype=np.float32), 1)


# ── Hybrid pipeline: GeoDeep detection → SAM3 refinement ────────

# Prompts where hybrid mode adds value (detection models that find
# discrete objects whose boundaries SAM3 can refine).
HYBRID_CAPABLE = {
    "tree":    "trees_yolov9",
    "car":     "cars",
    "vehicle": "cars",
    "plane":   "planes",
}


def has_hybrid_support(segment_prompt: str) -> bool:
    """Return True if a prompt can benefit from hybrid GeoDeep→SAM3."""
    return GEODEEP_AVAILABLE and segment_prompt in HYBRID_CAPABLE


def run_hybrid_segmentation(
    geotiff_path: str,
    segment_prompt: str,
    out_dir: str,
) -> tuple[bool, str, int]:
    """Hybrid pipeline: GeoDeep detects objects, SAM3 refines masks.

    1. Run GeoDeep detection model to find bounding boxes (CPU).
    2. Convert pixel-space bboxes → geographic coordinates (WGS84).
    3. Feed geographic bboxes to SAM3 ``generate_masks_by_boxes``.
    4. Save combined masks in LLMSat-compatible format.

    Parameters
    ----------
    geotiff_path : str
        Path to the input GeoTIFF.
    segment_prompt : str
        Segment name (e.g. "tree", "car").
    out_dir : str
        Output directory for this segment.

    Returns
    -------
    (has_data, mask_path, n_detections)
    """
    if not GEODEEP_AVAILABLE:
        return False, "", 0

    model_name = HYBRID_CAPABLE.get(segment_prompt)
    if model_name is None:
        return False, "", 0

    os.makedirs(out_dir, exist_ok=True)

    # ── Pre-flight diagnostics ────────────────────────────────────
    diag = _inspect_geotiff(geotiff_path)
    logger.info(
        "[Hybrid] START  prompt=%s  geodeep_model=%s  "
        "image=%dx%d  res=%.1f×%.1f cm/px",
        segment_prompt, model_name,
        diag["width"], diag["height"],
        diag["res_cm_x"], diag["res_cm_y"],
    )

    # ── Step 1: GeoDeep detection (CPU) ──────────────────────────
    t0 = time.time()
    bboxes_px, scores_det, classes_det = geodeep_run(
        geotiff_path, model_name, output_type="bsc",
    )
    t_detect = time.time() - t0

    if not len(bboxes_px):
        logger.info(
            "[Hybrid] GeoDeep found 0 objects (%.1fs) — nothing to refine",
            t_detect,
        )
        mask_path = os.path.join(out_dir, "langsam_mask.tif")
        unique_path = os.path.join(out_dir, f"sam3_{segment_prompt}_masks.tif")
        scores_path = os.path.join(out_dir, f"sam3_{segment_prompt}_scores.tif")
        with rasterio.open(geotiff_path) as src:
            _write_empty_masks(src.profile, src.height, src.width,
                               mask_path, unique_path, scores_path)
        return False, mask_path, 0

    n_detections = len(bboxes_px)
    det_score_list = [float(s) for s in scores_det]
    logger.info(
        "[Hybrid] Step 1 DONE  GeoDeep detections=%d  elapsed=%.1fs  "
        "scores: %s",
        n_detections, t_detect, _score_stats(det_score_list),
    )

    # ── Step 2: Convert pixel bboxes → geographic coords ─────────
    with rasterio.open(geotiff_path) as src:
        transform = src.transform

    geo_boxes = []
    for i, bbox_px in enumerate(bboxes_px):
        x1, y1, x2, y2 = bbox_px[:4]
        # rasterio transform: pixel (col, row) → geographic (x, y)
        west, north = transform * (float(x1), float(y1))
        east, south = transform * (float(x2), float(y2))
        # Ensure west < east, south < north
        west, east = min(west, east), max(west, east)
        south, north = min(south, north), max(south, north)
        geo_boxes.append([west, south, east, north])
        if i < 5 or i == n_detections - 1:  # log first 5 + last
            logger.debug(
                "[Hybrid]   box[%d] px=[%.0f,%.0f,%.0f,%.0f] → "
                "geo=[%.6f,%.6f,%.6f,%.6f]  score=%.3f",
                i, x1, y1, x2, y2, west, south, east, north,
                det_score_list[i],
            )

    logger.info(
        "[Hybrid] Step 2 DONE  converted %d pixel bboxes → WGS84",
        len(geo_boxes),
    )

    # ── Step 3: Feed to SAM3 for pixel-perfect masks ─────────────
    logger.info(
        "[Hybrid] Step 3  feeding %d boxes to SAM3 generate_masks_by_boxes...",
        len(geo_boxes),
    )
    t1 = time.time()
    from segmenter import _get_sam3

    sam = _get_sam3()
    sam.set_image(geotiff_path)

    box_labels = [True] * len(geo_boxes)
    sam.generate_masks_by_boxes(geo_boxes, box_labels, box_crs="EPSG:4326")
    t_sam = time.time() - t1

    mask_path = os.path.join(out_dir, "langsam_mask.tif")
    unique_path = os.path.join(out_dir, f"sam3_{segment_prompt}_masks.tif")
    scores_path = os.path.join(out_dir, f"sam3_{segment_prompt}_scores.tif")

    n_sam_masks = len(sam.masks) if sam.masks else 0
    logger.info(
        "[Hybrid] Step 3 DONE  SAM3 produced %d masks from %d boxes  "
        "elapsed=%.1fs",
        n_sam_masks, n_detections, t_sam,
    )

    if sam.masks:
        # Save unique instance masks and scores
        sam.save_masks(output=unique_path, save_scores=scores_path, unique=True)

        # Save binary mask
        tmp_binary = os.path.join(out_dir, "_hybrid_binary_tmp.tif")
        sam.save_masks(output=tmp_binary, unique=False)
        with rasterio.open(tmp_binary) as src:
            binary_data = src.read(1)
            profile = src.profile
        os.remove(tmp_binary)

        profile.update(dtype=rasterio.uint8, count=1)
        with rasterio.open(mask_path, "w", **profile) as dst:
            dst.write(binary_data.astype(np.uint8), 1)

        has_data = bool(binary_data.any())
        n_pixels = int(binary_data.astype(bool).sum())
        total_time = time.time() - t0
        logger.info(
            "[Hybrid] COMPLETE  prompt=%s  detections=%d → masks=%d  "
            "pixels=%s  coverage=%.2f%%  total=%.1fs (detect=%.1fs + SAM3=%.1fs)",
            segment_prompt, n_detections, n_sam_masks,
            f"{n_pixels:,}",
            (n_pixels / (diag["width"] * diag["height"])) * 100,
            total_time, t_detect, t_sam,
        )
    else:
        # SAM3 found nothing — write empty masks
        with rasterio.open(geotiff_path) as src:
            _write_empty_masks(src.profile, src.height, src.width,
                               mask_path, unique_path, scores_path)
        has_data = False
        logger.warning(
            "[Hybrid] SAM3 produced NO masks from %d GeoDeep boxes",
            n_detections,
        )

    return has_data, mask_path, n_detections


def detection_result_to_geojson_features(result: DetectionResult) -> list[dict]:
    """Extract GeoJSON features from a DetectionResult, tagged with model metadata.

    Each feature gets additional properties: ``source``, ``model``, ``detection_type``.
    """
    if not result.geojson:
        return []

    fc = json.loads(result.geojson)
    features = fc.get("features", [])

    for feat in features:
        props = feat.setdefault("properties", {})
        props["source"] = "geodeep"
        props["model"] = result.model_key
        props["detection_type"] = result.model_type
        if "segment" not in props:
            props["segment"] = result.model_key

    return features
