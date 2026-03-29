"""RF-DETR integration — transformer-based object detection (GPU).

RF-DETR (ICLR 2026) uses a DINOv2 backbone with deformable attention
for real-time, NMS-free object detection.  Achieves 90 mAP on aerial
imagery — significantly better than YOLO on satellite/drone data.

Models download automatically from Roboflow on first use (~30-130 MB).
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

logger = logging.getLogger("llmsat.rfdetector")

try:
    import rfdetr
    from PIL import Image as PILImage

    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False


# ── Model catalogue ──────────────────────────────────────────────

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHECKPOINTS_DIR = os.path.join(_PROJECT_ROOT, "checkpoints")

RFDETR_MODELS = {
    "rfdetr_base": {
        "class": "RFDETRBase",
        "weights": "rf-detr-base.pth",
        "label": "RF-DETR Base",
        "icon": "🎯",
        "description": "General-purpose detection (COCO 80 classes, 53 mAP, balanced speed/accuracy)",
    },
    "rfdetr_large": {
        "class": "RFDETRLarge",
        "weights": "rf-detr-large-2026.pth",
        "label": "RF-DETR Large",
        "icon": "🔬",
        "description": "High-accuracy detection (COCO 80 classes, 56.5 mAP, best for satellite)",
    },
}

# COCO classes relevant to aerial/satellite imagery
AERIAL_CLASSES = {
    "car", "truck", "bus", "motorcycle", "bicycle",
    "boat", "airplane",
    "train",
    "person",
    "bench", "umbrella",
    "sports ball", "kite",
}


@dataclass
class RFDetectionResult:
    """Result from a single RF-DETR detection run."""

    model_key: str
    label: str
    count: int = 0
    bboxes: list = field(default_factory=list)
    scores: list = field(default_factory=list)
    classes: list = field(default_factory=list)
    class_ids: list = field(default_factory=list)
    elapsed: float = 0.0
    image_width: int = 0
    image_height: int = 0
    error: Optional[str] = None


# ── Module-level model cache ─────────────────────────────────────
_model_cache: dict[str, object] = {}


def is_available() -> bool:
    """Return True if the rfdetr package is importable."""
    return RFDETR_AVAILABLE


def available_models() -> dict:
    """Return the RF-DETR model catalogue."""
    return RFDETR_MODELS


def _get_model(model_key: str):
    """Load or return cached RF-DETR model instance."""
    if model_key in _model_cache:
        return _model_cache[model_key]

    info = RFDETR_MODELS.get(model_key)
    if info is None:
        raise ValueError(f"Unknown RF-DETR model: {model_key}")

    cls_name = info["class"]
    weights_file = info.get("weights", "")
    model_cls = getattr(rfdetr, cls_name)

    # Use local checkpoint if available, otherwise let rfdetr download
    local_path = os.path.join(_CHECKPOINTS_DIR, weights_file)
    kwargs = {}
    if os.path.exists(local_path):
        kwargs["pretrain_weights"] = local_path
        logger.info("[RF-DETR] Loading %s from local checkpoint: %s", model_key, local_path)
    else:
        logger.info("[RF-DETR] Local checkpoint not found at %s — will download", local_path)

    t0 = time.time()
    model = model_cls(**kwargs)
    logger.info("[RF-DETR] Model loaded in %.1fs", time.time() - t0)

    _model_cache[model_key] = model
    return model


def run_detection(
    geotiff_path: str,
    model_key: str = "rfdetr_base",
    threshold: float = 0.3,
    aerial_only: bool = False,
) -> RFDetectionResult:
    """Run RF-DETR detection on a GeoTIFF.

    Parameters
    ----------
    geotiff_path : str
        Path to the input GeoTIFF (RGB, georeferenced).
    model_key : str
        Key from ``RFDETR_MODELS``.
    threshold : float
        Detection confidence threshold (0-1).
    aerial_only : bool
        If True, filter results to only aerial-relevant COCO classes.

    Returns
    -------
    RFDetectionResult
    """
    if not RFDETR_AVAILABLE:
        return RFDetectionResult(
            model_key=model_key,
            label=RFDETR_MODELS.get(model_key, {}).get("label", model_key),
            error="rfdetr package is not installed. Run: pip install rfdetr",
        )

    info = RFDETR_MODELS.get(model_key)
    if info is None:
        return RFDetectionResult(
            model_key=model_key, label=model_key,
            error=f"Unknown model: {model_key}",
        )

    # ── Load image ────────────────────────────────────────────────
    with rasterio.open(geotiff_path) as src:
        bands = src.read([1, 2, 3])
        img_h, img_w = src.height, src.width
    rgb = np.moveaxis(bands, 0, -1)  # (3,H,W) → (H,W,3)
    pil_image = PILImage.fromarray(rgb)

    logger.info(
        "[RF-DETR] run_detection  model=%s  image=%dx%d  threshold=%.2f  "
        "aerial_only=%s",
        model_key, img_w, img_h, threshold, aerial_only,
    )

    # ── Run detection ─────────────────────────────────────────────
    try:
        model = _get_model(model_key)
    except Exception as e:
        return RFDetectionResult(
            model_key=model_key, label=info["label"],
            error=f"Model load failed: {e}",
        )

    t0 = time.time()
    try:
        detections = model.predict(pil_image, threshold=threshold)
    except Exception as e:
        return RFDetectionResult(
            model_key=model_key, label=info["label"],
            error=f"Inference failed: {e}",
        )
    elapsed = time.time() - t0

    # ── Parse supervision.Detections ──────────────────────────────
    try:
        from rfdetr.assets.coco_classes import COCO_CLASSES as _COCO_MAP
    except ImportError:
        try:
            from rfdetr.util.coco_classes import COCO_CLASSES as _COCO_MAP
        except ImportError:
            _COCO_MAP = {i: str(i) for i in range(80)}
    # _COCO_MAP is a dict {1: "person", 2: "bicycle", 3: "car", ...}

    n_raw = len(detections) if detections is not None else 0

    bboxes = []
    scores = []
    classes = []
    class_ids = []

    if n_raw > 0:
        for i in range(n_raw):
            cid = int(detections.class_id[i])
            score = float(detections.confidence[i])
            cls_name = _COCO_MAP.get(cid, _COCO_MAP.get(cid + 1, str(cid)))

            if aerial_only and cls_name not in AERIAL_CLASSES:
                continue

            x1, y1, x2, y2 = detections.xyxy[i]
            bboxes.append([float(x1), float(y1), float(x2), float(y2)])
            scores.append(score)
            classes.append(cls_name)
            class_ids.append(cid)

    # ── Diagnostics ───────────────────────────────────────────────
    cls_counts = Counter(classes)
    score_arr = np.array(scores) if scores else np.array([])
    logger.info(
        "[RF-DETR] detection complete  model=%s  raw=%d  filtered=%d  "
        "elapsed=%.2fs  classes=%s",
        model_key, n_raw, len(bboxes), elapsed, dict(cls_counts),
    )
    if len(score_arr):
        logger.info(
            "[RF-DETR] confidence stats: min=%.3f  mean=%.3f  max=%.3f  std=%.3f",
            score_arr.min(), score_arr.mean(), score_arr.max(), score_arr.std(),
        )
        for cls_name in sorted(cls_counts):
            cls_scores = [s for s, c in zip(scores, classes) if c == cls_name]
            cs = np.array(cls_scores)
            logger.info(
                "[RF-DETR]   %-15s  count=%-4d  min=%.3f  mean=%.3f  max=%.3f",
                cls_name, cls_counts[cls_name], cs.min(), cs.mean(), cs.max(),
            )

    return RFDetectionResult(
        model_key=model_key,
        label=info["label"],
        count=len(bboxes),
        bboxes=bboxes,
        scores=scores,
        classes=classes,
        class_ids=class_ids,
        elapsed=elapsed,
        image_width=img_w,
        image_height=img_h,
    )


def result_to_geojson(
    result: RFDetectionResult,
    geotiff_path: str,
) -> str:
    """Convert an RFDetectionResult to a GeoJSON FeatureCollection.

    Transforms pixel bounding boxes → WGS84 geographic coordinates.
    """
    if not result.bboxes:
        return json.dumps({"type": "FeatureCollection", "features": []})

    with rasterio.open(geotiff_path) as src:
        transform = src.transform

    features = []
    for i, (bbox, score, cls) in enumerate(
        zip(result.bboxes, result.scores, result.classes)
    ):
        x1, y1, x2, y2 = bbox
        # pixel (col, row) → geographic via rasterio transform
        west, north = transform * (x1, y1)
        east, south = transform * (x2, y2)
        west, east = min(west, east), max(west, east)
        south, north = min(south, north), max(south, north)

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [west, north], [east, north],
                    [east, south], [west, south],
                    [west, north],
                ]],
            },
            "properties": {
                "segment": cls,
                "source": "rf-detr",
                "model": result.model_key,
                "score": round(score, 4),
                "class": cls,
                "instance_id": i + 1,
            },
        })

    return json.dumps({
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "generator": "LLMSat/RF-DETR",
            "model": result.model_key,
            "total_features": len(features),
        },
    }, indent=2)


def pixel_bboxes_to_geo(
    bboxes: list[list[float]],
    geotiff_path: str,
) -> list[list[float]]:
    """Convert pixel-space bboxes [x1,y1,x2,y2] to WGS84 [west,south,east,north].

    Used by the hybrid pipeline to feed RF-DETR detections into SAM3.
    """
    with rasterio.open(geotiff_path) as src:
        transform = src.transform

    geo_boxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        west, north = transform * (x1, y1)
        east, south = transform * (x2, y2)
        west, east = min(west, east), max(west, east)
        south, north = min(south, north), max(south, north)
        geo_boxes.append([west, south, east, north])

    return geo_boxes
