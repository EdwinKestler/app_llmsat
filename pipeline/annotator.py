"""Annotation helpers — merge, edit, and render detection annotations.

Combines SAM3 instance masks, RF-DETR detections, and Open Buildings
polygons into a unified editable annotation format for dataset building.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("llmsat.annotator")

# Colors for up to 12 classes (looped if more)
CLASS_COLORS = [
    (0, 255, 0),    (255, 50, 50),   (50, 50, 255),
    (255, 255, 0),  (255, 0, 255),   (0, 255, 255),
    (255, 128, 0),  (128, 0, 255),   (0, 128, 255),
    (128, 255, 0),  (255, 0, 128),   (0, 255, 128),
]


def merge_detections(
    sam3_annotations: list[dict] | None = None,
    rfdetr_annotations: list[dict] | None = None,
    buildings_annotations: list[dict] | None = None,
    iou_merge_threshold: float = 0.5,
) -> list[dict]:
    """Merge annotations from multiple sources into a unified list.

    Each annotation is a dict with:
    - ``bbox``: [x, y, w, h] in pixel coords (COCO format)
    - ``class``: class name string
    - ``score``: confidence (0-1)
    - ``source``: origin ("sam3_mask", "rfdetr", "open_buildings")
    - ``accepted``: bool (default True, user can reject)
    - ``id``: unique integer

    When boxes from different sources overlap (IoU > threshold),
    the higher-confidence one is kept.
    """
    all_anns = []
    for source_anns in [sam3_annotations, rfdetr_annotations, buildings_annotations]:
        if source_anns:
            all_anns.extend(source_anns)

    if not all_anns:
        return []

    # Sort by score descending
    all_anns.sort(key=lambda a: a.get("score", 0), reverse=True)

    # Greedy NMS-style deduplication across sources
    kept = []
    for ann in all_anns:
        if _is_duplicate(ann, kept, iou_merge_threshold):
            continue
        ann["accepted"] = ann.get("accepted", True)
        ann["id"] = len(kept) + 1
        kept.append(ann)

    logger.info(
        "[Annotator] merged %d annotations → %d after dedup (IoU=%.2f)",
        len(all_anns), len(kept), iou_merge_threshold,
    )
    return kept


def _bbox_iou(a: list[float], b: list[float]) -> float:
    """Compute IoU between two COCO-format bboxes [x, y, w, h]."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih

    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def _is_duplicate(
    ann: dict, kept: list[dict], threshold: float
) -> bool:
    """Check if ann overlaps with any already-kept annotation."""
    for k in kept:
        if _bbox_iou(ann["bbox"], k["bbox"]) > threshold:
            return True
    return False


def render_annotated_tile(
    rgb: np.ndarray,
    annotations: list[dict],
    class_names: Optional[list[str]] = None,
    show_rejected: bool = True,
) -> np.ndarray:
    """Draw bounding boxes and labels on a tile image.

    Parameters
    ----------
    rgb : ndarray
        (H, W, 3) uint8 image.
    annotations : list[dict]
        Annotations with ``bbox``, ``class``, ``score``, ``accepted``.
    class_names : list[str], optional
        Ordered class names for consistent coloring.
    show_rejected : bool
        If True, draw rejected annotations with dashed red outline.

    Returns
    -------
    ndarray
        Annotated image.
    """
    pil_img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(pil_img, "RGBA")

    # Build color map
    if class_names is None:
        class_names = sorted(set(a["class"] for a in annotations))
    color_map = {
        name: CLASS_COLORS[i % len(CLASS_COLORS)]
        for i, name in enumerate(class_names)
    }

    for ann in annotations:
        bx, by, bw, bh = ann["bbox"]
        x1, y1, x2, y2 = int(bx), int(by), int(bx + bw), int(by + bh)
        accepted = ann.get("accepted", True)

        if not accepted and not show_rejected:
            continue

        if accepted:
            color = color_map.get(ann["class"], (200, 200, 200))
            outline = (*color, 220)
            fill = (*color, 30)
        else:
            outline = (255, 0, 0, 150)
            fill = (255, 0, 0, 15)

        draw.rectangle([x1, y1, x2, y2], outline=outline, fill=fill, width=2)

        # Label
        label = f"{ann['class']} {ann.get('score', 0):.2f}"
        if not accepted:
            label = f"[X] {label}"
        # Draw label background
        text_bbox = draw.textbbox((x1, y1 - 12), label)
        draw.rectangle(text_bbox, fill=(*outline[:3], 180))
        draw.text((x1, y1 - 12), label, fill=(255, 255, 255, 255))

    return np.array(pil_img)


def annotations_to_summary(annotations: list[dict]) -> dict:
    """Compute summary statistics for a list of annotations.

    Returns
    -------
    dict
        ``{total, accepted, rejected, classes: {name: count}, sources: {name: count}}``
    """
    from collections import Counter

    accepted = [a for a in annotations if a.get("accepted", True)]
    rejected = [a for a in annotations if not a.get("accepted", True)]

    return {
        "total": len(annotations),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "classes": dict(Counter(a["class"] for a in accepted)),
        "sources": dict(Counter(a.get("source", "unknown") for a in accepted)),
    }
