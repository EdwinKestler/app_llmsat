"""Dataset builder — convert LLMSat segmentation outputs to COCO training datasets.

Converts SAM3 instance masks, RF-DETR detections, and Open Buildings polygons
into COCO JSON format suitable for RF-DETR fine-tuning.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

logger = logging.getLogger("llmsat.dataset_builder")

_DATASETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"
)


# ── Tiling ───────────────────────────────────────────────────────────

def tile_geotiff(
    geotiff_path: str,
    tile_size: int = 640,
    overlap: float = 0.1,
    min_content_ratio: float = 0.05,
) -> list[dict]:
    """Split a GeoTIFF into tiles for training.

    Parameters
    ----------
    geotiff_path : str
        Path to the source GeoTIFF.
    tile_size : int
        Tile width/height in pixels.
    overlap : float
        Overlap fraction between adjacent tiles (0-0.5).
    min_content_ratio : float
        Skip tiles where non-zero pixels are below this ratio (filters black edges).

    Returns
    -------
    list of dict
        Each dict: ``{rgb: ndarray, x_off: int, y_off: int, width: int, height: int}``.
    """
    with rasterio.open(geotiff_path) as src:
        img_h, img_w = src.height, src.width
        stride = int(tile_size * (1 - overlap))

    tiles = []
    for y_off in range(0, img_h, stride):
        for x_off in range(0, img_w, stride):
            w = min(tile_size, img_w - x_off)
            h = min(tile_size, img_h - y_off)
            if w < tile_size // 2 or h < tile_size // 2:
                continue

            with rasterio.open(geotiff_path) as src:
                window = Window(x_off, y_off, w, h)
                bands = src.read([1, 2, 3], window=window)

            rgb = np.moveaxis(bands, 0, -1)

            # Skip mostly-black tiles
            content = np.count_nonzero(rgb.sum(axis=2)) / (w * h)
            if content < min_content_ratio:
                continue

            # Pad to tile_size if at edge
            if w < tile_size or h < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=rgb.dtype)
                padded[:h, :w] = rgb
                rgb = padded

            tiles.append({
                "rgb": rgb,
                "x_off": x_off,
                "y_off": y_off,
                "width": w,
                "height": h,
            })

    logger.info(
        "[DatasetBuilder] tiled %s → %d tiles (%dx%d, stride=%d)",
        geotiff_path, len(tiles), tile_size, tile_size, stride,
    )
    return tiles


# ── Mask-to-bbox conversion ──────────────────────────────────────────

def instance_masks_to_bboxes(
    mask_path: str,
    class_name: str = "object",
) -> list[dict]:
    """Convert a SAM3 instance mask raster to bounding box annotations.

    Each unique non-zero value in the mask becomes one bounding box.

    Returns
    -------
    list of dict
        ``[{bbox: [x, y, w, h], class: str, score: float}, ...]``
        where bbox is in pixel coordinates (COCO format: x, y, width, height).
    """
    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    annotations = []
    for instance_id in np.unique(mask):
        if instance_id == 0:
            continue
        ys, xs = np.where(mask == instance_id)
        if len(xs) == 0:
            continue
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        if w < 3 or h < 3:
            continue
        annotations.append({
            "bbox": [x_min, y_min, w, h],
            "class": class_name,
            "score": 1.0,
            "source": "sam3_mask",
        })

    return annotations


def rfdetr_bboxes_to_annotations(
    bboxes: list[list[float]],
    scores: list[float],
    classes: list[str],
) -> list[dict]:
    """Convert RF-DETR detection results to annotation format.

    Input bboxes are [x1, y1, x2, y2] in pixel coords.
    Output bbox is [x, y, w, h] (COCO format).
    """
    annotations = []
    for bbox, score, cls in zip(bboxes, scores, classes):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        if w < 1 or h < 1:
            continue
        annotations.append({
            "bbox": [x1, y1, w, h],
            "class": cls,
            "score": float(score),
            "source": "rfdetr",
        })
    return annotations


# ── Annotations ↔ tile mapping ───────────────────────────────────────

def clip_annotations_to_tile(
    annotations: list[dict],
    x_off: int,
    y_off: int,
    tile_w: int,
    tile_h: int,
    min_visible: float = 0.3,
) -> list[dict]:
    """Clip global-coordinate annotations to a tile, discarding those mostly outside.

    Parameters
    ----------
    min_visible : float
        Minimum fraction of the original bbox area that must be inside the tile.
    """
    clipped = []
    for ann in annotations:
        bx, by, bw, bh = ann["bbox"]
        # Shift to tile-local coordinates
        lx = bx - x_off
        ly = by - y_off

        # Clip to tile bounds
        cx1 = max(0, lx)
        cy1 = max(0, ly)
        cx2 = min(tile_w, lx + bw)
        cy2 = min(tile_h, ly + bh)

        cw = cx2 - cx1
        ch = cy2 - cy1
        if cw < 2 or ch < 2:
            continue

        # Check how much of the original box is visible
        orig_area = bw * bh
        clipped_area = cw * ch
        if orig_area > 0 and (clipped_area / orig_area) < min_visible:
            continue

        clipped.append({
            **ann,
            "bbox": [float(cx1), float(cy1), float(cw), float(ch)],
        })

    return clipped


# ── COCO dataset creation ────────────────────────────────────────────

def create_dataset(
    tiles: list[dict],
    tile_annotations: list[list[dict]],
    class_names: list[str],
    project_name: str,
    train_split: float = 0.8,
    gsd_cm: float = 0.0,
    source_bbox: Optional[list[float]] = None,
) -> str:
    """Write tiles and annotations as a COCO-format dataset.

    Parameters
    ----------
    tiles : list[dict]
        Output of :func:`tile_geotiff`.
    tile_annotations : list[list[dict]]
        Per-tile annotation lists (parallel to ``tiles``).
    class_names : list[str]
        Ordered list of class names.
    project_name : str
        Dataset name (used as directory name under ``datasets/``).
    train_split : float
        Fraction of tiles for training (rest goes to val).
    gsd_cm : float
        Ground sampling distance in cm/px.
    source_bbox : list[float], optional
        Source bounding box [west, south, east, north].

    Returns
    -------
    str
        Path to the dataset directory.
    """
    dataset_dir = os.path.join(_DATASETS_DIR, project_name)
    images_train = os.path.join(dataset_dir, "images", "train")
    images_val = os.path.join(dataset_dir, "images", "val")
    ann_dir = os.path.join(dataset_dir, "annotations")
    for d in [images_train, images_val, ann_dir]:
        os.makedirs(d, exist_ok=True)

    # Build class name → id mapping (1-indexed for COCO)
    class_map = {name: i + 1 for i, name in enumerate(class_names)}
    coco_categories = [
        {"id": i + 1, "name": name} for i, name in enumerate(class_names)
    ]

    # Shuffle tiles deterministically for reproducible splits
    n_tiles = len(tiles)
    indices = list(range(n_tiles))
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    split_idx = int(n_tiles * train_split)
    train_indices = set(indices[:split_idx])

    train_images = []
    train_annotations = []
    val_images = []
    val_annotations = []
    ann_id = 1

    for idx in range(n_tiles):
        tile = tiles[idx]
        anns = tile_annotations[idx] if idx < len(tile_annotations) else []
        is_train = idx in train_indices

        # Save tile image
        filename = f"tile_{idx:04d}.png"
        dest_dir = images_train if is_train else images_val
        Image.fromarray(tile["rgb"]).save(os.path.join(dest_dir, filename))

        image_info = {
            "id": idx + 1,
            "file_name": filename,
            "width": tile["rgb"].shape[1],
            "height": tile["rgb"].shape[0],
        }

        img_list = train_images if is_train else val_images
        ann_list = train_annotations if is_train else val_annotations
        img_list.append(image_info)

        for ann in anns:
            cls_name = ann["class"]
            cat_id = class_map.get(cls_name)
            if cat_id is None:
                continue
            bx, by, bw, bh = ann["bbox"]
            ann_list.append({
                "id": ann_id,
                "image_id": idx + 1,
                "category_id": cat_id,
                "bbox": [round(bx, 1), round(by, 1), round(bw, 1), round(bh, 1)],
                "area": round(bw * bh, 1),
                "iscrowd": 0,
            })
            ann_id += 1

    # Write COCO JSON files
    for split, images, annotations in [
        ("train", train_images, train_annotations),
        ("val", val_images, val_annotations),
    ]:
        coco = {
            "images": images,
            "annotations": annotations,
            "categories": coco_categories,
        }
        path = os.path.join(ann_dir, f"{split}.json")
        with open(path, "w") as f:
            json.dump(coco, f, indent=2)

    # Write metadata
    metadata = {
        "project_name": project_name,
        "created": datetime.now(timezone.utc).isoformat(),
        "class_names": class_names,
        "gsd_cm": gsd_cm,
        "source_bbox": source_bbox,
        "n_tiles": n_tiles,
        "n_train": len(train_images),
        "n_val": len(val_images),
        "n_annotations_train": len(train_annotations),
        "n_annotations_val": len(val_annotations),
    }
    with open(os.path.join(dataset_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "[DatasetBuilder] created dataset '%s': %d train + %d val images, "
        "%d + %d annotations, %d classes",
        project_name, len(train_images), len(val_images),
        len(train_annotations), len(val_annotations), len(class_names),
    )
    return dataset_dir


# ── Dataset management ───────────────────────────────────────────────

def list_datasets() -> list[dict]:
    """List all datasets in the datasets/ directory."""
    if not os.path.isdir(_DATASETS_DIR):
        return []
    datasets = []
    for name in sorted(os.listdir(_DATASETS_DIR)):
        meta_path = os.path.join(_DATASETS_DIR, name, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            datasets.append(meta)
    return datasets


def load_dataset_metadata(project_name: str) -> Optional[dict]:
    """Load metadata for a specific dataset."""
    meta_path = os.path.join(_DATASETS_DIR, project_name, "metadata.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def dataset_dir(project_name: str) -> str:
    """Return the absolute path to a dataset directory."""
    return os.path.join(_DATASETS_DIR, project_name)


def delete_dataset(project_name: str) -> bool:
    """Delete a dataset directory. Returns True if deleted."""
    path = os.path.join(_DATASETS_DIR, project_name)
    if os.path.isdir(path):
        shutil.rmtree(path)
        return True
    return False
