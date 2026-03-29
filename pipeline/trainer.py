"""RF-DETR training wrapper — fine-tune, evaluate, and register custom models.

Wraps the RF-DETR training API with progress reporting, evaluation,
and model registration for use in the LLMSat detection pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional, Callable

logger = logging.getLogger("llmsat.trainer")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHECKPOINTS_DIR = os.path.join(_PROJECT_ROOT, "checkpoints")
_CUSTOM_MODELS_PATH = os.path.join(_CHECKPOINTS_DIR, "custom_models.json")


def check_training_deps() -> tuple[bool, str]:
    """Check if RF-DETR training dependencies are installed.

    Returns (ok, message).
    """
    try:
        import rfdetr  # noqa: F401
    except ImportError:
        return False, "rfdetr is not installed. Run: pip install rfdetr"

    try:
        import pytorch_lightning  # noqa: F401
    except ImportError:
        return False, 'Training extras not installed. Run: pip install "rfdetr[train,loggers]"'

    import torch
    if not torch.cuda.is_available():
        return False, "No CUDA GPU detected. Training requires a GPU."

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb < 6:
        return False, f"GPU has {vram_gb:.1f} GB VRAM. RF-DETR training needs at least 8 GB."

    return True, f"GPU ready: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB VRAM)"


def train_rfdetr(
    dataset_dir: str,
    output_dir: str,
    base_model: str = "rfdetr_base",
    epochs: int = 50,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    lr: float = 1e-4,
    early_stopping: bool = True,
    early_stopping_patience: int = 10,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> dict:
    """Fine-tune an RF-DETR model on a COCO-format dataset.

    Parameters
    ----------
    dataset_dir : str
        Path to COCO dataset (must contain ``images/train``, ``images/val``,
        ``annotations/train.json``, ``annotations/val.json``).
    output_dir : str
        Where to save checkpoints and logs.
    base_model : str
        Base model to fine-tune ("rfdetr_base" or "rfdetr_large").
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size per GPU.
    grad_accum_steps : int
        Gradient accumulation steps (effective batch = batch_size * grad_accum_steps).
    lr : float
        Learning rate.
    early_stopping : bool
        Enable early stopping on validation mAP.
    early_stopping_patience : int
        Epochs without improvement before stopping.
    progress_callback : callable, optional
        ``fn(message, fraction)`` called during training for UI updates.

    Returns
    -------
    dict
        ``{checkpoint_path, best_map50, epochs_trained, elapsed_seconds, error}``
    """
    import rfdetr

    os.makedirs(output_dir, exist_ok=True)

    # Select base model class
    model_classes = {
        "rfdetr_base": rfdetr.RFDETRBase,
        "rfdetr_large": rfdetr.RFDETRLarge,
    }
    model_cls = model_classes.get(base_model)
    if model_cls is None:
        return {"error": f"Unknown base model: {base_model}"}

    # Load base model from local checkpoint if available
    from pipeline.rfdetector import _CHECKPOINTS_DIR, RFDETR_MODELS
    base_info = RFDETR_MODELS.get(base_model, {})
    weights_file = base_info.get("weights", "")
    local_weights = os.path.join(_CHECKPOINTS_DIR, weights_file)

    kwargs = {}
    if os.path.exists(local_weights):
        kwargs["pretrain_weights"] = local_weights
        logger.info("[Trainer] Fine-tuning from local checkpoint: %s", local_weights)

    if progress_callback:
        progress_callback("Loading base model...", 0.0)

    t0 = time.time()
    try:
        model = model_cls(**kwargs)
    except Exception as e:
        return {"error": f"Failed to load base model: {e}"}

    if progress_callback:
        progress_callback("Starting training...", 0.05)

    # Read class names from dataset metadata
    meta_path = os.path.join(os.path.dirname(dataset_dir), "metadata.json")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(dataset_dir, "..", "metadata.json")
    class_names = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        class_names = meta.get("class_names")

    # Train
    try:
        model.train(
            dataset_dir=dataset_dir,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            lr=lr,
            output_dir=output_dir,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
            class_names=class_names,
        )
    except Exception as e:
        return {"error": f"Training failed: {e}"}

    elapsed = time.time() - t0

    # Find best checkpoint
    best_ckpt = _find_best_checkpoint(output_dir)
    if best_ckpt is None:
        return {"error": "Training completed but no checkpoint found"}

    # Copy best checkpoint to checkpoints/ dir
    final_name = os.path.basename(output_dir) + ".pth"
    final_path = os.path.join(_CHECKPOINTS_DIR, final_name)
    import shutil
    shutil.copy2(best_ckpt, final_path)

    result = {
        "checkpoint_path": final_path,
        "best_map50": _read_best_map(output_dir),
        "epochs_trained": epochs,
        "elapsed_seconds": elapsed,
        "error": None,
    }

    if progress_callback:
        progress_callback("Training complete!", 1.0)

    logger.info(
        "[Trainer] training complete  checkpoint=%s  mAP@50=%.3f  "
        "elapsed=%.0fs  epochs=%d",
        final_path, result["best_map50"] or 0, elapsed, epochs,
    )
    return result


def _find_best_checkpoint(output_dir: str) -> Optional[str]:
    """Find the best checkpoint file in a training output directory."""
    candidates = [
        os.path.join(output_dir, "best.pth"),
        os.path.join(output_dir, "checkpoint_best.pth"),
        os.path.join(output_dir, "checkpoints", "best.pth"),
    ]
    # Also search for any .pth file with "best" in the name
    if os.path.isdir(output_dir):
        for f in os.listdir(output_dir):
            if f.endswith(".pth") and "best" in f.lower():
                candidates.append(os.path.join(output_dir, f))
        # Fallback: last .pth file
        pth_files = sorted(
            [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".pth")],
            key=os.path.getmtime,
        )
        candidates.extend(pth_files)

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _read_best_map(output_dir: str) -> Optional[float]:
    """Try to read best mAP@50 from training logs."""
    # RF-DETR saves metrics in various formats; try common ones
    for fname in ["metrics.json", "results.json", "log.json"]:
        path = os.path.join(output_dir, fname)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data.get("best_map50", data.get("map50"))
                if isinstance(data, list) and data:
                    last = data[-1]
                    return last.get("map50", last.get("val/map50"))
            except Exception:
                pass
    return None


# ── Custom model registration ────────────────────────────────────────

def register_custom_model(
    name: str,
    checkpoint_path: str,
    class_names: list[str],
    base_model: str = "rfdetr_base",
    gsd_cm: float = 0.0,
    description: str = "",
) -> None:
    """Register a fine-tuned model so it appears in the detection dropdown.

    Writes to ``checkpoints/custom_models.json``.
    """
    os.makedirs(_CHECKPOINTS_DIR, exist_ok=True)

    registry = load_custom_models()
    registry[name] = {
        "checkpoint_path": checkpoint_path,
        "base_model": base_model,
        "class_names": class_names,
        "gsd_cm": gsd_cm,
        "description": description or f"Custom model: {', '.join(class_names)}",
    }

    with open(_CUSTOM_MODELS_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    logger.info("[Trainer] registered custom model '%s' → %s", name, checkpoint_path)


def load_custom_models() -> dict:
    """Load the custom model registry."""
    if not os.path.exists(_CUSTOM_MODELS_PATH):
        return {}
    try:
        with open(_CUSTOM_MODELS_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def unregister_custom_model(name: str) -> bool:
    """Remove a model from the registry. Returns True if removed."""
    registry = load_custom_models()
    if name in registry:
        del registry[name]
        with open(_CUSTOM_MODELS_PATH, "w") as f:
            json.dump(registry, f, indent=2)
        return True
    return False


def list_all_detection_models() -> dict:
    """Return all detection models (pretrained + custom) for the UI dropdown."""
    from pipeline.rfdetector import RFDETR_MODELS

    models = {}
    # Pretrained RF-DETR models
    for key, info in RFDETR_MODELS.items():
        models[key] = {
            "label": info["label"],
            "icon": info["icon"],
            "description": info["description"],
            "type": "pretrained",
        }

    # Custom fine-tuned models
    for name, info in load_custom_models().items():
        classes_str = ", ".join(info.get("class_names", []))
        models[f"custom_{name}"] = {
            "label": f"{name} (custom)",
            "icon": "🎓",
            "description": info.get("description", classes_str),
            "type": "custom",
            "checkpoint_path": info["checkpoint_path"],
            "base_model": info["base_model"],
            "class_names": info.get("class_names", []),
        }

    return models
