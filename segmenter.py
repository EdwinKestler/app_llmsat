# app_stremlit/segmenter.py
from __future__ import annotations
import os
import inspect
from typing import List, Optional, Any

import torch

# Prefer real 'samgeo' (segment-geospatial pins), fail clearly if missing.
try:
    import samgeo
    from samgeo.text_sam import LangSAM
except Exception as e:
    raise RuntimeError(
        "segment-geospatial must be installed (we pin 0.12.0). "
        "Run: pip install --force-reinstall segment-geospatial==0.12.0"
    ) from e


def _is_sam2_checkpoint(path: Optional[str]) -> bool:
    if not path:
        return False
    name = os.path.basename(path).lower()
    return ("sam2" in name) or ("hiera" in name)


def _filter_kwargs(fn: Any, kwargs: dict) -> dict:
    try:
        sig = inspect.signature(fn)
        return {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}
    except Exception:
        return {}


def _maybe_to_device(obj: Any, device: str) -> None:
    """Try to move any torch.nn.Module-ish obj to device."""
    try:
        if hasattr(obj, "to") and callable(obj.to):
            obj.to(device)
            return
    except Exception:
        pass
    for attr in ("model", "module"):
        try:
            m = getattr(obj, attr, None)
            if m is not None and hasattr(m, "to"):
                m.to(device)
        except Exception:
            pass


def _ensure_initialized(langsam: LangSAM, *, device: str, checkpoint: Optional[str]) -> None:
    """
    Ensure langsam has a model bound and a non-None processor.
    We try common init methods across versions; then verify and raise if still missing.
    """
    # If processor already present, we’re good.
    if getattr(langsam, "processor", None) is not None:
        return

    model_type = "sam2" if _is_sam2_checkpoint(checkpoint) else "vit_h"
    attempts = [
        ("load_model",    dict(model_type=model_type, checkpoint=checkpoint, device=device)),
        ("set_model",     dict(model_type=model_type, checkpoint=checkpoint, device=device)),
        ("initialize",    dict(model_type=model_type, checkpoint=checkpoint, device=device)),
        ("build_model",   dict(model_type=model_type, checkpoint=checkpoint, device=device)),
        ("load_sam2",     dict(checkpoint=checkpoint, device=device)),
        ("load_sam",      dict(checkpoint=checkpoint, device=device, model_type=model_type)),
        ("load_processor",dict(model_type=model_type)),
        ("build_processor",dict(model_type=model_type)),
        ("init_processor",dict(model_type=model_type)),
        ("initialize_processor",dict(model_type=model_type)),
        ("create_processor",dict(model_type=model_type)),
    ]
    # Try each method if present, passing only compatible kwargs.
    for name, kw in attempts:
        if hasattr(langsam, name):
            try:
                fn = getattr(langsam, name)
                fn(**_filter_kwargs(fn, kw))
                # Exit early if processor is now ready
                if getattr(langsam, "processor", None) is not None:
                    break
            except TypeError:
                continue
            except Exception:
                # keep trying next method
                continue

    # Alias any model handle to a common attribute if some code path expects it
    # (older internals sometimes look for .sam).
    if not hasattr(langsam, "sam"):
        for cand in ("sam2", "model", "_sam", "sam_model", "sam_module"):
            m = getattr(langsam, cand, None)
            if m is not None:
                setattr(langsam, "sam", m)
                break

    # Nudge the bound model (if any) to the requested device
    try:
        m = getattr(langsam, "sam", None)
        if m is None:
            # try other common holders
            m = getattr(langsam, "model", None) or getattr(langsam, "sam2", None)
        if m is not None:
            _maybe_to_device(m, device)
    except Exception:
        pass

    # Final sanity: processor must be non-None now
    if getattr(langsam, "processor", None) is None:
        ver = getattr(samgeo, "__version__", "<unknown>")
        raise RuntimeError(
            "LangSAM processor was not initialized by the installed segment-geospatial "
            f"(samgeo=={ver}). Try re-installing the pinned combo:\n"
            "  pip install --force-reinstall groundingdino-py==0.4.0\n"
            "  pip install --force-reinstall segment-geospatial==0.12.0\n"
            "Also ensure transformers==4.55.0 and opencv-python-headless==4.12.0.88 are installed.\n"
            "If this persists, we can switch to a direct SAM2 inference path (bypassing LangSAM)."
        )


class Segmenter:
    def __init__(
        self,
        device: str = "cuda",
        model_dir: str = "checkpoints",
        sam2_checkpoint: str = "sam2_hiera_l.pt",
        box_threshold: float = 0.24,
        text_threshold: float = 0.24,
    ):
        # Device selection with CUDA fallback
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.model_dir = model_dir
        self.sam2_checkpoint = (
            sam2_checkpoint if os.path.isabs(sam2_checkpoint) else os.path.join(model_dir, sam2_checkpoint)
        )
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        if self.device == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True

        ckpt: Optional[str] = self.sam2_checkpoint if os.path.exists(self.sam2_checkpoint) else None

        # Prefer the 0.12.0 signature; fall back gracefully.
        try:
            self.langsam = LangSAM(model="sam2", device=self.device, checkpoint=ckpt)
        except TypeError:
            # Older/newer variant: use model_type/checkpoint only
            try:
                self.langsam = LangSAM(model_type="sam2", checkpoint=ckpt)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to construct LangSAM with available signatures. Check samgeo version. Detail: {e}"
                )

        # Force initialize model/processor across versions
        _ensure_initialized(self.langsam, device=self.device, checkpoint=ckpt)

    def run_text_segmentation(self, image_path: str, text_prompts: List[str]):
        """
        Call LangSAM.predict across API variants:
          1) predict(image, text_prompt, ...)
          2) predict(image=..., text_prompt=..., ...)
          3) predict(image_path=..., text_prompts=[...], ...)
        """
        prompts_str = ", ".join(text_prompts) if isinstance(text_prompts, list) else str(text_prompts)

        # Try positional (older builds)
        try:
            return self.langsam.predict(
                image_path,                          # positional image
                prompts_str,                         # positional text_prompt
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
        except TypeError:
            pass

        # Try keyword (image + text_prompt)
        try:
            return self.langsam.predict(
                image=image_path,
                text_prompt=prompts_str,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
        except TypeError:
            pass

        # Try newer style (image_path + text_prompts list)
        try:
            return self.langsam.predict(
                image_path=image_path,
                text_prompts=text_prompts if isinstance(text_prompts, list) else [str(text_prompts)],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
        except TypeError as e:
            # Surface signature to help pin versions if needed
            try:
                sig = str(inspect.signature(self.langsam.predict))
            except Exception:
                sig = "<unavailable>"
            raise TypeError(
                "LangSAM.predict signature not recognized. "
                f"Tried positional and keyword variants. Last error: {e}. "
                f"Detected predict signature: {sig}"
            )
