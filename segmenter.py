# app_stremlit/segmenter.py
from __future__ import annotations
import os
from typing import List, Optional

# Prefer the real 'samgeo' if installed; otherwise fall back to our stub
try:
    from samgeo import SamGeo  # type: ignore
except Exception:
    SamGeo = None  # Optional in this codepath if you only use LangSAM

# MODIFIED (~lines 1–40): attempt real LangSAM first, then stub
try:
    from samgeo.text_sam import LangSAM  # type: ignore
except Exception:
    from samgeo_stub.text_sam import LangSAM  # local stub fallback

import torch


class Segmenter:
    def __init__(
        self,
        device: str = "cuda",
        model_dir: str = "checkpoints",
        sam2_checkpoint: str = "sam2_hiera_l.pt",
        box_threshold: float = 0.24,
        text_threshold: float = 0.24,
    ):
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.model_dir = model_dir
        self.sam2_checkpoint = os.path.join(model_dir, sam2_checkpoint)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Optional perf tuning
        if self.device == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True

        self.langsam = LangSAM(
            model="sam2",
            device=self.device,
            checkpoint=self.sam2_checkpoint if os.path.exists(self.sam2_checkpoint) else None,
        )

    def run_text_segmentation(self, image_path: str, text_prompts: List[str]):
        return self.langsam.predict(
            image_path=image_path,
            text_prompts=text_prompts,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
