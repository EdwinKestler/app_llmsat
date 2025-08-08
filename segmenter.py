# app_stremlit/segmenter.py
from __future__ import annotations
import os
from typing import List
import torch

try:
    from samgeo.text_sam import LangSAM  # real
    _USING_STUB = False
except Exception:
    from samgeo_stub.text_sam import LangSAM  # fallback
    _USING_STUB = True


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
        # if caller passed an absolute path, keep it; else join with model_dir
        self.sam2_checkpoint = sam2_checkpoint if os.path.isabs(sam2_checkpoint) else os.path.join(model_dir, sam2_checkpoint)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        if self.device == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True

        ckpt = self.sam2_checkpoint if os.path.exists(self.sam2_checkpoint) else None
        self.langsam = LangSAM(model="sam2", device=self.device, checkpoint=ckpt)

    def run_text_segmentation(self, image_path: str, text_prompts: List[str]):
        return self.langsam.predict(
            image_path=image_path,
            text_prompts=text_prompts,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
