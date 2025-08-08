# app_stremlit/samgeo_stub/text_sam.py
# Minimal stub to avoid crashes if real `samgeo.text_sam.LangSAM` is not installed.
# This DOES NOT perform real segmentation.

from typing import List, Optional

class LangSAM:
    def __init__(self, model="sam2", device="cpu", checkpoint: Optional[str] = None, **kwargs):
        self.model = model
        self.device = device
        self.checkpoint = checkpoint

    def predict(self, image_path: str, text_prompts: List[str], box_threshold: float = 0.24, text_threshold: float = 0.24):
        # Return an empty mask-like structure for compatibility.
        # Real impl should return masks and metadata.
        return {
            "masks": [],
            "scores": [],
            "labels": text_prompts,
            "meta": {
                "image": image_path,
                "device": self.device,
                "checkpoint": self.checkpoint,
                "stub": True,
            },
        }
