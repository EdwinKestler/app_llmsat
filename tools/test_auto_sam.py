# tools/test_auto_sam.py
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.config import PipelineConfig, load_config
from pipeline.pipeline import run_pipeline

if __name__ == "__main__":
    cfg = load_config(
        bbox=(-90.015147, 14.916566, -90.010159, 14.919471),
        zoom=18,
        tms_source="Satellite",  # Changed from imagery_source to match PipelineConfig
        data_dir="data",
        out_dir="output",  # Changed from output_dir to match PipelineConfig
        checkpoints_dir="checkpoints",  # Changed from model_dir to match PipelineConfig
        sam_checkpoint="sam_vit_h_4b8939.pth",
        device="cuda" if sys.platform != "win32" else "cpu",  # Fallback to CPU on Windows if needed
        box_threshold=0.24,
        text_threshold=0.24,
    )
    out = run_pipeline(cfg, text_prompts=["water"])  # Added text_prompts
    print("✅ OK:", out)
    for k, v in out.items():
        if isinstance(v, str) and os.path.exists(v):
            print("  -", k, "->", v)