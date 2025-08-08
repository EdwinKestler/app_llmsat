# app_stremlit/cli.py
import argparse
from typing import List

# CHANGED (~lines 1–10): import from submodules instead of package root
# BEFORE: from pipeline import load_config, run_pipeline
from pipeline.config import load_config
from pipeline.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the geospatial processing pipeline")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        required=True,
        help="Bounding box: west south east north (EPSG:4326)",
    )
    parser.add_argument("--out_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--model_dir", type=str, default="checkpoints", help="Model checkpoint directory")
    parser.add_argument("--sam2_checkpoint", type=str, default="sam2_hiera_l.pt", help="SAM2 checkpoint file name")
    parser.add_argument("--box_threshold", type=float, default=0.24)
    parser.add_argument("--text_threshold", type=float, default=0.24)
    parser.add_argument("--segments", nargs="+", default=["water"], help="Text prompts for segmentation")
    args = parser.parse_args()

    cfg = load_config(
        bbox=tuple(args.bbox),
        out_dir=args.out_dir,
        device=args.device,
        model_dir=args.model_dir,
        sam2_checkpoint=args.sam2_checkpoint,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    run_pipeline(cfg, text_prompts=args.segments)


if __name__ == "__main__":
    main()
