import argparse
from typing import List
from pipeline import load_config, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the geospatial processing pipeline"
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        required=True,
        help="Bounding box coordinates in EPSG:4326",
    )
    parser.add_argument(
        "--out-dir",
        default="output",
        help="Directory where outputs will be written",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Text prompt for LangSAM. Can be repeated.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run models on (cuda or cpu)",
    )
    args = parser.parse_args()

    bbox = tuple(args.bbox)
    prompts: List[str] = args.prompt
    config = load_config(bbox=bbox, out_dir=args.out_dir, device=args.device)
    outputs = run_pipeline(config, prompts)

    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()