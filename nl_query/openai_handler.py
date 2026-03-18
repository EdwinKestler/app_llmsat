import json
import os
from pathlib import Path
from typing import Iterable, Dict, List, Tuple, Optional

import geopandas as gpd
import pandas as pd
import altair as alt

from pipeline.config import PipelineConfig, load_config
from pipeline.pipeline import run_pipeline

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

SEGMENT_KEYWORDS: Dict[str, set[str]] = {
    "water": {"water", "river", "lake", "pond", "sea", "ocean"},
    "tree": {"tree", "trees", "forest", "woodland", "vegetation"},
    "building": {"building", "buildings", "house", "houses", "structure"},
    "road": {"road", "roads", "street", "highway", "path"},
}


def _estimate_utm_crs(gdf: gpd.GeoDataFrame) -> str:
    """Return an appropriate UTM EPSG code based on the centroid of *gdf*."""
    centroid = gdf.geometry.union_all().centroid
    zone = int((centroid.x + 180) / 6) + 1
    epsg = 32600 + zone if centroid.y >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def parse_user_text(question: str, *, client: Optional[OpenAI] = None) -> List[str]:
    question_lower = question.lower()
    if client is not None:
        try:
            response = client.responses.create(
                model=os.getenv("LLMSAT_OPENAI_MODEL", "gpt-5.4-nano"),
                input=question,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "set_segments",
                            "description": "Extract referenced segment types",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "segments": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    }
                                },
                                "required": ["segments"],
                            },
                        },
                    }
                ],
                tool_choice={"type": "function", "function": {"name": "set_segments"}},
            )
            tool_call = response.output[0].content[0].tool_call
            args = json.loads(tool_call["arguments"])
            segments = args.get("segments", [])
            if segments:
                return [s for s in segments if s in SEGMENT_KEYWORDS]
        except Exception:
            pass

    segments = []
    for segment, words in SEGMENT_KEYWORDS.items():
        if any(w in question_lower for w in words):
            segments.append(segment)
    return segments


def map_keywords_to_segments(keywords: Iterable[str]) -> List[str]:
    segments = []
    for kw in keywords:
        for segment, words in SEGMENT_KEYWORDS.items():
            if kw in words or kw == segment:
                segments.append(segment)
                break
    seen = set()
    return [s for s in segments if not (s in seen or seen.add(s))]


def _segment_file(out_dir: str, segment: str) -> Optional[Path]:
    """Find the GeoPackage for *segment* inside *out_dir*.

    Callers pass the per-segment directory (e.g. ``output/tree``) so we
    look for ``segments.gpkg`` directly there.
    """
    gpkg = Path(out_dir) / "segments.gpkg"
    if gpkg.exists():
        return gpkg
    # Fallback: legacy naming convention
    files = sorted(Path(out_dir).glob(f"segment_{segment}_*.gpkg"))
    return files[-1] if files else None


def fetch_segment_data(segment: str, out_dir: str) -> Tuple[gpd.GeoDataFrame, float]:
    gpkg = _segment_file(out_dir, segment)
    if gpkg is None:
        raise FileNotFoundError(f"No GeoPackage found for segment '{segment}' in '{out_dir}'")
    gdf = gpd.read_file(gpkg)
    if gdf.empty:
        return gdf, 0.0
    utm_crs = _estimate_utm_crs(gdf)
    area = gdf.to_crs(utm_crs).geometry.area.sum()
    return gdf, area


def ask(
    question: str,
    bbox: Iterable[float],
    *,
    config: Optional[PipelineConfig] = None,
    out_dir: str = "data",
    use_altair: bool = True,
):
    segments = parse_user_text(question)
    if not segments:
        raise ValueError("No known segment types referenced in question")

    if config is None:
        config = load_config(bbox=bbox, out_dir=out_dir)

    for seg in segments:
        seg_out_dir = os.path.join(config.out_dir, seg)
        if _segment_file(seg_out_dir, seg) is None:
            seg_config = PipelineConfig(**{**config.__dict__, "out_dir": seg_out_dir})
            run_pipeline(seg_config, text_prompts=[seg])

    data = []
    for seg in segments:
        gdf, area = fetch_segment_data(seg, os.path.join(config.out_dir, seg))
        data.append({"segment": seg, "area_m2": area})

    df = pd.DataFrame(data)
    chart = alt.Chart(df).mark_bar().encode(x="segment", y="area_m2")
    return chart, df, segments


def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Query pipeline outputs using natural language")
    parser.add_argument("question", help="Natural language question")
    parser.add_argument("bbox", nargs=4, type=float, help="Bounding box xmin ymin xmax ymax")
    parser.add_argument("--out-dir", default="data", help="Pipeline output directory")
    args = parser.parse_args()

    chart, df, _segments = ask(args.question, args.bbox, out_dir=args.out_dir)
    print(df)
    chart.display()


if __name__ == "__main__":
    _main()
