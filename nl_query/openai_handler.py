# Modified file app_stremlit/nl_query/openai_handler.py
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable, Dict, List, Tuple, Optional

import geopandas as gpd
import pandas as pd
import altair as alt

from pipeline.config import PipelineConfig, load_config  # Updated import at line ~23
from pipeline.pipeline import run_pipeline

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ... (rest of the file unchanged)
SEGMENT_KEYWORDS: Dict[str, set[str]] = {
    "water": {"water", "river", "lake", "pond", "sea", "ocean"},
    "tree": {"tree", "trees", "forest", "woodland", "vegetation"},
    "building": {"building", "buildings", "house", "houses", "structure"},
    "road": {"road", "roads", "street", "highway", "path"},
}

def parse_user_text(question: str, *, client: Optional[OpenAI] = None) -> List[str]:
    question_lower = question.lower()
    if client is not None:
        try:
            response = client.responses.create(
                model="gpt-4o-mini",
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
    files = sorted(Path(out_dir).glob(f"segment_{segment}_*.gpkg"))
    return files[-1] if files else None

def fetch_segment_data(segment: str, out_dir: str) -> Tuple[gpd.GeoDataFrame, float]:
    gpkg = _segment_file(out_dir, segment)
    if gpkg is None:
        raise FileNotFoundError(f"No GeoPackage found for segment '{segment}' in '{out_dir}'")
    gdf = gpd.read_file(gpkg)
    area = gdf.geometry.area.sum()
    return gdf, area

def ask(question: str, bbox: Iterable[float], *, out_dir: str = "data", use_altair: bool = True):
    segments = parse_user_text(question)
    if not segments:
        raise ValueError("No known segment types referenced in question")

    config = load_config(bbox=bbox, out_dir=out_dir)

    if any(_segment_file(config.out_dir, s) is None for s in segments):
        for seg in segments:
            seg_out_dir = os.path.join(config.out_dir, seg)
            seg_config = PipelineConfig(**{**config.__dict__, "out_dir": seg_out_dir})
            run_pipeline(seg_config, text_prompts=[seg])

    data = []
    for seg in segments:
        gdf, area = fetch_segment_data(seg, os.path.join(config.out_dir, seg))
        data.append({"segment": seg, "area_m2": area})

    df = pd.DataFrame(data)
    if use_altair:
        chart = alt.Chart(df).mark_bar().encode(x="segment", y="area_m2")
    else:
        import plotly.express as px
        chart = px.bar(df, x="segment", y="area_m2")
    return chart, df

def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Query pipeline outputs using natural language")
    parser.add_argument("question", help="Natural language question")
    parser.add_argument("bbox", nargs=4, type=float, help="Bounding box xmin ymin xmax ymax")
    parser.add_argument("--out-dir", default="data", help="Pipeline output directory")
    parser.add_argument(
        "--plotly", action="store_true", help="Use Plotly instead of Altair for the chart"
    )
    args = parser.parse_args()

    chart, df = ask(args.question, args.bbox, out_dir=args.out_dir, use_altair=not args.plotly)
    print(df)
    if hasattr(chart, "show"):
        chart.show()
    else:
        chart.display()

if __name__ == "__main__":
    _main()