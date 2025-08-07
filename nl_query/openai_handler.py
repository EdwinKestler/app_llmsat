from __future__ import annotations

"""Natural language interface for pipeline outputs via OpenAI models.

This module provides utilities to parse user questions, map the
resulting keywords to known segmentation types and retrieve the
corresponding geometries and areas from the processing pipeline.  An
``ask`` function is exposed for both API and CLI usage which ensures the
pipeline has been executed for the requested bounding box and returns a
simple chart/table of the results.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable, Dict, List, Tuple, Optional

import geopandas as gpd
import pandas as pd
import altair as alt

from pipeline import load_config, run_pipeline, PipelineConfig

try:  # Optional dependency - the package may not be installed.
    from openai import OpenAI
except Exception:  # pragma: no cover - optional import
    OpenAI = None  # type: ignore

# ---------------------------------------------------------------------------
# Keyword mapping utilities
# ---------------------------------------------------------------------------

# Maps canonical segment names to sets of keywords that may appear in a
# user's question.
SEGMENT_KEYWORDS: Dict[str, set[str]] = {
    "water": {"water", "river", "lake", "pond", "sea", "ocean"},
    "tree": {"tree", "trees", "forest", "woodland", "vegetation"},
    "building": {"building", "buildings", "house", "houses", "structure"},
    "road": {"road", "roads", "street", "highway", "path"},
}


def parse_user_text(question: str, *, client: Optional[OpenAI] = None) -> List[str]:
    """Parse a natural language question into segment keywords.

    Parameters
    ----------
    question: str
        Raw user question.
    client: OpenAI, optional
        Optional OpenAI client used for more sophisticated parsing via
        function calling.  When omitted, a simple keyword search is used.

    Returns
    -------
    list[str]
        List of canonical segment names referenced in ``question``.
    """

    question_lower = question.lower()

    # Attempt to use OpenAI function calling if a client is provided.
    if client is not None:
        try:  # pragma: no cover - network call
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
            tool_call = response.output[0].content[0].tool_call  # type: ignore[attr-defined]
            args = json.loads(tool_call["arguments"])  # type: ignore[index]
            segments = args.get("segments", [])
            if segments:
                return [s for s in segments if s in SEGMENT_KEYWORDS]
        except Exception:
            # Fall back to simple keyword search on any error or if the
            # response format is unexpected.
            pass

    segments = []
    for segment, words in SEGMENT_KEYWORDS.items():
        if any(w in question_lower for w in words):
            segments.append(segment)
    return segments


def map_keywords_to_segments(keywords: Iterable[str]) -> List[str]:
    """Map extracted keywords to canonical segment names."""
    segments = []
    for kw in keywords:
        for segment, words in SEGMENT_KEYWORDS.items():
            if kw in words or kw == segment:
                segments.append(segment)
                break
    # Remove duplicates while preserving order
    seen = set()
    return [s for s in segments if not (s in seen or seen.add(s))]


# ---------------------------------------------------------------------------
# Pipeline output helpers
# ---------------------------------------------------------------------------


def _segment_file(out_dir: str, segment: str) -> Optional[Path]:
    """Return path to the latest GeoPackage for ``segment`` if it exists."""
    files = sorted(Path(out_dir).glob(f"segment_{segment}_*.gpkg"))
    return files[-1] if files else None


def fetch_segment_data(segment: str, out_dir: str) -> Tuple[gpd.GeoDataFrame, float]:
    """Fetch geometries and total area for ``segment`` from pipeline outputs."""
    gpkg = _segment_file(out_dir, segment)
    if gpkg is None:
        raise FileNotFoundError(f"No GeoPackage found for segment '{segment}' in '{out_dir}'")
    gdf = gpd.read_file(gpkg)
    area = gdf.geometry.area.sum()
    return gdf, area


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ask(question: str, bbox: Iterable[float], *, out_dir: str = "data", use_altair: bool = True):
    """Answer ``question`` about a given ``bbox`` using pipeline outputs.

    If required data are missing the processing pipeline is executed.  The
    result is returned as a tuple of (chart, dataframe).
    """
    segments = parse_user_text(question)
    if not segments:
        raise ValueError("No known segment types referenced in question")

    config = load_config(bbox=bbox, out_dir=out_dir)

    # Run pipeline if required outputs are missing
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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main() -> None:  # pragma: no cover - CLI helper
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


if __name__ == "__main__":  # pragma: no cover
    _main()
