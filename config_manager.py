"""Centralized configuration management for LLMSat.

All user-configurable settings live in ``config.json`` in the project
root.  The Streamlit Settings tab writes to this file; every other
module reads from it via :func:`load_config` / :func:`get`.

Environment variables (``.env``) are used **only** for secrets
(API keys, tokens).  Everything else goes through ``config.json``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent
_CONFIG_PATH = _PROJECT_ROOT / "config.json"

# ── Default configuration ───────────────────────────────────────────

DEFAULTS: dict[str, Any] = {
    # Map & area
    "default_bbox": [-90.5115, 14.6304, -90.5015, 14.6394],
    "default_center": [14.6349, -90.5065],
    "default_zoom": 18,

    # Output
    "output_dir": "output",

    # Segmentation (SAM3)
    "sam3_backend": "meta",
    "sam3_model_id": "facebook/sam3",
    "confidence_threshold": 0.5,
    "mask_threshold": 0.5,

    # Open Buildings
    "open_buildings_dir": "openbuildings",
    "open_buildings_min_confidence": 0.65,
    "open_buildings_max_buildings": 50000,

    # OSM Roads
    "osm_overpass_url": "https://overpass-api.de/api/interpreter",
    "osm_timeout": 60,

    # OpenAI
    "openai_model": "gpt-5.4-nano",

    # ET Monitor (planned)
    "et_monitor_root": "",

    # UI
    "segment_colors": {
        "tree":     [0, 200, 0, 140],
        "building": [200, 0, 0, 140],
        "water":    [0, 80, 200, 140],
        "road":     [200, 200, 0, 140],
    },
}


# ── Public API ──────────────────────────────────────────────────────

def load_config() -> dict[str, Any]:
    """Load config from ``config.json``, merged with defaults."""
    cfg = DEFAULTS.copy()
    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH) as f:
                user = json.load(f)
            cfg.update(user)
        except (json.JSONDecodeError, OSError):
            pass  # Corrupt file — use defaults
    return cfg


def save_config(cfg: dict[str, Any]) -> None:
    """Write config to ``config.json``."""
    with open(_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def get(key: str, default: Any = None) -> Any:
    """Get a single config value."""
    cfg = load_config()
    return cfg.get(key, default)


def get_secret(key: str) -> str:
    """Get a secret from environment variables (loaded via .env).

    Secrets are NEVER stored in config.json.
    """
    return os.getenv(key, "")


def set_secret(key: str, value: str) -> None:
    """Write a secret to the ``.env`` file.

    Only updates the specific key, preserving other entries.
    """
    env_path = _PROJECT_ROOT / ".env"
    lines = []
    found = False

    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip().startswith(f"{key}="):
                    lines.append(f"{key}={value}\n")
                    found = True
                else:
                    lines.append(line)

    if not found:
        lines.append(f"{key}={value}\n")

    with open(env_path, "w") as f:
        f.writelines(lines)

    # Also set in current process
    os.environ[key] = value
