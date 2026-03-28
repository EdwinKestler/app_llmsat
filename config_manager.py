"""Centralized configuration management for LLMSat.

All user-configurable settings live in ``config.json`` in the project
root.  The Streamlit Settings tab writes to this file; every other
module reads from it via :func:`load_config` / :func:`get`.

Environment variables (``.env``) are used **only** for secrets
(API keys, tokens).  Everything else goes through ``config.json``.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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

    # GeoDeep CPU fallback
    "cpu_fallback": False,
    "hybrid_mode": False,

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


# ── Validation helpers ──────────────────────────────────────────────

def validate_bbox(bbox: list[float]) -> tuple[bool, str]:
    """Validate a bounding box. Returns (is_valid, error_message)."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False, "Bounding box must have exactly 4 values."
    try:
        west, south, east, north = [float(x) for x in bbox]
    except (TypeError, ValueError):
        return False, "All bbox values must be numbers."
    if any(not math.isfinite(x) for x in [west, south, east, north]):
        return False, "Bbox values must be finite (no NaN or Inf)."
    if not (-180 <= west <= 180 and -180 <= east <= 180):
        return False, "Longitude must be between -180 and 180."
    if not (-90 <= south <= 90 and -90 <= north <= 90):
        return False, "Latitude must be between -90 and 90."
    if west >= east:
        return False, "West must be less than East."
    if south >= north:
        return False, "South must be less than North."
    if (east - west) > 1.0 or (north - south) > 1.0:
        return False, "Bbox too large (max 1 degree per side, ~111 km)."
    return True, ""


def validate_url(url: str) -> tuple[bool, str]:
    """Validate that a URL is HTTPS and not a private/local address."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Invalid URL format."
    if parsed.scheme != "https":
        return False, "Only HTTPS URLs are allowed."
    if not parsed.hostname:
        return False, "URL must have a hostname."
    hostname = parsed.hostname.lower()
    # Block local/private hostnames
    blocked = ["localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"]
    if hostname in blocked or hostname.startswith("192.168.") or hostname.startswith("10.") or hostname.startswith("172."):
        return False, "Private/local URLs are not allowed."
    return True, ""


def validate_safe_path(path: str) -> tuple[bool, str]:
    """Validate that a path is safe (within project root, no traversal)."""
    try:
        resolved = Path(path).resolve()
    except Exception:
        return False, "Invalid path."
    project_root = _PROJECT_ROOT.resolve()
    # Allow absolute paths within project, or relative paths that resolve within project
    if not str(resolved).startswith(str(project_root)):
        # Also allow if path is relative and doesn't escape
        relative_resolved = (_PROJECT_ROOT / path).resolve()
        if not str(relative_resolved).startswith(str(project_root)):
            return False, f"Path must be within project directory: {project_root}"
    return True, ""


# ── Public API ──────────────────────────────────────────────────────

def load_config() -> dict[str, Any]:
    """Load config from ``config.json``, merged with defaults."""
    cfg = DEFAULTS.copy()
    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH) as f:
                user = json.load(f)
            if isinstance(user, dict):
                # Validate critical fields
                if "confidence_threshold" in user:
                    v = user["confidence_threshold"]
                    if not (isinstance(v, (int, float)) and 0 <= v <= 1):
                        user["confidence_threshold"] = DEFAULTS["confidence_threshold"]
                if "mask_threshold" in user:
                    v = user["mask_threshold"]
                    if not (isinstance(v, (int, float)) and 0 <= v <= 1):
                        user["mask_threshold"] = DEFAULTS["mask_threshold"]
                if "osm_overpass_url" in user:
                    ok, _ = validate_url(user["osm_overpass_url"])
                    if not ok:
                        user["osm_overpass_url"] = DEFAULTS["osm_overpass_url"]
                if "output_dir" in user:
                    ok, _ = validate_safe_path(user["output_dir"])
                    if not ok:
                        user["output_dir"] = DEFAULTS["output_dir"]
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
    """Write a secret to the ``.env`` file with restricted permissions.

    Only updates the specific key, preserving other entries.
    File is created with mode 0600 (owner-only read/write).
    """
    # Validate key name (prevent injection via key)
    if not key.isidentifier():
        raise ValueError(f"Invalid secret key: {key}")

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

    # Write with restricted permissions
    old_umask = os.umask(0o077)
    try:
        with open(env_path, "w") as f:
            f.writelines(lines)
    finally:
        os.umask(old_umask)

    # Ensure file permissions are correct even if file existed before
    try:
        os.chmod(env_path, 0o600)
    except OSError:
        pass  # Windows doesn't support chmod

    # Also set in current process
    os.environ[key] = value
