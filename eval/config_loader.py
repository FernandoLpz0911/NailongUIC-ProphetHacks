from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.config import CALIBRATION_ALPHA, DATA_DIR
from retrieval.category import detect_category

DEFAULT_ALPHA_PATH = DATA_DIR / "alpha_by_category.json"


def load_alpha_by_category(path: Path | None = None) -> dict[str, float]:
    """Load per-category calibration α overrides; empty dict if file missing."""
    alpha_path = path or DEFAULT_ALPHA_PATH
    if not alpha_path.exists():
        return {}
    payload = json.loads(alpha_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"alpha config must be a JSON object: {alpha_path}")
    return {str(k): float(v) for k, v in payload.items()}


def alpha_for_event(
    event: dict[str, Any],
    *,
    overrides: dict[str, float] | None = None,
    default: float | None = None,
) -> float:
    """Return tuned α for an event, or default when no override exists."""
    category = detect_category(event.get("title", ""), event.get("rules", ""))
    table = overrides if overrides is not None else load_alpha_by_category()
    fallback = CALIBRATION_ALPHA if default is None else default
    return table.get(category, fallback)
