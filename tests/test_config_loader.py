from __future__ import annotations

import json
from pathlib import Path

from eval.config_loader import alpha_for_event, load_alpha_by_category


def test_load_alpha_by_category_missing_returns_empty(tmp_path: Path) -> None:
    assert load_alpha_by_category(tmp_path / "missing.json") == {}


def test_alpha_for_event_uses_override(tmp_path: Path) -> None:
    path = tmp_path / "alpha_by_category.json"
    path.write_text(json.dumps({"crypto": 0.4, "general": 0.6}), encoding="utf-8")
    overrides = load_alpha_by_category(path)
    event = {"title": "Will Bitcoin hit $1M?", "rules": ""}
    assert alpha_for_event(event, overrides=overrides) == 0.4
