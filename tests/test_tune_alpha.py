from __future__ import annotations

import json
from pathlib import Path

from eval.config_loader import alpha_for_event, load_alpha_by_category
from eval.tune_alpha import tune_all, write_alpha_config
from retrieval.category import detect_category


def test_detect_category_keywords() -> None:
    assert detect_category("Will Bitcoin hit $200k?", "") == "crypto"
    assert detect_category("Will country X hold an election?", "") == "politics"
    assert detect_category("Will the Fed cut rates?", "") == "general"


def test_tune_alpha_writes_per_category(tmp_path: Path) -> None:
    events = tmp_path / "events.json"
    events.write_text(
        json.dumps(
            [
                {
                    "event_id": "E_FED",
                    "title": "Will the Fed cut rates?",
                    "rules": "Resolves YES on a cut.",
                    "market_stats": {
                        "Yes": {"last_price": 0.5, "yes_ask": 0.51, "no_ask": 0.49}
                    },
                    "outcome_yes": True,
                },
                {
                    "event_id": "E_BTC",
                    "title": "Will Bitcoin exceed $100k?",
                    "rules": "Resolves YES if BTC trades above $100k.",
                    "market_stats": {
                        "Yes": {"last_price": 0.5, "yes_ask": 0.51, "no_ask": 0.49}
                    },
                    "outcome_yes": False,
                },
            ]
        ),
        encoding="utf-8",
    )
    preds = tmp_path / "preds.json"
    preds.write_text(
        json.dumps(
            [
                {
                    "event_id": "E_FED",
                    "raw_prediction": {"YES": 0.9, "NO": 0.1},
                    "prediction": {"YES": 0.7, "NO": 0.3},
                },
                {
                    "event_id": "E_BTC",
                    "raw_prediction": {"YES": 0.2, "NO": 0.8},
                    "prediction": {"YES": 0.4, "NO": 0.6},
                },
            ]
        ),
        encoding="utf-8",
    )

    alphas = tune_all(events, preds, grid=[0.3, 0.5, 0.8])
    assert "general" in alphas
    assert "crypto" in alphas
    assert all(0.0 <= v <= 1.0 for v in alphas.values())

    out = tmp_path / "alpha_by_category.json"
    write_alpha_config(out, alphas)
    loaded = load_alpha_by_category(out)
    assert loaded["general"] == alphas["general"]

    event = {"title": "Will the Fed cut rates?", "rules": ""}
    assert alpha_for_event(event, overrides=loaded) == alphas["general"]
