from __future__ import annotations

import json
from pathlib import Path

from eval.dashboard import run_dashboard


def test_dashboard_summary_and_leaderboards(tmp_path: Path) -> None:
    events = tmp_path / "events.json"
    events.write_text(
        json.dumps(
            [
                {
                    "event_id": "WIN",
                    "title": "Fed rate cut",
                    "market_stats": {
                        "Yes": {"last_price": 0.4, "yes_ask": 0.41, "no_ask": 0.60}
                    },
                    "outcome_yes": True,
                },
                {
                    "event_id": "LOSS",
                    "title": "Bitcoin moon",
                    "market_stats": {
                        "Yes": {"last_price": 0.4, "yes_ask": 0.41, "no_ask": 0.60}
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
                {"event_id": "WIN", "prediction": {"YES": 0.8, "NO": 0.2}},
                {"event_id": "LOSS", "prediction": {"YES": 0.8, "NO": 0.2}},
            ]
        ),
        encoding="utf-8",
    )

    text = run_dashboard(events, preds)
    assert "Brier" in text
    assert "Avg Return" in text
    assert "Top 5 wins" in text
    assert "Top 5 losses" in text
    assert "WIN" in text
    assert "LOSS" in text
