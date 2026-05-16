from __future__ import annotations

import json
from pathlib import Path

from eval.compare import run_comparison


def test_compare_market_only(tmp_path: Path) -> None:
    events = tmp_path / "events.json"
    events.write_text(
        json.dumps(
            [
                {
                    "event_id": "E1",
                    "title": "Test",
                    "market_stats": {
                        "Yes": {"last_price": 0.6, "yes_ask": 0.61, "no_ask": 0.40}
                    },
                    "outcome_yes": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    rows, markdown = run_comparison(events)
    market_row = next(r for r in rows if r["mode"] == "market_only")
    assert market_row["events_scored"] == 1
    assert market_row["brier"] != "—"
    assert "market_only" in markdown
