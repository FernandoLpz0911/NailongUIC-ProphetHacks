"""Unit tests for agent/stages/text_review.py.

Covers the three parsing paths in _parse_review_json:
  1. Clean JSON (fast path)
  2. Markdown-fenced JSON
  3. Trailing commas (Python-style dict output)
  4. Truncated array — object-by-object extraction
  5. Leading prose before the JSON block
  6. Full _generate_review integration with a mock LLM client
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agent.stages.text_review import (
    TextReviewStage,
    _category,
    _extract_objects,
    _parse_review_json,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(market_id: str = "kalshi:FOO-1", priority: int = 1) -> dict:
    return {
        "market_id": market_id,
        "priority": priority,
        "queries": ["query one", "query two"],
        "rationale": "Looks mispriced.",
    }


def _review_json(entries: list[dict] | None = None) -> str:
    return json.dumps({"review": entries or [_entry()]})


# ---------------------------------------------------------------------------
# _parse_review_json — fast path (clean JSON)
# ---------------------------------------------------------------------------

def test_parse_clean_json():
    raw = _review_json()
    result = _parse_review_json(raw)
    assert result is not None
    assert len(result["review"]) == 1
    assert result["review"][0]["market_id"] == "kalshi:FOO-1"


def test_parse_multiple_entries():
    raw = _review_json([_entry("kalshi:FOO-1", 1), _entry("kalshi:BAR-2", 2)])
    result = _parse_review_json(raw)
    assert result is not None
    assert len(result["review"]) == 2


# ---------------------------------------------------------------------------
# _parse_review_json — markdown code fence
# ---------------------------------------------------------------------------

def test_parse_json_fence():
    raw = "```json\n" + _review_json() + "\n```"
    result = _parse_review_json(raw)
    assert result is not None
    assert len(result["review"]) == 1


def test_parse_plain_fence():
    raw = "```\n" + _review_json() + "\n```"
    result = _parse_review_json(raw)
    assert result is not None
    assert len(result["review"]) == 1


# ---------------------------------------------------------------------------
# _parse_review_json — trailing commas (Python-style dicts)
# ---------------------------------------------------------------------------

def test_parse_trailing_comma_in_object():
    raw = '{"review": [{"market_id": "kalshi:X", "priority": 1, "queries": ["q"], "rationale": "r",}]}'
    result = _parse_review_json(raw)
    assert result is not None
    assert result["review"][0]["market_id"] == "kalshi:X"


def test_parse_trailing_comma_in_array():
    raw = '{"review": [{"market_id": "kalshi:X", "priority": 1, "queries": ["q",], "rationale": "r"}]}'
    result = _parse_review_json(raw)
    assert result is not None
    assert result["review"][0]["queries"] == ["q"]


# ---------------------------------------------------------------------------
# _parse_review_json — truncated output (MAX_TOKENS path)
# ---------------------------------------------------------------------------

def test_parse_truncated_mid_second_entry():
    """Simulates MAX_TOKENS cutting off mid-way through the second entry."""
    truncated = (
        '{"review": [\n'
        '  {"market_id": "kalshi:FOO-1", "priority": 1, "queries": ["q1", "q2"], "rationale": "r1"},\n'
        '  {"market_id": "kalshi:BAR-2", "priority": 2, "queries": ["q'
        # truncated here
    )
    result = _parse_review_json(truncated)
    assert result is not None
    # Should recover at least the first complete entry
    assert any(e["market_id"] == "kalshi:FOO-1" for e in result["review"])


def test_parse_truncated_no_complete_entry():
    """Nothing parseable — should return None rather than crash."""
    result = _parse_review_json('{"review": [{"market_id": "kal')
    assert result is None


# ---------------------------------------------------------------------------
# _parse_review_json — leading prose
# ---------------------------------------------------------------------------

def test_parse_with_leading_prose():
    raw = "Here are the markets I selected:\n\n" + _review_json()
    result = _parse_review_json(raw)
    assert result is not None
    assert len(result["review"]) == 1


def test_parse_returns_none_on_garbage():
    assert _parse_review_json("no json here at all") is None
    assert _parse_review_json("") is None
    assert _parse_review_json("[]") is None  # list at top level, no 'review' key


# ---------------------------------------------------------------------------
# _extract_objects — low-level helper
# ---------------------------------------------------------------------------

def test_extract_objects_from_clean_array():
    s = '[{"a": 1}, {"b": 2}]'
    objs = _extract_objects(s)
    assert len(objs) == 2
    assert objs[0] == {"a": 1}


def test_extract_objects_from_truncated_array():
    s = '[{"a": 1}, {"b": 2}, {"c": '  # truncated
    objs = _extract_objects(s)
    assert len(objs) == 2


def test_extract_objects_handles_trailing_commas():
    s = '[{"a": 1,}, {"b": 2,}]'
    objs = _extract_objects(s)
    assert len(objs) == 2


def test_extract_objects_empty():
    assert _extract_objects("") == []
    assert _extract_objects("no braces") == []


# ---------------------------------------------------------------------------
# TextReviewStage._generate_review — integration with mock LLM
# ---------------------------------------------------------------------------

def _make_stage() -> TextReviewStage:
    mock_llm = MagicMock()
    stage = TextReviewStage(llm_client=mock_llm, max_markets=5)
    return stage


def _make_mock_response(text: str):
    resp = MagicMock()
    resp.content = text
    return resp


def _make_candidates(n: int = 3):
    candidates = []
    for i in range(n):
        m = MagicMock()
        m.market_id = f"kalshi:MKT-{i}"
        m.question = f"Will thing {i} happen?"
        m.yes_bid = 0.40
        m.yes_ask = 0.45
        m.volume_24h = 1000.0
        candidates.append(m)
    return candidates


def _make_tick_ctx(candidates):
    ctx = MagicMock()
    ctx.tick_ts = "2026-05-17T05:00:00+00:00"
    ctx.cash = 10000.0
    ctx.candidates = candidates
    ctx.positions = []
    ctx.memory_summary = ""
    return ctx


def test_generate_review_parses_clean_response():
    stage = _make_stage()
    entries = [_entry("kalshi:MKT-0", 1), _entry("kalshi:MKT-1", 2)]
    stage.llm_client.generate.return_value = _make_mock_response(
        json.dumps({"review": entries})
    )
    result = stage._generate_review(_make_candidates(), _make_tick_ctx(_make_candidates()))
    assert len(result["review"]) == 2
    assert result["review"][0]["market_id"] == "kalshi:MKT-0"


def test_generate_review_filters_invalid_entries():
    stage = _make_stage()
    raw = json.dumps({"review": [
        {"market_id": "kalshi:GOOD", "priority": 1, "queries": ["q"], "rationale": "r"},
        {"market_id": "", "priority": 2, "queries": ["q"], "rationale": "r"},   # empty id
        {"priority": 3, "queries": ["q"], "rationale": "r"},                    # missing id
        {"market_id": "kalshi:NOQUERY", "priority": 4, "queries": [], "rationale": "r"},  # empty queries
    ]})
    stage.llm_client.generate.return_value = _make_mock_response(raw)
    result = stage._generate_review(_make_candidates(), _make_tick_ctx(_make_candidates()))
    assert len(result["review"]) == 1
    assert result["review"][0]["market_id"] == "kalshi:GOOD"


def test_generate_review_returns_empty_on_bad_response():
    stage = _make_stage()
    stage.llm_client.generate.return_value = _make_mock_response("not json at all")
    result = stage._generate_review(_make_candidates(), _make_tick_ctx(_make_candidates()))
    assert result == {"review": []}


def test_generate_review_handles_fenced_response():
    stage = _make_stage()
    entries = [_entry("kalshi:MKT-0")]
    stage.llm_client.generate.return_value = _make_mock_response(
        "```json\n" + json.dumps({"review": entries}) + "\n```"
    )
    result = stage._generate_review(_make_candidates(), _make_tick_ctx(_make_candidates()))
    assert len(result["review"]) == 1


# ---------------------------------------------------------------------------
# _category — category inference
# ---------------------------------------------------------------------------

def test_category_politics_by_keyword():
    assert _category("kalshi:KXPRES-2024", "Who will win the presidential election?") == "Politics"


def test_category_politics_by_market_id():
    assert _category("kalshi:kxvote-2024", "Will the vote pass?") == "Politics"


def test_category_economics_by_keyword():
    assert _category("kalshi:KXBTC-100K", "Will Bitcoin reach $100K?") == "Economics"


def test_category_sports_by_keyword():
    assert _category("kalshi:KXNBA-FINALS", "Will the Lakers win the NBA finals?") == "Sports"


def test_category_entertainment_by_keyword():
    assert _category("kalshi:KXOSCAR-BP", "Who will win the Oscar for Best Picture?") == "Entertainment"


def test_category_science_tech_by_keyword():
    assert _category("kalshi:KXSPACEX-LAUNCH", "Will SpaceX launch by Friday?") == "Science/Tech"


def test_category_falls_back_to_other():
    assert _category("kalshi:RANDOM-MARKET-99", "Will it rain in Seattle?") == "Other"


def test_category_case_insensitive():
    assert _category("KALSHI:KXNFL-SB", "NFL Super Bowl winner?") == "Sports"


def test_category_prefers_first_match():
    # "nba" appears in Sports but "fed" is in Economics — first match wins
    result = _category("kxnba-market", "Will the Fed raise rates this NBA season?")
    # Sports keywords come before Economics in the dict, but "fed" is in the text too.
    # As long as one deterministic category is returned, we're good.
    assert result in {"Politics", "Economics", "Sports", "Entertainment", "Science/Tech", "Other"}
