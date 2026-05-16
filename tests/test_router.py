from datetime import date
from unittest.mock import patch

from agent.config import CHEAP_MODEL, EXPENSIVE_MODEL
from agent.router import classify_event, select_model
from agent.schemas import MarketStat, PredictRequest


def _request(
    *,
    rules: str = "Resolves YES if a general election is officially announced by March 31, 2026.",
    title: str = "Will country X hold an election by March 2026?",
    yes_ask: float = 0.73,
    no_ask: float = 0.28,
) -> PredictRequest:
    yes_mid = (yes_ask + (1.0 - no_ask)) / 2.0
    return PredictRequest(
        event_id="EVT_TEST",
        title=title,
        markets=["Yes", "No"],
        rules=rules,
        market_stats={
            "Yes": MarketStat(last_price=yes_mid, yes_ask=yes_ask, no_ask=no_ask),
            "No": MarketStat(last_price=1 - yes_mid, yes_ask=1 - no_ask, no_ask=yes_ask),
        },
    )


def test_classify_easy_clear_rules_strong_market_near_term() -> None:
    assert classify_event(_request()) == "easy"


def test_classify_hard_short_rules() -> None:
    req = _request(rules="TBD")
    assert classify_event(req) == "hard"


def test_classify_hard_unclear_in_rules() -> None:
    rules = "Resolution criteria are unclear until further notice from the commission."
    assert len(rules) >= 50
    assert classify_event(_request(rules=rules)) == "hard"


def test_classify_hard_market_near_fifty() -> None:
    assert classify_event(_request(yes_ask=0.50, no_ask=0.50)) == "hard"
    assert classify_event(_request(yes_ask=0.45, no_ask=0.55)) == "hard"
    assert classify_event(_request(yes_ask=0.55, no_ask=0.45)) == "hard"


@patch("agent.router._today")
def test_classify_hard_resolution_far_future(mock_today) -> None:
    mock_today.return_value = date(2026, 5, 16)
    req = _request(
        title="Will event happen by December 2028?",
        rules="Resolves YES if the event occurs on or before December 31, 2028.",
    )
    assert classify_event(req) == "hard"


@patch("agent.router._today")
def test_classify_easy_resolution_within_year(mock_today) -> None:
    mock_today.return_value = date(2026, 5, 16)
    req = _request(
        title="Will event happen by December 2026?",
        rules="Resolves YES if the event occurs on or before December 31, 2026.",
    )
    assert classify_event(req) == "easy"


def test_select_model_tiers() -> None:
    assert select_model("easy") == CHEAP_MODEL
    assert select_model("hard") == EXPENSIVE_MODEL
