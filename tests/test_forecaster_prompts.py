from __future__ import annotations

from agent.schemas import MarketStat, PredictRequest
from forecasting.forecaster import build_forecast_messages, forecast_prompt_name


def _easy_request() -> PredictRequest:
    return PredictRequest(
        event_id="evt-1",
        title="Will country X hold an election by March 2026?",
        markets=["Yes", "No"],
        rules="Resolves YES if a general election is officially announced by March 31, 2026.",
        market_stats={
            "Yes": MarketStat(last_price=0.72, yes_ask=0.73, no_ask=0.28),
            "No": MarketStat(last_price=0.28, yes_ask=0.27, no_ask=0.72),
        },
    )


def test_hard_event_uses_cot_prompt() -> None:
    req = PredictRequest(
        event_id="evt-2",
        title="Uncertain event",
        markets=["Yes", "No"],
        rules="TBD",
        market_stats={
            "Yes": MarketStat(last_price=0.5, yes_ask=0.5, no_ask=0.5),
            "No": MarketStat(last_price=0.5, yes_ask=0.5, no_ask=0.5),
        },
    )
    assert forecast_prompt_name(req) == "forecast_cot_v1.txt"
    messages = build_forecast_messages(req, [])
    user = messages[1]["content"]
    assert "base rate" in user.lower() or "Base rate" in user


def test_easy_event_uses_standard_prompt() -> None:
    assert forecast_prompt_name(_easy_request()) == "forecast_v1.txt"
