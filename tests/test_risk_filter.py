from __future__ import annotations

from agent.schemas import MarketStat, Prediction
from agent.sanitize import sanitize_prediction
from forecasting.calibration import market_probability


def test_risk_filter_caps_large_edge() -> None:
    stats = {"Yes": MarketStat(last_price=0.5, yes_ask=0.51, no_ask=0.49)}
    p_market = market_probability(stats, "Yes")
    bold = Prediction(YES=0.95, NO=0.05)
    final, _ = sanitize_prediction(bold, "test", stats, max_edge=0.30)
    assert abs(final.YES - p_market) <= 0.30 + 1e-9


def test_risk_filter_preserves_small_edge() -> None:
    stats = {"Yes": MarketStat(last_price=0.5, yes_ask=0.51, no_ask=0.49)}
    mild = Prediction(YES=0.58, NO=0.42)
    final, _ = sanitize_prediction(mild, "test", stats, max_edge=0.30)
    assert abs(final.YES - 0.58) < 1e-9


def test_risk_filter_nan_falls_back_to_market() -> None:
    stats = {"Yes": MarketStat(last_price=0.6)}
    bad = Prediction.model_construct(YES=float("nan"), NO=float("nan"))
    final, _ = sanitize_prediction(bad, "x", stats, max_edge=0.30)
    assert 0.59 < final.YES < 0.61
