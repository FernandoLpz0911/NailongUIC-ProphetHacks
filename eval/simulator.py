from __future__ import annotations

from agent.schemas import MarketStat, Prediction


def simulate_return(
    prediction: Prediction,
    market_stats: dict[str, MarketStat],
    *,
    outcome_yes: bool,
    risk_aversion: float = 1.0,
) -> float:
    """
    Simplified average-return proxy for local backtests.

    Stage 2+: align with Prophet Arena optimal-betting scorer.
    Positive when model edge agrees with realized outcome.
    """
    p_market = _market_yes(market_stats)
    p_model = prediction.YES
    edge = p_model - p_market
    realized = 1.0 if outcome_yes else 0.0
    return risk_aversion * edge * (realized - p_market)


def _market_yes(market_stats: dict[str, MarketStat]) -> float:
    stats = market_stats.get("Yes") or market_stats.get("YES")
    if stats and stats.last_price is not None:
        return stats.last_price
    if market_stats:
        first = next(iter(market_stats.values()))
        if first.last_price is not None:
            return first.last_price
    return 0.5
