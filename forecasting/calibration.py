from __future__ import annotations

from agent.schemas import MarketStat, Prediction


def market_probability(market_stats: dict[str, MarketStat], outcome: str = "Yes") -> float:
    """
    Best-effort market-implied probability from Kalshi stats.

    Uses midpoint of yes_ask / no_ask when available (game plan Stage 3).
    """
    stats = market_stats.get(outcome) or market_stats.get(outcome.upper())
    if stats is None and market_stats:
        stats = next(iter(market_stats.values()))

    if stats is None:
        return 0.5

    yes_ask = stats.yes_ask
    no_ask = stats.no_ask
    if yes_ask is not None and no_ask is not None:
        return max(0.0, min(1.0, (yes_ask + (1.0 - no_ask)) / 2.0))
    if stats.last_price is not None:
        return max(0.0, min(1.0, stats.last_price))
    return 0.5


def calibrate_vs_market(
    model: Prediction,
    market_stats: dict[str, MarketStat],
    *,
    alpha: float = 0.5,
) -> Prediction:
    """Blend model probability with market: p_final = α·p_model + (1-α)·p_market."""
    alpha = max(0.0, min(1.0, alpha))
    p_market = market_probability(market_stats, "Yes")
    yes = alpha * model.YES + (1.0 - alpha) * p_market
    yes = max(0.0, min(1.0, yes))
    return Prediction(YES=yes, NO=1.0 - yes)
