from __future__ import annotations

from agent.schemas import MarketStat, Prediction
from forecasting.calibration import market_probability

"""
Average-return proxy for local backtests (Trading Track).

Prophet Arena uses optimal betting vs live Kalshi prices. We approximate:
  return ≈ edge × (realized_payoff − market_price)

where edge = p_model − p_market and realized_payoff is 1 if YES resolved else 0.
Risk aversion scales how aggressively we bet on the edge (default 1.0 = linear).
"""


def simulate_return(
    prediction: Prediction,
    market_stats: dict[str, MarketStat],
    *,
    outcome_yes: bool,
    risk_aversion: float = 1.0,
) -> float:
    p_market = market_probability(market_stats, "Yes")
    p_model = prediction.YES
    edge = p_model - p_market
    realized = 1.0 if outcome_yes else 0.0
    # Payoff of a YES position at market price p_market when outcome resolves
    payoff_delta = realized - p_market
    return risk_aversion * edge * payoff_delta


def expected_return(
    prediction: Prediction,
    market_stats: dict[str, MarketStat],
    *,
    risk_aversion: float = 1.0,
) -> float:
    """Expected return if we treat model probability as belief."""
    p_market = market_probability(market_stats, "Yes")
    edge = prediction.YES - p_market
    # E[outcome - p_market] under model belief p_model
    p_model = prediction.YES
    return risk_aversion * edge * (p_model - p_market)
