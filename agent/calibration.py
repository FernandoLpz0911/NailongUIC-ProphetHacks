"""Market-anchored calibration math.

This is the single most impactful module for Trading-Track scoring: every
forecast is blended toward market consensus so we only deviate when we have
real evidence, which directly reduces variance (helps Sharpe) and bounds
losses (helps PnL).

Pure functions only — no I/O, no LLM calls. Tested in isolation by
`agent/tests/test_calibration.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.settings import CalibrationConfig

ALPHA_BY_CONFIDENCE = {
    "high":   "alpha_high",
    "medium": "alpha_medium",
    "low":    "alpha_low",
}


def market_mid(yes_bid: float | None, yes_ask: float | None) -> float | None:
    """Best-effort market mid for YES, guarded against missing or crossed quotes."""
    if yes_bid is None and yes_ask is None:
        return None
    if yes_bid is None:
        return _clamp01(float(yes_ask))  # type: ignore[arg-type]
    if yes_ask is None:
        return _clamp01(float(yes_bid))
    bid = float(yes_bid)
    ask = float(yes_ask)
    if ask < bid:
        bid, ask = ask, bid
    return _clamp01((bid + ask) / 2.0)


def blend(p_model: float, p_market: float, alpha: float) -> float:
    """Convex combination: `alpha` of model + `(1 - alpha)` of market.

    alpha=1.0 -> pure model. alpha=0.0 -> pure market. Out-of-range alphas
    are clipped to [0, 1] so callers can pass raw confidence values without
    re-validating.
    """
    a = min(1.0, max(0.0, alpha))
    return _clamp01(a * p_model + (1.0 - a) * p_market)


def cap_deviation(p_proposed: float, p_market: float, max_dev: float) -> float:
    """Hard-cap |p_proposed - p_market| at `max_dev`.

    This is the trading-track-specific risk filter from CustomAgentTradingRules.pdf:
    a forecast claiming >30% edge against the market is almost always wrong, so
    we clip to the cap and let the action stage decide if even the clipped
    edge is large enough to trade on.
    """
    if max_dev <= 0:
        return _clamp01(p_market)
    dev = p_proposed - p_market
    if dev > max_dev:
        return _clamp01(p_market + max_dev)
    if dev < -max_dev:
        return _clamp01(p_market - max_dev)
    return _clamp01(p_proposed)


def polymarket_consensus(
    market_history: dict | None,
    p_kalshi: float,
    *,
    config: CalibrationConfig,
) -> float | None:
    """If Polymarket agrees with Kalshi within the agreement band, return a
    volume-weighted consensus price; otherwise return None and let the caller
    fall back to the Kalshi mid alone.

    Why: when both major prediction markets agree, the consensus is a much
    stronger anchor than either alone. When they disagree, one of them is
    wrong but we don't know which, so it's safer to ignore Polymarket and
    let the model's edge speak via the standard alpha blend.

    Stale or thin Polymarket data is ignored entirely (volume floor).
    """
    if not market_history:
        return None
    yes = market_history.get("yes_price")
    volume = market_history.get("volume") or 0.0
    if yes is None:
        return None
    try:
        p_poly = float(yes)
        v = float(volume)
    except (TypeError, ValueError):
        return None
    if v < config.polymarket_min_volume_usd:
        return None
    if abs(p_poly - p_kalshi) > config.polymarket_agreement_band:
        return None
    # 50/50 average is a reasonable consensus when both venues agree this
    # tightly. Volume weighting would just amplify whichever side has more
    # liquidity, which is rarely the side with better information.
    return _clamp01((p_poly + p_kalshi) / 2.0)


def alpha_for_confidence(confidence: str, config: CalibrationConfig) -> float:
    """Look up the configured alpha for a retrieval confidence label."""
    attr = ALPHA_BY_CONFIDENCE.get(confidence, "alpha_low")
    return float(getattr(config, attr))


def dynamic_alpha(
    conf_model: float,
    *,
    config: CalibrationConfig,
    low_liquidity: bool = False,
) -> float:
    """Model-weight alpha adjusted for LLM ensemble confidence and market liquidity.

    Translates the guide's market-weight formula into code convention (alpha = model weight):
      market_weight = 0.60 + 0.15*(1-conf_model) - 0.10*low_liquidity
      model_weight  = 1 - market_weight

    We anchor the high-confidence baseline to config.alpha_medium so operators
    can tune it via env vars without code changes.
    """
    base = config.alpha_medium  # model weight at full confidence
    adj = -0.15 * (1.0 - conf_model)  # high variance → trust model less
    if low_liquidity:
        adj += 0.10  # thin market → trust model more
    return min(max(base + adj, 0.15), 0.85)


def epistemic_shrink(p: float, llm_var: float) -> float:
    """James-Stein shrinkage factor for Kelly sizing under LLM estimation uncertainty.

    Shrink = naive_var / (naive_var + LLM_var)

    When llm_var=0 (no ensemble), returns 1.0 (no shrinkage).
    When llm_var is large, reduces the Kelly fraction proportionally.
    """
    naive_var = p * (1.0 - p)
    total_var = naive_var + max(llm_var, 0.0)
    if total_var < 1e-9:
        return 1.0
    return naive_var / total_var


@dataclass(frozen=True)
class CalibratedForecast:
    """Output of `calibrate()` — the final p_yes plus the inputs that produced it."""

    p_final: float
    p_model: float
    p_market: float
    p_anchor: float
    alpha: float
    used_polymarket: bool
    clipped_to_cap: bool
    raw_gap: float = 0.0      # p_model - p_market before blending (for raw-gap gating)
    conf_model: float = 0.0   # LLM ensemble confidence [0,1]; 0 means not available


def calibrate(
    p_model: float,
    p_market: float,
    *,
    confidence: str,
    market_history: dict | None,
    config: CalibrationConfig,
    conf_model: float = 0.0,
    low_liquidity: bool = False,
    hours_to_resolution: float | None = None,
    category_alpha: float | None = None,
) -> CalibratedForecast:
    """End-to-end blend: pick anchor, blend, clip deviation, return all inputs.

    When conf_model > 0 (LLM ensemble available), uses dynamic_alpha() instead
    of the static confidence-string lookup. raw_gap is included in the output
    so the action stage can gate on the pre-blend model deviation.

    category_alpha overrides the computed alpha when the paper's per-category
    findings warrant it (e.g. Sports: trust market more; Politics: trust LLM more).

    hours_to_resolution decays alpha toward 0 inside the decay window (paper
    Fig.2: markets incorporate information faster than LLMs near resolution).
    """
    p_model = _clamp01(p_model)
    p_market = _clamp01(p_market)

    if conf_model > 0.0:
        alpha = dynamic_alpha(conf_model, config=config, low_liquidity=low_liquidity)
    else:
        alpha = alpha_for_confidence(confidence, config)

    # Category override takes precedence over the confidence-derived alpha.
    if category_alpha is not None:
        alpha = float(category_alpha)

    # Time-to-resolution decay: ramp alpha from full → 0 over decay_hours.
    # At hours_to_resolution=0, alpha=0 (pure market). At >= decay_hours, no change.
    if hours_to_resolution is not None:
        decay_hours = getattr(config, "resolution_decay_hours", 72.0)
        time_weight = min(1.0, hours_to_resolution / max(decay_hours, 1.0))
        alpha = alpha * time_weight

    poly = polymarket_consensus(market_history, p_market, config=config)
    if poly is not None:
        anchor = poly
        used_polymarket = True
    else:
        anchor = p_market
        used_polymarket = False

    blended = blend(p_model, anchor, alpha)
    p_final = cap_deviation(blended, p_market, config.max_deviation)
    clipped = abs(blended - p_final) > 1e-9

    return CalibratedForecast(
        p_final=p_final,
        p_model=p_model,
        p_market=p_market,
        p_anchor=anchor,
        alpha=alpha,
        used_polymarket=used_polymarket,
        clipped_to_cap=clipped,
        raw_gap=p_model - p_market,
        conf_model=conf_model,
    )


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)
