"""Strategy evaluation metrics — Brier, calibration, Calmar, IR, edge realization.

Reads the structured JSONL logs written by CalibratedForecastStage and
RiskAwareActionStage and computes the metrics specified in the Elite plan.

Designed to be invoked manually (`python -m agent.metrics`) or imported by
a dashboard. Pure computation — no LLM calls, no I/O beyond reading the logs.

Evaluation tiers (paper-inspired sample-size gates):
  - Tier 1 (≥50 ticks, ≥3 resolved trades): basic sanity
  - Tier 2 (≥500 ticks, ≥10 resolved trades): Brier, win rate, calibration plot
  - Tier 3 (≥3000 ticks, ≥100 resolved trades): IR vs market baseline, full validation
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DECISION_DIR = Path("data/decisions")
FORECAST_LOG = DECISION_DIR / "forecast.jsonl"
ACTION_LOG = DECISION_DIR / "action.jsonl"
FILL_LOG = DECISION_DIR / "fills.jsonl"
RESOLUTION_LOG = DECISION_DIR / "resolutions.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------

def brier_score(resolutions: list[dict]) -> float | None:
    """Mean squared error between predicted probability and realized outcome.

    Lower is better. Random forecast → 0.25. Skilled forecaster → < 0.18.
    """
    if not resolutions:
        return None
    sq_errors: list[float] = []
    for r in resolutions:
        p = r.get("p_predicted")
        o = r.get("outcome")  # 1 if YES, 0 if NO
        if p is None or o is None:
            continue
        sq_errors.append((float(p) - float(o)) ** 2)
    if not sq_errors:
        return None
    return sum(sq_errors) / len(sq_errors)


# ---------------------------------------------------------------------------
# Calibration buckets (10% bins)
# ---------------------------------------------------------------------------

@dataclass
class CalibrationBucket:
    bin_lo: float
    bin_hi: float
    n: int = 0
    mean_p_predicted: float = 0.0
    realized_yes_rate: float = 0.0

    def as_dict(self) -> dict:
        return {
            "bin":                f"{self.bin_lo:.1f}-{self.bin_hi:.1f}",
            "n":                  self.n,
            "mean_p_predicted":   round(self.mean_p_predicted, 4),
            "realized_yes_rate":  round(self.realized_yes_rate, 4),
            "calibration_error":  round(abs(self.mean_p_predicted - self.realized_yes_rate), 4),
        }


def calibration_buckets(resolutions: list[dict], n_bins: int = 10) -> list[CalibrationBucket]:
    """Group predictions into n_bins equal-width bins on [0,1] and compute
    realized YES frequency in each bin. The "perfect" calibrator has
    realized_yes_rate ≈ mean_p_predicted in every bin."""
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for r in resolutions:
        p = r.get("p_predicted")
        o = r.get("outcome")
        if p is None or o is None:
            continue
        idx = min(n_bins - 1, max(0, int(float(p) * n_bins)))
        buckets[idx].append((float(p), int(o)))

    results: list[CalibrationBucket] = []
    for i, bucket in enumerate(buckets):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if not bucket:
            results.append(CalibrationBucket(lo, hi))
            continue
        ps = [p for p, _ in bucket]
        os_ = [o for _, o in bucket]
        results.append(CalibrationBucket(
            bin_lo=lo, bin_hi=hi, n=len(bucket),
            mean_p_predicted=sum(ps) / len(ps),
            realized_yes_rate=sum(os_) / len(os_),
        ))
    return results


def expected_calibration_error(resolutions: list[dict], n_bins: int = 10) -> float | None:
    """ECE: weighted average of |bucket_p - bucket_realized| across bins.

    Paper's primary calibration metric. Lower is better. Strong models ≤ 0.05.
    """
    buckets = calibration_buckets(resolutions, n_bins=n_bins)
    total_n = sum(b.n for b in buckets)
    if total_n == 0:
        return None
    weighted_err = 0.0
    for b in buckets:
        if b.n > 0:
            weighted_err += b.n * abs(b.mean_p_predicted - b.realized_yes_rate)
    return weighted_err / total_n


# ---------------------------------------------------------------------------
# Edge realization rate
# ---------------------------------------------------------------------------

def edge_realization_rate(
    resolutions: list[dict],
    edge_threshold: float = 0.05,
) -> float | None:
    """Of resolved trades with |edge| > edge_threshold at entry, what fraction
    had (realized - p_market_entry) directionally match the predicted edge?

    Strong strategy: >0.55.  Random: 0.5.
    """
    matching: list[int] = []
    for r in resolutions:
        edge = r.get("edge_predicted")
        p_market_entry = r.get("p_market_entry")
        outcome = r.get("outcome")
        if edge is None or p_market_entry is None or outcome is None:
            continue
        if abs(float(edge)) < edge_threshold:
            continue
        realized_edge = float(outcome) - float(p_market_entry)
        match = (realized_edge * float(edge)) > 0  # same sign
        matching.append(1 if match else 0)
    if not matching:
        return None
    return sum(matching) / len(matching)


# ---------------------------------------------------------------------------
# Sharpe, Calmar, IR (need equity series)
# ---------------------------------------------------------------------------

def annualized_sharpe(returns_per_tick: list[float], ticks_per_year: int = 96 * 252) -> float | None:
    """Sharpe ratio annualized for 15-min ticks (96 ticks/day × 252 trading days).

    Uses mean / std × √N — standard Sharpe with zero risk-free rate.
    """
    if len(returns_per_tick) < 2:
        return None
    mu = statistics.mean(returns_per_tick)
    sd = statistics.stdev(returns_per_tick)
    if sd <= 1e-9:
        return None
    return (mu / sd) * math.sqrt(ticks_per_year)


def max_drawdown(equity_series: list[float]) -> float | None:
    """Maximum peak-to-trough drawdown of an equity series. Returns absolute value."""
    if not equity_series:
        return None
    peak = equity_series[0]
    max_dd = 0.0
    for e in equity_series:
        if e > peak:
            peak = e
        dd = (peak - e) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def calmar_ratio(equity_series: list[float], ticks_per_year: int = 96 * 252) -> float | None:
    """CAGR / |Max Drawdown|."""
    if len(equity_series) < 2:
        return None
    start, end = equity_series[0], equity_series[-1]
    if start <= 0:
        return None
    n_ticks = len(equity_series) - 1
    years = n_ticks / ticks_per_year
    if years <= 0:
        return None
    cagr = (end / start) ** (1 / years) - 1
    mdd = max_drawdown(equity_series) or 0.0
    if mdd <= 1e-9:
        return None
    return cagr / mdd


def information_ratio(
    strategy_returns: list[float],
    benchmark_returns: list[float],
    ticks_per_year: int = 96 * 252,
) -> float | None:
    """IR = annualized mean of (strategy - benchmark) / std of (strategy - benchmark).

    Benchmark = "buy market consensus" — what you'd earn betting at exact
    market prices. IR > 0 means real edge.
    """
    if len(strategy_returns) != len(benchmark_returns) or len(strategy_returns) < 2:
        return None
    diffs = [s - b for s, b in zip(strategy_returns, benchmark_returns)]
    mu = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    if sd <= 1e-9:
        return None
    return (mu / sd) * math.sqrt(ticks_per_year)


# ---------------------------------------------------------------------------
# Evaluation tier classifier
# ---------------------------------------------------------------------------

@dataclass
class EvaluationTier:
    tier: int  # 0, 1, 2, 3
    name: str
    trustworthy_metrics: list[str] = field(default_factory=list)
    pending_metrics: list[str] = field(default_factory=list)
    reason: str = ""


def classify_tier(n_ticks: int, n_resolved: int) -> EvaluationTier:
    """Returns which evaluation tier we're in based on accumulated data."""
    if n_ticks < 50 or n_resolved < 3:
        return EvaluationTier(
            tier=0, name="WARMING_UP",
            pending_metrics=["Brier", "WinRate", "Sharpe", "Calmar", "IR", "EdgeRealization"],
            reason=f"Tier 0: need ≥50 ticks (have {n_ticks}) and ≥3 resolved (have {n_resolved})",
        )
    if n_ticks < 500 or n_resolved < 10:
        return EvaluationTier(
            tier=1, name="SANITY",
            trustworthy_metrics=["sizing_function", "logging", "tick_throughput"],
            pending_metrics=["Brier", "WinRate", "Sharpe", "Calmar", "IR"],
            reason=f"Tier 1: need ≥500 ticks (have {n_ticks}) and ≥10 resolved (have {n_resolved})",
        )
    if n_ticks < 3000 or n_resolved < 100:
        return EvaluationTier(
            tier=2, name="ADAPTIVITY",
            trustworthy_metrics=["Brier", "WinRate", "Calibration", "Sharpe", "Calmar"],
            pending_metrics=["IR", "EdgeRealization", "FullStrategyValidation"],
            reason=f"Tier 2: need ≥3000 ticks (have {n_ticks}) and ≥100 resolved (have {n_resolved})",
        )
    return EvaluationTier(
        tier=3, name="STRATEGY_VALIDATION",
        trustworthy_metrics=[
            "Brier", "WinRate", "Calibration", "Sharpe", "Calmar",
            "IR", "EdgeRealization", "FullStrategyValidation",
        ],
        reason=f"Tier 3: full statistical confidence ({n_ticks} ticks, {n_resolved} resolved)",
    )


# ---------------------------------------------------------------------------
# CLI / summary entry point
# ---------------------------------------------------------------------------

def summarize() -> dict[str, Any]:
    """Compute every metric over the current decision logs and return a dict."""
    forecasts = _read_jsonl(FORECAST_LOG)
    actions = _read_jsonl(ACTION_LOG)
    fills = _read_jsonl(FILL_LOG)
    resolutions = _read_jsonl(RESOLUTION_LOG)

    n_ticks_seen = len({a.get("tick_ts") for a in actions})
    n_buy_decisions = sum(1 for a in actions if a.get("decision") == "BUY_OK")
    n_skip_decisions = sum(1 for a in actions if str(a.get("decision", "")).startswith("SKIP_"))
    n_resolved = len(resolutions)

    tier = classify_tier(n_ticks_seen, n_resolved)

    summary: dict[str, Any] = {
        "tier":             {"id": tier.tier, "name": tier.name, "reason": tier.reason},
        "n_ticks_seen":     n_ticks_seen,
        "n_buy_decisions":  n_buy_decisions,
        "n_skip_decisions": n_skip_decisions,
        "n_fills":          len(fills),
        "n_resolved":       n_resolved,
        "trustworthy":      tier.trustworthy_metrics,
        "pending":          tier.pending_metrics,
    }

    brier = brier_score(resolutions)
    ece = expected_calibration_error(resolutions)
    edge_real = edge_realization_rate(resolutions)
    buckets = [b.as_dict() for b in calibration_buckets(resolutions)]

    summary["metrics"] = {
        "brier_score":          brier,
        "ece":                  ece,
        "edge_realization":     edge_real,
        "calibration_buckets":  buckets,
    }

    return summary


def main() -> None:
    import sys
    s = summarize()
    print(json.dumps(s, indent=2, default=str))
    sys.exit(0)


if __name__ == "__main__":
    main()
