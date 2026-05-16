"""Deterministic ActionStage: forecasts -> sized, capped trade intents.

Why deterministic (no second LLM call):
  * Removes one LLM round-trip per market -> ~40% latency win per tick.
  * Removes a $0.01-$0.05 cost per market -> material savings over 1344 ticks.
  * Makes sizing reproducible from (p_yes, market_quote, equity, positions),
    which is the right contract for a Trading-Track agent scored on Sharpe + PnL.

Sizing pipeline for each forecast (in order):
  1. Compute edge for BUY YES and BUY NO; pick the better side.
  2. Drop if best edge < MIN_EDGE (or MIN_EDGE_RELAXED when under the
     14-trade floor).
  3. Half-Kelly fraction of equity, capped at `max_position_pct_of_equity`.
  4. Cap notional at MAX_NOTIONAL_PER_MARKET (1000) per market.
  5. Cap running gross exposure at MAX_GROSS_EXPOSURE (10_000).
  6. Skip if would open a NEW position while at MAX_OPEN_POSITIONS (30).
  7. Skip if would exceed MAX_TRADES_PER_TICK (20).
  8. If holding the opposite side, emit a SELL on the held side instead of
     a BUY on the opposite (per CustomAgentTradingRules.pdf — the server
     silently drops unfilled opposite shares otherwise).

Output schema matches the SDK contract so ExperimentRunner can submit without
adapter glue: each intent has run_id, tick_ts, market_id, question, action,
side, shares, rationale.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from ai_prophet.trade.agent.stages.action import ActionStage
from ai_prophet.trade.agent.stages.base import StageResult
from ai_prophet.trade.core import TickContext
from ai_prophet.trade.core.tick_context import CandidateMarket
from ai_prophet.trade.llm import LLMClient

from agent.settings import RiskConfig, TradingConstraints

logger = logging.getLogger(__name__)


class RiskAwareActionStage(ActionStage):
    """Replaces the SDK ActionStage with a deterministic, rule-bound sizer."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        constraints: TradingConstraints | None = None,
        risk: RiskConfig | None = None,
        min_size_usd: float | None = None,
    ) -> None:
        # `llm_client` is unused but accepted for SDK signature compatibility.
        self.constraints = constraints or TradingConstraints()
        self.risk = risk or RiskConfig()
        super().__init__(
            llm_client=llm_client,
            min_size_usd=min_size_usd if min_size_usd is not None else self.risk.min_intent_size_usd,
        )

    @property
    def name(self) -> str:
        return "action"

    # ------------------------------------------------------------------
    # Stage entry point
    # ------------------------------------------------------------------

    def execute(
        self,
        tick_ctx: TickContext,
        previous_results: dict[str, StageResult],
    ) -> StageResult:
        if "forecast" not in previous_results:
            return StageResult(
                stage_name=self.name, success=False, data={},
                error="Forecast stage result not found",
            )

        forecasts: dict[str, dict[str, Any]] = (
            previous_results["forecast"].data.get("forecasts", {}) or {}
        )
        if not forecasts:
            return StageResult(
                stage_name=self.name, success=True,
                data={"intents": [], "decisions": {}},
            )

        # Decide effective min_edge for this tick.
        min_edge = self._effective_min_edge(tick_ctx)
        equity = float(tick_ctx.equity)
        existing_market_ids = {p.market_id for p in tick_ctx.positions}
        n_open_positions = len(existing_market_ids)

        # Score every (market, side) opportunity; sort by edge desc so we
        # spend the per-tick budget on the highest-EV ideas first.
        scored: list[dict[str, Any]] = []
        for market_id, forecast in forecasts.items():
            market_info = tick_ctx.get_candidate(market_id)
            if market_info is None:
                logger.warning("Action: market %s missing from candidates", market_id)
                continue

            try:
                p_yes = float(forecast.get("p_yes", 0.5))
            except (TypeError, ValueError):
                p_yes = 0.5

            decision = self._score_market(
                market_id=market_id,
                p_yes=p_yes,
                rationale=forecast.get("rationale", "") or "",
                market=market_info,
                tick_ctx=tick_ctx,
                min_edge=min_edge,
            )
            if decision is not None:
                scored.append(decision)

        # Highest |edge| first.
        scored.sort(key=lambda d: abs(d["edge"]), reverse=True)

        # Apply per-tick + gross-exposure + open-positions caps.
        intents: list[dict[str, Any]] = []
        decisions: dict[str, dict[str, Any]] = {}
        running_notional = 0.0
        new_positions_opened = 0

        for d in scored:
            if len(intents) >= self.constraints.max_trades_per_tick:
                logger.info("Action: hit MAX_TRADES_PER_TICK=%d", self.constraints.max_trades_per_tick)
                break

            mid = d["market_id"]
            is_new_position = mid not in existing_market_ids

            if is_new_position and n_open_positions + new_positions_opened >= self.constraints.max_open_positions:
                logger.info("Action: skip %s, would exceed MAX_OPEN_POSITIONS", mid)
                continue

            size_usd = float(d["size_usd"])
            if running_notional + size_usd > self.constraints.max_gross_exposure:
                trimmed = self.constraints.max_gross_exposure - running_notional
                if trimmed < self.min_size_usd:
                    logger.info("Action: skip %s, gross exposure exhausted", mid)
                    continue
                logger.info(
                    "Action: trim %s from $%.0f to $%.0f to respect MAX_GROSS_EXPOSURE",
                    mid, size_usd, trimmed,
                )
                size_usd = trimmed
                d["size_usd"] = size_usd
                d["shares"] = self._size_to_shares(size_usd, d["price"])
                # Skip if trimming dropped us under per-market min.
                if size_usd < self.min_size_usd:
                    continue

            intent = self._to_intent(d, tick_ctx, equity_now=equity)
            if intent is None:
                continue

            intents.append(intent)
            decisions[mid] = self._decision_summary(d, equity_now=equity)
            running_notional += size_usd
            if is_new_position:
                new_positions_opened += 1

        logger.info(
            "Action stage complete: %d intents from %d forecasts "
            "(min_edge=%.3f, gross_notional=$%.0f)",
            len(intents), len(forecasts), min_edge, running_notional,
        )

        return StageResult(
            stage_name=self.name,
            success=True,
            data={"intents": intents, "decisions": decisions},
        )

    # ------------------------------------------------------------------
    # Per-market scoring
    # ------------------------------------------------------------------

    def _score_market(
        self,
        *,
        market_id: str,
        p_yes: float,
        rationale: str,
        market: CandidateMarket,
        tick_ctx: TickContext,
        min_edge: float,
    ) -> dict[str, Any] | None:
        """Compute (side, edge, sizing) for one market; return None to skip.

        Output keys: market_id, action, side, price, edge, size_usd, shares,
        rationale, p_yes, p_no, question, is_position_flip.
        """
        p_yes = _clamp01(p_yes)
        p_no = 1.0 - p_yes

        yes_ask = float(market.yes_ask)
        no_ask = float(market.no_ask)

        edge_yes = p_yes - yes_ask if yes_ask > 0 else -1.0
        edge_no = p_no - no_ask if no_ask > 0 else -1.0

        # Pick the side with the larger positive edge.
        if edge_yes >= edge_no:
            best_side = "YES"
            best_edge = edge_yes
            best_price = yes_ask
        else:
            best_side = "NO"
            best_edge = edge_no
            best_price = no_ask

        if best_edge < min_edge:
            return None  # No real edge, no trade.

        existing = tick_ctx.get_position(market_id)

        # ------------------------------------------------------------------
        # Position-flip-as-sell rule: never BUY opposite side of an existing
        # position. Emit SELL on the held side to reduce/close instead.
        # ------------------------------------------------------------------
        if existing is not None and existing.side != best_side:
            shares_held = float(existing.shares)
            if shares_held <= 0:
                # Shouldn't happen but be defensive.
                pass
            else:
                # Sell down the held side. Price for executing the SELL is
                # best_bid on the held side (we hit the bid).
                held_side = existing.side
                sell_price = (
                    float(market.yes_bid) if held_side == "YES" else float(market.no_bid)
                )
                if sell_price <= 0:
                    return None
                sell_notional = shares_held * sell_price
                return {
                    "market_id":        market_id,
                    "action":           "SELL",
                    "side":             held_side,
                    "price":            sell_price,
                    "edge":             best_edge,  # reason we wanted to flip
                    "size_usd":         sell_notional,
                    "shares":           shares_held,
                    "rationale": (
                        f"Flip-as-sell: held {held_side} {shares_held:.2f} shares; "
                        f"forecast prefers {best_side} (edge {best_edge:+.3f}). "
                        f"Selling to flat, will re-enter next tick. {rationale}"
                    ).strip(),
                    "p_yes":             p_yes,
                    "p_no":              p_no,
                    "question":          market.question,
                    "is_position_flip":  True,
                }

        # ------------------------------------------------------------------
        # Standard BUY path.
        # ------------------------------------------------------------------
        if best_price <= 0 or best_price >= 1.0:
            return None  # Degenerate or resolved-near-expiry market.

        # Half-Kelly: f = 0.5 * edge / (1 - price). Capped at the position cap.
        kelly_frac = self.risk.kelly_fraction * (best_edge / max(1.0 - best_price, 1e-6))
        kelly_frac = max(0.0, min(self.risk.max_position_pct_of_equity, kelly_frac))

        equity = float(tick_ctx.equity)
        size_usd = kelly_frac * equity

        # Per-market notional cap.
        size_usd = min(size_usd, self.constraints.max_notional_per_market)

        if size_usd < self.min_size_usd:
            return None

        shares = self._size_to_shares(size_usd, best_price)
        if shares <= 0:
            return None

        return {
            "market_id":         market_id,
            "action":            "BUY",
            "side":              best_side,
            "price":             best_price,
            "edge":              best_edge,
            "size_usd":          size_usd,
            "shares":            shares,
            "rationale": (
                f"edge={best_edge:+.3f} (p_yes={p_yes:.3f}, price={best_price:.3f}); "
                f"half-Kelly size ${size_usd:.0f} = {kelly_frac:.2%} of ${equity:.0f}. "
                f"{rationale}"
            ).strip(),
            "p_yes":             p_yes,
            "p_no":              p_no,
            "question":          market.question,
            "is_position_flip":  False,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _effective_min_edge(self, tick_ctx: TickContext) -> float:
        """Relax min_edge below the floor when on pace to miss 14 fills."""
        floor = self.risk.trade_floor_count
        if tick_ctx.total_fills < floor:
            relaxed = self.risk.min_edge_relaxed
            if relaxed < self.risk.min_edge:
                logger.info(
                    "Action: relaxing min_edge %.3f -> %.3f (only %d/%d lifetime fills)",
                    self.risk.min_edge, relaxed, tick_ctx.total_fills, floor,
                )
                return relaxed
        return self.risk.min_edge

    @staticmethod
    def _size_to_shares(size_usd: float, price: float) -> float:
        if price <= 0:
            return 0.0
        return round(size_usd / price, 2)

    @staticmethod
    def _to_intent(
        d: dict[str, Any], tick_ctx: TickContext, *, equity_now: float,
    ) -> dict[str, Any] | None:
        if d.get("shares", 0) <= 0:
            return None
        return {
            "run_id":    tick_ctx.run_id,
            "tick_ts":   tick_ctx.tick_ts,
            "market_id": d["market_id"],
            "question":  d.get("question", d["market_id"]),
            "action":    d["action"],
            "side":      d["side"],
            "shares":    f"{d['shares']:.2f}",
            "rationale": d.get("rationale", ""),
        }

    @staticmethod
    def _decision_summary(d: dict[str, Any], *, equity_now: float) -> dict[str, Any]:
        rec = (
            "BUY_YES" if d["action"] == "BUY" and d["side"] == "YES"
            else "BUY_NO" if d["action"] == "BUY" and d["side"] == "NO"
            else f"SELL_{d['side']}"
        )
        return {
            "recommendation":   rec,
            "size_usd":         round(float(d["size_usd"]), 2),
            "rationale":        d.get("rationale", ""),
            "edge":             round(float(d.get("edge", 0.0)), 4),
            "is_position_flip": bool(d.get("is_position_flip", False)),
        }


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)
