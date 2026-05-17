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
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from ai_prophet.trade.agent.stages.action import ActionStage
from ai_prophet.trade.agent.stages.base import StageResult
from ai_prophet.trade.core import TickContext
from ai_prophet.trade.core.tick_context import CandidateMarket
from ai_prophet.trade.llm import LLMClient

from agent.calibration import resolution_proximity_multiplier
from agent.settings import RiskConfig, TradingConstraints
from agent.stages.text_review import _category

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
        # Decide effective min_edge for this tick.
        min_edge = self._effective_min_edge(tick_ctx)
        equity = float(tick_ctx.equity)
        existing_market_ids = {p.market_id for p in tick_ctx.positions}
        n_open_positions = len(existing_market_ids)
        
        current_port_var = 0.0
        if equity > 0:
            for pos in tick_ctx.positions:
                mark = float(pos.current_price)
                weight = float(pos.shares * pos.current_price) / equity
                pos_var = mark * (1.0 - mark)
                current_port_var += (weight ** 2) * pos_var

        # Score every (market, side) opportunity; sort by edge desc so we
        # spend the per-tick budget on the highest-EV ideas first.
        scored: list[dict[str, Any]] = []
        for market_id, forecast in forecasts.items():
            market_info = tick_ctx.get_candidate(market_id)
            if market_info is None and not market_id.startswith("kalshi:"):
                market_info = tick_ctx.get_candidate("kalshi:" + market_id)
                if market_info is not None:
                    market_id = "kalshi:" + market_id  # normalize for intent submission
            if market_info is None:
                logger.warning("Action: market %s missing from candidates", market_id)
                continue

            try:
                p_yes = float(forecast.get("p_yes", 0.5))
            except (TypeError, ValueError):
                p_yes = 0.5

            # raw_gap and llm_var come from the forecast sidecar populated by
            # CalibratedForecastStage. Falls back gracefully when running under
            # tests or the SDK base stage (no sidecar entry → no extra gating).
            # Try both the normalized ID and the bare ID (review may omit prefix).
            from agent.stages.calibrated_forecast import TICK_FORECAST_EXTRAS
            bare_id = (
                market_id[len("kalshi:"):]
                if market_id.startswith("kalshi:") else market_id
            )
            extras = (
                TICK_FORECAST_EXTRAS.get(market_id)
                or TICK_FORECAST_EXTRAS.get(bare_id, {})
            )
            raw_gap_val = extras.get("raw_gap")
            raw_gap: float | None = (
                float(raw_gap_val) if raw_gap_val is not None else None
            )
            try:
                llm_var = float(extras.get("llm_var", 0.0))
            except (TypeError, ValueError):
                llm_var = 0.0
            conf_model_val = extras.get("conf_model")
            conf_model: float | None = (
                float(conf_model_val)
                if conf_model_val is not None else None
            )

            decision = self._score_market(
                market_id=market_id,
                p_yes=p_yes,
                rationale=forecast.get("rationale", "") or "",
                market=market_info,
                tick_ctx=tick_ctx,
                min_edge=min_edge,
                raw_gap=raw_gap,
                llm_var=llm_var,
                current_port_var=current_port_var,
                conf_model=conf_model,
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

        # Count open positions by category (cross-tick diversity cap).
        open_cat_counts: dict[str, int] = {}
        for pos in tick_ctx.positions:
            pos_info = tick_ctx.get_candidate(pos.market_id)
            pos_q = pos_info.question if pos_info else ""
            cat = _category(pos.market_id, pos_q)
            open_cat_counts[cat] = open_cat_counts.get(cat, 0) + 1

        # Per-tick new-buy count by category (enforces max_intents_per_category).
        category_counts: dict[str, int] = {}

        # 1. Mechanical exits — take-profit and stop-loss before forecast BUYs.
        tp_scored, tp_handled = self._take_profit_intents(tick_ctx)
        sl_scored, sl_handled = self._stop_loss_intents(tick_ctx)
        exit_handled = tp_handled | sl_handled
        for d in tp_scored + sl_scored:
            if len(intents) >= self.constraints.max_trades_per_tick:
                logger.info("Action: mechanical exit blocked by MAX_TRADES_PER_TICK")
                break
            intent = self._to_intent(d, tick_ctx, equity_now=equity)
            if intent is None:
                continue
            intents.append(intent)
            decisions[d["market_id"]] = self._decision_summary(d, equity_now=equity)

        # 2. Forecast-based BUYs (and flip-as-sell).
        for d in scored:
            if len(intents) >= self.constraints.max_trades_per_tick:
                logger.info("Action: hit MAX_TRADES_PER_TICK=%d", self.constraints.max_trades_per_tick)
                break

            mid = d["market_id"]
            if mid in exit_handled:
                continue  # already exited mechanically this tick
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

            cat = _category(mid, d.get("question", ""))
            # Per-tick new-buy cap.
            if category_counts.get(cat, 0) >= self.constraints.max_intents_per_category:
                logger.info(
                    "Action: skip %s [%s], per-tick cap %d reached",
                    mid, cat, self.constraints.max_intents_per_category,
                )
                continue
            # Cross-tick total open cap: existing + new intents this tick.
            total_in_cat = (
                open_cat_counts.get(cat, 0) + category_counts.get(cat, 0)
            )
            if total_in_cat >= self.constraints.max_open_per_category:
                logger.info(
                    "Action: skip %s [%s], total open cap %d reached",
                    mid, cat, self.constraints.max_open_per_category,
                )
                continue

            intent = self._to_intent(d, tick_ctx, equity_now=equity)
            if intent is None:
                continue

            intents.append(intent)
            decisions[mid] = self._decision_summary(d, equity_now=equity)
            category_counts[cat] = category_counts.get(cat, 0) + 1
            running_notional += size_usd
            if is_new_position:
                new_positions_opened += 1

        logger.info(
            "Action stage complete: %d intents (%d tp-sells, %d sl-sells, %d forecasts) "
            "(min_edge=%.3f, gross_notional=$%.0f)",
            len(intents), len(tp_scored), len(sl_scored),
            len(forecasts), min_edge, running_notional,
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
        raw_gap: float | None = None,
        llm_var: float = 0.0,
        current_port_var: float = 0.0,
        conf_model: float | None = None,
    ) -> dict[str, Any] | None:
        """Compute (side, edge, sizing) for one market; return None to skip.

        Output keys: market_id, action, side, price, edge, size_usd, shares,
        rationale, p_yes, p_no, question, is_position_flip.

        Gate logic (guide §2.2):
          - When raw_gap is provided (ensemble path): gate on |raw_gap| >= min_raw_gap.
          - Otherwise (tests / base SDK): gate on blended edge >= min_edge.
        Sizing uses the blended edge in both cases (conservative Kelly input).
        """
        # Confidence gate: when ensemble sampling nearly or fully failed,
        # p_model defaults to 0.5 (no information). Skip rather than trade
        # on a meaningless uniform prior disguised as a large edge.
        if conf_model is not None and conf_model < self.risk.min_conf_model:
            logger.debug(
                "Action: skip %s, conf_model=%.2f < min %.2f"
                " (ensemble sampling failed)",
                market_id, conf_model, self.risk.min_conf_model,
            )
            return None

        # Time-to-resolution gate: prefer contracts that resolve inside the
        # 14-day comp window. Computed once here so the BUY path can both
        # hard-skip far-out markets and shrink mid-horizon ones in sizing.
        # The flip-as-sell path below intentionally bypasses this (we always
        # want to close stale positions, even on far-dated markets).
        days_to_res = _days_to_resolution(market, tick_ctx)
        time_mult = resolution_proximity_multiplier(
            days_to_res if days_to_res is not None else 0.0,
            near_term_days=self.risk.near_term_horizon_days,
            max_days=self.risk.max_days_to_resolution,
            floor=self.risk.far_term_size_floor,
        ) if days_to_res is not None else 1.0

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

        # Gate: raw-gap (pre-blend) when available; blended edge otherwise.
        if raw_gap is not None:
            if abs(raw_gap) < self.risk.min_raw_gap:
                return None
            if abs(raw_gap) > self.risk.max_raw_gap_hard:
                logger.debug(
                    "Action: skip %s, |raw_gap| %.2f > max_raw_gap_hard %.2f"
                    " (model knowledge likely stale)",
                    market_id, abs(raw_gap), self.risk.max_raw_gap_hard,
                )
                return None
        else:
            if best_edge < min_edge:
                return None

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
        # Standard BUY path with Volatility Penalization.
        # ------------------------------------------------------------------
        # Hard time-to-resolution gate. Days outside [0, max_days_to_resolution]
        # collapse to time_mult == 0; skip opening a new position there.
        if days_to_res is not None and time_mult <= 0.0:
            logger.debug(
                "Action: skip %s, days_to_resolution=%.1f outside [0, %.1f]",
                market_id, days_to_res, self.risk.max_days_to_resolution,
            )
            return None

        if best_price <= 0 or best_price >= 1.0:
            return None  # Degenerate or resolved-near-expiry market.

        # Spread gate: wide spreads eat edge at entry. Exempt from flip-as-sell.
        yes_bid = float(market.yes_bid)
        no_bid = float(market.no_bid)
        if best_side == "YES":
            spread_mid = (yes_ask + yes_bid) / 2.0
            spread_pct = (yes_ask - yes_bid) / spread_mid if spread_mid > 0 else 1.0
        else:
            spread_mid = (no_ask + no_bid) / 2.0
            spread_pct = (no_ask - no_bid) / spread_mid if spread_mid > 0 else 1.0
        if spread_pct > self.risk.max_spread_pct:
            logger.debug(
                "Action: skip %s, spread %.1f%% > max %.1f%%",
                market_id, spread_pct * 100, self.risk.max_spread_pct * 100,
            )
            return None

        # Epistemic shrinkage (James-Stein): reduce when LLM ensemble shows high
        # variance, since σ²_position = p(1-p) + Var(LLM_samples).
        from agent.calibration import epistemic_shrink
        shrink = epistemic_shrink(p_yes, llm_var)
        
        # 1. Base Exact Kelly: f* = edge / (1 - price)
        base_kelly = best_edge / max(1.0 - best_price, 1e-6)
        
        # 2. Volatility Penalization (Sharpe Optimization)
        
        asset_var = best_price * (1.0 - best_price)
        asset_std_dev = max(asset_var ** 0.5, 1e-6)
        
        # Base dampener: penalizes variance for standard coin-toss bets
        base_dampener = min(1.0, 0.4 / asset_std_dev)
        
        # --- TUNED: The Mid-Curve Sizing Paradox Bypass ---
        # Adjusted for steady, noticeable PnL generation.
        # We begin releasing the brakes at an 8% edge (0.08) rather than 15%, 
        # allowing the agent to capture high-quality (but not perfectly rare) setups.
        if best_edge > 0.08:
            # Scales smoothly: begins recovering at 0.08 edge, fully bypasses penalty at 0.33 edge.
            # A multiplier of 4.0 ensures the transition is steady, preventing sudden 
            # massive spikes in exposure that would ruin the Sharpe ratio.
            recovery_factor = min(1.0, (best_edge - 0.08) * 4.0) 
            sharpe_dampener = base_dampener + ((1.0 - base_dampener) * recovery_factor)
        else:
            sharpe_dampener = base_dampener
        # ------------------------------------------------

        # B. Portfolio-Level Penalty (Heat Check)
        heat_penalty = max(0.1, 1.0 - (current_port_var / self.risk.target_portfolio_var))

        # C. Calculate Adjusted Size
        kelly_frac = (
            self.risk.kelly_fraction
            * shrink
            * base_kelly
            * sharpe_dampener
            * heat_penalty
            * time_mult        # bias capital toward sooner-resolving contracts
        )

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
                f"adj-Kelly size ${size_usd:.0f} (heat_pen={heat_penalty:.2f}, "
                f"sharpe_pen={sharpe_dampener:.2f}, time_mult={time_mult:.2f}"
                f"{f', days_to_res={days_to_res:.1f}' if days_to_res is not None else ''})."
                f" {rationale}"
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

    def _take_profit_intents(
        self, tick_ctx: TickContext
    ) -> tuple[list[dict[str, Any]], set[str]]:
        """Return scored dicts for positions with >= take_profit_threshold gain.

        Uses the same dict shape as _score_market so _to_intent and
        _decision_summary can consume them without special-casing.
        """
        scored: list[dict[str, Any]] = []
        handled: set[str] = set()
        threshold = self.risk.take_profit_threshold

        for pos in tick_ctx.positions:
            entry = float(pos.avg_entry_price)
            mark = float(pos.current_price)
            if entry <= 0:
                continue
            gain_pct = (mark - entry) / entry
            if gain_pct < threshold:
                continue

            market_info = tick_ctx.get_candidate(pos.market_id)
            if market_info is not None:
                sell_price = (
                    float(market_info.yes_bid) if pos.side == "YES"
                    else float(market_info.no_bid)
                )
                question = market_info.question
            else:
                sell_price = mark
                question = pos.market_id

            if sell_price <= 0:
                continue

            shares = float(pos.shares)
            scored.append({
                "market_id":        pos.market_id,
                "action":           "SELL",
                "side":             pos.side,
                "price":            sell_price,
                "edge":             gain_pct,
                "size_usd":         shares * sell_price,
                "shares":           shares,
                "rationale": (
                    f"Take-profit: mark {mark:.3f} vs entry {entry:.3f} "
                    f"(+{gain_pct:.1%} >= {threshold:.0%} threshold)."
                ),
                "question":         question,
                "is_position_flip": False,
            })
            handled.add(pos.market_id)

        if scored:
            logger.info("Action: %d take-profit sell(s) triggered", len(scored))
        return scored, handled

    def _stop_loss_intents(
        self, tick_ctx: TickContext
    ) -> tuple[list[dict[str, Any]], set[str]]:
        """Return scored dicts for positions with >= stop_loss_threshold loss.

        Same dict shape as _score_market so _to_intent/_decision_summary work.
        """
        scored: list[dict[str, Any]] = []
        handled: set[str] = set()
        threshold = self.risk.stop_loss_threshold  # positive, e.g. 0.20

        for pos in tick_ctx.positions:
            entry = float(pos.avg_entry_price)
            mark = float(pos.current_price)
            if entry <= 0:
                continue
            loss_pct = (mark - entry) / entry  # negative when a loss
            if loss_pct > -threshold:
                continue

            market_info = tick_ctx.get_candidate(pos.market_id)
            if market_info is not None:
                sell_price = (
                    float(market_info.yes_bid) if pos.side == "YES"
                    else float(market_info.no_bid)
                )
                question = market_info.question
            else:
                sell_price = mark
                question = pos.market_id

            if sell_price <= 0:
                continue

            shares = float(pos.shares)
            scored.append({
                "market_id":        pos.market_id,
                "action":           "SELL",
                "side":             pos.side,
                "price":            sell_price,
                "edge":             abs(loss_pct),
                "size_usd":         shares * sell_price,
                "shares":           shares,
                "rationale": (
                    f"Stop-loss: mark {mark:.3f} vs entry {entry:.3f} "
                    f"({loss_pct:.1%} <= -{threshold:.0%} threshold)."
                ),
                "question":         question,
                "is_position_flip": False,
            })
            handled.add(pos.market_id)

        if scored:
            logger.info("Action: %d stop-loss sell(s) triggered", len(scored))
        return scored, handled

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


def _days_to_resolution(
    market: CandidateMarket, tick_ctx: TickContext
) -> float | None:
    """Days between the tick boundary and the market's resolution time.

    Returns None when the market has no usable resolution_time (defensive — we
    treat that as "unknown horizon" and let the trade proceed at full size
    rather than over-filtering on malformed data).
    """
    res = getattr(market, "resolution_time", None)
    if not isinstance(res, datetime):
        return None
    tick_ts = tick_ctx.tick_ts
    # Both should be tz-aware per the SDK contract; if either is naive, assume
    # UTC so the subtraction doesn't raise.
    if res.tzinfo is None:
        res = res.replace(tzinfo=timezone.utc)
    if tick_ts.tzinfo is None:
        tick_ts = tick_ts.replace(tzinfo=timezone.utc)
    return (res - tick_ts).total_seconds() / 86400.0
