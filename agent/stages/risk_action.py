"""Nailong Elite ActionStage — deterministic, duration-aware, concentration-safe.

Sizing pipeline for each forecast (in order):
  1. Pick the better side (YES or NO) by edge.
  2. Gate on raw_gap (pre-blend model deviation) or blended edge.
  3. Base Kelly with epistemic shrinkage and Sharpe dampening.
  4. **Duration penalty**: size *= 1 / (1 + days_to_resolution / half_life).
     Long-dated markets get less size because capital is locked longer
     and we won't see resolution within the 14-day eval window.
  5. Heat penalty (portfolio variance).
  6. Hard caps: per-market notional ($1000), gross exposure ($10k),
     open positions (30), trades per tick (20), trades per 24h (100).
  7. **Long-dated cap**: combined exposure on markets > 180 days out
     cannot exceed 30% of equity.
  8. **Same-event cap**: combined exposure on markets sharing an event
     prefix (e.g. all 2028 Dem primary markets) cannot exceed $300.
  9. Take-profit / stop-loss exits run before BUYs.
 10. Position flips: SELL the held side AND BUY the new side in the
     same tick (per guide — both intents in one submission).
 11. Per-decision structured JSONL log to data/decisions/action.jsonl.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from ai_prophet.trade.agent.stages.action import ActionStage
from ai_prophet.trade.agent.stages.base import StageResult
from ai_prophet.trade.core import TickContext
from ai_prophet.trade.core.tick_context import CandidateMarket
from ai_prophet.trade.llm import LLMClient

from agent.settings import RiskConfig, TradingConstraints, CalibrationConfig
from agent.stages.text_review import _category

logger = logging.getLogger(__name__)

_ACTION_LOG_PATH = Path("data/decisions/action.jsonl")
_FILL_LOG_PATH = Path("data/decisions/fills.jsonl")

# Rolling 24h fill counter, persisted in-memory across ticks within one
# process. SDK restarts will reset this — that's acceptable for the eval
# window because the server-side daily cap is what we ultimately respect.
_DAILY_FILL_TS: deque[float] = deque(maxlen=200)  # timestamps of recent fills

# Trailing event prefix pattern. We strip the most specific suffix to group:
#   KXDEMPRIMARY-2028-SLOTKIN  -> KXDEMPRIMARY-2028
#   KXNBAMVP-2026-DONCIC       -> KXNBAMVP-2026
_EVENT_PREFIX_RE = re.compile(r"^(.+?-\d{2,4})(?:-[A-Z0-9]+)+$")


def _event_prefix(market_id: str) -> str:
    """Strip the candidate-specific suffix from a Kalshi market_id to get the
    underlying event grouping. Used for same-event concentration caps."""
    mid = market_id.removeprefix("kalshi:")
    m = _EVENT_PREFIX_RE.match(mid)
    return m.group(1) if m else mid


class RiskAwareActionStage(ActionStage):
    """Replaces the SDK ActionStage with a deterministic, rule-bound sizer."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        constraints: TradingConstraints | None = None,
        risk: RiskConfig | None = None,
        calibration: CalibrationConfig | None = None,
        min_size_usd: float | None = None,
        target_total_ticks: int = 1500,
    ) -> None:
        # `llm_client` is unused but accepted for SDK signature compatibility.
        self.constraints = constraints or TradingConstraints()
        self.risk = risk or RiskConfig()
        self.calibration = calibration or CalibrationConfig()
        self.target_total_ticks = target_total_ticks
        super().__init__(
            llm_client=llm_client,
            min_size_usd=min_size_usd if min_size_usd is not None else self.risk.min_intent_size_usd,
        )
        # Best-effort: ensure log dir exists.
        try:
            _ACTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    @property
    def name(self) -> str:
        return "action"

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _write_action_log(self, entry: dict) -> None:
        try:
            with open(_ACTION_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as exc:
            logger.debug("Action log write failed: %s", exc)

    def _write_fill_log(self, entry: dict) -> None:
        try:
            with open(_FILL_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as exc:
            logger.debug("Fill log write failed: %s", exc)

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
        tick_ts = getattr(tick_ctx, "tick_ts", None)
        min_edge = self._effective_min_edge(tick_ctx)
        equity = float(tick_ctx.equity)
        existing_market_ids = {p.market_id for p in tick_ctx.positions}
        n_open_positions = len(existing_market_ids)

        # Portfolio variance for heat penalty.
        current_port_var = 0.0
        if equity > 0:
            for pos in tick_ctx.positions:
                mark = float(pos.current_price)
                weight = float(pos.shares * pos.current_price) / equity
                pos_var = mark * (1.0 - mark)
                current_port_var += (weight ** 2) * pos_var

        # Existing exposure by event prefix (for same-event cap).
        event_exposure: dict[str, float] = defaultdict(float)
        long_dated_exposure = 0.0
        for pos in tick_ctx.positions:
            notional = float(pos.shares * pos.current_price)
            event_exposure[_event_prefix(pos.market_id)] += notional

        # Daily rolling fill count.
        now_ts = self._tick_ts_to_epoch(tick_ts)
        self._prune_daily_fills(now_ts)
        daily_remaining = max(
            0,
            self.constraints.max_trades_per_day - len(_DAILY_FILL_TS),
        )

        # Score every (market, side) opportunity.
        scored: list[dict[str, Any]] = []
        for market_id, forecast in forecasts.items():
            market_info = tick_ctx.get_candidate(market_id)
            if market_info is None and not market_id.startswith("kalshi:"):
                market_info = tick_ctx.get_candidate("kalshi:" + market_id)
                if market_info is not None:
                    market_id = "kalshi:" + market_id

            if market_info is None:
                logger.warning("Action: market %s missing from candidates", market_id)
                continue

            try:
                p_yes = float(forecast.get("p_yes", 0.5))
            except (TypeError, ValueError):
                p_yes = 0.5

            from agent.stages.calibrated_forecast import TICK_FORECAST_EXTRAS
            extras = TICK_FORECAST_EXTRAS.get(market_id, {})
            raw_gap_val = extras.get("raw_gap")
            raw_gap: float | None = (
                float(raw_gap_val) if raw_gap_val is not None else None
            )
            try:
                llm_var = float(extras.get("llm_var", 0.0))
            except (TypeError, ValueError):
                llm_var = 0.0
            try:
                sigma = float(extras.get("sigma", 0.10))
            except (TypeError, ValueError):
                sigma = 0.10
            try:
                conf_model = float(extras.get("conf_model", 0.0))
            except (TypeError, ValueError):
                conf_model = 0.0
            days_to_resolution = extras.get("days_to_resolution")
            if days_to_resolution is not None:
                try:
                    days_to_resolution = float(days_to_resolution)
                except (TypeError, ValueError):
                    days_to_resolution = None

            decision = self._score_market(
                market_id=market_id,
                p_yes=p_yes,
                rationale=forecast.get("rationale", "") or "",
                market=market_info,
                tick_ctx=tick_ctx,
                min_edge=min_edge,
                raw_gap=raw_gap,
                llm_var=llm_var,
                sigma=sigma,
                conf_model=conf_model,
                days_to_resolution=days_to_resolution,
                current_port_var=current_port_var,
            )
            if decision is not None:
                scored.append(decision)

        scored.sort(key=lambda d: abs(d["edge"]), reverse=True)

        intents: list[dict[str, Any]] = []
        decisions: dict[str, dict[str, Any]] = {}
        running_notional = 0.0
        new_positions_opened = 0
        category_counts: dict[str, int] = {}
        intents_this_tick = 0

        # 1. Take-profit + stop-loss exits.
        tp_scored, tp_handled = self._take_profit_intents(tick_ctx)
        sl_scored, sl_handled = self._stop_loss_intents(tick_ctx)
        exit_handled = tp_handled | sl_handled
        for d in tp_scored + sl_scored:
            if intents_this_tick >= self.constraints.max_trades_per_tick:
                break
            if daily_remaining <= 0:
                logger.info("Action: skip TP/SL — daily fill cap reached (%d/100)", len(_DAILY_FILL_TS))
                break
            intent = self._to_intent(d, tick_ctx, equity_now=equity)
            if intent is None:
                continue
            intents.append(intent)
            intents_this_tick += 1
            daily_remaining -= 1
            # Free the event exposure since we're selling.
            event_exposure[_event_prefix(d["market_id"])] -= float(d.get("size_usd", 0.0))
            decisions[d["market_id"]] = self._decision_summary(d, equity_now=equity)
            self._write_action_log({
                "tick_ts": str(tick_ts), "market_id": d["market_id"],
                "action": "SELL", "side": d.get("side"),
                "edge": d.get("edge"), "size_usd": d.get("size_usd"),
                "decision": "TP" if d in tp_scored else "SL",
                "rationale": d.get("rationale", "")[:200],
            })

        # 2. Forecast-driven BUYs (and same-tick flip = SELL + BUY).
        for d in scored:
            if intents_this_tick >= self.constraints.max_trades_per_tick:
                logger.info("Action: hit MAX_TRADES_PER_TICK=%d", self.constraints.max_trades_per_tick)
                break
            if daily_remaining <= 0:
                logger.info("Action: daily fill cap reached (%d/100) — stopping BUY phase", len(_DAILY_FILL_TS))
                break

            mid = d["market_id"]
            if mid in exit_handled:
                continue

            is_new_position = mid not in existing_market_ids

            # Open-positions cap.
            if is_new_position and n_open_positions + new_positions_opened >= self.constraints.max_open_positions:
                self._log_skip(d, "MAX_OPEN_POSITIONS", tick_ts)
                continue

            # Handle position flip: SELL held side first, then BUY new side
            # IN THE SAME TICK. Both go out as separate intents in one request.
            if d.get("is_position_flip"):
                # Emit the SELL leg.
                sell_intent = self._to_intent(d, tick_ctx, equity_now=equity)
                if sell_intent is None:
                    continue
                if intents_this_tick + 2 > self.constraints.max_trades_per_tick:
                    # Not enough room for both legs.
                    logger.info("Action: not enough tick budget to flip %s", mid)
                    continue
                if daily_remaining < 2:
                    logger.info("Action: not enough daily budget to flip %s", mid)
                    continue
                intents.append(sell_intent)
                intents_this_tick += 1
                daily_remaining -= 1
                decisions[mid] = self._decision_summary(d, equity_now=equity)
                self._write_action_log({
                    "tick_ts": str(tick_ts), "market_id": mid,
                    "action": "SELL", "side": d.get("side"),
                    "decision": "FLIP_LEG_1_SELL",
                    "rationale": d.get("rationale", "")[:200],
                })
                # Now compute the BUY leg on the new side.
                buy_leg = d.get("flip_buy_leg")
                if buy_leg is not None:
                    buy_intent = self._to_intent(buy_leg, tick_ctx, equity_now=equity)
                    if buy_intent is not None:
                        intents.append(buy_intent)
                        intents_this_tick += 1
                        daily_remaining -= 1
                        new_positions_opened += 1
                        event_exposure[_event_prefix(mid)] += float(buy_leg.get("size_usd", 0.0))
                        if buy_leg.get("days_to_resolution") and buy_leg["days_to_resolution"] > self.risk.long_dated_threshold_days:
                            long_dated_exposure += float(buy_leg.get("size_usd", 0.0))
                        self._write_action_log({
                            "tick_ts": str(tick_ts), "market_id": mid,
                            "action": "BUY", "side": buy_leg.get("side"),
                            "size_usd": buy_leg.get("size_usd"),
                            "decision": "FLIP_LEG_2_BUY",
                            "rationale": buy_leg.get("rationale", "")[:200],
                        })
                continue

            size_usd = float(d["size_usd"])

            # Gross exposure cap.
            if running_notional + size_usd > self.constraints.max_gross_exposure:
                trimmed = self.constraints.max_gross_exposure - running_notional
                if trimmed < self.min_size_usd:
                    self._log_skip(d, "MAX_GROSS_EXPOSURE", tick_ts)
                    continue
                size_usd = trimmed
                d["size_usd"] = size_usd
                d["shares"] = self._size_to_shares(size_usd, d["price"])
                if size_usd < self.min_size_usd:
                    continue

            # Same-event concentration cap.
            ev_prefix = _event_prefix(mid)
            existing_event_notional = event_exposure[ev_prefix]
            if existing_event_notional + size_usd > self.risk.same_event_max_notional:
                room = self.risk.same_event_max_notional - existing_event_notional
                if room < self.min_size_usd:
                    self._log_skip(d, f"SAME_EVENT_CAP[{ev_prefix}]", tick_ts)
                    continue
                size_usd = room
                d["size_usd"] = size_usd
                d["shares"] = self._size_to_shares(size_usd, d["price"])

            # Long-dated capital share cap.
            days = d.get("days_to_resolution")
            if days is not None and days > self.risk.long_dated_threshold_days:
                max_long_dated = equity * self.risk.long_dated_max_share
                if long_dated_exposure + size_usd > max_long_dated:
                    room = max_long_dated - long_dated_exposure
                    if room < self.min_size_usd:
                        self._log_skip(d, "LONG_DATED_CAP", tick_ts)
                        continue
                    size_usd = room
                    d["size_usd"] = size_usd
                    d["shares"] = self._size_to_shares(size_usd, d["price"])

            # Category cap.
            cat = _category(mid, d.get("question", ""))
            if category_counts.get(cat, 0) >= self.constraints.max_intents_per_category:
                self._log_skip(d, f"CATEGORY_CAP[{cat}]", tick_ts)
                continue

            intent = self._to_intent(d, tick_ctx, equity_now=equity)
            if intent is None:
                continue

            intents.append(intent)
            intents_this_tick += 1
            daily_remaining -= 1
            decisions[mid] = self._decision_summary(d, equity_now=equity)
            category_counts[cat] = category_counts.get(cat, 0) + 1
            running_notional += size_usd
            event_exposure[ev_prefix] += size_usd
            if days is not None and days > self.risk.long_dated_threshold_days:
                long_dated_exposure += size_usd
            if is_new_position:
                new_positions_opened += 1

            self._write_action_log({
                "tick_ts":            str(tick_ts),
                "market_id":          mid,
                "category":           cat,
                "action":             "BUY",
                "side":               d.get("side"),
                "edge":               round(float(d.get("edge", 0.0)), 4),
                "size_usd":           round(size_usd, 2),
                "p_yes":              round(float(d.get("p_yes", 0.0)), 4),
                "p_market":           round(float(d.get("price", 0.0)), 4),
                "days_to_resolution": days,
                "decision":           "BUY_OK",
                "rationale":          d.get("rationale", "")[:200],
            })
            # Record the fill timestamp for daily-cap tracking.
            _DAILY_FILL_TS.append(now_ts)

        logger.info(
            "Action stage complete: %d intents (%d tp, %d sl, %d forecasts) "
            "(min_edge=%.3f, gross=$%.0f, long_dated=$%.0f, daily_used=%d/100)",
            len(intents), len(tp_scored), len(sl_scored), len(forecasts),
            min_edge, running_notional, long_dated_exposure, len(_DAILY_FILL_TS),
        )

        return StageResult(
            stage_name=self.name,
            success=True,
            data={"intents": intents, "decisions": decisions},
        )

    # ------------------------------------------------------------------
    # Daily cap helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tick_ts_to_epoch(tick_ts: Any) -> float:
        if tick_ts is None:
            import time as _t
            return _t.time()
        try:
            import datetime as _dt
            if hasattr(tick_ts, "timestamp"):
                return float(tick_ts.timestamp())
            return float(_dt.datetime.fromisoformat(str(tick_ts)).timestamp())
        except Exception:
            import time as _t
            return _t.time()

    def _prune_daily_fills(self, now_ts: float) -> None:
        """Drop fills older than 24h from the rolling counter."""
        cutoff = now_ts - 24 * 3600
        while _DAILY_FILL_TS and _DAILY_FILL_TS[0] < cutoff:
            _DAILY_FILL_TS.popleft()

    def _log_skip(self, d: dict, reason: str, tick_ts: Any) -> None:
        self._write_action_log({
            "tick_ts": str(tick_ts), "market_id": d.get("market_id"),
            "decision": f"SKIP_{reason}",
            "edge": d.get("edge"), "size_usd": d.get("size_usd"),
        })
        logger.info("Action: skip %s [%s]", d.get("market_id"), reason)

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
        sigma: float = 0.10,
        conf_model: float = 0.0,
        days_to_resolution: float | None = None,
        current_port_var: float = 0.0,
    ) -> dict[str, Any] | None:
        """Compute (side, edge, sizing) for one market; return None to skip."""
        p_yes = _clamp01(p_yes)
        p_no = 1.0 - p_yes

        yes_ask = float(market.yes_ask)
        no_ask = float(market.no_ask)

        edge_yes = p_yes - yes_ask if yes_ask > 0 else -1.0
        edge_no = p_no - no_ask if no_ask > 0 else -1.0

        if edge_yes >= edge_no:
            best_side = "YES"
            best_edge = edge_yes
            best_price = yes_ask
        else:
            best_side = "NO"
            best_edge = edge_no
            best_price = no_ask

        # Gate.
        if raw_gap is not None:
            if abs(raw_gap) < self.risk.min_raw_gap:
                return None
        else:
            if best_edge < min_edge:
                return None

        existing = tick_ctx.get_position(market_id)

        # ------------------------------------------------------------------
        # Position flip — SELL held side AND BUY new side in same tick.
        # ------------------------------------------------------------------
        if existing is not None and existing.side != best_side:
            shares_held = float(existing.shares)
            if shares_held <= 0:
                pass
            else:
                held_side = existing.side
                sell_price = (
                    float(market.yes_bid) if held_side == "YES" else float(market.no_bid)
                )
                if sell_price <= 0:
                    return None
                sell_notional = shares_held * sell_price

                # Compute the BUY leg as if no position existed.
                buy_leg = self._size_buy(
                    market_id=market_id,
                    p_yes=p_yes, p_no=p_no,
                    best_side=best_side, best_edge=best_edge, best_price=best_price,
                    market=market,
                    tick_ctx=tick_ctx,
                    rationale=f"Flip BUY leg after selling {held_side}. {rationale}",
                    llm_var=llm_var, sigma=sigma, conf_model=conf_model,
                    days_to_resolution=days_to_resolution,
                    current_port_var=current_port_var,
                )

                return {
                    "market_id":         market_id,
                    "action":            "SELL",
                    "side":              held_side,
                    "price":             sell_price,
                    "edge":              best_edge,
                    "size_usd":          sell_notional,
                    "shares":            shares_held,
                    "rationale": (
                        f"Flip-SELL: held {held_side} {shares_held:.2f} shares; "
                        f"forecast prefers {best_side} (edge {best_edge:+.3f}). "
                        f"Selling to flat + opening {best_side} in same tick. {rationale}"
                    ).strip(),
                    "p_yes":             p_yes,
                    "p_no":              p_no,
                    "question":          market.question,
                    "is_position_flip":  True,
                    "flip_buy_leg":      buy_leg,
                    "days_to_resolution": days_to_resolution,
                }

        # Standard BUY path.
        return self._size_buy(
            market_id=market_id,
            p_yes=p_yes, p_no=p_no,
            best_side=best_side, best_edge=best_edge, best_price=best_price,
            market=market,
            tick_ctx=tick_ctx,
            rationale=rationale,
            llm_var=llm_var, sigma=sigma, conf_model=conf_model,
            days_to_resolution=days_to_resolution,
            current_port_var=current_port_var,
        )

    def _size_buy(
        self, *,
        market_id: str, p_yes: float, p_no: float,
        best_side: str, best_edge: float, best_price: float,
        market: CandidateMarket, tick_ctx: TickContext, rationale: str,
        llm_var: float, sigma: float, conf_model: float,
        days_to_resolution: float | None,
        current_port_var: float,
    ) -> dict[str, Any] | None:
        """Compute the sizing dict for a BUY decision (or None to skip)."""
        if best_price <= 0 or best_price >= 1.0:
            return None

        from agent.calibration import epistemic_shrink
        shrink = epistemic_shrink(p_yes, llm_var)

        # 1. Base Kelly.
        base_kelly = best_edge / max(1.0 - best_price, 1e-6)

        # 2. Variance dampener with high-edge bypass.
        asset_var = best_price * (1.0 - best_price)
        asset_std_dev = max(asset_var ** 0.5, 1e-6)
        base_dampener = min(1.0, 0.4 / asset_std_dev)
        if best_edge > 0.08:
            recovery_factor = min(1.0, (best_edge - 0.08) * 4.0)
            sharpe_dampener = base_dampener + ((1.0 - base_dampener) * recovery_factor)
        else:
            sharpe_dampener = base_dampener

        # 3. Heat penalty.
        target_max_var = getattr(self.risk, "target_portfolio_var", 0.05)
        heat_penalty = max(0.1, 1.0 - (current_port_var / target_max_var))

        # 4. Duration penalty: 1 / (1 + days / half_life).
        if days_to_resolution is None:
            duration_penalty = 1.0
        else:
            half_life = max(1.0, self.risk.duration_half_life_days)
            duration_penalty = 1.0 / (1.0 + days_to_resolution / half_life)

        # 5. Sigma-aware sizing: high sigma → smaller positions.
        sigma_penalty = max(0.3, 1.0 - 2.0 * max(0.0, sigma - 0.10))

        kelly_frac = (
            self.risk.kelly_fraction
            * shrink
            * base_kelly
            * sharpe_dampener
            * heat_penalty
            * duration_penalty
            * sigma_penalty
        )

        kelly_frac = max(0.0, min(self.risk.max_position_pct_of_equity, kelly_frac))

        equity = float(tick_ctx.equity)
        size_usd = kelly_frac * equity
        size_usd = min(size_usd, self.constraints.max_notional_per_market)

        if size_usd < self.min_size_usd:
            return None

        shares = self._size_to_shares(size_usd, best_price)
        if shares <= 0:
            return None

        return {
            "market_id":          market_id,
            "action":             "BUY",
            "side":               best_side,
            "price":              best_price,
            "edge":               best_edge,
            "size_usd":           size_usd,
            "shares":             shares,
            "rationale": (
                f"edge={best_edge:+.3f} p_yes={p_yes:.3f} price={best_price:.3f}; "
                f"Kelly ${size_usd:.0f} "
                f"(dur_pen={duration_penalty:.2f} sigma_pen={sigma_penalty:.2f} "
                f"heat={heat_penalty:.2f} sharpe={sharpe_dampener:.2f}). {rationale}"
            ).strip(),
            "p_yes":              p_yes,
            "p_no":               p_no,
            "question":           market.question,
            "is_position_flip":   False,
            "days_to_resolution": days_to_resolution,
        }

    # ------------------------------------------------------------------
    # Helpers — min_edge relaxation, take-profit, stop-loss
    # ------------------------------------------------------------------

    def _effective_min_edge(self, tick_ctx: TickContext) -> float:
        """Relax min_edge only in the final ~final_stretch_ticks ticks if still
        below the trade floor. This stops day-1 over-trading.

        Uses target_total_ticks - tick_ctx.tick_index to estimate ticks remaining.
        """
        floor = self.risk.trade_floor_count
        tick_index = getattr(tick_ctx, "tick_index", None)
        if tick_index is None:
            tick_index = getattr(tick_ctx, "tick_num", 0) or 0
        try:
            tick_index = int(tick_index)
        except Exception:
            tick_index = 0
        ticks_remaining = max(0, self.target_total_ticks - tick_index)
        final_stretch = getattr(self.risk, "final_stretch_ticks", 100)

        if ticks_remaining <= final_stretch and tick_ctx.total_fills < floor:
            relaxed = self.risk.min_edge_relaxed
            if relaxed < self.risk.min_edge:
                logger.info(
                    "Action: relaxing min_edge %.3f -> %.3f (fills=%d/%d, ticks_left=%d)",
                    self.risk.min_edge, relaxed,
                    tick_ctx.total_fills, floor, ticks_remaining,
                )
                return relaxed
        return self.risk.min_edge

    def _take_profit_intents(
        self, tick_ctx: TickContext
    ) -> tuple[list[dict[str, Any]], set[str]]:
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
                "market_id":         pos.market_id,
                "action":            "SELL",
                "side":              pos.side,
                "price":             sell_price,
                "edge":              gain_pct,
                "size_usd":          shares * sell_price,
                "shares":            shares,
                "rationale": (
                    f"Take-profit: mark {mark:.3f} vs entry {entry:.3f} "
                    f"(+{gain_pct:.1%} >= {threshold:.0%})."
                ),
                "question":          question,
                "is_position_flip":  False,
            })
            handled.add(pos.market_id)

        if scored:
            logger.info("Action: %d take-profit sell(s) triggered", len(scored))
        return scored, handled

    def _stop_loss_intents(
        self, tick_ctx: TickContext
    ) -> tuple[list[dict[str, Any]], set[str]]:
        scored: list[dict[str, Any]] = []
        handled: set[str] = set()
        threshold = self.risk.stop_loss_threshold

        for pos in tick_ctx.positions:
            entry = float(pos.avg_entry_price)
            mark = float(pos.current_price)
            if entry <= 0:
                continue
            loss_pct = (mark - entry) / entry
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
                "market_id":         pos.market_id,
                "action":            "SELL",
                "side":              pos.side,
                "price":             sell_price,
                "edge":              abs(loss_pct),
                "size_usd":          shares * sell_price,
                "shares":            shares,
                "rationale": (
                    f"Stop-loss: mark {mark:.3f} vs entry {entry:.3f} "
                    f"({loss_pct:.1%} <= -{threshold:.0%})."
                ),
                "question":          question,
                "is_position_flip":  False,
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
