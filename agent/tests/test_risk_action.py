"""Unit tests for agent/stages/risk_action.py.

Phase 3 gate: covers all four caps + the position-flip-as-sell rule +
the minimum-edge gate + the 14-trade-minimum relaxation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from ai_prophet.trade.agent.stages.base import StageResult
from ai_prophet.trade.core.tick_context import CandidateMarket, Position, TickContext

from agent.settings import RiskConfig, TradingConstraints
from agent.stages.risk_action import RiskAwareActionStage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _tick_boundary(minutes: int = 15) -> datetime:
    """Snap to a 15-min boundary (the SDK enforces this in TickContext.__post_init__)."""
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    snapped = now.replace(minute=(now.minute // minutes) * minutes)
    return snapped


def make_market(
    market_id: str,
    *,
    question: str = "Q",
    yes_bid: float = 0.48,
    yes_ask: float = 0.52,
    volume_24h: float = 5000.0,
) -> CandidateMarket:
    yes_mark = (yes_bid + yes_ask) / 2.0
    return CandidateMarket(
        market_id=market_id,
        question=question,
        description=None,
        resolution_time=datetime.now(timezone.utc) + timedelta(days=10),
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        yes_mark=yes_mark,
        no_bid=1.0 - yes_ask,
        no_ask=1.0 - yes_bid,
        no_mark=1.0 - yes_mark,
        volume_24h=volume_24h,
        quote_ts=datetime.now(timezone.utc),
    )


def make_position(market_id: str, side: str, shares: float, avg_entry: float) -> Position:
    return Position(
        market_id=market_id,
        side=side,
        shares=Decimal(str(shares)),
        avg_entry_price=Decimal(str(avg_entry)),
        current_price=Decimal(str(avg_entry)),
        unrealized_pnl=Decimal("0"),
        realized_pnl=Decimal("0"),
        updated_at=datetime.now(timezone.utc),
    )


def make_tick(
    *,
    candidates: list[CandidateMarket],
    positions: list[Position] | None = None,
    cash: float = 10000.0,
    equity: float = 10000.0,
    total_fills: int = 50,
) -> TickContext:
    tick_ts = _tick_boundary()
    return TickContext(
        run_id="test:0",
        tick_ts=tick_ts,
        data_asof_ts=tick_ts,
        candidate_set_id="cs_test",
        submission_deadline=tick_ts + timedelta(minutes=9),
        server_now=tick_ts,
        candidates=tuple(candidates),
        cash=Decimal(str(cash)),
        equity=Decimal(str(equity)),
        total_pnl=Decimal("0"),
        positions=tuple(positions or []),
        total_fills=total_fills,
    )


def forecast_result(forecasts: dict[str, dict]) -> dict[str, StageResult]:
    return {
        "forecast": StageResult(
            stage_name="forecast",
            success=True,
            data={"forecasts": forecasts},
        ),
    }


def make_stage(**risk_overrides) -> RiskAwareActionStage:
    risk = RiskConfig(**risk_overrides) if risk_overrides else RiskConfig()
    return RiskAwareActionStage(
        llm_client=None,
        constraints=TradingConstraints(),
        risk=risk,
    )


# ---------------------------------------------------------------------------
# Edge gate
# ---------------------------------------------------------------------------

def test_no_intent_when_edge_below_min():
    # market: ask 0.52, p_yes 0.54 -> edge 0.02 < default 0.05.
    m = make_market("M1", yes_bid=0.48, yes_ask=0.52)
    tick = make_tick(candidates=[m])
    forecasts = {"M1": {"p_yes": 0.54, "rationale": "tiny edge"}}

    stage = make_stage()
    out = stage.execute(tick, forecast_result(forecasts))

    assert out.success
    assert out.data["intents"] == []


def test_buy_yes_when_edge_clears_threshold():
    # ask 0.40, p_yes 0.60 -> edge 0.20, well over 0.05.
    m = make_market("M1", yes_bid=0.36, yes_ask=0.40)
    tick = make_tick(candidates=[m])
    forecasts = {"M1": {"p_yes": 0.60, "rationale": "real edge"}}

    stage = make_stage()
    out = stage.execute(tick, forecast_result(forecasts))

    intents = out.data["intents"]
    assert len(intents) == 1
    assert intents[0]["action"] == "BUY"
    assert intents[0]["side"] == "YES"
    assert float(intents[0]["shares"]) > 0


def test_buy_no_when_negative_yes_edge_but_positive_no_edge():
    # ask 0.80 (high), p_yes 0.30 -> NO ask = 1 - 0.20 = 0.20.
    # p_no = 0.70, edge_no = 0.70 - 0.20 = 0.50.
    m = make_market("M1", yes_bid=0.78, yes_ask=0.80)
    tick = make_tick(candidates=[m])
    forecasts = {"M1": {"p_yes": 0.30, "rationale": "no side"}}

    stage = make_stage()
    out = stage.execute(tick, forecast_result(forecasts))

    intents = out.data["intents"]
    assert len(intents) == 1
    assert intents[0]["side"] == "NO"


# ---------------------------------------------------------------------------
# Sizing caps
# ---------------------------------------------------------------------------

def test_size_capped_by_max_position_pct_of_equity():
    # Massive edge would otherwise size higher; we cap at 5% of $10k = $500.
    m = make_market("M1", yes_bid=0.05, yes_ask=0.10)
    tick = make_tick(candidates=[m], equity=10_000.0)
    forecasts = {"M1": {"p_yes": 0.95, "rationale": "huge edge"}}

    stage = make_stage()
    out = stage.execute(tick, forecast_result(forecasts))
    decisions = out.data["decisions"]
    assert decisions["M1"]["size_usd"] <= 500.0 + 1e-6


def test_size_capped_by_max_notional_per_market_with_large_equity():
    # With huge equity, the 5% cap = $50k, so the $1k per-market cap binds.
    m = make_market("M1", yes_bid=0.05, yes_ask=0.10)
    tick = make_tick(candidates=[m], equity=1_000_000.0)
    forecasts = {"M1": {"p_yes": 0.95, "rationale": "huge edge"}}

    stage = make_stage()
    out = stage.execute(tick, forecast_result(forecasts))
    decisions = out.data["decisions"]
    assert decisions["M1"]["size_usd"] <= 1000.0 + 1e-6


def test_gross_exposure_cap_skips_later_intents():
    # 11 markets each priced cheap with full $1k size would total $11k.
    # MAX_GROSS_EXPOSURE = $10k -> last one should be trimmed or skipped.
    markets = [
        make_market(f"M{i}", yes_bid=0.05, yes_ask=0.10) for i in range(12)
    ]
    tick = make_tick(candidates=markets, equity=1_000_000.0)
    forecasts = {f"M{i}": {"p_yes": 0.95, "rationale": "edge"} for i in range(12)}

    stage = make_stage()
    out = stage.execute(tick, forecast_result(forecasts))
    total = sum(d["size_usd"] for d in out.data["decisions"].values())
    assert total <= 10_000.0 + 1e-6


# ---------------------------------------------------------------------------
# Position caps
# ---------------------------------------------------------------------------

def test_skips_new_position_when_at_max_open_positions():
    # 30 existing positions; new opportunity in M31 must be skipped.
    existing = [make_position(f"H{i}", "YES", 10.0, 0.5) for i in range(30)]
    new_market = make_market("M_NEW", yes_bid=0.36, yes_ask=0.40)
    tick = make_tick(
        candidates=[new_market],
        positions=existing,
        equity=1_000_000.0,
    )
    forecasts = {"M_NEW": {"p_yes": 0.60, "rationale": ""}}

    stage = make_stage()
    out = stage.execute(tick, forecast_result(forecasts))
    assert out.data["intents"] == []


def test_max_trades_per_tick_truncates():
    # 25 fresh markets with strong edge but tiny equity so per-position cap
    # keeps each trade small enough that MAX_GROSS_EXPOSURE doesn't bind
    # first: equity=$5k -> 5% cap = $250/trade, 25*$250=$6.25k < $10k cap,
    # so MAX_TRADES_PER_TICK=20 should be the binding limit.
    # Use max_intents_per_category=25 so the category cap doesn't bind first.
    markets = [
        make_market(f"M{i}", yes_bid=0.36, yes_ask=0.40) for i in range(25)
    ]
    tick = make_tick(candidates=markets, equity=5_000.0)
    forecasts = {f"M{i}": {"p_yes": 0.60, "rationale": ""} for i in range(25)}

    constraints = TradingConstraints(max_intents_per_category=25)
    stage = _make_stage_with_constraints(constraints)
    out = stage.execute(tick, forecast_result(forecasts))
    assert len(out.data["intents"]) == 20  # MAX_TRADES_PER_TICK


# ---------------------------------------------------------------------------
# Position-flip-as-sell rule
# ---------------------------------------------------------------------------

def test_position_flip_emits_sell_on_held_side():
    # Hold YES, but forecast now prefers NO.
    # Nailong Elite: same-tick SELL+BUY flip — emit SELL YES followed by
    # BUY NO (both intents in one submission, per hackathon guide).
    m = make_market("M1", yes_bid=0.78, yes_ask=0.80)
    held = make_position("M1", "YES", shares=50.0, avg_entry=0.50)
    tick = make_tick(candidates=[m], positions=[held])
    forecasts = {"M1": {"p_yes": 0.20, "rationale": "flip"}}

    stage = make_stage()
    out = stage.execute(tick, forecast_result(forecasts))

    intents = out.data["intents"]
    # The flip may emit 1 (SELL only, no BUY leg sized) or 2 (SELL + BUY)
    # depending on whether the BUY leg passes the min-size gate.
    assert len(intents) >= 1
    # First intent is always the SELL leg on the held side.
    assert intents[0]["action"] == "SELL"
    assert intents[0]["side"] == "YES"
    assert float(intents[0]["shares"]) == 50.0
    decision = out.data["decisions"]["M1"]
    assert decision["is_position_flip"] is True
    # If a BUY leg was emitted, it must be on the new side.
    if len(intents) == 2:
        assert intents[1]["action"] == "BUY"
        assert intents[1]["side"] == "NO"


def test_no_flip_when_already_aligned():
    # Hold YES, forecast still prefers YES -> standard BUY YES (sizing path).
    m = make_market("M1", yes_bid=0.36, yes_ask=0.40)
    held = make_position("M1", "YES", shares=20.0, avg_entry=0.30)
    tick = make_tick(candidates=[m], positions=[held])
    forecasts = {"M1": {"p_yes": 0.60, "rationale": "still good"}}

    stage = make_stage()
    out = stage.execute(tick, forecast_result(forecasts))
    intents = out.data["intents"]
    assert len(intents) == 1
    assert intents[0]["action"] == "BUY"
    assert intents[0]["side"] == "YES"


# ---------------------------------------------------------------------------
# 14-trade-minimum guard
# ---------------------------------------------------------------------------

def test_min_edge_relaxed_when_under_trade_floor():
    # Nailong Elite: relaxation only fires when ticks_remaining <= 100 AND
    # total_fills < floor. We use target_total_ticks=0 so any tick is
    # treated as "in the final stretch".
    m = make_market("M1", yes_bid=0.36, yes_ask=0.40)
    tick = make_tick(candidates=[m], total_fills=0)
    forecasts = {"M1": {"p_yes": 0.44, "rationale": ""}}

    stage = RiskAwareActionStage(
        llm_client=None,
        constraints=TradingConstraints(),
        risk=RiskConfig(),
        target_total_ticks=0,  # forces "final stretch" regardless of tick_index
    )
    out = stage.execute(tick, forecast_result(forecasts))
    assert len(out.data["intents"]) == 1


def test_min_edge_not_relaxed_outside_final_stretch():
    # New: even if under the floor, min_edge stays strict when there are
    # plenty of ticks remaining.
    m = make_market("M1", yes_bid=0.36, yes_ask=0.40)
    tick = make_tick(candidates=[m], total_fills=0)
    forecasts = {"M1": {"p_yes": 0.44, "rationale": ""}}

    stage = RiskAwareActionStage(
        llm_client=None,
        constraints=TradingConstraints(),
        risk=RiskConfig(),
        target_total_ticks=10000,  # huge horizon → no relaxation
    )
    out = stage.execute(tick, forecast_result(forecasts))
    assert out.data["intents"] == []


def test_min_edge_not_relaxed_when_at_or_above_floor():
    # Same edge 0.04 fails the default min_edge 0.05 when total_fills >= floor.
    m = make_market("M1", yes_bid=0.36, yes_ask=0.40)
    tick = make_tick(candidates=[m], total_fills=14)
    forecasts = {"M1": {"p_yes": 0.44, "rationale": ""}}

    stage = make_stage()
    out = stage.execute(tick, forecast_result(forecasts))
    assert out.data["intents"] == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_below_min_intent_size_skipped():
    # Tiny equity + tight edge -> Kelly fraction yields a sub-$5 trade -> skip.
    m = make_market("M1", yes_bid=0.48, yes_ask=0.50)
    tick = make_tick(candidates=[m], equity=10.0)
    forecasts = {"M1": {"p_yes": 0.60, "rationale": ""}}

    stage = make_stage(min_intent_size_usd=5.0)
    out = stage.execute(tick, forecast_result(forecasts))
    assert out.data["intents"] == []


def test_no_forecasts_returns_empty_success():
    tick = make_tick(candidates=[])
    stage = make_stage()
    out = stage.execute(tick, forecast_result({}))
    assert out.success
    assert out.data["intents"] == []


# ---------------------------------------------------------------------------
# Take-profit (mechanical SELL when mark >= entry * 1.25)
# ---------------------------------------------------------------------------

def test_take_profit_fires_when_gain_exceeds_threshold():
    m = make_market("M1", yes_bid=0.75, yes_ask=0.78)
    # Bought YES at 0.50, now marks 0.75 -> +50% gain, well over 25%.
    from dataclasses import replace as dc_replace
    held = dc_replace(
        make_position("M1", "YES", shares=100.0, avg_entry=0.50),
        current_price=Decimal("0.75"),
    )
    tick = make_tick(candidates=[m], positions=[held])

    stage = make_stage()
    out = stage.execute(tick, forecast_result({}))

    assert len(out.data["intents"]) == 1
    intent = out.data["intents"][0]
    assert intent["action"] == "SELL"
    assert intent["side"] == "YES"
    assert float(intent["shares"]) == 100.0


def test_take_profit_does_not_fire_below_threshold():
    m = make_market("M1", yes_bid=0.60, yes_ask=0.63)
    # Bought YES at 0.50, now marks 0.60 -> +20% gain, under the 25% threshold.
    held = make_position("M1", "YES", shares=50.0, avg_entry=0.50)
    from dataclasses import replace as dc_replace
    held = dc_replace(held, current_price=Decimal("0.60"))
    tick = make_tick(candidates=[m], positions=[held])

    stage = make_stage()
    out = stage.execute(tick, forecast_result({}))

    assert out.data["intents"] == []


def test_take_profit_fires_on_no_side():
    m = make_market("M1", yes_bid=0.30, yes_ask=0.33)
    # Bought NO at 0.55 (1 - yes_ask=0.45), now NO marks 0.70 -> +27% gain.
    held = make_position("M1", "NO", shares=80.0, avg_entry=0.55)
    from dataclasses import replace as dc_replace
    held = dc_replace(held, current_price=Decimal("0.70"))
    tick = make_tick(candidates=[m], positions=[held])

    stage = make_stage()
    out = stage.execute(tick, forecast_result({}))

    assert len(out.data["intents"]) == 1
    assert out.data["intents"][0]["side"] == "NO"


def test_take_profit_precedes_forecast_buys():
    # Take-profit SELL on M1, plus a new BUY forecast on M2 -> SELL listed first.
    tp_market = make_market("M1", yes_bid=0.75, yes_ask=0.78)
    new_market = make_market("M2", yes_bid=0.36, yes_ask=0.40)
    from dataclasses import replace as dc_replace
    held = make_position("M1", "YES", shares=100.0, avg_entry=0.50)
    held = dc_replace(held, current_price=Decimal("0.75"))
    tick = make_tick(candidates=[tp_market, new_market], positions=[held])

    stage = make_stage()
    out = stage.execute(tick, forecast_result({"M2": {"p_yes": 0.60, "rationale": ""}}))

    actions = [(i["market_id"], i["action"]) for i in out.data["intents"]]
    assert ("M1", "SELL") in actions
    assert ("M2", "BUY") in actions
    assert actions.index(("M1", "SELL")) < actions.index(("M2", "BUY"))


def test_take_profit_skips_market_from_forecast_loop():
    # If take-profit fires on M1, the forecast loop should not also BUY M1.
    m = make_market("M1", yes_bid=0.75, yes_ask=0.78)
    from dataclasses import replace as dc_replace
    held = make_position("M1", "YES", shares=100.0, avg_entry=0.50)
    held = dc_replace(held, current_price=Decimal("0.75"))
    tick = make_tick(candidates=[m], positions=[held])

    # Forecast says M1 is still a good BUY; take-profit should override.
    stage = make_stage()
    out = stage.execute(tick, forecast_result({"M1": {"p_yes": 0.90, "rationale": ""}}))

    assert len(out.data["intents"]) == 1
    assert out.data["intents"][0]["action"] == "SELL"


def test_take_profit_threshold_configurable():
    m = make_market("M1", yes_bid=0.63, yes_ask=0.66)
    from dataclasses import replace as dc_replace
    held = make_position("M1", "YES", shares=50.0, avg_entry=0.50)
    held = dc_replace(held, current_price=Decimal("0.63"))
    tick = make_tick(candidates=[m], positions=[held])

    # Default 25% threshold: +26% gain should fire.
    stage_default = make_stage()
    out_default = stage_default.execute(tick, forecast_result({}))
    assert len(out_default.data["intents"]) == 1

    # Custom 40% threshold: +26% gain should NOT fire.
    from agent.settings import RiskConfig
    stage_high = RiskAwareActionStage(
        llm_client=None,
        constraints=TradingConstraints(),
        risk=RiskConfig(take_profit_threshold=0.40),
    )
    out_high = stage_high.execute(tick, forecast_result({}))
    assert out_high.data["intents"] == []


# ---------------------------------------------------------------------------
# Stop-loss (mechanical SELL when mark <= entry * (1 - threshold))
# ---------------------------------------------------------------------------

def test_stop_loss_fires_when_loss_exceeds_threshold():
    m = make_market("M1", yes_bid=0.35, yes_ask=0.38)
    # Bought YES at 0.50, now marks 0.35 -> -30% loss, over 20% threshold.
    from dataclasses import replace as dc_replace
    held = dc_replace(
        make_position("M1", "YES", shares=100.0, avg_entry=0.50),
        current_price=Decimal("0.35"),
    )
    tick = make_tick(candidates=[m], positions=[held])

    stage = make_stage()
    out = stage.execute(tick, forecast_result({}))

    assert len(out.data["intents"]) == 1
    intent = out.data["intents"][0]
    assert intent["action"] == "SELL"
    assert intent["side"] == "YES"
    assert "Stop-loss" in intent["rationale"]


def test_stop_loss_does_not_fire_below_threshold():
    m = make_market("M1", yes_bid=0.47, yes_ask=0.49)
    # Bought YES at 0.50, now marks 0.47 -> -6% loss, under 10% threshold.
    from dataclasses import replace as dc_replace
    held = dc_replace(
        make_position("M1", "YES", shares=50.0, avg_entry=0.50),
        current_price=Decimal("0.47"),
    )
    tick = make_tick(candidates=[m], positions=[held])

    stage = make_stage()
    out = stage.execute(tick, forecast_result({}))

    assert out.data["intents"] == []


def test_stop_loss_fires_on_no_side():
    m = make_market("M1", yes_bid=0.68, yes_ask=0.72)
    # Bought NO at 0.55 (no_ask = 1 - yes_bid = 0.32? let's use a direct value).
    # entry=0.55, mark=0.40 -> -27% loss.
    from dataclasses import replace as dc_replace
    held = dc_replace(
        make_position("M1", "NO", shares=80.0, avg_entry=0.55),
        current_price=Decimal("0.40"),
    )
    tick = make_tick(candidates=[m], positions=[held])

    stage = make_stage()
    out = stage.execute(tick, forecast_result({}))

    assert len(out.data["intents"]) == 1
    assert out.data["intents"][0]["side"] == "NO"


def test_stop_loss_skips_market_from_forecast_loop():
    m = make_market("M1", yes_bid=0.35, yes_ask=0.38)
    from dataclasses import replace as dc_replace
    held = dc_replace(
        make_position("M1", "YES", shares=100.0, avg_entry=0.50),
        current_price=Decimal("0.35"),
    )
    tick = make_tick(candidates=[m], positions=[held])

    # Even if the forecast still says BUY, stop-loss should prevent re-buy this tick.
    stage = make_stage()
    out = stage.execute(tick, forecast_result({"M1": {"p_yes": 0.60, "rationale": ""}}))

    assert len(out.data["intents"]) == 1
    assert out.data["intents"][0]["action"] == "SELL"


def test_stop_loss_threshold_configurable():
    m = make_market("M1", yes_bid=0.47, yes_ask=0.49)
    from dataclasses import replace as dc_replace
    held = dc_replace(
        make_position("M1", "YES", shares=50.0, avg_entry=0.50),
        current_price=Decimal("0.47"),
    )
    tick = make_tick(candidates=[m], positions=[held])

    # Default 10% threshold (Elite): -6% loss should NOT fire.
    stage_default = make_stage()
    assert stage_default.execute(tick, forecast_result({})).data["intents"] == []

    # Custom 5% threshold: -6% loss SHOULD fire.
    from agent.settings import RiskConfig
    stage_tight = RiskAwareActionStage(
        llm_client=None,
        constraints=TradingConstraints(),
        risk=RiskConfig(stop_loss_threshold=0.05),
    )
    out = stage_tight.execute(tick, forecast_result({}))
    assert len(out.data["intents"]) == 1
    assert out.data["intents"][0]["action"] == "SELL"


def test_stop_loss_and_take_profit_independent():
    # Two positions: one hits take-profit, one hits stop-loss. Both should fire.
    tp_market = make_market("TP", yes_bid=0.75, yes_ask=0.78)
    sl_market = make_market("SL", yes_bid=0.35, yes_ask=0.38)
    from dataclasses import replace as dc_replace
    tp_pos = dc_replace(
        make_position("TP", "YES", shares=100.0, avg_entry=0.50),
        current_price=Decimal("0.75"),
    )
    sl_pos = dc_replace(
        make_position("SL", "YES", shares=80.0, avg_entry=0.50),
        current_price=Decimal("0.35"),
    )
    tick = make_tick(candidates=[tp_market, sl_market], positions=[tp_pos, sl_pos])

    stage = make_stage()
    out = stage.execute(tick, forecast_result({}))

    assert len(out.data["intents"]) == 2
    market_ids = {i["market_id"] for i in out.data["intents"]}
    assert market_ids == {"TP", "SL"}


# ---------------------------------------------------------------------------
# Category concentration cap
# ---------------------------------------------------------------------------

def _make_stage_with_constraints(constraints: TradingConstraints, **risk_overrides) -> RiskAwareActionStage:
    risk = RiskConfig(**risk_overrides) if risk_overrides else RiskConfig()
    return RiskAwareActionStage(llm_client=None, constraints=constraints, risk=risk)


def test_category_cap_limits_intents_per_category():
    # 3 Sports markets all with strong edge -> cap at 2 intents for Sports.
    markets = [
        make_market(f"M{i}", yes_bid=0.36, yes_ask=0.40, question=f"Will the NBA team {i} win?")
        for i in range(3)
    ]
    tick = make_tick(candidates=markets, equity=100_000.0)
    forecasts = {f"M{i}": {"p_yes": 0.60, "rationale": ""} for i in range(3)}

    constraints = TradingConstraints(max_intents_per_category=2)
    stage = _make_stage_with_constraints(constraints)
    out = stage.execute(tick, forecast_result(forecasts))

    assert len(out.data["intents"]) == 2


def test_category_cap_does_not_block_other_categories():
    # 2 Sports + 2 Politics + 2 Economics: all 6 should get through.
    markets = [
        make_market("S1", yes_bid=0.36, yes_ask=0.40, question="Will the NBA finals go to game 7?"),
        make_market("S2", yes_bid=0.36, yes_ask=0.40, question="Will the NFL draft pick surprise?"),
        make_market("P1", yes_bid=0.36, yes_ask=0.40, question="Who will win the presidential election?"),
        make_market("P2", yes_bid=0.36, yes_ask=0.40, question="Will the senate vote pass?"),
        make_market("E1", yes_bid=0.36, yes_ask=0.40, question="Will the Fed raise interest rates?"),
        make_market("E2", yes_bid=0.36, yes_ask=0.40, question="Will Bitcoin hit $100k?"),
    ]
    tick = make_tick(candidates=markets, equity=100_000.0)
    forecasts = {m.market_id: {"p_yes": 0.60, "rationale": ""} for m in markets}

    constraints = TradingConstraints(max_intents_per_category=2)
    stage = _make_stage_with_constraints(constraints)
    out = stage.execute(tick, forecast_result(forecasts))

    assert len(out.data["intents"]) == 6


def test_category_cap_configurable_to_one():
    # max_intents_per_category=1 -> only 1 intent per category.
    markets = [
        make_market("S1", yes_bid=0.36, yes_ask=0.40, question="Will the NBA team win?"),
        make_market("S2", yes_bid=0.36, yes_ask=0.40, question="Will the NFL game go into overtime?"),
        make_market("E1", yes_bid=0.36, yes_ask=0.40, question="Will the Fed cut rates?"),
        make_market("E2", yes_bid=0.36, yes_ask=0.40, question="Will Bitcoin rally?"),
    ]
    tick = make_tick(candidates=markets, equity=100_000.0)
    forecasts = {m.market_id: {"p_yes": 0.60, "rationale": ""} for m in markets}

    constraints = TradingConstraints(max_intents_per_category=1)
    stage = _make_stage_with_constraints(constraints)
    out = stage.execute(tick, forecast_result(forecasts))

    assert len(out.data["intents"]) == 2  # 1 Sports + 1 Economics


def test_category_cap_respects_sort_by_edge():
    # 3 Sports markets with different edges -> cap keeps the 2 with highest edge.
    markets = [
        make_market("S_HIGH", yes_bid=0.20, yes_ask=0.25, question="Will the NBA champ repeat?"),
        make_market("S_MED",  yes_bid=0.36, yes_ask=0.40, question="Will the NBA finals go 7 games?"),
        make_market("S_LOW",  yes_bid=0.45, yes_ask=0.50, question="Will the NBA player score 40?"),
    ]
    tick = make_tick(candidates=markets, equity=100_000.0)
    forecasts = {
        "S_HIGH": {"p_yes": 0.70, "rationale": ""},  # edge 0.45
        "S_MED":  {"p_yes": 0.60, "rationale": ""},  # edge 0.20
        "S_LOW":  {"p_yes": 0.58, "rationale": ""},  # edge 0.08
    }

    constraints = TradingConstraints(max_intents_per_category=2)
    stage = _make_stage_with_constraints(constraints)
    out = stage.execute(tick, forecast_result(forecasts))

    intent_ids = {i["market_id"] for i in out.data["intents"]}
    assert len(intent_ids) == 2
    assert "S_LOW" not in intent_ids  # lowest edge dropped by cap
