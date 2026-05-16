"""Unit tests for agent/calibration.py.

Phase 2 gate: covers (a) high-confidence deviation passes through,
(b) low-confidence pulls toward market, (c) >max_dev deviation is clipped,
(d) Polymarket disagreement falls back to Kalshi.
"""

from __future__ import annotations

import pytest

from agent.calibration import (
    blend,
    calibrate,
    cap_deviation,
    market_mid,
    polymarket_consensus,
)
from agent.settings import CalibrationConfig


@pytest.fixture
def cfg() -> CalibrationConfig:
    # Tight defaults so tests don't depend on .env values.
    return CalibrationConfig(
        alpha_high=0.7,
        alpha_medium=0.5,
        alpha_low=0.25,
        max_deviation=0.30,
        polymarket_agreement_band=0.05,
        polymarket_min_volume_usd=10_000.0,
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_market_mid_normal():
    assert market_mid(0.40, 0.42) == pytest.approx(0.41)


def test_market_mid_crossed_quote_repaired():
    # If ask < bid (crossed market), we sort them and still return a sane mid.
    assert market_mid(0.42, 0.40) == pytest.approx(0.41)


def test_market_mid_missing_one_side():
    assert market_mid(None, 0.7) == pytest.approx(0.7)
    assert market_mid(0.3, None) == pytest.approx(0.3)
    assert market_mid(None, None) is None


def test_blend_pure_model():
    assert blend(0.8, 0.5, alpha=1.0) == pytest.approx(0.8)


def test_blend_pure_market():
    assert blend(0.8, 0.5, alpha=0.0) == pytest.approx(0.5)


def test_blend_clips_alpha_out_of_range():
    assert blend(0.8, 0.5, alpha=2.0) == pytest.approx(0.8)
    assert blend(0.8, 0.5, alpha=-1.0) == pytest.approx(0.5)


def test_cap_deviation_within_cap():
    assert cap_deviation(0.65, 0.50, max_dev=0.30) == pytest.approx(0.65)


def test_cap_deviation_clips_high():
    assert cap_deviation(0.95, 0.50, max_dev=0.30) == pytest.approx(0.80)


def test_cap_deviation_clips_low():
    assert cap_deviation(0.10, 0.50, max_dev=0.30) == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Polymarket consensus
# ---------------------------------------------------------------------------

def test_polymarket_consensus_agreement(cfg):
    history = {"yes_price": 0.62, "no_price": 0.38, "volume": 50_000.0}
    # Kalshi at 0.60, Polymarket at 0.62, within 0.05 band -> averaged
    result = polymarket_consensus(history, p_kalshi=0.60, config=cfg)
    assert result == pytest.approx(0.61)


def test_polymarket_consensus_disagreement_returns_none(cfg):
    history = {"yes_price": 0.20, "no_price": 0.80, "volume": 50_000.0}
    # Kalshi at 0.60, Polymarket at 0.20 -> disagree by 0.40, ignored
    assert polymarket_consensus(history, p_kalshi=0.60, config=cfg) is None


def test_polymarket_consensus_low_volume_returns_none(cfg):
    history = {"yes_price": 0.62, "volume": 100.0}
    assert polymarket_consensus(history, p_kalshi=0.60, config=cfg) is None


def test_polymarket_consensus_missing_returns_none(cfg):
    assert polymarket_consensus(None, p_kalshi=0.60, config=cfg) is None
    assert polymarket_consensus({}, p_kalshi=0.60, config=cfg) is None


# ---------------------------------------------------------------------------
# End-to-end calibrate(): the four plan-mandated gate scenarios
# ---------------------------------------------------------------------------

def test_calibrate_high_confidence_deviation_passes_through(cfg):
    # High confidence -> alpha 0.7 -> we mostly trust the model.
    # Model says 0.70, market says 0.50, no Polymarket signal.
    # Expected: 0.7*0.70 + 0.3*0.50 = 0.64, well inside the 0.30 cap.
    out = calibrate(
        p_model=0.70, p_market=0.50,
        confidence="high", market_history=None, config=cfg,
    )
    assert out.p_final == pytest.approx(0.64)
    assert not out.clipped_to_cap
    assert not out.used_polymarket


def test_calibrate_low_confidence_pulls_toward_market(cfg):
    # Low confidence -> alpha 0.25 -> mostly anchor to market.
    # Model says 0.70, market says 0.50 -> 0.25*0.70 + 0.75*0.50 = 0.55.
    out = calibrate(
        p_model=0.70, p_market=0.50,
        confidence="low", market_history=None, config=cfg,
    )
    assert out.p_final == pytest.approx(0.55)
    assert abs(out.p_final - out.p_market) < abs(out.p_model - out.p_market)


def test_calibrate_clips_extreme_deviation(cfg):
    # Even with alpha=0.7, a 0.40 model deviation gets clipped at 0.30.
    # Model 0.95, market 0.50, high confidence -> blended 0.815,
    # which is 0.315 above market -> clipped to 0.80.
    out = calibrate(
        p_model=0.95, p_market=0.50,
        confidence="high", market_history=None, config=cfg,
    )
    assert out.p_final == pytest.approx(0.80)
    assert out.clipped_to_cap


def test_calibrate_polymarket_agreement_used_as_anchor(cfg):
    # Polymarket and Kalshi both near 0.60 -> consensus 0.60 used as anchor.
    # Model says 0.80, high confidence (alpha 0.7) ->
    # 0.7*0.80 + 0.3*0.60 = 0.74.
    history = {"yes_price": 0.60, "volume": 50_000.0}
    out = calibrate(
        p_model=0.80, p_market=0.60,
        confidence="high", market_history=history, config=cfg,
    )
    assert out.used_polymarket
    assert out.p_anchor == pytest.approx(0.60)
    assert out.p_final == pytest.approx(0.74)


def test_calibrate_polymarket_disagreement_falls_back_to_kalshi(cfg):
    # Polymarket disagrees badly with Kalshi -> Polymarket ignored,
    # anchor falls back to Kalshi mid.
    history = {"yes_price": 0.20, "volume": 50_000.0}
    out = calibrate(
        p_model=0.80, p_market=0.60,
        confidence="medium", market_history=history, config=cfg,
    )
    assert not out.used_polymarket
    assert out.p_anchor == pytest.approx(0.60)
    # alpha=0.5 -> 0.5*0.80 + 0.5*0.60 = 0.70
    assert out.p_final == pytest.approx(0.70)
