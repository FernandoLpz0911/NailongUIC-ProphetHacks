from agent.schemas import MarketStat, Prediction
from eval.metrics import brier_score
from forecasting.calibration import calibrate_vs_market, market_probability
from forecasting.ensemble import weighted_average


def test_market_probability_midpoint() -> None:
    stats = {"Yes": MarketStat(yes_ask=0.73, no_ask=0.28)}
    p = market_probability(stats, "Yes")
    assert 0.7 < p < 0.76


def test_calibrate_blends_toward_market() -> None:
    model = Prediction(YES=0.9, NO=0.1)
    stats = {"Yes": MarketStat(last_price=0.5)}
    blended = calibrate_vs_market(model, stats, alpha=0.5)
    assert 0.65 < blended.YES < 0.75


def test_ensemble_average() -> None:
    result = weighted_average(
        [Prediction(YES=0.6, NO=0.4), Prediction(YES=0.8, NO=0.2)],
    )
    assert abs(result.YES - 0.7) < 1e-6


def test_brier_perfect() -> None:
    assert brier_score(1.0, True) == 0.0
