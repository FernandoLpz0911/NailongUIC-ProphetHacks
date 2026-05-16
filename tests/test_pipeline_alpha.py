from agent.pipeline import _calibration_alpha
from agent.schemas import PredictRequest


def test_calibration_uses_category_alpha() -> None:
    request = PredictRequest(
        event_id="e1",
        title="Bitcoin hits new high",
        markets=["Yes", "No"],
        rules="Resolves Yes if BTC exceeds $150k by year end on major exchanges.",
        market_stats={},
    )
    alpha = _calibration_alpha(request, used_live_search=True, num_docs=5)
    # crypto default in alpha_by_category.json is 0.45, boosted by docs
    assert 0.3 <= alpha <= 0.8


def test_low_agreement_reduces_alpha() -> None:
    request = PredictRequest(
        event_id="e2",
        title="Test",
        markets=["Yes", "No"],
        rules="Clear resolution criteria for the event.",
        market_stats={},
    )
    base = _calibration_alpha(request, used_live_search=True, num_docs=3, low_agreement=False)
    reduced = _calibration_alpha(request, used_live_search=True, num_docs=3, low_agreement=True)
    assert reduced < base
