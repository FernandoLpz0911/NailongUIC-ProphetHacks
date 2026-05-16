from __future__ import annotations

from agent.schemas import PredictRequest, PredictResponse, Prediction


async def run_predict_pipeline(request: PredictRequest) -> PredictResponse:
    """
    Orchestrates retrieval → forecasting → calibration.

    Stage 1 returns a stub; P2/P3 wire in via imports from sibling packages.
    """
    # TODO(Stage 2): retrieval.build_context(request) + forecasting.predict(...)
    _ = request.market_stats
    return PredictResponse(
        event_id=request.event_id,
        prediction=Prediction(YES=0.5, NO=0.5),
        rationale="Stub prediction — wire retrieval and forecasting in Stage 2.",
    )
