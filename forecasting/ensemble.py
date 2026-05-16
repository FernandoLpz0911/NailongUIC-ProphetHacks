from __future__ import annotations

from agent.schemas import Prediction


def weighted_average(predictions: list[Prediction], weights: list[float] | None = None) -> Prediction:
    if not predictions:
        raise ValueError("predictions must not be empty")

    if weights is None:
        weights = [1.0 / len(predictions)] * len(predictions)

    if len(weights) != len(predictions):
        raise ValueError("weights length must match predictions")

    total_weight = sum(weights)
    yes = sum(p.YES * w for p, w in zip(predictions, weights, strict=True)) / total_weight
    no = 1.0 - yes
    return Prediction(YES=yes, NO=no)
