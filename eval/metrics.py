from __future__ import annotations


def brier_score(probability_yes: float, outcome_yes: bool) -> float:
    outcome = 1.0 if outcome_yes else 0.0
    return (probability_yes - outcome) ** 2
