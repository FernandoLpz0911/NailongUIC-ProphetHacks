from __future__ import annotations

from pathlib import Path

from agent.config import PROMPTS_DIR
from agent.schemas import PredictRequest, Prediction
from retrieval.search import SearchDocument


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


async def forecast(
    request: PredictRequest,
    context: list[SearchDocument],
) -> tuple[Prediction, str]:
    """
    Run the forecasting model (single-model baseline → ensemble in Stage 3).

    Stage 1 stub; P3 replaces with OpenRouter-backed prompts.
    """
    _ = (load_prompt("system_v1.txt"), context)
    return (
        Prediction(YES=0.5, NO=0.5),
        "Stub forecast — implement prompts and JSON parsing in Stage 2.",
    )
