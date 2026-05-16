from __future__ import annotations

import asyncio
import json
import logging
import time

from agent.config import MODEL_CALL_TIMEOUT_SECONDS, OPENROUTER_API_KEY, PROMPTS_DIR
from agent.router import classify_event, route_model
from agent.openrouter.client import OpenRouterClient
from agent.openrouter.pricing import estimate_cost_usd
from agent.schemas import PredictRequest, Prediction
from forecasting.parser import parse_with_retry
from retrieval.search import SearchDocument

logger = logging.getLogger(__name__)


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def forecast_prompt_name(request: PredictRequest) -> str:
    if classify_event(request) == "hard":
        return "forecast_cot_v1.txt"
    return "forecast_v1.txt"


def build_forecast_messages(
    request: PredictRequest,
    context: list[SearchDocument],
    *,
    prompt_name: str | None = None,
) -> list[dict[str, str]]:
    system = load_prompt("system_v1.txt")
    template = load_prompt(prompt_name or forecast_prompt_name(request))
    news_block = _format_context(context)
    market_block = json.dumps(
        {k: v.model_dump() for k, v in request.market_stats.items()},
        indent=2,
    )
    user_content = (
        f"{template}\n\n"
        f"## Event\nTitle: {request.title}\n"
        f"Rules: {request.rules}\n"
        f"Markets: {request.markets}\n"
        f"## Market stats\n{market_block}\n"
        f"## News context\n{news_block}\n"
        "Respond with JSON only."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


async def forecast(
    request: PredictRequest,
    context: list[SearchDocument],
    *,
    model: str | None = None,
    temperature: float = 0.3,
) -> tuple[Prediction, str, dict]:
    """
    Run forecaster via OpenRouter.

    Returns (prediction, rationale, meta) where meta has model, latency_ms, cost_usd.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not configured")

    tier, selected = route_model(request)
    if model is not None:
        selected = model

    messages = build_forecast_messages(request, context)
    client = OpenRouterClient()
    started = time.perf_counter()
    payload = await asyncio.wait_for(
        client.chat(
            messages,
            model=selected,
            temperature=temperature,
            max_tokens=2048,
            timeout=120.0,
        ),
        timeout=MODEL_CALL_TIMEOUT_SECONDS,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    text = OpenRouterClient.extract_text(payload)
    prediction, rationale = await parse_with_retry(text)

    model_used = str(payload.get("_model_used") or selected)
    usage = payload.get("usage") or {}
    cost_usd = estimate_cost_usd(model_used, usage)

    meta = {
        "model": model_used,
        "model_used": model_used,
        "tier": tier,
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
        "usage": usage,
    }
    return prediction, rationale, meta


def _format_context(docs: list[SearchDocument]) -> str:
    if not docs:
        return "(no news retrieved)"
    lines = []
    for index, doc in enumerate(docs, start=1):
        lines.append(f"{index}. [{doc.title}]({doc.url})\n   {doc.snippet[:400]}")
    return "\n".join(lines)
