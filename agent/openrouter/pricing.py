from __future__ import annotations

from typing import Any

# USD per 1M tokens (input, output) — rough defaults for cost logging
MODEL_RATES: dict[str, tuple[float, float]] = {
    "anthropic/claude-sonnet-4": (3.0, 15.0),
    "anthropic/claude-sonnet-4.6": (3.0, 15.0),
    "google/gemini-2.5-flash": (0.15, 0.60),
    "deepseek/deepseek-v3.2": (0.27, 1.10),
    "deepseek/deepseek-r1": (0.55, 2.20),
}


def estimate_cost_usd(model: str, usage: dict[str, Any]) -> float:
    if not usage:
        return 0.0
    if "cost" in usage and usage["cost"] is not None:
        return float(usage["cost"])
    if "total_cost" in usage and usage["total_cost"] is not None:
        return float(usage["total_cost"])

    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    input_rate, output_rate = MODEL_RATES.get(model, (1.0, 3.0))
    return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000.0
