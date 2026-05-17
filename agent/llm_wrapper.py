"""Cost-tracking proxy around any LLMClient.

We don't edit the vendored SDK. Instead this wrapper delegates every method
to the underlying client and records token usage + USD cost to agent.spend
on each call. The kill switch in agent.spend.is_killed() then becomes live.

Pricing table is approximate (OpenRouter / Google pricing as of May 2026).
"""

from __future__ import annotations

import logging
from typing import Any

from ai_prophet.trade.llm import LLMClient
from ai_prophet.trade.llm.base import LLMRequest, LLMResponse

from agent.spend import record_call

logger = logging.getLogger(__name__)

# Approximate per-million-token pricing (USD).
# Format: {model_prefix: (input_per_million, output_per_million)}
_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash":  (0.15, 0.60),
    "gemini-2.5-pro":    (1.25, 10.0),
    "gemini-3.5-pro":    (1.25, 10.0),
    "gemini-3.1-pro":    (1.25, 10.0),
    "claude-sonnet-4":   (3.0,  15.0),
    "claude-haiku-4-5":  (0.80, 4.0),
    "gpt-5":             (5.0,  25.0),
    "gpt-4o":            (5.0,  15.0),
    "deepseek-v3":       (0.27, 1.10),
    "deepseek-r1":       (0.55, 2.20),
}

_DEFAULT_PRICING = (1.0, 5.0)  # safety fallback so cost is never logged as 0


def _price_for_model(model: str) -> tuple[float, float]:
    model_lc = (model or "").lower()
    for prefix, price in _PRICING.items():
        if prefix in model_lc:
            return price
    return _DEFAULT_PRICING


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_per_m, out_per_m = _price_for_model(model)
    return (prompt_tokens / 1_000_000.0) * in_per_m + (completion_tokens / 1_000_000.0) * out_per_m


class CostTrackingLLMClient:
    """Proxy that records cost on every generate() / generate_json() call.

    Implements the LLMClient interface via __getattr__ delegation so any new
    methods the SDK adds keep working without code changes here.
    """

    def __init__(self, inner: LLMClient, provider: str, stage: str | None = None) -> None:
        self._inner = inner
        self._provider = provider
        self._stage = stage
        self.model = getattr(inner, "model", "unknown")

    # The two methods the pipeline actually calls.

    def generate(self, request: LLMRequest, **kw: Any) -> LLMResponse:
        resp = self._inner.generate(request, **kw)
        self._record(resp)
        return resp

    def generate_json(self, messages: Any, tool: Any = None, **kw: Any) -> dict:
        # Some implementations don't return token counts on generate_json,
        # so we approximate from the raw response if present.
        result = self._inner.generate_json(messages, tool=tool, **kw)
        # Best-effort cost recording: peek at last response on the underlying
        # client if it exposes one; otherwise fall back to a flat estimate.
        last = getattr(self._inner, "_last_response", None)
        if isinstance(last, LLMResponse):
            self._record(last)
        else:
            # Conservative estimate: average call ~ 1500 prompt + 200 completion.
            self._record_estimate(prompt_tokens=1500, completion_tokens=200)
        return result

    def _record(self, resp: LLMResponse) -> None:
        try:
            cost = _estimate_cost(
                resp.model,
                int(resp.prompt_tokens or 0),
                int(resp.completion_tokens or 0),
            )
            record_call(
                provider=self._provider,
                model=resp.model or self.model,
                usd_cost=cost,
                prompt_tokens=int(resp.prompt_tokens or 0),
                completion_tokens=int(resp.completion_tokens or 0),
                stage=self._stage,
            )
        except Exception as exc:
            logger.debug("CostTrackingLLMClient record failed: %s", exc)

    def _record_estimate(self, prompt_tokens: int, completion_tokens: int) -> None:
        try:
            cost = _estimate_cost(self.model, prompt_tokens, completion_tokens)
            record_call(
                provider=self._provider, model=self.model,
                usd_cost=cost,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                stage=self._stage,
            )
        except Exception as exc:
            logger.debug("CostTrackingLLMClient estimate record failed: %s", exc)

    # Fall through everything else to the underlying client.
    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)
