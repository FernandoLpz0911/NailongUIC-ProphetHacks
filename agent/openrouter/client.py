from __future__ import annotations

import os
from typing import Any

import httpx

from agent.config import DEFAULT_MODEL, FALLBACK_MODELS, OPENROUTER_API_KEY, OPENROUTER_BASE_URL


class OpenRouterClient:
    """Thin wrapper around OpenRouter chat completions with model fallbacks."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        default_model: str = DEFAULT_MODEL,
        fallback_models: list[str] | None = None,
    ) -> None:
        self.api_key = api_key or OPENROUTER_API_KEY
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.fallback_models = fallback_models if fallback_models is not None else FALLBACK_MODELS

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        models = [model or self.default_model, *self.fallback_models]
        last_error: Exception | None = None

        async with httpx.AsyncClient(timeout=120.0) as client:
            for candidate in models:
                try:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://github.com"),
                            "X-Title": os.getenv("OPENROUTER_APP_NAME", "prophet-hacks-agent"),
                        },
                        json={
                            "model": candidate,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                    payload["_model_used"] = candidate
                    return payload
                except Exception as exc:  # noqa: BLE001 — try next model in fallback chain
                    last_error = exc

        raise RuntimeError(f"all OpenRouter models failed: {last_error}")

    @staticmethod
    def extract_text(payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content") or "")
