from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.config import CHEAP_MODEL, OPENROUTER_API_KEY
from agent.openrouter.client import OpenRouterClient
from agent.schemas import Prediction

logger = logging.getLogger(__name__)


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start : end + 1])


def parse_prediction(payload: dict[str, Any]) -> tuple[Prediction, str]:
    raw = payload.get("prediction") or payload
    rationale = str(payload.get("rationale") or payload.get("reasoning") or "").strip()
    yes = float(raw.get("YES") or raw.get("Yes") or raw.get("yes"))
    no = float(raw.get("NO") or raw.get("No") or raw.get("no", 1.0 - yes))
    total = yes + no
    if total <= 0:
        raise ValueError("invalid probabilities")
    yes /= total
    return Prediction(YES=yes, NO=1.0 - yes), rationale


async def parse_with_retry(
    text: str,
    *,
    max_attempts: int = 2,
) -> tuple[Prediction, str]:
    last_error: Exception | None = None
    current_text = text

    for attempt in range(max_attempts):
        try:
            payload = extract_json_object(current_text)
            return parse_prediction(payload)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt + 1 >= max_attempts or not OPENROUTER_API_KEY:
                break
            current_text = await _fix_json(current_text)

    raise ValueError(f"could not parse model output: {last_error}")


async def _fix_json(bad_text: str) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("no API key for JSON repair")
    client = OpenRouterClient()
    payload = await client.chat(
        [
            {
                "role": "user",
                "content": (
                    "Fix this into valid JSON with keys prediction.YES, prediction.NO, rationale:\n"
                    f"{bad_text[:4000]}"
                ),
            }
        ],
        model=CHEAP_MODEL,
        temperature=0.0,
        max_tokens=512,
        timeout=45.0,
    )
    return OpenRouterClient.extract_text(payload)
