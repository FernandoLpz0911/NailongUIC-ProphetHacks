from __future__ import annotations

import json
import re
from typing import Any

from agent.schemas import Prediction


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object from model output."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start : end + 1])


def parse_prediction(payload: dict[str, Any]) -> Prediction:
    raw = payload.get("prediction") or payload
    yes = float(raw.get("YES") or raw.get("Yes") or raw.get("yes"))
    no = float(raw.get("NO") or raw.get("No") or raw.get("no", 1.0 - yes))
    total = yes + no
    if total <= 0:
        raise ValueError("invalid probabilities")
    yes /= total
    return Prediction(YES=yes, NO=1.0 - yes)
