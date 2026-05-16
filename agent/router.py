from __future__ import annotations

import re
from datetime import date
from typing import Literal

from agent.config import CHEAP_MODEL, EXPENSIVE_MODEL
from agent.schemas import PredictRequest

Tier = Literal["easy", "hard"]

_FAR_FUTURE_DAYS = 365

_MONTHS: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

_MONTH_DAY_YEAR = re.compile(
    r"\b(" + "|".join(_MONTHS) + r")\w*\s+(\d{1,2}),?\s+(20\d{2})\b",
    re.IGNORECASE,
)
_MONTH_YEAR = re.compile(
    r"\b(" + "|".join(_MONTHS) + r")\w*\s+(20\d{2})\b",
    re.IGNORECASE,
)
_BEFORE_MONTH = re.compile(
    r"\b(?:by|before|until|on)\s+(" + "|".join(_MONTHS) + r")\w*(?:\s+(\d{1,2}))?,?\s*(20\d{2})\b",
    re.IGNORECASE,
)
_YEAR_ONLY = re.compile(r"\b(20\d{2})\b")


def classify_event(request: PredictRequest) -> Tier:
    """Route easy vs hard events for model selection (no API calls)."""
    if _rules_vague(request.rules):
        return "hard"
    if _market_near_fifty(request):
        return "hard"
    if _resolution_far_in_future(request):
        return "hard"
    return "easy"


def select_model(tier: Tier) -> str:
    if tier == "hard":
        return EXPENSIVE_MODEL
    return CHEAP_MODEL


def _rules_vague(rules: str) -> bool:
    text = rules.strip()
    if len(text) < 50:
        return True
    lower = text.lower()
    return "tbd" in lower or "unclear" in lower


def _market_near_fifty(request: PredictRequest) -> bool:
    from forecasting.calibration import market_probability

    p_yes = round(market_probability(request.market_stats, "Yes"), 4)
    return 0.45 <= p_yes <= 0.55


def _resolution_far_in_future(request: PredictRequest) -> bool:
    combined = f"{request.title} {request.rules}"
    dates = _parse_resolution_dates(combined)
    if not dates:
        return False
    today = _today()
    return any((resolved - today).days > _FAR_FUTURE_DAYS for resolved in dates)


def _today() -> date:
    return date.today()


def _parse_resolution_dates(text: str) -> list[date]:
    found: list[date] = []
    for match in _MONTH_DAY_YEAR.finditer(text):
        found.append(_month_day_year(match.group(1), int(match.group(2)), int(match.group(3))))
    for match in _BEFORE_MONTH.finditer(text):
        day = int(match.group(2)) if match.group(2) else 1
        found.append(_month_day_year(match.group(1), day, int(match.group(3))))
    for match in _MONTH_YEAR.finditer(text):
        month = _month_num(match.group(1))
        year = int(match.group(2))
        if month is not None:
            found.append(date(year, month, 28))
    for match in _YEAR_ONLY.finditer(text):
        found.append(date(int(match.group(1)), 12, 31))
    return found


def _month_day_year(month_token: str, day: int, year: int) -> date:
    month = _month_num(month_token)
    if month is None:
        return date(year, 12, 31)
    try:
        return date(year, month, min(day, 28))
    except ValueError:
        return date(year, month, 28)


def _month_num(token: str) -> int | None:
    return _MONTHS.get(token.lower().rstrip("."))


def route_model(request: PredictRequest) -> tuple[Tier, str]:
    tier = classify_event(request)
    return tier, select_model(tier)
