from __future__ import annotations

from dataclasses import dataclass

from agent.schemas import PredictRequest
from retrieval.search import SearchDocument


@dataclass(frozen=True)
class GuardFlags:
    force_market: bool


def count_real_docs(context: list[SearchDocument]) -> int:
    """Documents that are not local stubs and have a usable URL."""
    return sum(
        1
        for doc in context
        if doc.url.strip() and "local/stub" not in doc.url
    )


def guard_flags(request: PredictRequest, context: list[SearchDocument]) -> GuardFlags:
    """
    Ambiguous-event guard: thin rules or retrieval → anchor to market (zero edge).

    force_market when rules < 40 chars, rules contain TBD, or fewer than 2 real docs.
    """
    rules = request.rules.strip()
    if len(rules) < 40:
        return GuardFlags(force_market=True)
    if "tbd" in rules.lower():
        return GuardFlags(force_market=True)
    if count_real_docs(context) < 2:
        return GuardFlags(force_market=True)
    return GuardFlags(force_market=False)
