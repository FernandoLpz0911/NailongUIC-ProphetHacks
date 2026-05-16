from __future__ import annotations

import re
from typing import Literal

Category = Literal["politics", "crypto", "sports", "general"]

CATEGORIES: tuple[Category, ...] = ("politics", "crypto", "sports", "general")

_POLITICS = (
    "election",
    "president",
    "congress",
    "senate",
    "parliament",
    "vote",
    "gop",
    "democrat",
    "republican",
    "white house",
    "prime minister",
    "legislation",
    "geopolitic",
    "nato",
    "sanction",
    "tariff",
    "supreme court",
    "cabinet",
    "ballot",
    "referendum",
    "impeach",
    "governor",
    "legislature",
)
_CRYPTO = (
    "bitcoin",
    "btc",
    "ethereum",
    "eth ",
    " crypto",
    "blockchain",
    "solana",
    "coinbase",
    "defi",
    "nft",
    "token",
    "altcoin",
    "dogecoin",
    "binance",
    "memecoin",
    "stablecoin",
    "xrp",
    "ripple",
)
_SPORTS = (
    "nfl",
    "nba",
    "mlb",
    "nhl",
    "mls",
    "super bowl",
    "world cup",
    "championship",
    "playoff",
    "mvp",
    "premier league",
    "uefa",
    "olympic",
    "tennis",
    "golf",
    "nascar",
    "formula 1",
    "f1 ",
    " vs ",
    "match winner",
    "game winner",
    "stanley cup",
    "march madness",
    "grand slam",
    "pga",
    "ufc",
    "boxing",
)

_WORD = re.compile(r"[a-z0-9]+")


def detect_category(title: str, rules: str) -> Category:
    """Classify an event using keyword heuristics on title and rules."""
    text = f"{title} {rules}".lower()
    tokens = set(_WORD.findall(text))

    scores = {
        "politics": _score_keywords(text, tokens, _POLITICS),
        "crypto": _score_keywords(text, tokens, _CRYPTO),
        "sports": _score_keywords(text, tokens, _SPORTS),
    }
    best = max(scores, key=scores.get)
    if scores[best] <= 0:
        return "general"
    return best  # type: ignore[return-value]


def _score_keywords(text: str, tokens: set[str], keywords: tuple[str, ...]) -> int:
    score = 0
    for kw in keywords:
        kw = kw.strip()
        if " " in kw:
            if kw in text:
                score += 2
        elif kw in tokens or kw in text:
            score += 1
    return score
