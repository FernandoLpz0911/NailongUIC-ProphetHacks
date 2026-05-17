# P2 — Retrieval Module

> **Partially superseded by [`PROPHET_HACKS_TRADING_PLAN.md`](PROPHET_HACKS_TRADING_PLAN.md) on 2026-05-16.**
> The retrieval module itself (`retrieval/retrieval.py`) is correct and still in use. However,
> the "How P3 uses confidence for alpha" section below refers to a `p_final = α·p_model + (1-α)·p_market`
> blend that now lives in [`agent/calibration.py`](agent/calibration.py); the SDK pipeline (not a
> standalone `POST /predict`) is the consumer. Treat the call signature and return shape below
> as authoritative; treat the integration notes as historical.


**Owner:** Andres (Teammate 2)
**File:** `retrieval/retrieval.py`
**Status:** ✅ Complete — Stages 2 through 5

---

## What this module does

Every time the agent receives an event from Prophet Arena, it needs real-world context before it can forecast a probability. That's this module's job.

Given an event ID, title, and resolution rules, `get_context()` returns a structured dict containing:
- Relevant news chunks from the web (filtered, ranked, deduped)
- A Polymarket price signal if a matching market exists
- A category label (crypto, finance, politics, sports, general)
- A confidence score (high / medium / low)

The forecaster (P3) uses this output to decide how far to deviate from the Kalshi market price. The confidence score directly drives the alpha (market-anchoring weight) in the calibration step.

---

## How to use it

### Install dependencies
```bash
pip install tavily-python diskcache python-dotenv openai httpx duckduckgo-search
```

### Add to `.env`
```
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxx
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxx
```

### Call from your code
```python
from retrieval import get_context

context = get_context(
    event_id="EVT_1023",
    title="Will the Fed cut rates in June 2026?",
    rules="Resolves YES if the Federal Reserve announces a rate cut at the June 2026 FOMC meeting.",
    resolution_date="2026-06-18"  # optional — enables recency boost for near-term events
)
```

### Return shape
```python
{
    "event_id":       "EVT_1023",
    "retrieved_at":   "2026-05-16T18:07:39+00:00",
    "category":       "finance",
    "confidence":     "medium",
    "market_history": {
        "source":    "polymarket",
        "question":  "Will the Fed cut rates in June 2026?",
        "yes_price": 0.515,
        "no_price":  0.485,
        "volume":    729047.32,
        "url":       "https://polymarket.com/event/fed-june-2026"
    },
    "chunks": [
        {
            "title":          "Fed signals potential rate cut at June meeting",
            "url":            "https://reuters.com/...",
            "snippet":        "Federal Reserve officials indicated...",
            "source":         "reuters.com",
            "published_date": "2026-05-15",
            "quality":        "high"
        },
        ...
    ]
}
```

---

## P3 — How to use confidence for alpha

The `confidence` field maps directly to your alpha (market-anchoring weight):

| confidence | recommended alpha | meaning |
|------------|------------------|---------|
| `"high"`   | 0.6 – 0.8        | Strong evidence — lean on the model, deviate from market |
| `"medium"` | 0.4 – 0.5        | Mixed evidence — blend model and market evenly |
| `"low"`    | 0.2 – 0.3        | Weak evidence — stay close to market price, don't risk it |

If `market_history` is present, the `yes_price` and `no_price` are a Polymarket consensus price. This is the strongest signal available and should be weighted heavily alongside the Kalshi `market_stats` already in the event payload.

---

## What each stage added

### Stage 2 — Core pipeline
- Real Tavily API search (3 queries per event, 5 results each)
- LLM-generated queries via Gemini 2.5 Flash Lite on OpenRouter — smarter than using the raw event title as a search query
- 24-hour disk cache keyed on `event_id` — same event hit twice returns instantly with no API call
- Deduplication by URL
- Fallback: if LLM query generation fails for any reason, falls back to the event title

### Stage 3 — Source quality + market signal
- **Source quality filter:** blocked domains (Reddit, content farms, tabloids) are dropped entirely. High-quality domains (Reuters, AP, Bloomberg, gov sites, CoinDesk, etc.) are ranked first.
- **Polymarket integration:** searches Polymarket's public API for a matching market and returns current yes/no prices and trading volume. This is the single strongest external signal — prediction markets aggregate all publicly available information into a price.
- **Confidence scoring:** updated to factor in both chunk count and proportion of high-quality sources, not just raw count.

### Stage 4 — Category-aware retrieval
- **Category detection:** auto-detects event category (crypto, finance, politics, sports, general) from the event title using keyword matching.
- **Priority domains per category:** crypto events rank CoinDesk/CoinTelegraph first; finance events rank Bloomberg/FT/WSJ/Fed first; politics events rank Politico/Reuters/AP first; sports events rank ESPN/official leagues first.
- **Recency boost:** if `resolution_date` is provided and the event resolves within 7 days, articles from the last 48 hours are prepended before sorting so near-term signals surface at the top.

### Stage 5 — Reliability + prewarm
- **DuckDuckGo fallback:** if Tavily hits a rate limit or errors, the module automatically switches to DuckDuckGo for the remaining queries. Zero crashes, zero empty context.
- **Cache prewarm:** `prewarm_cache(events)` accepts a list of events and pre-fetches context for all of them. Run this before the 10-day eval window opens so every first call is an instant cache hit.

---

## CSV logging

Every call to `get_context()` — cache hit or miss — appends a row to `retrieval_log.csv` in the working directory. This is a debugging and audit tool.

Columns: `event_id`, `retrieved_at`, `category`, `confidence`, `chunk_count`, `high_quality_count`, `polymarket_yes_price`, `polymarket_no_price`, `polymarket_volume`, `top_sources`, `top_titles`

Example row:
```
EVT_001, 2026-05-16T18:09:58+00:00, finance, medium, 10, 3, 0.515, 0.485, 729047.32,
federalreserve.gov | calendarx.com | chicagofed.org,
"Calendar: June 2026 - Federal Reserve Board | The Fed - April 28-29 FOMC Meeting"
```

Add to `.gitignore` — this is a local debug file, not for the repo:
```
retrieval_log.csv
search_cache/
.env
```

---

## Test script

```python
# test_handoff.py — run this to verify P2 output before wiring into /predict
import json
from retrieval import get_context

events = [
    ("EVT_001", "Will the Fed cut rates in June 2026?",
     "Resolves YES if the Federal Reserve announces a rate cut at the June 2026 FOMC meeting."),
    ("EVT_002", "Will Bitcoin exceed $120,000 by July 2026?",
     "Resolves YES if BTC/USD closes above $120,000 on any day before July 1 2026."),
    ("EVT_003", "Will France hold a snap election before September 2026?",
     "Resolves YES if the French president officially dissolves the National Assembly before Sep 1 2026."),
]

for event_id, title, rules in events:
    context = get_context(event_id, title, rules)
    print(f"\n{'='*50}")
    print(f"Event:      {title}")
    print(f"Category:   {context['category']}")
    print(f"Confidence: {context['confidence']}")
    print(f"Chunks:     {len(context['chunks'])}")
    print(f"Polymarket: {context.get('market_history')}")

# Uncomment once P3 has their module ready
# from forecaster import get_prediction
# prediction = get_prediction(event, context)
# print(json.dumps(prediction, indent=2))
```

---

## File structure

```
retrieval/
├── retrieval.py         # this module
├── test_handoff.py      # shared test script for P2 + P3 integration
├── .env                 # API keys (never commit)
├── search_cache/        # diskcache folder (never commit)
└── retrieval_log.csv    # debug log (never commit)
```

---

## Questions

Ping Andres on Discord. The handoff interface (`get_context` return shape) is fixed — P1 and P3 can build against it without waiting on any further changes from P2.
