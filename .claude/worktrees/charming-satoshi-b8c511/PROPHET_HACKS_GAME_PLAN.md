# Prophet Hacks 2026 — Trading Track Game Plan

> **Superseded by [`PROPHET_HACKS_TRADING_PLAN.md`](PROPHET_HACKS_TRADING_PLAN.md) on 2026-05-16.**
> This document was written before the team realized the Trading Track requires a tick-based
> custom agent against the Core API (not a `POST /predict` endpoint). It is kept in place as
> historical context only — every architectural assumption below is wrong for the track we
> actually entered. Do not implement against it.


**Event:** Prophet Hacks (AI Forecasting Hackathon), May 16–17, 2026
**Track:** Trading (scored on Average Return via Prophet Arena)
**Team size:** 4
**Budget:** $200 OpenRouter.ai credits
**Submission deadline:** Sunday May 17, 5:00 PM
**Evaluation window:** 10 days post-submission (continuous live trading)

---

## 1. The single most important thing to internalize

**Trading Track ≠ Forecasting Track.**

The Forecasting Track is scored on Brier score (raw accuracy of probabilistic predictions). The Trading Track is scored on **Average Return** — your probability output is fed into an *optimal betting simulator* that bets against the live Kalshi `market_stats` (yes_ask, no_ask, last_price) included in every event payload. You get paid when your probabilities differ from the market price *in the right direction*.

This means:

- A perfectly accurate forecast that matches the market price exactly earns you **zero return**.
- A confident, well-calibrated forecast that disagrees with the market and turns out right earns **big return**.
- A confident forecast that disagrees with the market and turns out wrong **loses big**.
- **Calibration > confidence.** Markets are usually right. Only deviate from market price when you have strong, defensible evidence.

Every design decision below flows from this fact.

---

## 2. What the agent must do

The agent contract from Prophet Arena's official spec:

**Input (POST body to your endpoint, per event):**
```json
{
  "event_id": "EVT_1023",
  "title": "Will country X hold an election by March 2026?",
  "markets": ["Yes", "No"],
  "rules": "Resolves YES if a general or presidential election is officially announced…",
  "market_stats": {
    "Yes": {"last_price": 0.72, "yes_ask": 0.73, "no_ask": 0.28},
    "No":  {"last_price": 0.28, "yes_ask": 0.27, "no_ask": 0.72}
  }
}
```

**Output (return body):**
```json
{
  "event_id": "EVT_1023",
  "prediction": {"YES": 0.72, "NO": 0.28},
  "rationale": "Recent polling data suggests a high likelihood of early elections."
}
```

**Hard constraints:**
- Probabilities must sum to 1.0
- ≤ 3600 seconds (1 hour) per event
- Output must be valid JSON or it is not scored
- Submission = zip + a single run command the organizers execute in a standardized environment

**Integration with `ai-prophet`:** Two paths, pick one:
1. **Local module:** `prophet forecast predict --events events.json --local your_module`
2. **HTTP endpoint:** `prophet forecast predict --events events.json --agent-url http://localhost:8000/predict`

We'll build (2) because it's easier to test, scale, and swap models.

---

## 3. Architecture (vertical slice)

```
┌─────────────────────────────────────────────────────────────────┐
│  Prophet Arena server  ──POST event──▶  Our FastAPI /predict    │
└─────────────────────────────────────────────────────────────────┘
                                            │
        ┌───────────────────────────────────┼─────────────────────┐
        ▼                                   ▼                     ▼
   ┌──────────┐                      ┌──────────────┐      ┌──────────┐
   │ Retrieval│  news + context  ──▶ │  Ensemble    │  ──▶ │ Edge +   │
   │ (web srch│                      │  forecaster  │      │ Calibrate│
   │  + cache)│                      │ (OpenRouter) │      │ vs market│
   └──────────┘                      └──────────────┘      └────┬─────┘
                                                                ▼
                                                       prediction + rationale
```

Four pieces, four owners. Clean handoffs at the box boundaries.

---

## 4. Roles

| # | Role | Owner | Primary deliverables |
|---|------|-------|----------------------|
| **P1** | **Lead / Agent Architect** | Esteven | FastAPI server, OpenRouter client, model routing, packaging, final submission |
| **P2** | **Retrieval Engineer** | Teammate 2 | News/web search per event, context builder, source caching |
| **P3** | **Forecasting & Calibration Engineer** | Teammate 3 | Prompts, multi-model ensemble, probability extraction, calibration |
| **P4** | **Eval & Trading Strategy Engineer** | Teammate 4 | Backtest harness, Brier + Avg-Return simulator, edge-vs-market math, spend monitoring |

P1 (Esteven) also acts as integration owner — anyone whose change breaks `main` fixes it before the next stage gate.

---

## 5. Stages, tasks, and gates

Six stages. Each has explicit **gate conditions**: nobody moves to the next stage until they're true. A gate failure means stop and fix, not push through.

### Stage 0 — Pre-kickoff (today, before 9 AM)

**Goal:** Zero friction at the starting line.

- [ ] All 4 members have GitHub accounts and Python 3.11+
- [ ] All 4 have OpenRouter accounts; team OpenRouter key generated and shared via 1Password / encrypted note
- [ ] Single shared repo created on GitHub, all 4 are collaborators
- [ ] Discord server joined (https://discord.gg/NbYM8CDP)
- [ ] Skim of this doc by everyone — 15 minutes
- [ ] Read https://www.prophetarena.co/research/agent-leaderboard-rules — 10 minutes
- [ ] Read README at https://github.com/ai-prophet/ai-prophet — 10 minutes

**Gate:** Everyone can run `pip install ai-prophet` and `prophet --help` works on their machine.

---

### Stage 1 — Setup & architecture lock (Sat 9:00 AM – 11:00 AM, ~2 hr)

**Goal:** Project skeleton with all four boxes stubbed out, end-to-end "hello world" call works.

| Owner | Task | Done when |
|-------|------|-----------|
| **P1** | Init repo, `.env` template, `requirements.txt`, FastAPI skeleton with stub `POST /predict` returning `{"YES":0.5,"NO":0.5}` | `uvicorn` runs, `curl` to `/predict` returns valid stub |
| **P1** | OpenRouter client wrapper with model name + fallback list | A single call to Claude Sonnet 4.6 via OpenRouter returns text |
| **P2** | Pick web search API (recommend: **Tavily** free tier or **Exa**), get key, write `search(query) -> [docs]` stub | Stub returns 3 docs for "Fed rate decision May 2026" |
| **P3** | First-pass prompt template: `system` + event JSON + rules + news → probability JSON | Prompt returns parseable JSON on 3 hand-crafted test events |
| **P4** | Clone `ai-prophet`, run `prophet forecast events --deadline 2026-05-25 --out events.json`, save to `data/` | `events.json` has ≥ 50 live Kalshi events on disk |

**Gate:**
- Repo has CI (a single `pytest -q` GitHub Action) that runs and passes
- `curl -X POST localhost:8000/predict -d @data/sample_event.json` returns a valid `{"prediction": {...}, "rationale": "..."}` response (even if probabilities are stubbed)
- All four have committed at least once

---

### Stage 2 — Vertical-slice MVP (Sat 11:00 AM – 5:00 PM, ~6 hr)

**Goal:** End-to-end agent that handles real events with a single-model baseline. This is the baseline you must beat.

| Owner | Task | Done when |
|-------|------|-----------|
| **P1** | Wire retrieval (P2) + forecaster (P3) into `/predict`. Add request timeout (180s per event for MVP — well under the 1hr limit). Add structured logging of (event_id, model, latency, cost) | A real event from `events.json` POSTed to `/predict` returns a real probability within 3 min |
| **P1** | Add OpenRouter cost tracker — log per-call USD spend to a SQLite file | Total spend after a 10-event run is reported on stdout |
| **P2** | Convert search stub → real Tavily/Exa calls. Add a 24-hour disk cache (`shelve` or `diskcache`) keyed on `(event_id, query)` | Same event run twice → second run uses cache, no API call |
| **P2** | News-pipeline prompt: take event title + rules, generate 2–3 *focused* search queries with an LLM, fetch top 5 results per query, return top 10 deduped chunks | Visibly relevant news returned for 5 sample events |
| **P3** | Real forecasting prompt with: event title, rules, news context, current market_stats. Instruct the model to (a) reason step by step, (b) output a JSON probability, (c) include a one-sentence rationale | Returns calibrated-looking probabilities (not all 0.5, not all 0.99) on 10 events |
| **P3** | Robust JSON parsing with retry and "fix-it" prompt if the model returns malformed output | 100% parseable output across 20 test events |
| **P4** | Local evaluation harness: load `events.json`, hit `/predict` for each event, save predictions to `submission.json`. Compute Brier score against a small set of already-resolved events (use `events_test.json` from repo) | `python eval.py` prints Brier score and per-event log |
| **P4** | **Average-Return simulator** (the critical one): given a prediction dict and market_stats, compute the expected return under the same "optimal betting" the Prophet Arena scorer uses. Document the assumed risk-aversion. | `simulate_return(pred, market_stats, outcome)` returns a number; ranking 10 dummy strategies gives sensible ordering |

**Gate:**
- Baseline agent runs end-to-end on 20 real events without crashing
- Brier score on the resolved subset is **better than 0.25** (random baseline) — likely 0.18–0.22 with single Claude Sonnet
- Average per-event cost is logged and is **under $0.05**
- A first submission is uploaded via `prophet forecast submit --submission submission.json` so the team's name appears on the platform

---

### Stage 3 — First real submission and feedback loop (Sat 5:00 PM – 11:00 PM, ~6 hr)

**Goal:** Stop being a baseline. Build the diagnostic feedback loop that drives every later improvement.

| Owner | Task | Done when |
|-------|------|-----------|
| **P1** | **Model router:** Cheap model (DeepSeek V3.2 or Gemini Flash) for easy events, expensive (Claude Opus 4.7 / GPT-5) for hard ones. "Hard" = vague rules, far resolution date, or market_stats near 0.50. Add a `model_used` field to logs | Router measurably saves cost (≥30%) vs. always-Opus on a 30-event run |
| **P1** | Concurrency: `asyncio.gather` over N=5 events at a time, with semaphore to respect OpenRouter rate limits | 30 events processed in under 5 min wall-clock |
| **P2** | **Source quality filter:** Drop low-quality domains (`reddit.com` comment threads, content farms). Prefer Reuters, AP, gov.\*, official press releases. Add a per-source recency boost. | Visible improvement on a hand-graded sample of 10 events |
| **P2** | **Targeted retrieval for trading:** For each event, also pull Kalshi market chart / Polymarket equivalent if it exists — markets are the strongest signal. (Polymarket has a public API; Kalshi events are already in the payload) | Market history retrieved for 5 sample events |
| **P3** | **Ensemble of 2–3 models:** Run Claude Opus 4.7 + GPT-5 + DeepSeek R1 in parallel, take a weighted average of probabilities. Weights start equal; P4 will tune them later. | Ensemble output beats best single-model Brier by ≥0.005 on 20 events |
| **P3** | **Calibration step:** After getting raw probability `p_model`, compute `p_market` from `market_stats` (use `(1 - no_ask + yes_ask) / 2` as best market estimate). Output `p_final = α * p_model + (1 - α) * p_market`, where `α` ∈ [0.3, 0.8] depending on retrieval confidence. **This is the single most impactful trading change.** | Average Return on backtest set is *positive* and better than baseline |
| **P4** | Run baseline + ensemble + calibrated on the same 50-event backtest. Produce a comparison table (Brier, Avg Return, $ cost). Share in Slack. | Comparison table posted |
| **P4** | **Budget projection:** From cost-per-event measured so far, project 10-day total spend assuming N events/day. Flag risk if projected > $150 | Spreadsheet exists, projection cell visible |

**Gate:**
- Second submission uploaded with ensemble + calibration. Brier < 0.18 on backtest.
- Average Return on backtest is **positive** (better than just bidding the market).
- Projected 10-day spend is under **$150** (leaving $50 buffer).
- Sleep schedule planned: rotate so 2 are always awake. **Nobody pulls a full all-nighter.** Tired code is buggy code, and bugs lose more points than incremental features.

---

### Stage 4 — Trading-track-specific optimization (Sat 11:00 PM – Sun 9:00 AM, ~10 hr with sleep rotation)

**Goal:** Squeeze out edge. This stage is where Trading Track winners and losers separate.

| Owner | Task | Done when |
|-------|------|-----------|
| **P1** | **Prompt caching:** OpenRouter / Anthropic supports prompt caching. Cache the system prompt + rules-explanation prefix, save up to 90% on input tokens for the static portion. | Per-event cost drops by ≥40% on repeated runs |
| **P1** | **Hard timeout protection:** wrap every model call with `asyncio.wait_for(call, timeout=180)`. Fallback to market-anchored probability `p_final = p_market` if all models fail. Never crash, never return malformed output. | Inject 10 random timeouts in test → 100% return valid JSON |
| **P2** | **Per-category retrieval profiles:** Crypto/finance events → CoinGecko + Yahoo Finance. Politics → Politico + Reuters. Sports → ESPN + official league sites. Auto-detect category from event title | Category detection works on 30 sample events |
| **P2** | **News freshness:** For events resolving in <7 days, weight news from last 48 hours much higher | Visible in retrieved context on near-term events |
| **P3** | **Self-consistency check:** Call the forecaster twice (different temperatures or different prompts) and only commit to a non-market-anchored prediction if both agree within 0.10. Otherwise increase α toward market | Backtest Brier improves on uncertain events |
| **P3** | **Chain-of-thought structure for hard events:** For events where the model says it's uncertain, switch to a structured reasoning prompt: list base rate → list update evidence → final probability. (DeepSeek R1 or Claude Opus 4.7 thinking mode) | Hard-event subset Brier improves |
| **P3** | **Reject ambiguous events safely:** If `rules` are ambiguous or context is too thin, return `p_final = p_market` (zero edge, zero risk). Better than betting against the market with bad info | Sample of 5 ambiguous events all return market-anchored output |
| **P4** | **Tune α (market-anchoring weight) per category:** Use the backtest to find the α that maximizes Avg Return for each category. Politics may want α=0.7, crypto may want α=0.4 | Per-category α values committed to config |
| **P4** | **Risk filter:** If `|p_model - p_market| > 0.40`, that's a huge claimed edge — almost always wrong. Cap deviation at 0.30 unless retrieval confidence is very high. | Backtest shows fewer large losses |
| **P4** | **Continuous backtest dashboard:** Streamlit or just a `print()` table — shows Brier, Avg Return, $ cost, # events, top 5 wins, top 5 losses. Refresh after every model/prompt change. | Dashboard runs locally; team checks it before each commit |

**Gate:**
- Backtest Avg Return is positive and ≥ 1.5× the baseline Avg Return from Stage 3.
- Brier score ≤ 0.16 on backtest.
- Cost-per-event ≤ $0.03 average.
- Third submission uploaded with all of the above.
- Get **at least 5 hours of sleep** across the team before the final stage.

---

### Stage 5 — Hardening and final submission (Sun 9:00 AM – 5:00 PM, ~8 hr)

**Goal:** Make it bulletproof. The agent must run unattended for 10 days. Anything that crashes loses you points for every event it misses.

| Owner | Task | Done when |
|-------|------|-----------|
| **P1** | **Containerize:** `Dockerfile` that builds a clean image, single `docker run` command starts the agent on port 8000. Test from a fresh checkout. | Fresh-clone → `docker build && docker run` → endpoint works |
| **P1** | **Graceful degradation chain:** Opus → Sonnet → DeepSeek → Gemini Flash → market-anchored fallback. Each step has a max retries and timeout. | Kill OpenRouter access in test → fallback chain returns valid output |
| **P1** | **Single run command** documented in `README.md`: organizers should be able to `unzip submission.zip && bash run.sh` and have a working agent on `:8000/predict`. | Test on a teammate's clean machine |
| **P2** | **Search API key rotation:** if Tavily/Exa hits rate limit, fall back to a second provider or a free option (DuckDuckGo via `duckduckgo-search`). | Inject rate-limit error → second provider used |
| **P2** | **Final cache prewarm:** for the top 50 live events on the platform right now, prewarm news so the first prod run is fast | Cache file committed (if small) or documented build step |
| **P3** | **Prompt freeze:** stop changing prompts. Commit them to `prompts/` directory with version numbers. | Prompts are in files, not strings in code |
| **P3** | **Output sanitizer:** belt-and-braces — after generating the prediction, validate `0 ≤ p ≤ 1`, sum to 1, no NaN, rationale ≤ 500 chars. If any check fails, fallback to market-anchored. | Fuzz test with 100 malformed model outputs → 100% valid final output |
| **P4** | **Final backtest run:** 100-event backtest with the frozen system. Record Brier, Avg Return, cost. Add to README as evidence. | Numbers in README, committed |
| **P4** | **10-day spend simulation:** estimate based on observed events/day on the platform and current cost-per-event. Set a hard kill-switch if cumulative spend > $180. | Kill-switch tested by setting threshold to $0.01 → agent falls back to market |
| **ALL** | **README.md:** what the agent does, how it works, how to run, backtest numbers, model & cost breakdown, fallback behavior. This is also graded. | README reviewed by all 4 |
| **ALL** | **Final submission package:** zip the repo (sans `.git`, `.env`, `__pycache__`), include `run.sh` and `README.md`, follow organizers' submission instructions | Zip uploaded by 4:00 PM at the latest |

**Gate (the hardest one):**
- A teammate clones a fresh copy of the zip, follows the README, and gets a working `/predict` endpoint in under 10 minutes — without asking the original author any questions.
- Final submission uploaded by **4:00 PM Sunday**. The 5:00 PM deadline is a hard wall, not a target.
- All four team members sign off on the README in Slack.

---

## 6. OpenRouter budget plan

**Total budget:** $200 over (~30 hr build + 10 days eval) = ~11 days of spend.

**Recommended split:**
- Build & test (30 hr): **$30** ceiling. Stop and re-evaluate if you cross this.
- 10-day evaluation: **$150**. Roughly **$15/day**.
- Buffer: **$20** for re-submissions, fixes, post-mortem.

**Recommended models on OpenRouter (May 2026):**

| Use case | Model | Approx cost | When to call |
|---------|-------|-------------|--------------|
| Query generation, JSON repair, routing | `google/gemini-2.5-flash` or `deepseek/deepseek-v3.2` | $0.10–$0.30 / 1M tokens | Always, for any sub-task that doesn't need top-tier reasoning |
| Reasoning backbone | `anthropic/claude-sonnet-4.6` | $3 / $15 per 1M | Default forecaster |
| Hard events / final reasoning | `anthropic/claude-opus-4.7` or `openai/gpt-5` | $5 / $25 per 1M | Only when market is near 0.5 or rules are complex |
| Cheap reasoning sanity check | `deepseek/deepseek-r1` | ~$0.55 / $2.20 per 1M | Ensemble third vote |

**Hard rules:**
- Never call Opus or GPT-5 in retrieval — those are Sonnet-or-cheaper jobs.
- Cache the system prompt + rules explanation. This is a free 40–70% cost win.
- Log every call's cost. If a single event costs more than $0.10, that's a bug.

---

## 7. Stretch goals (only after Stage 5 gate is green)

If you have working bandwidth before the 5 PM deadline:

1. **Best-PR-to-AI-Prophet prize ($300 + $100):** Find a small but real improvement in the `ai-prophet` open-source package (a bug fix, a missing fallback, a better error message) and open a clean PR. This is a separate award track and can be done in parallel by whoever finishes their main task first.

2. **Sponsor track:** Check Discord — sponsors (Fleet AI, Kalshi, Sigma Lab) may post their own bonus challenges during the event. These are usually 1–2 hour add-ons with their own prizes.

3. **Trading-specific bonus:** Build a "high-conviction-only" variant of the agent that returns `p_final = p_market` (zero bet) on 70% of events and only bets when confidence is genuinely high. Sometimes this outperforms the ensemble. Submit it via resubmission and compare.

---

## 8. Communication and rituals

- **Standups:** 3 in the build phase — Saturday 11 AM, Saturday 5 PM, Sunday 9 AM. 5 minutes each, in a voice channel. "What I did, what I'm doing, what's blocked."
- **Slack/Discord channels:** `#general`, `#prs` (auto-feed from GitHub), `#cost-alerts` (where the budget tracker posts), `#help-im-stuck`.
- **Commit and push every 90 minutes minimum.** Even half-finished. We can't help you with code that lives only on your laptop.
- **Always-on dashboard:** P4's backtest dashboard runs on a shared screen / shared link. Everyone glances at it before merging.
- **"Two-pizza" rule for PRs:** if a PR touches more than two of the four boxes (server, retrieval, forecaster, eval), it's too big — split it.

---

## 9. Failure modes to actively guard against

1. **Over-engineering the retrieval pipeline at the expense of the forecaster.** Retrieval is supporting cast. The forecaster is the lead.
2. **Falling in love with one model.** Always ensemble. The OpenRouter premise is "pick the right tool per call."
3. **Forgetting Trading Track ≠ Forecasting Track.** Every prompt change should be evaluated on Avg Return too, not just Brier.
4. **Submitting once and walking away.** Resubmission is allowed and free. Aim for 3 submissions: end of Stage 2 (baseline), end of Stage 3 (calibrated), end of Stage 5 (final).
5. **Crashing during evaluation.** A crashed agent returns no prediction → effectively a zero bet on that event. Worse: a malformed JSON output isn't scored at all. The fallback chain in Stage 5 exists specifically to prevent this.
6. **Blowing the budget in build.** $30 is plenty for 30 hours of testing if you cache and use small models for non-reasoning sub-tasks.
7. **Spending the last 4 hours adding features instead of hardening.** Stage 5 has zero new features for a reason.

---

## 10. Definition of success

We will know we won (or could have) if:
- Final agent has Brier ≤ 0.16 and positive Avg Return on backtest
- Total 11-day spend is between $100 and $180 (under = underutilized, over = unsafe)
- Agent runs unattended for 10 days with zero crashes and zero malformed outputs
- Submission packaging works on a fresh clone in under 10 minutes
- Every team member can explain what every other team member built

The Korea trip is a $2,000 prize. The Trading Track has one winner. Build accordingly.

---

*Last updated: May 16, 2026 — pre-kickoff.*
