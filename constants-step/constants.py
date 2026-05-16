"""
Trading rule constants for the Prophet agent.

These values are enforced by the Prophet Arena API on every trade submission.
They are defined here once and imported wherever needed — extraction,
hypothesis, verify, sizing, and the main tick loop.

Source: Prophet Hacks participant guide + Trade Quick Start docs.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Tick timing
# ---------------------------------------------------------------------------

TICK_INTERVAL_MINUTES: int = 15          # How often a new tick window opens
TICK_LEASE_MINUTES: int = 10             # Time allowed to submit trades per tick
TICK_INTERVAL_SECONDS: int = TICK_INTERVAL_MINUTES * 60
TICK_LEASE_SECONDS: int = TICK_LEASE_MINUTES * 60

# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

STARTING_CASH: float = 10_000.0         # Simulated cash at experiment start

# ---------------------------------------------------------------------------
# Position limits  (API rejects trades that breach these)
# ---------------------------------------------------------------------------

MAX_OPEN_POSITIONS: int = 30            # Max number of simultaneously open positions
MAX_NOTIONAL_PER_MARKET: float = 1_000.0  # Max dollar exposure in any single market
MAX_GROSS_EXPOSURE: float = 10_000.0    # Max total dollar exposure across all positions

# ---------------------------------------------------------------------------
# Trade limits
# ---------------------------------------------------------------------------

MAX_TRADES_PER_TICK: int = 10           # Hard cap on trade intents per tick
TRADE_FEES: float = 0.0                 # No fees on Prophet Arena

# ---------------------------------------------------------------------------
# Payout
# ---------------------------------------------------------------------------

PAYOUT_YES: float = 1.0                 # YES resolves to $1 per share
PAYOUT_NO: float = 0.0                  # NO resolves to $0 per share

# ---------------------------------------------------------------------------
# Extraction stage thresholds  (tunable, informed by trading rules above)
# ---------------------------------------------------------------------------

EXTRACTION_MAX_OUTPUT: int = 25         # Markets passed to hypothesis per tick
EXTRACTION_MIN_SPREAD: float = 0.08     # Minimum bid-ask spread to have edge potential
EXTRACTION_MIN_VOLUME: float = 500.0    # Minimum 24h volume for liquidity
EXTRACTION_MIN_ASK: float = 0.10        # Ignore near-certain NO outcomes
EXTRACTION_MAX_ASK: float = 0.90        # Ignore near-certain YES outcomes
EXTRACTION_MAX_DAYS: int = 30           # Ignore markets resolving too far out

# ---------------------------------------------------------------------------
# Hypothesis stage
# ---------------------------------------------------------------------------

HYPOTHESIS_MIN_EDGE: float = 0.10       # Minimum edge (our prob - market ask) to trade
HYPOTHESIS_MAX_RETRIES: int = 2         # Max Verify → Hypothesis loop-backs per market

# ---------------------------------------------------------------------------
# Sizing  (Kelly-based, conservative)
# ---------------------------------------------------------------------------

KELLY_FRACTION: float = 0.25            # Quarter-Kelly to limit variance
MIN_TRADE_SIZE: float = 50.0            # Don't bother with tiny positions
MAX_TRADE_SIZE: float = MAX_NOTIONAL_PER_MARKET  # Hard ceiling per market