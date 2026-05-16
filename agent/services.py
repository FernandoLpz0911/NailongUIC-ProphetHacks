from __future__ import annotations

from agent.openrouter.client import OpenRouterClient
from agent.openrouter.cost_tracker import CostTracker

_client: OpenRouterClient | None = None
_cost_tracker: CostTracker | None = None


def get_openrouter_client() -> OpenRouterClient:
    global _client
    if _client is None:
        _client = OpenRouterClient()
    return _client


def get_cost_tracker() -> CostTracker:
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker
