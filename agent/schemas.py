from typing import Any

from pydantic import BaseModel, Field, field_validator


class MarketStat(BaseModel):
    last_price: float | None = None
    yes_ask: float | None = None
    no_ask: float | None = None


class PredictRequest(BaseModel):
    event_id: str
    title: str
    markets: list[str]
    rules: str
    market_stats: dict[str, MarketStat] = Field(default_factory=dict)


class Prediction(BaseModel):
    YES: float
    NO: float

    @field_validator("YES", "NO")
    @classmethod
    def probability_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        return value


class PredictResponse(BaseModel):
    event_id: str
    prediction: Prediction
    rationale: str


def normalize_market_keys(market_stats: dict[str, Any]) -> dict[str, MarketStat]:
    normalized: dict[str, MarketStat] = {}
    for key, value in market_stats.items():
        if isinstance(value, MarketStat):
            normalized[key] = value
        else:
            normalized[key] = MarketStat.model_validate(value)
    return normalized
