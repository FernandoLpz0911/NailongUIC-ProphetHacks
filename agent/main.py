from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent.logging_utils import setup_logging
from agent.pipeline import run_predict_pipeline
from agent.schemas import PredictRequest, PredictResponse
from agent.services import get_cost_tracker


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging()
    yield


app = FastAPI(
    title="Prophet Hacks Trading Agent",
    description="HTTP agent for Prophet Arena Trading Track (POST /predict).",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/stats")
async def stats() -> dict[str, float]:
    return {"total_spend_usd": get_cost_tracker().total_spend()}


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest) -> PredictResponse:
    return await run_predict_pipeline(body)
