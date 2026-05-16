from fastapi import FastAPI

from agent.pipeline import run_predict_pipeline
from agent.schemas import PredictRequest, PredictResponse

app = FastAPI(
    title="Prophet Hacks Trading Agent",
    description="HTTP agent for Prophet Arena Trading Track (POST /predict).",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest) -> PredictResponse:
    return await run_predict_pipeline(body)
