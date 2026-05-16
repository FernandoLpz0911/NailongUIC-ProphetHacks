"""Main FastAPI server and agent pipeline for Prophet Hacks."""

import os
import asyncio
import logging
import httpx
from fastapi import FastAPI, Request
from dotenv import load_dotenv

from extraction import extraction
from verification import verification

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

app = FastAPI()
logger = logging.getLogger(__name__)


async def call_openrouter(prompt: str) -> str:
    """Handles the async HTTP call to OpenRouter (Claude Sonnet 4.6)."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost:8000",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "anthropic/claude-sonnet-4.6",
        "messages": [
            {
                "role": "system",
                "content": extraction.FORECASTER_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2
    }

    async with httpx.AsyncClient(timeout=170.0) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


@app.post("/predict")
async def predict(request: Request):
    """Endpoint hit by the Prophet Arena Scorer."""
    market_id = "UNKNOWN"
    try:
        event_payload = await request.json()
        market_id = event_payload.get("market_id", "UNKNOWN")
        quote = event_payload.get("quote", {})

        context = "No live news available yet."

        prompt = extraction.build_user_prompt(event_payload, context)

        llm_response = await asyncio.wait_for(
            call_openrouter(prompt), timeout=175.0
        )

        final_prediction = extraction.extract_and_validate_prediction(
            llm_response, market_id, quote
        )

        if final_prediction is None:
            logger.warning(
                "Skipping trade for %s due to extraction failure.", market_id
            )
            return {"status": "skipped", "reason": "extraction_failure"}

        is_valid = verification.verify_prediction(final_prediction)

        if not is_valid:
            logger.warning(
                "Skipping trade for %s due to verification failure.", market_id
            )
            return {"status": "skipped", "reason": "verification_failure"}

        return final_prediction

    except (
        ValueError,
        TypeError,
        KeyError,
        asyncio.TimeoutError,
        httpx.HTTPError
    ) as e:
        logger.error("Pipeline crashed for %s: %s", market_id, e)
        return {"status": "error", "reason": str(e)}
