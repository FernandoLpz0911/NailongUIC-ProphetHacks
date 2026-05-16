FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent agent/
COPY retrieval retrieval/
COPY forecasting forecasting/
COPY eval eval/
COPY prompts prompts/
COPY data/sample_event.json data/sample_event.json

EXPOSE 8000
CMD ["uvicorn", "agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
