FROM python:3.11-slim

WORKDIR /app

# git is needed by some SDK imports; curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps before copying app code so Docker can cache this layer
COPY requirements.txt .
COPY packages/ packages/
RUN pip install --no-cache-dir -r requirements.txt

# Copy team-authored code
COPY agent/       agent/
COPY retrieval/   retrieval/
COPY constants/   constants/
COPY scripts/     scripts/

# Runtime directories — overridden by a bind-mount in production so data
# survives container restarts.  Created here so they exist in the image too.
RUN mkdir -p data/traces data/cache

# Verify the CLI is importable at build time (fails fast on bad installs)
RUN python -m agent.run --help > /dev/null

HEALTHCHECK --interval=5m --timeout=30s --start-period=90s \
    CMD python -c "import agent.run" || exit 1

# COST_DB_PATH is overridden at runtime to live inside the mounted data dir
ENV COST_DB_PATH=/app/data/costs.sqlite \
    LOG_LEVEL=INFO

CMD ["python", "-m", "agent.run", "--slug", "nailong_v01"]
