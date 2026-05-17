#!/usr/bin/env bash
# =============================================================================
# /opt/nailong/vm-deploy.sh — called by Cloud Build on every push to main
#
# Usage:  vm-deploy.sh <image:tag>
#
# What it does:
#   1. Fetches the full .env from GCP Secret Manager (never touches disk as root)
#   2. Pulls the new Docker image
#   3. Gracefully stops the running agent (gives it 30 s to finish a tick)
#   4. Starts the new container with the data volume and env vars injected
# =============================================================================
set -eu

IMAGE="${1:?Usage: vm-deploy.sh <image:tag>}"
CONTAINER_NAME="nailong-agent"
DATA_DIR="/opt/nailong/data"
PROJECT_ID="project-c31af50d-0e9f-45fd-a5c"
SECRET_NAME="nailong-env"
ENV_FILE="/run/nailong-env"          # tmpfs — survives only this script run

echo "[deploy] Image: ${IMAGE}"
echo "[deploy] $(date -u)"

# ── 1. Fetch .env from Secret Manager into a tmpfs file ──────────────────────
# /run is a tmpfs on Linux — secret is never written to disk.
echo "[deploy] Fetching .env from Secret Manager..."
gcloud secrets versions access latest \
  --secret="${SECRET_NAME}" \
  --project="${PROJECT_ID}" \
  > "${ENV_FILE}"
chmod 600 "${ENV_FILE}"

# ── 2. Configure Docker auth for Artifact Registry ───────────────────────────
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

# ── 3. Pull new image ─────────────────────────────────────────────────────────
echo "[deploy] Pulling ${IMAGE}..."
docker pull "${IMAGE}"

# ── 4. Stop existing container gracefully (SIGTERM → 30 s → SIGKILL) ─────────
if docker inspect "${CONTAINER_NAME}" &>/dev/null; then
  echo "[deploy] Stopping existing container..."
  docker stop --time=30 "${CONTAINER_NAME}" || true
  docker rm "${CONTAINER_NAME}"             || true
fi

# ── 5. Ensure data directory exists ──────────────────────────────────────────
mkdir -p "${DATA_DIR}"

# ── 6. Run new container ──────────────────────────────────────────────────────
echo "[deploy] Starting ${CONTAINER_NAME}..."
docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  --env-file "${ENV_FILE}" \
  -v "${DATA_DIR}:/app/data" \
  --log-driver=gcplogs \
  --log-opt gcp-project="${PROJECT_ID}" \
  --log-opt labels=container_name \
  "${IMAGE}"

# ── 7. Wipe the secret from tmpfs immediately ─────────────────────────────────
rm -f "${ENV_FILE}"

echo "[deploy] Container started. Logs:"
sleep 5
docker logs --tail 20 "${CONTAINER_NAME}"

echo "[deploy] Done."
