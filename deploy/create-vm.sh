#!/usr/bin/env bash
# =============================================================================
# deploy/create-vm.sh — ONE-TIME GCP infrastructure setup
#
# Run this ONCE from your local machine (with gcloud auth + billing enabled).
# After this script completes, every push to main will auto-deploy via
# Cloud Build (see cloudbuild.yaml).
#
# Prerequisites:
#   gcloud auth login
#   gcloud config set project project-c31af50d-0e9f-45fd-a5c
#   Billing must be enabled on the project.
# =============================================================================
set -eu

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ID="project-c31af50d-0e9f-45fd-a5c"
REGION="us-central1"
ZONE="us-central1-a"
VM_NAME="nailong-agent"
MACHINE_TYPE="e2-medium"
SA_NAME="nailong-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
REPO_NAME="nailong"
GITHUB_OWNER="FernandoLpz0911"
GITHUB_REPO="NailongUIC-ProphetHacks"
# Branch the Cloud Build trigger watches.  Set to the repo's default branch
# (currently `refactor-connection`).  Override with: GITHUB_BRANCH=foo bash ...
GITHUB_BRANCH="${GITHUB_BRANCH:-refactor-connection}"

echo "==> Setting project to ${PROJECT_ID}"
gcloud config set project "${PROJECT_ID}"

# ── 1. Enable required APIs ────────────────────────────────────────────────────
echo "==> Enabling GCP APIs..."
gcloud services enable \
  cloudbuild.googleapis.com \
  compute.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  iap.googleapis.com \
  --quiet

# ── 2. Artifact Registry repository ───────────────────────────────────────────
echo "==> Creating Artifact Registry repo '${REPO_NAME}'..."
gcloud artifacts repositories create "${REPO_NAME}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Nailong trading agent images" \
  --quiet 2>/dev/null || echo "    (repo already exists, skipping)"

# ── 3. VM service account ─────────────────────────────────────────────────────
echo "==> Creating service account ${SA_EMAIL}..."
gcloud iam service-accounts create "${SA_NAME}" \
  --display-name="Nailong agent VM service account" \
  --quiet 2>/dev/null || echo "    (service account already exists, skipping)"

# Wait for SA to propagate before adding IAM bindings
echo "    Waiting for service account to propagate..."
for i in 1 2 3 4 5; do
  if gcloud iam service-accounts describe "${SA_EMAIL}" --quiet >/dev/null 2>&1; then
    echo "    Service account ready."
    break
  fi
  echo "    Not ready yet, retrying in 5s... (${i}/5)"
  sleep 5
done

# Grant: pull Docker images from Artifact Registry
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/artifactregistry.reader" \
  --quiet

# Grant: read secrets (for the .env file at deploy/runtime)
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/secretmanager.secretAccessor" \
  --quiet

# Grant: write logs to Cloud Logging
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/logging.logWriter" \
  --quiet

# ── 4. Cloud Build service account IAM ────────────────────────────────────────
# Cloud Build needs to: SSH to the VM (osAdminLogin + IAP) and push images
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')
CB_SA="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"

echo "==> Granting Cloud Build SA SSH + Artifact Registry access..."
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="${CB_SA}" \
  --role="roles/compute.osAdminLogin" \
  --quiet

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="${CB_SA}" \
  --role="roles/iap.tunnelResourceAccessor" \
  --quiet

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="${CB_SA}" \
  --role="roles/artifactregistry.writer" \
  --quiet

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="${CB_SA}" \
  --role="roles/compute.viewer" \
  --quiet

# Cloud Build needs serviceAccountUser on the VM's SA before it can SSH into a
# VM that runs as that SA.  Without this, `gcloud compute ssh/scp` fails with
# "User does not have permission" on the upload-deploy-script + deploy steps.
echo "==> Granting Cloud Build SA serviceAccountUser on ${SA_EMAIL}..."
gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
  --member="${CB_SA}" \
  --role="roles/iam.serviceAccountUser" \
  --quiet

# CLOUD_LOGGING_ONLY in cloudbuild.yaml requires explicit logWriter on the CB SA
# in projects created after April-2024 (no default GCS bucket → no fallback).
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="${CB_SA}" \
  --role="roles/logging.logWriter" \
  --quiet

# ── 5. Create the Secret for the .env file ────────────────────────────────────
echo ""
echo "==> Creating Secret Manager secret 'nailong-env'..."
echo "    (You will fill this in manually — see instructions below)"
gcloud secrets create nailong-env \
  --replication-policy="automatic" \
  --quiet 2>/dev/null || echo "    (secret already exists, skipping)"

# Add a placeholder version so the secret is valid
echo "PA_SERVER_API_KEY=REPLACE_ME" | \
  gcloud secrets versions add nailong-env --data-file=- --quiet 2>/dev/null || true

echo ""
echo "  *** ACTION REQUIRED ***"
echo "  Go to: https://console.cloud.google.com/security/secret-manager?project=${PROJECT_ID}"
echo "  Click 'nailong-env' → 'Edit Secret' → 'Add Version'."
echo "  Paste the full contents of your .env file (with real API keys) as the secret value."
echo "  Then click 'Add New Version'."
echo ""

# ── 6. Compute Engine VM ──────────────────────────────────────────────────────
echo "==> Creating Compute Engine VM '${VM_NAME}' (${MACHINE_TYPE} in ${ZONE})..."

# Startup script: installs Docker and configures the deployment directories
STARTUP_SCRIPT=$(cat <<'STARTUP'
#!/bin/bash
set -e

# Install Docker if not present
if ! command -v docker &>/dev/null; then
  apt-get update -q
  apt-get install -y -q ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/debian/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) \
    signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/debian \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | tee /etc/apt/sources.list.d/docker.list > /dev/null
  apt-get update -q
  apt-get install -y -q docker-ce docker-ce-cli containerd.io
fi

# Create deployment directory
mkdir -p /opt/nailong/data

# Configure Docker to authenticate with Artifact Registry using the VM SA
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet || true

echo "Startup complete."
STARTUP
)

gcloud compute instances create "${VM_NAME}" \
  --zone="${ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=20GB \
  --boot-disk-type=pd-standard \
  --service-account="${SA_EMAIL}" \
  --scopes=cloud-platform \
  --tags=nailong-agent \
  --metadata="startup-script=${STARTUP_SCRIPT}" \
  --no-address \
  --quiet 2>/dev/null || echo "    (VM already exists, skipping)"

# Enable IAP for the VM (allows Cloud Build to SSH without a public IP)
gcloud compute firewall-rules create allow-iap-ssh \
  --direction=INGRESS \
  --priority=1000 \
  --network=default \
  --action=ALLOW \
  --rules=tcp:22 \
  --source-ranges=35.235.240.0/20 \
  --target-tags=nailong-agent \
  --quiet 2>/dev/null || echo "    (firewall rule already exists, skipping)"

# ── 7. Upload the VM-side deploy script ───────────────────────────────────────
echo "==> Waiting 60 s for VM startup script to finish..."
sleep 60

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "==> Uploading vm-deploy.sh to VM..."
gcloud compute scp "${SCRIPT_DIR}/vm-deploy.sh" "${VM_NAME}:/opt/nailong/vm-deploy.sh" \
  --zone="${ZONE}" \
  --tunnel-through-iap \
  --quiet
gcloud compute ssh "${VM_NAME}" \
  --zone="${ZONE}" \
  --tunnel-through-iap \
  --command="chmod +x /opt/nailong/vm-deploy.sh" \
  --quiet

# ── 8. Cloud Build GitHub trigger ─────────────────────────────────────────────
echo ""
echo "==> Cloud Build GitHub trigger setup"
echo "    Cloud Build requires the GitHub App to be installed manually before"
echo "    a trigger can be created via CLI."
echo ""
echo "    Follow these steps once:"
echo "    1. Open https://console.cloud.google.com/cloud-build/triggers?project=${PROJECT_ID}"
echo "    2. Click 'Connect Repository' → 'GitHub (Cloud Build GitHub App)'"
echo "    3. Authorise the app and select '${GITHUB_OWNER}/${GITHUB_REPO}'"
echo "    4. Then run the following command to create the trigger:"
echo ""
echo "    gcloud builds triggers create github \\"
echo "      --project=${PROJECT_ID} \\"
echo "      --repo-name=${GITHUB_REPO} \\"
echo "      --repo-owner=${GITHUB_OWNER} \\"
echo "      --branch-pattern='^${GITHUB_BRANCH}\$' \\"
echo "      --build-config=cloudbuild.yaml \\"
echo "      --name=nailong-deploy-on-push \\"
echo "      --description='Build and deploy on push to ${GITHUB_BRANCH}'"
echo ""
echo "    If a trigger already exists on the wrong branch, update it with:"
echo ""
echo "    gcloud builds triggers update github nailong-deploy-on-push \\"
echo "      --project=${PROJECT_ID} \\"
echo "      --branch-pattern='^${GITHUB_BRANCH}\$'"
echo ""

echo "==================================================================="
echo " Infrastructure ready."
echo ""
echo " Next steps:"
echo "   1. Fill in the nailong-env secret with your real .env contents"
echo "      (see URL printed above)."
echo "   2. Connect GitHub repo in Cloud Build console (see URL above)."
echo "   3. Run the gcloud builds triggers create command above."
echo "   4. Push a commit to ${GITHUB_BRANCH} — Cloud Build will build + deploy."
echo "==================================================================="
