#!/usr/bin/env bash
# =============================================================================
# deploy/setup.sh — Run this ONCE in your WSL/Linux terminal.
# Handles gcloud install (if missing), login, GCP provisioning, first deploy.
#
# Usage:
#   bash /mnt/c/porper_code_projs/NailongUIC-ProphetHacks/deploy/setup.sh
# =============================================================================
set -eu

PROJECT_ID="project-c31af50d-0e9f-45fd-a5c"
REPO_ROOT="/mnt/c/porper_code_projs/NailongUIC-ProphetHacks"

# ── Find or install gcloud ────────────────────────────────────────────────────
if command -v gcloud >/dev/null 2>&1; then
    GCLOUD="$(command -v gcloud)"
elif [ -f "$HOME/google-cloud-sdk/bin/gcloud" ]; then
    GCLOUD="$HOME/google-cloud-sdk/bin/gcloud"
else
    echo "==> gcloud not found. Installing to ~/google-cloud-sdk ..."
    TARBALL="$HOME/google-cloud-cli-linux-x86_64.tar.gz"
    # Reuse the already-downloaded tarball if it exists in the project dir
    if [ -f "$REPO_ROOT/google-cloud-cli-linux-x86_64.tar.gz" ]; then
        cp "$REPO_ROOT/google-cloud-cli-linux-x86_64.tar.gz" "$TARBALL"
    else
        curl -fsSL -o "$TARBALL" \
            "https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz"
    fi
    tar -xf "$TARBALL" -C "$HOME"
    bash "$HOME/google-cloud-sdk/install.sh" --quiet --path-update=true
    GCLOUD="$HOME/google-cloud-sdk/bin/gcloud"
    echo "    gcloud installed at $GCLOUD"
fi

echo "==> Using gcloud: $GCLOUD  (version: $("$GCLOUD" version | head -1))"

# ── Authenticate ──────────────────────────────────────────────────────────────
echo ""
echo "==> Step 1/3: Google login"
echo "    A browser window will open. Sign in with the Google account"
echo "    that owns project '${PROJECT_ID}', then come back here."
echo ""
"$GCLOUD" auth login
"$GCLOUD" config set project "$PROJECT_ID"
echo "    Logged in as: $("$GCLOUD" config get-value account)"

# ── Provision GCP infrastructure ──────────────────────────────────────────────
echo ""
echo "==> Step 2/3: Provisioning GCP infrastructure..."
echo ""
chmod +x "$REPO_ROOT/deploy/create-vm.sh" "$REPO_ROOT/deploy/vm-deploy.sh"
bash "$REPO_ROOT/deploy/create-vm.sh"

# ── Final instructions ────────────────────────────────────────────────────────
echo ""
echo "==> Step 3/3: Two manual steps remaining (GCP Console UI):"
echo ""
echo "  A) Fill in your API keys as the 'nailong-env' secret:"
echo "     https://console.cloud.google.com/security/secret-manager?project=${PROJECT_ID}"
echo "     -> 'nailong-env' -> 'Add New Version' -> paste your .env contents"
echo ""
echo "  B) Connect GitHub to Cloud Build:"
echo "     https://console.cloud.google.com/cloud-build/triggers?project=${PROJECT_ID}"
echo "     -> 'Connect Repository' -> 'GitHub (Cloud Build GitHub App)'"
echo "     -> Authorise and select 'FernandoLpz0911/NailongUIC-ProphetHacks'"
echo "     -> Then run the gcloud builds triggers create command shown above"
echo ""
echo "Once done, push any commit to main to trigger an automatic build+deploy."
