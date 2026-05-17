# GCP Deployment Guide

Project: `project-c31af50d-0e9f-45fd-a5c`  
Region: `us-central1` (zone `us-central1-a`)  
VM: `nailong-agent` (`e2-medium`, no public IP — accessed via IAP)  
Image registry: `us-central1-docker.pkg.dev/project-c31af50d-0e9f-45fd-a5c/nailong/agent`

---

## Architecture

```
GitHub push to default branch (refactor-connection)
        │
        ▼
  Cloud Build trigger
        │
  ┌─────┴──────────────────────────────────────────┐
  │  cloudbuild.yaml                               │
  │  1. lint  (Ruff)                               │
  │  2. test  (pytest – packages + agent/tests)    │
  │  3. docker build                               │
  │  4. docker push → Artifact Registry            │
  │  5. gcloud compute ssh → vm-deploy.sh          │
  └────────────────────────────────────────────────┘
        │
        ▼ (IAP tunnel, no public IP needed)
  Compute Engine VM  nailong-agent
        │
        ├── pulls secret  nailong-env  from Secret Manager
        ├── docker pull  <new image>
        └── docker run   nailong-agent container
                          └── /app/data  →  /opt/nailong/data (persistent volume)
```

---

> **Default branch note** — this repo's default branch is `refactor-connection`,
> not `main`.  All trigger references below use that branch; change `--branch-pattern`
> if the default ever moves.

## One-time setup (run once, then CI/CD takes over)

### Prerequisites

```bash
# Install gcloud CLI: https://cloud.google.com/sdk/docs/install
gcloud auth login
gcloud config set project project-c31af50d-0e9f-45fd-a5c
# Confirm billing is enabled on the project in the GCP Console
```

### Step 1 — Run the provisioning script

From the repo root (Linux/macOS/WSL):

```bash
chmod +x deploy/create-vm.sh deploy/vm-deploy.sh
bash deploy/create-vm.sh
```

On Windows PowerShell (run from WSL or Git Bash — the script uses bash):

```powershell
# Open Git Bash or WSL, cd to repo root, then:
bash deploy/create-vm.sh
```

The script creates:
- Artifact Registry repo `nailong`
- Service account `nailong-sa` with Secret Manager + Artifact Registry read access
- Compute Engine VM `nailong-agent` (e2-medium, Debian 12, no public IP)
- IAP firewall rule so Cloud Build can SSH in
- Empty Secret Manager secret `nailong-env`

### Step 2 — Fill in the secret

1. Open [Secret Manager](https://console.cloud.google.com/security/secret-manager?project=project-c31af50d-0e9f-45fd-a5c)
2. Click `nailong-env` → **Actions** → **Add New Version**
3. Paste the full contents of your `.env` file (with real API keys) into the "Secret value" box
4. Click **Add New Version**

The secret is the entire `.env` file, for example:

```
PA_SERVER_URL=https://api.aiprophet.dev
PA_SERVER_API_KEY=prophet_REAL_KEY_HERE
GEMINI_API_KEY=AIza...
OPENROUTER_API_KEY=sk-or-...
TAVILY_API_KEY=tvly-...
COST_DB_PATH=/app/data/costs.sqlite
KILL_SWITCH_USD=100
LLM_ENSEMBLE_N=6
...
```

**Never commit real keys to git.** The `.env` file is gitignored.

### Step 3 — Connect GitHub to Cloud Build

This step requires the GCP Console (cannot be fully scripted):

1. Open [Cloud Build Triggers](https://console.cloud.google.com/cloud-build/triggers?project=project-c31af50d-0e9f-45fd-a5c)
2. Click **Connect Repository**
3. Select **GitHub (Cloud Build GitHub App)**
4. Authorize the app and select `FernandoLpz0911/NailongUIC-ProphetHacks`
5. Click **Connect**

### Step 4 — Create the Cloud Build trigger

After connecting the repo, run:

```bash
gcloud builds triggers create github \
  --project=project-c31af50d-0e9f-45fd-a5c \
  --repo-name=NailongUIC-ProphetHacks \
  --repo-owner=FernandoLpz0911 \
  --branch-pattern='^refactor-connection$' \
  --build-config=cloudbuild.yaml \
  --name=nailong-deploy-on-push \
  --description='Lint, test, build, and deploy on push to default branch'
```

If a trigger already exists pointing at the wrong branch (e.g. `main`), update
it in place instead of recreating it:

```bash
gcloud builds triggers update github nailong-deploy-on-push \
  --project=project-c31af50d-0e9f-45fd-a5c \
  --branch-pattern='^refactor-connection$'
```

### Step 5 — First deploy

Push any commit to the default branch (`refactor-connection`) to trigger the
first build:

```bash
git add .
git commit -m "chore: add GCP deployment config"
git push origin refactor-connection
```

Watch the build at [Cloud Build History](https://console.cloud.google.com/cloud-build/builds?project=project-c31af50d-0e9f-45fd-a5c).

---

## After initial deployment

### Check the agent is running

```bash
gcloud compute ssh nailong-agent \
  --zone=us-central1-a \
  --tunnel-through-iap \
  --command="docker ps && docker logs --tail 40 nailong-agent"
```

### Tail live logs

```bash
gcloud compute ssh nailong-agent \
  --zone=us-central1-a \
  --tunnel-through-iap \
  --command="docker logs -f nailong-agent"
```

### Or stream from Cloud Logging (no SSH needed)

```bash
gcloud logging read \
  'resource.type="gce_instance" AND resource.labels.instance_id="nailong-agent"' \
  --project=project-c31af50d-0e9f-45fd-a5c \
  --limit=50 \
  --format='value(textPayload)'
```

### Restart the agent without a code deploy

```bash
gcloud compute ssh nailong-agent \
  --zone=us-central1-a \
  --tunnel-through-iap \
  --command="docker restart nailong-agent"
```

### Update the .env (e.g. rotate an API key)

1. Update the secret in Secret Manager (add a new version)
2. Redeploy by pushing a commit or manually triggering Cloud Build

---

## Cost estimate

| Resource | Size | Est. monthly |
|---|---|---|
| Compute Engine e2-medium | 24/7 for 14 days | ~$6 (14 days) |
| Artifact Registry storage | <1 GB | ~$0.10 |
| Cloud Build | ~5 min/build × N builds | ~$0 (free tier covers it) |
| Secret Manager | 1 secret | ~$0 |
| Cloud Logging | low volume | ~$0 |

The VM costs roughly **$0.04/hour**. Total GCP infra cost for the 14-day eval window is under **$10** excluding your LLM/search API costs.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Cloud Build fails at `upload-deploy-script` with "Permission denied" | Run: `gcloud projects add-iam-policy-binding PROJECT_ID --member="serviceAccount:PROJECT_NUMBER@cloudbuild.gserviceaccount.com" --role="roles/compute.osAdminLogin"` |
| Cloud Build SSH/SCP fails with `Required 'iam.serviceAccounts.actAs' permission` | The Cloud Build SA needs `serviceAccountUser` on the VM SA. Run: `gcloud iam service-accounts add-iam-policy-binding nailong-sa@PROJECT_ID.iam.gserviceaccount.com --member="serviceAccount:PROJECT_NUMBER@cloudbuild.gserviceaccount.com" --role="roles/iam.serviceAccountUser"` |
| Build fails immediately with `logging must be set to GCS_ONLY` or `bucket required` | Cloud Logging-only mode needs `logWriter`. Run: `gcloud projects add-iam-policy-binding PROJECT_ID --member="serviceAccount:PROJECT_NUMBER@cloudbuild.gserviceaccount.com" --role="roles/logging.logWriter"` |
| Push to default branch doesn't fire the Cloud Build trigger | The trigger's branch-pattern probably still says `^main$`. Update it: `gcloud builds triggers update github nailong-deploy-on-push --branch-pattern='^refactor-connection$'` |
| Container exits immediately | Check logs: `docker logs nailong-agent`. Usually a missing env var — verify the `nailong-env` secret has the correct `.env` contents |
| `gcloud compute ssh` hangs | IAP firewall rule may be missing — re-run `create-vm.sh` to recreate it |
| Agent stops after first tick | Check `costs.sqlite` inside the container: `docker exec nailong-agent sqlite3 /app/data/costs.sqlite "SELECT SUM(usd_cost) FROM costs"`. Kill switch (`KILL_SWITCH_USD`) may have tripped |
