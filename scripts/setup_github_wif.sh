#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:-nrsi-web-prod-20260524}"
REPO="${2:-VelarIQ/nrsi}"
POOL_ID="${3:-github-actions-pool}"
PROVIDER_ID="${4:-nrsi-repo-provider}"
SA_NAME="${5:-nrsi-github-deployer}"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"

echo "Configuring GitHub OIDC deploy identity:"
echo "  project=${PROJECT_ID}"
echo "  repo=${REPO}"
echo "  service-account=${SA_EMAIL}"

gcloud iam service-accounts describe "${SA_EMAIL}" --project "${PROJECT_ID}" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "${SA_NAME}" \
    --project "${PROJECT_ID}" \
    --display-name="NRSI GitHub Cloud Run Deployer"

for role in \
  roles/run.admin \
  roles/iam.serviceAccountUser \
  roles/cloudbuild.builds.editor \
  roles/artifactregistry.writer \
  roles/storage.admin \
  roles/serviceusage.serviceUsageConsumer; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${role}" \
    --condition=None \
    --quiet >/dev/null
done

gcloud iam workload-identity-pools describe "${POOL_ID}" \
  --project "${PROJECT_ID}" \
  --location global >/dev/null 2>&1 || \
  gcloud iam workload-identity-pools create "${POOL_ID}" \
    --project "${PROJECT_ID}" \
    --location global \
    --display-name "GitHub Actions Pool"

gcloud iam workload-identity-pools providers describe "${PROVIDER_ID}" \
  --project "${PROJECT_ID}" \
  --location global \
  --workload-identity-pool "${POOL_ID}" >/dev/null 2>&1 || \
  gcloud iam workload-identity-pools providers create-oidc "${PROVIDER_ID}" \
    --project "${PROJECT_ID}" \
    --location global \
    --workload-identity-pool "${POOL_ID}" \
    --display-name "VelarIQ nrsi GitHub Provider" \
    --issuer-uri "https://token.actions.githubusercontent.com" \
    --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.actor=assertion.actor,attribute.ref=assertion.ref" \
    --attribute-condition "assertion.repository=='${REPO}'"

gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
  --project "${PROJECT_ID}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/attribute.repository/${REPO}" \
  --condition=None \
  --quiet >/dev/null

echo ""
echo "GitHub workflow settings:"
echo "  workload_identity_provider=projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/providers/${PROVIDER_ID}"
echo "  service_account=${SA_EMAIL}"
