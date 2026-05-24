#!/usr/bin/env bash
set -euo pipefail

# Deploy the static NRSI website to Cloud Run.
# Usage:
#   ./scripts/deploy_gcp.sh [PROJECT_ID] [REGION] [SERVICE_NAME]
#
# Defaults:
#   PROJECT_ID from current gcloud config
#   REGION=us-central1
#   SERVICE_NAME=nrsi-site

PROJECT_ID="${1:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${2:-us-central1}"
SERVICE_NAME="${3:-nrsi-site}"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:$(date +%Y%m%d-%H%M%S)"

if [ -z "${PROJECT_ID}" ] || [ "${PROJECT_ID}" = "(unset)" ]; then
  echo "No PROJECT_ID set. Pass one as arg1 or run: gcloud config set project <id>"
  exit 1
fi

echo "Deploying ${SERVICE_NAME} to project=${PROJECT_ID}, region=${REGION}"

gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  --project "${PROJECT_ID}"

gcloud builds submit \
  --project "${PROJECT_ID}" \
  --tag "${IMAGE}" \
  website

gcloud run deploy "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --image "${IMAGE}" \
  --port 8080 \
  --allow-unauthenticated

SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --format='value(status.url)')"

echo "Deployment complete:"
echo "  Cloud Run URL: ${SERVICE_URL}"
echo ""
echo "Next:"
echo "  1) Keep GitHub Pages as fallback or disable it after cutover."
echo "  2) Map custom domain with:"
echo "     gcloud beta run domain-mappings create --project ${PROJECT_ID} --service ${SERVICE_NAME} --domain nrsi.ai --region ${REGION}"
