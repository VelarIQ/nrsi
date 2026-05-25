#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:-nrsi-web-prod-20260524}"
REGION="${2:-us-central1}"
SERVICE_NAME="${3:-nrsi-alert-relay}"

if [[ -z "${RESEND_API_KEY:-}" ]]; then
  echo "Missing RESEND_API_KEY environment variable."
  exit 1
fi
if [[ -z "${RESEND_FROM:-}" ]]; then
  echo "Missing RESEND_FROM environment variable (example: 'NRSI Alerts <alerts@yourdomain.com>')."
  exit 1
fi
if [[ -z "${ALERT_RECIPIENTS:-}" ]]; then
  echo "Missing ALERT_RECIPIENTS environment variable (comma-separated emails)."
  exit 1
fi
if [[ -z "${ALERT_WEBHOOK_TOKEN:-}" ]]; then
  echo "Missing ALERT_WEBHOOK_TOKEN environment variable."
  exit 1
fi

IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:$(date +%Y%m%d-%H%M%S)"

echo "Deploying ${SERVICE_NAME} in ${PROJECT_ID}/${REGION}"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --project "${PROJECT_ID}" >/dev/null

gcloud builds submit \
  --project "${PROJECT_ID}" \
  --tag "${IMAGE}" \
  alert-relay

gcloud run deploy "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --image "${IMAGE}" \
  --port 8080 \
  --allow-unauthenticated \
  --set-env-vars "RESEND_API_KEY=${RESEND_API_KEY},RESEND_FROM=${RESEND_FROM},ALERT_RECIPIENTS=${ALERT_RECIPIENTS},ALERT_WEBHOOK_TOKEN=${ALERT_WEBHOOK_TOKEN}"

SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" --project "${PROJECT_ID}" --region "${REGION}" --format='value(status.url)')"
echo "Relay deployed: ${SERVICE_URL}/monitoring-alert/${ALERT_WEBHOOK_TOKEN}"
