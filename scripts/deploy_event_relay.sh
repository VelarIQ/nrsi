#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:-nrsi-web-prod-20260524}"
REGION="${2:-us-central1}"
SERVICE_NAME="${3:-nrsi-event-relay}"
EVENT_TOKEN="${EVENT_WEBHOOK_TOKEN:-}"
SIGNING_SECRET="${EVENT_WEBHOOK_SIGNING_SECRET:-}"
BIGQUERY_DATASET="${BIGQUERY_DATASET:-nrsi_events}"
BIGQUERY_TABLE="${BIGQUERY_TABLE:-product_events}"
STORE_PLAIN_EMAIL="${STORE_PLAIN_EMAIL:-false}"
EVENT_SIGNATURE_MAX_SKEW_SECONDS="${EVENT_SIGNATURE_MAX_SKEW_SECONDS:-300}"
RUNTIME_SERVICE_ACCOUNT="${RUNTIME_SERVICE_ACCOUNT:-nrsi-event-relay-runtime@${PROJECT_ID}.iam.gserviceaccount.com}"

if [[ -z "${EVENT_TOKEN}" ]]; then
  echo "EVENT_WEBHOOK_TOKEN must be set." >&2
  exit 1
fi
if [[ -z "${SIGNING_SECRET}" ]]; then
  echo "EVENT_WEBHOOK_SIGNING_SECRET must be set." >&2
  exit 1
fi

IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:$(date +%Y%m%d-%H%M%S)"

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --project "${PROJECT_ID}" >/dev/null
gcloud builds submit --project "${PROJECT_ID}" --tag "${IMAGE}" event-relay

ENV_VARS=(
  "EVENT_WEBHOOK_TOKEN=${EVENT_TOKEN}"
  "EVENT_WEBHOOK_SIGNING_SECRET=${SIGNING_SECRET}"
  "BIGQUERY_PROJECT=${PROJECT_ID}"
  "BIGQUERY_DATASET=${BIGQUERY_DATASET}"
  "BIGQUERY_TABLE=${BIGQUERY_TABLE}"
  "STORE_PLAIN_EMAIL=${STORE_PLAIN_EMAIL}"
  "EVENT_SIGNATURE_MAX_SKEW_SECONDS=${EVENT_SIGNATURE_MAX_SKEW_SECONDS}"
)

gcloud run deploy "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --image "${IMAGE}" \
  --allow-unauthenticated \
  --service-account "${RUNTIME_SERVICE_ACCOUNT}" \
  --set-env-vars "$(IFS=,; echo "${ENV_VARS[*]}")"

URL="$(gcloud run services describe "${SERVICE_NAME}" --project "${PROJECT_ID}" --region "${REGION}" --format='value(status.url)')"
echo "Event relay deployed: ${URL}"
