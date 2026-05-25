#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:-nrsi-web-prod-20260524}"
SA_NAME="${2:-nrsi-event-relay-runtime}"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create "${SA_NAME}" \
  --project "${PROJECT_ID}" \
  --display-name "NRSI Event Relay Runtime" >/dev/null 2>&1 || true

for _ in 1 2 3 4 5; do
  if gcloud iam service-accounts describe "${SA_EMAIL}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member "serviceAccount:${SA_EMAIL}" \
  --role "roles/bigquery.dataEditor" \
  --condition=None >/dev/null

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member "serviceAccount:${SA_EMAIL}" \
  --role "roles/bigquery.jobUser" \
  --condition=None >/dev/null

echo "Event relay runtime service account ready: ${SA_EMAIL}"
