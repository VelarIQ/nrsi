#!/usr/bin/env bash
set -euo pipefail

# Deploy the Next.js NRSI website to Cloud Run.
# Usage:
#   ./scripts/deploy_gcp.sh [PROJECT_ID] [REGION] [SERVICE_NAME]
#
# Defaults:
#   PROJECT_ID=nrsi-web-prod-20260524
#   REGION=us-central1
#   SERVICE_NAME=nrsi-site

PROJECT_ID="${1:-nrsi-web-prod-20260524}"
REGION="${2:-us-central1}"
SERVICE_NAME="${3:-nrsi-site}"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:$(date +%Y%m%d-%H%M%S)"
DEPLOY_FLAGS=()

echo "Deploying ${SERVICE_NAME} to project=${PROJECT_ID}, region=${REGION}"

# Core APIs should be pre-enabled during project bootstrap.
# This keeps the deploy identity least-privilege (no serviceusage.admin needed).

BUILD_ID="$(
  gcloud builds submit \
    --project "${PROJECT_ID}" \
    --tag "${IMAGE}" \
    --async \
    --format='value(id)' \
    web-next
)"

if [[ -z "${BUILD_ID}" ]]; then
  echo "Failed to create Cloud Build job." >&2
  exit 1
fi

echo "Cloud Build started: ${BUILD_ID}"
while true; do
  BUILD_STATUS="$(gcloud builds describe "${BUILD_ID}" --project "${PROJECT_ID}" --format='value(status)')"
  case "${BUILD_STATUS}" in
    SUCCESS)
      break
      ;;
    QUEUED|WORKING|PENDING)
      sleep 5
      ;;
    *)
      echo "Cloud Build failed: ${BUILD_STATUS}" >&2
      gcloud builds describe "${BUILD_ID}" --project "${PROJECT_ID}" --format='value(logUrl)' >&2 || true
      exit 1
      ;;
  esac
done

ENV_VARS=()
if [[ -n "${NEXT_PUBLIC_TURNSTILE_SITE_KEY:-}" ]]; then
  ENV_VARS+=("NEXT_PUBLIC_TURNSTILE_SITE_KEY=${NEXT_PUBLIC_TURNSTILE_SITE_KEY}")
fi
if [[ -n "${TURNSTILE_SECRET_KEY:-}" ]]; then
  ENV_VARS+=("TURNSTILE_SECRET_KEY=${TURNSTILE_SECRET_KEY}")
fi
if [[ -n "${LAUNCH_INTEREST_WEBHOOK_URL:-}" ]]; then
  ENV_VARS+=("LAUNCH_INTEREST_WEBHOOK_URL=${LAUNCH_INTEREST_WEBHOOK_URL}")
fi
if [[ -n "${LAUNCH_INTEREST_WEBHOOK_TOKEN:-}" ]]; then
  ENV_VARS+=("LAUNCH_INTEREST_WEBHOOK_TOKEN=${LAUNCH_INTEREST_WEBHOOK_TOKEN}")
fi
if [[ -n "${LAUNCH_INTEREST_WEBHOOK_TIMEOUT_MS:-}" ]]; then
  ENV_VARS+=("LAUNCH_INTEREST_WEBHOOK_TIMEOUT_MS=${LAUNCH_INTEREST_WEBHOOK_TIMEOUT_MS}")
fi
if [[ -n "${ANALYTICS_WEBHOOK_URL:-}" ]]; then
  ENV_VARS+=("ANALYTICS_WEBHOOK_URL=${ANALYTICS_WEBHOOK_URL}")
fi
if [[ -n "${ANALYTICS_WEBHOOK_TOKEN:-}" ]]; then
  ENV_VARS+=("ANALYTICS_WEBHOOK_TOKEN=${ANALYTICS_WEBHOOK_TOKEN}")
fi
if [[ -n "${ANALYTICS_WEBHOOK_TIMEOUT_MS:-}" ]]; then
  ENV_VARS+=("ANALYTICS_WEBHOOK_TIMEOUT_MS=${ANALYTICS_WEBHOOK_TIMEOUT_MS}")
fi
if [[ -n "${LAUNCH_INTEREST_STAGING_WEBHOOK_URL:-}" ]]; then
  ENV_VARS+=("LAUNCH_INTEREST_STAGING_WEBHOOK_URL=${LAUNCH_INTEREST_STAGING_WEBHOOK_URL}")
fi
if [[ -n "${ANALYTICS_STAGING_WEBHOOK_URL:-}" ]]; then
  ENV_VARS+=("ANALYTICS_STAGING_WEBHOOK_URL=${ANALYTICS_STAGING_WEBHOOK_URL}")
fi
if [[ -n "${WEBHOOK_CANARY_PERCENT:-}" ]]; then
  ENV_VARS+=("WEBHOOK_CANARY_PERCENT=${WEBHOOK_CANARY_PERCENT}")
fi
if [[ -n "${EVENT_WEBHOOK_SIGNING_SECRET:-}" ]]; then
  ENV_VARS+=("EVENT_WEBHOOK_SIGNING_SECRET=${EVENT_WEBHOOK_SIGNING_SECRET}")
fi
if [[ -n "${GOOGLE_OAUTH_CLIENT_ID:-}" ]]; then
  ENV_VARS+=("GOOGLE_OAUTH_CLIENT_ID=${GOOGLE_OAUTH_CLIENT_ID}")
fi
if [[ -n "${NEXT_PUBLIC_GOOGLE_OAUTH_CLIENT_ID:-}" ]]; then
  ENV_VARS+=("NEXT_PUBLIC_GOOGLE_OAUTH_CLIENT_ID=${NEXT_PUBLIC_GOOGLE_OAUTH_CLIENT_ID}")
fi
if [[ -n "${AUTH_ALLOWED_DOMAINS:-}" ]]; then
  ENV_VARS+=("AUTH_ALLOWED_DOMAINS=${AUTH_ALLOWED_DOMAINS}")
fi
if [[ -n "${AUTH_SUPER_ADMIN_EMAILS:-}" ]]; then
  ENV_VARS+=("AUTH_SUPER_ADMIN_EMAILS=${AUTH_SUPER_ADMIN_EMAILS}")
fi
if [[ -n "${AUTH_ADMIN_EMAILS:-}" ]]; then
  ENV_VARS+=("AUTH_ADMIN_EMAILS=${AUTH_ADMIN_EMAILS}")
fi
if [[ -n "${AUTH_SESSION_SECRET:-}" ]]; then
  ENV_VARS+=("AUTH_SESSION_SECRET=${AUTH_SESSION_SECRET}")
fi
if [[ -n "${AUTH_SESSION_TTL_SECONDS:-}" ]]; then
  ENV_VARS+=("AUTH_SESSION_TTL_SECONDS=${AUTH_SESSION_TTL_SECONDS}")
fi
if [[ ${#ENV_VARS[@]} -gt 0 ]]; then
  echo "Applying runtime env vars."
  gcloud run deploy "${SERVICE_NAME}" \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --image "${IMAGE}" \
    --port 8080 \
    --allow-unauthenticated \
    --update-env-vars "^|^$(IFS='|'; echo "${ENV_VARS[*]}")"
else
  gcloud run deploy "${SERVICE_NAME}" \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --image "${IMAGE}" \
    --port 8080 \
    --allow-unauthenticated
fi

SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --format='value(status.url)')"

echo "Deployment complete:"
echo "  Cloud Run URL: ${SERVICE_URL}"
echo ""
echo "Next:"
echo "  1) Cloud Run deploy workflow: .github/workflows/deploy-cloud-run.yml"
echo "  2) Map custom domain with:"
echo "     gcloud beta run domain-mappings create --project ${PROJECT_ID} --service ${SERVICE_NAME} --domain nrsi.ai --region ${REGION}"
