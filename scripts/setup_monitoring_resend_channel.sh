#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:-nrsi-web-prod-20260524}"
RELAY_URL="${2:-}"
CHANNEL_DISPLAY_NAME="${3:-nrsi-resend-alerts-webhook}"
ALERT_WEBHOOK_TOKEN="${4:-${ALERT_WEBHOOK_TOKEN:-}}"

if [[ -z "${RELAY_URL}" ]]; then
  echo "Usage: $0 <project-id> <relay-url> [channel-display-name]"
  echo "Example relay-url: https://nrsi-alert-relay-xxxx-uc.a.run.app/monitoring-alert"
  exit 1
fi
if [[ -z "${ALERT_WEBHOOK_TOKEN}" ]]; then
  echo "Missing ALERT_WEBHOOK_TOKEN (arg4 or env var)."
  exit 1
fi

RELAY_HOOK_URL="${RELAY_URL%/}/${ALERT_WEBHOOK_TOKEN}"

gcloud services enable monitoring.googleapis.com --project "${PROJECT_ID}" >/dev/null

CHANNEL_NAME="$(gcloud alpha monitoring channels list --project "${PROJECT_ID}" --format='value(name)' --filter="displayName=\"${CHANNEL_DISPLAY_NAME}\"" | head -n 1)"
if [[ -z "${CHANNEL_NAME}" ]]; then
  gcloud alpha monitoring channels create \
    --project "${PROJECT_ID}" \
    --display-name "${CHANNEL_DISPLAY_NAME}" \
    --description "Routes Cloud Monitoring incidents through Cloud Run relay to Resend." \
    --type webhook_tokenauth \
    --channel-labels "url=${RELAY_HOOK_URL}" >/dev/null
  CHANNEL_NAME="$(gcloud alpha monitoring channels list --project "${PROJECT_ID}" --format='value(name)' --filter="displayName=\"${CHANNEL_DISPLAY_NAME}\"" | head -n 1)"
fi

if [[ -z "${CHANNEL_NAME}" ]]; then
  echo "Failed to create or resolve notification channel."
  exit 1
fi

CHANNEL_NAME="${CHANNEL_NAME}" PROJECT_ID="${PROJECT_ID}" python3 - <<'PY'
import json
import os
import subprocess
import tempfile

project = os.environ["PROJECT_ID"]
channel = os.environ["CHANNEL_NAME"]
target_policies = {
    "nrsi-ai-uptime-alert",
    "nrsi-site-5xx-alert",
    "nrsi-site-latency-alert",
}

raw = subprocess.check_output(
    ["gcloud", "monitoring", "policies", "list", "--project", project, "--format=json"],
    text=True,
)
policies = json.loads(raw)

for policy in policies:
    if policy.get("displayName") not in target_policies:
        continue
    channels = policy.get("notificationChannels", [])
    if channel in channels:
        continue
    channels.append(channel)
    policy["notificationChannels"] = channels
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
        json.dump(policy, f)
        path = f.name
    try:
        subprocess.check_call(
            ["gcloud", "monitoring", "policies", "update", policy["name"], "--project", project, "--policy-from-file", path]
        )
    finally:
        os.unlink(path)
PY

echo "Notification channel attached to alert policies: ${CHANNEL_NAME}"
