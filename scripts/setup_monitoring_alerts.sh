#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:-nrsi-web-prod-20260524}"
DOMAIN="${2:-nrsi.ai}"
SERVICE_NAME="${3:-nrsi-site}"
EVENT_SERVICE_NAME="${4:-nrsi-event-relay}"
export PROJECT_ID SERVICE_NAME
export EVENT_SERVICE_NAME

gcloud services enable monitoring.googleapis.com --project "${PROJECT_ID}" >/dev/null
gcloud services enable logging.googleapis.com --project "${PROJECT_ID}" >/dev/null

if ! gcloud monitoring uptime list-configs --project "${PROJECT_ID}" --format='value(displayName)' | awk -v n="nrsi-ai-https" 'BEGIN{f=1} $0==n{f=0} END{exit f}'; then
  gcloud monitoring uptime create nrsi-ai-https \
    --project "${PROJECT_ID}" \
    --resource-type uptime-url \
    --resource-labels "project_id=${PROJECT_ID},host=${DOMAIN}" \
    --protocol https \
    --path / \
    --period 1 \
    --timeout 10 \
    --regions usa-iowa,usa-virginia,europe \
    --validate-ssl=true >/dev/null
fi

WAITLIST_FILTER="resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${SERVICE_NAME}\" AND (textPayload:\"waitlist_rate_limited\" OR textPayload:\"waitlist_honeypot_triggered\" OR textPayload:\"waitlist_bot_like_user_agent\" OR textPayload:\"waitlist_turnstile_failed\")"
if gcloud logging metrics describe nrsi_waitlist_rate_limited_count --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud logging metrics update nrsi_waitlist_rate_limited_count \
    --project "${PROJECT_ID}" \
    --description="Count waitlist rate limit and bot blocks" \
    --log-filter="${WAITLIST_FILTER}" >/dev/null
else
  gcloud logging metrics create nrsi_waitlist_rate_limited_count \
    --project "${PROJECT_ID}" \
    --description="Count waitlist rate limit and bot blocks" \
    --log-filter="${WAITLIST_FILTER}" >/dev/null
fi

ERROR_FILTER="resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${SERVICE_NAME}\" AND (textPayload:\"launch_interest_failed\" OR textPayload:\"launch_analytics_failed\")"
if gcloud logging metrics describe nrsi_launch_api_error_count --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud logging metrics update nrsi_launch_api_error_count \
    --project "${PROJECT_ID}" \
    --description="Count launch API server errors" \
    --log-filter="${ERROR_FILTER}" >/dev/null
else
  gcloud logging metrics create nrsi_launch_api_error_count \
    --project "${PROJECT_ID}" \
    --description="Count launch API server errors" \
    --log-filter="${ERROR_FILTER}" >/dev/null
fi

RELAY_AUTH_FILTER="resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${EVENT_SERVICE_NAME}\" AND textPayload:\"product_event_signature_failed\""
if gcloud logging metrics describe nrsi_event_relay_auth_fail_count --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud logging metrics update nrsi_event_relay_auth_fail_count \
    --project "${PROJECT_ID}" \
    --description="Count product event relay signature/auth failures" \
    --log-filter="${RELAY_AUTH_FILTER}" >/dev/null
else
  gcloud logging metrics create nrsi_event_relay_auth_fail_count \
    --project "${PROJECT_ID}" \
    --description="Count product event relay signature/auth failures" \
    --log-filter="${RELAY_AUTH_FILTER}" >/dev/null
fi

RELAY_INGEST_FILTER="resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${EVENT_SERVICE_NAME}\" AND textPayload:\"product_event_received\""
if gcloud logging metrics describe nrsi_event_relay_ingest_count --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud logging metrics update nrsi_event_relay_ingest_count \
    --project "${PROJECT_ID}" \
    --description="Count product event relay ingested events" \
    --log-filter="${RELAY_INGEST_FILTER}" >/dev/null
else
  gcloud logging metrics create nrsi_event_relay_ingest_count \
    --project "${PROJECT_ID}" \
    --description="Count product event relay ingested events" \
    --log-filter="${RELAY_INGEST_FILTER}" >/dev/null
fi

python3 - <<'PY'
import json
import os
import subprocess
import tempfile

project = os.environ["PROJECT_ID"]
service = os.environ["SERVICE_NAME"]
event_service = os.environ["EVENT_SERVICE_NAME"]

def run(cmd):
    return subprocess.check_output(cmd, text=True).strip()

def policy_exists(name):
    out = run(["gcloud", "monitoring", "policies", "list", "--project", project, "--format=value(displayName)"])
    return name in {line.strip() for line in out.splitlines() if line.strip()}

uptime_name = run([
    "gcloud", "monitoring", "uptime", "list-configs",
    "--project", project,
    "--filter", "displayName=nrsi-ai-https",
    "--format=value(name)",
])
check_id = uptime_name.split("/")[-1]

policies = [
    {
        "displayName": "nrsi-ai-uptime-alert",
        "documentation": {"content": "nrsi.ai uptime check is failing.", "mimeType": "text/markdown"},
        "conditions": [{
            "displayName": "Uptime check failing",
            "conditionThreshold": {
                "filter": f'metric.type="monitoring.googleapis.com/uptime_check/check_passed" AND resource.type="uptime_url" AND metric.label.check_id="{check_id}"',
                "comparison": "COMPARISON_LT",
                "thresholdValue": 1,
                "duration": "180s",
                "trigger": {"count": 1},
                "aggregations": [{"alignmentPeriod": "60s", "perSeriesAligner": "ALIGN_NEXT_OLDER"}],
            },
        }],
    },
    {
        "displayName": "nrsi-site-5xx-alert",
        "documentation": {"content": "Cloud Run nrsi-site is emitting HTTP 5xx responses.", "mimeType": "text/markdown"},
        "conditions": [{
            "displayName": "Cloud Run 5xx requests > 0",
            "conditionThreshold": {
                "filter": f'metric.type="run.googleapis.com/request_count" AND resource.type="cloud_run_revision" AND resource.label."service_name"="{service}" AND metric.label."response_code_class"="5xx"',
                "comparison": "COMPARISON_GT",
                "thresholdValue": 0,
                "duration": "300s",
                "trigger": {"count": 1},
                "aggregations": [{"alignmentPeriod": "60s", "perSeriesAligner": "ALIGN_RATE"}],
            },
        }],
    },
    {
        "displayName": "nrsi-site-latency-alert",
        "documentation": {"content": "Cloud Run nrsi-site p95 latency exceeds 2 seconds.", "mimeType": "text/markdown"},
        "conditions": [{
            "displayName": "Cloud Run p95 latency > 2s",
            "conditionThreshold": {
                "filter": f'metric.type="run.googleapis.com/request_latencies" AND resource.type="cloud_run_revision" AND resource.label."service_name"="{service}"',
                "comparison": "COMPARISON_GT",
                "thresholdValue": 2000,
                "duration": "300s",
                "trigger": {"count": 1},
                "aggregations": [{"alignmentPeriod": "60s", "perSeriesAligner": "ALIGN_PERCENTILE_95"}],
            },
        }],
    },
    {
        "displayName": "nrsi-waitlist-abuse-alert",
        "documentation": {"content": "Waitlist endpoint is experiencing bot or rate-limit pressure.", "mimeType": "text/markdown"},
        "conditions": [{
            "displayName": "Waitlist abuse events > 5/min",
            "conditionThreshold": {
                "filter": f'metric.type="logging.googleapis.com/user/nrsi_waitlist_rate_limited_count" AND resource.type="cloud_run_revision" AND resource.label."service_name"="{service}"',
                "comparison": "COMPARISON_GT",
                "thresholdValue": 5,
                "duration": "60s",
                "trigger": {"count": 1},
                "aggregations": [{"alignmentPeriod": "60s", "perSeriesAligner": "ALIGN_DELTA"}],
            },
        }],
    },
    {
        "displayName": "nrsi-launch-api-error-alert",
        "documentation": {"content": "Launch API is emitting backend errors.", "mimeType": "text/markdown"},
        "conditions": [{
            "displayName": "Launch API error events > 0",
            "conditionThreshold": {
                "filter": f'metric.type="logging.googleapis.com/user/nrsi_launch_api_error_count" AND resource.type="cloud_run_revision" AND resource.label."service_name"="{service}"',
                "comparison": "COMPARISON_GT",
                "thresholdValue": 0,
                "duration": "120s",
                "trigger": {"count": 1},
                "aggregations": [{"alignmentPeriod": "60s", "perSeriesAligner": "ALIGN_DELTA"}],
            },
        }],
    },
    {
        "displayName": "nrsi-event-relay-5xx-alert",
        "documentation": {"content": "Event relay Cloud Run is emitting HTTP 5xx responses.", "mimeType": "text/markdown"},
        "conditions": [{
            "displayName": "Event relay 5xx requests > 0",
            "conditionThreshold": {
                "filter": f'metric.type="run.googleapis.com/request_count" AND resource.type="cloud_run_revision" AND resource.label."service_name"="{event_service}" AND metric.label."response_code_class"="5xx"',
                "comparison": "COMPARISON_GT",
                "thresholdValue": 0,
                "duration": "300s",
                "trigger": {"count": 1},
                "aggregations": [{"alignmentPeriod": "60s", "perSeriesAligner": "ALIGN_RATE"}],
            },
        }],
    },
    {
        "displayName": "nrsi-event-relay-auth-failure-alert",
        "documentation": {"content": "Event relay auth/signature failures exceeded threshold.", "mimeType": "text/markdown"},
        "conditions": [{
            "displayName": "Relay auth failures > 5/min",
            "conditionThreshold": {
                "filter": f'metric.type="logging.googleapis.com/user/nrsi_event_relay_auth_fail_count" AND resource.type="cloud_run_revision" AND resource.label."service_name"="{event_service}"',
                "comparison": "COMPARISON_GT",
                "thresholdValue": 5,
                "duration": "60s",
                "trigger": {"count": 1},
                "aggregations": [{"alignmentPeriod": "60s", "perSeriesAligner": "ALIGN_DELTA"}],
            },
        }],
    },
    {
        "displayName": "nrsi-event-relay-throughput-drop-alert",
        "documentation": {"content": "Event relay ingestion throughput dropped below expected baseline.", "mimeType": "text/markdown"},
        "conditions": [{
            "displayName": "Relay ingest rate < 1/min for 15m",
            "conditionThreshold": {
                "filter": f'metric.type="logging.googleapis.com/user/nrsi_event_relay_ingest_count" AND resource.type="cloud_run_revision" AND resource.label."service_name"="{event_service}"',
                "comparison": "COMPARISON_LT",
                "thresholdValue": 1,
                "duration": "900s",
                "trigger": {"count": 1},
                "aggregations": [{"alignmentPeriod": "60s", "perSeriesAligner": "ALIGN_DELTA"}],
            },
        }],
    },
]

for item in policies:
    if policy_exists(item["displayName"]):
        continue
    policy = {
        "displayName": item["displayName"],
        "combiner": "OR",
        "enabled": True,
        "documentation": item["documentation"],
        "conditions": item["conditions"],
    }
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
        json.dump(policy, f)
        path = f.name
    try:
        subprocess.check_call(["gcloud", "monitoring", "policies", "create", "--project", project, "--policy-from-file", path])
    finally:
        os.unlink(path)
PY

echo "Monitoring configuration completed for ${PROJECT_ID}."
