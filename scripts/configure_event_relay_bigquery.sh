#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:-nrsi-web-prod-20260524}"
DATASET="${2:-nrsi_events}"
TABLE="${3:-product_events}"
TABLE_TTL_DAYS="${4:-30}"

gcloud services enable bigquery.googleapis.com --project "${PROJECT_ID}" >/dev/null

bq --project_id="${PROJECT_ID}" --location=US mk --dataset --description "NRSI product event relay sink" "${PROJECT_ID}:${DATASET}" >/dev/null 2>&1 || true

cat > /tmp/nrsi_event_table_schema.json <<'EOF'
[
  {"name":"event_id","type":"STRING","mode":"REQUIRED"},
  {"name":"event_type","type":"STRING","mode":"REQUIRED"},
  {"name":"source","type":"STRING","mode":"REQUIRED"},
  {"name":"timestamp","type":"TIMESTAMP","mode":"REQUIRED"},
  {"name":"schema_version","type":"STRING","mode":"NULLABLE"},
  {"name":"record","type":"JSON","mode":"NULLABLE"},
  {"name":"received_at","type":"TIMESTAMP","mode":"REQUIRED"}
]
EOF

bq --project_id="${PROJECT_ID}" mk --table --time_partitioning_type=DAY --time_partitioning_expiration=$((TABLE_TTL_DAYS*24*60*60)) --schema=/tmp/nrsi_event_table_schema.json "${PROJECT_ID}:${DATASET}.${TABLE}" >/dev/null 2>&1 || true

rm -f /tmp/nrsi_event_table_schema.json
echo "BigQuery sink ready: ${PROJECT_ID}.${DATASET}.${TABLE} (TTL ${TABLE_TTL_DAYS} days)"
