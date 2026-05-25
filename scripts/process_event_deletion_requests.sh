#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:-nrsi-web-prod-20260524}"
DATASET="${2:-nrsi_events}"
PROD_TABLE="${3:-product_events}"
STAGING_TABLE="${4:-product_events_staging}"
LOOKBACK_HOURS="${LOOKBACK_HOURS:-24}"
DRY_RUN="${DRY_RUN:-false}"

START_TIME="$(python3 -c "from datetime import datetime, timedelta, timezone; print((datetime.now(timezone.utc)-timedelta(hours=${LOOKBACK_HOURS})).strftime('%Y-%m-%dT%H:%M:%SZ'))")"
FILTER="resource.type=\"cloud_run_revision\" AND jsonPayload.event=\"product_event_delete_requested\" AND timestamp>=\"${START_TIME}\""

TMP_IDS="$(mktemp)"
trap 'rm -f "${TMP_IDS}"' EXIT

gcloud logging read "${FILTER}" \
  --project "${PROJECT_ID}" \
  --format=json \
  --limit=1000 | python3 -c 'import json,sys; rows=json.load(sys.stdin); ids=sorted({((r.get("jsonPayload") or {}).get("event_id") or "").strip() for r in rows}); [print(i) for i in ids if i]' > "${TMP_IDS}"

if [[ ! -s "${TMP_IDS}" ]]; then
  echo "No deletion requests found in the last ${LOOKBACK_HOURS} hours."
  exit 0
fi

mapfile -t EVENT_IDS < "${TMP_IDS}"
echo "Found ${#EVENT_IDS[@]} deletion request(s)."

build_delete_sql() {
  local table="$1"
  local quoted=""
  for id in "${EVENT_IDS[@]}"; do
    escaped="${id//\'/\\\'}"
    if [[ -n "${quoted}" ]]; then
      quoted="${quoted},"
    fi
    quoted="${quoted}'${escaped}'"
  done
  cat <<EOF
DELETE FROM \`${PROJECT_ID}.${DATASET}.${table}\`
WHERE event_id IN (${quoted})
EOF
}

for table in "${PROD_TABLE}" "${STAGING_TABLE}"; do
  SQL="$(build_delete_sql "${table}")"
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY_RUN=true, skipping delete for ${table}"
    echo "${SQL}"
  else
    bq --project_id="${PROJECT_ID}" query --nouse_legacy_sql "${SQL}" >/dev/null
    echo "Processed deletion requests in ${PROJECT_ID}.${DATASET}.${table}"
  fi
done
