#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:-nrsi-web-prod-20260524}"
PROD_SERVICE="${2:-nrsi-event-relay}"
STAGING_SERVICE="${3:-nrsi-event-relay-staging}"

TMP_JSON="$(mktemp)"
trap 'rm -f "${TMP_JSON}"' EXIT

cat > "${TMP_JSON}" <<EOF
{
  "displayName": "NRSI Event Relay Operations",
  "mosaicLayout": {
    "columns": 12,
    "tiles": [
      {
        "xPos": 0,
        "yPos": 0,
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Relay Ingest Count (Prod)",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\\"logging.googleapis.com/user/nrsi_event_relay_ingest_count\\" AND resource.type=\\"cloud_run_revision\\" AND resource.label.\\"service_name\\"=\\"${PROD_SERVICE}\\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_DELTA"
                    }
                  }
                },
                "plotType": "LINE"
              }
            ]
          }
        }
      },
      {
        "xPos": 6,
        "yPos": 0,
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Relay Ingest Count (Staging)",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\\"run.googleapis.com/request_count\\" AND resource.type=\\"cloud_run_revision\\" AND resource.label.\\"service_name\\"=\\"${STAGING_SERVICE}\\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_RATE"
                    }
                  }
                },
                "plotType": "LINE"
              }
            ]
          }
        }
      },
      {
        "xPos": 0,
        "yPos": 4,
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Signature/Auth Failures",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\\"logging.googleapis.com/user/nrsi_event_relay_auth_fail_count\\" AND resource.type=\\"cloud_run_revision\\" AND resource.label.\\"service_name\\"=\\"${PROD_SERVICE}\\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_DELTA"
                    }
                  }
                },
                "plotType": "STACKED_BAR"
              }
            ]
          }
        }
      },
      {
        "xPos": 6,
        "yPos": 4,
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Relay 5xx Rate (Prod)",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\\"run.googleapis.com/request_count\\" AND resource.type=\\"cloud_run_revision\\" AND resource.label.\\"service_name\\"=\\"${PROD_SERVICE}\\" AND metric.label.\\"response_code_class\\"=\\"5xx\\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_RATE"
                    }
                  }
                },
                "plotType": "LINE"
              }
            ]
          }
        }
      }
    ]
  }
}
EOF

existing="$(
  gcloud monitoring dashboards list --project "${PROJECT_ID}" --format=json | python3 -c '
import json,sys
rows=json.load(sys.stdin)
for row in rows:
    if row.get("displayName")=="NRSI Event Relay Operations":
        print(row.get("name",""))
        break
'
)"
if [[ -n "${existing}" ]]; then
  gcloud monitoring dashboards update "${existing}" --project "${PROJECT_ID}" --config-from-file "${TMP_JSON}" >/dev/null
  echo "Updated dashboard: ${existing}"
else
  gcloud monitoring dashboards create --project "${PROJECT_ID}" --config-from-file "${TMP_JSON}" >/dev/null
  echo "Created dashboard: NRSI Event Relay Operations"
fi
