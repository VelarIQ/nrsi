# NRSI Incident Runbook

This runbook covers the production `nrsi.ai` web + event relay pipeline.

## 1) Initial triage

1. Confirm active impact:
   - `https://nrsi.ai/`
   - `https://nrsi.ai/login`
   - `https://nrsi.ai/api/analytics` (expect `400` on `{}`)
   - `https://nrsi-event-relay-924270273440.us-central1.run.app/health`
2. Check Cloud Run revisions:
   - `gcloud run services describe nrsi-site --project nrsi-web-prod-20260524 --region us-central1`
   - `gcloud run services describe nrsi-event-relay --project nrsi-web-prod-20260524 --region us-central1`
3. Check open alerts in Cloud Monitoring and the `NRSI Event Relay Operations` dashboard.

## 2) Event relay failures

### Signature/auth failures spike

- Symptom: alert `nrsi-event-relay-auth-failure-alert`
- Validate shared secret parity:
  - `nrsi-site`: `EVENT_WEBHOOK_SIGNING_SECRET`
  - relay services: `EVENT_WEBHOOK_SIGNING_SECRET`
- If mismatch, redeploy `nrsi-site` and relays with aligned secret.

### BigQuery write failures

- Symptom: relay logs show `bigquery_exception` or insert errors.
- Validate runtime identity:
  - relay service account must be `nrsi-event-relay-runtime@nrsi-web-prod-20260524.iam.gserviceaccount.com`
- Required roles:
  - `roles/bigquery.dataEditor`
  - `roles/bigquery.jobUser`
- Reapply with:
  - `./scripts/setup_event_relay_service_account.sh nrsi-web-prod-20260524 nrsi-event-relay-runtime`

### Throughput drop

- Symptom: alert `nrsi-event-relay-throughput-drop-alert`
- Validate upstream webhook routes on `nrsi-site`:
  - `LAUNCH_INTEREST_WEBHOOK_URL`
  - `ANALYTICS_WEBHOOK_URL`
- Validate canary routing remains secondary:
  - staging URLs in `*_STAGING_WEBHOOK_URL`
  - `WEBHOOK_CANARY_PERCENT` in expected range.

## 3) Website deploy failures

1. Check latest run:
   - `gh run list --workflow "Deploy NRSI Website (Cloud Run)" --limit 1`
2. Inspect failed logs:
   - `gh run view <run-id> --log-failed`
3. If build/deploy fails in CI but local deploy is required urgently:
   - `./scripts/deploy_gcp.sh nrsi-web-prod-20260524 us-central1 nrsi-site`
4. Ensure CI identity still valid:
   - WIF provider and deploy SA configured via `scripts/setup_github_wif.sh`.

## 4) Rollback

For `nrsi-site`:

1. List revisions:
   - `gcloud run revisions list --service nrsi-site --project nrsi-web-prod-20260524 --region us-central1`
2. Route 100% traffic to known-good revision:
   - `gcloud run services update-traffic nrsi-site --project nrsi-web-prod-20260524 --region us-central1 --to-revisions <REV>=100`

For relays, use equivalent `gcloud run services update-traffic` commands for:
- `nrsi-event-relay`
- `nrsi-event-relay-staging`

## 5) Post-incident checklist

- Capture root cause and timeline in issue tracker.
- Add or tune alert thresholds if detection lagged.
- Run production acceptance check:
  - homepage/login `200`
  - relay unsigned event `401`
  - analytics + launch-interest `200`
  - BigQuery ingress row verification
- Update `docs/event-data-governance.md` and architecture knowledge pack if behavior changed.
