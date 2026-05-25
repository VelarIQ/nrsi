# NRSI Product Event Data Governance

This document defines retention, privacy, and deletion workflows for product events ingested by `nrsi-event-relay`.

## Data Minimization

- Relay accepts a strict allowlist of record fields:
  - `name`, `email`, `company`, `intent`, `ip`, `eventName`, `path`, `ts`, `extras`, `schemaVersion`, `idempotencyKey`.
- Emails are hashed to `email_sha256` before persistence.
- Plain email storage is disabled by default (`STORE_PLAIN_EMAIL=false`).

## Retention

- BigQuery sink is day-partitioned and configured with table expiration.
- Default retention target is 30 days via:
  - `scripts/configure_event_relay_bigquery.sh <project> <dataset> <table> <ttl_days>`

## Deletion Workflow

- Relay exposes `POST /event-delete/<token>` with payload:
  - `{ "eventId": "<id>" }`
- Requests are logged as `product_event_delete_requested`.
- A downstream job should execute the matching BigQuery delete and append an audit record.

## Integrity and Replay Protection

- Upstream events are signed with HMAC SHA-256 (`EVENT_WEBHOOK_SIGNING_SECRET`).
- Signature max skew is enforced (`EVENT_SIGNATURE_MAX_SKEW_SECONDS`, default 300s).
- Event IDs are deduplicated in relay memory for 24 hours.
- BigQuery inserts use `event_id` as row insert ID for dedupe-safe retries.
