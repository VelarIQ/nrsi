import hashlib
import hmac
import json
import os
import time
from datetime import datetime, timezone
from typing import Any

from flask import Flask, Response, request
from google.cloud import bigquery


app = Flask(__name__)
_BQ_CLIENT = None
_SEEN_EVENT_IDS = {}
_SEEN_TTL_SECONDS = 24 * 60 * 60


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _extract_payload() -> dict:
    body = request.get_json(silent=True)
    if isinstance(body, dict):
        return body
    return {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cleanup_seen(now_s: int) -> None:
    stale = [k for k, exp in _SEEN_EVENT_IDS.items() if exp <= now_s]
    for key in stale:
        _SEEN_EVENT_IDS.pop(key, None)


def _is_duplicate(event_id: str) -> bool:
    if not event_id:
        return False
    now_s = int(time.time())
    _cleanup_seen(now_s)
    if event_id in _SEEN_EVENT_IDS:
        return True
    _SEEN_EVENT_IDS[event_id] = now_s + _SEEN_TTL_SECONDS
    return False


def _signing_payload(event_id: str, ts: str, record_type: str, source: str) -> str:
    return f"{event_id}|{ts}|{record_type}|{source}"


def _verify_signature(payload: dict) -> tuple[bool, str]:
    secret = _env("EVENT_WEBHOOK_SIGNING_SECRET")
    if not secret:
        return True, "skipped_no_secret"

    signature = str(payload.get("signature") or "")
    signed_at = str(payload.get("signedAt") or "")
    event_id = str(payload.get("eventId") or "")
    record_type = str(payload.get("type") or "")
    source = str(payload.get("source") or "")

    if not signature or not signed_at or not event_id:
        return False, "missing_signature_fields"

    try:
        signed_epoch = int(signed_at)
    except Exception:
        return False, "invalid_signedAt"

    max_skew = int(_env("EVENT_SIGNATURE_MAX_SKEW_SECONDS", "300") or "300")
    now_epoch = int(time.time())
    if abs(now_epoch - signed_epoch) > max_skew:
        return False, "signature_expired"

    message = _signing_payload(event_id, signed_at, record_type, source).encode("utf-8")
    expected = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        return False, "signature_mismatch"
    return True, "verified"


def _allowlisted_record_fields(record: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "name",
        "email",
        "company",
        "intent",
        "ip",
        "eventName",
        "path",
        "ts",
        "extras",
        "schemaVersion",
        "idempotencyKey",
    }
    filtered = {}
    for key in allowed:
        if key in record:
            filtered[key] = record[key]
    return filtered


def _privacy_record(record: dict[str, Any]) -> dict[str, Any]:
    value = dict(record)
    email = str(value.get("email") or "")
    if email:
        value["email_sha256"] = hashlib.sha256(email.encode("utf-8")).hexdigest()
        if _env("STORE_PLAIN_EMAIL", "false").lower() != "true":
            value.pop("email", None)
    return value


def _get_bq_client():
    global _BQ_CLIENT
    if _BQ_CLIENT is None:
        _BQ_CLIENT = bigquery.Client()
    return _BQ_CLIENT


def _bigquery_table() -> str:
    project = _env("BIGQUERY_PROJECT") or _env("GOOGLE_CLOUD_PROJECT")
    dataset = _env("BIGQUERY_DATASET", "nrsi_events")
    table = _env("BIGQUERY_TABLE", "product_events")
    if not project:
        return ""
    return f"{project}.{dataset}.{table}"


def _write_bigquery(event: dict[str, Any]) -> tuple[bool, str]:
    table = _bigquery_table()
    if not table:
        return False, "missing_bigquery_target"
    try:
        client = _get_bq_client()
        errors = client.insert_rows_json(table, [event], row_ids=[event.get("event_id", "") or None])
        if errors:
            return False, f"bigquery_insert_errors:{errors}"
        return True, "ok"
    except Exception as exc:
        return False, f"bigquery_exception:{exc}"


@app.get("/health")
def health() -> Response:
    return Response("ok", status=200, mimetype="text/plain")


@app.post("/event/<token>")
def event_ingest(token: str) -> Response:
    expected_token = _env("EVENT_WEBHOOK_TOKEN")
    if not expected_token or token != expected_token:
        return Response("unauthorized", status=401, mimetype="text/plain")

    payload = _extract_payload()
    verified, verify_reason = _verify_signature(payload)
    if not verified:
        print(
            json.dumps(
                {"severity": "WARNING", "event": "product_event_signature_failed", "reason": verify_reason[:120]}
            ),
            flush=True,
        )
        return Response("unauthorized", status=401, mimetype="text/plain")

    event_id = str(payload.get("eventId") or "")
    if _is_duplicate(event_id):
        print(
            json.dumps({"severity": "NOTICE", "event": "product_event_duplicate_ignored", "event_id": event_id[:120]}),
            flush=True,
        )
        return Response("duplicate", status=202, mimetype="text/plain")

    record_type = str(payload.get("type") or "unknown")
    record_source = str(payload.get("source") or "unknown")
    record_ts = str(payload.get("ts") or _now_iso())
    schema_version = str(payload.get("schemaVersion") or "2026-05-25.v1")
    record = payload.get("record", {}) if isinstance(payload.get("record"), dict) else {}
    record = _privacy_record(_allowlisted_record_fields(record))

    event_doc = {
        "event_id": event_id or hashlib.sha256(f"{record_source}|{record_type}|{record_ts}".encode("utf-8")).hexdigest(),
        "event_type": record_type[:80],
        "source": record_source[:80],
        "timestamp": record_ts[:80],
        "schema_version": schema_version[:40],
        "record": json.dumps(record, separators=(",", ":"), sort_keys=True),
        "received_at": _now_iso(),
    }

    bq_ok, bq_detail = _write_bigquery(event_doc)
    print(
        json.dumps(
            {
                "severity": "NOTICE",
                "event": "product_event_received",
                "event_id": event_doc["event_id"][:120],
                "type": record_type[:80],
                "source": record_source[:80],
                "timestamp": record_ts[:80],
                "signature": verify_reason[:40],
                "bigquery": "ok" if bq_ok else "failed",
                "bigquery_detail": bq_detail[:500],
                "record": record,
            }
        ),
        flush=True,
    )
    return Response("ok", status=200, mimetype="text/plain")


@app.post("/event-delete/<token>")
def event_delete(token: str) -> Response:
    expected_token = _env("EVENT_WEBHOOK_TOKEN")
    if not expected_token or token != expected_token:
        return Response("unauthorized", status=401, mimetype="text/plain")

    payload = _extract_payload()
    event_id = str(payload.get("eventId") or "").strip()
    if not event_id:
        return Response("missing_event_id", status=400, mimetype="text/plain")

    print(
        json.dumps(
            {
                "severity": "NOTICE",
                "event": "product_event_delete_requested",
                "event_id": event_id[:120],
                "requested_at": _now_iso(),
            }
        ),
        flush=True,
    )
    # Retention/deletion workflow: process this request downstream from logs or workflow jobs.
    return Response("accepted", status=202, mimetype="text/plain")
