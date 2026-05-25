import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone

from flask import Flask, Response, request


app = Flask(__name__)

RESEND_API_URL = "https://api.resend.com/emails"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _extract_payload() -> dict:
    body = request.get_json(silent=True)
    if isinstance(body, dict):
        return body
    return {}


def _build_email(payload: dict) -> tuple[str, str, str]:
    incident = payload.get("incident", {}) if isinstance(payload.get("incident"), dict) else {}
    policy_name = incident.get("policy_name") or payload.get("policy_name") or "NRSI Alert"
    state = (incident.get("state") or payload.get("state") or "unknown").upper()
    incident_id = incident.get("incident_id") or payload.get("incident_id") or "unknown"
    summary = incident.get("summary") or payload.get("summary") or "Monitoring alert received."
    started_at = incident.get("started_at") or payload.get("started_at")

    started_display = "unknown"
    if started_at:
        try:
            ts = datetime.fromtimestamp(float(started_at), tz=timezone.utc)
            started_display = ts.isoformat()
        except Exception:
            started_display = str(started_at)

    subject = f"[NRSI Monitoring] {state} - {policy_name}"
    html = f"""
    <div style="font-family:Arial,sans-serif;line-height:1.5">
      <h2 style="margin:0 0 8px;">{subject}</h2>
      <p style="margin:0 0 12px;">{summary}</p>
      <table cellpadding="6" cellspacing="0" border="1" style="border-collapse:collapse;border-color:#d1d5db">
        <tr><td><strong>Policy</strong></td><td>{policy_name}</td></tr>
        <tr><td><strong>State</strong></td><td>{state}</td></tr>
        <tr><td><strong>Incident ID</strong></td><td>{incident_id}</td></tr>
        <tr><td><strong>Started At (UTC)</strong></td><td>{started_display}</td></tr>
      </table>
      <p style="margin-top:12px;">Raw payload:</p>
      <pre style="white-space:pre-wrap;background:#f7f7f7;padding:10px;border-radius:8px;">{json.dumps(payload, indent=2)}</pre>
    </div>
    """
    return subject, html, str(incident_id)


def _send_resend(subject: str, html: str, incident_id: str) -> tuple[bool, str]:
    api_key = _env("RESEND_API_KEY")
    from_email = _env("RESEND_FROM")
    recipients = [item.strip() for item in _env("ALERT_RECIPIENTS").split(",") if item.strip()]
    if not api_key or not from_email or not recipients:
        return False, "Missing RESEND_API_KEY, RESEND_FROM, or ALERT_RECIPIENTS"

    payload = {
        "from": from_email,
        "to": recipients,
        "subject": subject,
        "html": html,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        RESEND_API_URL,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Idempotency-Key": f"monitoring-alert/{incident_id}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8")
            if 200 <= resp.status < 300:
                return True, body
            return False, f"Unexpected status {resp.status}: {body}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTPError {exc.code}: {exc.read().decode('utf-8', errors='replace')}"
    except Exception as exc:
        return False, str(exc)


@app.get("/health")
def health() -> Response:
    return Response("ok", status=200, mimetype="text/plain")


@app.post("/monitoring-alert/<token>")
def monitoring_alert(token: str) -> Response:
    expected_token = _env("ALERT_WEBHOOK_TOKEN")
    if not expected_token or token != expected_token:
        return Response("unauthorized", status=401, mimetype="text/plain")

    payload = _extract_payload()
    subject, html, incident_id = _build_email(payload)
    ok, detail = _send_resend(subject, html, incident_id)
    if not ok:
        return Response(f"send_failed: {detail}", status=500, mimetype="text/plain")
    return Response("sent", status=200, mimetype="text/plain")

