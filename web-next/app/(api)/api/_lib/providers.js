const DEFAULT_TIMEOUT_MS = 4000;
const SCHEMA_VERSION = "2026-05-25.v2";

function hashString(input) {
  let hash = 0;
  for (let i = 0; i < input.length; i += 1) {
    hash = (hash << 5) - hash + input.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

async function sha256Hex(input) {
  const bytes = new TextEncoder().encode(input);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

async function hmacSha256Hex(secret, message) {
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const signature = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(message));
  return Array.from(new Uint8Array(signature))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

async function postJsonWithTimeout(url, body, timeoutMs = DEFAULT_TIMEOUT_MS) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal
    });
    return response;
  } finally {
    clearTimeout(timeout);
  }
}

export async function deliverLaunchInterest(record) {
  const webhookUrl = process.env.LAUNCH_INTEREST_WEBHOOK_URL || "";
  const webhookToken = process.env.LAUNCH_INTEREST_WEBHOOK_TOKEN || "";
  const stagingUrl = process.env.LAUNCH_INTEREST_STAGING_WEBHOOK_URL || "";
  const signingSecret = process.env.EVENT_WEBHOOK_SIGNING_SECRET || "";
  const canaryPercent = Number(process.env.WEBHOOK_CANARY_PERCENT || 0);

  if (!webhookUrl) {
    // Cloud Logging is the default durable sink when no external webhook is configured.
    return { delivered: false, skipped: true };
  }

  const idempotencyKey =
    record.idempotencyKey || (await sha256Hex(`${record.email}|${record.company}|${record.intent}|${record.schemaVersion || SCHEMA_VERSION}`));
  const eventId = `launch-${idempotencyKey}`;
  const signedAt = `${Math.floor(Date.now() / 1000)}`;
  const signaturePayload = `${eventId}|${signedAt}|launch_interest|nrsi-web`;
  const signature = signingSecret ? await hmacSha256Hex(signingSecret, signaturePayload) : "";

  const payload = {
    source: "nrsi-web",
    type: "launch_interest",
    record,
    ts: new Date().toISOString(),
    schemaVersion: record.schemaVersion || SCHEMA_VERSION,
    eventId,
    signedAt,
    signature
  };

  try {
    const response = await postJsonWithTimeout(
      webhookUrl,
      webhookToken ? { ...payload, token: webhookToken } : payload,
      Number(process.env.LAUNCH_INTEREST_WEBHOOK_TIMEOUT_MS || DEFAULT_TIMEOUT_MS)
    );

    if (!response.ok) {
      return { delivered: false, skipped: false, error: `launch_interest_webhook_failed:${response.status}` };
    }

    if (stagingUrl && canaryPercent > 0 && hashString(idempotencyKey) % 100 < canaryPercent) {
      await postJsonWithTimeout(stagingUrl, payload, Number(process.env.LAUNCH_INTEREST_WEBHOOK_TIMEOUT_MS || DEFAULT_TIMEOUT_MS));
    }

    return { delivered: true, skipped: false, eventId };
  } catch (error) {
    return { delivered: false, skipped: false, error: String(error) };
  }
}

export async function deliverAnalytics(record) {
  const webhookUrl = process.env.ANALYTICS_WEBHOOK_URL || "";
  const webhookToken = process.env.ANALYTICS_WEBHOOK_TOKEN || "";
  const stagingUrl = process.env.ANALYTICS_STAGING_WEBHOOK_URL || "";
  const signingSecret = process.env.EVENT_WEBHOOK_SIGNING_SECRET || "";
  const canaryPercent = Number(process.env.WEBHOOK_CANARY_PERCENT || 0);

  if (!webhookUrl) {
    return { delivered: false, skipped: true };
  }

  const idempotencyKey =
    record.idempotencyKey ||
    (await sha256Hex(`${record.eventName}|${record.path}|${record.ts}|${record.ip}|${record.schemaVersion || SCHEMA_VERSION}`));
  const eventId = `analytics-${idempotencyKey}`;
  const signedAt = `${Math.floor(Date.now() / 1000)}`;
  const signaturePayload = `${eventId}|${signedAt}|analytics_event|nrsi-web`;
  const signature = signingSecret ? await hmacSha256Hex(signingSecret, signaturePayload) : "";

  const payload = {
    source: "nrsi-web",
    type: "analytics_event",
    record,
    ts: new Date().toISOString(),
    schemaVersion: record.schemaVersion || SCHEMA_VERSION,
    eventId,
    signedAt,
    signature
  };

  try {
    const response = await postJsonWithTimeout(
      webhookUrl,
      webhookToken ? { ...payload, token: webhookToken } : payload,
      Number(process.env.ANALYTICS_WEBHOOK_TIMEOUT_MS || DEFAULT_TIMEOUT_MS)
    );

    if (!response.ok) {
      return { delivered: false, skipped: false, error: `analytics_webhook_failed:${response.status}` };
    }

    if (stagingUrl && canaryPercent > 0 && hashString(idempotencyKey) % 100 < canaryPercent) {
      await postJsonWithTimeout(stagingUrl, payload, Number(process.env.ANALYTICS_WEBHOOK_TIMEOUT_MS || DEFAULT_TIMEOUT_MS));
    }

    return { delivered: true, skipped: false, eventId };
  } catch (error) {
    return { delivered: false, skipped: false, error: String(error) };
  }
}
