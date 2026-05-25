const RATE_LIMIT_STORE = globalThis.__nrsiRateLimitStore || new Map();
if (!globalThis.__nrsiRateLimitStore) {
  globalThis.__nrsiRateLimitStore = RATE_LIMIT_STORE;
}

function compact(nowMs) {
  for (const [key, value] of RATE_LIMIT_STORE.entries()) {
    if (value.expiresAt <= nowMs) RATE_LIMIT_STORE.delete(key);
  }
}

export function getClientIp(request) {
  const forwarded = request.headers.get("x-forwarded-for");
  if (forwarded) return forwarded.split(",")[0].trim();
  const realIp = request.headers.get("x-real-ip");
  if (realIp) return realIp.trim();
  return "unknown";
}

export function maskIp(ip) {
  if (!ip || ip === "unknown") return "unknown";
  if (ip.includes(":")) {
    const segments = ip.split(":");
    return `${segments[0] || ""}:${segments[1] || ""}::`;
  }
  const parts = ip.split(".");
  if (parts.length !== 4) return "masked";
  return `${parts[0]}.${parts[1]}.x.x`;
}

export function isLikelyBot(ua = "") {
  return /(bot|spider|crawler|curl|python|wget|postman|insomnia|headless|httpclient)/i.test(ua);
}

export function rateLimit({ key, max, windowMs }) {
  const nowMs = Date.now();
  compact(nowMs);

  const existing = RATE_LIMIT_STORE.get(key);
  if (!existing || existing.expiresAt <= nowMs) {
    RATE_LIMIT_STORE.set(key, { count: 1, expiresAt: nowMs + windowMs });
    return { limited: false, remaining: max - 1 };
  }

  existing.count += 1;
  if (existing.count > max) {
    return { limited: true, remaining: 0 };
  }

  return { limited: false, remaining: Math.max(0, max - existing.count) };
}

export async function verifyTurnstile({ token, remoteip, secret }) {
  if (!secret) return { success: true, skipped: true };
  if (!token) return { success: false, errors: ["missing-input-response"] };

  const formData = new URLSearchParams();
  formData.set("secret", secret);
  formData.set("response", token);
  if (remoteip && remoteip !== "unknown") formData.set("remoteip", remoteip);

  const response = await fetch("https://challenges.cloudflare.com/turnstile/v0/siteverify", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: formData.toString()
  });

  if (!response.ok) {
    return { success: false, errors: ["turnstile-http-failure"] };
  }

  const payload = await response.json();
  return {
    success: Boolean(payload?.success),
    errors: Array.isArray(payload?.["error-codes"]) ? payload["error-codes"] : []
  };
}
