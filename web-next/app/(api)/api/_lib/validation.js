const ALLOWED_INTENTS = new Set(["builder", "design_partner", "enterprise", "investor", "media"]);

export function asShortString(value, maxLen = 200) {
  return String(value || "")
    .trim()
    .slice(0, maxLen);
}

export function isValidEmail(email) {
  if (!email) return false;
  if (email.length > 240) return false;
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

export function validateLaunchInterest(payload) {
  const name = asShortString(payload?.name, 200);
  const email = asShortString(payload?.email, 240).toLowerCase();
  const company = asShortString(payload?.company, 200);
  const intent = asShortString(payload?.intent || "builder", 80);
  const website = asShortString(payload?.website, 200);
  const turnstileToken = asShortString(payload?.turnstileToken, 2048);
  const startedAt = Number(payload?.startedAt || 0);

  if (!name || name.length < 2) {
    return { ok: false, error: "invalid_name" };
  }
  if (!isValidEmail(email)) {
    return { ok: false, error: "invalid_email" };
  }
  if (!company || company.length < 2) {
    return { ok: false, error: "invalid_company" };
  }
  if (!ALLOWED_INTENTS.has(intent)) {
    return { ok: false, error: "invalid_intent" };
  }

  return {
    ok: true,
    value: {
      name,
      email,
      company,
      intent,
      website,
      startedAt,
      turnstileToken
    }
  };
}

export function validateAnalytics(payload) {
  const eventName = asShortString(payload?.eventName, 80);
  if (!eventName || eventName.length < 2) return { ok: false, error: "invalid_event_name" };

  const path = asShortString(payload?.path || "/", 300);
  const ts = Number(payload?.ts || Date.now());
  const extras = {};

  for (const [key, value] of Object.entries(payload || {})) {
    if (key === "eventName" || key === "path" || key === "ts") continue;
    if (typeof value === "string") extras[key.slice(0, 40)] = asShortString(value, 200);
    else if (typeof value === "number" || typeof value === "boolean") extras[key.slice(0, 40)] = value;
  }

  return {
    ok: true,
    value: { eventName, path, ts: Number.isFinite(ts) ? ts : Date.now(), extras }
  };
}
