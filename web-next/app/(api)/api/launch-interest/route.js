import { deliverLaunchInterest } from "../_lib/providers";
import { getClientIp, isLikelyBot, maskIp, rateLimit, verifyTurnstile } from "../_lib/security";
import { validateLaunchInterest } from "../_lib/validation";

export const runtime = "nodejs";
const SCHEMA_VERSION = "2026-05-25.v2";

export async function POST(request) {
  try {
    const ip = getClientIp(request);
    const ua = request.headers.get("user-agent") || "";
    const turnstileSecret = process.env.TURNSTILE_SECRET_KEY || "";

    const rate = rateLimit({ key: `waitlist:${ip}`, max: 6, windowMs: 60_000 });
    if (rate.limited) {
      console.warn(JSON.stringify({ severity: "WARNING", event: "waitlist_rate_limited", ip: maskIp(ip) }));
      return Response.json({ ok: false, error: "rate_limited" }, { status: 429 });
    }

    if (isLikelyBot(ua)) {
      console.warn(
        JSON.stringify({
          severity: "WARNING",
          event: "waitlist_bot_like_user_agent",
          ip: maskIp(ip),
          ua: ua.slice(0, 120)
        })
      );
      return Response.json({ ok: false, error: "blocked" }, { status: 400 });
    }

    const payload = await request.json();
    const validated = validateLaunchInterest(payload);
    if (!validated.ok) {
      return Response.json({ ok: false, error: validated.error }, { status: 400 });
    }

    if (validated.value.website) {
      console.warn(JSON.stringify({ severity: "WARNING", event: "waitlist_honeypot_triggered", ip: maskIp(ip) }));
      return Response.json({ ok: false, error: "blocked" }, { status: 400 });
    }

    const elapsedMs = Date.now() - validated.value.startedAt;
    if (!Number.isFinite(validated.value.startedAt) || elapsedMs < 1500 || elapsedMs > 86_400_000) {
      console.warn(
        JSON.stringify({
          severity: "WARNING",
          event: "waitlist_invalid_submission_timing",
          ip: maskIp(ip),
          elapsedMs
        })
      );
      return Response.json({ ok: false, error: "invalid_submission" }, { status: 400 });
    }

    if (turnstileSecret) {
      const turnstile = await verifyTurnstile({
        token: validated.value.turnstileToken,
        remoteip: ip,
        secret: turnstileSecret
      });
      if (!turnstile.success) {
        console.warn(
          JSON.stringify({
            severity: "WARNING",
            event: "waitlist_turnstile_failed",
            ip: maskIp(ip),
            errors: turnstile.errors
          })
        );
        return Response.json({ ok: false, error: "captcha_failed" }, { status: 400 });
      }
    }

    const record = {
      ts: new Date().toISOString(),
      name: validated.value.name,
      email: validated.value.email,
      company: validated.value.company,
      intent: validated.value.intent,
      ip: maskIp(ip),
      schemaVersion: SCHEMA_VERSION,
      idempotencyKey: `${validated.value.email}|${validated.value.company}|${validated.value.intent}|${SCHEMA_VERSION}`
    };

    const delivery = await deliverLaunchInterest(record);
    if (delivery.error) {
      console.warn(
        JSON.stringify({
          severity: "WARNING",
          event: "launch_interest_delivery_failed",
          error: delivery.error,
          ip: record.ip
        })
      );
    }
    console.log(
      JSON.stringify({
        severity: "NOTICE",
        event: "launch_interest_captured",
        schemaVersion: SCHEMA_VERSION,
        eventId: delivery.eventId || "",
        intent: record.intent,
        ip: record.ip,
        externalDelivery: delivery.delivered ? "ok" : "skipped"
      })
    );
    return Response.json({ ok: true });
  } catch (error) {
    console.error(
      JSON.stringify({
        severity: "ERROR",
        event: "launch_interest_failed",
        error: String(error)
      })
    );
    return Response.json({ ok: false, error: "server_error" }, { status: 500 });
  }
}
