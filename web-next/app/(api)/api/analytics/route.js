import { deliverAnalytics } from "../_lib/providers";
import { getClientIp, maskIp, rateLimit } from "../_lib/security";
import { validateAnalytics } from "../_lib/validation";

export const runtime = "nodejs";
const SCHEMA_VERSION = "2026-05-25.v2";

export async function POST(request) {
  try {
    const ip = getClientIp(request);
    const rate = rateLimit({ key: `analytics:${ip}`, max: 240, windowMs: 60_000 });
    if (rate.limited) {
      console.warn(JSON.stringify({ severity: "WARNING", event: "analytics_rate_limited", ip: maskIp(ip) }));
      return Response.json({ ok: false, error: "rate_limited" }, { status: 429 });
    }

    const payload = await request.json();
    const validated = validateAnalytics(payload);
    if (!validated.ok) {
      return Response.json({ ok: false, error: validated.error }, { status: 400 });
    }

    const record = {
      ...validated.value,
      ip: maskIp(ip),
      schemaVersion: SCHEMA_VERSION,
      idempotencyKey: `${validated.value.eventName}|${validated.value.path}|${validated.value.ts}|${maskIp(ip)}|${SCHEMA_VERSION}`
    };

    const delivery = await deliverAnalytics(record);
    if (delivery.error) {
      console.warn(
        JSON.stringify({
          severity: "WARNING",
          event: "analytics_delivery_failed",
          error: delivery.error,
          ip: record.ip
        })
      );
    }
    console.log(
      JSON.stringify({
        severity: "NOTICE",
        event: "launch_analytics",
        schemaVersion: SCHEMA_VERSION,
        eventId: delivery.eventId || "",
        name: record.eventName,
        path: record.path,
        ip: record.ip,
        externalDelivery: delivery.delivered ? "ok" : "skipped"
      })
    );
    return Response.json({ ok: true });
  } catch (error) {
    console.error(
      JSON.stringify({
        severity: "ERROR",
        event: "launch_analytics_failed",
        error: String(error)
      })
    );
    return Response.json({ ok: false, error: "bad_payload" }, { status: 400 });
  }
}
