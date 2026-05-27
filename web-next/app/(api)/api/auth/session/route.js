import { NextResponse } from "next/server";
import { getSessionCookieName, getSessionMaxAge, createSessionToken, verifySessionToken } from "../../_lib/session";
import { hasAllowedDomain, roleForEmail } from "../../_lib/rbac";

export const runtime = "nodejs";

const CLIENT_ID = process.env.GOOGLE_OAUTH_CLIENT_ID || "924270273440-nlcokkui1a4l6a7dm2gi50jt26n93i9j.apps.googleusercontent.com";

async function verifyGoogleCredential(credential) {
  const url = new URL("https://oauth2.googleapis.com/tokeninfo");
  url.searchParams.set("id_token", credential);
  const response = await fetch(url, { method: "GET", cache: "no-store" });
  if (!response.ok) return null;
  const payload = await response.json();
  if (payload.aud !== CLIENT_ID) return null;
  if (!payload.email || payload.email_verified !== "true") return null;
  return payload;
}

export async function POST(request) {
  try {
    const body = await request.json();
    const credential = String(body?.credential || "");
    if (!credential) {
      return NextResponse.json({ ok: false, error: "missing_credential" }, { status: 400 });
    }

    const claims = await verifyGoogleCredential(credential);
    if (!claims) {
      return NextResponse.json({ ok: false, error: "invalid_google_token" }, { status: 401 });
    }

    const email = String(claims.email).toLowerCase();
    if (!hasAllowedDomain(email)) {
      return NextResponse.json({ ok: false, error: "domain_not_allowed" }, { status: 403 });
    }

    const role = roleForEmail(email);
    const token = createSessionToken({ email, role });
    const session = verifySessionToken(token);

    const response = NextResponse.json({ ok: true, session });
    response.cookies.set(getSessionCookieName(), token, {
      httpOnly: true,
      secure: true,
      sameSite: "lax",
      path: "/",
      maxAge: getSessionMaxAge()
    });
    return response;
  } catch (error) {
    console.error(JSON.stringify({ severity: "ERROR", event: "auth_session_create_failed", error: String(error) }));
    return NextResponse.json({ ok: false, error: "server_error" }, { status: 500 });
  }
}

export async function GET(request) {
  const token = request.cookies.get(getSessionCookieName())?.value || "";
  const session = verifySessionToken(token);
  if (!session) return NextResponse.json({ ok: false, error: "unauthorized" }, { status: 401 });
  return NextResponse.json({ ok: true, session });
}

export async function DELETE() {
  const response = NextResponse.json({ ok: true });
  response.cookies.set(getSessionCookieName(), "", {
    httpOnly: true,
    secure: true,
    sameSite: "lax",
    path: "/",
    maxAge: 0
  });
  return response;
}
