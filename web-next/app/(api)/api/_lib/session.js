import crypto from "crypto";
import { getPermissions } from "./rbac";

const SESSION_COOKIE = "nrsi_auth";
const SESSION_MAX_AGE_SECONDS = Number(process.env.AUTH_SESSION_TTL_SECONDS || 60 * 60 * 8);
const SESSION_SECRET = process.env.AUTH_SESSION_SECRET || "dev-insecure-change-me";

function toBase64Url(value) {
  return Buffer.from(value).toString("base64url");
}

function fromBase64Url(value) {
  return Buffer.from(value, "base64url").toString("utf8");
}

function signPayload(encodedPayload) {
  return crypto.createHmac("sha256", SESSION_SECRET).update(encodedPayload).digest("base64url");
}

export function createSessionToken({ email, role }) {
  const now = Math.floor(Date.now() / 1000);
  const payload = {
    email,
    role,
    permissions: getPermissions(role),
    loginAt: new Date().toISOString(),
    iat: now,
    exp: now + SESSION_MAX_AGE_SECONDS
  };
  const encodedPayload = toBase64Url(JSON.stringify(payload));
  const encodedSig = signPayload(encodedPayload);
  return `${encodedPayload}.${encodedSig}`;
}

export function verifySessionToken(token) {
  if (!token || typeof token !== "string" || !token.includes(".")) return null;
  const [encodedPayload, encodedSig] = token.split(".");
  const expectedSig = signPayload(encodedPayload);
  const actualSig = Buffer.from(encodedSig || "");
  const expectedSigBuf = Buffer.from(expectedSig);
  if (actualSig.length !== expectedSigBuf.length || !crypto.timingSafeEqual(actualSig, expectedSigBuf)) {
    return null;
  }

  try {
    const payload = JSON.parse(fromBase64Url(encodedPayload));
    const now = Math.floor(Date.now() / 1000);
    if (!payload?.email || !payload?.role || !payload?.exp || payload.exp < now) return null;
    return payload;
  } catch {
    return null;
  }
}

export function getSessionCookieName() {
  return SESSION_COOKIE;
}

export function getSessionMaxAge() {
  return SESSION_MAX_AGE_SECONDS;
}
