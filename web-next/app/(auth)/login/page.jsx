"use client";

import Script from "next/script";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

const SESSION_KEY = "nrsi_site_session_v1";
const CLIENT_ID = "924270273440-nlcokkui1a4l6a7dm2gi50jt26n93i9j.apps.googleusercontent.com";
const ALLOWED_DOMAIN = "velariq.ai";

function parseJwt(token) {
  try {
    const payload = token.split(".")[1];
    const normalized = payload.replace(/-/g, "+").replace(/_/g, "/");
    return JSON.parse(atob(normalized));
  } catch {
    return null;
  }
}

export default function LoginPage() {
  const router = useRouter();
  const [error, setError] = useState("");
  const [gisReady, setGisReady] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const nextPath = params.get("next") || "/dashboard";
    if (!gisReady || !window.google?.accounts?.id) return;

    window.google.accounts.id.initialize({
      client_id: CLIENT_ID,
      callback: (response) => {
        const claims = parseJwt(response.credential || "");
        if (!claims?.email || !claims.email_verified) {
          setError("Could not verify Google account.");
          return;
        }
        if (claims.hd !== ALLOWED_DOMAIN) {
          setError(`Use your ${ALLOWED_DOMAIN} workspace account.`);
          return;
        }
        const email = claims.email.toLowerCase();
        const role = email.startsWith("admin@") || email.includes("+admin") ? "admin" : "user";
        localStorage.setItem(
          SESSION_KEY,
          JSON.stringify({
            email,
            role,
            authMethod: "google",
            loginAt: new Date().toISOString()
          })
        );
        if (nextPath === "/admin" && role !== "admin") {
          router.replace("/dashboard");
          return;
        }
        router.replace(nextPath);
      }
    });

    window.google.accounts.id.renderButton(document.getElementById("googleButton"), {
      theme: "outline",
      size: "large",
      width: 320
    });
  }, [gisReady, router]);

  return (
    <main className="section fade-in">
      <div className="container auth-shell panel page-hero">
        <p className="eyebrow">Secure Access</p>
        <h1>Atherion Workspace Login</h1>
        <p className="muted">Enterprise SSO is required. Continue with your authorized Google Workspace identity.</p>
        <div id="googleButton" style={{ marginTop: 12 }} />
        <p className="auth-error">{error}</p>
      </div>
      <Script src="https://accounts.google.com/gsi/client" strategy="afterInteractive" onLoad={() => setGisReady(true)} />
    </main>
  );
}
