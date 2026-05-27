"use client";

import Script from "next/script";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

type Session = {
  role: "user" | "admin" | "super_admin";
};

type AuthResponse = {
  ok?: boolean;
  error?: string;
  session?: Session;
};

type GoogleCredentialResponse = {
  credential: string;
};

type GoogleAccounts = {
  id: {
    initialize: (input: { client_id: string; callback: (response: GoogleCredentialResponse) => void }) => void;
    renderButton: (element: HTMLElement | null, options: Record<string, string | number>) => void;
  };
};

const CLIENT_ID =
  process.env.NEXT_PUBLIC_GOOGLE_OAUTH_CLIENT_ID ||
  "924270273440-nlcokkui1a4l6a7dm2gi50jt26n93i9j.apps.googleusercontent.com";

export default function LoginPage() {
  const router = useRouter();
  const [error, setError] = useState("");
  const [gisReady, setGisReady] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const nextPath = params.get("next") || "/dashboard";
    const google = (window as Window & { google?: { accounts?: GoogleAccounts } }).google?.accounts;

    if (!gisReady || !google?.id) return;

    google.id.initialize({
      client_id: CLIENT_ID,
      callback: async (response) => {
        try {
          setError("");
          const authResponse = await fetch("/api/auth/session", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ credential: response.credential })
          });
          const body = (await authResponse.json()) as AuthResponse;

          if (!authResponse.ok || !body?.ok || !body?.session) {
            if (body?.error === "domain_not_allowed") {
              setError("Use your authorized workspace account.");
              return;
            }
            setError("Could not verify Google account.");
            return;
          }

          if (nextPath === "/admin" && body.session.role !== "admin" && body.session.role !== "super_admin") {
            router.replace("/dashboard");
            return;
          }

          router.replace(nextPath);
        } catch {
          setError("Authentication failed. Please try again.");
        }
      }
    });

    google.id.renderButton(document.getElementById("googleButton"), {
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
