"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

const SESSION_KEY = "nrsi_site_session_v1";

export default function AuthGate({ requireRole, children }) {
  const router = useRouter();
  const [session, setSession] = useState(null);

  useEffect(() => {
    const raw = localStorage.getItem(SESSION_KEY);
    if (!raw) {
      router.replace("/login");
      return;
    }
    const parsed = JSON.parse(raw);
    if (requireRole && parsed.role !== requireRole) {
      router.replace("/dashboard");
      return;
    }
    setSession(parsed);
  }, [requireRole, router]);

  if (!session) return null;

  return children(session);
}
