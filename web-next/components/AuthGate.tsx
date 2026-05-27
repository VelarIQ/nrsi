"use client";

import { useEffect, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";

type UserRole = "user" | "admin" | "super_admin";

type Session = {
  email: string;
  role: UserRole;
  permissions?: string[];
  loginAt?: string;
  authMethod?: string;
};

type AuthGateProps = {
  requireRole?: UserRole;
  requirePermission?: string;
  children: (session: Session) => ReactNode;
};

const ROLE_PRIORITY: Record<UserRole, number> = {
  user: 10,
  admin: 50,
  super_admin: 100
};

function hasRoleAtLeast(role: string, requiredRole: UserRole): boolean {
  const safeRole: UserRole = role === "admin" || role === "super_admin" ? role : "user";
  return ROLE_PRIORITY[safeRole] >= ROLE_PRIORITY[requiredRole];
}

function hasPermission(session: Session | null, requiredPermission?: string): boolean {
  if (!requiredPermission) return true;
  return Array.isArray(session?.permissions) && session.permissions.includes(requiredPermission);
}

export default function AuthGate({ requireRole, requirePermission, children }: AuthGateProps) {
  const router = useRouter();
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    const run = async () => {
      try {
        const response = await fetch("/api/auth/session", { method: "GET", cache: "no-store" });
        const body = (await response.json()) as { ok?: boolean; session?: Session };

        if (!response.ok || !body?.ok || !body?.session) {
          router.replace("/login");
          return;
        }

        const currentSession = body.session;
        if (requireRole && !hasRoleAtLeast(currentSession.role, requireRole)) {
          router.replace("/dashboard");
          return;
        }

        if (!hasPermission(currentSession, requirePermission)) {
          router.replace("/dashboard");
          return;
        }

        if (!cancelled) {
          setSession(currentSession);
        }
      } catch {
        router.replace("/login");
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    void run();

    return () => {
      cancelled = true;
    };
  }, [requireRole, requirePermission, router]);

  if (loading || !session) return null;
  return <>{children(session)}</>;
}
