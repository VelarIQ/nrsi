"use client";

import Link from "next/link";
import AuthGate from "../../../components/AuthGate";

export default function DashboardPage() {
  return (
    <main className="section fade-in">
      <div className="container">
        <AuthGate>
          {(session) => (
            <>
              <div className="panel page-hero">
                <p className="eyebrow">Workspace</p>
                <h1>Atherion User Workspace</h1>
                <p className="muted">
                  Signed in as {session.email} ({session.role})
                </p>
                {session.role === "admin" ? (
                  <Link className="btn primary" href="/admin">
                    Open Admin Dashboard
                  </Link>
                ) : null}
              </div>
              <div className="kpi-grid" style={{ marginTop: 16 }}>
                <div className="kpi-card">
                  <p className="muted">Availability</p>
                  <p className="kpi-value">99.95%</p>
                  <p className="muted">health checks passing across active regions</p>
                </div>
                <div className="kpi-card">
                  <p className="muted">Policy Gates</p>
                  <p className="kpi-value">3 Active</p>
                  <p className="muted">deterministic, safety, and auth route controls</p>
                </div>
                <div className="kpi-card">
                  <p className="muted">Current Session</p>
                  <p className="kpi-value">{session.authMethod || "google"}</p>
                  <p className="muted">last sign in {new Date(session.loginAt).toLocaleString()}</p>
                </div>
              </div>
            </>
          )}
        </AuthGate>
      </div>
    </main>
  );
}
