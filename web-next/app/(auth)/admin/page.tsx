"use client";

import Link from "next/link";
import AuthGate from "../../../components/AuthGate";

type Session = {
  email: string;
  role: "admin" | "super_admin";
};

export default function AdminPage() {
  return (
    <main className="section fade-in">
      <div className="container">
        <AuthGate requireRole="admin">
          {(session: Session) => (
            <>
              <div className="panel page-hero">
                <p className="eyebrow">Operations</p>
                <h1>Atherion Admin Console</h1>
                <p className="muted">Authorized admin: {session.email}</p>
                <div style={{ marginTop: 12 }}>
                  <Link className="btn" href="/dashboard">
                    Back to Dashboard
                  </Link>
                </div>
              </div>

              <div className="kpi-grid" style={{ marginTop: 16 }}>
                <div className="kpi-card">
                  <p className="muted">Active Policies</p>
                  <p className="kpi-value">3</p>
                  <p className="muted">policy engine gates currently enforced</p>
                </div>
                <div className="kpi-card">
                  <p className="muted">Incidents (24h)</p>
                  <p className="kpi-value">0</p>
                  <p className="muted">no unresolved production incidents in this window</p>
                </div>
                <div className="kpi-card">
                  <p className="muted">Alert Routing</p>
                  <p className="kpi-value">Fallback Email</p>
                  <p className="muted">monitoring channel is active while Resend is paused</p>
                </div>
              </div>
            </>
          )}
        </AuthGate>
      </div>
    </main>
  );
}
