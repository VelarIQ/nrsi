export default function EnterprisePage() {
  return (
    <main className="section fade-in">
      <div className="container page-shell">
        <section className="panel page-hero">
          <p className="eyebrow">Enterprise</p>
          <h1>Production adoption with governance, reliability, and clear accountability.</h1>
          <p className="muted">
            Enterprise programs include architecture onboarding, staged rollout controls, and operator runbooks aligned to your risk
            profile.
          </p>
        </section>

        <section className="content-grid-2">
          <article className="panel">
            <h3>Program tracks</h3>
            <ul className="list">
              <li>30-day pilot with production shadow traffic</li>
              <li>Control-plane hardening and policy calibration</li>
              <li>Org-wide enablement and ownership transfer</li>
            </ul>
          </article>
          <article className="panel">
            <h3>What’s included</h3>
            <ul className="list">
              <li>Dedicated support lane and escalation protocol</li>
              <li>Incident review templates and RACI model</li>
              <li>Security and architecture workshops</li>
            </ul>
          </article>
        </section>
      </div>
    </main>
  );
}
