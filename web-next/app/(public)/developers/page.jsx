export default function DevelopersPage() {
  return (
    <main className="section fade-in">
      <div className="container page-shell">
        <section className="panel page-hero">
          <p className="eyebrow">Developers</p>
          <h1>Ship quickly with policy-aware foundations and production guardrails.</h1>
          <p className="muted">
            Use the runtime APIs, auth patterns, and deployment scripts to move from proof-of-concept to enterprise-ready delivery.
          </p>
        </section>

        <section className="content-grid-2">
          <article className="panel">
            <h3>Quickstart flow</h3>
            <div className="timeline">
              <div className="timeline-step">
                <strong>1. Clone + configure</strong>
                <p className="muted">Set env, auth providers, and project-level runtime settings.</p>
              </div>
              <div className="timeline-step">
                <strong>2. Select mode routing</strong>
                <p className="muted">Define deterministic vs exploratory pathways per route.</p>
              </div>
              <div className="timeline-step">
                <strong>3. Deploy + monitor</strong>
                <p className="muted">Ship to Cloud Run and attach alert policies before launch.</p>
              </div>
            </div>
          </article>
          <article className="panel">
            <h3>Reference surfaces</h3>
            <table className="table">
              <thead>
                <tr>
                  <th>Surface</th>
                  <th>Purpose</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Runtime API</td>
                  <td>Message routing, policy enforcement, metadata</td>
                </tr>
                <tr>
                  <td>Auth layer</td>
                  <td>Workspace SSO and role segmentation</td>
                </tr>
                <tr>
                  <td>Ops scripts</td>
                  <td>Deploy, alert provisioning, incident hooks</td>
                </tr>
              </tbody>
            </table>
          </article>
        </section>
      </div>
    </main>
  );
}
