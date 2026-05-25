export default function PlatformPage() {
  return (
    <main className="section fade-in">
      <div className="container page-shell">
        <section className="panel page-hero">
          <p className="eyebrow">Runtime Architecture</p>
          <h1>Platform surfaces built for auditable reasoning and controlled execution.</h1>
          <p className="muted">
            NRSI combines traffic routing, policy gates, conversation memory, and service-level observability into one deployable
            runtime stack.
          </p>
        </section>

        <section className="stats-row">
          <div className="stat-pill">
            <p className="muted">Control Planes</p>
            <h4>Policy + Mode + Safety</h4>
          </div>
          <div className="stat-pill">
            <p className="muted">Execution Planes</p>
            <h4>Code + Live + Deterministic</h4>
          </div>
          <div className="stat-pill">
            <p className="muted">Data Planes</p>
            <h4>History + Metrics + Proof</h4>
          </div>
        </section>

        <section className="content-grid-2">
          <article className="panel">
            <h3>Core capabilities</h3>
            <ul className="list">
              <li>Mode-aware routing with deterministic fallbacks</li>
              <li>Auth-bound conversation surfaces with role access</li>
              <li>SLO-backed deployment on managed Cloud Run</li>
              <li>Alerting and incident hooks for ops response</li>
            </ul>
          </article>
          <article className="panel">
            <h3>Where teams use it</h3>
            <ul className="list">
              <li>Mission-critical assistants that require traceability</li>
              <li>Enterprise copilots with policy and compliance boundaries</li>
              <li>Research stacks transitioning from prototype to production</li>
              <li>Hybrid deterministic/probabilistic chat workflows</li>
            </ul>
          </article>
        </section>
      </div>
    </main>
  );
}
