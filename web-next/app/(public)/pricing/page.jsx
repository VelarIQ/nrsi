export default function PricingPage() {
  return (
    <main className="section fade-in">
      <div className="container page-shell">
        <section className="panel page-hero">
          <p className="eyebrow">Pricing</p>
          <h1>Start free, scale with governance and dedicated support.</h1>
          <p className="muted">
            Every tier includes core runtime access. Commercial tiers add compliance controls, support SLAs, and enterprise rollout
            services.
          </p>
        </section>

        <section className="pricing-grid">
          <article className="panel pricing-card">
            <span className="badge">Community</span>
            <h3>Open Builder</h3>
            <p className="price">$0</p>
            <ul className="list">
              <li>OSS runtime and docs</li>
              <li>Public issue support</li>
              <li>Baseline observability hooks</li>
            </ul>
          </article>
          <article className="panel pricing-card featured">
            <span className="badge">Most Selected</span>
            <h3>Growth</h3>
            <p className="price">$499/mo</p>
            <ul className="list">
              <li>Managed deployment templates</li>
              <li>Priority support channel</li>
              <li>Policy and routing tuning sessions</li>
            </ul>
          </article>
          <article className="panel pricing-card">
            <span className="badge">Enterprise</span>
            <h3>Platform Scale</h3>
            <p className="price">Custom</p>
            <ul className="list">
              <li>Dedicated architecture advisory</li>
              <li>Security and compliance alignment</li>
              <li>SLA-backed ops and incident pathways</li>
            </ul>
          </article>
        </section>
      </div>
    </main>
  );
}
