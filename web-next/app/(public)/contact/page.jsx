export default function ContactPage() {
  return (
    <main className="section fade-in">
      <div className="container page-shell">
        <section className="panel page-hero">
          <p className="eyebrow">Contact</p>
          <h1>Bring your use case and we’ll map the right deployment path.</h1>
          <p className="muted">
            Use this channel for enterprise pilots, technical due diligence, and implementation advisory requests.
          </p>
        </section>
        <section className="content-grid-2">
          <article className="panel">
            <h3>Commercial inquiries</h3>
            <p className="muted">leighton@velariq.ai</p>
            <p className="muted">Expect response within one business day.</p>
          </article>
          <article className="panel">
            <h3>What to include</h3>
            <ul className="list">
              <li>Deployment environment (cloud / on-prem / hybrid)</li>
              <li>Expected traffic and SLA targets</li>
              <li>Compliance and policy requirements</li>
            </ul>
          </article>
        </section>
      </div>
    </main>
  );
}
