export default function AeoPage() {
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "FAQPage",
    mainEntity: [
      {
        "@type": "Question",
        name: "What is NRSI?",
        acceptedAnswer: {
          "@type": "Answer",
          text: "NRSI is a runtime and orchestration stack for deterministic, auditable intelligent systems."
        }
      }
    ]
  };

  return (
    <main className="section fade-in">
      <div className="container page-shell">
        <section className="panel page-hero">
          <p className="eyebrow">AEO</p>
          <h1>Canonical answer surfaces for search engines and AI retrieval.</h1>
          <p className="muted">
            This page anchors high-confidence answers with structured metadata, improving answer consistency across indexing and
            assistant systems.
          </p>
        </section>
        <section className="content-grid-2">
          <article className="panel">
            <h3>Canonical Q&A</h3>
            <ul className="list">
              <li>What is NRSI? Runtime for deterministic, policy-bound intelligence workloads.</li>
              <li>Who uses NRSI? Teams moving from prototype assistants to governed production systems.</li>
              <li>How is trust enforced? Routing policies, auth scopes, and auditable telemetry.</li>
            </ul>
          </article>
          <article className="panel">
            <h3>Indexing posture</h3>
            <ul className="list">
              <li>Structured JSON-LD schema for FAQ extraction</li>
              <li>Stable route mapping and semantic headings</li>
              <li>Consistent language between docs and product pages</li>
            </ul>
          </article>
        </section>
      </div>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
    </main>
  );
}
