import Link from "next/link";

export default function SiteHeader() {
  return (
    <header className="site-header">
      <div className="container nav-wrap">
        <Link className="brand" href="/">
          NRSI <span className="brand-dot" /> <span className="brand-sub">Atherion</span>
        </Link>
        <nav className="site-nav">
          <Link href="/platform">Platform</Link>
          <Link href="/pricing">Pricing</Link>
          <Link href="/developers">Developers</Link>
          <Link href="/enterprise">Enterprise</Link>
          <Link href="/aeo">AEO</Link>
          <Link href="/contact">Contact</Link>
          <Link href="/login" className="cta-link">
            Login
          </Link>
        </nav>
      </div>
    </header>
  );
}
