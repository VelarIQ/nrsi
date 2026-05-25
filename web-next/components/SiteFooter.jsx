export default function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="container footer-inner">
        <p>© {new Date().getFullYear()} Atherion Group (VelarIQ Inc.). All rights reserved.</p>
        <p>NRSI | AI/ML language runtime, protocol intelligence, and enterprise software orchestration.</p>
      </div>
    </footer>
  );
}
