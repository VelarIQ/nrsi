import SiteHeader from "../../components/SiteHeader";
import SiteFooter from "../../components/SiteFooter";
import ThemeSync from "../../components/ThemeSync";
import ScrollRevealInit from "../../components/ScrollRevealInit";

export default function PublicLayout({ children }) {
  return (
    <>
      <ScrollRevealInit />
      <ThemeSync />
      <SiteHeader />
      {children}
      <SiteFooter />
    </>
  );
}
