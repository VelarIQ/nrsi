import "./globals.css";
import RouteStage from "../components/RouteStage";

export const metadata = {
  title: "NRSI",
  description: "Neuromorphic runtime and protocol intelligence platform."
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <RouteStage>{children}</RouteStage>
      </body>
    </html>
  );
}
