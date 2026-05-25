import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  outputFileTracingRoot: path.join(__dirname, ".."),
  async rewrites() {
    return {
      beforeFiles: [
        {
          source: "/",
          destination: "/nrsi-ghost-in-the-machine.html"
        }
      ]
    };
  },
  async headers() {
    return [
      {
        source: "/",
        headers: [
          {
            key: "Cache-Control",
            value: "no-store, no-cache, must-revalidate"
          }
        ]
      }
    ];
  }
};

export default nextConfig;
