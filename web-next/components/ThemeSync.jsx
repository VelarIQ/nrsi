"use client";

import { useEffect } from "react";

const STORAGE_KEY = "nrsi_hero_mode";
const MODES = new Set(["knot", "orbit", "pulse"]);

function applyMode(mode) {
  const safeMode = MODES.has(mode) ? mode : "knot";
  document.body.setAttribute("data-hero-mode", safeMode);
}

export default function ThemeSync() {
  useEffect(() => {
    applyMode(localStorage.getItem(STORAGE_KEY) || "knot");

    const onMode = (event) => {
      const mode = event?.detail?.mode;
      if (mode) applyMode(mode);
    };

    window.addEventListener("nrsi:hero-mode", onMode);
    return () => window.removeEventListener("nrsi:hero-mode", onMode);
  }, []);

  return null;
}
