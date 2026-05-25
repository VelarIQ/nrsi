"use client";

import { useEffect } from "react";

export default function ScrollRevealInit() {
  useEffect(() => {
    const nodes = Array.from(document.querySelectorAll(".reveal"));
    if (!nodes.length) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.14, rootMargin: "0px 0px -40px 0px" }
    );

    nodes.forEach((node, idx) => {
      node.style.transitionDelay = `${Math.min(0.44, idx * 0.045)}s`;
      observer.observe(node);
    });

    return () => observer.disconnect();
  }, []);

  return null;
}
