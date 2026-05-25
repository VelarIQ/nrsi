"use client";

import { usePathname } from "next/navigation";

export default function RouteStage({ children }) {
  const pathname = usePathname();

  return (
    <div key={pathname} className="route-stage">
      {children}
    </div>
  );
}
