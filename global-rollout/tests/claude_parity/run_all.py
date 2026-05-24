#!/usr/bin/env python3
"""Run every Claude-parity battery + probe sequentially.

Designed to be invoked from CI after `docker compose up -d` has been
run. Honours the same env vars the individual scripts expect:

    NRS_BASE        platform-api base URL (default: http://localhost:8000)
    NRS_API_KEY     /v1/chat/completions key (battery scripts)
    NRS_JWT_PATH    file containing a JWT for the autonomous probe
                    (defaults to /tmp/.nrs_jwt)

Exits non-zero if any battery script returns non-zero. Each script
is run as a subprocess so a partial failure still surfaces the rest
of the diagnostics in the CI log.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

API_KEY_FILE = Path("/tmp/.nrs_api_key")

BATTERIES = [
    "nrs_parity_battery.py",
    "nrs_stretch_battery.py",
    "nrs_advanced_battery.py",
    "nrs_adversarial_battery.py",
    "nrs_multiturn_battery.py",
    "nrs_extended_battery.py",
    "nrs_mode_probe.py",
    "nrs_autonomous_probe.py",
    "audit_default_impulses.py",
    "nrs_human_layer_probe.py",
]


def main() -> int:
    api_key = os.environ.get("NRS_API_KEY", "").strip()
    if api_key and not API_KEY_FILE.exists():
        API_KEY_FILE.write_text(f"NRS_API_KEY={api_key}\n", encoding="utf-8")

    here = Path(__file__).resolve().parent
    failures: list[str] = []
    for script in BATTERIES:
        path = here / script
        if not path.exists():
            print(f"SKIP {script} (missing)")
            continue
        print(f"\n========== {script} ==========")
        rc = subprocess.call([sys.executable, str(path)], env=os.environ.copy())
        if rc != 0:
            failures.append(f"{script} (exit {rc})")
    if failures:
        print("\nFAIL — batteries that did not pass:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll Claude-parity batteries passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
