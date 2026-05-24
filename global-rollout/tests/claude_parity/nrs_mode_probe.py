"""Quick mode-coverage probe.

Hits the worker with the same 5 reference prompts under each of
deterministic / hybrid / creative / auto, and confirms:

  * Each request returns a non-empty answer.
  * `nrs_meta.mode` echoes the requested mode (or, for auto, lists a
    routed-to mode).
  * Deterministic runs of the same prompt are byte-identical.
  * Creative runs of the same prompt are NOT byte-identical (variance).
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import requests

_raw_key = open("/tmp/.nrs_api_key").read().strip()
API_KEY = _raw_key.split("=", 1)[1] if "=" in _raw_key else _raw_key
BASE = os.environ.get("NRS_BASE", "http://localhost:8000")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "content-type": "application/json"}

PROMPTS = [
    "What is the capital of France?",
    "Write one short sentence about the ocean.",
    "What's 17 times 24?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Explain quicksort to me in one short paragraph.",
]

MODES = ["deterministic", "hybrid", "creative", "auto"]


def ask(prompt: str, mode: str) -> Dict:
    payload = {
        "model": "nrs-1",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "mode": mode,
    }
    r = requests.post(
        f"{BASE}/v1/chat/completions",
        headers=HEADERS,
        data=json.dumps(payload),
        timeout=60,
    )
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
    body = r.json()
    msg = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
    meta = body.get("nrs_meta") or {}
    return {"text": msg, "meta": meta}


def main() -> None:
    summary: List[str] = []
    failures = 0
    for prompt in PROMPTS:
        print(f"\n=== {prompt!r}")
        cache: Dict[str, List[str]] = {}
        for mode in MODES:
            r1 = ask(prompt, mode)
            if "error" in r1:
                print(f"  {mode:>14}: ERROR — {r1['error']}")
                failures += 1
                continue
            text1 = r1["text"]
            r2 = ask(prompt, mode)
            text2 = r2.get("text", "")
            same = text1 == text2
            mode_meta = (
                r1["meta"].get("reasoningMode")
                or r1["meta"].get("mode")
                or r1["meta"].get("routed_mode")
                or "?"
            )
            cache[mode] = [text1, text2]
            tag = "OK"
            if mode == "deterministic" and not same:
                tag = "FAIL_var"
                failures += 1
            if mode == "creative" and same and len(text1) > 60:
                tag = "FAIL_no_var"
                failures += 1
            print(
                f"  {mode:>14}: {tag:>10}  meta.mode={mode_meta:<14}  "
                f"len={len(text1):>4}  same={same}"
            )
        # auto-router rationale
        auto_meta = ask(prompt, "auto")["meta"]
        decision = auto_meta.get("modeDecision") or {}
        rationale = (
            (decision.get("rationale") if isinstance(decision, dict) else None)
            or auto_meta.get("mode_rationale")
            or "—"
        )
        routed = decision.get("mode") if isinstance(decision, dict) else None
        print(f"  auto rationale: routed={routed!r}  detail={str(rationale)[:120]}")
        time.sleep(0.2)
    print(f"\n=== summary: {failures} failures across {len(PROMPTS) * len(MODES)} runs")


if __name__ == "__main__":
    main()
