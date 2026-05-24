"""Advanced 12-prompt battery — prompts not pre-baked into NRS
fast-paths. Mirrors how Claude.ai would handle real-world questions."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Callable, Dict, List, Tuple

import requests

_raw_key = open("/tmp/.nrs_api_key").read().strip()
API_KEY = _raw_key.split("=", 1)[1] if "=" in _raw_key else _raw_key
BASE = os.environ.get("NRS_BASE", "http://localhost:8000")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "content-type": "application/json"}

NONCE = f"adv-{int(time.time())}"


def _wrap(p: str, idx: int) -> str:
    return f"{NONCE}-{idx:02d}: {p}"


PROMPTS: List[Dict] = [
    {
        "id": "A00",
        "prompt": "if I invest $1000 every month into an S&P 500 index fund averaging 8% annual return, how much will I have in 30 years?",
        "claude_style": "Future value of annuity calc, ~$1.49M.",
        "max_chars": 900,
        "rubric": [
            ("number_in_million_range", lambda r: re.search(r"\$?1\.[34][0-9]?\s*(?:million|m\b|M\b)|\$1,?[34][0-9]{2},?\d{3}", r) is not None or "1.49" in r or "1.4" in r or "1.5" in r),
            ("mentions_compound_or_FV", lambda r: any(k in r.lower() for k in ("compound", "future value", "annuity", "growth"))),
        ],
    },
    {
        "id": "A01",
        "prompt": "what's the difference between let, const, and var in JavaScript?",
        "claude_style": "Block-scope vs function-scope, hoisting, reassignability.",
        "max_chars": 900,
        "rubric": [
            ("mentions_block_scope", lambda r: "block" in r.lower() or "scope" in r.lower()),
            ("mentions_hoisting_or_var", lambda r: "hoist" in r.lower() or "var" in r.lower()),
            ("mentions_const_immutable", lambda r: "const" in r.lower()),
        ],
    },
    {
        "id": "A02",
        "prompt": "i make 60k and have $20k in credit card debt at 22% interest. should i invest in stocks or pay off the debt first?",
        "claude_style": "Pay off CC debt first — guaranteed 22% return beats market.",
        "max_chars": 1400,
        "rubric": [
            ("recommends_pay_debt", lambda r: any(k in r.lower() for k in ("pay off", "pay down", "pay it off", "debt first"))),
            ("mentions_interest_rate", lambda r: "22" in r or "interest" in r.lower()),
            ("mentions_market_or_return", lambda r: any(k in r.lower() for k in ("market", "return", "stocks", "s&p", "average"))),
        ],
    },
    {
        "id": "A03",
        "prompt": "summarize the central conflict in 'The Great Gatsby' in 2-3 sentences.",
        "claude_style": "Gatsby's pursuit of Daisy and the corruption of the American Dream.",
        "max_chars": 750,
        "rubric": [
            ("mentions_gatsby", lambda r: "gatsby" in r.lower()),
            ("mentions_daisy_or_dream", lambda r: any(k in r.lower() for k in ("daisy", "american dream", "wealth", "love"))),
        ],
    },
    {
        "id": "A04",
        "prompt": "translate 'where is the bathroom' into Spanish, French, and Japanese.",
        "claude_style": "Donde está el baño / Où est la salle de bain / トイレはどこですか",
        "max_chars": 500,
        "rubric": [
            ("spanish", lambda r: "baño" in r.lower() or "donde" in r.lower() or "dónde" in r.lower()),
            ("french", lambda r: "où" in r.lower() or "salle de bain" in r.lower() or "toilette" in r.lower()),
            ("japanese", lambda r: "トイレ" in r or "どこ" in r or "wa doko" in r.lower() or "doko" in r.lower()),
        ],
    },
    {
        "id": "A05",
        "prompt": "what's a good first book to read about Stoicism?",
        "claude_style": "Meditations by Marcus Aurelius, or Letters from a Stoic by Seneca.",
        "max_chars": 800,
        "rubric": [
            ("mentions_meditations_or_seneca", lambda r: any(k in r.lower() for k in ("meditations", "marcus aurelius", "seneca", "epictetus", "letters from a stoic"))),
        ],
    },
    {
        "id": "A06",
        "prompt": "write a SQL query to find the top 3 customers by total order amount in the last 30 days from a table 'orders' with columns customer_id, order_date, amount.",
        "claude_style": "SELECT customer_id, SUM(amount) ... GROUP BY ... ORDER BY DESC LIMIT 3.",
        "max_chars": 700,
        "rubric": [
            ("contains_select", lambda r: "SELECT" in r.upper()),
            ("contains_sum", lambda r: "SUM(" in r.upper()),
            ("contains_group_by", lambda r: "GROUP BY" in r.upper()),
            ("contains_limit_3", lambda r: ("LIMIT 3" in r.upper()) or ("TOP 3" in r.upper())),
        ],
    },
    {
        "id": "A07",
        "prompt": "explain the prisoner's dilemma in 2 short paragraphs.",
        "claude_style": "Two suspects, dominant strategy = defect, Pareto worse.",
        "max_chars": 900,
        "rubric": [
            ("mentions_dilemma", lambda r: "prisoner" in r.lower() or "dilemma" in r.lower()),
            ("mentions_defect_or_cooperate", lambda r: any(k in r.lower() for k in ("defect", "cooperate", "betray", "confess", "stay silent"))),
            ("not_too_long", lambda r: len(r) < 1100),
        ],
    },
    {
        "id": "A08",
        "prompt": "i'm a Python developer and want to learn Rust. what should I focus on first?",
        "claude_style": "Ownership/borrowing model, lifetimes, traits vs classes, Cargo.",
        "max_chars": 2000,
        "rubric": [
            ("mentions_ownership_or_borrow", lambda r: any(k in r.lower() for k in ("ownership", "borrow", "borrowing", "lifetime"))),
            ("mentions_rust_or_python_diff", lambda r: any(k in r.lower() for k in ("rust", "python", "memory", "compiler", "type system"))),
        ],
    },
    {
        "id": "A09",
        "prompt": "what's the chemical reaction when you mix baking soda and vinegar? give the equation.",
        "claude_style": "NaHCO3 + CH3COOH → CO2 + H2O + sodium acetate.",
        "max_chars": 800,
        "rubric": [
            ("mentions_co2", lambda r: "CO2" in r or "CO₂" in r or "carbon dioxide" in r.lower()),
            ("mentions_water", lambda r: "H2O" in r or "H₂O" in r or "water" in r.lower()),
            ("mentions_sodium_or_acetate", lambda r: any(k in r.lower() for k in ("sodium", "acetate", "nahco", "naoac"))),
        ],
    },
    {
        "id": "A10",
        "prompt": "i'm 32, healthy, and want to start strength training. give me a 3-day-per-week beginner routine.",
        "claude_style": "Push/Pull/Legs or Full-body 3x/week, ~3 sets x 5-8 reps compound lifts.",
        "max_chars": 1400,
        "rubric": [
            ("mentions_three_days", lambda r: "3" in r or "three" in r.lower()),
            ("mentions_compound_lift", lambda r: any(k in r.lower() for k in ("squat", "deadlift", "bench", "press", "row", "pull-up", "pullup", "pull up"))),
            ("structured_per_day", lambda r: r.lower().count("day") >= 2 or "monday" in r.lower() or "session" in r.lower()),
        ],
    },
    {
        "id": "A11",
        "prompt": "explain quantum entanglement in plain English in one paragraph.",
        "claude_style": "Two particles linked so measuring one instantly affects the other regardless of distance.",
        "max_chars": 900,
        "rubric": [
            ("mentions_particle_or_pair", lambda r: any(k in r.lower() for k in ("particle", "pair", "photon", "electron"))),
            ("mentions_correlation_or_distance", lambda r: any(k in r.lower() for k in ("correlated", "linked", "regardless of distance", "no matter how far", "instantly"))),
            ("not_too_long", lambda r: len(r) < 900),
        ],
    },
]


def post_completion(prompt: str, stream: bool = False) -> Tuple[float, str, List[str], int]:
    url = f"{BASE}/v1/chat/completions"
    body = {
        "model": "nrs-1",
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "max_tokens": 800,
    }
    t0 = time.time()
    if not stream:
        r = requests.post(url, headers=HEADERS, data=json.dumps(body), timeout=60)
        dt = time.time() - t0
        try:
            j = r.json()
            content = j["choices"][0]["message"]["content"]
        except Exception:
            content = r.text
        return dt, content, [], 1
    body["stream"] = True
    r = requests.post(url, headers=HEADERS, data=json.dumps(body), stream=True, timeout=60)
    chunks: List[str] = []
    events: List[str] = []
    nrs_meta_count = 0
    for line in r.iter_lines():
        if not line:
            continue
        s = line.decode("utf-8", errors="ignore")
        if s.startswith("event:"):
            ev = s.split(":", 1)[1].strip()
            events.append(ev)
            if ev == "nrs_meta":
                nrs_meta_count += 1
        elif s.startswith("data:"):
            payload = s[5:].strip()
            if payload == "[DONE]":
                continue
            try:
                j = json.loads(payload)
                delta = j.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    chunks.append(delta)
            except Exception:
                pass
    dt = time.time() - t0
    return dt, "".join(chunks), events, nrs_meta_count


def evaluate(idx: int, item: Dict) -> Dict:
    prompt = _wrap(item["prompt"], idx)
    print(f"\n--- {item['id']} ---", flush=True)
    print(f"prompt: {item['prompt']}", flush=True)
    dt, content, _evs, meta_n = post_completion(prompt, stream=True)
    findings = []
    pass_count = 0
    for name, fn in item["rubric"]:
        try:
            ok = bool(fn(content))
        except Exception as exc:
            ok = False
            name = f"{name} ERROR:{exc}"
        findings.append(("PASS" if ok else "FAIL") + " " + name)
        if ok:
            pass_count += 1
    too_long = len(content) > item.get("max_chars", 1500)
    if too_long:
        findings.append(f"FAIL too_long({len(content)}>{item['max_chars']})")
    n_rubric = len(item["rubric"])
    verdict = "PASS" if (pass_count == n_rubric and not too_long) else (
        "PARTIAL" if (pass_count >= max(1, n_rubric // 2) and not too_long) else "FAIL"
    )
    print(f"stream {verdict:<8} dt={round(dt,2)}  chunks={len(content.split())}  meta={meta_n}  len={len(content)}", flush=True)
    print(f"   findings: {findings}", flush=True)
    print(f"   content: {content[:300]!r}", flush=True)
    return {
        "id": item["id"],
        "verdict": verdict,
        "findings": findings,
        "meta": meta_n,
        "len": len(content),
        "dt": round(dt, 2),
        "content": content,
    }


def main():
    print(f"=== NRS advanced battery — nonce={NONCE} ===", flush=True)
    out = [evaluate(i, item) for i, item in enumerate(PROMPTS)]
    summary = {
        "pass": sum(1 for r in out if r["verdict"] == "PASS"),
        "partial": sum(1 for r in out if r["verdict"] == "PARTIAL"),
        "fail": sum(1 for r in out if r["verdict"] == "FAIL"),
        "meta": sum(1 for r in out if r["meta"] > 0),
        "total": len(out),
    }
    print("\n=== ADVANCED SUMMARY ===")
    print(json.dumps(summary, indent=2))
    json.dump({"summary": summary, "results": out}, open("/tmp/nrs_advanced_results.json", "w"), indent=2)
    with open("/tmp/nrs_advanced_log.txt", "w") as fh:
        fh.write(f"=== NRS advanced battery — nonce={NONCE} ===\n\n")
        for r in out:
            fh.write(f"--- {r['id']} ---\n")
            fh.write(f"  verdict={r['verdict']} dt={r['dt']} meta={r['meta']} len={r['len']}\n")
            fh.write(f"  findings: {r['findings']}\n")
            fh.write(f"  content:\n{r['content']}\n\n")
    print("Wrote /tmp/nrs_advanced_results.json + /tmp/nrs_advanced_log.txt")


if __name__ == "__main__":
    main()
