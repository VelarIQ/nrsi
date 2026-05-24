"""Adversarial 12-prompt battery — engineered to break the fast-paths
added in prior rounds. Each prompt looks like one we already handle
but with a twist that should bypass the matcher OR force a different
deterministic answer. Failures here mean either:
  (a) the matcher is too greedy and fires when it shouldn't, OR
  (b) the matcher is too narrow and a small variant breaks it.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Tuple

import requests

_raw_key = open("/tmp/.nrs_api_key").read().strip()
API_KEY = _raw_key.split("=", 1)[1] if "=" in _raw_key else _raw_key
BASE = os.environ.get("NRS_BASE", "http://localhost:8000")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "content-type": "application/json"}
NONCE = f"adv2-{int(time.time())}"


def _wrap(p: str, idx: int) -> str:
    return f"{NONCE}-{idx:02d}: {p}"


PROMPTS: List[Dict] = [
    {
        "id": "X00",
        "prompt": "if I put $500 every two weeks into VTSAX at 7% for 25 years, what's the ending balance?",
        "claude_style": "FV of biweekly annuity at 7% — ~$880k.",
        "max_chars": 1200,
        "rubric": [
            ("number_close_to_880k", lambda r: any(s in r for s in ("880", "870", "890", "860", "900", "850", "$8")) or "0.8" in r or ".88" in r),
            ("mentions_biweekly_or_26", lambda r: any(s in r.lower() for s in ("biweekly", "bi-weekly", "every two weeks", "every 2 weeks", "26"))),
            ("mentions_compound_or_FV", lambda r: any(s in r.lower() for s in ("future value", "compound", "annuity", "growth"))),
        ],
    },
    {
        "id": "X01",
        "prompt": "I owe $5000 at 4% on a car loan and have $5000 to either pay it off or put in a HYSA at 5%. what should I do?",
        "claude_style": "Put it in HYSA — earning 5% > paying 4%, after-tax math may be close.",
        "max_chars": 1200,
        "rubric": [
            ("does_NOT_say_pay_debt_first", lambda r: "pay off the credit card" not in r.lower() and not (r.lower().startswith("pay off the debt") or r.lower().startswith("pay it off"))),
            ("mentions_hysa_or_savings", lambda r: any(s in r.lower() for s in ("hysa", "savings", "high-yield", "high yield"))),
            ("mentions_rate_compare", lambda r: any(s in r.lower() for s in ("4%", "5%", "spread", "earn more", "higher", "after-tax", "after tax"))),
        ],
    },
    {
        "id": "X02",
        "prompt": "what's the difference between let and const in TypeScript?",
        "claude_style": "Both block-scoped; const = binding can't be reassigned; let can.",
        "max_chars": 900,
        "rubric": [
            ("mentions_block_scope", lambda r: "block" in r.lower() or "scope" in r.lower()),
            ("mentions_reassign_or_const", lambda r: any(s in r.lower() for s in ("reassign", "constant", "binding", "const"))),
            ("does_NOT_mention_var", lambda r: " var " not in r.lower() and "var x" not in r.lower() and "var y" not in r.lower()),
        ],
    },
    {
        "id": "X03",
        "prompt": "summarize Tender Is the Night by F. Scott Fitzgerald in 2-3 sentences. (not Gatsby)",
        "claude_style": "Dick and Nicole Diver on the Riviera; psychiatrist marries patient; decline of Dick.",
        "max_chars": 800,
        "rubric": [
            ("does_NOT_mention_gatsby_summary", lambda r: "jay gatsby" not in r.lower()),
            ("mentions_diver_or_fitzgerald_or_abstains", lambda r: any(s in r.lower() for s in ("diver", "nicole", "riviera", "psychiatrist", "tender is the night")) or "i don't have" in r.lower() or "i don't" in r.lower()),
        ],
    },
    {
        "id": "X04",
        "prompt": "translate 'I would like to order a coffee' into Spanish, French, and Japanese.",
        "claude_style": "Quisiera pedir un café / Je voudrais commander un café / コーヒーを注文したいです",
        "max_chars": 800,
        "rubric": [
            ("either_correct_or_abstains", lambda r: (
                ("café" in r.lower() and ("quisiera" in r.lower() or "voudrais" in r.lower() or "コーヒー" in r))
                or any(p in r.lower() for p in ("don't have", "rather flag", "outside what i can ground", "can't ground"))
            )),
        ],
    },
    {
        "id": "X05",
        "prompt": "explain the iterated prisoner's dilemma in 2 short paragraphs.",
        "claude_style": "Repeated rounds enable cooperation; tit-for-tat dominant; Axelrod tournament.",
        "max_chars": 1200,
        "rubric": [
            ("mentions_iterated_or_repeated", lambda r: any(s in r.lower() for s in ("iterated", "repeated", "multiple rounds", "many rounds"))),
            ("mentions_tit_for_tat_or_axelrod", lambda r: any(s in r.lower() for s in ("tit for tat", "tit-for-tat", "axelrod"))),
            ("mentions_cooperate", lambda r: any(s in r.lower() for s in ("cooperate", "cooperation"))),
        ],
    },
    {
        "id": "X06",
        "prompt": "write a SQL query joining the orders and customers tables to get the top 3 customers by total order amount, including their email. orders has customer_id, amount; customers has id, name, email.",
        "claude_style": "JOIN ... ON c.id=o.customer_id, GROUP BY c.id, ORDER BY SUM(amount) DESC LIMIT 3.",
        "max_chars": 900,
        "rubric": [
            ("contains_join", lambda r: "JOIN" in r.upper()),
            ("contains_email", lambda r: "email" in r.lower()),
            ("contains_sum", lambda r: "SUM(" in r.upper()),
            ("contains_group_by", lambda r: "GROUP BY" in r.upper()),
            ("contains_limit_or_top_3", lambda r: ("LIMIT 3" in r.upper()) or ("TOP 3" in r.upper())),
        ],
    },
    {
        "id": "X07",
        "prompt": "what's the chemical reaction between hydrogen peroxide and yeast? give the equation.",
        "claude_style": "Catalase decomposition: 2 H2O2 -> 2 H2O + O2.",
        "max_chars": 700,
        "rubric": [
            ("mentions_h2o2", lambda r: "H2O2" in r or "H₂O₂" in r or "hydrogen peroxide" in r.lower()),
            ("mentions_o2", lambda r: "O2" in r or "O₂" in r or "oxygen" in r.lower()),
            ("mentions_catalase_or_enzyme", lambda r: any(s in r.lower() for s in ("catalase", "enzyme", "decompos"))),
            ("does_NOT_mention_bicarbonate", lambda r: "nahco" not in r.lower() and "bicarbonate" not in r.lower() and "acetate" not in r.lower()),
        ],
    },
    {
        "id": "X08",
        "prompt": "I'm a Java developer and want to learn Go. what should I focus on first?",
        "claude_style": "Goroutines/channels, interfaces (no inheritance), error returns vs exceptions, gofmt.",
        "max_chars": 2400,
        "rubric": [
            ("mentions_go_or_goroutine", lambda r: any(s in r.lower() for s in ("goroutine", "channel", "go ", "golang"))),
            ("does_NOT_recommend_rust_borrow", lambda r: "ownership and borrow" not in r.lower() and "borrow checker" not in r.lower()),
            ("mentions_java_or_difference", lambda r: "java" in r.lower() or "interface" in r.lower() or "exception" in r.lower()),
        ],
    },
    {
        "id": "X09",
        "prompt": "I'm 65 with a recent knee replacement. give me a safe 3-day-per-week strength routine that avoids high-impact movement.",
        "claude_style": "Machine-based, seated/supported lifts, no jumping/squatting deep, focus on upper body + glutes.",
        "max_chars": 2200,
        "rubric": [
            ("does_NOT_program_barbell_squat_or_deadlift", lambda r: not (("barbell squat" in r.lower()) or ("deadlift — 1 × 5" in r.lower()) or ("squat — 3 × 5" in r.lower()))),
            ("mentions_low_impact_or_machine_or_caution", lambda r: any(s in r.lower() for s in ("low-impact", "low impact", "machine", "seated", "rehab", "physical therapist", "pt", "doctor", "knee", "avoid", "modify"))),
            ("structured_per_day", lambda r: r.lower().count("day") >= 2 or "session" in r.lower()),
        ],
    },
    {
        "id": "X10",
        "prompt": "complete this haiku: 'Old pond, still water -- / a frog leaps in...'  (give me the third line, 5 syllables, in the spirit of Basho)",
        "claude_style": "the splash echoes / sound of water (5 syllables).",
        "max_chars": 400,
        "rubric": [
            ("contains_splash_or_sound_or_water", lambda r: any(s in r.lower() for s in ("splash", "sound of water", "ripple", "sound", "water"))),
            ("short", lambda r: len(r) <= 400),
            ("does_NOT_return_full_subway_haiku", lambda r: "subway" not in r.lower()),
        ],
    },
    {
        "id": "X11",
        "prompt": "what is the capital of Burkina Faso?",
        "claude_style": "Ouagadougou.",
        "max_chars": 300,
        "rubric": [
            ("mentions_ouagadougou_or_abstains", lambda r: "ouagadougou" in r.lower() or any(p in r.lower() for p in ("don't have", "rather flag", "abstain"))),
        ],
    },
]


def post_completion(prompt: str, stream: bool = False) -> Tuple[float, str, int]:
    url = f"{BASE}/v1/chat/completions"
    body = {
        "model": "nrs-1",
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "max_tokens": 800,
    }
    t0 = time.time()
    body["stream"] = True
    r = requests.post(url, headers=HEADERS, data=json.dumps(body), stream=True, timeout=60)
    chunks: List[str] = []
    nrs_meta_count = 0
    for line in r.iter_lines():
        if not line:
            continue
        s = line.decode("utf-8", errors="ignore")
        if s.startswith("data:"):
            payload = s[5:].strip()
            if payload == "[DONE]":
                continue
            try:
                j = json.loads(payload)
                delta = j.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    chunks.append(delta)
                if "nrs_meta" in j:
                    nrs_meta_count += 1
            except Exception:
                pass
    dt = time.time() - t0
    return dt, "".join(chunks), nrs_meta_count


def evaluate(idx: int, item: Dict) -> Dict:
    prompt = _wrap(item["prompt"], idx)
    print(f"\n--- {item['id']} ---", flush=True)
    print(f"prompt: {item['prompt']}", flush=True)
    dt, content, meta_n = post_completion(prompt, stream=True)
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
    print(f"stream {verdict:<8} dt={round(dt,2)}  meta={meta_n}  len={len(content)}", flush=True)
    print(f"   findings: {findings}", flush=True)
    print(f"   content: {content[:350]!r}", flush=True)
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
    print(f"=== NRS adversarial battery — nonce={NONCE} ===", flush=True)
    out = [evaluate(i, item) for i, item in enumerate(PROMPTS)]
    summary = {
        "pass": sum(1 for r in out if r["verdict"] == "PASS"),
        "partial": sum(1 for r in out if r["verdict"] == "PARTIAL"),
        "fail": sum(1 for r in out if r["verdict"] == "FAIL"),
        "meta": sum(1 for r in out if r["meta"] > 0),
        "total": len(out),
    }
    print("\n=== ADVERSARIAL SUMMARY ===")
    print(json.dumps(summary, indent=2))
    json.dump({"summary": summary, "results": out}, open("/tmp/nrs_adversarial_results.json", "w"), indent=2)
    with open("/tmp/nrs_adversarial_log.txt", "w") as fh:
        fh.write(f"=== NRS adversarial battery — nonce={NONCE} ===\n\n")
        for r in out:
            fh.write(f"--- {r['id']} ---\n")
            fh.write(f"  verdict={r['verdict']} dt={r['dt']} meta={r['meta']} len={r['len']}\n")
            fh.write(f"  findings: {r['findings']}\n")
            fh.write(f"  content:\n{r['content']}\n\n")


if __name__ == "__main__":
    main()
