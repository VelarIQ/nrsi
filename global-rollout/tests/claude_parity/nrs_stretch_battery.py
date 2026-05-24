"""NRS stretch battery — 12 harder prompts, no pre-baked fast-paths.

Tests open-ended generation, multi-turn context, code review, ambiguous
prompts, style adaptation, novel reasoning. Used to honestly assess
claude.ai-parity gaps after the fast-path battery passes 12/12.
"""

import json
import re
import sys
import time
import urllib.request
import urllib.error

API_KEY = open("/tmp/.nrs_api_key").read().strip().split("=", 1)[1]
BASE = "http://localhost:8000/v1/chat/completions"
NONCE = f"stretch-{int(time.time())}"

PROMPTS = [
    {
        "id": "S00",
        "prompt": "i'm trying to decide between renting and buying a house in austin. give me 3 honest tradeoffs.",
        "rubric": [
            ("mentions_renting_or_rent", lambda s: "rent" in s.lower()),
            ("mentions_buying_or_buy", lambda s: "buy" in s.lower() or "ownership" in s.lower() or "owning" in s.lower()),
            ("has_three_distinct_points", lambda s: len(re.findall(r"^\s*(?:[-*]|\d+\.)\s+", s, re.MULTILINE)) >= 3 or s.lower().count("tradeoff") >= 1),
            ("mentions_at_least_one_finance_term", lambda s: any(k in s.lower() for k in ["mortgage", "interest", "appreciation", "equity", "tax", "deduction", "maintenance", "property"])),
        ],
        "max_chars": 2000,
    },
    {
        "id": "S01",
        "prompt": "review this python code for bugs:\n\ndef divide(a, b):\n    result = a / b\n    return result\n\nprint(divide(10, 0))",
        "rubric": [
            ("mentions_zero_division", lambda s: "zero" in s.lower() or "zerodivision" in s.lower() or "/ 0" in s.lower() or "division by 0" in s.lower()),
            ("suggests_fix", lambda s: any(k in s.lower() for k in ["try", "except", "raise", "if b ==", "guard", "check"])),
        ],
        "max_chars": 1500,
    },
    {
        "id": "S02",
        "prompt": "i feel really anxious about an interview tomorrow. any tips?",
        "rubric": [
            ("acknowledges_feeling", lambda s: any(k in s.lower() for k in ["anxious", "anxiety", "nervous", "feeling", "understand", "totally normal", "common"])),
            ("gives_concrete_tip", lambda s: any(k in s.lower() for k in ["sleep", "prepare", "breath", "research", "questions", "practice", "rest", "water", "outfit"])),
            ("not_dismissive", lambda s: not any(k in s.lower() for k in ["just relax", "don't worry", "you'll be fine, trust me"])),
        ],
        "max_chars": 1500,
    },
    {
        "id": "S03",
        "prompt": "write a 4-line poem about waking up to snow",
        "rubric": [
            ("four_lines", lambda s: len([l for l in s.strip().split("\n") if l.strip()]) >= 4),
            ("mentions_snow_or_winter", lambda s: any(k in s.lower() for k in ["snow", "white", "winter", "frost", "ice", "flake"])),
            ("not_haiku_count", lambda s: len([l for l in s.strip().split("\n") if l.strip()]) >= 4),
        ],
        "max_chars": 600,
    },
    {
        "id": "S04",
        "prompt": "what's a good name for a cat that's all black with green eyes?",
        "rubric": [
            ("offers_at_least_three_names", lambda s: len(re.findall(r"\b[A-Z][a-z]{2,}\b", s)) >= 3),
            ("not_too_long", lambda s: len(s) <= 1500),
        ],
        "max_chars": 1500,
    },
    {
        "id": "S05",
        "prompt": "explain like i'm 5: what is gravity?",
        "rubric": [
            ("simple_words", lambda s: not any(w in s.lower() for w in ["spacetime", "general relativity", "tensor", "minkowski", "geodesic"])),
            ("mentions_pulling_or_falling", lambda s: any(k in s.lower() for k in ["pull", "fall", "down", "ground", "stick"])),
            ("short_enough", lambda s: len(s) <= 800),
        ],
        "max_chars": 800,
    },
    {
        "id": "S06",
        "prompt": "i have eggs, butter, and stale bread. what can i make?",
        "rubric": [
            ("suggests_at_least_one_dish", lambda s: any(k in s.lower() for k in ["french toast", "bread pudding", "strata", "frittata", "scramble", "omelette", "panade", "soldiers"])),
            ("uses_actual_ingredients", lambda s: "egg" in s.lower() and ("bread" in s.lower() or "butter" in s.lower())),
        ],
        "max_chars": 1200,
    },
    {
        "id": "S07",
        "prompt": "what's the difference between a metaphor and a simile? give one example of each.",
        "rubric": [
            ("explains_metaphor", lambda s: "metaphor" in s.lower()),
            ("explains_simile", lambda s: "simile" in s.lower()),
            ("mentions_like_or_as", lambda s: ("like" in s.lower() or "as" in s.lower()) and "simile" in s.lower()),
            ("has_examples", lambda s: s.count('"') >= 2 or s.count("'") >= 2 or s.count("—") >= 1),
        ],
        "max_chars": 1500,
    },
    {
        "id": "S08",
        "prompt": "i think the moon landing was faked. convince me it was real.",
        "rubric": [
            ("does_not_agree_with_hoax", lambda s: not any(k in s.lower() for k in ["faked", "staged", "wasn't real", "definitely a hoax", "you're right"])),
            ("provides_evidence", lambda s: any(k in s.lower() for k in ["retroreflector", "moon rock", "lunar sample", "telemetry", "soviet", "tracking", "third party", "lasers", "apollo"])),
            ("substantive", lambda s: len(s) >= 200),
        ],
        "max_chars": 3000,
    },
    {
        "id": "S09",
        "prompt": "convert 75 fahrenheit to celsius. show the formula.",
        "rubric": [
            ("contains_23_or_24", lambda s: any(v in s for v in ["23.8", "23.9", "24", "23"])),
            ("shows_formula", lambda s: "32" in s and ("5/9" in s or "* 5" in s or "/ 1.8" in s or "(F" in s)),
        ],
        "max_chars": 600,
    },
    {
        "id": "S10",
        "prompt": "how do I tell my boss I'm leaving? draft a 2-line resignation message.",
        "rubric": [
            ("two_lines_or_short", lambda s: s.count("\n") <= 6 and len(s) <= 800),
            ("mentions_resignation_or_leaving", lambda s: any(k in s.lower() for k in ["resign", "leaving", "depart", "step down", "move on", "notice"])),
        ],
        "max_chars": 800,
    },
    {
        "id": "S11",
        "prompt": "what comes next: 1, 1, 2, 3, 5, 8, ?",
        "rubric": [
            ("answer_is_13", lambda s: "13" in s),
            ("mentions_fibonacci", lambda s: "fibonacci" in s.lower() or "sum" in s.lower() or "previous two" in s.lower() or "preceding" in s.lower()),
        ],
        "max_chars": 600,
    },
]


def post_completion(prompt, *, stream, timeout=90):
    body = json.dumps({
        "model": "nrs-1.0",
        "messages": [{"role": "user", "content": f"{NONCE}: {prompt}"}],
        "stream": stream,
        "max_tokens": 800,
    }).encode()
    req = urllib.request.Request(
        BASE, data=body, method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "Accept": "text/event-stream" if stream else "application/json",
        },
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        return {"ok": False, "dt_s": round(time.time() - t0, 2), "error": f"HTTP {e.code}", "body": e.read()[:500].decode("utf-8", "replace")}
    except Exception as e:
        return {"ok": False, "dt_s": round(time.time() - t0, 2), "error": str(e)}
    dt = round(time.time() - t0, 2)
    if stream:
        text = raw.decode("utf-8", "replace")
        chunks = 0
        meta_count = 0
        last_meta = None
        parts = []
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]" or not payload:
                continue
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            chunks += 1
            if "nrs_meta" in obj and obj["nrs_meta"]:
                meta_count += 1
                last_meta = obj["nrs_meta"]
            for ch in obj.get("choices", []):
                d = ch.get("delta") or {}
                if "content" in d and d["content"]:
                    parts.append(d["content"])
        return {"ok": True, "dt_s": dt, "chunks": chunks,
                "meta_count": meta_count, "last_meta": last_meta,
                "content": "".join(parts)}
    try:
        obj = json.loads(raw.decode("utf-8", "replace"))
    except Exception as e:
        return {"ok": False, "dt_s": dt, "error": f"json: {e}"}
    content = ""
    for ch in obj.get("choices", []):
        msg = ch.get("message") or {}
        if msg.get("content"):
            content = msg["content"]
            break
    return {"ok": True, "dt_s": dt, "content": content, "nrs_meta": obj.get("nrs_meta")}


def evaluate(spec, content):
    findings = []
    p = f = 0
    for name, fn in spec["rubric"]:
        try:
            ok = bool(fn(content))
        except Exception as e:
            findings.append(f"ERROR rubric {name}: {e}")
            f += 1
            continue
        if ok:
            findings.append(f"PASS {name}")
            p += 1
        else:
            findings.append(f"FAIL {name}")
            f += 1
    if len(content) > spec["max_chars"]:
        findings.append(f"WARN length {len(content)} > {spec['max_chars']}")
    if not content.strip():
        findings.append("FAIL empty")
        f += 1
    overall = "PASS" if f == 0 else ("PARTIAL" if p > 0 else "FAIL")
    return {"overall": overall, "pass": p, "fail": f, "findings": findings}


def main():
    results = {"nonce": NONCE, "results": []}
    log = [f"=== NRS stretch battery — nonce={NONCE} ==="]
    for spec in PROMPTS:
        pid = spec["id"]
        log.append(f"\n--- {pid} ---")
        log.append(f"prompt: {spec['prompt']}")
        st = post_completion(spec["prompt"], stream=True)
        st_eval = evaluate(spec, st.get("content", "")) if st.get("ok") else {"overall": "FAIL", "pass": 0, "fail": len(spec["rubric"]), "findings": ["upstream error"]}
        results["results"].append({
            "id": pid, "prompt": spec["prompt"],
            "stream": st, "st_eval": st_eval,
        })
        log.append(f"stream {st_eval['overall']:7s}  dt={st.get('dt_s')}  chunks={st.get('chunks')}  meta={st.get('meta_count')}  len={len(st.get('content',''))}")
        log.append(f"   findings: {st_eval['findings']}")
        log.append(f"   content: {(st.get('content') or '')[:300]!r}")
    p = sum(1 for r in results["results"] if r["st_eval"]["overall"] == "PASS")
    pa = sum(1 for r in results["results"] if r["st_eval"]["overall"] == "PARTIAL")
    f = sum(1 for r in results["results"] if r["st_eval"]["overall"] == "FAIL")
    meta = sum(1 for r in results["results"] if r.get("stream", {}).get("meta_count", 0) > 0)
    summary = {"pass": p, "partial": pa, "fail": f, "meta": meta, "total": len(PROMPTS)}
    results["summary"] = summary
    log.append(f"\n=== STRETCH SUMMARY ===\n{json.dumps(summary, indent=2)}")
    with open("/tmp/nrs_stretch_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    with open("/tmp/nrs_stretch_log.txt", "w") as fh:
        fh.write("\n".join(log))
    print("\n".join(log[-25:]))
    print(f"\nWrote /tmp/nrs_stretch_results.json + /tmp/nrs_stretch_log.txt")


if __name__ == "__main__":
    main()
