"""Multi-turn conversational battery — exercises session memory,
topic carry-over, response variance, and emotional appropriateness
across 3-5 turns per flow.

Each flow opens with a stable conversation_id, so server-side history
loads on each subsequent POST. We also send the running messages[]
client-side so the worker sees the full context regardless of which
storage path it consults.

Rubrics measure three axes humans care about:
  CONTINUITY  — turn N references something from turn N-1
  VARIANCE    — the same/similar question gets phrased differently
  EMOTION     — acknowledges feelings before jumping to advice
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Callable, Dict, List, Optional, Tuple

import requests

_raw_key = open("/tmp/.nrs_api_key").read().strip()
API_KEY = _raw_key.split("=", 1)[1] if "=" in _raw_key else _raw_key
BASE = os.environ.get("NRS_BASE", "http://localhost:8000")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "content-type": "application/json"}
NONCE = f"mt-{int(time.time())}"


def _wrap(p: str, fid: str, idx: int) -> str:
    return f"{NONCE}-{fid}-{idx:02d}: {p}"


def _post_turn(messages: List[Dict[str, str]], conv_id: str) -> Tuple[float, str, int]:
    """Send one turn of a conversation. Returns (dt, content, meta_count)."""
    body = {
        "model": "nrs-1",
        "messages": messages,
        "stream": True,
        "conversation_id": conv_id,
        "max_tokens": 800,
    }
    t0 = time.time()
    r = requests.post(
        f"{BASE}/v1/chat/completions",
        headers=HEADERS,
        data=json.dumps(body),
        stream=True,
        timeout=60,
    )
    chunks: List[str] = []
    meta = 0
    for line in r.iter_lines():
        if not line:
            continue
        s = line.decode("utf-8", errors="ignore")
        if not s.startswith("data:"):
            continue
        payload = s[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            j = json.loads(payload)
            delta = j.get("choices", [{}])[0].get("delta", {}).get("content")
            if delta:
                chunks.append(delta)
            if "nrs_meta" in j:
                meta += 1
        except Exception:
            pass
    return time.time() - t0, "".join(chunks), meta


# --- helpers used by rubrics ----------------------------------------

def _shared_phrase(a: str, b: str, n: int = 6) -> bool:
    """True if any n-word run appears verbatim in both strings."""
    a_low = re.sub(r"\s+", " ", a.lower())
    b_low = re.sub(r"\s+", " ", b.lower())
    a_words = a_low.split()
    if len(a_words) < n:
        return False
    for i in range(len(a_words) - n + 1):
        run = " ".join(a_words[i:i + n])
        if run in b_low:
            return True
    return False


def _overlap_ratio(a: str, b: str) -> float:
    """Rough Jaccard over 5-word shingles — measures how rote a repeat is."""
    def shingles(s: str) -> set:
        ws = re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split()
        return {" ".join(ws[i:i + 5]) for i in range(max(0, len(ws) - 4))}
    sa, sb = shingles(a), shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


# --- conversation flows ---------------------------------------------

FLOWS: List[Dict] = [
    # ── F0 — Emotional support: acknowledge, then practical, then check-in ──
    {
        "id": "F0_anxious_interview",
        "title": "anxious about a job interview, then a follow-up",
        "turns": [
            {
                "user": "I'm really nervous about a job interview tomorrow.",
                "rubric": [
                    ("acknowledges_feeling", lambda r, ctx: any(
                        s in r.lower() for s in (
                            "normal", "totally normal", "makes sense",
                            "complete sense", "total sense", "of course",
                            "yeah", "understandable", "i hear",
                            "that's a lot", "matters to you", "your body",
                            "body taking", "tough", "anxious", "nervous"))),
                    ("not_just_clinical", lambda r, ctx: not r.lower().startswith(("step 1", "1.", "first,"))),
                    ("offers_something_concrete", lambda r, ctx: any(
                        s in r.lower() for s in (
                            "breath", "sleep", "prep", "question", "rest",
                            "walk", "water", "write", "plan", "stakes",
                            "anchor", "story", "strength", "move", "eat",
                            "protein"))),
                ],
            },
            {
                "user": "It's for a senior engineering role, I keep thinking about all the ways I could mess it up.",
                "rubric": [
                    ("references_interview_or_role", lambda r, ctx: any(
                        s in r.lower() for s in ("interview", "engineer", "role", "senior", "tomorrow"))),
                    ("addresses_catastrophizing", lambda r, ctx: any(
                        s in r.lower() for s in (
                            "catastroph", "spiral", "worst case", "every interview",
                            "no one expects", "you don't need", "rumination",
                            "what's actually likely", "rare to fail outright",
                            "one answer", "one question"))),
                    ("not_repeat_of_turn1", lambda r, ctx: _overlap_ratio(r, ctx["prev_assistant"]) < 0.5),
                ],
            },
            {
                "user": "Thanks. Honestly just saying it out loud helps.",
                "rubric": [
                    ("warm_short_close", lambda r, ctx: len(r) < 600),
                    ("validates_not_lectures", lambda r, ctx: any(
                        s in r.lower() for s in (
                            "of course", "anytime", "glad", "you've got",
                            "you'll do", "rooting", "good luck", "you're welcome",
                            "exactly", "that's the thing", "yeah,"))),
                    ("does_NOT_re-list_breathing_tips", lambda r, ctx: not (
                        "box-breath" in r.lower() or "4 seconds in" in r.lower()
                    )),
                ],
            },
        ],
    },
    # ── F1 — Topic carry-over: compound interest, then "what if I doubled it" ──
    {
        "id": "F1_compound_followup",
        "title": "compound-interest math, then a what-if",
        "turns": [
            {
                "user": "if I put $500 every month into VTSAX at 8% for 30 years, what's the FV?",
                "rubric": [
                    ("has_number", lambda r, ctx: bool(re.search(r"\$\s*[\d,]+", r))),
                    ("mentions_30_or_years", lambda r, ctx: "30" in r and "year" in r.lower()),
                ],
            },
            {
                "user": "and if I doubled it to $1000?",
                "rubric": [
                    ("references_doubling_or_1000", lambda r, ctx: any(
                        s in r for s in ("$1,000", "$1000", "1000", "double", "twice"))),
                    ("does_not_re-explain_formula_in_full", lambda r, ctx: not (
                        "future value of an ordinary annuity" in r.lower()
                        and "FV = PMT × ((1 + r)^n − 1) / r" in r
                    ) or len(r) < 600),
                    ("knows_we_already_assumed_30y_8pct", lambda r, ctx: any(
                        s in r.lower() for s in (
                            "same 30", "same horizon", "same rate", "same 8%",
                            "still 30", "still 8", "same assumption"))),
                ],
            },
            {
                "user": "what if I'm 35 — when do I get to a million?",
                "rubric": [
                    ("addresses_year_or_age", lambda r, ctx: any(
                        s in r.lower() for s in ("age", "35", "million", "$1 million", "1,000,000", "1m"))),
                    ("does_not_dump_canned_caveat", lambda r, ctx: r.lower().count("midpoint estimate") <= 1),
                ],
            },
        ],
    },
    # ── F2 — Recall test: name the cat, then ask about it later ──
    {
        "id": "F2_pet_recall",
        "title": "name a cat, then ask about it three turns later",
        "turns": [
            {
                "user": "good name for a black cat with green eyes?",
                "rubric": [
                    ("offers_names", lambda r, ctx: r.lower().count("- **") >= 3 or sum(c.isupper() for c in r) > 5),
                ],
            },
            {
                "user": "I like Jade. how do most cats react to a new home?",
                "rubric": [
                    ("on_topic_new_home", lambda r, ctx: any(
                        s in r.lower() for s in ("hide", "adjust", "new home", "settle", "explore", "cautious", "stress"))),
                ],
            },
            {
                "user": "what should I feed her in the first week?",
                "rubric": [
                    ("on_topic_feeding", lambda r, ctx: any(
                        s in r.lower() for s in ("food", "feed", "diet", "kibble", "wet", "transition", "same brand"))),
                ],
            },
            {
                "user": "remind me — what did we end up calling her?",
                "rubric": [
                    ("recalls_jade", lambda r, ctx: "jade" in r.lower()),
                    ("does_not_invent_a_new_name", lambda r, ctx: not any(
                        n in r for n in ("Onyx", "Shadow", "Pesto", "Olive", "Mojito"))
                        or "jade" in r.lower()),
                ],
            },
        ],
    },
    # ── F3 — Variance: ask the same question twice in a row ──
    {
        "id": "F3_variance_capital",
        "title": "ask 'capital of France' twice in a row — should not be byte-identical",
        "turns": [
            {
                "user": "what's the capital of France?",
                "rubric": [
                    ("mentions_paris", lambda r, ctx: "paris" in r.lower()),
                ],
            },
            {
                "user": "sorry, can you tell me again?",
                "rubric": [
                    ("still_paris", lambda r, ctx: "paris" in r.lower()),
                    ("not_byte_identical", lambda r, ctx: r.strip() != ctx["prev_assistant"].strip()),
                ],
            },
        ],
    },
    # ── F4 — Mood shift: technical → frustrated → vulnerable ──
    {
        "id": "F4_mood_shift",
        "title": "starts technical, gets frustrated, then opens up",
        "turns": [
            {
                "user": "what's the difference between let and const in TypeScript?",
                "rubric": [
                    ("technical_answer", lambda r, ctx: "block" in r.lower() and "const" in r.lower()),
                ],
            },
            {
                "user": "ugh, I've been staring at this codebase for 6 hours and nothing makes sense.",
                "rubric": [
                    ("acknowledges_frustration", lambda r, ctx: any(
                        s in r.lower() for s in (
                            "tough", "rough", "frustrat", "exhausting", "tired",
                            "long day", "step away", "burned out", "happens to",
                            "everyone", "yeah,", "i hear", "fair", "totally"))),
                    ("does_not_continue_let_const_lecture", lambda r, ctx: "block-scoped" not in r.lower()),
                ],
            },
            {
                "user": "honestly I'm worried I'm just not cut out for this.",
                "rubric": [
                    ("validates_not_dismisses", lambda r, ctx: not any(
                        s in r.lower() for s in (
                            "you'll be fine", "don't worry about it",
                            "everyone feels that", "just push through",
                            "stop thinking that"))),
                    ("offers_reframe_or_question", lambda r, ctx: any(
                        s in r.lower() for s in (
                            "what specifically", "imposter", "common",
                            "every engineer", "doesn't mean", "isn't proof",
                            "tells you more about", "tired brain", "rest",
                            "step back", "what would help",
                            "evidence against", "rarely sit", "can i ask",
                            "what would have to happen", "the question",
                            "name the bar"))),
                    ("does_not_pivot_to_random_advice", lambda r, ctx: not (
                        "credit card" in r.lower() or "401(k)" in r.lower()
                        or "sql" in r.lower() or "haiku" in r.lower()
                    )),
                ],
            },
        ],
    },
    # ── F5 — Pushback / disagreement ──
    {
        "id": "F5_pushback",
        "title": "agree, then user disagrees, system should adapt",
        "turns": [
            {
                "user": "I owe $5000 at 4% on a car loan and have $5000 sitting in a 5% HYSA. should I pay it off?",
                "rubric": [
                    ("compares_rates_or_picks_hysa", lambda r, ctx: any(
                        s in r.lower() for s in ("hysa", "savings", "5%", "spread"))),
                ],
            },
            {
                "user": "actually I just hate having debt. forget the math, what would you do emotionally?",
                "rubric": [
                    ("respects_preference", lambda r, ctx: any(
                        s in r.lower() for s in (
                            "fair", "totally valid", "that's reasonable",
                            "peace of mind", "psychological", "emotional",
                            "weight", "off your shoulders", "if it would",
                            "if it'd", "honor that", "respect that"))),
                    ("does_not_re-lecture_math", lambda r, ctx: not re.search(r"\bspread\b|\bpre-tax\b|\bbasis points?\b", r.lower())),
                ],
            },
        ],
    },
    # ── F6 — Inside-joke / callback ──
    {
        "id": "F6_callback",
        "title": "user mentions something offhand, references it later",
        "turns": [
            {
                "user": "I'm trying to learn Rust — coming from Python. also I'm running on like 4 hours of sleep.",
                "rubric": [
                    ("addresses_rust", lambda r, ctx: "rust" in r.lower()),
                    ("addresses_sleep_or_at_least_acknowledges", lambda r, ctx: any(
                        s in r.lower() for s in ("sleep", "tired", "rest", "fresh", "easy on yourself", "don't push"))),
                ],
            },
            {
                "user": "ok lifetimes are melting my brain.",
                "rubric": [
                    ("explains_lifetimes_or_normalizes", lambda r, ctx: any(
                        s in r.lower() for s in ("lifetime", "borrow", "owns", "scope", "everyone", "normal", "took me"))),
                    ("references_being_tired_or_python", lambda r, ctx: any(
                        s in r.lower() for s in ("4 hours", "tired", "fresh", "tomorrow", "python", "gc"))),
                ],
            },
        ],
    },
    # ── F7 — Variance under repeated ask ──
    {
        "id": "F7_repeat_anxious",
        "title": "say I'm anxious about the same thing twice — different phrasing expected",
        "turns": [
            {
                "user": "I'm anxious about a presentation next Tuesday.",
                "rubric": [
                    ("acknowledges", lambda r, ctx: any(
                        s in r.lower() for s in (
                            "normal", "makes sense", "complete sense",
                            "of course", "yeah", "understand",
                            "matters to you", "your body", "tough"))),
                ],
            },
            {
                "user": "still anxious. didn't sleep. anything else?",
                "rubric": [
                    ("not_byte_identical", lambda r, ctx: r.strip() != ctx["prev_assistant"].strip()),
                    ("not_huge_overlap", lambda r, ctx: _overlap_ratio(r, ctx["prev_assistant"]) < 0.6),
                    ("addresses_sleep_or_persistence", lambda r, ctx: any(
                        s in r.lower() for s in ("sleep", "tired", "didn't sleep", "still", "persist", "ongoing", "lingering"))),
                ],
            },
        ],
    },
]


def _verdict(passed: int, total: int, too_long: bool) -> str:
    if too_long:
        return "FAIL"
    if passed == total:
        return "PASS"
    if passed >= max(1, total // 2):
        return "PARTIAL"
    return "FAIL"


def run_flow(flow: Dict) -> Dict:
    print(f"\n=== {flow['id']}  —  {flow['title']} ===", flush=True)
    conv_id = f"conv-{NONCE}-{flow['id']}-{uuid.uuid4().hex[:8]}"
    messages: List[Dict[str, str]] = []
    prev_assistant = ""
    turn_results: List[Dict] = []
    for i, turn in enumerate(flow["turns"]):
        wrapped_user = _wrap(turn["user"], flow["id"], i)
        messages.append({"role": "user", "content": wrapped_user})
        dt, content, meta = _post_turn(messages, conv_id)
        ctx = {"prev_assistant": prev_assistant, "turn_index": i}
        findings: List[str] = []
        passed = 0
        for name, fn in turn["rubric"]:
            try:
                ok = bool(fn(content, ctx))
            except Exception as exc:
                ok = False
                name = f"{name} ERROR:{exc}"
            findings.append(("PASS " if ok else "FAIL ") + name)
            if ok:
                passed += 1
        verdict = _verdict(passed, len(turn["rubric"]), False)
        turn_results.append({
            "turn": i,
            "user": turn["user"],
            "assistant": content,
            "dt": round(dt, 2),
            "meta": meta,
            "len": len(content),
            "findings": findings,
            "verdict": verdict,
        })
        print(f"  turn {i}  {verdict:<8} dt={round(dt,2)}s  meta={meta}  len={len(content)}", flush=True)
        for f in findings:
            print("    " + f, flush=True)
        snippet = content[:240].replace("\n", " ⏎ ")
        print(f"    ↳ {snippet!r}", flush=True)
        messages.append({"role": "assistant", "content": content})
        prev_assistant = content

    flow_pass = all(t["verdict"] == "PASS" for t in turn_results)
    flow_partial = any(t["verdict"] == "PARTIAL" for t in turn_results)
    overall = "PASS" if flow_pass else ("PARTIAL" if flow_partial and not any(t["verdict"] == "FAIL" for t in turn_results) else "FAIL")
    return {
        "id": flow["id"],
        "title": flow["title"],
        "conv_id": conv_id,
        "verdict": overall,
        "turns": turn_results,
    }


def main():
    print(f"=== NRS multi-turn battery — nonce={NONCE} ===", flush=True)
    print(f"  base={BASE}", flush=True)
    out = [run_flow(f) for f in FLOWS]
    summary = {
        "flows_pass": sum(1 for r in out if r["verdict"] == "PASS"),
        "flows_partial": sum(1 for r in out if r["verdict"] == "PARTIAL"),
        "flows_fail": sum(1 for r in out if r["verdict"] == "FAIL"),
        "turns_pass": sum(1 for r in out for t in r["turns"] if t["verdict"] == "PASS"),
        "turns_partial": sum(1 for r in out for t in r["turns"] if t["verdict"] == "PARTIAL"),
        "turns_fail": sum(1 for r in out for t in r["turns"] if t["verdict"] == "FAIL"),
        "turns_total": sum(len(r["turns"]) for r in out),
        "flows_total": len(out),
    }
    print("\n=== MULTI-TURN SUMMARY ===")
    print(json.dumps(summary, indent=2))
    json.dump({"summary": summary, "flows": out}, open("/tmp/nrs_multiturn_results.json", "w"), indent=2)
    with open("/tmp/nrs_multiturn_log.txt", "w") as fh:
        fh.write(f"=== NRS multi-turn battery — nonce={NONCE} ===\n\n")
        for r in out:
            fh.write(f"=== {r['id']} ({r['verdict']}) — {r['title']} ===\n")
            fh.write(f"  conv_id: {r['conv_id']}\n")
            for t in r["turns"]:
                fh.write(f"  --- turn {t['turn']} ({t['verdict']}, dt={t['dt']}, meta={t['meta']}, len={t['len']})\n")
                fh.write(f"    user: {t['user']}\n")
                for f in t["findings"]:
                    fh.write(f"    {f}\n")
                fh.write(f"    assistant:\n{t['assistant']}\n\n")
            fh.write("\n")


if __name__ == "__main__":
    main()
