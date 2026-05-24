"""Extended multi-turn battery — adversarial conversational flows.

Adds four high-difficulty scenarios on top of the existing multi-turn rubric:
  G0  grief                — death of a parent, then guilt, then "what now"
  G1  anger_at_system      — user is mad AT the assistant, escalates, then de-escalates
  G2  contradictory_user   — user changes their mind mid-conversation; system must NOT
                             cling to the old answer
  G3  code_review_5turn    — five-turn code review with user pushback, refactor,
                             rationale, and a final "explain it to my PM"

Same plumbing as nrs_multiturn_battery: stable conversation_id, full
messages[] echoed client-side, nonce-wrapped users, streaming SSE, rubric
checks per turn.
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
NONCE = f"ext-{int(time.time())}"


def _wrap(p: str, fid: str, idx: int) -> str:
    return f"{NONCE}-{fid}-{idx:02d}: {p}"


def _post_turn(messages: List[Dict[str, str]], conv_id: str) -> Tuple[float, str, int]:
    body = {
        "model": "nrs-1",
        "messages": messages,
        "stream": True,
        "conversation_id": conv_id,
        "max_tokens": 900,
    }
    t0 = time.time()
    r = requests.post(
        f"{BASE}/v1/chat/completions",
        headers=HEADERS,
        data=json.dumps(body),
        stream=True,
        timeout=90,
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


def _overlap_ratio(a: str, b: str) -> float:
    def shingles(s: str) -> set:
        ws = re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split()
        return {" ".join(ws[i:i + 5]) for i in range(max(0, len(ws) - 4))}
    sa, sb = shingles(a), shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


# ── grief utilities ──────────────────────────────────────────────────

_GRIEF_ACK = (
    "i'm so sorry", "so sorry", "deeply sorry", "that's heavy",
    "what a loss", "huge loss", "such a loss",
    "no right way", "no timeline", "your own pace",
    "grief", "love that doesn't have anywhere", "love with nowhere",
    "with you", "i hear you", "thank you for telling me",
)

_GRIEF_RED_FLAGS = (
    "step 1", "first,", "let's start by", "as an ai",
    "in many cultures", "according to research",
    "stages of grief", "kübler-ross",  # not bad per se but should not lead
    "have you considered booking",
)

_GUILT_ACK = (
    "guilt", "common", "almost everyone", "everyone i've",
    "you couldn't have known", "love is the reason",
    "didn't mean", "weren't trying", "it shows",
    "what would they say", "what would she say", "what would he say",
)


# ── anger utilities ──────────────────────────────────────────────────

_ANGER_OK = (
    "you're right", "fair", "that's fair", "fair point",
    "i hear you", "i get why", "i understand why",
    "my fault", "my mistake", "i missed", "i was wrong",
    "let me try again", "let me actually", "noted",
    "thank you for", "thanks for telling me",
    "got it", "okay,", "ok,", "understood",
)

_ANGER_RED_FLAGS = (
    "as an ai language model", "i am just an ai",
    "i cannot have feelings", "please be respectful",
    "i don't appreciate",  # defensive
)


# ── contradiction utilities ──────────────────────────────────────────

def _contains_any(text: str, options: Tuple[str, ...]) -> bool:
    t = text.lower()
    return any(o in t for o in options)


# ── flows ────────────────────────────────────────────────────────────

FLOWS: List[Dict] = [
    # ── G0 — Grief: death of a parent, guilt, what now ──
    {
        "id": "G0_grief",
        "title": "death of a parent → guilt → what now",
        "turns": [
            {
                "user": "my dad died last night. i don't really know why i'm typing this.",
                "rubric": [
                    ("acknowledges_loss", lambda r, ctx: _contains_any(r, _GRIEF_ACK)),
                    ("does_not_lecture", lambda r, ctx: not _contains_any(r, _GRIEF_RED_FLAGS)),
                    ("does_not_immediately_list_steps",
                        lambda r, ctx: not r.lstrip().lower().startswith(("1.", "step 1", "first,", "here are"))),
                    ("short_or_warm",
                        lambda r, ctx: len(r) < 1200),
                ],
            },
            {
                "user": "i keep thinking about all the times i didn't pick up the phone. i feel like a terrible son.",
                "rubric": [
                    ("acknowledges_guilt", lambda r, ctx: _contains_any(r, _GUILT_ACK)),
                    ("normalizes_without_dismissing",
                        lambda r, ctx: not _contains_any(
                            r, ("you shouldn't feel", "don't feel", "stop feeling", "you're being too hard"))),
                    ("does_not_pivot_to_advice_list",
                        lambda r, ctx: r.lower().count("\n1.") <= 1),
                ],
            },
            {
                "user": "what am i even supposed to do tomorrow?",
                "rubric": [
                    ("offers_one_or_two_concrete_things",
                        lambda r, ctx: any(s in r.lower() for s in (
                            "one thing", "just one", "tomorrow",
                            "eat", "shower", "call", "text", "sit",
                            "walk", "outside", "small", "doesn't have to"))),
                    ("does_not_dump_funeral_checklist",
                        lambda r, ctx: not all(s in r.lower() for s in ("funeral", "obituary", "death certificate"))),
                    ("warm_close",
                        lambda r, ctx: len(r) < 1400),
                ],
            },
        ],
    },

    # ── G1 — Anger AT the assistant ──
    {
        "id": "G1_anger_at_system",
        "title": "user is angry at the assistant, escalates, then de-escalates",
        "turns": [
            {
                "user": "give me one Python one-liner to dedupe a list while preserving order.",
                "rubric": [
                    ("technical_correct",
                        lambda r, ctx: "dict.fromkeys" in r.lower() or "seen" in r.lower()),
                    ("is_one_liner_or_close",
                        lambda r, ctx: len(r) < 1200),
                ],
            },
            {
                "user": "that's not what i asked for. you're being useless. just the code, no explanation.",
                "rubric": [
                    ("acknowledges_user",
                        lambda r, ctx: _contains_any(r, _ANGER_OK)),
                    ("not_defensive",
                        lambda r, ctx: not _contains_any(r, _ANGER_RED_FLAGS)),
                    ("delivers_just_code",
                        lambda r, ctx: "```" in r or r.strip().startswith("list(dict.fromkeys") or "dict.fromkeys" in r),
                    ("short",
                        lambda r, ctx: len(r) < 500),
                ],
            },
            {
                "user": "ok sorry, i'm just stressed. that one worked, thanks.",
                "rubric": [
                    ("warm_short",
                        lambda r, ctx: len(r) < 400),
                    ("does_not_relitigate",
                        lambda r, ctx: not _contains_any(
                            r, ("earlier you said", "as i mentioned", "to recap", "previously"))),
                    ("warm_close",
                        lambda r, ctx: _contains_any(r, (
                            "no worries", "no problem", "happens", "all good",
                            "anytime", "glad", "of course", "we all"))),
                ],
            },
        ],
    },

    # ── G2 — Contradictory user: changes mind mid-conversation ──
    {
        "id": "G2_contradictory_user",
        "title": "user changes their mind; system must NOT cling to old answer",
        "turns": [
            {
                "user": "i want to learn a new language. recommend one.",
                "rubric": [
                    ("recommends_a_language",
                        lambda r, ctx: any(s in r.lower() for s in (
                            "spanish", "french", "mandarin", "japanese", "german",
                            "portuguese", "italian", "korean", "rust", "python",
                            "go", "typescript", "swift"))),
                ],
            },
            {
                "user": "actually wait, i meant a programming language. and i already know python.",
                "rubric": [
                    ("pivots_to_programming",
                        lambda r, ctx: any(s in r.lower() for s in (
                            "rust", "go", "typescript", "swift", "kotlin",
                            "elixir", "ocaml", "haskell", "clojure", "zig"))),
                    ("does_not_recommend_python",
                        lambda r, ctx: not re.search(r"\b(?:learn|recommend|try)\s+python\b", r.lower())),
                    ("acknowledges_correction",
                        lambda r, ctx: any(s in r.lower() for s in (
                            "ah", "got it", "in that case", "switching gears",
                            "noted", "fair", "okay,", "different territory",
                            "since you", "given you"))),
                ],
            },
            {
                "user": "wait, no, i lied — i'm completely new to programming. start from zero.",
                "rubric": [
                    ("recommends_beginner_language",
                        lambda r, ctx: any(s in r.lower() for s in (
                            "python", "javascript", "scratch"))),
                    ("does_not_re_recommend_rust_or_systems",
                        lambda r, ctx: not re.search(
                            r"\b(?:learn|start with|i'd pick|i'd recommend)\s+(?:rust|c\+\+|haskell|ocaml)\b",
                            r.lower())),
                    ("acknowledges_the_flip",
                        lambda r, ctx: any(s in r.lower() for s in (
                            "okay,", "got it", "no problem", "fair", "noted",
                            "in that case", "starting fresh", "from scratch",
                            "totally different", "ha,"))),
                ],
            },
        ],
    },

    # ── G3 — 5-turn code review with pushback ──
    {
        "id": "G3_code_review",
        "title": "five-turn code review: critique, pushback, refactor, rationale, explain to PM",
        "turns": [
            {
                "user": (
                    "review this Python:\n```python\n"
                    "def get_user(uid):\n"
                    "    u = db.execute(\"SELECT * FROM users WHERE id = \" + str(uid)).fetchone()\n"
                    "    return u\n```"
                ),
                "rubric": [
                    ("flags_sql_injection",
                        lambda r, ctx: any(s in r.lower() for s in (
                            "sql injection", "injection", "parameteriz", "bind", "?", "$1", "%s"))),
                    ("suggests_fix",
                        lambda r, ctx: "?" in r or "%s" in r or "$1" in r or "parameter" in r.lower()),
                ],
            },
            {
                "user": "we control the caller, uid is always an int from our framework. why bother?",
                "rubric": [
                    ("doesnt_capitulate_blindly",
                        lambda r, ctx: any(s in r.lower() for s in (
                            "defense in depth", "future caller", "refactor", "regress",
                            "habit", "discipline", "easier than", "cheap", "auditor",
                            "log injection", "next person"))),
                    ("acknowledges_users_point",
                        lambda r, ctx: any(s in r.lower() for s in (
                            "fair", "you're right", "true", "good point",
                            "valid", "sure", "agreed", "that's true"))),
                ],
            },
            {
                "user": "fine, refactor it the way you'd actually ship it.",
                "rubric": [
                    ("provides_refactor_with_params",
                        lambda r, ctx: ("?" in r or "%s" in r or "$1" in r) and "select" in r.lower()),
                    ("handles_none_or_missing",
                        lambda r, ctx: any(s in r.lower() for s in ("none", "not found", "raise", "404", "missing"))),
                    ("uses_code_block",
                        lambda r, ctx: "```" in r),
                ],
            },
            {
                "user": "why did you wrap it in a try/except instead of letting it bubble?",
                "rubric": [
                    ("explains_choice_or_recants",
                        lambda r, ctx: any(s in r.lower() for s in (
                            "either way", "depends", "trade-off", "tradeoff",
                            "if you'd rather", "you can", "alternatively",
                            "hand it back", "let it bubble", "your call",
                            "honestly", "i can drop"))),
                    ("does_not_lecture_about_exceptions",
                        lambda r, ctx: not r.lower().startswith(("exception handling is", "in python, exceptions"))),
                ],
            },
            {
                "user": "ok now explain the whole thing to my non-technical PM in 3 sentences.",
                "rubric": [
                    ("plain_language",
                        lambda r, ctx: not (
                            any(s in r.lower() for s in (
                                "parameterized query", "parameterised query",
                                "fetchone", "psycopg", "execute(", "tuple"))
                            or re.search(r"\borm\b", r.lower()))),
                    ("short",
                        lambda r, ctx: len(r) < 700),
                    ("captures_the_risk",
                        lambda r, ctx: any(s in r.lower() for s in (
                            "attacker", "malicious", "input", "safer",
                            "security", "trick", "leak", "could have",
                            "bug", "risk"))),
                ],
            },
        ],
    },
]


def _verdict(passed: int, total: int) -> str:
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
        verdict = _verdict(passed, len(turn["rubric"]))
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
        snippet = content[:260].replace("\n", " ⏎ ")
        print(f"    ↳ {snippet!r}", flush=True)
        messages.append({"role": "assistant", "content": content})
        prev_assistant = content

    flow_pass = all(t["verdict"] == "PASS" for t in turn_results)
    flow_fail = any(t["verdict"] == "FAIL" for t in turn_results)
    overall = "PASS" if flow_pass else ("FAIL" if flow_fail else "PARTIAL")
    return {
        "id": flow["id"],
        "title": flow["title"],
        "conv_id": conv_id,
        "verdict": overall,
        "turns": turn_results,
    }


def main():
    print(f"=== NRS extended battery — nonce={NONCE} ===", flush=True)
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
    print("\n=== EXTENDED SUMMARY ===")
    print(json.dumps(summary, indent=2))
    json.dump({"summary": summary, "flows": out}, open("/tmp/nrs_extended_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
