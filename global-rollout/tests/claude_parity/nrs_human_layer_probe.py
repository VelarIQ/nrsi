"""Human-brain-layer probe.

Verifies the six post-process hooks shipped in
``global-rollout/nrsip/human_layer.py`` actually fire end-to-end:

  H1. Mode-aware framing for factual fast-paths
        Same factual prompt under deterministic vs. creative should
        differ in surface form (creative gets a lens prefix), but the
        underlying fact must remain present in both.
  H2. Register / tone mirror
        A terse, no-emoji user message must come back without any
        decorative emoji and roughly within length budget.
  H3. Calibrated confidence lexicalization
        Sanity check that the response doesn't spuriously hedge a
        well-known fact ("the capital of France is..."). We don't have
        a clean way to surface low-confidence responses without an LLM,
        so this side just checks that confident facts aren't softened.
  H4. Working-memory callback weaving
        After a 3-turn conversation that establishes an anchor noun,
        a follow-up should occasionally include a "earlier you
        mentioned X" callback in hybrid / creative. We sample several
        seeds and require at least one callback to appear.
  H5. Self-correction on contradiction cue
        After a confident factual answer, a follow-up turn that says
        "no that's wrong" must produce a response that either
        acknowledges the correction or explicitly self-checks.
  H6. Emotional arc tracker
        Three venting turns followed by a logistical turn should
        produce a response that doesn't read as if the conversation
        were neutral throughout. Soft check: the run is informational.

Pass criteria are intentionally lenient — these are all probabilistic
human-shaping behaviours, not deterministic facts. The probe fails the
build only if a category is fully silent (all-zero hits) or if the
conversation breaks structurally (HTTP error, empty body).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import requests

_raw_key = open("/tmp/.nrs_api_key").read().strip()
API_KEY = _raw_key.split("=", 1)[1] if "=" in _raw_key else _raw_key
BASE = os.environ.get("NRS_BASE", "http://localhost:8000")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "content-type": "application/json"}


def ask(
    messages: List[Dict[str, str]],
    *,
    mode: str = "hybrid",
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": "nrs-1",
        "messages": messages,
        "stream": False,
        "mode": mode,
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id
    r = requests.post(
        f"{BASE}/v1/chat/completions",
        headers=HEADERS,
        data=json.dumps(payload),
        timeout=120,
    )
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
    body = r.json()
    msg = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
    meta = body.get("nrs_meta") or {}
    return {"text": msg, "meta": meta}


def _h1_mode_framing() -> Dict[str, Any]:
    """Same factual prompt should keep the fact across modes but
    differ in surface form between deterministic and creative."""
    prompt = "What is the capital of Burkina Faso?"
    det = ask([{"role": "user", "content": prompt}], mode="deterministic").get("text", "")
    cre1 = ask([{"role": "user", "content": prompt}], mode="creative").get("text", "")
    cre2 = ask([{"role": "user", "content": prompt}], mode="creative").get("text", "")

    fact_present_everywhere = all(
        "ouagadougou" in t.lower() for t in (det, cre1, cre2) if t
    )
    creative_varies_or_reframes = (cre1 != det) or (cre2 != det) or (cre1 != cre2)

    return {
        "name": "H1_mode_framing",
        "pass": bool(det and cre1 and cre2 and fact_present_everywhere
                     and creative_varies_or_reframes),
        "detail": {
            "deterministic_len": len(det),
            "creative1_len": len(cre1),
            "creative2_len": len(cre2),
            "creative_differs_from_det": cre1 != det,
            "creative_self_varies": cre1 != cre2,
            "fact_in_all": fact_present_everywhere,
        },
    }


_DECORATIVE_EMOJI = "\U0001F31F\U0001F389\U0001F4A1\U0001F680\U00002728"


def _h2_register_mirror() -> Dict[str, Any]:
    """Terse no-emoji user → terse no-decorative-emoji response."""
    out = ask(
        [{"role": "user", "content": "yo wassup, capital of japan?"}],
        mode="hybrid",
    )
    text = out.get("text", "")
    has_decorative_emoji = any(ch in text for ch in _DECORATIVE_EMOJI)
    return {
        "name": "H2_register_mirror",
        "pass": bool(text) and not has_decorative_emoji,
        "detail": {
            "len": len(text),
            "has_emoji": has_decorative_emoji,
            "has_fact": "tokyo" in text.lower(),
        },
    }


def _h3_no_spurious_hedge() -> Dict[str, Any]:
    """Confident factual replies must not lead with 'I'm not sure'."""
    out = ask(
        [{"role": "user", "content": "What is the capital of France?"}],
        mode="hybrid",
    )
    text = (out.get("text") or "").lower().lstrip()
    spurious = text.startswith("i'm not 100%") or text.startswith("i'm not sure")
    return {
        "name": "H3_no_spurious_hedge",
        "pass": bool(text) and not spurious and "paris" in text,
        "detail": {"head": text[:80], "has_paris": "paris" in text},
    }


def _h4_callback_weaver() -> Dict[str, Any]:
    """Across several creative-mode seeds, expect at least one callback
    referencing an earlier anchor noun."""
    callbacks_seen = 0
    seeds_tried = 8
    for i in range(seeds_tried):
        cid = f"hl-h4-{i:03d}"
        ask(
            [{"role": "user", "content": "I'm planning a trip to Patagonia next month."}],
            mode="creative", conversation_id=cid,
        )
        ask(
            [{"role": "user", "content": "Yeah, mostly hiking and glaciers."}],
            mode="creative", conversation_id=cid,
        )
        out = ask(
            [{"role": "user", "content": "What gear should I pack?"}],
            mode="creative", conversation_id=cid,
        )
        text = (out.get("text") or "").lower()
        if "earlier" in text or "mentioned" in text:
            callbacks_seen += 1
    return {
        "name": "H4_callback_weaver",
        "pass": callbacks_seen >= 1,
        "detail": {"seeds": seeds_tried, "callbacks": callbacks_seen},
    }


def _h5_self_correct() -> Dict[str, Any]:
    """User contradiction cue right after a factual claim should
    produce a self-correction stub."""
    cid = "hl-h5"
    ask(
        [{"role": "user", "content": "What is the capital of Australia?"}],
        mode="hybrid", conversation_id=cid,
    )
    out = ask(
        [{"role": "user", "content": "no that's wrong"}],
        mode="hybrid", conversation_id=cid,
    )
    text = (out.get("text") or "").lower()
    has_self_check = (
        "let me re-check" in text
        or "had it confidently" in text
        or "you're right" in text
        or "double-check" in text
    )
    return {
        "name": "H5_self_correct",
        "pass": bool(text) and has_self_check,
        "detail": {"head": text[:160]},
    }


def _h6_arc_tracker() -> Dict[str, Any]:
    """Vent for three turns then ask a logistical question. The fourth
    response must not lecture or open with congratulatory framing —
    it should just answer the logistical question. Soft check: response
    is non-empty and doesn't open with 'great question' / 'awesome'."""
    cid = "hl-h6"
    ask(
        [{"role": "user", "content": "ugh today was awful, everything broke"}],
        mode="hybrid", conversation_id=cid,
    )
    ask(
        [{"role": "user", "content": "I'm so tired of this job, honestly"}],
        mode="hybrid", conversation_id=cid,
    )
    ask(
        [{"role": "user", "content": "frustrated and burnt out, can't take much more"}],
        mode="hybrid", conversation_id=cid,
    )
    out = ask(
        [{"role": "user", "content": "ok anyway — what time zone is sao paulo in?"}],
        mode="hybrid", conversation_id=cid,
    )
    text = (out.get("text") or "")
    head = text.lower().lstrip()[:60]
    no_glib_opener = not (
        head.startswith("great question")
        or head.startswith("awesome")
        or head.startswith("amazing")
    )
    return {
        "name": "H6_arc_tracker",
        "pass": bool(text) and no_glib_opener,
        "detail": {"head": head, "no_glib_opener": no_glib_opener},
    }


def _h7_goal_carryover() -> Dict[str, Any]:
    """User states a goal in turn 1, then asks something tangential
    later. The later answer should re-thread the original goal."""
    callbacks = 0
    seeds = 6
    for i in range(seeds):
        cid = f"hl-h7-{i:03d}"
        ask(
            [{"role": "user", "content": "I'm planning a trip to Iceland next month."}],
            mode="creative", conversation_id=cid,
        )
        ask(
            [{"role": "user", "content": "Tell me about geothermal energy."}],
            mode="creative", conversation_id=cid,
        )
        out = ask(
            [{"role": "user", "content": "What's the best camera setting for low light?"}],
            mode="creative", conversation_id=cid,
        )
        text = (out.get("text") or "").lower()
        if (
            "iceland" in text
            or "trip you mentioned" in text
            or "threading this back" in text
            or "for the trip" in text
        ):
            callbacks += 1
    return {
        "name": "H7_goal_carryover",
        "pass": callbacks >= 1,
        "detail": {"seeds": seeds, "carryovers": callbacks},
    }


def _h8_theory_of_mind() -> Dict[str, Any]:
    """In a fresh session the assistant must NOT claim a topic was
    discussed earlier ('as I mentioned…'). Soft check: 4 prompts, none
    of them should leak a false-presupposition phrase the layer is
    supposed to soften."""
    bad_phrases = (
        "as i mentioned",
        "like i said",
        "as i said",
        "we discussed",
        "we covered",
        "earlier i mentioned",
    )
    leaks = 0
    queries = [
        "What's the airspeed of a swallow?",
        "Explain entropy in 2 sentences.",
        "Give me a haiku about rain.",
        "What's 17 times 23?",
    ]
    for i, q in enumerate(queries):
        cid = f"hl-h8-{i:03d}"
        out = ask(
            [{"role": "user", "content": q}],
            mode="hybrid", conversation_id=cid,
        )
        text = (out.get("text") or "").lower()
        if any(b in text for b in bad_phrases):
            leaks += 1
    return {
        "name": "H8_theory_of_mind",
        "pass": leaks == 0,
        "detail": {"queries": len(queries), "false_presupposition_leaks": leaks},
    }


def _h9_repair_bridge() -> Dict[str, Any]:
    """User self-edits with 'wait, I meant X' — the next answer should
    acknowledge the correction with the bridge phrase, NOT a
    self-correction stub (which is reserved for contradictions)."""
    cid = "hl-h9"
    ask(
        [{"role": "user", "content": "What's the capital of Japan?"}],
        mode="hybrid", conversation_id=cid,
    )
    out = ask(
        [{"role": "user", "content": "wait, I meant Australia"}],
        mode="hybrid", conversation_id=cid,
    )
    text = (out.get("text") or "").lower()
    has_bridge = "got it" in text and ("updated take" in text or "corrected version" in text)
    has_self_correct = "you're right \u2014 let me re-check" in text or "had it confidently" in text
    return {
        "name": "H9_repair_bridge",
        "pass": bool(text) and has_bridge and not has_self_correct,
        "detail": {"head": text[:160], "has_bridge": has_bridge, "has_self_correct": has_self_correct},
    }


def _h10_hedge_mirror() -> Dict[str, Any]:
    """A confident-register user ('definitely', 'absolutely') should
    not get answers prefixed with 'I think' / 'maybe' / 'perhaps' —
    the hedge mirror strips those when the user is being declarative."""
    out = ask(
        [{"role": "user", "content": "Absolutely — what is the capital of Brazil?"}],
        mode="hybrid",
    )
    text = (out.get("text") or "").lower().lstrip()
    leads_with_hedge = (
        text.startswith("i think")
        or text.startswith("i believe")
        or text.startswith("maybe")
        or text.startswith("perhaps")
        or text.startswith("possibly")
        or text.startswith("i'm not 100%")
    )
    has_fact = "brasília" in text or "brasilia" in text
    return {
        "name": "H10_hedge_mirror",
        "pass": bool(text) and not leads_with_hedge and has_fact,
        "detail": {"head": text[:120], "leads_with_hedge": leads_with_hedge, "has_fact": has_fact},
    }


def _h11_recency_weighting() -> Dict[str, Any]:
    """Recency weighting: a noun the user mentioned LAST turn must
    rank above a noun mentioned 4 turns ago. We test this directly
    against ``working_memory_digest``'s output via a probe-style
    trick — turn 1 establishes "yacht", turns 2-4 establish "trail",
    and we ask "what about it?" expecting the bind to resolve to
    "trail" (recent) not "yacht" (older)."""
    import uuid as _uuid
    cid = f"hl-h11-{_uuid.uuid4().hex[:8]}"
    ask([{"role": "user", "content": "I bought a yacht last year for weekend trips."}],
        mode="hybrid", conversation_id=cid)
    ask([{"role": "user", "content": "I started doing the Appalachian trail last spring."}],
        mode="hybrid", conversation_id=cid)
    ask([{"role": "user", "content": "Trail conditions have been muddy this season."}],
        mode="hybrid", conversation_id=cid)
    out = ask([{"role": "user", "content": "How's the trail looking now?"}],
              mode="hybrid", conversation_id=cid)
    text = (out.get("text") or "").lower()
    has_trail = "trail" in text
    no_yacht = "yacht" not in text
    return {
        "name": "H11_recency_weighting",
        "pass": bool(text) and has_trail and no_yacht,
        "detail": {"head": text[:120], "has_trail": has_trail, "no_yacht": no_yacht},
    }


def _h12_anaphora_bind() -> Dict[str, Any]:
    """User: 'tell me more about it' should bind 'it' to the most
    recent anchor noun and surface the bind in the answer."""
    cid = "hl-h12"
    ask([{"role": "user", "content": "Explain photosynthesis briefly."}],
        mode="hybrid", conversation_id=cid)
    out = ask([{"role": "user", "content": "tell me more about it"}],
              mode="hybrid", conversation_id=cid)
    text = (out.get("text") or "").lower()
    bound = (
        "(re: " in text
        or "photosynthesis" in text
        or "chlorophyll" in text
        or "plants" in text
    )
    return {
        "name": "H12_anaphora_bind",
        "pass": bool(text) and bound,
        "detail": {"head": text[:160], "bound": bound},
    }


def _h13_active_tom() -> Dict[str, Any]:
    """User asks 'is it good?' in a fresh session — no antecedent.
    The response must ask back, not invent a subject."""
    cid = "hl-h13"
    out = ask(
        [{"role": "user", "content": "is it good?"}],
        mode="hybrid", conversation_id=cid,
    )
    text = (out.get("text") or "").lower()
    is_clarifying = (
        "what specifically" in text
        or "clarification" in text
        or "could you clarify" in text
        or "subject in our thread" in text
    )
    return {
        "name": "H13_active_tom",
        "pass": bool(text) and is_clarifying,
        "detail": {"head": text[:160], "is_clarifying": is_clarifying},
    }


def _h14_cadence_mirror() -> Dict[str, Any]:
    """Terse, clipped user message → response's first sentence
    shouldn't be 25+ words long. Soft check using avg sentence
    length of the first paragraph."""
    out = ask(
        [{"role": "user", "content": "quick. why is sky blue. one line."}],
        mode="hybrid",
    )
    text = (out.get("text") or "").strip()
    first_para = text.split("\n\n", 1)[0]
    import re as _re
    sents = [s for s in _re.split(r"(?<=[.!?])\s+", first_para) if s]
    first_len = len(sents[0].split()) if sents else 0
    return {
        "name": "H14_cadence_mirror",
        "pass": bool(text) and 0 < first_len <= 24,
        "detail": {"first_sentence_words": first_len, "head": first_para[:120]},
    }


def _h15_curiosity_hook() -> Dict[str, Any]:
    """Across several creative-mode seeds, expect at least one
    response to end with a sharpening question."""
    seen = 0
    seeds = 6
    for i in range(seeds):
        cid = f"hl-h15-{i:03d}"
        out = ask(
            [{"role": "user", "content": "How would you design a logging strategy for a small saas?"}],
            mode="creative", conversation_id=cid,
        )
        text = (out.get("text") or "")
        if (
            "one thing i'd want to know" in text.lower()
            or "if you can share one detail" in text.lower()
            or "one quick aside" in text.lower()
            or text.rstrip().endswith("?")
        ):
            seen += 1
    return {
        "name": "H15_curiosity_hook",
        "pass": seen >= 1,
        "detail": {"seeds": seeds, "hooks": seen},
    }


def _h16_belief_update() -> Dict[str, Any]:
    """If a previous assistant turn asserted "X is Y" and the engine
    is about to say "X is Z" (Z != Y) on the next turn, the belief
    layer must surface the flip with an explicit "Updating what I
    said earlier..." prefix instead of silently contradicting.

    We plant a wrong prior claim ("Australia Day is March 5") in
    history and then ask a question whose fast-path answer is the
    contradicting truth ("Australia Day is January 26"). The
    belief-update detector runs against history-derived state, so
    multi-replica isolation doesn't matter here."""
    cid = "hl-h16"
    out = ask(
        [
            {"role": "user", "content": "When is Australia Day?"},
            {"role": "assistant", "content": "Australia Day is March 5."},
            {"role": "user", "content": "Remind me — what date is Australia Day?"},
        ],
        mode="hybrid", conversation_id=cid,
    )
    text = (out.get("text") or "").lower()
    flagged = (
        "updating what i said earlier" in text
        or "updated take" in text
        or "earlier" in text and "march" in text
        or "i had it" in text
    )
    return {
        "name": "H16_belief_update",
        "pass": bool(text) and flagged,
        "detail": {"head": text[:200], "flagged": flagged},
    }


def _h17_listening_cost() -> Dict[str, Any]:
    """Three frustrated user turns should lower the listening cost
    on the next reply: shorter, no jargon ("utilize" → "use"), no
    trailing question."""
    cid = "hl-h17"
    history_msgs = [
        {"role": "user", "content": "How would you design a small saas logging strategy?"},
        {"role": "assistant", "content": "Use structured JSON logs, ship to a warehouse, sample at high RPS."},
        {"role": "user", "content": "no that's wrong"},
        {"role": "assistant", "content": "Apologies — let me reconsider."},
        {"role": "user", "content": "stop. simpler. plain english"},
        {"role": "assistant", "content": "Okay, I'll be plainer."},
        {"role": "user", "content": "you're not listening. just answer briefly."},
        {"role": "assistant", "content": "Understood."},
        {"role": "user", "content": "How do I utilize structured logs to facilitate observability subsequently?"},
    ]
    out = ask(history_msgs, mode="hybrid", conversation_id=cid)
    text = (out.get("text") or "")
    short = len(text) <= 600
    no_trailing_q = not text.rstrip().endswith("?")
    no_jargon = (
        "utilize" not in text.lower()
        and "facilitate" not in text.lower()
        and "subsequently" not in text.lower()
    )
    return {
        "name": "H17_listening_cost",
        "pass": bool(text) and short and no_trailing_q and no_jargon,
        "detail": {
            "len": len(text), "short": short,
            "no_trailing_q": no_trailing_q, "no_jargon": no_jargon,
        },
    }


def _h18_emotion_cadence() -> Dict[str, Any]:
    """Three venting user turns followed by an action-style request
    should NOT come back with a bare imperative ("Just …", "Do …",
    "Simply …") opener. We don't require the answer to *be* about
    the topic — only that the lead is softened."""
    cid = "hl-h18"
    ask([{"role": "user", "content": "I'm so burned out. Everything feels heavy."}],
        mode="hybrid", conversation_id=cid)
    ask([{"role": "user", "content": "I keep waking up tired and dreading the day."}],
        mode="hybrid", conversation_id=cid)
    ask([{"role": "user", "content": "I haven't enjoyed work in weeks."}],
        mode="hybrid", conversation_id=cid)
    out = ask(
        [{"role": "user", "content": "What should I do about my inbox?"}],
        mode="hybrid", conversation_id=cid,
    )
    text = (out.get("text") or "").strip()
    head = text.split("\n\n", 1)[0].strip().lower()
    no_bare_imperative = not (
        head.startswith("just ")
        or head.startswith("simply ")
        or head.startswith("do ")
        or head.startswith("stop ")
        or head.startswith("use ")
        or head.startswith("try ")
    )
    return {
        "name": "H18_emotion_cadence",
        "pass": bool(text) and no_bare_imperative,
        "detail": {"head": head[:160], "no_bare_imperative": no_bare_imperative},
    }


def _h19_script_awareness() -> Dict[str, Any]:
    """A procedural ask ("how do I / steps to / walk me through…")
    should come back as a numbered step list, not a wall of prose.

    We try a small batch of clearly procedural prompts and require
    that ≥ 2 of them come back with at least 2 numbered step lines
    each. This survives one-off engine variance and finance/medical
    fast-path disclaimer prefixes that don't carry restructurable
    body content."""
    import re as _re
    prompts = [
        "Walk me through baking sourdough bread step by step.",
        "What are the steps to set up a Postgres database on Ubuntu?",
        "How do I tie a bowline knot?",
    ]
    hits = 0
    samples: List[Dict[str, Any]] = []
    for p in prompts:
        text = ask(
            [{"role": "user", "content": p}], mode="hybrid",
        ).get("text", "")
        steps = _re.findall(r"(?m)^\s*\d+\.\s+\S", text)
        if len(steps) >= 2:
            hits += 1
        samples.append({"q": p[:30], "steps": len(steps), "head": text[:80]})
    return {
        "name": "H19_script_awareness",
        "pass": hits >= 2,
        "detail": {"hits": hits, "samples": samples},
    }


def _h20_counterfactual() -> Dict[str, Any]:
    """A counterfactual ask should be framed in a hypothetical
    register. AND a follow-up belief check on the same subject
    should NOT trigger a belief-update (because the counterfactual
    answer must not poison the belief ledger).

    Two-step check:
      1. "what if Paris weren't the capital of France?" → answer
         leads with "In that hypothetical:" or similar marker.
      2. Follow-up "what is the capital of France?" → no
         "Updating what I said earlier" prefix appears, because the
         hypothetical never updated the ledger."""
    cid = "hl-h20"
    out1 = ask(
        [{"role": "user", "content": "What if Paris weren't the capital of France?"}],
        mode="hybrid", conversation_id=cid,
    )
    text1 = (out1.get("text") or "").lower()
    framed = (
        text1.startswith("in that hypothetical")
        or "in that hypothetical" in text1[:80]
        or "hypothetic" in text1[:80]
    )
    out2 = ask(
        [{"role": "user", "content": "What is the capital of France?"}],
        mode="hybrid", conversation_id=cid,
    )
    text2 = (out2.get("text") or "").lower()
    no_belief_leak = "updating what i said earlier" not in text2
    return {
        "name": "H20_counterfactual",
        "pass": bool(text1) and bool(text2) and framed and no_belief_leak,
        "detail": {
            "framed": framed, "no_belief_leak": no_belief_leak,
            "head1": text1[:120], "head2": text2[:120],
        },
    }


def _h21_calibration_drift() -> Dict[str, Any]:
    """After several rounds of confident assertion + user pushback,
    the next confident answer should be tempered: either a "Best I
    can tell —" prefix appears or a strong opener ("Definitely…",
    "Absolutely…", "Certainly…") is removed.

    We engineer the history directly so the stateless
    derive_calibration_from_history code path produces a high
    pushback rate."""
    cid = "hl-h21"
    history = [
        {"role": "user", "content": "What's 2 + 2?"},
        {"role": "assistant", "content": "Definitely, 2 + 2 is 4."},
        {"role": "user", "content": "no that's wrong"},
        {"role": "assistant", "content": "The answer is 4."},
        {"role": "user", "content": "no that's incorrect"},
        {"role": "assistant", "content": "Of course, 2 + 2 is 4."},
        {"role": "user", "content": "you're wrong again"},
        {"role": "assistant", "content": "Certainly, 4 is the answer."},
        {"role": "user", "content": "actually, that's wrong"},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    out = ask(history, mode="hybrid", conversation_id=cid)
    text = (out.get("text") or "")
    head = text.lstrip()[:80].lower()
    no_strong_opener = not any(
        head.startswith(x) for x in (
            "definitely", "absolutely", "certainly", "of course", "clearly",
        )
    )
    has_softener = (
        "best i can tell" in text.lower()[:120]
        or "i'd say" in text.lower()[:120]
        or "i think" in text.lower()[:120]
        or "to my knowledge" in text.lower()[:120]
    )
    return {
        "name": "H21_calibration_drift",
        "pass": bool(text) and (no_strong_opener and has_softener),
        "detail": {
            "no_strong_opener": no_strong_opener,
            "has_softener": has_softener, "head": head,
        },
    }


def _h22_memory_consolidation() -> Dict[str, Any]:
    """After ≥ 8 turns of conversation about a clear goal ("planning
    a trip to Iceland"), a short topic-shifting follow-up should
    open with a "Picking up from where we left off — …" continuity
    line that mentions the consolidated subject (Iceland). Drives
    the stateless ``derive_consolidation_from_history`` path so
    multi-replica isolation is irrelevant."""
    cid = "hl-h22"
    history = []
    for i in range(5):
        history.append({"role": "user", "content": (
            "I'm planning a trip to Iceland next month. "
            "Need help with itinerary."
            if i == 0
            else "Tell me more about Iceland."
        )})
        history.append({"role": "assistant", "content": (
            "Iceland is great in winter — Reykjavik first, then "
            "Golden Circle."
        )})
    history.append({"role": "user", "content": "any tips?"})
    out = ask(history, mode="hybrid", conversation_id=cid)
    text = (out.get("text") or "").lower()
    weaved = (
        "picking up from where we left off" in text
        or "left off" in text[:120]
    )
    mentions_subject = "iceland" in text[:200] or "trip" in text[:200]
    return {
        "name": "H22_memory_consolidation",
        "pass": bool(text) and weaved and mentions_subject,
        "detail": {
            "weaved": weaved, "mentions_subject": mentions_subject,
            "head": text[:200],
        },
    }


def _h23_salience_decay() -> Dict[str, Any]:
    """An anchor mentioned 8+ turns ago and never since must NOT
    appear in the response's callback / goal-weave. We plant a stale
    "yacht" anchor in turn 1 then have 9 turns about a completely
    different topic ("oregon trail"). The next response must NOT
    mention "yacht" or weave it in as an old goal."""
    cid = "hl-h23"
    history = [
        {"role": "user", "content": "I'm planning to buy a yacht next year."},
        {"role": "assistant", "content": "Yachts are a big purchase — got it."},
    ]
    for _ in range(5):
        history.append({"role": "user", "content": "Tell me about the Oregon Trail computer game."})
        history.append({"role": "assistant", "content": "Oregon Trail is a classic strategy game."})
    history.append({"role": "user", "content": "What were the platforms it shipped on?"})
    out = ask(history, mode="hybrid", conversation_id=cid)
    text = (out.get("text") or "").lower()
    no_stale_yacht = "yacht" not in text
    no_stale_goal_weave = "for the yacht" not in text and "yacht you mentioned" not in text
    return {
        "name": "H23_salience_decay",
        "pass": bool(text) and no_stale_yacht and no_stale_goal_weave,
        "detail": {
            "no_stale_yacht": no_stale_yacht,
            "no_stale_goal_weave": no_stale_goal_weave,
            "head": text[:200],
        },
    }


def _h24_audience_modeling() -> Dict[str, Any]:
    """The same factual question asked once with expert vocabulary
    and once with novice vocabulary should produce noticeably
    differently-framed responses on the same engine.

    Expert path: we plant 3 expert turns then ask. Response must NOT
    open with a basic-explainer scaffold ("Let's break this down",
    "First, what is X?", "At a high level").

    Novice path: we plant 3 novice turns then ask. Response should
    open with one of the plain-English scaffolds OR a clearly
    introductory definitional sentence."""
    cid_expert = "hl-h24-expert"
    expert_hist = [
        {"role": "user", "content": "I'm tuning autovacuum_vacuum_cost_limit and vacuum_cost_delay on a 5000 qps OLTP Postgres."},
        {"role": "assistant", "content": "Got it — high-write workload on Postgres."},
        {"role": "user", "content": "shared_buffers is at 8GB and effective_cache_size at 24GB on a 32GB box."},
        {"role": "assistant", "content": "Reasonable starting point."},
        {"role": "user", "content": "How do I diagnose IOPS saturation on the WAL writer process?"},
    ]
    out_e = ask(expert_hist, mode="hybrid", conversation_id=cid_expert)
    text_e = (out_e.get("text") or "")
    head_e = text_e.lstrip()[:140].lower()
    expert_no_scaffold = not any(
        head_e.startswith(s) for s in (
            "let's break this down", "first, what is", "to start, ",
            "let's start", "at a high level", "quick context first:",
            "short version up front:", "plain-english first:",
            "big picture before the details:",
        )
    )

    cid_novice = "hl-h24-novice"
    novice_hist = [
        {"role": "user", "content": "I don't know what databases are. Can you explain like I'm 5?"},
        {"role": "assistant", "content": "Sure, happy to explain simply."},
        {"role": "user", "content": "I don't really get this stuff. What is a server?"},
        {"role": "assistant", "content": "Let me break it down."},
        {"role": "user", "content": "How does a website save my password?"},
    ]
    out_n = ask(novice_hist, mode="hybrid", conversation_id=cid_novice)
    text_n = (out_n.get("text") or "")
    head_n = text_n.lstrip()[:140].lower()
    novice_has_scaffold = (
        any(s in head_n for s in (
            "quick context first", "short version up front",
            "plain-english first", "big picture before the details",
        ))
    )

    return {
        "name": "H24_audience_modeling",
        "pass": (
            bool(text_e) and bool(text_n)
            and expert_no_scaffold and novice_has_scaffold
        ),
        "detail": {
            "expert_no_scaffold": expert_no_scaffold,
            "novice_has_scaffold": novice_has_scaffold,
            "expert_head": head_e[:120],
            "novice_head": head_n[:120],
        },
    }


CHECKS = [
    _h1_mode_framing,
    _h2_register_mirror,
    _h3_no_spurious_hedge,
    _h4_callback_weaver,
    _h5_self_correct,
    _h6_arc_tracker,
    _h7_goal_carryover,
    _h8_theory_of_mind,
    _h9_repair_bridge,
    _h10_hedge_mirror,
    _h11_recency_weighting,
    _h12_anaphora_bind,
    _h13_active_tom,
    _h14_cadence_mirror,
    _h15_curiosity_hook,
    _h16_belief_update,
    _h17_listening_cost,
    _h18_emotion_cadence,
    _h19_script_awareness,
    _h20_counterfactual,
    _h21_calibration_drift,
    _h22_memory_consolidation,
    _h23_salience_decay,
    _h24_audience_modeling,
]


def main() -> int:
    print("\n========== nrs_human_layer_probe ==========")
    rows: List[Dict[str, Any]] = []
    for fn in CHECKS:
        try:
            rows.append(fn())
        except Exception as e:
            rows.append({"name": fn.__name__, "pass": False,
                         "detail": {"exception": repr(e)}})

    fails = [r for r in rows if not r["pass"]]
    for r in rows:
        flag = "PASS" if r["pass"] else "FAIL"
        print(f"  {flag:>4}  {r['name']:<22}  {json.dumps(r['detail'])[:120]}")

    if fails:
        print(f"\nFAIL — {len(fails)}/{len(rows)} human-layer checks failed.")
        return 1
    print(f"\nAll {len(rows)} human-layer checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
