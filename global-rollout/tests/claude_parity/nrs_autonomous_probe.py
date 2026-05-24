"""Autonomous + impulse + brain-chat probe.

Checks the surfaces the chat UI uses for AI-to-AI / friend-of-friend
interactions but that none of the parity batteries actually exercise:

  GET  /peers/me           — self profile
  GET  /peers/impulses     — system AI brains visible in the slideout
  GET  /peers/online       — presence
  GET  /peers/friends      — buddy list
  POST /peers/{h}/converse — seed an autonomous AI-to-AI conversation
  POST /peers/{h}/brain/chat — single-turn brain chat

Reports each route's status + a tiny content snapshot so we know
which parts of the autonomous surface are wired up locally.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import requests

try:
    JWT = open("/tmp/.nrs_jwt").read().strip()
except FileNotFoundError:
    JWT = ""
BASE = os.environ.get("NRS_BASE", "http://localhost:8000")
HEADERS = {"Authorization": f"Bearer {JWT}", "content-type": "application/json"}


def hit(method: str, path: str, body: Dict[str, Any] | None = None) -> Dict[str, Any]:
    url = f"{BASE}{path}"
    try:
        r = requests.request(
            method, url, headers=HEADERS,
            data=json.dumps(body) if body is not None else None,
            timeout=30,
        )
    except Exception as e:
        return {"status": "exc", "detail": str(e)}
    out: Dict[str, Any] = {"status": r.status_code}
    try:
        out["body"] = r.json()
    except Exception:
        out["body"] = r.text[:200]
    return out


def main() -> None:
    print("--- self ---")
    me = hit("GET", "/peers/me")
    print(f"  /peers/me -> {me['status']}")
    me_handle = (me.get("body") or {}).get("simple_name") if isinstance(me.get("body"), dict) else None
    print(f"  my handle: @{me_handle}")

    print("\n--- system bots (impulses) ---")
    imp = hit("GET", "/peers/impulses")
    print(f"  /peers/impulses -> {imp['status']}")
    impulses = (imp.get("body") or {}).get("impulses") if isinstance(imp.get("body"), dict) else None
    if isinstance(impulses, list):
        for i in impulses[:8]:
            print(f"    @{i.get('simple_name'):<22} {i.get('public_name')}")
        print(f"  total impulses: {len(impulses)}")
    else:
        print(f"  body snippet: {str(imp.get('body'))[:200]}")

    print("\n--- friends + presence ---")
    friends = hit("GET", "/peers/friends")
    online = hit("GET", "/peers/online")
    print(f"  /peers/friends -> {friends['status']}  ({len(friends.get('body') or [])} friends)")
    print(f"  /peers/online  -> {online['status']}  ({len(online.get('body') or [])} online)")

    # Pick a target for autonomous tests: prefer a real impulse
    target_handle = None
    if isinstance(impulses, list) and impulses:
        target_handle = impulses[0].get("simple_name")
    if not target_handle:
        target_handle = "nrs"  # fallback that any local seed should expose

    print(f"\n--- autonomous conversation seed against @{target_handle} ---")
    convo = hit("POST", f"/peers/{target_handle}/converse", {
        "opening_message": "Hey — let's plan a 3-day weekend trip with $400 budget.",
        "rounds": 3,
    })
    print(f"  POST /peers/{target_handle}/converse -> {convo['status']}")
    if convo["status"] == 200:
        body = convo["body"]
        print(f"    convo_id: {body.get('id')}")
        print(f"    rounds:   {body.get('rounds')}")
        print(f"    status:   {body.get('status')}")
        print(f"    opening:  {(body.get('opening') or '')[:80]}")
    else:
        print(f"    body: {str(convo.get('body'))[:300]}")

    # NEW: peer-groups CRUD smoke
    print("\n--- peer-groups CRUD smoke ---")
    g0 = hit("GET", "/peers/groups")
    print(f"  GET /peers/groups -> {g0['status']}")
    if g0["status"] == 200:
        body = g0["body"]
        print(f"    smart groups: {[g['id'] for g in body.get('smart', [])]}")
        print(f"    user groups:  {len(body.get('user', []))}")
    create = hit("POST", "/peers/groups", {"name": "Probe Folder"})
    print(f"  POST /peers/groups -> {create['status']}")
    new_gid = (create.get("body") or {}).get("id") if isinstance(create.get("body"), dict) else None
    if new_gid:
        # add a bot to it
        bot = "impulse:nrs-core"
        if isinstance(impulses, list) and impulses:
            bot = impulses[0].get("user_id", bot)
        addm = hit("PUT", f"/peers/groups/{new_gid}/members/{bot}")
        print(f"  PUT  /peers/groups/{new_gid}/members/{bot} -> {addm['status']}")
        members = hit("GET", f"/peers/groups/{new_gid}/members")
        print(f"  GET  /peers/groups/{new_gid}/members -> {members['status']}  members={len((members.get('body') or {}).get('members', []))}")
        rm = hit("DELETE", f"/peers/groups/{new_gid}")
        print(f"  DELETE /peers/groups/{new_gid} -> {rm['status']}")

    print(f"\n--- single-turn brain chat against @{target_handle} ---")
    brain = hit("POST", f"/peers/{target_handle}/brain/chat", {
        "message": "what would you actually do with a $400 long-weekend budget?",
    })
    print(f"  POST /peers/{target_handle}/brain/chat -> {brain['status']}")
    if brain["status"] == 200 and isinstance(brain.get("body"), dict):
        b = brain["body"]
        print(f"    from_brain:    {b.get('from_brain')}")
        print(f"    is_friend:     {b.get('is_friend')}")
        print(f"    is_public:     {b.get('is_public_brain')}")
        print(f"    response[:240]: {(b.get('response') or '')[:240]}")
    else:
        print(f"    body: {str(brain.get('body'))[:300]}")


if __name__ == "__main__":
    main()
