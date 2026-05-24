#!/usr/bin/env python3
"""Audit: every entry in :mod:`platform-api/default_impulses` must

  (a) be discoverable via ``GET /peers/groups/smart:bots/members``,
  (b) be addable to a user-named group via
      ``PUT /peers/groups/{gid}/members/{uid}`` (i.e. the impulse-
      prefixed UID must resolve through ``_resolve_member_uid``),
  (c) round-trip back through ``GET /peers/groups/{gid}/members``.

Honours the same env vars as the rest of the parity suite:

    NRS_BASE      platform-api base URL (default http://localhost:8000)
    NRS_JWT_PATH  file with a JWT for the probing user
                  (default /tmp/.nrs_jwt)
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("NRS_BASE", "http://localhost:8000")
JWT_PATH = os.environ.get("NRS_JWT_PATH", "/tmp/.nrs_jwt")
try:
    JWT = open(JWT_PATH).read().strip()
except FileNotFoundError:
    JWT = ""

H = {"Authorization": f"Bearer {JWT}", "content-type": "application/json"}

EXPECTED = [
    "impulse-nrs",
    "impulse-coach",
    "impulse-scholar",
    "impulse-coder",
    "impulse-medic",
    "impulse-counsel",
    "impulse-cfo",
    "impulse-tutor",
    "impulse-critic",
    "impulse-translator",
    "impulse-brainstorm",
    "impulse-summarizer",
]


def hit(method: str, path: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{BASE}{path}", data=data, headers=H, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            payload = r.read()
            if r.status == 204 or not payload:
                return r.status, {}
            return r.status, json.loads(payload)
    except urllib.error.HTTPError as e:
        try:
            body_err = json.loads(e.read() or b"{}")
        except Exception:
            body_err = {}
        return e.code, body_err


def _uids_from(members_payload: dict) -> set[str]:
    members = members_payload.get("members", []) if isinstance(members_payload, dict) else []
    out: set[str] = set()
    for m in members:
        if not isinstance(m, dict):
            continue
        for key in ("member_uid", "user_id", "uid"):
            v = m.get(key)
            if isinstance(v, str) and v:
                out.add(v)
                break
    return out


def main() -> int:
    if not JWT:
        print(f"FAIL: no JWT at {JWT_PATH}; mint one before invoking this audit.")
        return 1

    st, body = hit("GET", "/peers/groups/smart:bots/members")
    print(f"smart:bots -> HTTP {st}")
    seen = _uids_from(body)
    print(f"  smart:bots members: {len(seen)}")
    missing_in_smart = [u for u in EXPECTED if u not in seen]
    extra_in_smart = sorted(seen - set(EXPECTED))
    print(f"  missing: {missing_in_smart}")
    print(f"  extra:   {extra_in_smart}")

    st_g, gbody = hit("POST", "/peers/groups", {"name": "ImpulseAudit"})
    gid = gbody.get("id") if isinstance(gbody, dict) else None
    print(f"\ncreate folder -> HTTP {st_g}  gid={gid}")
    if not gid:
        return 1

    results: list[tuple[str, int]] = []
    for uid in EXPECTED:
        st_add, _ = hit("PUT", f"/peers/groups/{gid}/members/{uid}")
        results.append((uid, st_add))

    print("\nadd-to-folder:")
    ok = 0
    for uid, s in results:
        flag = "OK  " if s in (200, 204) else "FAIL"
        if s in (200, 204):
            ok += 1
        print(f"  {flag} {uid}  http={s}")

    st_l, lbody = hit("GET", f"/peers/groups/{gid}/members")
    got = _uids_from(lbody)
    print(f"\nfolder list -> HTTP {st_l}  {len(got)}/{len(EXPECTED)} expected")
    missing_in_folder = [u for u in EXPECTED if u not in got]
    if missing_in_folder:
        print(f"  missing: {missing_in_folder}")

    hit("DELETE", f"/peers/groups/{gid}")

    passed = (
        not missing_in_smart
        and ok == len(EXPECTED)
        and not missing_in_folder
    )
    print(f"\nAUDIT {'PASS' if passed else 'FAIL'}  (added {ok}/{len(EXPECTED)})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
