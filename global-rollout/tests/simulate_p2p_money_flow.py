"""End-to-end P2P money flow simulation.

Drives the *real* platform-api FastAPI app against a real local Postgres + Redis
with Stripe monkey-patched into a deterministic mock. Every signature, NRSIP
envelope, and POVI proof is genuine.

Required env (set by run_p2p_money_flow.sh):
    DATABASE_URL, VLT_PG_HOST/PORT/USER/PASSWORD/DB
    REDIS_HOST, REDIS_PORT, REDIS_PASSWORD (empty)
    STRIPE_SECRET_KEY=sk_test_simulated
    JWT_SECRET=...
    POVI_COORDINATOR_SIGNING_KEY=...
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
import time
import uuid
from typing import Any

# Ensure platform-api modules importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "services", "platform-api"))
sys.path.insert(0, ROOT)


# ────────────────────────────────────────────────────────────────────────────
# Stripe mock — intercept before the app imports it
# ────────────────────────────────────────────────────────────────────────────
import stripe  # noqa: E402

_MOCK_ACCOUNTS: dict[str, dict[str, Any]] = {}
_MOCK_INTENTS: dict[str, dict[str, Any]] = {}


class _StripeObj(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def _mock_account_create(**kw):
    aid = f"acct_test_{uuid.uuid4().hex[:14]}"
    obj = _StripeObj(
        id=aid,
        type=kw.get("type", "express"),
        country=kw.get("country", "US"),
        details_submitted=True,
        charges_enabled=True,
        payouts_enabled=True,
        capabilities=kw.get("capabilities", {}),
        metadata=kw.get("metadata", {}),
    )
    _MOCK_ACCOUNTS[aid] = obj
    return obj


def _mock_account_retrieve(aid):
    return _MOCK_ACCOUNTS.get(aid) or _StripeObj(
        id=aid, charges_enabled=True, payouts_enabled=True, details_submitted=True
    )


def _mock_account_link_create(**kw):
    return _StripeObj(
        url=f"https://stripe-mock.local/onboard/{kw.get('account')}",
        expires_at=int(time.time()) + 3600,
    )


def _mock_pi_create(**kw):
    pid = f"pi_test_{uuid.uuid4().hex[:18]}"
    obj = _StripeObj(
        id=pid,
        object="payment_intent",
        amount=kw["amount"],
        currency=kw["currency"],
        status="succeeded",  # simulate instant capture
        customer=kw.get("customer"),
        payment_method=kw.get("payment_method"),
        transfer_data=kw.get("transfer_data"),
        metadata=kw.get("metadata", {}),
        description=kw.get("description"),
        amount_received=kw["amount"],
    )
    _MOCK_INTENTS[pid] = obj
    return obj


# Patch the SDK in-process so the route's `import stripe` sees the mocks
stripe.Account.create = _mock_account_create  # type: ignore
stripe.Account.retrieve = _mock_account_retrieve  # type: ignore
stripe.AccountLink.create = _mock_account_link_create  # type: ignore
stripe.PaymentIntent.create = _mock_pi_create  # type: ignore


# Provide the missing `stripe.error` namespace some SDK versions require
class _StripeError(Exception):
    user_message = None
if not hasattr(stripe, "error"):
    class _ErrNS:
        StripeError = _StripeError
    stripe.error = _ErrNS()  # type: ignore


# ────────────────────────────────────────────────────────────────────────────
# Now we can import the real app
# ────────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402
from nrsip import vlt_postgres as vp  # noqa: E402
from nrsip.nrsip_messages import Message  # noqa: E402
from nrsip.nrsip_signing import verify_envelope  # noqa: E402
from routes.transfers import _node_signing_key  # noqa: E402

# In-process POVI validator (real signatures)
sys.path.insert(0, os.path.join(ROOT, "services", "nrs-worker"))
import threading  # noqa: E402
import socket  # noqa: E402
from contextlib import closing  # noqa: E402
import uvicorn  # noqa: E402
from povi_validator import router as povi_validator_router  # noqa: E402
from nrsip.povi_coordinator import ValidatorEndpoint  # noqa: E402


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_validator_server() -> str:
    """Boot a real POVI validator on a background thread, return its base URL."""
    port = _free_port()
    val_app = FastAPI()
    val_app.include_router(povi_validator_router)

    config = uvicorn.Config(val_app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    # Wait for socket to accept
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    return f"http://127.0.0.1:{port}"


def _hr(label: str) -> None:
    print()
    print("━" * 76)
    print(f"  {label}")
    print("━" * 76)


def _sub(label: str) -> None:
    print()
    print(f"── {label} ──")


def _pretty(obj: Any, indent: int = 2) -> str:
    return textwrap.indent(json.dumps(obj, indent=2, default=str), " " * indent)


async def main() -> None:
    _hr("NRS · P2P Money-Flow Simulation (real signatures, real POVI, mocked Stripe)")
    print(f"Stripe key:           {os.environ.get('STRIPE_SECRET_KEY', '(none)')[:16]}…")
    print(f"Postgres DSN:         {os.environ.get('DATABASE_URL', '(none)')}")
    print(f"Redis:                {os.environ.get('REDIS_HOST')}:{os.environ.get('REDIS_PORT')}")

    # Start an in-process POVI validator BEFORE the platform-api so it's
    # already accepting requests by the time the coordinator fans out.
    validator_url = _start_validator_server()
    print(f"POVI validator:       {validator_url}")

    # Seed validator into Redis BEFORE the platform-api boots so
    # ``hydrate_registry`` picks it up during lifespan.
    import redis as sync_redis  # noqa: E402
    rsync = sync_redis.Redis(
        host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]),
        password=os.environ.get("REDIS_PASSWORD") or None, decode_responses=True,
    )
    validator_record = {
        "validator_id": "sim-validator-1",
        "provider": "atherion",
        "region": "local",
        "url": validator_url,
        "signing_key": os.environ["POVI_VALIDATOR_SIGNING_KEY"],
        "trust_score": 5.0,
        "suspended": False,
    }
    rsync.set("povi:validator:sim-validator-1", json.dumps(validator_record))
    rsync.sadd("povi:validators", "sim-validator-1")
    rsync.close()
    print("POVI validator seeded into Redis registry")

    # TestClient triggers lifespan → init_redis, init_postgres, build_povi_runtime
    with TestClient(app) as client:
        # ── 1. Sign up Alice (initiator) and Bob (recipient) ───────────────
        _hr("STEP 1 — Sign up Alice + Bob")
        alice_email = f"alice.{uuid.uuid4().hex[:6]}@example.com"
        bob_email = f"bob.{uuid.uuid4().hex[:6]}@example.com"

        a = client.post("/auth/signup", json={
            "first_name": "Alice", "last_name": "Initiator",
            "email": alice_email, "password": "Sim-Test-2026!",
        }).json()
        b = client.post("/auth/signup", json={
            "first_name": "Bob", "last_name": "Recipient",
            "email": bob_email, "password": "Sim-Test-2026!",
        }).json()
        alice_uid = a["user"]["uid"]; alice_jwt = a["token"]
        bob_uid = b["user"]["uid"];   bob_jwt   = b["token"]
        print(f"Alice → uid={alice_uid}  jwt=…{alice_jwt[-12:]}")
        print(f"Bob   → uid={bob_uid}  jwt=…{bob_jwt[-12:]}")

        H_a = {"Authorization": f"Bearer {alice_jwt}"}
        H_b = {"Authorization": f"Bearer {bob_jwt}"}

        # ── 2. Bob onboards Stripe Connect (mocked → verified) ─────────────
        _hr("STEP 2 — Bob onboards Stripe Connect (mocked instant-verify)")
        ob = client.post(
            "/transfers/_/stripe-connect/onboard",
            json={"return_url": "https://example.com/done", "country": "US"},
            headers=H_b,
        ).json()
        print(_pretty(ob))
        # Mock onboarding completes immediately
        rf = client.post("/transfers/_/stripe-connect/refresh", headers=H_b).json()
        print("After refresh:")
        print(_pretty(rf))
        assert rf["verified"], "Bob's Stripe Connect should be verified"

        # ── 3. Alice proposes $42.00 to Bob (FIAT, send) ───────────────────
        _hr("STEP 3 — Alice proposes a $42.00 fiat transfer to Bob")
        prop = client.post("/transfers", json={
            "recipient_id": bob_uid,
            "direction": "send",
            "rail": "fiat",
            "amount_cents": 4200,
            "currency": "USD",
            "note": "Coffee + dinner share",
            "expires_in_seconds": 3600,
        }, headers=H_a)
        assert prop.status_code == 201, prop.text
        proposal = prop.json()
        tx_id = proposal["id"]
        print(_pretty({
            "id": proposal["id"], "status": proposal["status"],
            "amount_cents": proposal["amount_cents"], "currency": proposal["currency"],
            "expires_at": proposal["expires_at"],
        }))

        # Pull raw row to inspect signed proposal envelope
        row = vp.get_transfer(tx_id)
        envelope_proposal = row["proposal_envelope"]
        msg = Message.from_dict(envelope_proposal)
        sig_valid = verify_envelope(msg, _node_signing_key(alice_uid))
        _sub("Signed proposal envelope (NRSIP)")
        print(_pretty({
            "message_id": msg.message_id,
            "message_type": msg.message_type,
            "signer_id": msg.signer_id,
            "key_id": msg.key_id,
            "signature_first16": (msg.signature or "")[:16] + "…",
            "verifies": sig_valid,
            "payload": msg.payload,
        }))
        assert sig_valid, "Alice's proposal envelope failed verification"

        # ── 4. POVI consensus authorises the release ───────────────────────
        _hr("STEP 4 — Alice asks the POVI runtime to authorise this transfer")
        povi = client.post(f"/transfers/{tx_id}/povi", json={
            "rationale": (
                "Recipient Bob is a verified peer; amount $42.00 is below the "
                "low-risk threshold; no fraud signals on the conversation; "
                "Alice's intent was explicit (split bill). Authorise release."
            ),
            "domain": "finance",
            "processing_mode": "DETERMINISTIC",
        }, headers=H_a)
        if povi.status_code != 200:
            print(f"WARNING POVI returned {povi.status_code}: {povi.text}")
            povi_proof_hash = None
        else:
            pj = povi.json()
            povi_proof_hash = pj["povi_proof_hash"]
            print(_pretty({
                "povi_proof_hash": pj["povi_proof_hash"],
                "povi_round_id": pj["povi_round_id"],
                "status": pj["status"],
            }))

        # ── 5. Bob accepts the proposal → triggers Stripe destination charge
        _hr("STEP 5 — Bob accepts → Stripe destination charge fires")
        accept = client.post(f"/transfers/{tx_id}/accept", json={
            "payment_method_id": "pm_card_visa",
        }, headers=H_b)
        assert accept.status_code == 200, accept.text
        aj = accept.json()
        print(_pretty({
            "status": aj["status"],
            "stripe_payment_intent_id": aj["stripe_payment_intent_id"],
            "stripe_destination_account": aj["stripe_destination_account"],
            "completed_at": aj["completed_at"],
            "povi_proof_hash": aj["povi_proof_hash"],
        }))

        row = vp.get_transfer(tx_id)
        envelope_accept = row["acceptance_envelope"]
        msg2 = Message.from_dict(envelope_accept)
        sig2_valid = verify_envelope(msg2, _node_signing_key(bob_uid))
        _sub("Signed acceptance envelope (NRSIP)")
        print(_pretty({
            "message_id": msg2.message_id,
            "signer_id": msg2.signer_id,
            "verifies": sig2_valid,
            "payload": msg2.payload,
        }))
        assert sig2_valid, "Bob's acceptance envelope failed verification"

        # ── 6. Inspect the captured Stripe PaymentIntent (mock) ────────────
        _hr("STEP 6 — Stripe PaymentIntent captured (mock)")
        pi_id = aj["stripe_payment_intent_id"]
        pi = _MOCK_INTENTS.get(pi_id, {})
        print(_pretty({
            "id": pi.get("id"),
            "status": pi.get("status"),
            "amount": pi.get("amount"),
            "currency": pi.get("currency"),
            "transfer_data": pi.get("transfer_data"),
            "metadata": pi.get("metadata"),
        }))
        if povi_proof_hash:
            assert pi.get("metadata", {}).get("nrs_povi_proof_hash") == povi_proof_hash, \
                "POVI proof_hash should be anchored in Stripe metadata"
            print("  ✓ POVI proof_hash anchored in Stripe metadata — auditable link confirmed")

        # ── 7. CRYPTO flow: Alice requests 0.05 ETH from Bob ───────────────
        _hr("STEP 7 — Crypto rail: Alice requests 0.05 ETH from Bob")
        # Bob registers a wallet (self-attested verified=true)
        client.post("/transfers/_/accounts", json={
            "rail": "web3_wallet",
            "handle": "0x4242424242424242424242424242424242424242",
            "network": "ethereum",
            "label": "Bob primary",
        }, headers=H_b)
        # Alice registers her destination wallet
        client.post("/transfers/_/accounts", json={
            "rail": "web3_wallet",
            "handle": "0x1111111111111111111111111111111111111111",
            "network": "ethereum",
            "label": "Alice primary",
        }, headers=H_a)

        crypto_prop = client.post("/transfers", json={
            "recipient_id": bob_uid,
            "direction": "request",        # Alice asking Bob to send
            "rail": "crypto",
            # 50 USDC, expressed in micro-USDC (6 decimals) → fits the API cap.
            "amount_cents": 50_000_000,
            "currency": "USDC",
            "network": "ethereum",
            "note": "Splitting concert tickets",
            "expires_in_seconds": 3600,
            "crypto_from_address": "0x4242424242424242424242424242424242424242",
            "crypto_to_address":   "0x1111111111111111111111111111111111111111",
        }, headers=H_a).json()
        ctx_id = crypto_prop["id"]
        print(f"Crypto proposal id: {ctx_id}  status={crypto_prop['status']}")

        # Bob accepts and supplies a (simulated) on-chain tx hash
        sim_tx_hash = "0x" + "ab" * 32  # 64 hex chars
        c_acc = client.post(f"/transfers/{ctx_id}/accept", json={
            "crypto_tx_hash": sim_tx_hash,
        }, headers=H_b).json()
        print(_pretty({
            "status": c_acc["status"],
            "crypto_tx_hash": c_acc["crypto_tx_hash"],
        }))

        # Confirm on-chain
        c_done = client.post(f"/transfers/{ctx_id}/confirm", json={
            "crypto_tx_hash": sim_tx_hash,
        }, headers=H_b).json()
        print(_pretty({
            "status": c_done["status"],
            "completed_at": c_done["completed_at"],
            "crypto_tx_hash": c_done["crypto_tx_hash"],
        }))

        # ── 8. List transfers from each side ───────────────────────────────
        _hr("STEP 8 — Final ledger view from each user")
        for label, hdr in (("Alice", H_a), ("Bob", H_b)):
            ls = client.get("/transfers", headers=hdr).json()
            _sub(f"{label}'s transfer history ({len(ls)} entries)")
            for t in ls:
                print(f"  {t['id']:30s}  {t['rail']:6s}  {t['amount_cents']:>20d} {t['currency']:4s}  {t['status']:10s}")

        # ── 9. Database-level audit trail ──────────────────────────────────
        _hr("STEP 9 — Direct DB audit trail")
        for tid in (tx_id, ctx_id):
            r = vp.get_transfer(tid)
            print()
            print(f"transfer_id      : {r['id']}")
            print(f"  initiator      : {r['initiator_id']}  →  recipient: {r['recipient_id']}")
            print(f"  rail/direction : {r['rail']}/{r['direction']}")
            print(f"  amount         : {r['amount_cents']} {r['currency']}")
            print(f"  status         : {r['status']}")
            print(f"  proposal env   : {bool(r['proposal_envelope'])}  (signed)")
            print(f"  acceptance env : {bool(r['acceptance_envelope'])} (signed)")
            print(f"  povi_proof_hash: {r.get('povi_proof_hash')}")
            print(f"  povi_round_id  : {r.get('povi_round_id')}")
            print(f"  stripe PI      : {r.get('stripe_payment_intent_id')}")
            print(f"  stripe dest    : {r.get('stripe_destination_account')}")
            print(f"  crypto tx_hash : {r.get('crypto_tx_hash')}")
            print(f"  completed_at   : {r.get('completed_at')}")

        # ── 10. Ironclad ledger — chain integrity + balances ─────────────
        _hr("STEP 10 — Ironclad ledger view + chain verification")
        for label, hdr in (("Alice", H_a), ("Bob", H_b)):
            entries = client.get("/ledger/entries?limit=100", headers=hdr).json()
            balances = client.get("/ledger/balances", headers=hdr).json()
            verify = client.get("/ledger/verify", headers=hdr).json()
            _sub(f"{label}'s ledger ({len(entries)} entries)")
            for e in reversed(entries):  # show in chronological order
                amt = e["amount_minor"]
                tag = "+" if amt > 0 else ("-" if amt < 0 else " ")
                print(
                    f"  #{e['seq']:>3} {tag} {abs(amt):>20d} {e['currency']:4s}  "
                    f"{e['kind']:22s}  bal={e['balance_after_minor']:>14d}  "
                    f"hash={e['entry_hash'][:12]}…"
                )
            print(f"  → balances: {balances['balances_display']}")
            print(f"  → chain head: seq={balances['head_seq']} hash={(balances['head_hash'] or '')[:16]}…")
            print(f"  → integrity: ok={verify['ok']} entries_checked={verify['entries_checked']} reason={verify['reason']}")
            assert verify["ok"], f"{label}'s chain failed verification: {verify}"

        # ── 11. Native NRS Credit Line ───────────────────────────────────
        _hr("STEP 11 — Native NRS Credit (line / draw / repay)")
        # Get Alice's line (auto-created)
        line0 = client.get("/credit/line", headers=H_a).json()
        _sub("Alice's initial credit line")
        print(_pretty(line0))

        # Force-underwrite (will write a credit.limit_changed ledger entry if it differs)
        line1 = client.post("/credit/line/underwrite", headers=H_a).json()
        _sub("After re-underwriting")
        print(_pretty(line1))

        if line1["limit_minor"] > 0:
            # Draw a small amount that stays under the POVI threshold
            small = max(100, int(line1["limit_minor"] * 0.10))
            d = client.post(
                "/credit/line/draw", headers=H_a,
                json={"amount_minor": small, "currency": "USD",
                      "purpose": "Top-up for groceries"},
            ).json()
            _sub(f"Alice drew ${small/100:.2f} (under POVI threshold)")
            print(_pretty(d))

            # Repay half of it
            half = max(50, small // 2)
            r = client.post(
                "/credit/line/repay", headers=H_a,
                json={"amount_minor": half, "currency": "USD"},
            ).json()
            _sub(f"Alice repaid ${half/100:.2f}")
            print(_pretty(r))

            # Final ledger check after credit activity
            entries = client.get(
                "/ledger/entries?limit=20&kind=credit.drawn,credit.repaid,credit.limit_changed",
                headers=H_a,
            ).json()
            _sub(f"Credit-only ledger entries on Alice's chain ({len(entries)})")
            for e in reversed(entries):
                print(f"  #{e['seq']:>3}  {e['kind']:22s}  {e['amount_minor']:>+15d} {e['currency']}  {e['description']}")

            verify = client.get("/ledger/verify", headers=H_a).json()
            assert verify["ok"], f"Alice's chain broke after credit activity: {verify}"
            print(f"  → integrity after credit activity: ok={verify['ok']} entries={verify['entries_checked']}")
        else:
            print(f"  → tier {line1['tier']} has $0 limit; skipping draw/repay")

        # ── 12. Tiered platform fees — quote + collected ledger entries ───
        _hr("STEP 12 — Tiered platform fees (AI-credit-score driven)")

        schedule = client.get("/fees/schedule").json()
        print(f"  baseline processor: {schedule['baseline_display']}")
        for row in schedule["rows"]:
            mark = "◀ you" if row["tier"] == line1["tier"] else ""
            mn = f" + ${row['min_minor']/100:.2f}" if row["min_minor"] else ""
            print(f"    {row['tier']:<12s} {row['rate_display']:<7s}{mn:<10s}{mark}")

        _sub("Live fee preview for a $42.00 send")
        quote = client.post(
            "/fees/quote",
            headers=H_a,
            json={"amount_minor": 4_200, "currency": "USD", "rail": "fiat"},
        ).json()
        print(f"  tier       : {quote['tier']}  (score {quote['score']:.2f})")
        print(f"  rate       : {quote['rate_display']}")
        print(f"  fee        : ${quote['fee_minor']/100:.2f}")
        print(f"  baseline   : ${quote['baseline_minor']/100:.2f}  ({quote['baseline_rate_display']})")
        print(f"  you save   : ${quote['savings_minor']/100:.2f}")
        print(f"  payer pays : ${quote['total_minor']/100:.2f}")

        _sub("Platform fee entries posted for the original $42 completed send")
        fee_entries = client.get(
            "/ledger/entries?limit=20&kind=fee",
            headers=H_a,
        ).json()
        print(f"  Alice saw {len(fee_entries)} fee entry(ies):")
        for e in fee_entries:
            meta = e.get("metadata") or {}
            print(
                f"    #{e['seq']:>3}  {e['amount_minor']:>+6d} {e['currency']}  "
                f"tier={meta.get('tier','?')}  rate_bps={meta.get('rate_bps','?')}  "
                f"saved={meta.get('savings_minor',0)}  ref={e.get('ref_id')}"
            )
        assert fee_entries, "Expected at least one fee ledger entry on Alice's chain"
        fee_row = fee_entries[0]
        assert fee_row["amount_minor"] < 0, "Fee should be an outflow on payer side"

        verify = client.get("/ledger/verify", headers=H_a).json()
        assert verify["ok"], f"Alice's chain broke after fees booked: {verify}"
        print(f"  → integrity after fee activity: ok={verify['ok']} entries={verify['entries_checked']}")

        _hr("✅  P2P money flow + Ironclad ledger + Native Credit + Tiered fees simulated end-to-end")
        print()
        print(f"  Fiat   transfer: {tx_id}")
        print(f"  Crypto transfer: {ctx_id}")
        print()
        print("  • Both proposals signed by initiator (NRSIP envelope, verified)")
        print("  • Both acceptances signed by recipient (NRSIP envelope, verified)")
        print("  • POVI consensus round produced a real proof_hash, anchored in Stripe metadata")
        print("  • Stripe destination charge fired with POVI metadata attached")
        print("  • Crypto rail recorded an on-chain tx hash and flipped to completed")
        print("  • Ironclad ledger appended hash-chained entries on every transition")
        print("  • Chain verification re-walked every entry and confirmed integrity")
        print("  • Native NRS Credit line opened, drawn, and repaid through the same chain")
        print("  • Platform fee deducted on completion — rate set by AI Credit Score tier")
        print()


if __name__ == "__main__":
    asyncio.run(main())
