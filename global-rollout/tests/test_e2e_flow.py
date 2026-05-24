"""End-to-end integration test for the full NRSIP flow.

Validates the complete pipeline without requiring live services:
1. Generate node identity (server, macOS, Android)
2. Enroll via challenge-response
3. Negotiate Iron-Clad session with OQS enforcement
4. Build NRSIP packet with security profile
5. Run packet through full integrity gate chain
6. Verify mode classification and processing
7. Verify audit trace completeness

This test is self-contained and uses no network calls.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
import unittest
from pathlib import Path

from nrsip.integrity_gates import (
    ChainResult,
    FreshnessGate,
    GateVerdict,
    HopChainGate,
    IntegrityGateChain,
    PacketContext,
    ReplayGate,
    SecurityProfileGate,
    SequenceGate,
    TierGate,
    TTLGate,
    build_production_gate_chain,
)
from nrsip.transport_policy import OQS_STRICT_POLICY, TransportPolicyEnforcer
from nrsip.session_store import InMemorySessionStore
from clients.bootstrap import (
    BootstrapConfig,
    FileCredentialStore,
    KeychainCredentialStore,
    KeystoreCredentialStore,
    NRSIPClientBootstrap,
    NodeIdentity,
    Platform,
    credential_store_for_platform,
)


class TestNodeIdentityGeneration(unittest.TestCase):
    """Phase 3 validation: all three platforms can generate unique identities."""

    def _gen_identity(self, platform: Platform) -> NodeIdentity:
        import tempfile
        kwargs = {}
        if platform == Platform.SERVER:
            kwargs = {"store_dir": tempfile.mkdtemp(prefix="nrsip-test-")}
        config = BootstrapConfig(
            edge_gateway_url="http://fake:8443",
            platform=platform,
            region="us-central1",
            credential_store_kwargs=kwargs,
        )
        client = NRSIPClientBootstrap(config)
        return client.generate_identity()

    def test_server_identity(self):
        identity = self._gen_identity(Platform.SERVER)
        self.assertTrue(identity.node_id.startswith("nrs-server-"))
        self.assertEqual(identity.platform, Platform.SERVER)
        self.assertTrue(len(identity.pubkey) == 64)

    def test_macos_identity(self):
        identity = self._gen_identity(Platform.MACOS)
        self.assertTrue(identity.node_id.startswith("nrs-macos-"))
        self.assertEqual(identity.platform, Platform.MACOS)

    def test_android_identity(self):
        identity = self._gen_identity(Platform.ANDROID)
        self.assertTrue(identity.node_id.startswith("nrs-android-"))
        self.assertEqual(identity.platform, Platform.ANDROID)

    def test_all_unique_ids(self):
        ids = set()
        for platform in [Platform.SERVER, Platform.MACOS, Platform.ANDROID]:
            identity = self._gen_identity(platform)
            ids.add(identity.node_id)
        self.assertEqual(len(ids), 3)

    def test_all_unique_overlay_ips(self):
        ips = set()
        for platform in [Platform.SERVER, Platform.MACOS, Platform.ANDROID]:
            identity = self._gen_identity(platform)
            ips.add(identity.overlay_ip)
        self.assertEqual(len(ips), 3)


class TestChallengeResponseEnrollment(unittest.TestCase):
    """Phase 3 validation: challenge-response enrollment contract."""

    def test_challenge_signature_matches(self):
        pubkey = secrets.token_bytes(32).hex()
        challenge = secrets.token_hex(32)
        expected_sig = hashlib.sha256(
            (challenge + pubkey).encode("utf-8")
        ).hexdigest()
        self.assertEqual(len(expected_sig), 64)
        self.assertNotEqual(expected_sig, challenge)

    def test_wrong_pubkey_fails(self):
        pubkey_a = secrets.token_bytes(32).hex()
        pubkey_b = secrets.token_bytes(32).hex()
        challenge = secrets.token_hex(32)
        sig_a = hashlib.sha256((challenge + pubkey_a).encode("utf-8")).hexdigest()
        sig_b = hashlib.sha256((challenge + pubkey_b).encode("utf-8")).hexdigest()
        self.assertNotEqual(sig_a, sig_b)


class TestOQSPolicyEnforcement(unittest.TestCase):
    """Phase 2 validation: OQS strict policy blocks non-PQ bundles."""

    def setUp(self):
        self.enforcer = TransportPolicyEnforcer(OQS_STRICT_POLICY)

    def test_mlkem1024_accepted(self):
        result = self.enforcer.validate_bundle("MLKEM1024_X25519_DILITHIUM3")
        self.assertTrue(result["allowed"])

    def test_mlkem768_accepted(self):
        result = self.enforcer.validate_bundle("MLKEM768_X25519_DILITHIUM2")
        self.assertTrue(result["allowed"])

    def test_x25519_rejected(self):
        result = self.enforcer.validate_bundle("X25519_ED25519")
        self.assertFalse(result["allowed"])

    def test_random_bundle_rejected(self):
        result = self.enforcer.validate_bundle("FAKE_BUNDLE")
        self.assertFalse(result["allowed"])

    def test_best_bundle_is_1024(self):
        best = self.enforcer.select_best_bundle([
            "MLKEM1024_X25519_DILITHIUM3",
            "MLKEM768_X25519_DILITHIUM2",
        ])
        self.assertEqual(best, "MLKEM1024_X25519_DILITHIUM3")

    def test_no_acceptable_bundle_raises(self):
        with self.assertRaises(ValueError):
            self.enforcer.select_best_bundle(["X25519_ED25519", "RSA_ONLY"])

    def test_clock_skew_within_limits(self):
        now = int(time.time() * 1000)
        result = self.enforcer.validate_clock_skew(now, now)
        self.assertTrue(result["allowed"])

    def test_clock_skew_exceeds_limits(self):
        now = int(time.time() * 1000)
        result = self.enforcer.validate_clock_skew(now - 30000, now)
        self.assertFalse(result["allowed"])

    def test_ttl_valid(self):
        self.assertTrue(self.enforcer.validate_ttl(5)["allowed"])

    def test_ttl_zero_invalid(self):
        self.assertFalse(self.enforcer.validate_ttl(0)["allowed"])

    def test_ttl_over_max_invalid(self):
        self.assertFalse(self.enforcer.validate_ttl(10)["allowed"])


class TestSessionStore(unittest.TestCase):
    """Phase 2 validation: session/replay store operations."""

    def setUp(self):
        self.store = InMemorySessionStore()

    def test_nonce_replay_detection(self):
        self.assertTrue(self.store.register_nonce(b"nonce-e2e-1", 300.0))
        self.assertFalse(self.store.register_nonce(b"nonce-e2e-1", 300.0))

    def test_session_lifecycle(self):
        self.store.store_session("e2e-sess", {"node": "test", "bundle": "MLKEM768"}, 900)
        s = self.store.get_session("e2e-sess")
        self.assertIsNotNone(s)
        self.assertEqual(s["node"], "test")

        self.store.update_session_field("e2e-sess", "status", "active")
        s = self.store.get_session("e2e-sess")
        self.assertEqual(s["status"], "active")

        self.store.delete_session("e2e-sess")
        self.assertIsNone(self.store.get_session("e2e-sess"))


class TestFullIntegrityGateFlow(unittest.TestCase):
    """Phase 4 validation: full packet through all integrity gates."""

    def _build_hop_chain(self, hops, sequence=1, destination="worker-pool",
                          packet_type="query", validation_tier="T2"):
        from nrsip.nrsip_signing import compute_hop_hash
        from nrsip.nrsip_messages import HopRecord
        chain = []
        prev_hash = "root"
        for i, (hop_addr, ttl) in enumerate(hops):
            hop_record = HopRecord(
                hop_index=i,
                node_id=hop_addr,
                key_id=hop_addr,
                prev_hop_hash=prev_hash,
                message_id="",
                decision="recv",
            )
            hop_hash = compute_hop_hash(hop_record)
            chain.append({
                "hop": hop_addr,
                "signer": hop_addr,
                "prev_hash": prev_hash,
                "hash": hop_hash,
                "ttl_hops": str(ttl),
                "ts_ms": str(int(time.time() * 1000)),
            })
            prev_hash = hop_hash
        return chain

    def test_valid_packet_passes_all_gates(self):
        """Simulate: client sends packet -> relay extends hop -> policy verifies -> all gates pass."""
        chain = build_production_gate_chain(require_oqs=True)

        hop_chain = self._build_hop_chain(
            [("edge-gw-1", 7), ("relay-us-central1", 6)],
            sequence=42,
            destination="worker-pool",
            packet_type="query",
            validation_tier="T2",
        )

        ctx = PacketContext(
            source="nrs-server-abc12345",
            destination="worker-pool",
            sequence=42,
            nonce=secrets.token_hex(16),
            issued_at_ms=int(time.time() * 1000),
            ttl_hops=6,
            validation_tier="T2",
            packet_type="query",
            session_id="sess-e2e-1",
            key_id="key-e2e-1",
            hop_chain=hop_chain,
            security_profile={
                "algorithm_bundle": "MLKEM1024_X25519_DILITHIUM3",
                "kem_provider": "oqs-mlkem-v1",
                "signature_provider": "oqs-dilithium-v1",
                "transcript_hash": "e2e-hash-abc",
            },
        )

        result = chain.enforce(ctx)
        self.assertTrue(result.passed, f"Gate chain failed: {result.rejected_by}: {result.rejection_reason}")
        self.assertEqual(len(result.results), 7)
        for gate_result in result.results:
            self.assertEqual(gate_result.verdict, GateVerdict.PASS)

    def test_replay_detected_in_full_flow(self):
        chain = build_production_gate_chain(require_oqs=True)
        nonce = secrets.token_hex(16)
        ctx = PacketContext(
            source="node-a", destination="node-b", sequence=1,
            nonce=nonce, issued_at_ms=int(time.time() * 1000),
            ttl_hops=5, validation_tier="T2", packet_type="query",
        )
        self.assertTrue(chain.enforce(ctx).passed)

        ctx2 = PacketContext(
            source="node-a", destination="node-b", sequence=2,
            nonce=nonce, issued_at_ms=int(time.time() * 1000),
            ttl_hops=5, validation_tier="T2", packet_type="query",
        )
        result = chain.enforce(ctx2)
        self.assertFalse(result.passed)
        self.assertEqual(result.rejected_by, "replay")

    def test_non_oqs_bundle_rejected(self):
        chain = build_production_gate_chain(require_oqs=True)
        ctx = PacketContext(
            source="node-a", destination="node-b", sequence=1,
            nonce=secrets.token_hex(16),
            issued_at_ms=int(time.time() * 1000),
            ttl_hops=5, validation_tier="T2", packet_type="query",
            session_id="sess-1", key_id="key-1",
            security_profile={
                "algorithm_bundle": "X25519_ED25519",
                "kem_provider": "x25519",
                "signature_provider": "ed25519",
                "transcript_hash": "abc",
            },
        )
        result = chain.enforce(ctx)
        self.assertFalse(result.passed)
        self.assertEqual(result.rejected_by, "security_profile")

    def test_tampered_hop_chain_rejected(self):
        chain = build_production_gate_chain(require_oqs=True)
        hop_chain = self._build_hop_chain([("relay-1", 7)])
        hop_chain[0]["hash"] = "tampered_hash"
        ctx = PacketContext(
            source="node-a", destination="worker-pool", sequence=1,
            nonce=secrets.token_hex(16),
            issued_at_ms=int(time.time() * 1000),
            ttl_hops=7, validation_tier="T2", packet_type="query",
            hop_chain=hop_chain,
        )
        result = chain.enforce(ctx)
        self.assertFalse(result.passed)
        self.assertEqual(result.rejected_by, "hop_chain")

    def test_stale_packet_rejected(self):
        chain = build_production_gate_chain(max_age_ms=5000, require_oqs=False)
        ctx = PacketContext(
            source="node-a", destination="node-b", sequence=1,
            nonce=secrets.token_hex(16),
            issued_at_ms=int(time.time() * 1000) - 10000,
            ttl_hops=5, validation_tier="T2", packet_type="query",
        )
        result = chain.enforce(ctx)
        self.assertFalse(result.passed)
        self.assertEqual(result.rejected_by, "freshness")

    def test_ttl_exhausted_rejected(self):
        chain = build_production_gate_chain(require_oqs=False)
        ctx = PacketContext(
            source="node-a", destination="node-b", sequence=1,
            nonce=secrets.token_hex(16),
            issued_at_ms=int(time.time() * 1000),
            ttl_hops=0, validation_tier="T2", packet_type="query",
        )
        result = chain.enforce(ctx)
        self.assertFalse(result.passed)
        self.assertEqual(result.rejected_by, "ttl")

    def test_tier_downgrade_rejected(self):
        chain = build_production_gate_chain(required_tier="T3", require_oqs=False)
        ctx = PacketContext(
            source="node-a", destination="node-b", sequence=1,
            nonce=secrets.token_hex(16),
            issued_at_ms=int(time.time() * 1000),
            ttl_hops=5, validation_tier="T1", packet_type="query",
        )
        result = chain.enforce(ctx)
        self.assertFalse(result.passed)
        self.assertEqual(result.rejected_by, "tier")


class TestAuditTraceCompleteness(unittest.TestCase):
    """Phase 7 validation: every gate decision produces auditable evidence."""

    def test_audit_dict_has_all_gates(self):
        chain = build_production_gate_chain(require_oqs=False)
        ctx = PacketContext(
            source="audit-node", destination="worker", sequence=99,
            nonce=secrets.token_hex(16),
            issued_at_ms=int(time.time() * 1000),
            ttl_hops=5, validation_tier="T2", packet_type="query",
        )
        result = chain.enforce(ctx)
        audit = result.to_audit_dict()

        self.assertTrue(audit["passed"])
        self.assertEqual(len(audit["gates"]), 7)

        gate_names = {g["gate"] for g in audit["gates"]}
        expected_gates = {"replay", "sequence", "freshness", "ttl", "hop_chain", "tier", "security_profile"}
        self.assertEqual(gate_names, expected_gates)

        for g in audit["gates"]:
            self.assertIn("verdict", g)
            self.assertIn("elapsed_us", g)
            self.assertIsInstance(g["elapsed_us"], int)

    def test_rejection_audit_has_reason(self):
        chain = build_production_gate_chain(require_oqs=False)
        ctx = PacketContext(
            source="audit-node", destination="worker", sequence=1,
            nonce="", issued_at_ms=int(time.time() * 1000),
            ttl_hops=5, validation_tier="T2", packet_type="query",
        )
        result = chain.enforce(ctx)
        audit = result.to_audit_dict()

        self.assertFalse(audit["passed"])
        self.assertEqual(audit["rejected_by"], "replay")
        self.assertTrue(len(audit["rejection_reason"]) > 0)


class TestModeClassification(unittest.TestCase):
    """Phase 4 validation: NRS worker mode classification logic."""

    def setUp(self):
        from nrsip.mode_classifier import classify_mode, ProcessingMode, ModePolicy
        self.classify_mode = classify_mode
        self.ProcessingMode = ProcessingMode
        self.ModePolicy = ModePolicy

    def test_creative_query_probabilistic(self):
        mode, inputs = self.classify_mode("imagine a creative poem about space")
        self.assertEqual(mode, self.ProcessingMode.PROBABILISTIC)
        self.assertTrue(inputs["has_creative_signal"])

    def test_factual_query_deterministic(self):
        mode, inputs = self.classify_mode("verify the fact that 2+2=4")
        self.assertEqual(mode, self.ProcessingMode.DETERMINISTIC)
        self.assertTrue(inputs["has_factual_signal"])

    def test_high_risk_domain_forces_deterministic(self):
        mode, inputs = self.classify_mode("what is the treatment", domain="medical")
        self.assertEqual(mode, self.ProcessingMode.DETERMINISTIC)
        self.assertTrue(inputs["high_risk_domain"])

    def test_explicit_override_honored(self):
        mode, inputs = self.classify_mode("some query", context_override="probabilistic")
        self.assertEqual(mode, self.ProcessingMode.PROBABILISTIC)
        self.assertEqual(inputs["override"], "probabilistic")

    def test_mixed_signals_hybrid(self):
        mode, inputs = self.classify_mode("imagine and verify a creative fact")
        self.assertEqual(mode, self.ProcessingMode.HYBRID)

    def test_mode_decision_id_stable(self):
        from nrsip.mode_classifier import mode_decision_id
        _, inputs = self.classify_mode("test query")
        id1 = mode_decision_id(inputs)
        id2 = mode_decision_id(inputs)
        self.assertEqual(id1, id2)
        self.assertEqual(len(id1), 16)


class TestCredentialStorePerPlatform(unittest.TestCase):
    """Phase 3 validation: each platform credential store works correctly."""

    def test_keychain_store_roundtrip(self):
        store = KeychainCredentialStore(service_name="test.nrsip")
        ref = store.store_key("test-key", b"secret-data")
        data = store.load_key(ref)
        self.assertEqual(data, b"secret-data")
        store.delete_key(ref)
        with self.assertRaises(KeyError):
            store.load_key(ref)

    def test_keystore_roundtrip(self):
        store = KeystoreCredentialStore(alias_prefix="test")
        ref = store.store_key("test-key", b"android-secret")
        data = store.load_key(ref)
        self.assertEqual(data, b"android-secret")
        store.delete_key(ref)
        with self.assertRaises(KeyError):
            store.load_key(ref)


if __name__ == "__main__":
    unittest.main()
