"""NRS Core Subsystems Test Suite.

Validates all critical subsystems: H-Score, VLT, Provenance, NRSIP Transport,
Content Safety, Emergency, Binary Neurons, PRISM Gate, Ironclad Session, and
Hardware Determinism.

Usage:
    python -m unittest tests.test_core_subsystems -v
"""

from __future__ import annotations
import hashlib
import os
import time
import unittest

from nrsip.nrs_core import HScoreCalculator, HScoreResult, VLT, VLTLayer, EvictionPolicy
from nrsip.provenance import (
    ProvenanceBuilder, AppendOnlyProvenanceLog, ProvenanceEvent, SectionLabel,
)
from nrsip.nrsip_messages import (
    Message, NRSIPHeader, PaymentHeader, QoSPriority, ValidationTier,
)
from nrsip.nrsip_signing import (
    sign_envelope, verify_envelope, KeyResolver,
    sign_hop, verify_hop, sign_payment_proof, verify_payment_proof,
)
from nrsip.nrsip_transport import (
    TransportPipeline, QoSScheduler, NonceLedger, build_signed_message,
)
from nrsip.content_safety import ContentSafetyEngine, Severity, PolicyAction
from nrsip.emergency import (
    Killswitch, QuarantineManager, GuardedPipeline, EmergencyLevel,
)
from nrsip.binary_neurons import BinaryNeuronBank
from nrsip.prism_gate import PRISMGate
from nrsip.ironclad_session import IroncladSession
from nrsip.hardware_determinism import (
    HardwareDeterminismController, HardwareProfile, HardwareReading,
)


# ───────────────────────────────────────────────────────────────────
# 1. H-Score
# ───────────────────────────────────────────────────────────────────


class TestHScore(unittest.TestCase):
    """H-Score calculation: entropy, consistency, entailment → verdict."""

    def setUp(self):
        self.calc = HScoreCalculator()

    def test_identical_samples_high_score(self):
        samples = ["The capital of France is Paris."] * 5
        result = self.calc.score(
            query="What is the capital of France?",
            response="The capital of France is Paris.",
            samples=samples,
            ground_truth=["Paris is the capital of France."],
        )
        self.assertIsInstance(result, HScoreResult)
        self.assertGreaterEqual(result.consistency, 0.9)
        self.assertGreaterEqual(result.score, 0.5)

    def test_diverse_samples_lower_score(self):
        samples = [
            "Paris is the capital.",
            "London is the capital.",
            "Berlin is the capital.",
            "Tokyo is the capital.",
            "Rome is the capital.",
        ]
        result = self.calc.score(
            query="What is the capital of France?",
            response="The capital of France is Paris.",
            samples=samples,
        )
        uniform = self.calc.compute_consistency(["same"] * 5)
        self.assertLess(result.consistency, uniform)

    def test_empty_ground_truth(self):
        result = self.calc.score(
            query="Tell me something.",
            response="Here is something.",
            samples=["Here is something.", "Here is something."],
            ground_truth=[],
        )
        self.assertIsInstance(result.score, float)
        self.assertIn(result.verdict, (
            "validated", "acceptable", "suspicious", "hallucination",
        ))

    def test_hallucination_verdict(self):
        result = self.calc.score(
            query="What year did WW2 end?",
            response="World War 2 ended in 2035.",
            samples=[
                "WW2 ended in 1999.",
                "WW2 ended in 2010.",
                "WW2 ended in 2035.",
                "WW2 ended in 1888.",
            ],
            ground_truth=["World War 2 ended in 1945."],
        )
        if result.score < 0.5:
            self.assertEqual(result.verdict, "hallucination")


# ───────────────────────────────────────────────────────────────────
# 2. VLT (Verifiable Layered Trust)
# ───────────────────────────────────────────────────────────────────


class TestVLT(unittest.TestCase):
    """VLT layer hierarchy, search, eviction, and immutability."""

    def setUp(self):
        self.vlt = VLT(l1_capacity=4, l2_capacity=8, l3_capacity=16)

    def test_store_and_recall(self):
        self.vlt.store("fact-1", "neurons fire", layer=VLTLayer.L2_SESSION,
                       domain="neuroscience", confidence=0.9)
        items = self.vlt.search(domain="neuroscience")
        self.assertTrue(any(it.key == "fact-1" for it in items))
        self.assertEqual(self.vlt.recall("fact-1", layer=VLTLayer.L2_SESSION),
                         "neurons fire")

    def test_l4_immutability(self):
        self.vlt.store("axiom-1", "e=mc2", layer=VLTLayer.L4_ARCHIVAL,
                       immutable=True, confidence=1.0)
        self.assertEqual(self.vlt.recall("axiom-1", layer=VLTLayer.L4_ARCHIVAL),
                         "e=mc2")
        with self.assertRaises(Exception):
            self.vlt.store("axiom-1", "overwrite", layer=VLTLayer.L4_ARCHIVAL,
                           immutable=True, confidence=1.0)

    def test_eviction_policy(self):
        for i in range(10):
            self.vlt.store(f"eph-{i}", f"val-{i}", layer=VLTLayer.L1_EPHEMERAL)
        self.assertLessEqual(self.vlt.layer_size(VLTLayer.L1_EPHEMERAL), 4)

    def test_layer_hierarchy(self):
        layers = [VLTLayer.L1_EPHEMERAL, VLTLayer.L2_SESSION,
                  VLTLayer.L3_PERSISTENT, VLTLayer.L4_ARCHIVAL]
        for i, layer in enumerate(layers[:-1]):
            self.assertLess(layer.value, layers[i + 1].value)


# ───────────────────────────────────────────────────────────────────
# 3. Provenance Chain
# ───────────────────────────────────────────────────────────────────


class TestProvenanceChain(unittest.TestCase):
    """Append-only provenance log integrity and builder finalization."""

    def test_chain_integrity(self):
        log = AppendOnlyProvenanceLog()
        log.append("mode_select", {"mode": "deterministic"})
        log.append("lobe_fire", {"lobe": "analytical"})
        log.append("h_score", {"score": 0.92})
        ok, msg = log.verify_chain()
        self.assertTrue(ok, msg)

    def test_tamper_detection(self):
        log = AppendOnlyProvenanceLog()
        log.append("step_a", {"x": 1})
        log.append("step_b", {"x": 2})
        log.events[0].data["x"] = 999
        ok, _ = log.verify_chain()
        self.assertFalse(ok)

    def test_builder_finalizes_with_signature(self):
        builder = ProvenanceBuilder(query="test query", instance_id="node-1")
        builder.set_mode("deterministic", "T2", "science")
        builder.set_h_score(0.88, "validated")
        builder.set_confidence(0.88)

        def _signer(data: bytes) -> str:
            return hashlib.sha256(data).hexdigest()

        chain = builder.finalize("The answer is 42.", signing_fn=_signer)
        self.assertTrue(chain.verify(verify_fn=lambda d, s: s == hashlib.sha256(d).hexdigest()))

    def test_event_log_linkage(self):
        log = AppendOnlyProvenanceLog()
        for i in range(5):
            log.append(f"evt_{i}", {"i": i})
        events = log.events
        for idx in range(1, len(events)):
            recomputed_prev = events[idx - 1].compute_hash()
            self.assertEqual(events[idx].prev_hash, recomputed_prev,
                             f"Broken linkage at index {idx}")


# ───────────────────────────────────────────────────────────────────
# 4. NRSIP Transport
# ───────────────────────────────────────────────────────────────────


_SIGNING_KEY = "test-secret-key-256bit-000000001"
_KEY_ID = "key-alpha"
_NODE_A = "node-alpha"
_NODE_B = "node-beta"


def _make_pipeline(node_id=_NODE_A, key_id=_KEY_ID, signing_key=_SIGNING_KEY,
                   supported_tiers=None):
    resolver = KeyResolver()
    resolver.register(key_id, signing_key)
    tiers = supported_tiers or {ValidationTier.T1_CORTICAL}
    return TransportPipeline(
        node_id=node_id, node_key_id=key_id, signing_key=signing_key,
        key_resolver=resolver, supported_tiers=tiers,
    )


class TestNRSIPTransport(unittest.TestCase):
    """NRSIP envelope signing, hop chains, nonce replay, TTL, tier gating."""

    def test_sign_and_verify_envelope(self):
        msg = build_signed_message(
            signer_id=_NODE_A, key_id=_KEY_ID, signing_key=_SIGNING_KEY,
            payload={"query": "hello"},
        )
        self.assertTrue(verify_envelope(msg, _SIGNING_KEY))
        msg.payload["query"] = "tampered"
        self.assertFalse(verify_envelope(msg, _SIGNING_KEY))

    def test_hop_chain_linkage(self):
        from nrsip.nrsip_signing import verify_hop_chain_linkage
        from nrsip.nrsip_messages import HopRecord

        prev_hash = ""
        hops = []
        for i in range(4):
            hop = HopRecord(
                hop_index=i, node_id=f"n{i}", key_id=_KEY_ID,
                prev_hop_hash=prev_hash, message_id="msg-001",
            )
            hop = sign_hop(hop, _SIGNING_KEY)
            prev_hash = hop.hop_hash
            hops.append(hop)

        ok, reason = verify_hop_chain_linkage(hops)
        self.assertTrue(ok, reason)

    def test_nonce_replay_rejection(self):
        pipeline = _make_pipeline()
        shared_nonce = "replay-nonce-fixed-001"
        pay1 = PaymentHeader(
            nonce=shared_nonce, payer_address=_NODE_A,
            proof_key_id=_KEY_ID,
        )
        pay1.proof_signature = sign_payment_proof(pay1, _SIGNING_KEY)
        msg1 = build_signed_message(
            signer_id=_NODE_A, key_id=_KEY_ID, signing_key=_SIGNING_KEY,
            payload={"q": "first"}, payment=pay1,
        )
        r1 = pipeline.receive(msg1)
        self.assertTrue(r1.accepted, r1.reason)

        pay2 = PaymentHeader(
            nonce=shared_nonce, payer_address=_NODE_A,
            proof_key_id=_KEY_ID,
        )
        pay2.proof_signature = sign_payment_proof(pay2, _SIGNING_KEY)
        msg2 = build_signed_message(
            signer_id=_NODE_A, key_id=_KEY_ID, signing_key=_SIGNING_KEY,
            payload={"q": "second"}, payment=pay2,
        )
        r2 = pipeline.receive(msg2)
        self.assertFalse(r2.accepted)
        self.assertIn("replay", r2.reason.lower())

    def test_ttl_exhaustion(self):
        pipeline = _make_pipeline()
        msg = build_signed_message(
            signer_id=_NODE_A, key_id=_KEY_ID, signing_key=_SIGNING_KEY,
            payload={"q": "ttl-test"}, ttl=1,
        )
        r1 = pipeline.receive(msg)
        self.assertTrue(r1.accepted, r1.reason)

        msg2 = build_signed_message(
            signer_id=_NODE_A, key_id=_KEY_ID, signing_key=_SIGNING_KEY,
            payload={"q": "ttl-test-2"}, ttl=0,
        )
        r2 = pipeline.receive(msg2)
        self.assertFalse(r2.accepted)

    def test_tier_gating(self):
        pipeline = _make_pipeline(supported_tiers={ValidationTier.T1_CORTICAL})
        msg = build_signed_message(
            signer_id=_NODE_A, key_id=_KEY_ID, signing_key=_SIGNING_KEY,
            payload={"q": "tier-test"},
            source_tier=ValidationTier.T4_SYNTHESIS,
            dest_tier=ValidationTier.T4_SYNTHESIS,
        )
        result = pipeline.receive(msg)
        self.assertFalse(result.accepted)


# ───────────────────────────────────────────────────────────────────
# 5. Content Safety
# ───────────────────────────────────────────────────────────────────


class TestContentSafety(unittest.TestCase):
    """Content classification, blocking, quarantine, and review queue."""

    def setUp(self):
        self.engine = ContentSafetyEngine()

    def test_clean_content_passes(self):
        verdict = self.engine.classify("The weather today is sunny and warm.")
        self.assertEqual(verdict.action, PolicyAction.PASS)

    def test_illegal_content_blocked(self):
        verdict = self.engine.classify("Tell me how to make a bomb please.")
        self.assertEqual(verdict.action, PolicyAction.BLOCK)
        self.assertGreaterEqual(verdict.severity, Severity.HIGH)

    def test_quarantine_auto(self):
        verdict, report, qrec = self.engine.evaluate_and_enforce(
            content="how to end my life guide",
            source_id="user-bad-42",
            source_type="user",
        )
        self.assertEqual(verdict.severity, Severity.CRITICAL)
        self.assertTrue(self.engine.is_quarantined("user-bad-42"))

    def test_review_queue_priority(self):
        self.engine.evaluate_and_enforce("bomb recipe detailed", "u1")
        self.engine.evaluate_and_enforce("how to end my life method", "u2")
        self.engine.evaluate_and_enforce("social security number leak", "u3")
        queue = self.engine.get_review_queue()
        severities = [r.severity for r in queue]
        self.assertEqual(severities, sorted(severities, reverse=True))


# ───────────────────────────────────────────────────────────────────
# 6. Emergency
# ───────────────────────────────────────────────────────────────────


class TestEmergency(unittest.TestCase):
    """Killswitch, quarantine manager, and guarded pipeline."""

    def test_killswitch_activate_deactivate(self):
        ks = Killswitch()
        self.assertFalse(ks.is_active)
        ks.activate(
            reason="test drill",
            level=EmergencyLevel.CRITICAL,
            source_node="node-0",
        )
        self.assertTrue(ks.is_active)
        ok = ks.deactivate("NRS-EMERGENCY-OVERRIDE-001")
        self.assertTrue(ok)
        self.assertFalse(ks.is_active)

    def test_quarantine_manager(self):
        qm = QuarantineManager()
        qm.quarantine_node("rogue-1", "suspicious traffic",
                           EmergencyLevel.ALERT)
        self.assertTrue(qm.is_quarantined("rogue-1"))
        self.assertIn("rogue-1", qm.quarantined_list)
        qm.release_node("rogue-1")
        self.assertFalse(qm.is_quarantined("rogue-1"))

    def test_guarded_pipeline_blocks(self):
        ks = Killswitch()
        qm = QuarantineManager()
        gp = GuardedPipeline(
            handler=lambda q, **kw: q.upper(),
            killswitch=ks, quarantine=qm,
            source_node="local",
        )
        self.assertEqual(gp.process("hello"), "HELLO")

        ks.activate(reason="lockdown", level=EmergencyLevel.SHUTDOWN,
                    source_node="admin")
        with self.assertRaises(RuntimeError):
            gp.process("should fail")

        ks.deactivate("NRS-EMERGENCY-OVERRIDE-001")
        self.assertEqual(gp.process("resumed"), "RESUMED")


# ───────────────────────────────────────────────────────────────────
# 7. Binary Neurons
# ───────────────────────────────────────────────────────────────────


class TestBinaryNeurons(unittest.TestCase):
    """Binary neuron bank: fire count, determinism, and diversity."""

    def setUp(self):
        self.bank = BinaryNeuronBank(total_neurons=1000, active_k=10)

    def test_fire_returns_k_active(self):
        result = self.bank.fire("What is quantum entanglement?")
        self.assertEqual(result.active_count, 10)
        self.assertEqual(len(result.active_ids), 10)

    def test_determinism(self):
        query = "Explain photosynthesis."
        r1 = self.bank.fire(query)
        r2 = self.bank.fire(query)
        self.assertEqual(r1.active_ids, r2.active_ids)
        self.assertEqual(r1.query_hash, r2.query_hash)

    def test_different_queries_different_activations(self):
        r_a = self.bank.fire("How does gravity work?")
        r_b = self.bank.fire("What is the recipe for sourdough bread?")
        self.assertNotEqual(r_a.active_ids, r_b.active_ids)


# ───────────────────────────────────────────────────────────────────
# 8. PRISM Gate
# ───────────────────────────────────────────────────────────────────


class TestPRISMGate(unittest.TestCase):
    """PRISM confidence gate: layer thresholds and dedup."""

    def setUp(self):
        self.gate = PRISMGate()

    def test_l2_passes_with_sufficient_confidence(self):
        verdict = self.gate.evaluate(
            value="some fact",
            layer="L2",
            confidence=0.6,
            domain="general",
            source="unit-test",
        )
        self.assertTrue(verdict.passed, verdict.reason)

    def test_l4_blocks_low_confidence(self):
        verdict = self.gate.evaluate(
            value="uncertain claim",
            layer="L4",
            confidence=0.5,
            domain="general",
            source="unit-test",
        )
        self.assertFalse(verdict.passed)
        self.assertIn("confidence", verdict.reason.lower())

    def test_dedup_blocks_repeat(self):
        payload = "identical payload for dedup test"
        v1 = self.gate.evaluate(payload, "L1", 0.9,
                                "general", "unit-test")
        v2 = self.gate.evaluate(payload, "L1", 0.9,
                                "general", "unit-test")
        self.assertFalse(v2.passed)
        self.assertIn("dedup", v2.reason.lower())


# ───────────────────────────────────────────────────────────────────
# 9. Ironclad Session
# ───────────────────────────────────────────────────────────────────


class TestIroncladSession(unittest.TestCase):
    """Symmetric session encryption: roundtrip, tamper, rotation."""

    def setUp(self):
        self.session = IroncladSession(
            shared_secret=b"super-secret-key-material-32byte",
            key_rotation_interval=5,
        )

    def test_encrypt_decrypt_roundtrip(self):
        plaintext = b"NRS deterministic inference output."
        ct = self.session.encrypt(plaintext)
        self.assertEqual(self.session.decrypt(ct), plaintext)

    def test_tamper_detection(self):
        ct = self.session.encrypt(b"sensitive data")
        tampered = bytearray(ct)
        tampered[len(tampered) // 2] ^= 0xFF
        with self.assertRaises(ValueError):
            self.session.decrypt(bytes(tampered))

    def test_key_rotation(self):
        initial_key_id = self.session.current_key.key_id
        ciphertexts = []
        for i in range(8):
            ciphertexts.append(self.session.encrypt(f"msg-{i}".encode()))
        rotated_key_id = self.session.current_key.key_id
        self.assertNotEqual(initial_key_id, rotated_key_id)
        for i, ct in enumerate(ciphertexts):
            self.assertEqual(self.session.decrypt(ct), f"msg-{i}".encode())


# ───────────────────────────────────────────────────────────────────
# 10. Hardware Determinism
# ───────────────────────────────────────────────────────────────────


class TestHardwareDeterminism(unittest.TestCase):
    """Hardware envelope validation: power, frequency, DVFS."""

    def setUp(self):
        self.profile = HardwareProfile(
            gpu_power_budget_w=200.0,
            gpu_power_tolerance_w=10.0,
            gpu_frequency_mhz=1600,
            gpu_frequency_tolerance_mhz=50,
            dvfs_enabled=False,
        )
        self.controller = HardwareDeterminismController(self.profile)

    def test_valid_reading(self):
        reading = HardwareReading(power_w=200.0, frequency_mhz=1600,
                                  temperature_c=65.0)
        violations = self.controller.validate_reading(reading)
        self.assertEqual(len(violations), 0)

    def test_over_power(self):
        reading = HardwareReading(power_w=250.0, frequency_mhz=1600,
                                  temperature_c=70.0)
        violations = self.controller.validate_reading(reading)
        self.assertTrue(any(v.violation_type == "power_envelope"
                            for v in violations))

    def test_dvfs_violation(self):
        dvfs_profile = HardwareProfile(
            gpu_power_budget_w=200.0,
            gpu_power_tolerance_w=10.0,
            gpu_frequency_mhz=1600,
            gpu_frequency_tolerance_mhz=50,
            dvfs_enabled=True,
        )
        ctrl = HardwareDeterminismController(dvfs_profile)
        reading = HardwareReading(power_w=200.0, frequency_mhz=1600,
                                  temperature_c=60.0)
        violations = ctrl.validate_reading(reading)
        self.assertTrue(any(v.violation_type == "dvfs_active"
                            for v in violations))


# ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main()
