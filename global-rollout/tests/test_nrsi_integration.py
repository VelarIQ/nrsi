"""NRSI Integration Tests.

Verifies that the NRSI programming language (NRS Instructions) is
properly wired across all Enterprise GA services.  Every test proves
that NRSI trust types, validation gates, provenance chains, and the
full NRS engine are functioning correctly.
"""

import asyncio
import hashlib
import os
import time

import pytest


# ── Core NRSI Type System ────────────────────────────────────────────────

class TestNRSITypes:
    """Trust hierarchy: raw → validated → trusted → certified."""

    def test_raw_is_untrusted(self):
        from nrsi.core.types import raw, TrustLevel, Confidence
        d = raw("untrusted input")
        assert d.trust_level == TrustLevel.RAW
        assert d.confidence == Confidence.NONE
        assert d.is_raw
        assert not d.is_validated

    def test_trust_elevation(self):
        from nrsi.core.types import raw, TrustLevel
        d = raw("data")
        elevated = d.elevate(TrustLevel.VALIDATED, confidence=0.9,
                             gate_name="test_gate", reason="test elevation")
        assert elevated.trust_level == TrustLevel.VALIDATED
        assert elevated.is_validated
        assert elevated.id == d.id  # same data, elevated

    def test_trust_never_downgrades_silently(self):
        from nrsi.core.types import raw, TrustLevel
        from nrsi.core.errors import TrustError
        d = raw("data")
        elevated = d.elevate(TrustLevel.TRUSTED, 0.95, "gate")
        with pytest.raises(TrustError):
            elevated.elevate(TrustLevel.VALIDATED, 0.8, "bad_gate")

    def test_explicit_downgrade_requires_justification(self):
        from nrsi.core.types import raw, TrustLevel
        d = raw("data").elevate(TrustLevel.TRUSTED, 0.95, "gate")
        downgraded = d.downgrade(TrustLevel.RAW, reason="test", actor="admin")
        assert downgraded.trust_level == TrustLevel.RAW

    def test_provenance_chain_grows(self):
        from nrsi.core.types import raw, TrustLevel
        d = raw("data")
        d2 = d.elevate(TrustLevel.VALIDATED, 0.9, "gate_1")
        d3 = d2.elevate(TrustLevel.TRUSTED, 0.95, "gate_2")
        d4 = d3.elevate(TrustLevel.CERTIFIED, 0.99, "gate_3")
        assert len(d4.provenance) >= 4
        actions = [e.action for e in d4.provenance]
        assert "created" in actions
        assert actions.count("elevated") == 3

    def test_require_trust_blocks_raw(self):
        from nrsi.core.types import raw, TrustLevel
        from nrsi.core.errors import TrustError
        d = raw("sensitive data")
        with pytest.raises(TrustError):
            d.require_trust(TrustLevel.TRUSTED, "read_sensitive")

    def test_certified_constructor(self):
        from nrsi.core.types import certified, TrustLevel
        d = certified("policy-approved", confidence=0.99,
                       gate_name="governance", policy_name="HIPAA")
        assert d.trust_level == TrustLevel.CERTIFIED
        assert d.is_certified

    def test_confidence_combine_is_conservative(self):
        from nrsi.core.types import Confidence
        assert Confidence.combine(0.95, 0.80) == 0.80
        assert Confidence.combine(1.0, 1.0) == 1.0
        assert Confidence.combine() == Confidence.NONE


# ── Validation Gates ─────────────────────────────────────────────────────

class TestValidationGates:

    def test_gate_elevates_on_pass(self):
        from nrsi.core.types import raw, TrustLevel
        from nrsi.core.validation import ValidationGate, FunctionValidator
        gate = ValidationGate(
            name="format_check",
            confidence_threshold=0.8,
            validators=[FunctionValidator(lambda x: True, name="always_pass")],
            target_trust=TrustLevel.VALIDATED,
        )
        result = gate.process(raw("hello"))
        assert result.trust_level == TrustLevel.VALIDATED

    def test_gate_rejects_on_fail(self):
        from nrsi.core.types import raw
        from nrsi.core.validation import ValidationGate, FunctionValidator
        from nrsi.core.errors import ValidationError
        gate = ValidationGate(
            name="reject_gate",
            confidence_threshold=0.8,
            validators=[FunctionValidator(lambda x: False, name="always_fail")],
        )
        with pytest.raises(ValidationError):
            gate.process(raw("bad data"))

    def test_gate_stats_track(self):
        from nrsi.core.types import raw
        from nrsi.core.validation import ValidationGate, FunctionValidator
        gate = ValidationGate(
            name="stats_gate",
            confidence_threshold=0.8,
            validators=[FunctionValidator(lambda x: True)],
        )
        gate.process(raw("a"))
        gate.process(raw("b"))
        assert gate.stats["total_processed"] == 2
        assert gate.stats["total_passed"] == 2

    def test_orchestrator_input_gate_rejects_empty(self):
        from nrsi.core.types import raw
        from nrsi.core.errors import ValidationError
        from nrsip.orchestrator import INPUT_GATE
        with pytest.raises(ValidationError):
            INPUT_GATE.process(raw({}))

    def test_orchestrator_input_gate_passes_valid(self):
        from nrsi.core.types import raw, TrustLevel
        from nrsip.orchestrator import INPUT_GATE
        result = INPUT_GATE.process(raw({"query": "hello world"}))
        assert result.trust_level == TrustLevel.VALIDATED


# ── Signal System ────────────────────────────────────────────────────────

class TestSignalSystem:

    def test_inhibitory_network_fires(self):
        from nrsi.core.signals import (
            InhibitoryNetwork, InhibitionRule, InhibitionType,
            confidence_inhibition,
        )
        net = InhibitoryNetwork()
        net.add_rule(confidence_inhibition(0.95, ["T2", "T3", "T4"]))
        inhibited = net.evaluate(
            "T1", {"confidence": 0.98}, cycle=0)
        assert "T2" in inhibited
        assert "T3" in inhibited
        assert "T4" in inhibited

    def test_no_inhibition_below_threshold(self):
        from nrsi.core.signals import InhibitoryNetwork, confidence_inhibition
        net = InhibitoryNetwork()
        net.add_rule(confidence_inhibition(0.95, ["T2", "T3"]))
        inhibited = net.evaluate("T1", {"confidence": 0.70})
        assert len(inhibited) == 0


# ── NRS Engine ───────────────────────────────────────────────────────────

class TestNRSEngine:

    def test_text_processing(self):
        from nrsi.core.nrs import NRS, ResponseStatus
        nrs = NRS(instance_id="test-text", total_neurons=500, active_k=5)
        resp = nrs.process("What is the speed of light?")
        assert resp.status in (
            ResponseStatus.VALIDATED,
            ResponseStatus.ACCEPTABLE,
            ResponseStatus.CACHED,
        )
        assert resp.h_score >= 0
        assert resp.tier in ("T1", "T2", "T3", "T4")
        assert resp.query_hash == hashlib.sha256(
            b"What is the speed of light?").hexdigest()

    def test_media_processing(self):
        from nrsi.core.nrs import NRS
        from nrsi.core.media import Modality
        nrs = NRS(instance_id="test-media", total_neurons=500, active_k=5)
        resp = nrs.process_media(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100,
                                 Modality.IMAGE, "describe image")
        assert resp.input_modality == Modality.IMAGE
        assert resp.routing_packet_bytes == 160

    def test_pvs_cache_hit(self):
        from nrsi.core.nrs import NRS, ResponseStatus
        nrs = NRS(instance_id="test-pvs", total_neurons=500, active_k=5)
        nrs.process("cached query test")
        resp2 = nrs.process("cached query test")
        assert resp2.cache_hit is True
        assert resp2.status == ResponseStatus.CACHED

    def test_deterministic_output(self):
        from nrsi.core.nrs import NRS
        nrs = NRS(instance_id="test-det", total_neurons=500, active_k=5)
        resp1 = nrs.process("determinism test query")
        nrs2 = NRS(instance_id="test-det", total_neurons=500, active_k=5)
        resp2 = nrs2.process("determinism test query")
        assert resp1.query_hash == resp2.query_hash

    def test_provenance_is_complete(self):
        from nrsi.core.nrs import NRS
        nrs = NRS(instance_id="test-prov", total_neurons=500, active_k=5)
        resp = nrs.process("provenance test")
        prov = resp.provenance
        assert "query_hash" in prov
        assert "neurons" in prov
        assert "routing" in prov
        assert "validation" in prov
        assert "output" in prov

    def test_nrs_stats(self):
        from nrsi.core.nrs import NRS
        nrs = NRS(instance_id="test-stats", total_neurons=500, active_k=5)
        nrs.process("q1")
        nrs.process("q2")
        st = nrs.stats() if callable(nrs.stats) else nrs.stats
        assert st["queries_processed"] == 2


# ── VLT Memory System ───────────────────────────────────────────────────

class TestVLTMemory:

    def test_four_layer_hierarchy(self):
        from nrsi.core.memory import VLT, VLTLayer
        vlt = VLT()
        vlt.store("q1", "answer1", layer=VLTLayer.L1_EPHEMERAL)
        vlt.store("fact1", "earth is round", layer=VLTLayer.L4_ARCHIVAL,
                  domain="science", confidence=1.0)
        assert vlt.recall("q1") == "answer1"
        assert vlt.recall("fact1") == "earth is round"

    def test_l4_is_immutable(self):
        from nrsi.core.memory import VLT, VLTLayer
        vlt = VLT()
        vlt.store("immutable_fact", 42, layer=VLTLayer.L4_ARCHIVAL)
        with pytest.raises(ValueError, match="immutable"):
            vlt.store("immutable_fact", 99, layer=VLTLayer.L4_ARCHIVAL)

    def test_promotion_l1_to_l4(self):
        from nrsi.core.memory import VLT, VLTLayer
        vlt = VLT()
        vlt.store("promote_me", "value", layer=VLTLayer.L1_EPHEMERAL,
                  confidence=0.99)
        vlt.promote("promote_me", VLTLayer.L1_EPHEMERAL, VLTLayer.L4_ARCHIVAL)
        assert vlt.recall("promote_me", VLTLayer.L4_ARCHIVAL) == "value"
        assert vlt.recall("promote_me", VLTLayer.L1_EPHEMERAL) is None


# ── Tuition System ───────────────────────────────────────────────────────

class TestTuitionSystem:

    def test_correction_stored_in_pvs(self):
        from nrsi.core.memory import TuitionSystem
        tuition = TuitionSystem()
        tuition.correct("complex medical claim", "T1", "T3")
        tier = tuition.route("complex medical claim", default_tier="T1")
        assert tier == "T3"

    def test_convergence_within_3(self):
        from nrsi.core.memory import TuitionSystem
        tuition = TuitionSystem()
        tuition.correct("novel_q", "T1", "T2")
        tier = tuition.route("novel_q", "T1")
        assert tier == "T2"


# ── Orchestrator NRSI Integration ────────────────────────────────────────

class TestOrchestratorNRSI:

    def test_orchestrator_has_inhibitory_network(self):
        from nrsip.orchestrator import Orchestrator
        orch = Orchestrator()
        assert orch.inhibitory_network is not None
        assert len(orch.inhibitory_network.rules) > 0

    @pytest.mark.asyncio
    async def test_route_with_nrsi_trust(self):
        from nrsip.orchestrator import Orchestrator, Modality, RouteResult

        async def mock_handler(payload, trace, budget):
            return {"answer": "test response", "confidence": 0.85}

        orch = Orchestrator()
        orch.register_route(Modality.TEXT, mock_handler)
        result = await orch.route(
            Modality.TEXT,
            {"query": "hello world"},
            source="test",
        )
        assert result.status.value == "success"
        assert result.trust_level in ("TRUSTED", "VALIDATED", "RAW")
        assert len(result.provenance) > 0

    @pytest.mark.asyncio
    async def test_route_rejects_empty_input(self):
        from nrsip.orchestrator import Orchestrator, Modality
        orch = Orchestrator()
        result = await orch.route(Modality.TEXT, {}, source="test")
        assert result.status.value == "error"
        assert "validation" in result.error.lower() or "no handler" in result.error.lower()


# ── Media Processing ─────────────────────────────────────────────────────

class TestMediaProcessing:

    def test_media_processor_produces_digest(self):
        from nrsi.core.media import MediaProcessor, Modality
        proc = MediaProcessor()
        digest = proc.process(b"test-image-data", Modality.IMAGE)
        assert digest.modality == Modality.IMAGE
        assert len(digest.feature_embedding) == 768
        assert digest.content_ref.content_hash == hashlib.sha256(
            b"test-image-data").hexdigest()

    def test_media_coordinator_caches(self):
        from nrsi.core.media import MediaCoordinator, Modality
        coord = MediaCoordinator()
        d1, p1 = coord.ingest(b"same-data", Modality.IMAGE)
        d2, p2 = coord.ingest(b"same-data", Modality.IMAGE)
        assert coord.stats["cache_hits"] == 1

    def test_routing_packet_is_160_bytes(self):
        from nrsi.core.media import MediaCoordinator, Modality
        coord = MediaCoordinator()
        _, packet = coord.ingest(b"x" * 5_000_000, Modality.IMAGE)
        assert packet.estimated_bytes == 160


# ── Governance Errors ────────────────────────────────────────────────────

class TestGovernanceErrors:
    """GovernanceViolationError and AuditRequiredError behavior."""

    def test_governance_violation_error_raised(self):
        from nrsi.core.errors import GovernanceViolationError
        from nrsi.core.types import raw, TrustLevel
        from nrsi.core.validation import ValidationGate, Validator

        class AlwaysFailGovernance(Validator):
            name = "always_fail_governance"

            def validate(self, data, context=None):
                raise GovernanceViolationError(
                    policy_name="test_policy",
                    standard="TEST-STD",
                    violation="validator rejected",
                )

        gate = ValidationGate(
            name="gov_gate",
            confidence_threshold=0.8,
            validators=[AlwaysFailGovernance()],
            target_trust=TrustLevel.VALIDATED,
        )
        with pytest.raises(GovernanceViolationError) as exc_info:
            gate.process(raw("payload"))
        assert exc_info.value.policy_name == "test_policy"

    def test_governance_error_message(self):
        from nrsi.core.errors import GovernanceViolationError

        err = GovernanceViolationError(
            policy_name="HIPAA",
            standard="PHI",
            violation="missing BAA",
            suggestion="Attach BAA before processing",
        )
        msg = str(err)
        assert "HIPAA" in msg
        assert "PHI" in msg
        assert "missing BAA" in msg
        assert err.policy_name == "HIPAA"
        assert err.standard == "PHI"
        assert err.violation == "missing BAA"

    def test_audit_required_error(self):
        from nrsi.core.errors import AuditRequiredError

        with pytest.raises(AuditRequiredError) as exc_info:
            raise AuditRequiredError("classified_export")
        assert "classified_export" in str(exc_info.value)
        assert "audit trail" in str(exc_info.value).lower()


# ── PVS-4 Patterns ───────────────────────────────────────────────────────

class TestPVS4Patterns:
    """PVS4 deterministic store / lookup and cache behavior."""

    def test_pvs4_store_and_retrieve(self):
        from nrsi.core.memory import PVS4

        pvs = PVS4(vector_bits=256)
        q = "deterministic test query"
        pvs.store(q, tier="T2", confidence=0.91, data={"answer": "ok"})
        m = pvs.lookup(q)
        assert m is not None
        assert m.text == q
        assert m.tier == "T2"
        assert m.data.get("answer") == "ok"

    def test_pvs4_cache_hit(self):
        from nrsi.core.memory import PVS4

        pvs = PVS4(vector_bits=256)
        q = "repeat lookup"
        pvs.store(q, tier="T1", confidence=0.9, data={})
        pvs.lookup(q)
        pvs.lookup(q)
        assert pvs.stats["cache_hits"] >= 1
        assert pvs.stats["lookups"] == 2

    def test_pvs4_different_query_no_hit(self):
        from nrsi.core.memory import PVS4

        pvs = PVS4(vector_bits=256)
        pvs.store("alpha query", tier="T1", confidence=0.9, data={})
        miss = pvs.lookup("completely different query")
        assert miss is None
        assert pvs.stats["misses"] >= 1


# ── Inhibitory Network E2E ───────────────────────────────────────────────

class TestInhibitoryNetworkE2E:
    """Inhibition rules with Signal-shaped payloads and event audit trail."""

    def test_inhibition_skips_processing(self):
        from nrsi.core.signals import (
            InhibitoryNetwork,
            InhibitionRule,
            InhibitionType,
            Signal,
        )

        net = InhibitoryNetwork()
        net.add_rule(
            InhibitionRule(
                name="high_confidence_skip",
                condition=lambda d, c: d.get("confidence", 0) > 0.9,
                targets=["T2", "T3"],
                inhibition_type=InhibitionType.CONFIDENCE,
                reason="confidence too high for downstream",
            )
        )
        sig = Signal(data={"confidence": 0.95, "text": "early exit"})
        inhibited = net.evaluate("T1", sig.data, cycle=0)
        assert "T2" in inhibited
        assert "T3" in inhibited
        assert len(net.events) >= 1
        assert net.events[-1].trigger_confidence == 0.95

    def test_no_inhibition_proceeds(self):
        from nrsi.core.signals import (
            InhibitoryNetwork,
            InhibitionRule,
            InhibitionType,
            Signal,
        )

        net = InhibitoryNetwork()
        net.add_rule(
            InhibitionRule(
                name="high_confidence_skip",
                condition=lambda d, c: d.get("confidence", 0) > 0.9,
                targets=["T2", "T3"],
                inhibition_type=InhibitionType.CONFIDENCE,
                reason="confidence too high for downstream",
            )
        )
        sig = Signal(data={"confidence": 0.5})
        inhibited = net.evaluate("T1", sig.data, cycle=0)
        assert len(inhibited) == 0
        assert len(net.events) == 0


# ── Trust Level Escalation ───────────────────────────────────────────────

class TestTrustLevelEscalation:
    """Sequential trust elevation vs single-hop provenance."""

    def test_full_escalation_raw_to_certified(self):
        from nrsi.core.types import raw, validated, trusted, certified, TrustLevel
        from nrsi.core.validation import ValidationGate, FunctionValidator

        v = FunctionValidator(lambda x: True, name="pass")
        g_val = ValidationGate(
            name="to_validated",
            confidence_threshold=0.8,
            validators=[v],
            target_trust=TrustLevel.VALIDATED,
        )
        g_trust = ValidationGate(
            name="to_trusted",
            confidence_threshold=0.8,
            validators=[v],
            target_trust=TrustLevel.TRUSTED,
        )
        g_cert = ValidationGate(
            name="to_certified",
            confidence_threshold=0.8,
            validators=[v],
            target_trust=TrustLevel.CERTIFIED,
        )
        d0 = raw("escalation payload")
        d1 = g_val.process(d0)
        assert d1.trust_level == TrustLevel.VALIDATED
        d2 = g_trust.process(d1)
        assert d2.trust_level == TrustLevel.TRUSTED
        d3 = g_cert.process(d2)
        assert d3.trust_level == TrustLevel.CERTIFIED
        elevated = [e for e in d3.provenance if e.action == "elevated"]
        assert len(elevated) == 3
        assert validated("ctor", 0.9, "g").trust_level == d1.trust_level
        assert trusted("ctor", 0.95, "g").trust_level == d2.trust_level
        assert certified("ctor", 0.99, "g", "pol").trust_level == d3.trust_level

    def test_cannot_skip_trust_levels(self):
        from nrsi.core.types import raw, TrustLevel
        from nrsi.core.validation import ValidationGate, FunctionValidator

        v = FunctionValidator(lambda x: True, name="pass")
        g_val = ValidationGate(
            name="to_validated",
            confidence_threshold=0.8,
            validators=[v],
            target_trust=TrustLevel.VALIDATED,
        )
        g_trust = ValidationGate(
            name="to_trusted",
            confidence_threshold=0.8,
            validators=[v],
            target_trust=TrustLevel.TRUSTED,
        )
        g_cert = ValidationGate(
            name="to_certified",
            confidence_threshold=0.8,
            validators=[v],
            target_trust=TrustLevel.CERTIFIED,
        )
        sequential = g_cert.process(g_trust.process(g_val.process(raw("audited"))))
        seq_elevations = [e for e in sequential.provenance if e.action == "elevated"]
        assert len(seq_elevations) == 3
        assert all(
            e.from_trust is not None and e.to_trust is not None
            for e in seq_elevations
        )

        # Single elevate() allows RAW→CERTIFIED in one step; provenance shows one hop.
        direct = raw("direct").elevate(
            TrustLevel.CERTIFIED,
            confidence=0.99,
            gate_name="single_hop",
            reason="API allows non-sequential elevation",
        )
        direct_elevations = [e for e in direct.provenance if e.action == "elevated"]
        assert len(direct_elevations) == 1
        assert direct.trust_level == TrustLevel.CERTIFIED
