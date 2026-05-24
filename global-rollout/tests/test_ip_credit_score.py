"""Comprehensive test suite for IP Credit Score system.

Tests every claim from the patents and documentation:
- 5-factor weighted scoring (exact weights verified)
- 6-tier classification (boundary conditions)
- Capability unlock thresholds (score + age requirements)
- Suspension at < 1.0 and ban at < 0.5
- PRISM-fed response quality (h_score → factor 1)
- Incident severity weighting
- Interaction volume logarithmic saturation
- Liveness score from uptime + alive_days
- Verification level KYC bonus
- Full credit report generation
- Score history tracking
- Ledger serialization round-trip
"""

import os
import time
import unittest

from nrsip.ip_credit_score import (
    BAN_THRESHOLD,
    CAPABILITIES,
    SUSPENSION_THRESHOLD,
    WEIGHT_INCIDENT_RATE,
    WEIGHT_INTERACTION_COUNT,
    WEIGHT_NODE_LIVENESS,
    WEIGHT_RESPONSE_QUALITY,
    WEIGHT_VERIFICATION_LEVEL,
    CapabilityEnforcer,
    CapabilityRequirement,
    CreditLedger,
    CreditTier,
    EntityType,
    FactorScores,
    IPCreditScoreEngine,
    IncidentRecord,
    IncidentType,
    InteractionRecord,
    TrustLevel,
    generate_credit_report,
    tier_for_score,
)
from nrsip.credit_store import MemoryStore, _dict_to_ledger, _ledger_to_dict


class TestWeightConstants(unittest.TestCase):
    """Verify the 5-factor weights match patent spec exactly."""

    def test_weights_sum_to_one(self):
        total = (
            WEIGHT_RESPONSE_QUALITY
            + WEIGHT_INCIDENT_RATE
            + WEIGHT_INTERACTION_COUNT
            + WEIGHT_NODE_LIVENESS
            + WEIGHT_VERIFICATION_LEVEL
        )
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_response_quality_weight(self):
        self.assertEqual(WEIGHT_RESPONSE_QUALITY, 0.25)

    def test_incident_rate_weight(self):
        self.assertEqual(WEIGHT_INCIDENT_RATE, 0.25)

    def test_interaction_count_weight(self):
        self.assertEqual(WEIGHT_INTERACTION_COUNT, 0.20)

    def test_node_liveness_weight(self):
        self.assertEqual(WEIGHT_NODE_LIVENESS, 0.20)

    def test_verification_level_weight(self):
        self.assertEqual(WEIGHT_VERIFICATION_LEVEL, 0.10)


class TestTierClassification(unittest.TestCase):
    """Verify the 6-tier boundary conditions from documentation."""

    def test_untrusted_range(self):
        for score in [0.0, 0.5, 0.99]:
            self.assertEqual(tier_for_score(score), CreditTier.UNTRUSTED,
                             f"Score {score} should be UNTRUSTED")

    def test_low_range(self):
        for score in [1.0, 1.5, 1.99]:
            self.assertEqual(tier_for_score(score), CreditTier.LOW,
                             f"Score {score} should be LOW")

    def test_developing_range(self):
        for score in [2.0, 2.5, 2.99]:
            self.assertEqual(tier_for_score(score), CreditTier.DEVELOPING,
                             f"Score {score} should be DEVELOPING")

    def test_established_range(self):
        for score in [3.0, 3.5, 3.99]:
            self.assertEqual(tier_for_score(score), CreditTier.ESTABLISHED,
                             f"Score {score} should be ESTABLISHED")

    def test_trusted_range(self):
        for score in [4.0, 4.25, 4.49]:
            self.assertEqual(tier_for_score(score), CreditTier.TRUSTED,
                             f"Score {score} should be TRUSTED")

    def test_exemplary_range(self):
        for score in [4.5, 4.75, 5.0]:
            self.assertEqual(tier_for_score(score), CreditTier.EXEMPLARY,
                             f"Score {score} should be EXEMPLARY")

    def test_clamping(self):
        self.assertEqual(tier_for_score(-1.0), CreditTier.UNTRUSTED)
        self.assertEqual(tier_for_score(6.0), CreditTier.EXEMPLARY)

    def test_all_six_tiers_exist(self):
        self.assertEqual(len(CreditTier), 6)


class TestScoreEngine(unittest.TestCase):
    """Test the full credit score computation engine."""

    def setUp(self):
        self.engine = IPCreditScoreEngine()

    def _make_ledger(self, **kwargs) -> CreditLedger:
        defaults = {
            "entity_id": "test-node-1",
            "entity_type": EntityType.NRS_INSTANCE,
            "nrsip_address": "2620:be1a:1:4::1",
            "trust_level": TrustLevel.ATTESTED,
            "enrolled_at_ms": int((time.time() - 90 * 86400) * 1000),  # 90 days ago
            "uptime_pct": 99.5,
            "total_interactions": 5000,
            "validated_interactions": 4700,
            "positive_feedback_count": 4700,
            "prism_score_sum": 4250.0,
            "prism_score_count": 5000,
        }
        defaults.update(kwargs)
        return CreditLedger(**defaults)

    def test_new_entity_starts_low(self):
        ledger = CreditLedger(
            entity_id="new-node",
            entity_type=EntityType.NRS_INSTANCE,
            trust_level=TrustLevel.ANONYMOUS,
            enrolled_at_ms=int(time.time() * 1000),
        )
        score = self.engine.compute(ledger)
        self.assertLess(score, 2.0)
        self.assertIn(ledger.current_tier, (CreditTier.UNTRUSTED, CreditTier.LOW))

    def test_mature_attested_node_scores_high(self):
        ledger = self._make_ledger()
        score = self.engine.compute(ledger)
        self.assertGreater(score, 3.5)
        self.assertIn(ledger.current_tier, (CreditTier.ESTABLISHED, CreditTier.TRUSTED, CreditTier.EXEMPLARY))

    def test_score_in_valid_range(self):
        ledger = self._make_ledger()
        score = self.engine.compute(ledger)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 5.0)

    def test_score_history_appended(self):
        ledger = self._make_ledger()
        self.engine.compute(ledger)
        self.engine.compute(ledger)
        self.assertEqual(len(ledger.score_history), 2)
        self.assertIn("factors", ledger.score_history[-1])
        self.assertIn("tier", ledger.score_history[-1])

    def test_response_quality_factor_uses_prism_scores(self):
        ledger = self._make_ledger(prism_score_sum=0.0, prism_score_count=0)
        f1_none = self.engine.compute_response_quality(ledger)
        self.assertEqual(f1_none, 0.0)

        ledger.prism_score_sum = 4500.0
        ledger.prism_score_count = 5000
        f1_high = self.engine.compute_response_quality(ledger)
        self.assertGreater(f1_high, 0.5)

    def test_incident_rate_degrades_score(self):
        clean = self._make_ledger()
        score_clean = self.engine.compute(clean)

        dirty = self._make_ledger()
        for _ in range(10):
            dirty.incidents.append(IncidentRecord(
                timestamp_ms=int(time.time() * 1000),
                incident_type=IncidentType.HALLUCINATION,
            ))
        score_dirty = self.engine.compute(dirty)
        self.assertLess(score_dirty, score_clean)

    def test_ban_severity_incidents_crush_score(self):
        ledger = self._make_ledger(total_interactions=100)
        ledger.incidents.append(IncidentRecord(
            timestamp_ms=int(time.time() * 1000),
            incident_type=IncidentType.BAN,
        ))
        self.engine.compute(ledger)
        f2 = self.engine.compute_incident_rate(ledger)
        self.assertLess(f2, 0.8)

    def test_interaction_count_logarithmic_saturation(self):
        low = self._make_ledger(total_interactions=10)
        mid = self._make_ledger(total_interactions=1000)
        high = self._make_ledger(total_interactions=10000)

        f3_low = self.engine.compute_interaction_count(low)
        f3_mid = self.engine.compute_interaction_count(mid)
        f3_high = self.engine.compute_interaction_count(high)

        self.assertLess(f3_low, f3_mid)
        self.assertLess(f3_mid, f3_high)
        self.assertAlmostEqual(f3_high, 1.0, places=1)

    def test_liveness_factor_combines_days_and_uptime(self):
        young_bad_uptime = self._make_ledger(
            enrolled_at_ms=int((time.time() - 5 * 86400) * 1000),
            uptime_pct=50.0,
        )
        old_great_uptime = self._make_ledger(
            enrolled_at_ms=int((time.time() - 200 * 86400) * 1000),
            uptime_pct=99.99,
        )
        f4_young = self.engine.compute_node_liveness(young_bad_uptime)
        f4_old = self.engine.compute_node_liveness(old_great_uptime)
        self.assertLess(f4_young, f4_old)

    def test_verification_level_scores(self):
        anon = self._make_ledger(trust_level=TrustLevel.ANONYMOUS)
        auth = self._make_ledger(trust_level=TrustLevel.AUTHENTICATED)
        attest = self._make_ledger(trust_level=TrustLevel.ATTESTED)

        f5_anon = self.engine.compute_verification_level(anon)
        f5_auth = self.engine.compute_verification_level(auth)
        f5_attest = self.engine.compute_verification_level(attest)

        self.assertAlmostEqual(f5_anon, 0.2)
        self.assertAlmostEqual(f5_auth, 0.6)
        self.assertAlmostEqual(f5_attest, 1.0)

    def test_record_interaction_updates_score(self):
        ledger = self._make_ledger()
        old_score = self.engine.compute(ledger)

        interaction = InteractionRecord(
            timestamp_ms=int(time.time() * 1000),
            prism_tier="T3",
            h_score=0.95,
            validated=True,
            domain="medical",
        )
        new_score = self.engine.record_interaction(ledger, interaction)
        self.assertEqual(ledger.total_interactions, 5001)
        self.assertEqual(ledger.validated_interactions, 4701)
        self.assertIsInstance(new_score, float)

    def test_record_incident_degrades_score(self):
        ledger = self._make_ledger()
        old_score = self.engine.compute(ledger)

        incident = IncidentRecord(
            timestamp_ms=int(time.time() * 1000),
            incident_type=IncidentType.SECURITY,
        )
        new_score = self.engine.record_incident(ledger, incident)
        self.assertLessEqual(new_score, old_score)
        self.assertEqual(len(ledger.incidents), 1)

    def test_resolve_incident_improves_score(self):
        ledger = self._make_ledger()
        incident = IncidentRecord(
            timestamp_ms=int(time.time() * 1000),
            incident_type=IncidentType.SECURITY,
        )
        score_with_incident = self.engine.record_incident(ledger, incident)
        score_after_resolve = self.engine.resolve_incident(ledger, 0)
        self.assertGreaterEqual(score_after_resolve, score_with_incident)

    def test_update_liveness_recomputes(self):
        ledger = self._make_ledger(uptime_pct=80.0)
        score_low = self.engine.compute(ledger)
        score_high = self.engine.update_liveness(ledger, 99.99)
        self.assertGreater(score_high, score_low)

    def test_set_trust_level_recomputes(self):
        ledger = self._make_ledger(trust_level=TrustLevel.ANONYMOUS)
        score_anon = self.engine.compute(ledger)
        score_attested = self.engine.set_trust_level(ledger, TrustLevel.ATTESTED)
        self.assertGreater(score_attested, score_anon)


class TestSuspensionAndBan(unittest.TestCase):
    """Test suspension at < 1.0 and ban at < 0.5."""

    def setUp(self):
        self.engine = IPCreditScoreEngine()

    def test_suspension_threshold(self):
        self.assertEqual(SUSPENSION_THRESHOLD, 1.0)

    def test_ban_threshold(self):
        self.assertEqual(BAN_THRESHOLD, 0.5)

    def test_low_score_triggers_suspension(self):
        ledger = CreditLedger(
            entity_id="bad-node",
            entity_type=EntityType.NRS_INSTANCE,
            trust_level=TrustLevel.ANONYMOUS,
            enrolled_at_ms=int(time.time() * 1000),
            total_interactions=100,
        )
        for _ in range(50):
            ledger.incidents.append(IncidentRecord(
                timestamp_ms=int(time.time() * 1000),
                incident_type=IncidentType.SECURITY,
            ))
        self.engine.compute(ledger)
        self.assertTrue(ledger.suspended)

    def test_extreme_incidents_trigger_ban(self):
        ledger = CreditLedger(
            entity_id="banned-node",
            entity_type=EntityType.NRS_INSTANCE,
            trust_level=TrustLevel.ANONYMOUS,
            enrolled_at_ms=int(time.time() * 1000),
            total_interactions=10,
        )
        for _ in range(20):
            ledger.incidents.append(IncidentRecord(
                timestamp_ms=int(time.time() * 1000),
                incident_type=IncidentType.BAN,
            ))
        self.engine.compute(ledger)
        self.assertTrue(ledger.banned)


class TestCapabilityEnforcer(unittest.TestCase):
    """Test capability unlock thresholds from documentation."""

    def setUp(self):
        self.enforcer = CapabilityEnforcer()
        self.engine = IPCreditScoreEngine()

    def _make_ledger_with_score(self, score: float, alive_days: float = 100) -> CreditLedger:
        enrolled_ms = int((time.time() - alive_days * 86400) * 1000)
        ledger = CreditLedger(
            entity_id="test-entity",
            entity_type=EntityType.NRS_INSTANCE,
            enrolled_at_ms=enrolled_ms,
            current_score=score,
            current_tier=tier_for_score(score),
            suspended=score < SUSPENSION_THRESHOLD,
            banned=score < BAN_THRESHOLD,
        )
        return ledger

    def test_basic_queries_requires_0_5(self):
        self.assertEqual(CAPABILITIES["basic_queries"].min_score, 0.5)
        self.assertEqual(CAPABILITIES["basic_queries"].min_alive_days, 0)

    def test_standard_access_requires_1_5_and_7d(self):
        self.assertEqual(CAPABILITIES["standard_access"].min_score, 1.5)
        self.assertEqual(CAPABILITIES["standard_access"].min_alive_days, 7)

    def test_domain_specific_requires_2_5_and_30d(self):
        self.assertEqual(CAPABILITIES["domain_specific"].min_score, 2.5)
        self.assertEqual(CAPABILITIES["domain_specific"].min_alive_days, 30)

    def test_cross_provider_requires_3_0_and_60d(self):
        self.assertEqual(CAPABILITIES["cross_provider"].min_score, 3.0)
        self.assertEqual(CAPABILITIES["cross_provider"].min_alive_days, 60)

    def test_priority_routing_requires_4_0(self):
        self.assertEqual(CAPABILITIES["priority_routing"].min_score, 4.0)

    def test_endorse_entities_requires_4_5(self):
        self.assertEqual(CAPABILITIES["endorse_entities"].min_score, 4.5)

    def test_can_basic_queries_above_threshold(self):
        ledger = self._make_ledger_with_score(0.7)
        self.assertTrue(self.enforcer.can(ledger, "basic_queries"))

    def test_cannot_basic_queries_below_threshold(self):
        ledger = self._make_ledger_with_score(0.3)
        self.assertFalse(self.enforcer.can(ledger, "basic_queries"))

    def test_age_requirement_enforced(self):
        ledger = self._make_ledger_with_score(2.5, alive_days=10)
        self.assertFalse(self.enforcer.can(ledger, "domain_specific"))

        ledger_old = self._make_ledger_with_score(2.5, alive_days=35)
        self.assertTrue(self.enforcer.can(ledger_old, "domain_specific"))

    def test_banned_entity_cannot_do_anything(self):
        ledger = self._make_ledger_with_score(0.3)
        self.assertTrue(ledger.banned)
        for cap_name in CAPABILITIES:
            self.assertFalse(self.enforcer.can(ledger, cap_name))

    def test_suspended_entity_only_basic_queries(self):
        ledger = self._make_ledger_with_score(0.8)
        self.assertTrue(ledger.suspended)
        self.assertTrue(self.enforcer.can(ledger, "basic_queries"))
        self.assertFalse(self.enforcer.can(ledger, "standard_access"))
        self.assertFalse(self.enforcer.can(ledger, "priority_routing"))

    def test_enforcement_decision_has_audit_fields(self):
        ledger = self._make_ledger_with_score(3.5, alive_days=100)
        decision = self.enforcer.enforcement_decision(ledger, "cross_provider")
        self.assertTrue(decision["allowed"])
        self.assertEqual(decision["capability"], "cross_provider")
        self.assertIn("entity_score", decision)
        self.assertIn("entity_tier", decision)
        self.assertIn("required_score", decision)
        self.assertIn("actual_alive_days", decision)

    def test_enforcement_decision_denied_reason(self):
        ledger = self._make_ledger_with_score(2.0, alive_days=100)
        decision = self.enforcer.enforcement_decision(ledger, "cross_provider")
        self.assertFalse(decision["allowed"])
        self.assertIn("score", decision["reason"])

    def test_unlocked_capabilities_returns_all(self):
        ledger = self._make_ledger_with_score(4.8, alive_days=200)
        caps = self.enforcer.unlocked_capabilities(ledger)
        self.assertEqual(len(caps), len(CAPABILITIES))
        for cap_name in CAPABILITIES:
            self.assertTrue(caps[cap_name], f"{cap_name} should be unlocked for 4.8/200d")


class TestCreditReport(unittest.TestCase):
    """Test the full IP Credit Report generation."""

    def test_report_matches_documentation_fields(self):
        engine = IPCreditScoreEngine()
        ledger = CreditLedger(
            entity_id="2620:be1a:3:4:0:0:11a9:0",
            entity_type=EntityType.NRS_INSTANCE,
            nrsip_address="2620:be1a:3:4:0:0:11a9:0",
            trust_level=TrustLevel.ATTESTED,
            enrolled_at_ms=int((time.time() - 245 * 86400) * 1000),
            uptime_pct=99.97,
            total_interactions=12847,
            validated_interactions=12100,
            positive_feedback_count=12100,
            prism_score_sum=11500.0,
            prism_score_count=12847,
            domain_certifications=["medical", "pharmacology"],
        )
        engine.compute(ledger)
        report = generate_credit_report(ledger)

        self.assertEqual(report.entity, "2620:be1a:3:4:0:0:11a9:0")
        self.assertEqual(report.entity_type, "nrs_instance")
        self.assertGreater(report.reliability_score, 3.0)
        self.assertIn(report.tier, [t.value for t in CreditTier])
        self.assertGreater(report.alive_days, 240)
        self.assertAlmostEqual(report.uptime_pct, 99.97)
        self.assertEqual(report.total_interactions, 12847)
        self.assertEqual(report.domain_certifications, ["medical", "pharmacology"])
        self.assertEqual(report.trust_level, "ATTESTED")
        self.assertFalse(report.suspended)
        self.assertFalse(report.banned)

    def test_report_to_dict_has_all_keys(self):
        engine = IPCreditScoreEngine()
        ledger = CreditLedger(
            entity_id="test",
            entity_type=EntityType.NRS_INSTANCE,
            enrolled_at_ms=int((time.time() - 100 * 86400) * 1000),
        )
        engine.compute(ledger)
        report = generate_credit_report(ledger)
        d = report.to_dict()

        required_keys = {
            "entity", "type", "nrsip_address", "reliability_score", "tier",
            "alive_days", "uptime_pct", "total_interactions", "positive_feedback_pct",
            "incidents", "unresolved_incidents", "domain_certifications",
            "cross_provider_enabled", "priority_routing", "can_endorse",
            "suspended", "banned", "trust_level", "factor_breakdown",
            "unlocked_capabilities",
        }
        for key in required_keys:
            self.assertIn(key, d, f"Missing key in credit report: {key}")

        fb = d["factor_breakdown"]
        self.assertIn("response_quality_25pct", fb)
        self.assertIn("incident_rate_25pct", fb)
        self.assertIn("interaction_count_20pct", fb)
        self.assertIn("node_liveness_20pct", fb)
        self.assertIn("verification_level_10pct", fb)


class TestLedgerSerialization(unittest.TestCase):
    """Test ledger round-trip through store serialization."""

    def test_round_trip(self):
        engine = IPCreditScoreEngine()
        original = CreditLedger(
            entity_id="round-trip-test",
            entity_type=EntityType.USER,
            nrsip_address="2620:be1a:2:1::abc",
            trust_level=TrustLevel.AUTHENTICATED,
            enrolled_at_ms=int((time.time() - 50 * 86400) * 1000),
            uptime_pct=98.5,
            total_interactions=500,
            validated_interactions=480,
            positive_feedback_count=475,
            prism_score_sum=420.0,
            prism_score_count=500,
            incidents=[
                IncidentRecord(
                    timestamp_ms=int(time.time() * 1000),
                    incident_type=IncidentType.HALLUCINATION,
                ),
            ],
            domain_certifications=["legal"],
        )
        engine.compute(original)

        d = _ledger_to_dict(original)
        restored = _dict_to_ledger(d)

        self.assertEqual(restored.entity_id, original.entity_id)
        self.assertEqual(restored.entity_type, original.entity_type)
        self.assertEqual(restored.trust_level, original.trust_level)
        self.assertEqual(restored.total_interactions, original.total_interactions)
        self.assertEqual(restored.current_score, original.current_score)
        self.assertEqual(restored.current_tier, original.current_tier)
        self.assertEqual(len(restored.incidents), 1)
        self.assertEqual(restored.incidents[0].incident_type, IncidentType.HALLUCINATION)
        self.assertEqual(restored.domain_certifications, ["legal"])

    def test_memory_store_crud(self):
        store = MemoryStore()
        engine = IPCreditScoreEngine()

        ledger = CreditLedger(
            entity_id="store-test",
            entity_type=EntityType.APPLICATION,
            enrolled_at_ms=int(time.time() * 1000),
        )
        engine.compute(ledger)

        store.save(ledger)
        self.assertEqual(store.count(), 1)
        self.assertIn("store-test", store.list_entities())

        loaded = store.load("store-test")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.entity_id, "store-test")
        self.assertEqual(loaded.current_score, ledger.current_score)

        store.delete("store-test")
        self.assertEqual(store.count(), 0)
        self.assertIsNone(store.load("store-test"))


class TestEntityTypes(unittest.TestCase):
    """Test all 4 entity types from the patent."""

    def test_four_entity_types(self):
        types = set(EntityType)
        self.assertEqual(types, {
            EntityType.NRS_INSTANCE,
            EntityType.USER,
            EntityType.APPLICATION,
            EntityType.SERVICE,
        })

    def test_three_trust_levels(self):
        levels = set(TrustLevel)
        self.assertEqual(levels, {
            TrustLevel.ANONYMOUS,
            TrustLevel.AUTHENTICATED,
            TrustLevel.ATTESTED,
        })


class TestIncidentTypes(unittest.TestCase):
    """All incident types specified in the documentation."""

    def test_all_incident_types(self):
        expected = {
            IncidentType.HALLUCINATION,
            IncidentType.COMPLAINT,
            IncidentType.BLOCK,
            IncidentType.BAN,
            IncidentType.SECURITY,
            IncidentType.ATTESTATION_FAILURE,
        }
        self.assertEqual(set(IncidentType), expected)

    def test_severity_ordering(self):
        from nrsip.ip_credit_score import INCIDENT_SEVERITY
        self.assertLess(INCIDENT_SEVERITY[IncidentType.COMPLAINT],
                        INCIDENT_SEVERITY[IncidentType.HALLUCINATION])
        self.assertLess(INCIDENT_SEVERITY[IncidentType.HALLUCINATION],
                        INCIDENT_SEVERITY[IncidentType.BLOCK])
        self.assertLess(INCIDENT_SEVERITY[IncidentType.BLOCK],
                        INCIDENT_SEVERITY[IncidentType.SECURITY])
        self.assertLess(INCIDENT_SEVERITY[IncidentType.SECURITY],
                        INCIDENT_SEVERITY[IncidentType.BAN])


class TestScoreProgression(unittest.TestCase):
    """Test realistic score progression from enrollment to TRUSTED."""

    def test_new_entity_progresses_with_interactions(self):
        engine = IPCreditScoreEngine()
        ledger = CreditLedger(
            entity_id="growing-node",
            entity_type=EntityType.NRS_INSTANCE,
            trust_level=TrustLevel.AUTHENTICATED,
            enrolled_at_ms=int((time.time() - 60 * 86400) * 1000),
            uptime_pct=99.0,
        )
        engine.compute(ledger)
        initial_score = ledger.current_score

        for i in range(200):
            interaction = InteractionRecord(
                timestamp_ms=int(time.time() * 1000),
                prism_tier="T2",
                h_score=0.85,
                validated=True,
            )
            engine.record_interaction(ledger, interaction)

        final_score = ledger.current_score
        self.assertGreater(final_score, initial_score)
        self.assertGreater(final_score, 2.0)


if __name__ == "__main__":
    unittest.main()
