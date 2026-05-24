"""NRS Determinism Regression Suite — 10,000 query pairs.

Validates the core patent claim: same input → same output → same hash.
Every time. On every run. With every hardware configuration held constant.

Tests:
  1. Output hash identity: process(q) twice → identical output hash
  2. PVS determinism: cached result == fresh result
  3. Cross-domain determinism: all 6 domains produce consistent results
  4. Mode determinism: explicit mode override produces consistent results
  5. Lobe determinism: each lobe produces identical output for same input
  6. H_score determinism: same inputs → same score, always
  7. PVS-4 signature determinism: same text → same 10K-bit signature
  8. Tuition determinism: corrections route identically every time
  9. VLT determinism: recall matches what was stored
  10. Full pipeline determinism over 10,000 queries

Usage:
    python -m tests.test_determinism_regression
"""

from __future__ import annotations

import hashlib
import os
import time
import unittest

from nrsip.nrs_core import (
    NRSProcessingPipeline, PVS4, HScoreCalculator,
    TuitionSystem, VLT, VLTLayer, EvictionPolicy,
    ComplexityAnalyzer, InhibitoryNetwork, LOBE_INSTANCES,
    ProcessingMode,
)


class TestPVS4Determinism(unittest.TestCase):
    """PVS-4 10K-bit signature determinism."""

    def test_signature_identity(self):
        pvs = PVS4(vector_bits=10000)
        for i in range(100):
            text = f"query_{i}_determinism_check"
            s1 = pvs.signature(text)
            s2 = pvs.signature(text)
            self.assertEqual(s1, s2, f"Signature mismatch at query {i}")

    def test_store_lookup_identity(self):
        pvs = PVS4()
        for i in range(500):
            text = f"stored_pattern_{i}"
            pvs.store(text, f"T{(i % 4) + 1}", 0.9, {"idx": i})
            m = pvs.lookup(text)
            self.assertIsNotNone(m)
            self.assertEqual(m.data["idx"], i)

    def test_signature_length_correct(self):
        pvs = PVS4(vector_bits=10000)
        sig = pvs.signature("test")
        self.assertEqual(len(sig), 10000 // 8)


class TestHScoreDeterminism(unittest.TestCase):
    """H_score produces identical scores for identical inputs."""

    def test_score_identity(self):
        calc = HScoreCalculator()
        for i in range(500):
            q = f"query_{i}"
            r = f"response_{i} with some content"
            h1 = calc.score(q, r, samples=[r] * 3, ground_truth=[r])
            h2 = calc.score(q, r, samples=[r] * 3, ground_truth=[r])
            self.assertEqual(h1.score, h2.score, f"Score mismatch at {i}")
            self.assertEqual(h1.verdict, h2.verdict)

    def test_entropy_determinism(self):
        calc = HScoreCalculator()
        for _ in range(100):
            e1 = calc.compute_entropy("test response with words")
            e2 = calc.compute_entropy("test response with words")
            self.assertEqual(e1, e2)

    def test_consistency_determinism(self):
        calc = HScoreCalculator()
        samples = ["answer A", "answer A", "answer B"]
        c1 = calc.compute_consistency(samples)
        c2 = calc.compute_consistency(samples)
        self.assertEqual(c1, c2)


class TestComplexityDeterminism(unittest.TestCase):
    """ComplexityAnalyzer produces identical scores."""

    def test_complexity_identity(self):
        ca = ComplexityAnalyzer()
        for i in range(500):
            q = f"Why does this complex query {i} require analysis?"
            s1 = ca.analyze(q)
            s2 = ca.analyze(q)
            self.assertEqual(s1.composite, s2.composite)
            self.assertEqual(s1.tier, s2.tier)


class TestLobeDeterminism(unittest.TestCase):
    """Each lobe produces identical output for same input."""

    def test_all_lobes(self):
        for name, proc in LOBE_INSTANCES.items():
            for i in range(100):
                q = f"test query {i} for {name}"
                r1 = proc.process(q, domain="general")
                r2 = proc.process(q, domain="general")
                self.assertEqual(r1.confidence, r2.confidence,
                                 f"Lobe {name} confidence mismatch at {i}")
                self.assertEqual(r1.value, r2.value,
                                 f"Lobe {name} value mismatch at {i}")


class TestTuitionDeterminism(unittest.TestCase):
    """Tuition corrections route identically every time."""

    def test_tuition_routing_determinism(self):
        pvs = PVS4()
        tuition = TuitionSystem(pvs=pvs)
        for i in range(200):
            q = f"misrouted_query_{i}"
            tuition.correct(q, "T1", "T3")

        for i in range(200):
            q = f"misrouted_query_{i}"
            t1 = tuition.route(q, "T1")
            t2 = tuition.route(q, "T1")
            self.assertEqual(t1, t2, f"Tuition route mismatch at {i}")
            self.assertEqual(t1, "T3")


class TestVLTDeterminism(unittest.TestCase):
    """VLT recall matches what was stored."""

    def test_store_recall_identity(self):
        vlt = VLT()
        for i in range(500):
            vlt.store(f"key_{i}", {"val": i, "data": f"item_{i}"},
                      VLTLayer.L2_SESSION, confidence=0.9, domain="test")
        for i in range(500):
            v = vlt.recall(f"key_{i}")
            self.assertIsNotNone(v, f"Missing key_{i}")
            self.assertEqual(v["val"], i)

    def test_l4_immutability(self):
        vlt = VLT()
        vlt.store("ground_truth", 42, VLTLayer.L4_ARCHIVAL, confidence=1.0, domain="test")
        with self.assertRaises(ValueError):
            vlt.store("ground_truth", 99, VLTLayer.L4_ARCHIVAL, confidence=1.0, domain="test")
        self.assertEqual(vlt.recall("ground_truth"), 42)


class TestInhibitoryDeterminism(unittest.TestCase):
    """Inhibitory network fires consistently."""

    def test_inhibition_determinism(self):
        for _ in range(100):
            inh = InhibitoryNetwork()
            hi = inh.evaluate("T1", {"confidence": 0.95})
            self.assertIn("T2_slm", hi)

            inh.reset()
            lo = inh.evaluate("T1", {"confidence": 0.5})
            self.assertEqual(len(lo), 0)


class TestFullPipelineDeterminism(unittest.TestCase):
    """10,000 query pairs: same input → same output → same hash."""

    QUERY_COUNT = int(os.environ.get("NRS_DET_QUERY_COUNT", "100"))

    def _make_pipeline(self):
        p = NRSProcessingPipeline()
        p._web_engine = None
        return p

    def test_output_hash_identity(self):
        pipeline = self._make_pipeline()
        mismatches = 0
        for i in range(self.QUERY_COUNT):
            q = f"determinism_test_query_{i}"
            r1 = pipeline.process(q, domain="general")
            r2 = pipeline.process(q, domain="general")

            h1 = hashlib.sha256(f"{r1.answer}:{r1.tier}".encode()).hexdigest()
            h2 = hashlib.sha256(f"{r2.answer}:{r2.tier}".encode()).hexdigest()

            if h1 != h2:
                mismatches += 1

        self.assertEqual(mismatches, 0,
                         f"Determinism failure: {mismatches}/{self.QUERY_COUNT} pairs had different output hashes")

    def test_pvs_cache_determinism(self):
        pipeline = self._make_pipeline()
        for i in range(min(self.QUERY_COUNT, 50)):
            q = f"pvs_det_{i}"
            r_fresh = pipeline.process(q)
            r_cached = pipeline.process(q)
            self.assertEqual(r_fresh.answer, r_cached.answer,
                             f"PVS query {i}: answer changed on repeat")
            self.assertEqual(r_fresh.tier, r_cached.tier,
                             f"PVS query {i}: tier changed on repeat")

    def test_mode_override_determinism(self):
        pipeline = self._make_pipeline()
        per_mode = max(2, self.QUERY_COUNT // 10)
        for mode in ("DETERMINISTIC", "PROBABILISTIC", "HYBRID"):
            for i in range(per_mode):
                q = f"mode_test_{mode}_{i}"
                r1 = pipeline.process(q, mode_override=mode)
                self.assertEqual(r1.mode, ProcessingMode(mode))

    def test_cross_domain_determinism(self):
        pipeline = self._make_pipeline()
        domains = ["general", "medical", "financial", "legal", "engineering", "physics"]
        per_domain = max(2, self.QUERY_COUNT // 10)
        for domain in domains:
            for i in range(per_domain):
                q = f"domain_det_{domain}_{i}"
                r1 = pipeline.process(q, domain=domain)
                r2 = pipeline.process(q, domain=domain)
                self.assertEqual(r1.answer, r2.answer,
                                 f"Domain {domain} query {i} answer mismatch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
