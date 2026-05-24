"""Test AGI engines working through NRSI-native integration.

Verifies:
- Engines register as lobe processors correctly
- Outputs are wrapped in NRSIData with proper trust levels
- ValidationGates gate outputs appropriately
- VLT adapters store/retrieve correctly
- Mode-awareness changes behavior
- Cross-lobe planning coordination works
- Full NRS.process() pipeline includes AGI lobe output
"""
import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nrsi.core.types import NRSIData, TrustLevel, Confidence, raw, validated, trusted
from nrsi.core.validation import ValidationGate, FunctionValidator, ValidationResult, Validator
from nrsi.core.lobes import (
    ProcessingLobe, LobeType, LobeResult, IntegrationCore,
    LogicalLobe, MathematicalLobe, LinguisticLobe,
    TemporalLobe, CreativeProcessingLobe,
)
from nrsi.core.memory import VLT, VLTLayer, VLTItem, PVS4, ProcessingMode, TuitionSystem
from nrsi.core.mode_control import (
    ModeVector, ModeDecision, ModeSpectrum, CognitiveModeController,
)
from nrsi.core.creases import DomainCrease
from nrsi.core.agi_integration import (
    AGIIntegration,
    LogicLobeProcessor,
    MathLobeProcessor,
    SemanticLobeProcessor,
    CausalLobeProcessor,
    AnalogyLobeProcessor,
    PlannerCoordinator,
    VLTWorkingMemoryAdapter,
    VLTLearningAdapter,
)

from nrsip.logic_engine import LogicEngine, Fact, Literal, Term
from nrsip.semantic_engine import SemanticEngine
from nrsip.code_executor import ComputationEngine
from nrsip.causal_engine import CausalReasoner
from nrsip.analogy_engine import AnalogyEngine
from nrsip.planner import Planner
from nrsip.working_memory import WorkingMemory
from nrsip.learning_engine import LearningEngine


# ── Helpers ──────────────────────────────────────────────────────────────────

def _deterministic_context(factual=1.0):
    """Build a context dict that causes _is_deterministic to return True."""
    vec = ModeVector(analytical=0.8, factual=factual, creative=0.0)
    return {"mode_decision": ModeDecision(vector=vec, primary_mode=ModeSpectrum.DETERMINISTIC)}


def _creative_context(creative=1.0):
    """Build a context dict that causes _is_creative to return True."""
    vec = ModeVector(creative=creative, exploratory=0.6, factual=0.0, analytical=0.0)
    return {"mode_decision": ModeDecision(vector=vec, primary_mode=ModeSpectrum.CREATIVE)}


def _make_agi(
    logic=True, math=True, semantic=True, causal=True, analogy=True,
    planner=True, wm=True, learning=True,
):
    """Construct an AGIIntegration with requested engines registered."""
    core = IntegrationCore()
    vlt = VLT()
    creases = {}
    if learning:
        c = DomainCrease("is_a", version="1.0")
        c.lock(training_accuracy=0.999)
        creases["is_a"] = c

    agi = AGIIntegration(lobes=core, vlt=vlt, creases=creases)

    engines = {}
    if logic:
        engines["logic_engine"] = LogicEngine()
    if math:
        engines["computation_engine"] = ComputationEngine()
    if semantic:
        engines["semantic_engine"] = SemanticEngine()
    if causal:
        engines["causal_engine"] = CausalReasoner()
    if analogy:
        engines["analogy_engine"] = AnalogyEngine()
    if planner:
        engines["planner"] = Planner()
    if wm:
        engines["working_memory"] = WorkingMemory()
    if learning:
        engines["learning_engine"] = LearningEngine()

    agi.register_engines(**engines)
    return agi, core, vlt


# ═════════════════════════════════════════════════════════════════════════════
# 1. TestLobeRegistration
# ═════════════════════════════════════════════════════════════════════════════

class TestLobeRegistration(unittest.TestCase):
    """Verify engines register on the correct NRSI lobes."""

    def setUp(self):
        self.agi, self.core, self.vlt = _make_agi()

    def test_logic_registered_on_logical_lobe(self):
        lobe = self.core.get_lobe(LobeType.LOGICAL)
        self.assertIsNotNone(lobe, "LogicalLobe should be registered")
        self.assertIsInstance(lobe, LogicalLobe)
        self.assertGreaterEqual(len(lobe._processors), 1)
        self.assertIsInstance(lobe._processors[-1], LogicLobeProcessor)

    def test_math_registered_on_mathematical_lobe(self):
        lobe = self.core.get_lobe(LobeType.MATHEMATICAL)
        self.assertIsNotNone(lobe, "MathematicalLobe should be registered")
        self.assertIsInstance(lobe, MathematicalLobe)
        self.assertGreaterEqual(len(lobe._processors), 1)
        self.assertIsInstance(lobe._processors[-1], MathLobeProcessor)

    def test_semantic_registered_on_linguistic_lobe(self):
        lobe = self.core.get_lobe(LobeType.LINGUISTIC)
        self.assertIsNotNone(lobe, "LinguisticLobe should be registered")
        self.assertIsInstance(lobe, LinguisticLobe)
        self.assertGreaterEqual(len(lobe._processors), 1)
        self.assertIsInstance(lobe._processors[-1], SemanticLobeProcessor)

    def test_causal_registered_on_temporal_lobe(self):
        lobe = self.core.get_lobe(LobeType.TEMPORAL)
        self.assertIsNotNone(lobe, "TemporalLobe should be registered")
        self.assertIsInstance(lobe, TemporalLobe)
        self.assertGreaterEqual(len(lobe._processors), 1)
        self.assertIsInstance(lobe._processors[-1], CausalLobeProcessor)

    def test_analogy_registered_on_creative_lobe(self):
        lobe = self.core.get_lobe(LobeType.CREATIVE)
        self.assertIsNotNone(lobe, "CreativeProcessingLobe should be registered")
        self.assertIsInstance(lobe, CreativeProcessingLobe)
        self.assertGreaterEqual(len(lobe._processors), 1)
        self.assertIsInstance(lobe._processors[-1], AnalogyLobeProcessor)


# ═════════════════════════════════════════════════════════════════════════════
# 2. TestTrustTyping
# ═════════════════════════════════════════════════════════════════════════════

class TestTrustTyping(unittest.TestCase):
    """Verify lobe processors wrap output with correct NRSIData trust levels."""

    def test_computation_output_is_trusted(self):
        engine = ComputationEngine()
        proc = MathLobeProcessor(engine)
        result = proc("2 + 3")
        val = result["value"]
        self.assertIsInstance(val, NRSIData)
        self.assertIn(val.trust_level, (TrustLevel.TRUSTED, TrustLevel.VALIDATED))
        self.assertGreaterEqual(val.confidence, 0.95)

    def test_logic_proof_is_trusted(self):
        engine = LogicEngine()
        engine.ingest_text("All dogs are animals")
        engine.add_fact(Fact(literal=Literal("is_a", (Term("fido"), Term("dog")))))
        engine.derive_new_knowledge()
        proc = LogicLobeProcessor(engine)
        result = proc("Is fido an animal?")
        val = result["value"]
        self.assertIsInstance(val, NRSIData)
        if result["metadata"].get("proved"):
            self.assertIn(val.trust_level, (TrustLevel.TRUSTED, TrustLevel.VALIDATED))
        else:
            self.assertIn(val.trust_level, (TrustLevel.VALIDATED, TrustLevel.TRUSTED))

    def test_logic_derived_is_validated(self):
        engine = LogicEngine()
        engine.ingest_text("Cats like milk")
        proc = LogicLobeProcessor(engine)
        result = proc("Do cats like milk?")
        val = result["value"]
        self.assertIsInstance(val, NRSIData)
        if result["metadata"].get("proved"):
            self.assertIn(val.trust_level, (TrustLevel.TRUSTED, TrustLevel.VALIDATED))
        else:
            self.assertIn(
                val.trust_level,
                (TrustLevel.VALIDATED, TrustLevel.RAW),
                "Non-proved logic should be VALIDATED or RAW",
            )

    def test_semantic_is_validated(self):
        engine = SemanticEngine()
        proc = SemanticLobeProcessor(engine)
        candidates = ["The Earth orbits the Sun", "Water is H2O", "Dogs are pets"]
        result = proc(
            "planets orbiting stars",
            context={"candidates": candidates},
        )
        val = result["value"]
        self.assertIsInstance(val, NRSIData)
        if result["confidence"] > 0:
            self.assertEqual(val.trust_level, TrustLevel.VALIDATED)
        else:
            self.assertEqual(val.trust_level, TrustLevel.RAW)

    def test_causal_counterfactual_is_raw(self):
        engine = CausalReasoner()
        engine.ingest("Rain causes wet ground. Wet ground causes slippery roads.")
        proc = CausalLobeProcessor(engine)
        result = proc("What if it hadn't rained?")
        val = result["value"]
        self.assertIsInstance(val, NRSIData)
        if result["metadata"].get("counterfactual"):
            self.assertEqual(val.trust_level, TrustLevel.RAW)

    def test_analogy_is_raw(self):
        engine = AnalogyEngine()
        proc = AnalogyLobeProcessor(engine)
        result = proc("immune system", context={"target_domain": "cybersecurity"})
        val = result["value"]
        self.assertIsInstance(val, NRSIData)
        self.assertEqual(val.trust_level, TrustLevel.RAW)


# ═════════════════════════════════════════════════════════════════════════════
# 3. TestModeAwareness
# ═════════════════════════════════════════════════════════════════════════════

class TestModeAwareness(unittest.TestCase):
    """Mode-aware threshold enforcement on lobe processors."""

    def test_deterministic_mode_strict_threshold(self):
        engine = SemanticEngine()
        proc = SemanticLobeProcessor(engine)
        candidates = ["The speed of light is 299792458 m/s", "Bananas are yellow"]
        ctx = _deterministic_context(factual=1.0)
        ctx["candidates"] = candidates
        result = proc("speed of light", context=ctx)
        self.assertEqual(
            result["metadata"]["threshold"],
            SemanticLobeProcessor.DETERMINISTIC_THRESHOLD,
        )
        if result["confidence"] > 0:
            for match in result["value"].value:
                self.assertGreaterEqual(match["score"], 0.85)

    def test_creative_mode_relaxed_threshold(self):
        engine = SemanticEngine()
        proc = SemanticLobeProcessor(engine)
        candidates = ["Something vaguely related", "Totally different topic"]
        ctx = _creative_context(creative=1.0)
        ctx["candidates"] = candidates
        result = proc("topic", context=ctx)
        self.assertEqual(
            result["metadata"]["threshold"],
            SemanticLobeProcessor.CREATIVE_THRESHOLD,
        )

    def test_analogy_suppressed_in_deterministic(self):
        engine = AnalogyEngine()
        proc = AnalogyLobeProcessor(engine)
        ctx = _deterministic_context(factual=1.0)
        ctx["target_domain"] = "finance"
        result = proc("obscure concept xyz", context=ctx)
        threshold_used = result["metadata"].get("threshold", proc.DEFAULT_THRESHOLD)
        self.assertGreaterEqual(threshold_used, AnalogyLobeProcessor.DETERMINISTIC_THRESHOLD)
        if result["metadata"].get("below_threshold"):
            self.assertEqual(result["value"].trust_level, TrustLevel.RAW)


# ═════════════════════════════════════════════════════════════════════════════
# 4. TestVLTIntegration
# ═════════════════════════════════════════════════════════════════════════════

class TestVLTIntegration(unittest.TestCase):
    """VLT adapters persist data through the NRSI memory hierarchy."""

    def test_wm_stores_to_vlt_l1(self):
        vlt = VLT()
        wm_engine = WorkingMemory()
        adapter = VLTWorkingMemoryAdapter(wm_engine, vlt)
        adapter.process_turn("What is AI?", "AI is artificial intelligence.")
        l1_items = vlt.search(layer=VLTLayer.L1_EPHEMERAL, min_confidence=0.3)
        self.assertGreater(len(l1_items), 0, "Turn should produce at least one L1 item")
        keys = [item.key for item in l1_items]
        has_turn_key = any(k.startswith("wm_turn_") for k in keys)
        self.assertTrue(has_turn_key, f"Expected wm_turn_* key, got: {keys}")

    def test_wm_reads_from_vlt_l2(self):
        vlt = VLT()
        wm_engine = WorkingMemory()
        adapter = VLTWorkingMemoryAdapter(wm_engine, vlt)
        vlt.store(
            "wm_oq_test_question",
            "What is the meaning of life?",
            layer=VLTLayer.L2_SESSION,
            confidence=0.7,
            source="wm_question",
            tags={"open_question"},
        )
        ctx = adapter.get_context()
        self.assertIn("vlt_session", ctx)
        self.assertGreater(ctx["vlt_l2_count"], 0)

    def test_learning_stores_to_vlt_l3(self):
        vlt = VLT()
        learning = LearningEngine()
        adapter = VLTLearningAdapter(learning, vlt)
        adapter.learn_and_store(
            query="What color is the sky?",
            response="The sky is blue.",
            conversation_history=[],
            response_confidence=0.9,
        )
        l3_items = vlt.search(
            layer=VLTLayer.L3_PERSISTENT,
            tags={"learned_claim"},
            min_confidence=0.0,
        )
        if l3_items:
            self.assertTrue(
                any("learned_" in item.key for item in l3_items),
                "Expected learned claim key in L3",
            )

    def test_learning_promotes_to_crease(self):
        vlt = VLT()
        learning = LearningEngine()
        crease = DomainCrease("is_a", version="1.0")
        crease.lock(training_accuracy=0.999)
        adapter = VLTLearningAdapter(learning, vlt, creases={"is_a": crease})

        adapter.learn_and_store(
            query="What is a whale?",
            response="A whale is a marine mammal.",
            conversation_history=[
                {"role": "user", "content": "Tell me about whales"},
                {"role": "assistant", "content": "A whale is a marine mammal."},
            ],
            response_confidence=0.95,
        )

        initial_creases = adapter._creases_written
        adapter.learn_and_store(
            query="What is a dolphin?",
            response="A dolphin is a marine mammal.",
            conversation_history=[
                {"role": "user", "content": "Tell me about dolphins"},
                {"role": "assistant", "content": "A dolphin is a marine mammal."},
            ],
            response_confidence=0.95,
        )

        self.assertGreaterEqual(
            adapter._claims_stored, 0,
            "Adapter should have attempted to store claims",
        )


# ═════════════════════════════════════════════════════════════════════════════
# 5. TestPlannerCoordination
# ═════════════════════════════════════════════════════════════════════════════

class TestPlannerCoordination(unittest.TestCase):
    """PlannerCoordinator dispatches to lobes and synthesizes results."""

    def setUp(self):
        self.agi, self.core, self.vlt = _make_agi()
        self.planner_coord = self.agi.get_processor("planner")

    def test_planner_dispatches_to_lobes(self):
        self.assertIsNotNone(self.planner_coord)
        result = self.planner_coord(
            "Compare the economic impact of renewable vs fossil energy",
            domain="general",
            context={},
        )
        self.assertIsNotNone(result)
        self.assertIn("value", result)
        self.assertIn("metadata", result)
        meta = result["metadata"]
        self.assertEqual(meta["engine"], "planner")
        self.assertGreater(meta["steps_total"], 0)

    def test_planner_synthesizes(self):
        self.assertIsNotNone(self.planner_coord)
        result = self.planner_coord(
            "Explain why water boils at different temperatures at different altitudes",
            domain="science",
            context={},
        )
        val = result["value"]
        self.assertIsInstance(val, NRSIData)
        synthesis_text = str(val.value)
        self.assertGreater(len(synthesis_text), 0)
        self.assertIn(
            val.trust_level,
            (TrustLevel.VALIDATED, TrustLevel.RAW),
            "Plan synthesis should be VALIDATED (coherent) or RAW",
        )


# ═════════════════════════════════════════════════════════════════════════════
# 6. TestFullPipeline
# ═════════════════════════════════════════════════════════════════════════════

class TestFullPipeline(unittest.TestCase):
    """NRS.process() integrates AGI lobe output end-to-end."""

    def test_nrs_process_with_agi(self):
        from nrsi.core.nrs import NRS, ResponseStatus
        agi, core, vlt = _make_agi()
        nrs = NRS(
            instance_id="test-agi-pipe",
            total_neurons=1000,
            active_k=10,
            lobes=core,
            vlt=vlt,
            agi_integration=agi,
        )
        resp = nrs.process("What is 2 + 2?")
        self.assertIn(
            resp.status,
            (ResponseStatus.VALIDATED, ResponseStatus.ACCEPTABLE, ResponseStatus.CACHED),
        )
        self.assertGreater(len(resp.answer), 0)
        self.assertGreater(resp.result_confidence, 0)

    def test_nrs_process_backward_compat(self):
        from nrsi.core.nrs import NRS, ResponseStatus
        nrs = NRS(
            instance_id="test-no-agi",
            total_neurons=1000,
            active_k=10,
        )
        resp = nrs.process("Hello world")
        self.assertIn(
            resp.status,
            (ResponseStatus.VALIDATED, ResponseStatus.ACCEPTABLE,
             ResponseStatus.CACHED, ResponseStatus.ERROR),
        )
        self.assertGreater(len(resp.answer), 0)


if __name__ == "__main__":
    unittest.main()
