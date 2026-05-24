"""Comprehensive test suite for NRS AGI cognitive engines.

Tests all 9 AGI engines for correctness, integration, and edge cases.
Run: python3 -m pytest global-rollout/tests/test_agi_engines.py -v
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nrsip.logic_engine import (
    LogicEngine, Fact, Rule, Literal, Term, Contradiction,
)
from nrsip.semantic_engine import SemanticEngine
from nrsip.learning_engine import LearningEngine
from nrsip.code_executor import ComputationEngine
from nrsip.planner import Planner
from nrsip.working_memory import WorkingMemory, MemoryItem, ItemType
from nrsip.self_improvement import SelfImprovementLoop, InteractionRecord
from nrsip.causal_engine import CausalReasoner
from nrsip.analogy_engine import AnalogyEngine


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LogicEngine
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogicEngine(unittest.TestCase):

    def setUp(self):
        self.engine = LogicEngine()

    def test_ingest_simple_fact(self):
        result = self.engine.ingest_text("All humans are mortal")
        self.assertGreater(result["rules_added"] + result["facts_added"], 0)
        qr = self.engine.query("Are humans mortal?")
        self.assertIsInstance(qr.answer, str)
        self.assertGreater(len(qr.answer), 0)

    def test_forward_chaining(self):
        self.engine.ingest_text("All dogs are animals")
        self.engine.add_fact(Fact(
            literal=Literal("is_a", (Term("fido"), Term("dog"))),
        ))
        derived = self.engine.derive_new_knowledge()
        all_facts = [str(f) for f in self.engine._forward.all_facts]
        has_animal = any("animal" in f for f in all_facts)
        self.assertTrue(has_animal, f"Expected derived animal fact, got: {all_facts}")

    def test_backward_chaining_proof(self):
        self.engine.ingest_text("All cats are animals")
        self.engine.add_fact(Fact(
            literal=Literal("is_a", (Term("whiskers"), Term("cat"))),
        ))
        self.engine.derive_new_knowledge()
        qr = self.engine.query("Is whiskers an animal?")
        if qr.proof is not None:
            self.assertTrue(qr.proof.proved)

    def test_contradiction_detection(self):
        self.engine.add_fact(Fact(
            literal=Literal("is_a", (Term("sky"), Term("blue"))),
        ))
        self.engine.add_fact(Fact(
            literal=Literal("is_a", (Term("sky"), Term("blue")), negated=True),
        ))
        contradictions = self.engine.check_consistency()
        self.assertGreater(len(contradictions), 0)
        self.assertIsInstance(contradictions[0], Contradiction)

    def test_empty_query(self):
        qr = self.engine.query("Is the sky blue?")
        self.assertIsInstance(qr.answer, str)
        self.assertFalse(qr.answered)

    def test_stats(self):
        stats = self.engine.stats
        self.assertIn("total_facts", stats)
        self.assertIn("total_rules", stats)
        self.assertIn("queries", stats)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SemanticEngine
# ═══════════════════════════════════════════════════════════════════════════════

class TestSemanticEngine(unittest.TestCase):

    def setUp(self):
        self.engine = SemanticEngine()

    def test_similarity_identical(self):
        score = self.engine.similarity(
            "The quick brown fox", "The quick brown fox",
        )
        self.assertGreaterEqual(score, 0.95)

    def test_similarity_related(self):
        score = self.engine.similarity(
            "The doctor treated the patient",
            "The physician healed the sick person",
        )
        self.assertGreater(score, 0.2)

    def test_similarity_unrelated(self):
        score = self.engine.similarity(
            "quantum physics equations",
            "chocolate cake recipe baking",
        )
        self.assertLess(score, 0.15)

    def test_index_and_search(self):
        self.engine.tfidf.add_document("doc1", "Python programming language")
        self.engine.tfidf.add_document("doc2", "Italian pasta cooking recipe")
        self.engine.tfidf.add_document("doc3", "Python code software development")
        results = self.engine.tfidf.query_similarity("python coding", top_k=3)
        self.assertGreater(len(results), 0)
        top_ids = [doc_id for doc_id, _ in results]
        self.assertIn("doc1", top_ids[:2])

    def test_concept_expansion(self):
        expanded = self.engine.expand_query("happy")
        self.assertIn("happy", expanded.lower())
        self.assertGreater(len(expanded.split()), 1)

    def test_best_match(self):
        candidates = [
            "The economy is growing",
            "Dogs are friendly pets",
            "Stock market investment returns",
        ]
        results = self.engine.best_match("financial growth", candidates)
        self.assertGreater(len(results), 0)

    def test_stats(self):
        stats = self.engine.stats
        self.assertIn("concept_synonym_keys", stats)
        self.assertIn("stop_words", stats)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LearningEngine
# ═══════════════════════════════════════════════════════════════════════════════

class TestLearningEngine(unittest.TestCase):

    def setUp(self):
        self.engine = LearningEngine()

    def test_learn_from_interaction(self):
        result = self.engine.learn_from_interaction(
            query="What causes rain?",
            response="Water evaporates, forms clouds, and precipitates as rain.",
            conversation_history=[
                {"role": "user", "content": "What causes rain?"},
                {"role": "assistant", "content": "Water evaporates and condenses."},
            ],
            response_confidence=0.8,
        )
        self.assertIn("claims_extracted", result)
        self.assertIn("claims_accepted", result)
        self.assertIsInstance(result["claims_extracted"], int)

    def test_claim_retrieval(self):
        self.engine.learn_from_interaction(
            query="Tell me about the sun",
            response="The sun is a star. The sun has enormous gravity.",
            conversation_history=[
                {"role": "user", "content": "Tell me about the sun"},
                {"role": "assistant", "content": "The sun is a star at the center."},
            ],
            response_confidence=0.85,
        )
        claims = self.engine.get_claims_for_query("star", top_k=5)
        self.assertIsInstance(claims, list)

    def test_duplicate_claim(self):
        for _ in range(2):
            self.engine.learn_from_interaction(
                query="What is water?",
                response="Water is a liquid.",
                conversation_history=[
                    {"role": "user", "content": "What is water?"},
                    {"role": "assistant", "content": "Water is a liquid."},
                ],
                response_confidence=0.9,
            )
        total = len(self.engine._claims)
        self.engine.learn_from_interaction(
            query="What is water?",
            response="Water is a liquid.",
            conversation_history=[
                {"role": "user", "content": "What is water?"},
                {"role": "assistant", "content": "Water is a liquid."},
            ],
            response_confidence=0.9,
        )
        self.assertEqual(len(self.engine._claims), total)

    def test_stats(self):
        stats = self.engine.stats
        self.assertIn("total_claims", stats)
        self.assertIn("total_interactions", stats)
        self.assertIn("acceptance_rate", stats)
        self.assertIn("claims_by_relation", stats)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ComputationEngine
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputationEngine(unittest.TestCase):

    def setUp(self):
        self.engine = ComputationEngine()

    def test_can_compute_math(self):
        self.assertTrue(self.engine.can_compute("What is 15 * 23?"))

    def test_compute_result(self):
        result = self.engine.compute("What is 15 * 23?")
        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertEqual(result.output, 345)

    def test_cannot_compute_text(self):
        self.assertFalse(self.engine.can_compute("Tell me about history"))

    def test_sandbox_security(self):
        result = self.engine.execute_code("open('/etc/passwd', 'r')")
        self.assertFalse(result.success)
        self.assertIn("Security violation", result.error)

    def test_compute_returns_none_for_non_math(self):
        result = self.engine.compute("Tell me about history")
        self.assertIsNone(result)

    def test_percentage(self):
        result = self.engine.compute("What is 25% of 200?")
        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertEqual(result.output, 50.0)

    def test_stats(self):
        self.engine.compute("What is 2 + 2?")
        stats = self.engine.stats
        self.assertIn("total_queries", stats)
        self.assertGreaterEqual(stats["total_queries"], 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Planner
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlanner(unittest.TestCase):

    def setUp(self):
        self.planner = Planner()

    def test_plan_creation(self):
        plan = self.planner.plan("What caused the fall of Rome?")
        self.assertIsNotNone(plan.goal)
        self.assertGreater(len(plan.steps), 0)
        self.assertEqual(plan.decomposition_type, "causal")

    def test_plan_execution(self):
        plan = self.planner.plan("What is gravity?")
        executed = self.planner.execute(
            plan,
            retrieve_fn=lambda q: f"Retrieved answer for: {q}",
            infer_fn=lambda q: f"Inferred: {q}",
        )
        completed = sum(
            1 for s in executed.steps if s.status.value == "completed"
        )
        self.assertGreater(completed, 0)

    def test_synthesis(self):
        plan = self.planner.plan("Explain photosynthesis")
        executed = self.planner.execute(
            plan,
            retrieve_fn=lambda q: "Plants use sunlight to make glucose.",
        )
        synthesis = self.planner.synthesize_results(executed)
        self.assertIsInstance(synthesis, str)
        self.assertGreater(len(synthesis), 0)

    def test_explain_plan(self):
        plan = self.planner.plan("Why is the sky blue?")
        explanation = self.planner.explain_plan(plan)
        self.assertIn("Step", explanation)
        self.assertIn("causal", explanation.lower())

    def test_complex_plan(self):
        plan = self.planner.plan("Compare capitalism and socialism")
        self.assertEqual(plan.decomposition_type, "comparative")
        self.assertGreater(len(plan.steps), 3)

    def test_stats(self):
        self.planner.plan("test query")
        stats = self.planner.stats
        self.assertIn("active_plans", stats)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. WorkingMemory
# ═══════════════════════════════════════════════════════════════════════════════

class TestWorkingMemory(unittest.TestCase):

    def setUp(self):
        self.wm = WorkingMemory()

    def test_process_turn(self):
        self.wm.process_turn(
            query="What is quantum computing?",
            response="Quantum computing uses qubits.",
        )
        ctx = self.wm.get_context_for_response()
        self.assertGreater(self.wm._turn_count, 0)
        self.assertIn("primary_topic", ctx)

    def test_context_retrieval(self):
        self.wm.process_turn(
            query="Tell me about dogs",
            response="Dogs are domesticated canines.",
        )
        ctx = self.wm.get_context_for_response()
        self.assertIn("primary_topic", ctx)
        self.assertIn("open_questions", ctx)
        self.assertIn("buffer_usage", ctx)
        self.assertIn("conversation_phase", ctx)

    def test_memory_capacity(self):
        for i in range(20):
            self.wm.attend(MemoryItem(
                content=f"item_{i}",
                item_type=ItemType.FACT,
                salience=0.5 + (i * 0.01),
            ))
        self.assertLessEqual(len(self.wm._buffer), self.wm.CAPACITY)

    def test_open_questions(self):
        self.wm.process_turn(
            query="What is dark matter? How old is the universe?",
            response="Dark matter is a hypothetical form of matter.",
        )
        questions = self.wm.get_open_questions()
        self.assertIsInstance(questions, list)

    def test_decay(self):
        self.wm.attend(MemoryItem(
            content="ephemeral",
            item_type=ItemType.FACT,
            salience=0.1,
            decay_rate=0.99,
        ))
        for _ in range(5):
            self.wm.decay_tick()
        saliences = [m.salience for m in self.wm._buffer.values()
                     if m.content == "ephemeral"]
        if saliences:
            self.assertLess(saliences[0], 0.01)

    def test_stats(self):
        stats = self.wm.stats
        self.assertIn("buffer_size", stats)
        self.assertIn("capacity", stats)
        self.assertIn("turn_count", stats)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SelfImprovementLoop
# ═══════════════════════════════════════════════════════════════════════════════

class TestSelfImprovementLoop(unittest.TestCase):

    def setUp(self):
        self.loop = SelfImprovementLoop()

    def test_record_interaction(self):
        record = InteractionRecord(
            query="What is AI?",
            domain="technology",
            mode="deterministic",
            confidence=0.85,
            latency_ms=120.0,
            had_facts=True,
            had_web_facts=False,
            had_computation=False,
            had_logic_proof=False,
            plan_used=False,
            response_length=200,
        )
        self.loop.record_interaction(record)
        self.assertEqual(self.loop._interaction_count, 1)

    def test_get_runtime_param(self):
        val = self.loop.get_runtime_param(
            "web_max_facts", domain="medical", default=5,
        )
        self.assertEqual(val, 5)

    def test_stats(self):
        stats = self.loop.stats
        self.assertIn("interaction_count", stats)
        self.assertIn("improvement_cycles", stats)
        self.assertIn("active_hypotheses", stats)
        self.assertIn("overall_performance", stats)

    def test_improve_cycle(self):
        result = self.loop.improve()
        self.assertIsNotNone(result)
        self.assertIsInstance(result.summary, str)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CausalReasoner
# ═══════════════════════════════════════════════════════════════════════════════

class TestCausalReasoner(unittest.TestCase):

    def setUp(self):
        self.reasoner = CausalReasoner()
        self.reasoner.ingest(
            "Smoking causes cancer. "
            "Pollution causes respiratory disease. "
            "Pollution leads to environmental damage. "
            "Environmental damage causes species extinction."
        )

    def test_ingest_causal(self):
        self.assertGreater(self.reasoner.graph.node_count, 0)
        self.assertGreater(self.reasoner.graph.edge_count, 0)

    def test_query_why(self):
        result = self.reasoner.query("Why does cancer happen?")
        self.assertEqual(result.query_type, "why")
        self.assertIsInstance(result.explanation, str)
        self.assertGreater(len(result.explanation), 0)

    def test_query_what_if(self):
        result = self.reasoner.query("What if not smoking?")
        self.assertIn(result.query_type, ("counterfactual", "intervention"))
        self.assertIsInstance(result.explanation, str)

    def test_forward_chain(self):
        result = self.reasoner.query("What are the effects of pollution?")
        self.assertEqual(result.query_type, "effects")
        self.assertIsInstance(result.explanation, str)

    def test_backward_chain(self):
        chains = self.reasoner.graph.backward_chain("species extinction")
        self.assertIsInstance(chains, list)

    def test_counterfactual(self):
        cf = self.reasoner.graph.counterfactual("pollution")
        self.assertIsInstance(cf.affected_nodes, list)
        self.assertIsInstance(cf.explanation, str)

    def test_stats(self):
        stats = self.reasoner.stats
        self.assertIn("nodes", stats)
        self.assertIn("edges", stats)
        self.assertIn("queries_answered", stats)

    def test_unknown_node(self):
        result = self.reasoner.query("Why does unicornitis happen?")
        self.assertFalse(result.answered)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. AnalogyEngine
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnalogyEngine(unittest.TestCase):

    def setUp(self):
        self.engine = AnalogyEngine()

    def test_find_analogy(self):
        result = self.engine.find_analogy("heart")
        self.assertIsInstance(result.explanation, str)
        self.assertGreater(len(result.explanation), 0)

    def test_explain_by_analogy(self):
        explanation = self.engine.explain_by_analogy("dna", "blueprint")
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 10)

    def test_transfer_inference(self):
        inference = self.engine.transfer_inference(
            source="heart",
            source_property="powers",
            target="pump",
        )
        self.assertIsInstance(inference, str)
        self.assertGreater(len(inference), 10)

    def test_rate_analogy(self):
        score = self.engine.rate_analogy(
            "heart_circulatory", "pump_plumbing",
        )
        self.assertGreater(score, 0.5)

    def test_rate_analogy_unknown(self):
        score = self.engine.rate_analogy("xyzzy_domain", "abcde_domain")
        self.assertEqual(score, 0.0)

    def test_generate_analogy(self):
        result = self.engine.generate_analogy("heart", "pump")
        self.assertIsInstance(result.explanation, str)

    def test_stats_like_structure(self):
        result = self.engine.find_analogy("brain")
        self.assertIsNotNone(result.mapping)
        if result.mapping:
            self.assertIsInstance(result.mapping.overall_score, float)


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegrationSemanticLearning(unittest.TestCase):
    """Learning engine uses semantic engine for claim retrieval."""

    def test_semantic_plus_learning(self):
        semantic = SemanticEngine()
        learning = LearningEngine()
        learning.set_semantic_engine(semantic)

        learning.learn_from_interaction(
            query="What is photosynthesis?",
            response="Photosynthesis is the process where plants convert sunlight into energy.",
            conversation_history=[
                {"role": "user", "content": "What is photosynthesis?"},
                {"role": "assistant",
                 "content": "Photosynthesis is a process used by plants."},
            ],
            response_confidence=0.9,
        )
        claims = learning.get_claims_for_query("plants convert light energy")
        self.assertIsInstance(claims, list)


class TestIntegrationPlannerLogic(unittest.TestCase):
    """Planner's infer_fn calls logic engine."""

    def test_planner_with_logic(self):
        logic = LogicEngine()
        logic.ingest_text("All mammals are warm-blooded")
        logic.ingest_text("All dogs are mammals")

        def infer_fn(prompt: str) -> str:
            result = logic.query(prompt)
            return result.answer

        planner = Planner()
        plan = planner.plan("Prove that dogs are warm-blooded")
        executed = planner.execute(plan, infer_fn=infer_fn)
        self.assertIsNotNone(executed.synthesis)
        self.assertGreater(len(executed.synthesis), 0)


class TestIntegrationCausalPlanner(unittest.TestCase):
    """Why-query triggers both causal reasoner and planner."""

    def test_causal_with_planner(self):
        causal = CausalReasoner()
        causal.ingest("Deforestation causes soil erosion. Soil erosion leads to flooding.")

        def retrieve_fn(prompt: str) -> str:
            result = causal.query(prompt)
            return result.explanation

        planner = Planner()
        plan = planner.plan("Why does flooding happen?")
        executed = planner.execute(plan, retrieve_fn=retrieve_fn)
        self.assertIsNotNone(executed.synthesis)
        self.assertGreater(len(executed.synthesis), 0)


if __name__ == "__main__":
    unittest.main()
