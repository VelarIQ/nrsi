"""Tests for NRSI cognitive primitives — the 5 AGI gap-closing operations."""

import pytest
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nrsi.lang.cognitive_primitives import (
    nrsi_compose, CompositionResult,
    LearnableStore, StoredBelief,
    nrsi_semantic_distance, SemanticDistanceResult,
    nrsi_decompose, DecomposeResult,
    nrsi_intent_match, IntentMatchResult,
)


# ── compose tests ──────────────────────────────────────────────────────────

class TestCompose:
    def test_basic_synthesis(self):
        result = nrsi_compose(["The sky is blue.", "Water reflects light."])
        assert isinstance(result, CompositionResult)
        assert result.text
        assert result.confidence > 0
        assert result.fragment_count == 2

    def test_empty_sources(self):
        result = nrsi_compose([])
        assert result.confidence < 0.2

    def test_dict_sources(self):
        sources = [
            {"text": "Gravity pulls objects down.", "confidence": 0.9},
            {"text": "Mass creates gravitational fields.", "confidence": 0.85},
        ]
        result = nrsi_compose(sources, strategy="analytical")
        assert result.fragment_count == 2
        assert result.confidence > 0.5

    def test_deduplication(self):
        result = nrsi_compose(["The sky is blue.", "The sky is blue.", "Water is wet."])
        assert result.fragment_count == 2

    def test_strategies(self):
        facts = ["Fact one about topic.", "Fact two about the same topic."]
        for strat in ("synthesis", "contrastive", "narrative", "analytical"):
            r = nrsi_compose(facts, strategy=strat)
            assert r.strategy == strat

    def test_provenance_tracking(self):
        result = nrsi_compose(["A first fact.", "A second fact."])
        assert len(result.provenance) == 2
        assert all("frag_" in p for p in result.provenance)

    def test_confidence_propagation(self):
        high_conf = [{"text": "Very certain claim.", "confidence": 0.95}]
        low_conf = [{"text": "Uncertain speculation.", "confidence": 0.3}]
        r_high = nrsi_compose(high_conf)
        r_low = nrsi_compose(low_conf)
        assert r_high.confidence > r_low.confidence


# ── LearnableStore tests ───────────────────────────────────────────────────

class TestLearnableStore:
    def test_basic_store_and_query(self):
        store = LearnableStore(decay=0.001, reinforcement=0.1)
        was_new, h = store.store("Water boils at 100 degrees Celsius.", confidence=0.9)
        assert was_new is True
        results = store.query(min_confidence=0.5)
        assert len(results) >= 1

    def test_reinforcement(self):
        store = LearnableStore(reinforcement=0.15)
        store.store("The earth orbits the sun.", confidence=0.8)
        was_new, _ = store.store("The earth orbits the sun.", confidence=0.8)
        assert was_new is False
        results = store.query()
        assert results[0].confidence > 0.8

    def test_conflict_resolution(self):
        store = LearnableStore(conflict_resolution="revision")
        store.store("The capital of France is Paris.", confidence=0.9, domain="geography")
        store.store("The capital of France is not Paris.", confidence=0.95, domain="geography")
        results = store.query(domain="geography")
        assert len(results) == 1
        assert "not" in results[0].text

    def test_domain_filtering(self):
        store = LearnableStore()
        store.store("Physics fact", domain="science")
        store.store("Law fact", domain="law")
        science = store.query(domain="science")
        law = store.query(domain="law")
        assert len(science) == 1
        assert len(law) == 1

    def test_keyword_query(self):
        store = LearnableStore()
        store.store("Quantum entanglement is a phenomenon.", confidence=0.8)
        store.store("Classical mechanics describes motion.", confidence=0.8)
        results = store.query(keywords=["quantum"])
        assert len(results) == 1

    def test_disk_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            store1 = LearnableStore(backing="disk", backing_path=path)
            store1.store("Persistent belief.", confidence=0.9)
            assert len(store1) == 1

            store2 = LearnableStore(backing="disk", backing_path=path)
            assert len(store2) == 1
            results = store2.query()
            assert results[0].text == "Persistent belief."
        finally:
            os.unlink(path)

    def test_stats(self):
        store = LearnableStore()
        store.store("Fact A", confidence=0.7, domain="science")
        store.store("Fact B", confidence=0.8, domain="law")
        stats = store.stats
        assert stats["total_beliefs"] == 2
        assert "science" in stats["domains"]
        assert "law" in stats["domains"]


# ── semantic_distance tests ────────────────────────────────────────────────

class TestSemanticDistance:
    def test_identical_strings(self):
        r = nrsi_semantic_distance("quantum physics", "quantum physics")
        assert r.similarity > 0.8

    def test_completely_different(self):
        r = nrsi_semantic_distance("quantum physics", "chocolate cake recipe")
        assert r.similarity < 0.3

    def test_synonyms_boost(self):
        r1 = nrsi_semantic_distance("create a program", "build a program")
        r2 = nrsi_semantic_distance("create a program", "destroy a program")
        assert r1.similarity > r2.similarity

    def test_shared_concepts(self):
        r = nrsi_semantic_distance("machine learning algorithms", "learning algorithms for machines")
        assert len(r.shared_concepts) > 0

    def test_domain_awareness(self):
        r = nrsi_semantic_distance("software engineering", "software development")
        assert r.similarity > 0.3

    def test_returns_correct_type(self):
        r = nrsi_semantic_distance("hello", "world")
        assert isinstance(r, SemanticDistanceResult)
        assert 0.0 <= r.similarity <= 1.0


# ── decompose tests ────────────────────────────────────────────────────────

class TestDecompose:
    def test_causal_detection(self):
        r = nrsi_decompose("Why does the economy fluctuate?")
        assert r.decomposition_type == "causal"
        assert len(r.steps) > 0

    def test_comparative_detection(self):
        r = nrsi_decompose("Compare Python vs JavaScript")
        assert r.decomposition_type == "comparative"

    def test_explanatory_detection(self):
        r = nrsi_decompose("How does photosynthesis work?")
        assert r.decomposition_type == "explanatory"

    def test_recursive_depth(self):
        r1 = nrsi_decompose("Analyze climate change impacts", max_depth=1)
        r2 = nrsi_decompose("Analyze climate change impacts", max_depth=3)
        assert len(r2.steps) >= len(r1.steps)

    def test_empty_goal(self):
        r = nrsi_decompose("")
        assert r.estimated_confidence == 0.0

    def test_steps_have_descriptions(self):
        r = nrsi_decompose("Explain quantum computing")
        for step in r.steps:
            assert "action" in step
            assert "description" in step
            assert len(step["description"]) > 0

    def test_confidence_estimation(self):
        r = nrsi_decompose("Prove the Pythagorean theorem")
        assert r.estimated_confidence > 0.5


# ── intent_match tests ─────────────────────────────────────────────────────

class TestIntentMatch:
    def test_causal_intent(self):
        r = nrsi_intent_match("Why did the Roman Empire fall?")
        assert r.primary_intent == "causal"
        assert r.confidence > 0

    def test_temporal_intent(self):
        r = nrsi_intent_match("When did World War 2 start?")
        assert r.primary_intent == "temporal"

    def test_explanation_intent(self):
        r = nrsi_intent_match("How does a combustion engine work?")
        assert r.primary_intent == "explanation"

    def test_evaluative_intent(self):
        r = nrsi_intent_match("Should I invest in stocks or bonds?")
        assert r.primary_intent == "evaluative"

    def test_general_fallback(self):
        r = nrsi_intent_match("Hello there")
        assert r.primary_intent == "general"

    def test_belief_base_boosting(self):
        beliefs = [{"text": "The cause of inflation is money supply growth."}]
        r = nrsi_intent_match("What causes inflation?", beliefs)
        assert r.confidence > 0

    def test_matched_signals(self):
        r = nrsi_intent_match("Why does gravity cause objects to fall?")
        assert len(r.matched_signals) > 0

    def test_needs_dict(self):
        r = nrsi_intent_match("Compare and contrast democracy vs autocracy")
        assert isinstance(r.needs, dict)
        assert "analogy" in r.needs

    def test_returns_correct_type(self):
        r = nrsi_intent_match("test query")
        assert isinstance(r, IntentMatchResult)


# ── Language toolchain integration ─────────────────────────────────────────

class TestToolchainIntegration:
    """Verify the 5 primitives are recognized by the NRSI lexer/parser."""

    def test_lexer_recognizes_keywords(self):
        from nrsi.lang.lexer import Lexer, TokenType
        for kw, tt in [
            ("compose", TokenType.KW_COMPOSE),
            ("persist", TokenType.KW_PERSIST),
            ("semantic_distance", TokenType.KW_SEMANTIC_DISTANCE),
            ("decompose", TokenType.KW_DECOMPOSE),
            ("intent_match", TokenType.KW_INTENT_MATCH),
        ]:
            tokens = Lexer(kw).tokenize()
            assert tokens[0].type == tt, f"{kw} -> {tokens[0].type} != {tt}"

    def test_parser_compose_decl(self):
        from nrsi.lang.lexer import Lexer
        from nrsi.lang.parser import Parser, ComposeDecl
        src = 'compose answer from facts { strategy: "synthesis" }'
        tokens = Lexer(src).tokenize()
        module = Parser(tokens).parse()
        assert any(isinstance(d, ComposeDecl) for d in module.declarations)

    def test_parser_persist_decl(self):
        from nrsi.lang.lexer import Lexer
        from nrsi.lang.parser import Parser, PersistDecl
        src = 'persist store { decay: 0.01 }'
        tokens = Lexer(src).tokenize()
        module = Parser(tokens).parse()
        assert any(isinstance(d, PersistDecl) for d in module.declarations)

    def test_parser_semantic_distance_expr(self):
        from nrsi.lang.lexer import Lexer
        from nrsi.lang.parser import Parser, SemanticDistanceExpr
        src = 'let sim = semantic_distance("hello", "world")'
        tokens = Lexer(src).tokenize()
        module = Parser(tokens).parse()
        decls = module.declarations
        assert len(decls) >= 1

    def test_parser_decompose_expr(self):
        from nrsi.lang.lexer import Lexer
        from nrsi.lang.parser import Parser, DecomposeExpr
        src = 'let plan = decompose("explain quantum computing")'
        tokens = Lexer(src).tokenize()
        module = Parser(tokens).parse()
        assert len(module.declarations) >= 1

    def test_parser_intent_match_expr(self):
        from nrsi.lang.lexer import Lexer
        from nrsi.lang.parser import Parser, IntentMatchExpr
        src = 'let intent = intent_match("why gravity", "beliefs")'
        tokens = Lexer(src).tokenize()
        module = Parser(tokens).parse()
        assert len(module.declarations) >= 1

    def test_python_transpiler_compose(self):
        from nrsi.lang.runtime import compile_nrsi
        src = 'compose answer from facts { strategy: "synthesis" }'
        py = compile_nrsi(src)
        assert "nrsi_compose" in py

    def test_python_transpiler_persist(self):
        from nrsi.lang.runtime import compile_nrsi
        src = 'persist store { decay: 0.01 }'
        py = compile_nrsi(src)
        assert "LearnableStore" in py

    def test_swift_transpiler_compose(self):
        from nrsi.lang.lexer import Lexer
        from nrsi.lang.parser import Parser
        from nrsi.lang.targets import get_transpiler
        src = 'compose answer from facts { strategy: "synthesis" }'
        tokens = Lexer(src).tokenize()
        ast = Parser(tokens).parse()
        swift_t = get_transpiler("swift")()
        code = swift_t.transpile(ast)
        assert "NRSICognitive.compose" in code

    def test_kotlin_transpiler_compose(self):
        from nrsi.lang.lexer import Lexer
        from nrsi.lang.parser import Parser
        from nrsi.lang.targets import get_transpiler
        src = 'compose answer from facts { strategy: "synthesis" }'
        tokens = Lexer(src).tokenize()
        ast = Parser(tokens).parse()
        kt_t = get_transpiler("kotlin")()
        code = kt_t.transpile(ast)
        assert "NRSICognitive.compose" in code
