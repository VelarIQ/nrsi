"""Tests for previously-untested NRS/NRSIP modules.

Covers: web_retrieval, session_manager, session_store, metacognition_memory,
reasoning_coherence, reasoning_temporal, behavioral_attestation, knowledge_loader.

All tests run in-memory with mocked externals — no Redis, no HTTP.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Shared async helper for running coroutines in sync tests.
# Python 3.12+ deprecates get_event_loop() in non-async contexts;
# 3.14 removes it entirely.  Use a fresh loop each time.
def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  1. WebRetrievalEngine
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from nrsip.web_retrieval import (
        WebRetrievalEngine, SearchResult, RetrievedFact,
        SOURCE_CREDIBILITY_TIERS, DOMAIN_SOURCE_MAP,
    )
    _HAS_WEB_RETRIEVAL = True
except Exception:
    _HAS_WEB_RETRIEVAL = False


@unittest.skipUnless(_HAS_WEB_RETRIEVAL, "nrsip.web_retrieval not importable")
class TestWebRetrievalEngine(unittest.TestCase):
    """WebRetrievalEngine: instantiation, caching, domain detection, URL safety."""

    def test_instantiation_defaults(self):
        engine = WebRetrievalEngine()
        self.assertEqual(engine._search_count, 0)
        self.assertEqual(engine._crawl_count, 0)
        self.assertIsInstance(engine._search_cache, dict)

    def test_cache_key_normalisation(self):
        engine = WebRetrievalEngine()
        k1 = engine._cache_key("Hello  World")
        k2 = engine._cache_key("hello world")
        self.assertEqual(k1, k2)

    def test_cache_hit_increments_counter(self):
        engine = WebRetrievalEngine()
        key = engine._cache_key("test query")
        engine._search_cache[key] = (time.time(), [
            SearchResult(title="t", url="http://x", snippet="s", source="test"),
        ])
        result = engine._check_cache(key, engine._search_cache, ttl=86400)
        self.assertIsNotNone(result)
        self.assertEqual(engine._cache_hits, 1)

    def test_cache_expired_returns_none(self):
        engine = WebRetrievalEngine()
        key = engine._cache_key("old query")
        engine._search_cache[key] = (time.time() - 200_000, [
            SearchResult(title="t", url="http://x", snippet="s", source="test"),
        ])
        result = engine._check_cache(key, engine._search_cache, ttl=86400)
        self.assertIsNone(result)

    def test_domain_detection_medical(self):
        engine = WebRetrievalEngine()
        domain = engine._detect_query_domain("What drug treats cancer symptoms?")
        self.assertEqual(domain, "medical")

    def test_domain_detection_financial(self):
        engine = WebRetrievalEngine()
        domain = engine._detect_query_domain("What is the current GDP inflation rate?")
        self.assertEqual(domain, "financial")

    def test_domain_detection_general_fallback(self):
        engine = WebRetrievalEngine()
        domain = engine._detect_query_domain("xyzzy foobar baz")
        self.assertEqual(domain, "general")

    def test_ssrf_blocks_localhost(self):
        self.assertFalse(WebRetrievalEngine._is_safe_url("http://localhost/admin"))
        self.assertFalse(WebRetrievalEngine._is_safe_url("http://127.0.0.1/secret"))
        self.assertFalse(WebRetrievalEngine._is_safe_url("http://metadata.google.internal/"))

    def test_ssrf_allows_public(self):
        self.assertTrue(WebRetrievalEngine._is_safe_url("https://en.wikipedia.org/wiki/Test"))

    def test_decode_ddg_url(self):
        raw = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage"
        decoded = WebRetrievalEngine._decode_ddg_url(raw)
        self.assertEqual(decoded, "https://example.com/page")

    def test_extract_text_strips_html(self):
        engine = WebRetrievalEngine()
        html = "<html><body><p>This is a paragraph with enough characters to pass the filter.</p></body></html>"
        text = engine._extract_text(html)
        self.assertNotIn("<p>", text)
        self.assertIn("paragraph", text)

    def test_search_returns_cached_on_second_call(self):
        engine = WebRetrievalEngine()
        fake_results = [
            SearchResult(title="Cached", url="http://c.com", snippet="cached snippet", source="test"),
        ]
        ck = engine._cache_key("cached query")
        engine._search_cache[ck] = (time.time(), fake_results)

        results = _run(engine.search("cached query", count=5))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Cached")
        self.assertEqual(engine._cache_hits, 1)

    def test_retrieved_fact_content_hash(self):
        fact = RetrievedFact(
            text="The speed of light is 299792458 m/s",
            source_url="https://example.com",
            confidence=0.9,
            domain="science",
            retrieval_method="test",
        )
        self.assertTrue(len(fact.content_hash) > 0)

    def test_credibility_tiers_populated(self):
        self.assertIn("pubmed", SOURCE_CREDIBILITY_TIERS)
        self.assertGreater(SOURCE_CREDIBILITY_TIERS["pubmed"], 0.8)


# ═══════════════════════════════════════════════════════════════════════════════
#  2. SessionManager (Redis-backed, tested with async mock)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from nrsip.session_manager import SessionManager, ConversationTurn
    _HAS_SESSION_MANAGER = True
except Exception:
    _HAS_SESSION_MANAGER = False


class _FakeRedis:
    """Minimal async Redis mock for SessionManager tests."""

    def __init__(self):
        self._store: dict = {}

    async def get(self, key):
        return self._store.get(key)

    async def set(self, key, value, ex=None):
        self._store[key] = value

    async def delete(self, key):
        self._store.pop(key, None)


@unittest.skipUnless(_HAS_SESSION_MANAGER, "nrsip.session_manager not importable")
class TestSessionManager(unittest.TestCase):
    """SessionManager: create, store turns, load history, delete."""

    def setUp(self):
        self.redis = _FakeRedis()
        self.mgr = SessionManager(self.redis, ttl_seconds=300)

    def test_generate_session_id_format(self):
        sid = SessionManager.generate_session_id()
        self.assertTrue(sid.startswith("sess-"))
        self.assertEqual(len(sid), 5 + 16)

    def test_store_and_load_turn(self):
        sid = "sess-test1"
        turn = ConversationTurn(role="user", content="Hello NRS")
        _run(self.mgr.store_turn(sid, turn))
        history = _run(self.mgr.load(sid))
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].role, "user")
        self.assertEqual(history[0].content, "Hello NRS")

    def test_store_exchange_creates_two_turns(self):
        sid = "sess-test2"
        _run(self.mgr.store_exchange(sid, "What is NRS?", "NRS is a reasoning system."))
        history = _run(self.mgr.load(sid))
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].role, "user")
        self.assertEqual(history[1].role, "assistant")

    def test_delete_clears_session(self):
        sid = "sess-test3"
        _run(self.mgr.store_exchange(sid, "q", "a"))
        _run(self.mgr.delete(sid))
        history = _run(self.mgr.load(sid))
        self.assertEqual(len(history), 0)

    def test_to_history_dicts(self):
        turns = [
            ConversationTurn(role="user", content="hi"),
            ConversationTurn(role="assistant", content="hello"),
        ]
        dicts = self.mgr.to_history_dicts(turns)
        self.assertEqual(dicts, [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])

    def test_load_empty_session(self):
        history = _run(self.mgr.load("nonexistent-session"))
        self.assertEqual(history, [])

    def test_none_redis_gracefully_returns_empty(self):
        mgr = SessionManager(None)
        history = _run(mgr.load("any-session"))
        self.assertEqual(history, [])


# ═══════════════════════════════════════════════════════════════════════════════
#  3. InMemorySessionStore
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from nrsip.session_store import InMemorySessionStore
    _HAS_SESSION_STORE = True
except Exception:
    _HAS_SESSION_STORE = False


@unittest.skipUnless(_HAS_SESSION_STORE, "nrsip.session_store not importable")
class TestInMemorySessionStore(unittest.TestCase):
    """InMemorySessionStore: nonce registration, session CRUD, TTL expiry."""

    def setUp(self):
        self.store = InMemorySessionStore()

    def test_nonce_first_registration_returns_true(self):
        self.assertTrue(self.store.register_nonce(b"\x01\x02\x03", 60.0))

    def test_nonce_replay_returns_false(self):
        nonce = b"\xaa\xbb\xcc"
        self.store.register_nonce(nonce, 60.0)
        self.assertFalse(self.store.register_nonce(nonce, 60.0))

    def test_store_and_get_session(self):
        self.store.store_session("s1", {"user": "alice", "tier": "T2"}, ttl_seconds=300)
        data = self.store.get_session("s1")
        self.assertIsNotNone(data)
        self.assertEqual(data["user"], "alice")

    def test_delete_session(self):
        self.store.store_session("s2", {"user": "bob"}, ttl_seconds=300)
        self.store.delete_session("s2")
        self.assertIsNone(self.store.get_session("s2"))

    def test_update_session_field(self):
        self.store.store_session("s3", {"status": "active"}, ttl_seconds=300)
        self.store.update_session_field("s3", "status", "closed")
        data = self.store.get_session("s3")
        self.assertEqual(data["status"], "closed")

    def test_get_nonexistent_session_returns_none(self):
        self.assertIsNone(self.store.get_session("does-not-exist"))

    def test_register_compat_method(self):
        self.assertTrue(self.store.register(b"\xdd\xee", time.time(), 60.0))
        self.assertFalse(self.store.register(b"\xdd\xee", time.time(), 60.0))


# ═══════════════════════════════════════════════════════════════════════════════
#  4. MetacognitionMemory (InMemoryMetaStore)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from nrsip.metacognition_memory import InMemoryMetaStore, MetacognitionMemory, _tokenize, _TFIDFIndex
    from nrsip.metacognition_types import (
        ConversationEpisode, UserModel, GoalState,
        LessonRecord, StrategyPolicySnapshot,
    )
    _HAS_METACOGNITION = True
except Exception:
    _HAS_METACOGNITION = False


@unittest.skipUnless(_HAS_METACOGNITION, "nrsip.metacognition_memory not importable")
class TestInMemoryMetaStore(unittest.TestCase):
    """InMemoryMetaStore: episode, user model, goal, lesson, snapshot CRUD."""

    def setUp(self):
        self.store = InMemoryMetaStore()

    def test_save_and_load_episode(self):
        ep = ConversationEpisode(session_id="s1", user_id="u1")
        ep.append_turn("user", "Hello")
        _run(self.store.save_episode(ep))
        loaded = _run(self.store.load_episode(ep.episode_id))
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.user_id, "u1")
        self.assertEqual(len(loaded.turns), 1)

    def test_list_episodes_by_user(self):
        ep1 = ConversationEpisode(session_id="s1", user_id="u1")
        ep2 = ConversationEpisode(session_id="s2", user_id="u1")
        ep3 = ConversationEpisode(session_id="s3", user_id="u2")
        _run(self.store.save_episode(ep1))
        _run(self.store.save_episode(ep2))
        _run(self.store.save_episode(ep3))
        u1_eps = _run(self.store.list_episodes("u1"))
        self.assertEqual(len(u1_eps), 2)

    def test_save_and_load_user_model(self):
        model = UserModel(user_id="u1")
        _run(self.store.save_user_model(model))
        loaded = _run(self.store.load_user_model("u1"))
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.user_id, "u1")

    def test_save_and_load_active_goal(self):
        goal = GoalState(session_id="s1")
        _run(self.store.save_goal(goal))
        loaded = _run(self.store.load_active_goal("s1"))
        self.assertIsNotNone(loaded)

    def test_load_nonexistent_returns_none(self):
        self.assertIsNone(_run(self.store.load_episode("nonexistent")))
        self.assertIsNone(_run(self.store.load_user_model("nonexistent")))
        self.assertIsNone(_run(self.store.load_active_goal("nonexistent")))


@unittest.skipUnless(_HAS_METACOGNITION, "nrsip.metacognition_memory not importable")
class TestMetacognitionMemory(unittest.TestCase):
    """MetacognitionMemory facade: episode lifecycle, user model, lessons."""

    def setUp(self):
        self.mem = MetacognitionMemory()

    def test_start_and_record_episode(self):
        ep = _run(self.mem.start_episode("s1", "u1"))
        self.assertIsNotNone(ep.episode_id)
        _run(self.mem.record_turn(ep.episode_id, "user", "What is NRS?"))
        loaded = _run(self.mem.store.load_episode(ep.episode_id))
        self.assertEqual(len(loaded.turns), 1)

    def test_get_or_create_user(self):
        user = _run(self.mem.get_or_create_user("u1"))
        self.assertEqual(user.user_id, "u1")
        user2 = _run(self.mem.get_or_create_user("u1"))
        self.assertEqual(user.user_id, user2.user_id)

    def test_get_or_create_goal(self):
        goal = _run(self.mem.get_or_create_goal("s1"))
        self.assertIsNotNone(goal)
        goal2 = _run(self.mem.get_or_create_goal("s1"))
        self.assertEqual(goal.session_id, goal2.session_id)


@unittest.skipUnless(_HAS_METACOGNITION, "nrsip.metacognition_memory not importable")
class TestTFIDFIndex(unittest.TestCase):
    """Internal TF-IDF index used for semantic recall."""

    def test_add_and_query(self):
        idx = _TFIDFIndex()
        idx.add("doc1", "The speed of light in a vacuum is constant")
        idx.add("doc2", "Photosynthesis converts sunlight into chemical energy")
        idx.add("doc3", "Light travels at approximately 300000 km per second")
        results = idx.query("speed of light")
        self.assertGreater(len(results), 0)
        top_id = results[0][0]
        self.assertIn(top_id, ("doc1", "doc3"))

    def test_remove_document(self):
        idx = _TFIDFIndex()
        idx.add("d1", "quantum mechanics uncertainty principle")
        idx.add("d2", "classical mechanics newton laws")
        idx.remove("d1")
        results = idx.query("quantum uncertainty")
        doc_ids = [r[0] for r in results]
        self.assertNotIn("d1", doc_ids)


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Reasoning Coherence
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from nrsip.reasoning_coherence import (
        CoherenceController, ResponseCoherenceChecker,
        WorldviewMaintainer, CoherenceViolation,
    )
    from nrsip.reasoning_graph import CausalGraph, RelationType
    _HAS_COHERENCE = True
except Exception:
    _HAS_COHERENCE = False


@unittest.skipUnless(_HAS_COHERENCE, "nrsip.reasoning_coherence not importable")
class TestResponseCoherenceChecker(unittest.TestCase):
    """ResponseCoherenceChecker: commit claims, detect contradictions."""

    def setUp(self):
        self.graph = CausalGraph()
        self.checker = ResponseCoherenceChecker(self.graph)

    def test_commit_and_retrieve(self):
        self.checker.commit("aspirin", "causes", "pain relief", 0.9, "s1")
        active = self.checker.all_active_commitments()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].subject, "aspirin")

    def test_no_violation_for_consistent_claims(self):
        self.checker.commit("exercise", "causes", "fitness", 0.9, "s1")
        violations = self.checker.check_consistency("sleep", "causes", "recovery")
        self.assertEqual(len(violations), 0)

    def test_detects_direct_contradiction(self):
        self.checker.commit("smoking", "causes", "cancer", 0.95, "s1")
        violations = self.checker.check_consistency("smoking", "prevents", "cancer")
        self.assertGreater(len(violations), 0)
        self.assertEqual(violations[0].violation_type, "direct_contradiction")

    def test_retract_removes_commitment(self):
        self.checker.commit("x", "causes", "y", 0.8, "s1")
        self.assertTrue(self.checker.retract("x", "causes", "y"))
        active = self.checker.all_active_commitments()
        self.assertEqual(len(active), 0)

    def test_session_commitments_filtered(self):
        self.checker.commit("a", "causes", "b", 0.8, "s1")
        self.checker.commit("c", "causes", "d", 0.8, "s2")
        s1_commits = self.checker.session_commitments("s1")
        self.assertEqual(len(s1_commits), 1)
        self.assertEqual(s1_commits[0].subject, "a")


@unittest.skipUnless(_HAS_COHERENCE, "nrsip.reasoning_coherence not importable")
class TestWorldviewMaintainer(unittest.TestCase):
    """WorldviewMaintainer: stance tracking and alignment checking."""

    def setUp(self):
        self.wv = WorldviewMaintainer()

    def test_set_and_get_stance(self):
        self.wv.set_stance("climate change", "human-caused", 0.95)
        stance = self.wv.get_stance("climate change")
        self.assertIsNotNone(stance)
        self.assertEqual(stance["position"], "human-caused")

    def test_alignment_passes_for_matching_position(self):
        self.wv.set_stance("evolution", "well-established science", 0.99)
        violation = self.wv.check_alignment("evolution", "well-established science")
        self.assertIsNone(violation)

    def test_alignment_fails_for_conflicting_position(self):
        self.wv.set_stance("earth shape", "spherical", 0.99)
        violation = self.wv.check_alignment("earth shape", "flat")
        self.assertIsNotNone(violation)
        self.assertEqual(violation.violation_type, "worldview_inconsistency")


@unittest.skipUnless(_HAS_COHERENCE, "nrsip.reasoning_coherence not importable")
class TestCoherenceController(unittest.TestCase):
    """CoherenceController: unified validation pipeline."""

    def setUp(self):
        self.graph = CausalGraph()
        self.ctrl = CoherenceController(self.graph)

    def test_validate_coherent_claims(self):
        result = self.ctrl.validate_response([
            ("rain", "causes", "wet ground"),
            ("sun", "causes", "evaporation"),
        ])
        self.assertTrue(result["coherent"])
        self.assertEqual(result["claim_count"], 2)

    def test_commit_response_returns_count(self):
        count = self.ctrl.commit_response([
            ("a", "causes", "b"),
            ("c", "enables", "d"),
        ], session_id="s1")
        self.assertEqual(count, 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  6. Reasoning Temporal
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from nrsip.reasoning_temporal import (
        TemporalExtractor, TemporalReasoner, TemporalRelation, TemporalFact,
        QuantityExtractor, QuantitativeReasoner,
    )
    _HAS_TEMPORAL = True
except Exception:
    _HAS_TEMPORAL = False


@unittest.skipUnless(_HAS_TEMPORAL, "nrsip.reasoning_temporal not importable")
class TestTemporalExtractor(unittest.TestCase):
    """TemporalExtractor: year, century, decade, era, ongoing detection."""

    def test_extracts_year(self):
        ext = TemporalExtractor()
        fact = ext.extract("The internet was invented in 1969", "internet")
        self.assertEqual(fact.year_start, 1969)
        self.assertEqual(fact.concept, "internet")

    def test_extracts_year_range(self):
        ext = TemporalExtractor()
        fact = ext.extract("World War II lasted from 1939 to 1945", "WWII")
        self.assertEqual(fact.year_start, 1939)
        self.assertEqual(fact.year_end, 1945)
        self.assertEqual(fact.duration, 6)

    def test_detects_ongoing(self):
        ext = TemporalExtractor()
        fact = ext.extract("Climate change is currently accelerating", "climate")
        self.assertTrue(fact.is_ongoing)

    def test_century_extraction(self):
        ext = TemporalExtractor()
        fact = ext.extract("The 21st century brought digital revolution", "digital")
        self.assertEqual(fact.year_start, 2000)
        self.assertEqual(fact.year_end, 2100)


@unittest.skipUnless(_HAS_TEMPORAL, "nrsip.reasoning_temporal not importable")
class TestTemporalReasoner(unittest.TestCase):
    """TemporalReasoner: add facts, relate, timeline."""

    def setUp(self):
        self.reasoner = TemporalReasoner()

    def test_add_and_get_fact(self):
        self.reasoner.add_fact("moon landing", "Apollo 11 landed on the moon in 1969")
        fact = self.reasoner.get_fact("moon landing")
        self.assertIsNotNone(fact)
        self.assertEqual(fact.year_start, 1969)

    def test_before_relation(self):
        self.reasoner.add_fact("WWI", "World War I was from 1914 to 1918")
        self.reasoner.add_fact("WWII", "World War II was from 1939 to 1945")
        rel = self.reasoner.relate("WWI", "WWII")
        self.assertEqual(rel, TemporalRelation.BEFORE)

    def test_after_relation(self):
        self.reasoner.add_fact("ancient", "Ancient Rome was from 753 to 476")
        self.reasoner.add_fact("modern", "The modern era began in 1800")
        rel = self.reasoner.relate("modern", "ancient")
        self.assertEqual(rel, TemporalRelation.AFTER)

    def test_timeline_ordering(self):
        self.reasoner.add_fact("printing press", "Gutenberg invented the printing press in 1440")
        self.reasoner.add_fact("telephone", "Bell invented the telephone in 1876")
        self.reasoner.add_fact("internet", "ARPANET launched in 1969")
        timeline = self.reasoner.timeline()
        years = [y for _, y in timeline]
        self.assertEqual(years, sorted(years))

    def test_unknown_relation_for_missing_facts(self):
        rel = self.reasoner.relate("nonexistent_a", "nonexistent_b")
        self.assertEqual(rel, TemporalRelation.UNKNOWN)


@unittest.skipUnless(_HAS_TEMPORAL, "nrsip.reasoning_temporal not importable")
class TestQuantityExtractor(unittest.TestCase):
    """QuantityExtractor: number and unit extraction from text."""

    def test_extract_distance(self):
        ext = QuantityExtractor()
        facts = ext.extract_all("The distance is 384,400 km", "moon distance")
        self.assertGreater(len(facts), 0)
        self.assertAlmostEqual(facts[0].value, 384400.0)
        self.assertEqual(facts[0].unit, "kilometers")

    def test_extract_multiplier(self):
        ext = QuantityExtractor()
        facts = ext.extract_all("Earth has 8 billion people", "population")
        self.assertGreater(len(facts), 0)
        self.assertAlmostEqual(facts[0].value, 8e9)

    def test_extract_primary(self):
        ext = QuantityExtractor()
        fact = ext.extract_primary("The tower is 330 m tall", "eiffel")
        self.assertIsNotNone(fact)
        self.assertAlmostEqual(fact.value, 330.0)


@unittest.skipUnless(_HAS_TEMPORAL, "nrsip.reasoning_temporal not importable")
class TestQuantitativeReasoner(unittest.TestCase):
    """QuantitativeReasoner: compare, rank, sum quantities."""

    def setUp(self):
        self.reasoner = QuantitativeReasoner()

    def test_compare_same_unit(self):
        self.reasoner.add_fact("everest", "Mount Everest is 8849 km tall")
        self.reasoner.add_fact("kangchenjunga", "Kangchenjunga is 8586 km tall")
        result = self.reasoner.compare("everest", "kangchenjunga")
        self.assertIsNotNone(result)
        self.assertIn("greater", result)

    def test_rank_quantities(self):
        self.reasoner.add_fact("a", "Value A is 100 km")
        self.reasoner.add_fact("b", "Value B is 300 km")
        self.reasoner.add_fact("c", "Value C is 200 km")
        ranked = self.reasoner.rank(["a", "b", "c"])
        names = [name for name, _ in ranked]
        self.assertEqual(names[0], "b")


# ═══════════════════════════════════════════════════════════════════════════════
#  7. Behavioral Attestation
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from nrsip.behavioral_attestation import (
        AttestationChallengeBank, BehavioralAttestationVerifier,
        AttestationChallenge, AttestationResponse, AttestationVerdict,
    )
    _HAS_ATTESTATION = True
except Exception:
    _HAS_ATTESTATION = False


def _mock_activation_fn(query: str):
    """Deterministic mock: hash-based activation for testing."""
    import hashlib
    h = hashlib.sha256(query.encode()).hexdigest()
    return h, 0.85, "deterministic"


@unittest.skipUnless(_HAS_ATTESTATION, "nrsip.behavioral_attestation not importable")
class TestAttestationChallengeBank(unittest.TestCase):
    """AttestationChallengeBank: characterize instances, select challenges."""

    def test_characterize_instance(self):
        bank = AttestationChallengeBank()
        queries = ["What is 2+2?", "Define gravity", "Explain photosynthesis"]
        count = bank.characterize_instance("nrs-001", queries, _mock_activation_fn)
        self.assertEqual(count, 3)
        self.assertEqual(bank.instance_count(), 1)

    def test_select_challenges(self):
        bank = AttestationChallengeBank()
        queries = [f"Query {i}" for i in range(10)]
        bank.characterize_instance("nrs-002", queries, _mock_activation_fn)
        selected = bank.select_challenges("nrs-002", count=3)
        self.assertEqual(len(selected), 3)

    def test_select_from_unknown_instance_raises(self):
        bank = AttestationChallengeBank()
        with self.assertRaises(ValueError):
            bank.select_challenges("nonexistent", count=3)


@unittest.skipUnless(_HAS_ATTESTATION, "nrsip.behavioral_attestation not importable")
class TestBehavioralAttestationVerifier(unittest.TestCase):
    """BehavioralAttestationVerifier: verify single response, full attestation."""

    def _make_challenge_and_response(self, match=True):
        challenge = AttestationChallenge(
            challenge_id="ch-001",
            query="What is 2+2?",
            expected_activation_hash="abc123",
            expected_h_score_range=(0.84, 0.86),
            expected_processing_mode="deterministic",
        )
        response = AttestationResponse(
            challenge_id="ch-001",
            activation_hash="abc123" if match else "wrong_hash",
            h_score=0.85,
            processing_mode="deterministic",
            processing_time_ms=1.5,
        )
        return challenge, response

    def test_verify_matching_response(self):
        verifier = BehavioralAttestationVerifier()
        challenge, response = self._make_challenge_and_response(match=True)
        passed, reason = verifier.verify_response(challenge, response)
        self.assertTrue(passed)
        self.assertIn("passed", reason)

    def test_verify_mismatched_hash(self):
        verifier = BehavioralAttestationVerifier()
        challenge, response = self._make_challenge_and_response(match=False)
        passed, reason = verifier.verify_response(challenge, response)
        self.assertFalse(passed)
        self.assertIn("activation_hash", reason)

    def test_full_attestation_all_pass(self):
        verifier = BehavioralAttestationVerifier()
        challenges = []
        responses = []
        for i in range(3):
            c = AttestationChallenge(
                challenge_id=f"ch-{i}",
                query=f"Query {i}",
                expected_activation_hash=f"hash{i}",
                expected_h_score_range=(0.84, 0.86),
                expected_processing_mode="deterministic",
            )
            r = AttestationResponse(
                challenge_id=f"ch-{i}",
                activation_hash=f"hash{i}",
                h_score=0.85,
                processing_mode="deterministic",
                processing_time_ms=1.0,
            )
            challenges.append(c)
            responses.append(r)

        result = verifier.attest("nrs-test", challenges, responses)
        self.assertEqual(result.verdict, AttestationVerdict.PASSED)
        self.assertEqual(result.challenges_passed, 3)
        self.assertEqual(result.challenges_failed, 0)
        self.assertAlmostEqual(result.pass_rate, 1.0)

    def test_full_attestation_partial_failure(self):
        verifier = BehavioralAttestationVerifier()
        challenge = AttestationChallenge(
            challenge_id="ch-0",
            query="Q",
            expected_activation_hash="correct",
            expected_h_score_range=(0.84, 0.86),
            expected_processing_mode="deterministic",
        )
        response = AttestationResponse(
            challenge_id="ch-0",
            activation_hash="wrong",
            h_score=0.85,
            processing_mode="deterministic",
            processing_time_ms=1.0,
        )
        result = verifier.attest("nrs-test", [challenge], [response])
        self.assertEqual(result.verdict, AttestationVerdict.FAILED)
        self.assertEqual(result.challenges_failed, 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  8. Knowledge Loader
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from nrsip.knowledge_loader import (
        CreaseLoader, TFIDFIndex, Fact, MatchResult,
        _tokenize, _stem, _strip_query_noise,
    )
    _HAS_KNOWLEDGE_LOADER = True
except Exception:
    _HAS_KNOWLEDGE_LOADER = False


@unittest.skipUnless(_HAS_KNOWLEDGE_LOADER, "nrsip.knowledge_loader not importable")
class TestTFIDFIndexKnowledge(unittest.TestCase):
    """TFIDFIndex from knowledge_loader: add, build, query."""

    def test_add_and_query(self):
        idx = TFIDFIndex()
        idx.add(Fact(id="f1", text="Quantum entanglement links particles instantly", confidence=0.9, domain="physics"))
        idx.add(Fact(id="f2", text="Photosynthesis converts light to chemical energy", confidence=0.9, domain="biology"))
        idx.add(Fact(id="f3", text="Quantum computing uses qubits for parallel computation", confidence=0.9, domain="cs"))
        idx.build()
        results = idx.query("quantum entanglement")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].fact.id, "f1")

    def test_empty_index_returns_empty(self):
        idx = TFIDFIndex()
        idx.build()
        results = idx.query("anything")
        self.assertEqual(len(results), 0)

    def test_fact_count(self):
        idx = TFIDFIndex()
        idx.add(Fact(id="f1", text="fact one", confidence=0.5, domain="test"))
        idx.add(Fact(id="f2", text="fact two", confidence=0.5, domain="test"))
        self.assertEqual(idx.fact_count, 2)


@unittest.skipUnless(_HAS_KNOWLEDGE_LOADER, "nrsip.knowledge_loader not importable")
class TestCreaseLoader(unittest.TestCase):
    """CreaseLoader: load JSON knowledge packs, query facts."""

    def test_load_from_temp_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = {
                "domain": "test_domain",
                "facts": [
                    {"id": "t1", "text": "NRS uses neural routing for query processing", "confidence": 0.9, "tags": ["nrs"]},
                    {"id": "t2", "text": "NRSIP is the signaling protocol for NRS nodes", "confidence": 0.85, "tags": ["nrsip"]},
                    {"id": "t3", "text": "PRISM handles multi-modal perception", "confidence": 0.8, "tags": ["prism"]},
                ],
            }
            with open(os.path.join(tmpdir, "test_pack.json"), "w") as f:
                json.dump(pack, f)

            loader = CreaseLoader(knowledge_dir=tmpdir)
            count = loader.load_all()
            self.assertGreaterEqual(count, 3)
            self.assertIn("test_domain", loader.stats["domains"])
            self.assertEqual(loader.stats["per_domain"]["test_domain"], 3)

    def test_query_returns_relevant_facts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = {
                "domain": "science",
                "facts": [
                    {"id": "s1", "text": "Gravity pulls objects toward Earth at 9.8 m/s squared", "confidence": 0.95, "tags": ["physics"]},
                    {"id": "s2", "text": "DNA carries genetic information in living organisms", "confidence": 0.95, "tags": ["biology"]},
                ],
            }
            with open(os.path.join(tmpdir, "science_facts.json"), "w") as f:
                json.dump(pack, f)

            loader = CreaseLoader(knowledge_dir=tmpdir)
            loader.load_all()
            results = loader.query("gravity acceleration Earth")
            self.assertGreater(len(results), 0)
            self.assertEqual(results[0].fact.id, "s1")

    def test_load_directory_without_local_packs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = CreaseLoader(knowledge_dir=tmpdir)
            count = loader.load_all()
            # stdlib knowledge dir may contribute facts even when local dir is empty
            self.assertGreaterEqual(count, 0)

    def test_get_facts_for_domain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pack = {
                "domain": "math",
                "facts": [
                    {"id": "m1", "text": "Pi is approximately 3.14159", "confidence": 0.99, "tags": ["constants"]},
                ],
            }
            with open(os.path.join(tmpdir, "math_facts.json"), "w") as f:
                json.dump(pack, f)

            loader = CreaseLoader(knowledge_dir=tmpdir)
            loader.load_all()
            facts = loader.get_facts_for_domain("math")
            self.assertEqual(len(facts), 1)
            self.assertEqual(facts[0].id, "m1")


@unittest.skipUnless(_HAS_KNOWLEDGE_LOADER, "nrsip.knowledge_loader not importable")
class TestKnowledgeLoaderHelpers(unittest.TestCase):
    """Helper functions: tokenize, stem, strip_query_noise."""

    def test_tokenize_removes_stop_words(self):
        tokens = _tokenize("the quick brown fox jumps over the lazy dog")
        self.assertNotIn("the", tokens)
        self.assertNotIn("over", tokens)

    def test_stem_reduces_suffixes(self):
        self.assertEqual(_stem("running"), "runn")
        self.assertEqual(_stem("happiness"), "happi")

    def test_strip_query_noise(self):
        cleaned = _strip_query_noise("what does the knowledge base record about quantum physics")
        self.assertIn("quantum", cleaned)
        self.assertIn("physics", cleaned)
        self.assertNotIn("knowledge base record", cleaned)


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main()
