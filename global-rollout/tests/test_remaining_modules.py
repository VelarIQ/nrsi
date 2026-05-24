from __future__ import annotations
import os
import time
import unittest


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Signal Compression
# ═══════════════════════════════════════════════════════════════════════════════

class TestSignalCompression(unittest.TestCase):
    def _make_raw(self):
        from nrsip.signal_compression import RawActivation
        ids = list(range(50))
        sims = {i: 0.5 + i * 0.01 for i in ids}
        domain_signals = {"language": 20, "logic": 15, "math": 10, "creative": 5}
        return RawActivation(
            active_neuron_ids=ids,
            similarities=sims,
            domain_signals=domain_signals,
            query_hash="abc123",
        )

    def test_raw_to_digest(self):
        from nrsip.signal_compression import ActivationDigest
        raw = self._make_raw()
        digest = raw.compress_to_digest()
        self.assertIsInstance(digest, ActivationDigest)
        self.assertEqual(digest.active_count, 50)
        self.assertIn("language", digest.domain_distribution)

    def test_digest_to_routing(self):
        from nrsip.signal_compression import RoutingPacket
        raw = self._make_raw()
        digest = raw.compress_to_digest()
        packet = digest.to_routing_packet(tier="T2", confidence=0.9)
        self.assertIsInstance(packet, RoutingPacket)
        self.assertEqual(packet.tier, "T2")
        self.assertAlmostEqual(packet.confidence, 0.9)

    def test_compression_ratio(self):
        from nrsip.signal_compression import RawActivation, PRODUCTION_ACTIVE_K
        ids = list(range(500))
        sims = {i: 0.5 + i * 0.001 for i in ids}
        domain_signals = {"language": 200, "logic": 150, "math": 100}
        raw = RawActivation(
            active_neuron_ids=ids,
            similarities=sims,
            domain_signals=domain_signals,
            total_neurons=100_000,
            query_hash="compratio",
        )
        digest = raw.compress_to_digest()
        packet = digest.to_routing_packet()
        self.assertGreater(raw.estimated_bytes, digest.estimated_bytes)
        self.assertGreater(digest.estimated_bytes, packet.estimated_bytes)

    def test_serialize_deserialize(self):
        from nrsip.signal_compression import RoutingPacket
        raw = self._make_raw()
        packet = raw.compress_to_digest().to_routing_packet(
            tier="T1", confidence=0.85, h_score=0.92, instance_id="node-1",
        )
        wire = packet.serialize()
        self.assertGreater(len(wire), 0)
        restored = RoutingPacket.deserialize(wire)
        self.assertEqual(restored.tier, "T1")


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Media Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestMediaPipeline(unittest.TestCase):
    def test_process_text(self):
        from nrsip.media_pipeline import MediaProcessor, MediaDigest, Modality
        proc = MediaProcessor()
        digest = proc.process(b"hello world", Modality.TEXT)
        self.assertIsInstance(digest, MediaDigest)
        self.assertEqual(digest.modality, Modality.TEXT)

    def test_process_image(self):
        from nrsip.media_pipeline import MediaProcessor, Modality
        proc = MediaProcessor()
        digest = proc.process(b"\x89PNG fake image data", Modality.IMAGE)
        self.assertEqual(digest.modality, Modality.IMAGE)
        self.assertIn("objects", digest.entities)

    def test_digest_has_required_fields(self):
        from nrsip.media_pipeline import MediaProcessor, Modality
        proc = MediaProcessor()
        digest = proc.process(b"sample document content", Modality.DOCUMENT)
        self.assertIsInstance(digest.entities, dict)
        self.assertIsInstance(digest.extracted_text, str)
        self.assertEqual(len(digest.feature_embedding), 768)

    def test_content_ref_created(self):
        from nrsip.media_pipeline import MediaProcessor, Modality
        proc = MediaProcessor()
        digest = proc.process(b"audio bytes", Modality.AUDIO)
        self.assertTrue(len(digest.content_ref.content_hash) > 0)
        self.assertEqual(digest.content_ref.size_bytes, len(b"audio bytes"))


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Orchestration
# ═══════════════════════════════════════════════════════════════════════════════

class TestOrchestration(unittest.TestCase):
    def test_decompose_simple(self):
        from nrsip.orchestration import TaskDecomposer
        decomp = TaskDecomposer()
        tasks = decomp.decompose("What is the weather today?")
        self.assertGreaterEqual(len(tasks), 1)
        self.assertTrue(all(t.task_id for t in tasks))

    def test_decompose_multi_domain(self):
        from nrsip.orchestration import TaskDecomposer
        decomp = TaskDecomposer()
        tasks = decomp.decompose(
            "Calculate the financial risk of a medical treatment"
        )
        self.assertGreaterEqual(len(tasks), 2)
        domains = {t.domain for t in tasks}
        self.assertTrue(domains & {"medical", "financial"})

    def test_aggregate_results(self):
        from nrsip.orchestration import ResultAggregator, SubTaskResult, MergeStrategy
        agg = ResultAggregator()
        results = [
            SubTaskResult(
                task_id="t1", instance_id="i1", result="answer-1",
                confidence=0.9, h_score=0.85, h_verdict="validated",
                processing_mode="DETERMINISTIC", processing_time_ms=5.0,
            ),
            SubTaskResult(
                task_id="t2", instance_id="i2", result="answer-2",
                confidence=0.8, h_score=0.80, h_verdict="acceptable",
                processing_mode="PROBABILISTIC", processing_time_ms=8.0,
            ),
        ]
        composite = agg.aggregate("test query", results, MergeStrategy.SECTION_LABELED)
        self.assertEqual(composite.successful_sub_tasks, 2)
        self.assertGreater(composite.overall_h_score, 0)

    def test_coordinator_end_to_end(self):
        from nrsip.orchestration import (
            OrchestrationCoordinator, SubTaskResult, MergeStrategy,
        )
        coord = OrchestrationCoordinator()
        tasks = coord.plan("Explain medical treatment compliance regulations")
        self.assertGreaterEqual(len(tasks), 1)
        results = [
            SubTaskResult(
                task_id=t.task_id, instance_id="inst-1", result=f"result-{t.domain}",
                confidence=0.88, h_score=0.82, h_verdict="acceptable",
                processing_mode="HYBRID", processing_time_ms=4.0,
            )
            for t in tasks
        ]
        merged = coord.merge("Explain medical treatment compliance", results)
        self.assertGreater(merged.overall_confidence, 0)


# ═══════════════════════════════════════════════════════════════════════════════
#  4. PoVI (Proof of Validated Inference)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPoVI(unittest.TestCase):
    def _make_validators(self, n=3, agree=True):
        from nrsip.povi import ValidatorResult
        results = []
        for i in range(n):
            results.append(ValidatorResult(
                validator_id=f"v{i}",
                provider=f"provider-{i}",
                region=f"region-{i}",
                activation_hash="abc123" if agree else f"hash-{i}",
                h_score=0.92 if agree else 0.5 + i * 0.1,
                validation_result="correct" if agree else f"result-{i}",
                processing_mode="DETERMINISTIC",
                tier="T2",
                processing_time_ms=10.0,
            ))
        return results

    def test_generate_proof(self):
        from nrsip.povi import PoVIConsensusEngine, PoVIProof
        engine = PoVIConsensusEngine()
        results = self._make_validators(3, agree=True)
        consensus = engine.execute_round("query_hash_1", results)
        proof = engine.generate_proof(consensus, results)
        self.assertIsInstance(proof, PoVIProof)
        self.assertTrue(len(proof.proof_hash) > 0)

    def test_verify_proof_confirmed(self):
        from nrsip.povi import PoVIConsensusEngine, PoVIVerdict
        engine = PoVIConsensusEngine()
        results = self._make_validators(3, agree=True)
        consensus = engine.execute_round("query_hash_2", results)
        self.assertEqual(consensus.verdict, PoVIVerdict.CONFIRMED)
        self.assertEqual(consensus.agreeing_count, 3)

    def test_invalid_proof_rejected(self):
        from nrsip.povi import PoVIConsensusEngine, PoVIVerdict
        engine = PoVIConsensusEngine()
        results = self._make_validators(3, agree=False)
        consensus = engine.execute_round("query_hash_3", results)
        self.assertIn(consensus.verdict, (PoVIVerdict.REJECTED, PoVIVerdict.ACCEPTED))
        self.assertGreater(len(consensus.disagreeing_validators), 0)


# ═══════════════════════════════════════════════════════════════════════════════
#  5. IPv6 Addressing
# ═══════════════════════════════════════════════════════════════════════════════

class TestIPv6Addressing(unittest.TestCase):
    def test_address_creation(self):
        from nrsip.ipv6_addressing import encode_neuron_address
        import ipaddress
        addr = encode_neuron_address(
            provider_id=1, region_id=2, crease_id=100, neuron_id=42,
        )
        ipv6 = addr.ipv6
        self.assertIsInstance(ipv6, ipaddress.IPv6Address)
        self.assertTrue(len(addr.ipv6_str) > 0)

    def test_parse_roundtrip(self):
        from nrsip.ipv6_addressing import encode_neuron_address, decode_neuron_address
        original = encode_neuron_address(
            provider_id=5, region_id=3, crease_id=999, neuron_id=7777,
        )
        decoded = decode_neuron_address(original.ipv6)
        self.assertEqual(decoded.provider_id, original.provider_id)
        self.assertEqual(decoded.region_id, original.region_id)
        self.assertEqual(decoded.crease_id, original.crease_id)
        self.assertEqual(decoded.neuron_id, original.neuron_id)

    def test_prefix_matches(self):
        from nrsip.ipv6_addressing import encode_neuron_address, PREFIX_NETWORK
        addr = encode_neuron_address(
            provider_id=1, region_id=1, crease_id=0, neuron_id=1,
        )
        self.assertIn(addr.ipv6, PREFIX_NETWORK)

    def test_in_allocation(self):
        from nrsip.ipv6_addressing import encode_neuron_address
        addr = encode_neuron_address(
            provider_id=10, region_id=4, crease_id=500, neuron_id=100,
        )
        self.assertTrue(addr.in_allocation())


# ═══════════════════════════════════════════════════════════════════════════════
#  6. Address Revocation
# ═══════════════════════════════════════════════════════════════════════════════

class TestAddressRevocation(unittest.TestCase):
    def test_revoke_address(self):
        from nrsip.address_revocation import RevocationService, RevocationReason
        svc = RevocationService(issuer_id="admin-1")
        ann = svc.revoke(
            address="2620:be1a::1",
            entity_id="entity-bad",
            reason=RevocationReason.SECURITY_INCIDENT,
        )
        self.assertTrue(len(ann.revocation_id) > 0)
        self.assertTrue(len(ann.signature) > 0)

    def test_is_revoked(self):
        from nrsip.address_revocation import RevocationService, RevocationReason
        svc = RevocationService(issuer_id="admin-2")
        svc.revoke("2620:be1a::2", "entity-x", RevocationReason.ADMIN_ACTION)
        self.assertTrue(svc.store.is_revoked("2620:be1a::2"))
        self.assertFalse(svc.store.is_revoked("2620:be1a::999"))

    def test_revocation_announcement_verify(self):
        from nrsip.address_revocation import RevocationService, RevocationReason
        svc = RevocationService(issuer_id="admin-3")
        ann = svc.revoke("2620:be1a::3", "entity-y", RevocationReason.KEY_COMPROMISE)
        self.assertTrue(ann.verify())
        ann.signature = "tampered"
        self.assertFalse(ann.verify())


# ═══════════════════════════════════════════════════════════════════════════════
#  7. Payments
# ═══════════════════════════════════════════════════════════════════════════════

class TestPayments(unittest.TestCase):
    def test_pay_per_query(self):
        from nrsip.payments import PaymentChannel, PaymentCurrency
        ch = PaymentChannel(
            channel_id="ch-1", payer="alice", payee="bob",
            prefunded_amount=10.0, currency=PaymentCurrency.USD,
        )
        rec = ch.pay_per_query(0.01, query_hash="q1", povi_proof_hash="proof1")
        self.assertTrue(rec.validated)
        self.assertEqual(rec.amount, 0.01)
        self.assertAlmostEqual(ch.balance, 10.0 - 0.01)

    def test_channel_open_close(self):
        from nrsip.payments import PaymentChannel, ChannelState
        ch = PaymentChannel(
            channel_id="ch-2", payer="alice", payee="carol",
            prefunded_amount=5.0,
        )
        self.assertEqual(ch.state, ChannelState.ACTIVE)
        ch.pay_per_query(1.0, "q1")
        ch.pay_per_query(2.0, "q2")
        summary = ch.close()
        self.assertEqual(ch.state, ChannelState.CLOSED)
        self.assertAlmostEqual(summary["total_paid"], 3.0)
        self.assertEqual(summary["total_queries"], 2)

    def test_marketplace_listing(self):
        from nrsip.payments import ServiceMarketplace
        mp = ServiceMarketplace()
        listing = mp.register_service(
            provider="velariq",
            capability="medical_nrs",
            price_per_query=0.05,
            trust_score=4.5,
            domains=["medical"],
        )
        self.assertTrue(listing.active)
        found = mp.discover(capability="medical_nrs")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].provider, "velariq")


# ═══════════════════════════════════════════════════════════════════════════════
#  8. Registry Hierarchy
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegistryHierarchy(unittest.TestCase):
    def _build_registry(self):
        from nrsip.registry_hierarchy import (
            RootRegistry, ProviderRegistry, ProviderBlock, RegistryEntry,
        )
        root = RootRegistry("americas")
        provider_reg = ProviderRegistry("p1", "ProviderOne", "2620:be1a:0001::/48")
        provider_reg.register(RegistryEntry(
            entity_id="inst-1", entity_type="nrs_instance",
            provider="ProviderOne", region="na-east",
            nrsip_address="2620:be1a::1",
            capabilities=["medical", "financial"],
            trust_score=4.0,
        ))
        block = ProviderBlock(
            provider_id="p1", provider_name="ProviderOne",
            address_prefix="2620:be1a:0001::/48", regions=["na-east"],
            registry_url="https://registry.providerone.com",
        )
        root.register_provider(block, provider_reg)
        return root, provider_reg

    def test_register_provider(self):
        root, _ = self._build_registry()
        self.assertEqual(root.provider_count(), 1)
        providers = root.list_providers()
        self.assertEqual(providers[0].provider_name, "ProviderOne")

    def test_local_cache_lookup(self):
        from nrsip.registry_hierarchy import (
            LocalRegistryCache, RegistryEntry,
        )
        cache = LocalRegistryCache()
        entry = RegistryEntry(
            entity_id="inst-cache", entity_type="nrs_instance",
            provider="CacheProvider", region="apac",
            nrsip_address="2620:be1a::100", trust_score=3.8,
        )
        cache.put("medical:apac:0.0", [entry])
        result = cache.get("medical:apac:0.0")
        self.assertIsNotNone(result)
        self.assertEqual(result[0].entity_id, "inst-cache")
        self.assertIsNone(cache.get("nonexistent"))

    def test_hierarchy_delegation(self):
        from nrsip.registry_hierarchy import FederatedResolver, LocalRegistryCache
        root, _ = self._build_registry()
        resolver = FederatedResolver(root, LocalRegistryCache())
        results = resolver.resolve(capability="medical")
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].entity_id, "inst-1")
        results2 = resolver.resolve(capability="medical")
        self.assertGreater(resolver.cache_hit_rate, 0)


# ═══════════════════════════════════════════════════════════════════════════════
#  9. NIF Governance
# ═══════════════════════════════════════════════════════════════════════════════

class TestNIFGovernance(unittest.TestCase):
    def test_protocol_version(self):
        from nrsip.nif_governance import NIFGovernance
        gov = NIFGovernance()
        ver = gov.current_version
        self.assertEqual(ver.version_string, "1.0.0")

    def test_provider_certification(self):
        from nrsip.nif_governance import NIFGovernance, CertificationLevel
        gov = NIFGovernance()
        cert = gov.certify_provider(
            provider_id="p-test",
            provider_name="TestProvider",
            level=CertificationLevel.STANDARD,
            test_results={"interop": True, "perf": True},
            auditor="nif-auditor",
        )
        self.assertTrue(gov.is_provider_certified("p-test"))
        self.assertEqual(cert.level, CertificationLevel.STANDARD)

    def test_address_allocation(self):
        from nrsip.nif_governance import NIFGovernance
        gov = NIFGovernance()
        alloc = gov.allocate_address_block(
            provider_id="p-alloc",
            provider_name="AllocProvider",
            prefix="2620:be1a:0001:0002::/64",
            region="na-east",
        )
        self.assertTrue(alloc.active)
        self.assertEqual(alloc.provider_id, "p-alloc")


# ═══════════════════════════════════════════════════════════════════════════════
#  10. Hierarchy Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestHierarchyPipeline(unittest.TestCase):
    def _build_pipeline(self):
        from nrsip.hierarchy import HierarchicalPipeline, Layer
        pipeline = HierarchicalPipeline(name="test-pipeline")
        pipeline.add_layer(Layer("sensory", 0, processor=lambda d: d.upper()))
        pipeline.add_layer(Layer("pattern", 1, processor=lambda d: f"[{d}]"))
        pipeline.add_layer(Layer(
            "reasoning", 2, processor=lambda d: d + "!",
            feedback_processor=lambda d: d.lower(),
        ))
        return pipeline

    def test_feedforward_pass(self):
        from nrsip.hierarchy import FlowDirection
        pipeline = self._build_pipeline()
        result = pipeline.process("hello", mode="single")
        self.assertTrue(result.success)
        self.assertEqual(result.final_output, "[HELLO]!")
        self.assertTrue(all(
            r.direction == FlowDirection.FEEDFORWARD
            for r in result.flow_records
        ))

    def test_cycle_mode(self):
        from nrsip.hierarchy import FlowDirection
        pipeline = self._build_pipeline()
        result = pipeline.process("hello", mode="cycle")
        self.assertTrue(result.success)
        ff_count = sum(1 for r in result.flow_records if r.direction == FlowDirection.FEEDFORWARD)
        fb_count = sum(1 for r in result.flow_records if r.direction == FlowDirection.FEEDBACK)
        self.assertGreater(ff_count, 0)
        self.assertGreater(fb_count, 0)

    def test_flow_records_created(self):
        from nrsip.hierarchy import FlowRecord
        pipeline = self._build_pipeline()
        result = pipeline.process("test", mode="single")
        self.assertGreater(len(result.flow_records), 0)
        for rec in result.flow_records:
            self.assertIsInstance(rec, FlowRecord)
            self.assertTrue(rec.success)


# ═══════════════════════════════════════════════════════════════════════════════
#  11. Generative Engine
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerativeEngine(unittest.TestCase):
    def test_synthesis_mode(self):
        from nrsip.generative_engine import GenerativeEngine, GenerationMode
        engine = GenerativeEngine()
        result = engine.generate("quantum computing", mode=GenerationMode.SYNTHESIS)
        self.assertIn("text", result)
        self.assertTrue(len(result["text"]) > 0)
        self.assertEqual(result["method"], "synthesis")

    def test_interpolation_mode(self):
        from nrsip.generative_engine import GenerativeEngine, GenerationMode
        engine = GenerativeEngine()
        result = engine.generate(
            "the cat sat on the mat", mode=GenerationMode.INTERPOLATION,
        )
        self.assertIn("text", result)
        self.assertEqual(result["method"], "interpolation")

    def test_chat(self):
        from nrsip.generative_engine import GenerativeEngine
        engine = GenerativeEngine()
        response = engine.chat("How does NRS work?", [])
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)


# ═══════════════════════════════════════════════════════════════════════════════
#  12. Domain Creases
# ═══════════════════════════════════════════════════════════════════════════════

class TestDomainCreases(unittest.TestCase):
    def test_create_crease(self):
        from nrsip.domain_creases import CreaseRegistry, DomainCrease
        registry = CreaseRegistry()
        crease = registry.create("medical", version="1.0")
        self.assertIsInstance(crease, DomainCrease)
        crease.add_fact("aspirin_dose", "325mg", layer=1)
        crease.lock(0.999)
        self.assertEqual(crease.total_facts, 1)

    def test_query_crease(self):
        from nrsip.domain_creases import CreaseRegistry
        registry = CreaseRegistry()
        crease = registry.create("physics")
        crease.add_fact("speed_of_light", "299792458 m/s")
        crease.add_fact("planck_constant", "6.626e-34 J·s")
        result = crease.query("speed_of_light")
        self.assertEqual(result, "299792458 m/s")
        self.assertIsNone(crease.query("nonexistent_key"))

    def test_integration_core(self):
        from nrsip.domain_creases import (
            IntegrationCore, ProcessingLobe, LobeType, DomainCrease,
        )
        core = IntegrationCore()
        ling = ProcessingLobe(LobeType.LINGUISTIC, 2.0)
        logic = ProcessingLobe(LobeType.LOGICAL, 1.5)
        crease = DomainCrease("general")
        crease.add_fact("test_key", "test_value")
        ling.attach_crease(crease)
        core.register_lobe(ling)
        core.register_lobe(logic)
        result = core.process_multi(
            "analyze this query",
            [LobeType.LINGUISTIC, LobeType.LOGICAL],
        )
        self.assertIn("lobes_activated", result)
        self.assertEqual(len(result["lobes_activated"]), 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  13. NRS Validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestNRSValidation(unittest.TestCase):
    def test_passing_validator(self):
        from nrsip.nrs_validation import ValidationGate, FunctionValidator
        gate = ValidationGate(
            name="test_gate",
            confidence_threshold=0.5,
            validators=[FunctionValidator(
                lambda data, ctx=None: (True, 0.95, "looks good"),
                name="always_pass",
            )],
        )
        result = gate.process({"value": "high quality data"})
        self.assertIsNotNone(result)

    def test_failing_validator(self):
        from nrsip.nrs_validation import ValidationGate, FunctionValidator
        from nrsip.nrs_errors import ValidationError
        gate = ValidationGate(
            name="strict_gate",
            confidence_threshold=0.95,
            validators=[FunctionValidator(
                lambda data, ctx=None: (False, 0.2, "low quality"),
                name="always_fail",
            )],
        )
        with self.assertRaises(ValidationError):
            gate.process("bad data")

    def test_gate_stats(self):
        from nrsip.nrs_validation import ValidationGate, FunctionValidator
        gate = ValidationGate(
            name="stats_gate",
            confidence_threshold=0.5,
            validators=[FunctionValidator(
                lambda data, ctx=None: (True, 0.99, "ok"),
                name="ok_validator",
            )],
        )
        gate.process("data1")
        gate.process("data2")
        stats = gate.stats
        self.assertEqual(stats["total_processed"], 2)
        self.assertEqual(stats["passed"], 2)
        self.assertEqual(stats["failed"], 0)


# ═══════════════════════════════════════════════════════════════════════════════
#  14. NRS Knowledge
# ═══════════════════════════════════════════════════════════════════════════════

class TestNRSKnowledge(unittest.TestCase):
    def test_register_fact(self):
        from nrsip.nrs_knowledge import KnowledgeBase, KnowledgePattern
        kb = KnowledgeBase("test")
        pat = KnowledgePattern("physics", version="1.0.0", immutable=True)
        pat.register("c", 299_792_458, confidence=1.0, source="NIST", unit="m/s")
        kb.add(pat)
        self.assertEqual(kb.query("physics", "c"), 299_792_458)

    def test_immutable_mutation_blocked(self):
        from nrsip.nrs_knowledge import KnowledgePattern
        from nrsip.nrs_errors import KnowledgeMutationError
        pat = KnowledgePattern("math", version="1.0.0", immutable=True)
        pat.register("pi", 3.14159, confidence=1.0, source="textbook")
        with self.assertRaises(KnowledgeMutationError):
            pat.register("pi", 3.14, confidence=0.9, source="other")

    def test_evolve_version(self):
        from nrsip.nrs_knowledge import KnowledgePattern
        v1 = KnowledgePattern("chem", version="1.0.0", immutable=True)
        v1.register("h2o", "water", confidence=1.0, source="textbook")
        v1.register("nacl", "salt", confidence=1.0, source="textbook")
        v2 = v1.evolve("2.0.0")
        self.assertTrue(v2.has("h2o"))
        self.assertTrue(v2.has("nacl"))
        v2.register("co2", "carbon dioxide", confidence=0.99, source="lab")
        self.assertTrue(v2.has("co2"))


# ═══════════════════════════════════════════════════════════════════════════════
#  15. NRS Plasticity
# ═══════════════════════════════════════════════════════════════════════════════

class TestNRSPlasticity(unittest.TestCase):
    def test_threshold_adaptation(self):
        from nrsip.nrs_plasticity import PlasticityManager
        pm = PlasticityManager()
        pm.add_threshold("test_t", initial=0.5, target_pass_rate=0.8,
                         learning_rate=0.05, window_size=20)
        for i in range(25):
            passed = i % 2 == 0
            pm.record_outcome("test_t", passed=passed, value=0.6 if passed else 0.3)
        threshold = pm.get_threshold("test_t")
        self.assertNotEqual(threshold.value, 0.5)

    def test_weight_reinforcement(self):
        from nrsip.nrs_plasticity import PlasticityManager
        pm = PlasticityManager()
        pm.add_weight("lobe_linguistic", initial=1.0)
        original = pm.get_weight("lobe_linguistic").value
        pm.reinforce_weight("lobe_linguistic", success=True, magnitude=1.0)
        self.assertGreater(pm.get_weight("lobe_linguistic").value, original)
        pm.reinforce_weight("lobe_linguistic", success=False, magnitude=1.0)

    def test_decay(self):
        from nrsip.nrs_plasticity import PlasticityManager
        pm = PlasticityManager()
        pm.add_weight("w1", initial=2.0, decay_rate=0.1)
        pm.add_weight("w2", initial=3.0, decay_rate=0.1)
        events = pm.decay_all_weights()
        self.assertGreaterEqual(len(events), 2)
        self.assertLess(pm.get_weight("w1").value, 2.0)
        self.assertLess(pm.get_weight("w2").value, 3.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  16. Pipeline Integration (NRSProcessingPipeline)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineIntegration(unittest.TestCase):
    def _make_pipeline(self):
        from nrsip.nrs_core import NRSProcessingPipeline
        return NRSProcessingPipeline(total_neurons=1000, active_k=10)

    def test_killswitch_blocks(self):
        from nrsip.emergency import EmergencyLevel
        pipe = self._make_pipeline()
        pipe.killswitch.activate(
            reason="test halt",
            level=EmergencyLevel.SHUTDOWN,
            source_node="unit_test",
        )
        with self.assertRaises(RuntimeError):
            pipe.process("any query")

    def test_process_basic(self):
        pipe = self._make_pipeline()
        pipe._web_engine = None
        result = pipe.process("What is photosynthesis?", domain="general")
        self.assertTrue(len(result.answer) > 0)
        self.assertIn(result.tier, ("T1", "T2", "T3", "T4"))
        self.assertGreater(result.neurons_fired, 0)

    def test_process_media(self):
        pipe = self._make_pipeline()
        pipe._web_engine = None
        from nrsip.media_pipeline import Modality
        result = pipe.process_media(b"fake image bytes", modality=Modality.IMAGE)
        self.assertIn("media", result.provenance)
        self.assertEqual(result.provenance["media"]["modality"], Modality.IMAGE)

    def test_plasticity_tracks(self):
        pipe = self._make_pipeline()
        pipe.process("Explain quantum entanglement")
        events = pipe.plasticity.events
        self.assertGreater(len(events), 0)


if __name__ == "__main__":
    unittest.main()
