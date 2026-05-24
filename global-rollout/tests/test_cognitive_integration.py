"""Test suite for the unified NRS brain system.

Covers:
  - CognitiveIntegrationLayer initialization
  - VLT <-> Graph sync
  - PVS <-> Semantic sync
  - Tuition <-> Reasoning adapter
  - Lobe cognitive wiring
  - Neuron <-> Bayesian adapter
  - Crease <-> Graph adapter
  - Plasticity <-> Maintenance adapter
  - NervousSystem <-> SocialCognition adapter
  - Generative <-> Grounded adapter
  - PoVI <-> Provenance adapter
  - NIF Governance adapter
  - Persistent store
  - Full unified pipeline end-to-end
  - think() cognitive response
  - think_stream() streaming
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from nrsip.cognitive_integration import (
    CognitiveIntegrationLayer,
    VLTGraphAdapter,
    PVSSemanticAdapter,
    TuitionReasoningAdapter,
    NeuronConfidenceAdapter,
    CreaseGraphAdapter,
    PlasticityMaintenanceAdapter,
    NervousSystemCognitiveAdapter,
    GenerativeGroundedAdapter,
    PoVIProvenanceAdapter,
    NIFGovernanceAdapter,
    CognitiveAugmentation,
)
from nrsip.nrs_core import (
    NRSProcessingPipeline,
    NRSProcessingResult,
    LOBE_INSTANCES,
    VLT,
    VLTLayer,
    PVS4,
    TuitionSystem,
)
from nrsip.reasoning_bridge import ReasoningBridge, CognitiveResponse
from nrsip.reasoning_graph import CausalGraph, RelationType
from nrsip.persistent_store import KnowledgeStore


_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = NRSProcessingPipeline(total_neurons=1000, active_k=10)
    return _pipeline


class TestCognitiveIntegrationInit:
    def test_pipeline_has_cognitive(self):
        p = _get_pipeline()
        assert p.cognitive is not None, "cognitive layer should be initialized"

    def test_cognitive_initialized(self):
        p = _get_pipeline()
        assert p.cognitive._initialized is True

    def test_all_adapters_active(self):
        p = _get_pipeline()
        stats = p.cognitive.stats
        assert stats["adapters_active"] >= 9, f"Expected >= 9 adapters, got {stats['adapters_active']}"

    def test_bridge_has_graph(self):
        p = _get_pipeline()
        graph = p.cognitive.bridge.graph
        assert graph.node_count > 0, "Graph should have nodes after init"
        assert graph.edge_count > 0, "Graph should have edges after init"

    def test_graph_enriched_from_vlt(self):
        p = _get_pipeline()
        # Threshold is cap-aware: when CI sets NRS_TEST_FACT_CAP_PER_PACK
        # (currently 50/pack -> ~2500 facts across ~50 packs), full
        # ingestion is suppressed by design. The assertion just has to
        # prove enrichment happened, not that we hit the prod node count.
        cap = os.environ.get("NRS_TEST_FACT_CAP_PER_PACK")
        if cap:
            try:
                expected_min = max(50, int(cap) * 10)
            except ValueError:
                expected_min = 500
        else:
            expected_min = 4000
        assert p.cognitive.bridge.graph.node_count > expected_min, (
            f"VLT should have added nodes (got {p.cognitive.bridge.graph.node_count}, "
            f"expected > {expected_min}, cap={cap or 'unset'})"
        )


class TestVLTGraphSync:
    def test_vlt_adapter_created(self):
        p = _get_pipeline()
        assert p.cognitive._vlt_adapter is not None

    def test_sync_vlt_to_graph(self):
        p = _get_pipeline()
        stats = p.cognitive._vlt_adapter.sync_vlt_to_graph(p.vlt)
        assert "scanned" in stats
        assert "imported" in stats

    def test_sync_graph_to_vlt(self):
        p = _get_pipeline()
        exported = p.cognitive._vlt_adapter.sync_graph_to_vlt(p.vlt, max_edges=10)
        assert isinstance(exported, int)


class TestPVSSemanticSync:
    def test_pvs_adapter_created(self):
        p = _get_pipeline()
        assert p.cognitive._pvs_adapter is not None

    def test_pvs_lookup(self):
        p = _get_pipeline()
        result = p.cognitive._pvs_adapter.lookup("climate change effects", p.vlt.pvs)
        assert "semantic_concepts" in result

    def test_enrich_pvs_with_graph(self):
        p = _get_pipeline()
        count = p.cognitive._pvs_adapter.enrich_pvs_with_graph(
            p.vlt.pvs, p.cognitive.bridge.graph, max_entries=5)
        assert isinstance(count, int)


class TestTuitionAdapter:
    def test_tuition_adapter_created(self):
        p = _get_pipeline()
        assert p.cognitive._tuition_adapter is not None


class TestLobeWiring:
    def test_all_lobes_have_cognitive_bridge(self):
        _get_pipeline()
        for name, lobe in LOBE_INSTANCES.items():
            assert lobe._cognitive_bridge is not None, f"Lobe {name} missing cognitive bridge"

    def test_linguistic_lobe_produces_output(self):
        _get_pipeline()
        lobe = LOBE_INSTANCES["linguistic"]
        result = lobe.process("What is photosynthesis?", "science")
        assert result.value, "Linguistic lobe should produce output"
        assert result.confidence > 0

    def test_logical_lobe_produces_output(self):
        _get_pipeline()
        lobe = LOBE_INSTANCES["logical"]
        result = lobe.process("If temperature rises then ice melts", "science")
        assert result.value
        assert result.confidence > 0

    def test_mathematical_lobe_produces_output(self):
        _get_pipeline()
        lobe = LOBE_INSTANCES["mathematical"]
        result = lobe.process("Calculate 2 + 2", "math")
        assert result.value
        assert result.confidence > 0

    def test_temporal_lobe_produces_output(self):
        _get_pipeline()
        lobe = LOBE_INSTANCES["temporal"]
        result = lobe.process("When was the Roman Empire founded?", "history")
        assert result.value
        assert result.confidence > 0


class TestNeuronBayesianAdapter:
    def test_adapter_created(self):
        p = _get_pipeline()
        assert p.cognitive._neuron_adapter is not None


class TestCreaseGraphAdapter:
    def test_adapter_created(self):
        p = _get_pipeline()
        assert p.cognitive._crease_adapter is not None

    def test_creases_synced(self):
        p = _get_pipeline()
        stats = p.cognitive._crease_adapter.sync_creases_to_graph(p.crease_registry)
        assert "domains" in stats


class TestPlasticityAdapter:
    def test_adapter_created(self):
        p = _get_pipeline()
        assert p.cognitive._plasticity_adapter is not None


class TestNervousSystemAdapter:
    def test_adapter_created(self):
        p = _get_pipeline()
        assert p.cognitive._nervous_adapter is not None


class TestGenerativeAdapter:
    def test_adapter_created(self):
        p = _get_pipeline()
        assert p.cognitive._generative_adapter is not None


class TestPoVIAdapter:
    def test_adapter_created(self):
        p = _get_pipeline()
        assert p.cognitive._povi_adapter is not None


class TestNIFAdapter:
    def test_adapter_created(self):
        p = _get_pipeline()
        assert p.cognitive._nif_adapter is not None


class TestPersistentStore:
    def test_store_created(self):
        p = _get_pipeline()
        assert p.cognitive._store is not None

    def test_save_graph(self):
        p = _get_pipeline()
        result = p.cognitive._store.save_graph(p.cognitive.bridge.graph)
        assert result["nodes"] > 0
        assert result["edges"] > 0

    def test_user_profile(self):
        p = _get_pipeline()
        p.cognitive.save_user_profile("test-user", {
            "display_name": "Tester",
            "interaction_count": 1,
        })
        profile = p.cognitive.get_user_profile("test-user")
        assert profile is not None
        assert profile["display_name"] == "Tester"

    def test_conversation_history(self):
        p = _get_pipeline()
        row_id = p.cognitive.save_conversation(
            "test-user", "test-session", "user", "Hello world", ["hello"], 0.5)
        assert row_id is not None


class TestUnifiedPipeline:
    def test_process_returns_result(self):
        p = _get_pipeline()
        result = p.process("What causes climate change?", domain="science")
        assert isinstance(result, NRSProcessingResult)
        assert result.answer
        assert result.confidence > 0

    def test_cognitive_augmentation_present(self):
        p = _get_pipeline()
        result = p.process("Explain photosynthesis", domain="science")
        assert result.cognitive_augmentation is not None, "Cognitive augmentation should be in result"

    def test_cognitive_augmentation_has_concepts(self):
        p = _get_pipeline()
        result = p.process("How does gravity work?", domain="science")
        if result.cognitive_augmentation:
            assert "concepts" in result.cognitive_augmentation
            assert len(result.cognitive_augmentation["concepts"]) > 0

    def test_different_domains(self):
        p = _get_pipeline()
        domains = ["science", "history", "mathematics", "general"]
        for domain in domains:
            result = p.process(f"Tell me about {domain}", domain=domain)
            assert result.answer, f"No answer for domain {domain}"

    def test_lobes_activated(self):
        p = _get_pipeline()
        result = p.process("Calculate 2+2", domain="mathematics")
        assert len(result.lobes_activated) > 0


class TestThink:
    def test_think_returns_cognitive_response(self):
        p = _get_pipeline()
        response = p.cognitive.bridge.think("What is water?")
        assert isinstance(response, CognitiveResponse)

    def test_think_has_text(self):
        p = _get_pipeline()
        response = p.cognitive.bridge.think("What is the sun?")
        assert response.text or response.grounded

    def test_think_has_epistemic(self):
        p = _get_pipeline()
        response = p.cognitive.bridge.think("What is quantum computing?")
        assert response.epistemic is not None

    def test_think_has_confidence(self):
        p = _get_pipeline()
        response = p.cognitive.bridge.think("What is DNA?")
        assert response.confidence >= 0


class TestThinkStream:
    def test_think_stream_yields_events(self):
        import asyncio

        p = _get_pipeline()

        async def _run():
            events = []
            async for event in p.cognitive.bridge.think_stream("What is gravity?"):
                events.append(event)
            return events

        events = asyncio.run(_run())
        assert len(events) > 0, "Stream should yield events"

        phases = [e["phase"] for e in events]
        assert "semantic_parse" in phases
        assert "core_reasoning" in phases
        assert "complete" in phases

    def test_think_stream_final_has_text(self):
        import asyncio

        p = _get_pipeline()

        async def _run():
            last = None
            async for event in p.cognitive.bridge.think_stream("What is photosynthesis?"):
                last = event
            return last

        last = asyncio.run(_run())
        assert last["phase"] == "complete"
        assert "text" in last


class TestPeriodicSync:
    def test_periodic_sync(self):
        p = _get_pipeline()
        stats = p.cognitive.periodic_sync(p)
        assert "vlt_to_graph" in stats


class TestPreProcess:
    def test_pre_process_returns_augmentation(self):
        p = _get_pipeline()
        aug = p.cognitive.pre_process("What is the speed of light?", "science")
        assert isinstance(aug, CognitiveAugmentation)
        assert len(aug.concepts) > 0
        assert aug.confidence > 0

    def test_pre_process_suggests_lobes(self):
        p = _get_pipeline()
        aug = p.cognitive.pre_process("When was Rome founded?", "history")
        assert len(aug.suggested_lobes) > 0


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
