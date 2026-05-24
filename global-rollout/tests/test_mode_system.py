"""Tests for the AGI Cognitive Mode Control System."""

import pytest
from nrsi.core.mode_control import (
    ModeVector, ModeSpectrum, CognitiveModeController, ModeDecision,
    NeuronStrategy, ToneDirective, AdversarialModeReviewer,
    UserModeProfileStore, DomainModeHistory, ModeShiftEvent,
    HIGH_RISK_DOMAINS,
)


class TestModeVector:
    def test_factual_primary_mode(self):
        v = ModeVector(analytical=0.8, factual=0.9, creative=0.1)
        assert v.primary_mode() == ModeSpectrum.DETERMINISTIC

    def test_creative_primary_mode(self):
        v = ModeVector(creative=0.9, exploratory=0.6)
        assert v.primary_mode() == ModeSpectrum.CREATIVE

    def test_hybrid_when_close(self):
        v = ModeVector(analytical=0.5, factual=0.5, creative=0.5)
        assert v.primary_mode() == ModeSpectrum.HYBRID

    def test_legacy_mode(self):
        assert ModeVector(factual=0.9).legacy_mode == "DETERMINISTIC"
        assert ModeVector(creative=0.9).legacy_mode == "PROBABILISTIC"

    def test_blend(self):
        a = ModeVector(factual=1.0, creative=0.0, analytical=0.0)
        b = ModeVector(factual=0.0, creative=1.0, analytical=0.0)
        blended = a.blend(b, 0.5)
        assert 0.4 < blended.factual < 0.6
        assert 0.4 < blended.creative < 0.6

    def test_normalize(self):
        v = ModeVector(analytical=2.0, factual=3.0)
        n = v.normalize()
        total = sum(getattr(n, f) for f in ModeVector._fields())
        assert abs(total - 1.0) < 0.01

    def test_distance(self):
        a = ModeVector(factual=1.0)
        b = ModeVector(factual=1.0)
        assert a.distance(b) < 0.001
        c = ModeVector(creative=1.0)
        assert a.distance(c) > 0.5

    def test_to_from_dict(self):
        v = ModeVector(analytical=0.3, creative=0.7, factual=0.5)
        d = v.to_dict()
        v2 = ModeVector.from_dict(d)
        assert abs(v.analytical - v2.analytical) < 0.001
        assert abs(v.creative - v2.creative) < 0.001

    def test_from_query_type(self):
        v = ModeVector.from_query_type("creative")
        assert v.creative > 0.5
        v2 = ModeVector.from_query_type("factual")
        assert v2.factual > 0.5


class TestCognitiveModeController:
    def setup_method(self):
        self.ctrl = CognitiveModeController()

    def test_factual_classification(self):
        dec = self.ctrl.classify("What is the capital of France?", tier="T1")
        assert dec.tvs_mode in ("DETERMINISTIC", "HYBRID")
        assert dec.vector.factual > 0.5

    def test_creative_override(self):
        dec = self.ctrl.classify("Write a poem", mode_override="CREATIVE")
        assert dec.legacy_mode == "PROBABILISTIC"
        assert dec.neuron_strategy.use_stochastic
        assert dec.vector.creative > 0.5

    def test_medical_domain_constraint(self):
        dec = self.ctrl.classify("Creative ideas", domain="medical", tier="T2")
        assert dec.vector.creative <= 0.31
        assert dec.vector.factual >= 0.69
        assert dec.domain_constraint != ""

    def test_complexity_adjustment_t4(self):
        dec = self.ctrl.classify("Analyze everything", domain="general", tier="T4")
        assert dec.vector.metacognitive > 0

    def test_adversarial_check_flag(self):
        dec = self.ctrl.classify("Tell a story", mode_override="CREATIVE", tier="T3")
        if dec.vector.creative > 0.5 and dec.vector.factual > 0.3:
            assert dec.do_adversarial_check

    def test_tone_directive_warm(self):
        dec = self.ctrl.classify("I feel sad", tier="T1")
        if dec.vector.empathetic > 0.5:
            assert dec.tone_directive.tone == "warm"

    def test_record_outcome(self):
        v = ModeVector(factual=0.9)
        self.ctrl.record_outcome(v, 0.95, "general", "factual", "u1")
        assert self.ctrl.stats["history_size"] == 1

    def test_stats(self):
        self.ctrl.classify("test", tier="T1")
        s = self.ctrl.stats
        assert s["classifications"] == 1


class TestStochasticNeurons:
    def test_deterministic_reproduces(self):
        from nrsi.core.neurons import BinaryNeuronBank
        bank = BinaryNeuronBank(total_neurons=10000, active_k=100)
        d1 = bank.activate_and_compress("test")
        d2 = bank.activate_and_compress("test")
        assert d1.signature == d2.signature

    def test_stochastic_varies(self):
        from nrsi.core.neurons import BinaryNeuronBank
        bank = BinaryNeuronBank(total_neurons=10000, active_k=100)
        s1 = bank.activate_stochastic("test", temperature=1.5, noise_sigma=0.3)
        s2 = bank.activate_stochastic("test", temperature=1.5, noise_sigma=0.3)
        assert s1.signature != s2.signature

    def test_activate_for_mode_deterministic(self):
        from nrsi.core.neurons import BinaryNeuronBank
        bank = BinaryNeuronBank(total_neurons=10000, active_k=100)
        strat = NeuronStrategy(use_stochastic=False)
        d = bank.activate_for_mode("test", strat)
        assert d.signature  # non-empty

    def test_activate_for_mode_stochastic(self):
        from nrsi.core.neurons import BinaryNeuronBank
        bank = BinaryNeuronBank(total_neurons=10000, active_k=100)
        strat = NeuronStrategy(use_stochastic=True, temperature=1.0, noise_sigma=0.2)
        d = bank.activate_for_mode("test", strat)
        assert d.signature


class TestCreativeLobe:
    def test_creative_lobe_with_attached(self):
        from nrsi.core.lobes import CreativeProcessingLobe
        from nrsi.core.memory import CreativeLobe as CreativeLearning
        lobe = CreativeProcessingLobe()
        cl = CreativeLearning()
        lobe.attach_creative_lobe(cl)
        result = lobe.process("Imagine a new color", domain="fiction")
        assert result.value is not None
        assert result.metadata.get("source") == "creative_lobe"

    def test_creative_lobe_without_attached(self):
        from nrsi.core.lobes import CreativeProcessingLobe
        lobe = CreativeProcessingLobe()
        result = lobe.process("Test query")
        assert result.lobe.value == "creative"

    def test_weighted_processing(self):
        from nrsi.core.lobes import IntegrationCore, LinguisticLobe, CreativeProcessingLobe
        core = IntegrationCore()
        core.register_lobe(LinguisticLobe())
        core.register_lobe(CreativeProcessingLobe())
        weights = [("linguistic", 0.6), ("creative", 0.7)]
        result = core.process_weighted("Create a metaphor", weights)
        assert "lobes_activated" in result
        assert result.get("creative_first") is True

    def test_weighted_no_creative_first(self):
        from nrsi.core.lobes import IntegrationCore, LinguisticLobe, LogicalLobe
        core = IntegrationCore()
        core.register_lobe(LinguisticLobe())
        core.register_lobe(LogicalLobe())
        weights = [("linguistic", 0.8), ("logical", 0.5)]
        result = core.process_weighted("Analyze this", weights)
        assert result.get("creative_first") is False


class TestAdversarialReview:
    def setup_method(self):
        self.reviewer = AdversarialModeReviewer()

    def test_no_claims(self):
        result = self.reviewer.review("once upon a time")
        assert result.recommendation == "pass"
        assert result.overall_factual_score == 1.0

    def test_with_entities(self):
        result = self.reviewer.review("Albert Einstein discovered gravity in 1687 near London.")
        assert len(result.flagged_claims) >= 0
        assert result.recommendation in ("pass", "warn", "block")

    def test_extract_factual(self):
        claims = self.reviewer._extract_factual_from_creative(
            "Paris is located 300 km from Berlin. January 2024 was cold."
        )
        assert len(claims) > 0


class TestUserModeProfileStore:
    def setup_method(self):
        self.store = UserModeProfileStore()

    def test_round_trip(self):
        self.store.update_profile("u1", ModeVector(creative=0.8), 0.9, "general")
        prof = self.store.get_profile("u1")
        assert prof is not None
        assert prof.interaction_count == 1
        assert prof.preferred_vector.creative > 0

    def test_ema_update(self):
        self.store.update_profile("u2", ModeVector(creative=0.8), 0.9)
        self.store.update_profile("u2", ModeVector(creative=0.2), 0.9)
        prof = self.store.get_profile("u2")
        assert prof.interaction_count == 2
        assert prof.preferred_vector.creative < 0.8

    def test_clear(self):
        self.store.update_profile("u3", ModeVector(factual=0.9), 0.9)
        self.store.clear_profile("u3")
        assert self.store.get_profile("u3") is None

    def test_empty_user_id(self):
        assert self.store.get_profile("") is None


class TestDomainModeHistory:
    def setup_method(self):
        self.dh = DomainModeHistory()

    def test_record_and_optimal(self):
        for i in range(5):
            self.dh.record("sci", "factual", ModeVector(factual=0.8 + i * 0.02), 0.9 + i * 0.01)
        opt = self.dh.optimal_vector("sci", "factual")
        assert opt is not None
        assert opt.factual > 0.7

    def test_no_history(self):
        assert self.dh.optimal_vector("unknown", "unknown") is None

    def test_stats(self):
        self.dh.record("med", "factual", ModeVector(factual=0.9), 0.95)
        stats = self.dh.get_stats("med")
        assert "factual" in stats["query_types"]


class TestNRSModeIntegration:
    def test_mode_fields_populated(self):
        from nrsi.core.nrs import NRS
        nrs = NRS(instance_id="test-mode", total_neurons=1000, active_k=10)
        resp = nrs.process("What is water?")
        assert resp.mode in ("DETERMINISTIC", "HYBRID", "PROBABILISTIC")
        assert resp.mode_vector
        assert isinstance(resp.mode_vector, dict)
        assert "analytical" in resp.mode_vector

    def test_creative_override(self):
        from nrsi.core.nrs import NRS
        nrs = NRS(instance_id="test-creative", total_neurons=1000, active_k=10)
        resp = nrs.process("Write a poem", mode_override="CREATIVE")
        assert resp.mode == "PROBABILISTIC"
        assert not resp.deterministic
        assert resp.tone_applied == "expressive"

    def test_medical_clamping(self):
        from nrsi.core.nrs import NRS
        nrs = NRS(instance_id="test-med", total_neurons=1000, active_k=10)
        resp = nrs.process("Creative story", domain="medical", mode_override="CREATIVE")
        assert resp.mode_vector.get("creative", 0) <= 0.31
        assert resp.mode_vector.get("factual", 0) >= 0.69

    def test_backward_compat_mode_string(self):
        from nrsi.core.nrs import NRS
        nrs = NRS(instance_id="test-compat", total_neurons=1000, active_k=10)
        resp = nrs.process("Hello")
        assert resp.mode in ("DETERMINISTIC", "HYBRID", "PROBABILISTIC")
        status = resp.status.value if hasattr(resp.status, "value") else resp.status
        assert status in ("validated", "acceptable", "cached", "blocked", "error")

    def test_mode_controller_in_stats(self):
        from nrsi.core.nrs import NRS
        nrs = NRS(instance_id="test-stats", total_neurons=1000, active_k=10)
        nrs.process("Test")
        st = nrs.stats() if callable(nrs.stats) else nrs.stats
        assert "mode_controller" in st
        assert st["mode_controller"]["classifications"] >= 1
