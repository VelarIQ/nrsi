"""Tests for the NRS orchestrator layer -- tracing, circuit breakers, resilience."""

import asyncio
import os
import pytest
import time

from nrsip.orchestrator import (
    Orchestrator, Modality, RouteResult, RouteStatus,
    RequestTrace, LatencyBudget, CircuitBreaker, CircuitState,
    RetryPolicy, SSRFGuard, ReputationTracker,
)


class TestRequestTrace:
    def test_create_trace(self):
        trace = RequestTrace(request_id="test-001", modality="text")
        assert trace.correlation_id
        assert trace.request_id == "test-001"
        assert len(trace.spans) == 0

    def test_span_lifecycle(self):
        trace = RequestTrace()
        span = trace.new_span("test_op", "test_svc")
        assert span.operation == "test_op"
        assert span.service == "test_svc"
        assert span.start_ms > 0
        trace.close_span(span, "ok")
        assert span.end_ms >= span.start_ms
        assert span.duration_ms >= 0

    def test_evidence_chain(self):
        trace = RequestTrace()
        trace.add_evidence("web", "retrieval", {"url": "https://example.com"})
        assert len(trace.evidence_chain) == 1
        assert trace.evidence_chain[0]["source"] == "web"

    def test_to_dict(self):
        trace = RequestTrace(request_id="test", modality="text")
        span = trace.new_span("op1", "svc1")
        trace.close_span(span)
        d = trace.to_dict()
        assert "correlation_id" in d
        assert d["span_count"] == 1
        assert len(d["spans"]) == 1


class TestLatencyBudget:
    def test_initial_budget(self):
        budget = LatencyBudget(total_ms=10000.0)
        assert budget.remaining_ms <= 10000.0
        assert not budget.exhausted

    def test_allocate(self):
        budget = LatencyBudget(total_ms=10000.0)
        allocation = budget.allocate("test", 0.5)
        assert allocation > 0
        assert allocation <= 5000.0

    def test_record(self):
        budget = LatencyBudget(total_ms=10000.0)
        budget.record("test", 100.0)
        assert "test_actual" in budget.stages


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(name="test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request()

    def test_opens_on_threshold(self):
        cb = CircuitBreaker(name="test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.allow_request()

    def test_success_resets_half_open(self):
        cb = CircuitBreaker(name="test", failure_threshold=2, recovery_window_s=0.01)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED


class TestRetryPolicy:
    @pytest.mark.asyncio
    async def test_succeeds_first_try(self):
        policy = RetryPolicy(max_retries=3)
        async def success(): return "ok"
        result = await policy.execute(success)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        policy = RetryPolicy(max_retries=2, base_delay_s=0.01)
        attempts = [0]
        async def flaky():
            attempts[0] += 1
            if attempts[0] < 3:
                raise ValueError("fail")
            return "ok"
        result = await policy.execute(flaky)
        assert result == "ok"
        assert attempts[0] == 3

    @pytest.mark.asyncio
    async def test_exhausts_retries(self):
        policy = RetryPolicy(max_retries=1, base_delay_s=0.01)
        async def always_fail():
            raise ValueError("always")
        with pytest.raises(ValueError):
            await policy.execute(always_fail)


class TestSSRFGuard:
    def test_blocks_localhost(self):
        assert not SSRFGuard.is_safe_url("http://localhost/admin")
        assert not SSRFGuard.is_safe_url("http://127.0.0.1/admin")

    def test_blocks_private_ranges(self):
        assert not SSRFGuard.is_safe_url("http://10.0.0.1/api")
        assert not SSRFGuard.is_safe_url("http://192.168.1.1/api")
        assert not SSRFGuard.is_safe_url("http://172.16.0.1/api")

    def test_blocks_metadata(self):
        assert not SSRFGuard.is_safe_url("http://169.254.169.254/latest/meta-data/")
        assert not SSRFGuard.is_safe_url("http://metadata.google/computeMetadata")

    def test_allows_public(self):
        assert SSRFGuard.is_safe_url("https://example.com/api")
        assert SSRFGuard.is_safe_url("https://google.com")


class TestReputationTracker:
    def test_initial_trust(self):
        tracker = ReputationTracker()
        assert tracker.is_trusted("new-domain.com")

    def test_track_success(self):
        tracker = ReputationTracker()
        tracker.record_success("good.com")
        tracker.record_success("good.com")
        assert tracker.is_trusted("good.com")

    def test_track_failures(self):
        tracker = ReputationTracker()
        for _ in range(10):
            tracker.record_failure("bad.com")
        assert not tracker.is_trusted("bad.com")

    def test_block_domain(self):
        tracker = ReputationTracker()
        tracker.block("blocked.com")
        assert not tracker.is_trusted("blocked.com")


class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_route_with_handler(self):
        orch = Orchestrator()
        async def text_handler(payload, trace, budget):
            return {"answer": "hello", "confidence": 0.9}
        orch.register_route(Modality.TEXT, text_handler)
        result = await orch.route(Modality.TEXT, {"query": "hi"}, source="test")
        assert result.answer == "hello"
        assert result.confidence == 0.9
        assert result.trace is not None

    @pytest.mark.asyncio
    async def test_route_no_handler(self):
        orch = Orchestrator()
        result = await orch.route(Modality.VIDEO, {"query": "hi"})
        assert result.status == RouteStatus.ERROR
        assert "No handler" in result.error

    @pytest.mark.asyncio
    async def test_circuit_open_blocks(self):
        orch = Orchestrator()
        cb = orch.circuits["text_model"]
        for _ in range(cb.failure_threshold):
            cb.record_failure()
        async def handler(payload, trace, budget):
            return {"answer": "ok"}
        orch.register_route(Modality.TEXT, handler)
        result = await orch.route(Modality.TEXT, {"query": "hi"})
        assert result.status == RouteStatus.CIRCUIT_OPEN

    def test_stats(self):
        orch = Orchestrator()
        stats = orch.stats
        assert "total_requests" in stats
        assert "circuits" in stats
        assert "reputation" in stats


class TestOrchestrationIntegration:
    @pytest.mark.asyncio
    async def test_full_trace_lifecycle(self):
        orch = Orchestrator()
        async def handler(payload, trace, budget):
            span = trace.new_span("model_call", "text-model")
            await asyncio.sleep(0.01)
            trace.close_span(span, "ok")
            trace.add_evidence("model", "generation", {"tokens": 50})
            return RouteResult(
                modality=Modality.TEXT, answer="response",
                confidence=0.85, status=RouteStatus.SUCCESS,
            )
        orch.register_route(Modality.TEXT, handler)
        result = await orch.route(Modality.TEXT, {"query": "test"})
        assert result.status == RouteStatus.SUCCESS
        assert result.trace is not None
        trace_dict = result.trace.to_dict()
        assert trace_dict["span_count"] >= 2
        assert len(result.trace.evidence_chain) == 1
