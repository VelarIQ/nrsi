"""NRSI Inference Router — single entry point for all NRS inference.

Every processing mode (text, image, audio, code, video, etc.) routes
through ``NRSInferenceRouter.process()`` which delegates to the NRSI
``NRS`` engine.  The nrsip/ layer remains transport-only; no inference
logic is imported from it here.

Usage by nrs-workers::

    from nrsi.core.inference_router import NRSInferenceRouter

    router = NRSInferenceRouter(instance_id="nrs-worker-us-central1")
    response = router.process("Explain photosynthesis", mode="text")
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("nrsi.inference_router")

try:
    import cupy as _cp
    _GPU_AVAILABLE = True
    _GPU_NAME: Optional[str] = None
    try:
        _GPU_NAME = _cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        _GPU_NAME = "cupy-device-0"
except ImportError:
    _GPU_AVAILABLE = False
    _GPU_NAME = None
    _cp = None


class NRSInferenceRouter:
    """Routes ALL inference requests through the NRSI NRS engine.

    Parameters
    ----------
    instance_id : str
        Identifier for this NRS instance (appears in logs and response
        metadata).
    total_neurons : int
        Neuron count forwarded to the NRS engine constructor.
    active_k : int
        Number of active neurons per query.
    use_gpu : bool
        If True *and* CuPy is importable, GPU acceleration is enabled
        inside the NRS engine and its subsystems.
    """

    def __init__(
        self,
        instance_id: str = "nrs-router-001",
        total_neurons: int = 100_000,
        active_k: int = 100,
        use_gpu: bool = True,
    ):
        self._instance_id = instance_id
        self._use_gpu = use_gpu and _GPU_AVAILABLE
        self._request_count = 0
        self._total_latency_ms = 0.0

        if self._use_gpu:
            os.environ.setdefault("NRS_GPU_ENABLED", "1")

        from nrsi.core.nrs import NRS
        self._engine = NRS(
            instance_id=instance_id,
            total_neurons=total_neurons,
            active_k=active_k,
        )

        logger.info(
            "NRSInferenceRouter ready: instance=%s gpu=%s gpu_device=%s neurons=%s active_k=%s",
            instance_id, self._use_gpu, _GPU_NAME or "none",
            total_neurons, active_k,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def process(
        self,
        query: str,
        mode: str = "text",
        domain: str = "",
        session_id: str = "",
        user_id: str = "",
        mode_override: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Run *query* through the NRSI NRS pipeline.

        Every mode — text, code, image description, audio transcript,
        etc. — enters here and is forwarded to ``NRS.process()``.

        Returns the ``NRSResponse`` produced by the engine.
        """
        t0 = time.time()
        self._request_count += 1
        req_id = self._request_count

        logger.info(
            "[req=%d] mode=%s domain=%s gpu=%s query_len=%d",
            req_id, mode, domain or "auto", self._use_gpu, len(query),
        )

        effective_override = mode_override or mode
        response = self._engine.process(
            query,
            domain=domain,
            session_id=session_id,
            user_id=user_id,
            mode_override=effective_override,
        )

        elapsed_ms = (time.time() - t0) * 1000
        self._total_latency_ms += elapsed_ms

        status = getattr(response, "status", "unknown")
        confidence = getattr(response, "confidence", 0.0)
        logger.info(
            "[req=%d] done in %.1fms status=%s confidence=%.3f mode=%s",
            req_id, elapsed_ms, status, confidence, mode,
        )

        return response

    # ── Introspection ─────────────────────────────────────────────────────

    @property
    def engine(self) -> Any:
        return self._engine

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def gpu_enabled(self) -> bool:
        return self._use_gpu

    @property
    def gpu_device(self) -> Optional[str]:
        return _GPU_NAME if self._use_gpu else None

    @property
    def stats(self) -> Dict[str, Any]:
        avg = (self._total_latency_ms / self._request_count) if self._request_count else 0.0
        return {
            "instance_id": self._instance_id,
            "requests": self._request_count,
            "total_latency_ms": round(self._total_latency_ms, 1),
            "avg_latency_ms": round(avg, 1),
            "gpu_enabled": self._use_gpu,
            "gpu_device": _GPU_NAME,
        }


# ═══════════════════════════════════════════════════════════════════════════
# TextInferenceLoadBalancer
# ═══════════════════════════════════════════════════════════════════════════
#
# Per the runtime contract, *all* text inference must be served by internal
# NRS text backends — never an external LLM. ``TextInferenceLoadBalancer``
# distributes those requests across a heterogeneous CPU + GPU fleet:
#
#   * Backends are typed ``cpu`` or ``gpu`` so we can prefer GPU for long
#     prompts (where attention cost dominates) and CPU for short ones (where
#     wire latency dominates).
#   * Per-backend in-flight counters are the load signal. The picker prefers
#     the lowest-loaded healthy backend within the chosen kind.
#   * A 30-second rolling failure window marks a backend UNHEALTHY when more
#     than half of recent attempts failed, so a flapping endpoint stops
#     poisoning the pool quickly.
#   * On a 5xx or timeout the request is retried *once* on a backend of the
#     OTHER kind, so a GPU outage degrades to CPU (and vice versa).
#
# Crucially, this class only talks to internal NRS text endpoints over
# httpx. There is no OpenAI / Anthropic / Gemini / Bedrock / etc. client in
# the import graph.
# ═══════════════════════════════════════════════════════════════════════════

import asyncio
import json
import threading
from collections import deque
from typing import Deque, List


class _BackendState:
    """In-process bookkeeping for a single text backend."""

    HEALTH_WINDOW_SEC: float = 30.0
    UNHEALTHY_FAILURE_RATIO: float = 0.5
    MIN_SAMPLES_FOR_HEALTH: int = 4

    __slots__ = (
        "id",
        "kind",
        "endpoint_url",
        "max_concurrency",
        "inflight",
        "total_requests",
        "total_failures",
        "_events",
        "_latencies",
        "_lock",
    )

    def __init__(
        self,
        backend_id: str,
        kind: str,
        endpoint_url: str,
        max_concurrency: int,
    ):
        self.id = backend_id
        self.kind = kind
        self.endpoint_url = endpoint_url.rstrip("/")
        self.max_concurrency = max(1, int(max_concurrency))
        self.inflight: int = 0
        self.total_requests: int = 0
        self.total_failures: int = 0
        # Each event: (timestamp, success_bool)
        self._events: Deque[tuple[float, bool]] = deque()
        # Successful-only latencies in ms, capped to last 256 for p50/p95.
        self._latencies: Deque[float] = deque(maxlen=256)
        self._lock = threading.Lock()

    # ── health ────────────────────────────────────────────────────────────
    def _trim(self, now: float) -> None:
        cutoff = now - self.HEALTH_WINDOW_SEC
        evs = self._events
        while evs and evs[0][0] < cutoff:
            evs.popleft()

    def is_healthy(self, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        with self._lock:
            self._trim(now)
            samples = len(self._events)
            if samples < self.MIN_SAMPLES_FOR_HEALTH:
                return True  # Insufficient evidence — treat as healthy.
            failures = sum(1 for _, ok in self._events if not ok)
            return (failures / samples) <= self.UNHEALTHY_FAILURE_RATIO

    # ── observation ───────────────────────────────────────────────────────
    def record(self, success: bool, latency_ms: float, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        with self._lock:
            self.total_requests += 1
            if not success:
                self.total_failures += 1
            elif latency_ms >= 0:
                self._latencies.append(float(latency_ms))
            self._events.append((now, success))
            self._trim(now)

    # ── snapshot ──────────────────────────────────────────────────────────
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            lats = sorted(self._latencies)
            n = len(lats)
            def _pct(p: float) -> float:
                if n == 0:
                    return 0.0
                idx = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
                return round(lats[idx], 2)
            return {
                "id": self.id,
                "kind": self.kind,
                "endpoint_url": self.endpoint_url,
                "max_concurrency": self.max_concurrency,
                "inflight": self.inflight,
                "total_requests": self.total_requests,
                "total_failures": self.total_failures,
                "p50_ms": _pct(50.0),
                "p95_ms": _pct(95.0),
                "healthy": True,  # filled in by snapshot()
            }


class LoadBalancerError(RuntimeError):
    """Raised when no healthy backend can serve a request."""


class TextInferenceLoadBalancer:
    """CPU↔GPU load balancer for NRS text inference.

    The constructor accepts an explicit list of backends; if empty it
    falls back to ``NRS_TEXT_BACKENDS`` (a JSON list) and finally to the
    convenience env vars ``NRS_TEXT_CPU_URL`` / ``NRS_TEXT_GPU_URL``.

    The balancer is intentionally self-contained: it imports nothing from
    third-party LLM SDKs. The transport is plain ``httpx`` against the
    internal NRS text endpoints. The endpoint contract is a POST that
    accepts ``{"prompt": str, "params": dict}`` and returns a JSON body —
    callers can wrap their own request shape on top by encoding it inside
    ``params``.
    """

    # Token thresholds for kind preference. Token count is approximated
    # from prompt+context character length divided by 4 (typical English).
    GPU_PREFERRED_MIN_TOKENS: int = 1024
    CPU_PREFERRED_MAX_TOKENS: int = 256
    DEFAULT_TIMEOUT_SEC: float = 60.0

    def __init__(
        self,
        backends: Optional[List[Dict[str, Any]]] = None,
        *,
        timeout_sec: Optional[float] = None,
        path: str = "/v1/text/infer",
    ):
        self._lock = threading.Lock()
        self._rr_index: Dict[str, int] = {"cpu": 0, "gpu": 0}
        self._timeout_sec = float(timeout_sec or self.DEFAULT_TIMEOUT_SEC)
        self._path = path if path.startswith("/") else f"/{path}"

        cfg = backends if backends is not None else self._load_from_env()
        self._backends: List[_BackendState] = []
        for entry in cfg:
            try:
                kind = str(entry["kind"]).lower().strip()
                if kind not in ("cpu", "gpu"):
                    logger.warning(
                        "TextInferenceLoadBalancer: skipping backend with unknown kind=%r",
                        entry.get("kind"),
                    )
                    continue
                self._backends.append(
                    _BackendState(
                        backend_id=str(entry.get("id") or f"{kind}-{len(self._backends)}"),
                        kind=kind,
                        endpoint_url=str(entry["endpoint_url"]),
                        max_concurrency=int(entry.get("max_concurrency", 4)),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "TextInferenceLoadBalancer: malformed backend %r: %s",
                    entry, exc,
                )

        logger.info(
            "TextInferenceLoadBalancer ready: cpu=%d gpu=%d (path=%s timeout=%.1fs)",
            sum(1 for b in self._backends if b.kind == "cpu"),
            sum(1 for b in self._backends if b.kind == "gpu"),
            self._path,
            self._timeout_sec,
        )

    # ── env loader ────────────────────────────────────────────────────────
    @staticmethod
    def _load_from_env() -> List[Dict[str, Any]]:
        raw = os.environ.get("NRS_TEXT_BACKENDS", "").strip()
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return parsed
                logger.warning(
                    "NRS_TEXT_BACKENDS must be a JSON list, got %s — ignoring",
                    type(parsed).__name__,
                )
            except json.JSONDecodeError as exc:
                logger.warning("NRS_TEXT_BACKENDS unparseable: %s", exc)
        out: List[Dict[str, Any]] = []
        cpu_url = os.environ.get("NRS_TEXT_CPU_URL", "").strip()
        gpu_url = os.environ.get("NRS_TEXT_GPU_URL", "").strip()
        if cpu_url:
            out.append({"id": "cpu-default", "kind": "cpu", "endpoint_url": cpu_url, "max_concurrency": 8})
        if gpu_url:
            out.append({"id": "gpu-default", "kind": "gpu", "endpoint_url": gpu_url, "max_concurrency": 4})
        return out

    # ── selection ─────────────────────────────────────────────────────────
    @staticmethod
    def _approx_tokens(prompt: str, params: Dict[str, Any]) -> int:
        """Rough token estimate. Uses an explicit param if provided so callers
        with a real tokenizer can override the heuristic."""
        explicit = params.get("estimated_tokens") if isinstance(params, dict) else None
        if isinstance(explicit, int) and explicit > 0:
            return explicit
        ctx_len = 0
        if isinstance(params, dict):
            ctx = params.get("context") or params.get("history") or ""
            if isinstance(ctx, str):
                ctx_len = len(ctx)
            elif isinstance(ctx, list):
                ctx_len = sum(len(str(x)) for x in ctx)
        return max(1, (len(prompt or "") + ctx_len) // 4)

    def _healthy(self, kind: str) -> List[_BackendState]:
        return [b for b in self._backends if b.kind == kind and b.is_healthy()]

    def pick_backend(self, prompt: str, params: Dict[str, Any]) -> _BackendState:
        """Select the best backend for ``prompt``+``params``.

        The picker is the public-named entry for callers that want to
        own the actual transport (e.g. SSE-proxy code in platform-api).
        For most callers the high-level :meth:`submit` is preferred.
        """
        tokens = self._approx_tokens(prompt, params)
        if tokens > self.GPU_PREFERRED_MIN_TOKENS:
            preferred, fallback = "gpu", "cpu"
        elif tokens < self.CPU_PREFERRED_MAX_TOKENS:
            preferred, fallback = "cpu", "gpu"
        else:
            preferred, fallback = self._round_robin_kind(), None  # type: ignore[assignment]

        with self._lock:
            for kind in (preferred, fallback):
                if not kind:
                    continue
                healthy = self._healthy(kind)
                if not healthy:
                    continue
                # Prefer lowest-inflight; tie-break by round-robin across the
                # tied set so we don't pin one backend forever.
                healthy.sort(key=lambda b: (b.inflight, self._rr_tiebreak(kind, b)))
                chosen = healthy[0]
                self._rr_index[kind] = (self._rr_index[kind] + 1) % max(1, len(healthy))
                return chosen
            raise LoadBalancerError(
                f"no healthy backend available (tokens={tokens}, preferred={preferred})"
            )

    def _rr_tiebreak(self, kind: str, b: _BackendState) -> int:
        # Stable tiebreaker that rotates with the round-robin index.
        try:
            return (self._backends.index(b) - self._rr_index.get(kind, 0)) % max(
                1, len(self._backends)
            )
        except ValueError:
            return 0

    def _round_robin_kind(self) -> str:
        """Pick whichever kind has any healthy backend, alternating when
        both do — used when prompt size is in the ambiguous middle band."""
        cpu_ok = any(b.kind == "cpu" and b.is_healthy() for b in self._backends)
        gpu_ok = any(b.kind == "gpu" and b.is_healthy() for b in self._backends)
        if cpu_ok and gpu_ok:
            # Alternate based on which kind has fewer in-flight overall.
            cpu_load = sum(b.inflight for b in self._backends if b.kind == "cpu")
            gpu_load = sum(b.inflight for b in self._backends if b.kind == "gpu")
            return "cpu" if cpu_load <= gpu_load else "gpu"
        return "cpu" if cpu_ok else "gpu"

    # ── checkout / checkin (sync helpers) ─────────────────────────────────
    def checkout(self, backend: _BackendState) -> None:
        with self._lock:
            backend.inflight += 1

    def checkin(self, backend: _BackendState, success: bool, latency_ms: float) -> None:
        with self._lock:
            backend.inflight = max(0, backend.inflight - 1)
        backend.record(success=success, latency_ms=latency_ms)

    # ── high-level submit ─────────────────────────────────────────────────
    async def submit(self, prompt: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send ``prompt`` to the best backend and return the parsed JSON.

        On 5xx or timeout, fail over once to a backend of the OTHER kind.
        Returns the JSON body augmented with a ``_lb`` field describing
        which backend served the request and its observed latency.
        """
        # Local import keeps the module importable in environments that
        # don't ship httpx (e.g. read-only doc tooling).
        try:
            import httpx  # type: ignore
        except ImportError as exc:  # pragma: no cover - hard dep at runtime
            raise LoadBalancerError(f"httpx is required for submit(): {exc}") from exc

        params = params or {}
        primary = self.pick_backend(prompt, params)
        attempts: List[tuple[_BackendState, str]] = [(primary, "primary")]

        # Schedule a single failover on a backend of the OTHER kind.
        other_kind = "gpu" if primary.kind == "cpu" else "cpu"
        with self._lock:
            failover_candidates = self._healthy(other_kind)
        if failover_candidates:
            failover_candidates.sort(key=lambda b: b.inflight)
            attempts.append((failover_candidates[0], "failover"))

        last_exc: Optional[Exception] = None
        async with httpx.AsyncClient(timeout=self._timeout_sec) as client:
            for backend, role in attempts:
                self.checkout(backend)
                t0 = time.time()
                try:
                    url = f"{backend.endpoint_url}{self._path}"
                    resp = await client.post(url, json={"prompt": prompt, "params": params})
                    elapsed_ms = (time.time() - t0) * 1000.0
                    if resp.status_code >= 500:
                        self.checkin(backend, success=False, latency_ms=elapsed_ms)
                        last_exc = LoadBalancerError(
                            f"{backend.id} returned {resp.status_code}"
                        )
                        logger.warning(
                            "TextInferenceLoadBalancer: %s backend %s returned %d, role=%s",
                            backend.kind, backend.id, resp.status_code, role,
                        )
                        continue
                    if resp.status_code >= 400:
                        # 4xx is a caller error — don't fail over, surface it.
                        self.checkin(backend, success=False, latency_ms=elapsed_ms)
                        raise LoadBalancerError(
                            f"{backend.id} 4xx {resp.status_code}: {resp.text[:200]}"
                        )
                    body = resp.json() if resp.content else {}
                    self.checkin(backend, success=True, latency_ms=elapsed_ms)
                    if isinstance(body, dict):
                        body["_lb"] = {
                            "backend_id": backend.id,
                            "kind": backend.kind,
                            "role": role,
                            "latency_ms": round(elapsed_ms, 2),
                        }
                    return body if isinstance(body, dict) else {"result": body, "_lb": {
                        "backend_id": backend.id, "kind": backend.kind,
                        "role": role, "latency_ms": round(elapsed_ms, 2),
                    }}
                except (asyncio.TimeoutError,) as exc:
                    elapsed_ms = (time.time() - t0) * 1000.0
                    self.checkin(backend, success=False, latency_ms=elapsed_ms)
                    last_exc = exc
                    logger.warning(
                        "TextInferenceLoadBalancer: timeout on %s backend %s (%.1fms) role=%s",
                        backend.kind, backend.id, elapsed_ms, role,
                    )
                    continue
                except Exception as exc:  # noqa: BLE001 — relay/transport
                    elapsed_ms = (time.time() - t0) * 1000.0
                    # httpx.TimeoutException etc. all land here.
                    is_timeout = "Timeout" in type(exc).__name__
                    self.checkin(backend, success=False, latency_ms=elapsed_ms)
                    last_exc = exc
                    logger.warning(
                        "TextInferenceLoadBalancer: %s on %s backend %s role=%s",
                        type(exc).__name__, backend.kind, backend.id, role,
                    )
                    if not is_timeout and "Connect" not in type(exc).__name__:
                        # Unknown failure — don't keep retrying, surface it.
                        break
                    continue

        raise LoadBalancerError(
            f"all backends failed for prompt (last_error={last_exc!r})"
        )

    # ── introspection ─────────────────────────────────────────────────────
    def snapshot(self) -> Dict[str, Any]:
        """Per-backend snapshot for ops dashboards / tests."""
        out: List[Dict[str, Any]] = []
        for b in self._backends:
            snap = b.snapshot()
            snap["healthy"] = b.is_healthy()
            out.append(snap)
        return {
            "path": self._path,
            "timeout_sec": self._timeout_sec,
            "backends": out,
            "totals": {
                "cpu": sum(1 for b in self._backends if b.kind == "cpu"),
                "gpu": sum(1 for b in self._backends if b.kind == "gpu"),
                "inflight": sum(b.inflight for b in self._backends),
                "requests": sum(b.total_requests for b in self._backends),
                "failures": sum(b.total_failures for b in self._backends),
            },
        }


# ── Module-level singleton ────────────────────────────────────────────────
_text_lb_singleton: Optional[TextInferenceLoadBalancer] = None
_text_lb_lock = threading.Lock()


def get_text_load_balancer() -> TextInferenceLoadBalancer:
    """Process-wide :class:`TextInferenceLoadBalancer` configured from env.

    First call lazily reads ``NRS_TEXT_BACKENDS`` (JSON) or the convenience
    pair ``NRS_TEXT_CPU_URL`` / ``NRS_TEXT_GPU_URL``. Subsequent calls
    return the same instance so per-backend in-flight counters and health
    history accumulate correctly.
    """
    global _text_lb_singleton
    if _text_lb_singleton is None:
        with _text_lb_lock:
            if _text_lb_singleton is None:
                _text_lb_singleton = TextInferenceLoadBalancer()
    return _text_lb_singleton


def reset_text_load_balancer() -> None:
    """Test-only — drop the singleton so the next ``get_text_load_balancer()``
    re-reads env."""
    global _text_lb_singleton
    with _text_lb_lock:
        _text_lb_singleton = None
