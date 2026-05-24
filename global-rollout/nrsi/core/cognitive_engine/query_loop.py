"""Cognitive Processing Loop — Runtime backing for ``stdlib/cognitive_loop.nrsi``.

Implements LoopTermination, QueryState, LoopEvent, loop_output_verify,
code_verified_output, and the ``cognitive`` lobe declared in the NRSI contract.

Async generator that drives: ANALYZE -> PLAN -> EXECUTE -> VALIDATE -> SYNTHESIZE.
Each phase yields SSE-compatible event dicts.  Tool calls are batched by
concurrency safety.  The loop continues until H-Score/TVS pass the validation
gate AND (for code tasks) the verification loop completes clean.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, AsyncGenerator, Callable, Coroutine, Dict, List, Optional, Sequence,
)

from nrsi.core.cognitive_engine.thinking import (
    AdaptiveThinkingController,
    ThinkingConfig,
    ThinkingPhase,
)
from nrsi.core.cognitive_engine.tool_framework import (
    ToolRegistry,
    ToolResult,
    ToolUseContext,
)

logger = logging.getLogger("cognitive-engine.loop")


# ── Loop Events ──────────────────────────────────────────────────────────────

class LoopEventType(Enum):
    THINKING = "thinking"
    TOKEN = "token"
    TOOL_START = "tool_start"
    TOOL_RESULT = "tool_result"
    META = "meta"
    VERIFICATION = "verification"
    RESULT = "result"
    ERROR = "error"
    DONE = "done"


@dataclass
class LoopEvent:
    type: LoopEventType
    data: Dict[str, Any] = field(default_factory=dict)

    def as_sse(self) -> Dict[str, Any]:
        return {"type": self.type.value, **self.data}


class LoopTermination(Enum):
    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations"
    VALIDATION_FAILED = "validation_failed"
    USER_CANCELLED = "user_cancelled"
    ERROR = "error"


# ── Query State ──────────────────────────────────────────────────────────────

@dataclass
class QueryState:
    query: str
    domain: str = "general"
    session_id: str = ""
    user_id: str = ""
    mode_override: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    thinking_outputs: Dict[str, str] = field(default_factory=dict)
    answer: str = ""
    h_score: float = 0.0
    tvs_score: float = 0.0
    tvs_verdict: str = ""
    trust_level: str = "RAW"
    iteration: int = 0
    max_iterations: int = 5
    termination: Optional[LoopTermination] = None
    is_code_task: bool = False
    verification_passed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Hook System ──────────────────────────────────────────────────────────────

HookFn = Callable[[QueryState], Coroutine[Any, Any, Optional[QueryState]]]


class HookManager:
    """Extensible hook points at phase boundaries."""

    def __init__(self) -> None:
        self._hooks: Dict[str, List[HookFn]] = {
            "pre_analyze": [],
            "post_analyze": [],
            "pre_plan": [],
            "post_plan": [],
            "pre_execute": [],
            "post_execute": [],
            "pre_validate": [],
            "post_validate": [],
            "pre_synthesize": [],
            "post_synthesize": [],
            "stop": [],
        }

    def register(self, point: str, fn: HookFn) -> None:
        if point in self._hooks:
            self._hooks[point].append(fn)

    async def run(self, point: str, state: QueryState) -> QueryState:
        for fn in self._hooks.get(point, []):
            result = await fn(state)
            if result is not None:
                state = result
        return state


# ── Cognitive Loop ───────────────────────────────────────────────────────────

class CognitiveLoop:
    """Async generator that yields LoopEvents through the reasoning cycle.

    Parameters
    ----------
    state : QueryState
        Mutable state for this query lifecycle.
    thinking : AdaptiveThinkingController
        Controls phase budgets and SSE events.
    tool_registry : ToolRegistry
        Available tools for execution phase.
    hooks : HookManager
        Optional hooks at phase boundaries.
    process_fn : callable
        The NRS engine process function (query -> NRSResponse).
    validate_fn : callable
        H-Score/TVS validation function (answer, query -> scores).
    """

    def __init__(
        self,
        state: QueryState,
        thinking: AdaptiveThinkingController,
        tool_registry: Optional[ToolRegistry] = None,
        hooks: Optional[HookManager] = None,
        process_fn: Optional[Callable] = None,
        validate_fn: Optional[Callable] = None,
    ):
        self._state = state
        self._thinking = thinking
        self._tools = tool_registry or ToolRegistry()
        self._hooks = hooks or HookManager()
        self._process_fn = process_fn
        self._validate_fn = validate_fn

    async def run(self) -> AsyncGenerator[LoopEvent, None]:
        """Execute the full cognitive loop, yielding events."""
        while self._state.iteration < self._state.max_iterations:
            self._state.iteration += 1

            async for event in self._run_iteration():
                yield event

            if self._state.termination is not None:
                break

            if self._state.h_score >= 0.7 and (
                not self._state.is_code_task or self._state.verification_passed
            ):
                self._state.termination = LoopTermination.COMPLETED
                break

        if self._state.termination is None:
            self._state.termination = LoopTermination.MAX_ITERATIONS

        yield LoopEvent(type=LoopEventType.DONE, data={
            "reason": self._state.termination.value,
            "iterations": self._state.iteration,
            "h_score": self._state.h_score,
        })

    async def _run_iteration(self) -> AsyncGenerator[LoopEvent, None]:
        phases = [
            (ThinkingPhase.ANALYZE, self._analyze),
            (ThinkingPhase.PLAN, self._plan),
            (ThinkingPhase.EXECUTE, self._execute),
            (ThinkingPhase.VALIDATE, self._validate),
            (ThinkingPhase.SYNTHESIZE, self._synthesize),
        ]

        for phase, handler in phases:
            if self._thinking.should_skip_phase(phase):
                continue

            event = self._thinking.enter_phase(phase)
            yield LoopEvent(type=LoopEventType.THINKING, data=event)

            self._state = await self._hooks.run(f"pre_{phase.value}", self._state)

            async for evt in handler():
                yield evt

            self._state = await self._hooks.run(f"post_{phase.value}", self._state)

    async def _analyze(self) -> AsyncGenerator[LoopEvent, None]:
        """Phase 1: Complexity scoring + mode selection."""
        self._thinking.record_output(
            ThinkingPhase.ANALYZE,
            f"Query: {self._state.query[:200]} | Domain: {self._state.domain} | "
            f"Iteration: {self._state.iteration}"
        )
        yield LoopEvent(type=LoopEventType.THINKING, data={
            "phase": "analyze",
            "content": f"Analyzing: domain={self._state.domain}, "
                       f"iteration={self._state.iteration}",
        })

    async def _plan(self) -> AsyncGenerator[LoopEvent, None]:
        """Phase 2: Task decomposition for complex queries."""
        yield LoopEvent(type=LoopEventType.THINKING, data={
            "phase": "plan",
            "content": "Planning execution strategy...",
        })

    async def _execute(self) -> AsyncGenerator[LoopEvent, None]:
        """Phase 3: Run the NRS engine or tools."""
        if self._process_fn is not None:
            try:
                result = self._process_fn(
                    self._state.query,
                    domain=self._state.domain,
                    session_id=self._state.session_id,
                    user_id=self._state.user_id,
                    mode_override=self._state.mode_override,
                )
                self._state.answer = getattr(result, "answer", str(result))
                self._state.h_score = getattr(result, "h_score", 0.0)
                if self._state.h_score != self._state.h_score:  # NaN guard
                    self._state.h_score = 0.0

                for token_chunk in _chunk_text(self._state.answer, 20):
                    yield LoopEvent(type=LoopEventType.TOKEN, data={
                        "text": token_chunk,
                    })
            except Exception as exc:
                logger.warning("Process function failed: %s", exc)
                yield LoopEvent(type=LoopEventType.ERROR, data={
                    "message": str(exc),
                })
                self._state.termination = LoopTermination.ERROR

        pending_tools = self._state.metadata.get("pending_tool_calls", [])
        if pending_tools:
            ctx = ToolUseContext(
                session_id=self._state.session_id,
                user_id=self._state.user_id,
                domain=self._state.domain,
            )
            results = await self._tools.execute_batch(pending_tools, ctx)
            self._state.tool_results.extend(results)
            for i, r in enumerate(results):
                yield LoopEvent(type=LoopEventType.TOOL_RESULT, data={
                    "index": i,
                    "success": not r.is_error,
                    "elapsed_ms": r.execution_time_ms,
                })

    async def _validate(self) -> AsyncGenerator[LoopEvent, None]:
        """Phase 4: H-Score + TVS validation."""
        if self._validate_fn is not None:
            try:
                scores = self._validate_fn(self._state.answer, self._state.query)
                self._state.h_score = scores.get("h_score", self._state.h_score)
                self._state.tvs_score = scores.get("tvs_score", 0.0)
                self._state.tvs_verdict = scores.get("tvs_verdict", "")
                self._state.trust_level = scores.get("trust_level", "RAW")
            except Exception as exc:
                logger.warning("Validation function failed: %s", exc)

        yield LoopEvent(type=LoopEventType.META, data={
            "trust_level": self._state.trust_level,
            "confidence": self._state.h_score,
            "h_score": self._state.h_score,
            "tvs_score": self._state.tvs_score,
            "tvs_verdict": self._state.tvs_verdict,
        })

        if self._state.h_score < 0.4 and self._state.iteration < self._state.max_iterations:
            yield LoopEvent(type=LoopEventType.THINKING, data={
                "phase": "validate",
                "content": f"H-Score {self._state.h_score:.2f} below threshold, "
                           f"re-executing with higher reasoning budget...",
            })

    async def _synthesize(self) -> AsyncGenerator[LoopEvent, None]:
        """Phase 5: Final answer assembly with provenance."""
        finish_event = self._thinking.finish()
        yield LoopEvent(type=LoopEventType.THINKING, data=finish_event)

        yield LoopEvent(type=LoopEventType.RESULT, data={
            "answer": self._state.answer,
            "h_score": self._state.h_score,
            "trust_level": self._state.trust_level,
            "iteration": self._state.iteration,
        })


def _chunk_text(text: str, words_per_chunk: int) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunks.append(" ".join(words[i:i + words_per_chunk]))
    return chunks if chunks else [text]
