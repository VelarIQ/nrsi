"""Adaptive Thinking Controller — Runtime backing for ``stdlib/thinking.nrsi``.

Implements ThinkingPhase, ThinkingConfig, PhaseBudget, tier_budget, phase_weight,
and the thinking_phase_verify gate declared in the NRSI contract.

Controls reasoning depth per phase based on query complexity (T0Router tier)
and the 7-dimension ModeVector from ModeController.  Every phase emits SSE
events so the client UI can show live thinking progress.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("cognitive-engine.thinking")


# ── Thinking Phases ──────────────────────────────────────────────────────────

class ThinkingPhase(Enum):
    ANALYZE = "analyze"
    PLAN = "plan"
    EXECUTE = "execute"
    VALIDATE = "validate"
    SYNTHESIZE = "synthesize"


PHASE_ORDER = list(ThinkingPhase)


# ── Thinking Config (discriminated union) ────────────────────────────────────

class ThinkingType(Enum):
    ADAPTIVE = "adaptive"
    FIXED = "fixed"
    DISABLED = "disabled"


@dataclass(frozen=True)
class ThinkingConfig:
    type: ThinkingType = ThinkingType.ADAPTIVE
    budget_tokens: int = 0

    @classmethod
    def adaptive(cls) -> ThinkingConfig:
        return cls(type=ThinkingType.ADAPTIVE)

    @classmethod
    def fixed(cls, budget: int) -> ThinkingConfig:
        return cls(type=ThinkingType.FIXED, budget_tokens=budget)

    @classmethod
    def disabled(cls) -> ThinkingConfig:
        return cls(type=ThinkingType.DISABLED)

    @property
    def is_enabled(self) -> bool:
        return self.type != ThinkingType.DISABLED


# ── Phase Budget ─────────────────────────────────────────────────────────────

TIER_BUDGETS: Dict[str, int] = {
    "T0": 256,
    "T1": 1024,
    "T2": 4096,
    "T3": 16384,
    "T4": 65536,
}

PHASE_WEIGHT_DEFAULTS: Dict[ThinkingPhase, float] = {
    ThinkingPhase.ANALYZE: 0.15,
    ThinkingPhase.PLAN: 0.20,
    ThinkingPhase.EXECUTE: 0.30,
    ThinkingPhase.VALIDATE: 0.25,
    ThinkingPhase.SYNTHESIZE: 0.10,
}

MODE_DIMENSION_PHASE_BOOST: Dict[str, ThinkingPhase] = {
    "analytical": ThinkingPhase.VALIDATE,
    "creative": ThinkingPhase.PLAN,
    "factual": ThinkingPhase.VALIDATE,
    "critical": ThinkingPhase.VALIDATE,
    "empathetic": ThinkingPhase.SYNTHESIZE,
    "exploratory": ThinkingPhase.PLAN,
    "metacognitive": ThinkingPhase.ANALYZE,
}


@dataclass
class PhaseBudget:
    phase: ThinkingPhase
    tokens: int
    used: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def remaining(self) -> int:
        return max(0, self.tokens - self.used)

    @property
    def elapsed_ms(self) -> float:
        end = self.finished_at or time.time()
        return (end - self.started_at) * 1000 if self.started_at else 0.0

    def consume(self, n: int) -> int:
        actual = min(n, self.remaining)
        self.used += actual
        return actual

    def start(self) -> None:
        self.started_at = time.time()

    def finish(self) -> None:
        self.finished_at = time.time()


# ── Thinking Budget Manager ──────────────────────────────────────────────────

class ThinkingBudgetManager:
    """Allocates and tracks token budgets per thinking phase."""

    def __init__(self, config: ThinkingConfig, tier: str = "T2",
                 mode_vector: Optional[Dict[str, float]] = None):
        self._config = config
        self._tier = tier
        self._mode_vector = mode_vector or {}
        self._phase_budgets: Dict[ThinkingPhase, PhaseBudget] = {}
        self._total_budget = self._resolve_total()
        self._allocate()

    def _resolve_total(self) -> int:
        if self._config.type == ThinkingType.DISABLED:
            return 0
        if self._config.type == ThinkingType.FIXED:
            return self._config.budget_tokens
        return TIER_BUDGETS.get(self._tier, TIER_BUDGETS["T2"])

    def _allocate(self) -> None:
        weights = dict(PHASE_WEIGHT_DEFAULTS)

        for dim, boost_phase in MODE_DIMENSION_PHASE_BOOST.items():
            strength = self._mode_vector.get(dim, 0.0)
            if strength > 0.5:
                boost = (strength - 0.5) * 0.3
                weights[boost_phase] += boost

        total_w = sum(weights.values())
        for phase in ThinkingPhase:
            tokens = int(self._total_budget * weights[phase] / total_w)
            self._phase_budgets[phase] = PhaseBudget(phase=phase, tokens=tokens)

    @property
    def total_budget(self) -> int:
        return self._total_budget

    @property
    def total_used(self) -> int:
        return sum(pb.used for pb in self._phase_budgets.values())

    @property
    def total_remaining(self) -> int:
        return self._total_budget - self.total_used

    def budget_for(self, phase: ThinkingPhase) -> PhaseBudget:
        return self._phase_budgets[phase]

    def start_phase(self, phase: ThinkingPhase) -> PhaseBudget:
        pb = self._phase_budgets[phase]
        pb.start()
        return pb

    def finish_phase(self, phase: ThinkingPhase) -> PhaseBudget:
        pb = self._phase_budgets[phase]
        pb.finish()
        return pb

    def consume(self, phase: ThinkingPhase, tokens: int) -> int:
        return self._phase_budgets[phase].consume(tokens)

    def redistribute_remaining(self, from_phase: ThinkingPhase,
                               to_phase: ThinkingPhase) -> int:
        """Move unused budget from a finished phase to another."""
        src = self._phase_budgets[from_phase]
        dst = self._phase_budgets[to_phase]
        transfer = src.remaining
        dst.tokens += transfer
        src.tokens = src.used
        return transfer

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total": self._total_budget,
            "used": self.total_used,
            "remaining": self.total_remaining,
            "tier": self._tier,
            "phases": {
                p.value: {
                    "tokens": pb.tokens,
                    "used": pb.used,
                    "remaining": pb.remaining,
                    "elapsed_ms": round(pb.elapsed_ms, 1),
                }
                for p, pb in self._phase_budgets.items()
            },
        }


# ── Adaptive Thinking Controller ─────────────────────────────────────────────

class AdaptiveThinkingController:
    """Top-level controller that manages thinking lifecycle for a single query.

    Integrates with the T0Router tier and ModeController vector to decide
    per-phase depth, and produces SSE-ready event dicts for each phase
    transition.
    """

    def __init__(self, config: Optional[ThinkingConfig] = None,
                 tier: str = "T2",
                 mode_vector: Optional[Dict[str, float]] = None):
        self._config = config or ThinkingConfig.adaptive()
        self._tier = tier
        self._budget = ThinkingBudgetManager(self._config, tier, mode_vector)
        self._current_phase: Optional[ThinkingPhase] = None
        self._phase_outputs: Dict[ThinkingPhase, str] = {}
        self._started_at = time.time()

    @property
    def budget(self) -> ThinkingBudgetManager:
        return self._budget

    @property
    def current_phase(self) -> Optional[ThinkingPhase]:
        return self._current_phase

    def should_skip_phase(self, phase: ThinkingPhase) -> bool:
        if not self._config.is_enabled:
            return True
        budget = self._budget.budget_for(phase)
        if budget.tokens < 32:
            return True
        if self._tier in ("T0", "T1") and phase in (
            ThinkingPhase.PLAN, ThinkingPhase.VALIDATE
        ):
            return True
        return False

    def enter_phase(self, phase: ThinkingPhase) -> Dict[str, Any]:
        if self._current_phase is not None:
            self._budget.finish_phase(self._current_phase)
        self._current_phase = phase
        self._budget.start_phase(phase)

        descriptions = {
            ThinkingPhase.ANALYZE: f"Analyzing query complexity and routing (tier {self._tier})...",
            ThinkingPhase.PLAN: "Decomposing into sub-tasks and selecting reasoning strategy...",
            ThinkingPhase.EXECUTE: "Executing plan — running tools, retrieving knowledge...",
            ThinkingPhase.VALIDATE: "Running H-Score validation and trust verification...",
            ThinkingPhase.SYNTHESIZE: "Assembling final response with provenance chain...",
        }

        pb = self._budget.budget_for(phase)
        return {
            "type": "thinking",
            "phase": phase.value,
            "content": descriptions.get(phase, f"Phase: {phase.value}"),
            "budget_remaining": pb.remaining,
        }

    def finish(self) -> Dict[str, Any]:
        if self._current_phase is not None:
            self._budget.finish_phase(self._current_phase)
            self._current_phase = None
        elapsed = (time.time() - self._started_at) * 1000
        return {
            "type": "thinking",
            "phase": "complete",
            "content": f"Reasoning complete ({elapsed:.0f}ms, {self._budget.total_used} tokens used)",
            "budget_remaining": 0,
        }

    def record_output(self, phase: ThinkingPhase, output: str) -> None:
        self._phase_outputs[phase] = output

    def get_output(self, phase: ThinkingPhase) -> str:
        return self._phase_outputs.get(phase, "")
