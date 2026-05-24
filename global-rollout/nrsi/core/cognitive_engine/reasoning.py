"""Multi-Strategy Reasoning — Runtime backing for ``stdlib/reasoning.nrsi``.

Implements ReasoningStrategy, Evidence, ReasoningStep, and the strategy-specific
gates (deductive_reasoning, inductive_reasoning, abductive_reasoning,
analogical_reasoning, causal_reasoning) plus resolve_conflict and
accumulate_evidence declared in the NRSI contract.

Built on the existing NRSI epistemic primitives (EpistemicType, ClaimRecord,
BeliefBase).
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("cognitive-engine.reasoning")


# ── Reasoning Strategy ───────────────────────────────────────────────────────

class ReasoningStrategy(Enum):
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"


STRATEGY_DESCRIPTIONS = {
    ReasoningStrategy.DEDUCTIVE: "Deriving conclusions from premises with logical certainty",
    ReasoningStrategy.INDUCTIVE: "Generalizing from specific observations",
    ReasoningStrategy.ABDUCTIVE: "Inferring the best explanation for observations",
    ReasoningStrategy.ANALOGICAL: "Drawing parallels from similar known cases",
    ReasoningStrategy.CAUSAL: "Tracing cause-effect relationships",
}


# ── Evidence ─────────────────────────────────────────────────────────────────

@dataclass
class Evidence:
    content: str
    source: str = ""
    confidence: float = 0.5
    strategy: Optional[ReasoningStrategy] = None
    timestamp: float = field(default_factory=time.time)
    evidence_hash: str = ""

    def __post_init__(self):
        if not self.evidence_hash:
            self.evidence_hash = hashlib.sha256(
                f"{self.content}:{self.source}".encode()
            ).hexdigest()[:16]


# ── Reasoning Step ───────────────────────────────────────────────────────────

@dataclass
class ReasoningStep:
    strategy: ReasoningStrategy
    conclusion: str
    evidence: List[Evidence] = field(default_factory=list)
    confidence: float = 0.5
    premises: List[str] = field(default_factory=list)
    step_index: int = 0
    parent_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def evidence_strength(self) -> float:
        if not self.evidence:
            return 0.0
        return sum(e.confidence for e in self.evidence) / len(self.evidence)


# ── Reasoning Chain ──────────────────────────────────────────────────────────

class ReasoningChain:
    """Ordered sequence of reasoning steps forming a complete argument."""

    def __init__(self, query: str, domain: str = "general"):
        self.query = query
        self.domain = domain
        self.steps: List[ReasoningStep] = []
        self._started_at = time.time()

    def add_step(
        self,
        strategy: ReasoningStrategy,
        conclusion: str,
        evidence: Optional[List[Evidence]] = None,
        confidence: float = 0.5,
        premises: Optional[List[str]] = None,
        parent_index: Optional[int] = None,
    ) -> ReasoningStep:
        step = ReasoningStep(
            strategy=strategy,
            conclusion=conclusion,
            evidence=evidence or [],
            confidence=confidence,
            premises=premises or [],
            step_index=len(self.steps),
            parent_index=parent_index,
        )
        self.steps.append(step)
        return step

    @property
    def overall_confidence(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.confidence for s in self.steps) / len(self.steps)

    @property
    def strategies_used(self) -> List[ReasoningStrategy]:
        return list(set(s.strategy for s in self.steps))

    @property
    def final_conclusion(self) -> str:
        if not self.steps:
            return ""
        return self.steps[-1].conclusion

    @property
    def elapsed_ms(self) -> float:
        return (time.time() - self._started_at) * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "domain": self.domain,
            "steps": [
                {
                    "index": s.step_index,
                    "strategy": s.strategy.value,
                    "conclusion": s.conclusion,
                    "confidence": round(s.confidence, 3),
                    "evidence_count": len(s.evidence),
                    "evidence_strength": round(s.evidence_strength, 3),
                    "premises": s.premises,
                }
                for s in self.steps
            ],
            "overall_confidence": round(self.overall_confidence, 3),
            "strategies_used": [s.value for s in self.strategies_used],
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


# ── Evidence Accumulator ─────────────────────────────────────────────────────

class EvidenceAccumulator:
    """Collects evidence from multiple reasoning strategies and detects conflicts."""

    def __init__(self):
        self._evidence: List[Evidence] = []
        self._by_strategy: Dict[ReasoningStrategy, List[Evidence]] = {
            s: [] for s in ReasoningStrategy
        }

    def add(self, evidence: Evidence) -> None:
        self._evidence.append(evidence)
        if evidence.strategy:
            self._by_strategy[evidence.strategy].append(evidence)

    def add_batch(self, items: List[Evidence]) -> None:
        for e in items:
            self.add(e)

    @property
    def total_evidence(self) -> int:
        return len(self._evidence)

    def evidence_for(self, strategy: ReasoningStrategy) -> List[Evidence]:
        return self._by_strategy.get(strategy, [])

    def strongest(self, n: int = 5) -> List[Evidence]:
        return sorted(self._evidence, key=lambda e: e.confidence, reverse=True)[:n]

    def detect_conflicts(self, threshold: float = 0.3) -> List[Tuple[Evidence, Evidence]]:
        """Find pairs of evidence that likely conflict (low mutual confidence)."""
        conflicts = []
        for i, a in enumerate(self._evidence):
            for b in self._evidence[i + 1:]:
                if a.strategy != b.strategy:
                    gap = abs(a.confidence - b.confidence)
                    if gap > threshold and min(a.confidence, b.confidence) < 0.4:
                        conflicts.append((a, b))
        return conflicts

    def aggregate_confidence(self) -> float:
        if not self._evidence:
            return 0.0
        weights = [e.confidence for e in self._evidence]
        return sum(w * w for w in weights) / sum(weights)


# ── Conflict Resolver ────────────────────────────────────────────────────────

class ConflictResolver:
    """Resolves conflicts between reasoning strategies.

    Uses a priority ordering and evidence strength to decide which
    conclusion to trust when strategies disagree.
    """

    STRATEGY_PRIORITY = {
        ReasoningStrategy.DEDUCTIVE: 5,
        ReasoningStrategy.CAUSAL: 4,
        ReasoningStrategy.INDUCTIVE: 3,
        ReasoningStrategy.ABDUCTIVE: 2,
        ReasoningStrategy.ANALOGICAL: 1,
    }

    def resolve(
        self,
        chain: ReasoningChain,
        accumulator: EvidenceAccumulator,
    ) -> ReasoningStep:
        """Pick the most trustworthy conclusion from a chain with conflicts."""
        if not chain.steps:
            return ReasoningStep(
                strategy=ReasoningStrategy.ABDUCTIVE,
                conclusion="Insufficient evidence to form conclusion",
                confidence=0.0,
            )

        scored = []
        for step in chain.steps:
            priority = self.STRATEGY_PRIORITY.get(step.strategy, 0)
            score = (
                step.confidence * 0.5
                + step.evidence_strength * 0.3
                + (priority / 5.0) * 0.2
            )
            scored.append((score, step))

        scored.sort(key=lambda x: x[0], reverse=True)
        winner = scored[0][1]

        if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.1:
            winner = ReasoningStep(
                strategy=winner.strategy,
                conclusion=winner.conclusion,
                evidence=winner.evidence,
                confidence=winner.confidence * 0.9,
                premises=winner.premises,
                step_index=winner.step_index,
                metadata={"contested": True, "runner_up": scored[1][1].strategy.value},
            )

        return winner
