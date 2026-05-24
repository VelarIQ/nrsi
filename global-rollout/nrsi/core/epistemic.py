"""NRSI Epistemic Extensions — Knowledge Types for AGI Reasoning.

The core NRSI type system (raw→validated→trusted→certified) expresses
HOW MUCH we trust data.  These extensions express WHY and HOW we know it.

A logic proof and a web search result may both be 'validated', but they
represent fundamentally different epistemic states.  A deductive proof is
certain given its premises.  An inductive generalization may be wrong.
A causal chain has propagation uncertainty.  An analogy is creative transfer.

These distinctions matter for:
  - Deciding when to use a fact in high-stakes reasoning
  - Knowing what can be contradicted and what can't
  - Understanding how confidence should propagate through chains
  - Tracking knowledge lifecycle as it evolves
  - Expressing temporal bounds on knowledge validity
  - Resolving conflicts between competing knowledge claims

Patent-covered: NRSI Epistemic Knowledge Type System, VelarIQ.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto, IntEnum
from typing import (
    Any,
    Dict,
    FrozenSet,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from nrsi.core.types import (
    NRSIData,
    Confidence,
    TrustLevel,
    ProvenanceEntry,
    raw,
    validated,
    trusted,
)
from nrsi.core.errors import TrustError
from nrsi.core.lobes import LobeType


T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. EpistemicType — How we know something
# ═══════════════════════════════════════════════════════════════════════════════

class EpistemicType(Enum):
    """The epistemic origin of a piece of knowledge.

    This is orthogonal to trust level — DEDUCTIVE knowledge can be raw
    (not yet validated) or trusted (validated proof chain).  But a trusted
    DEDUCTIVE fact carries stronger guarantees than a trusted INDUCTIVE one.
    """

    DEDUCTIVE = auto()       # Proved by formal logic (modus ponens, resolution, …)
    INDUCTIVE = auto()       # Generalized from specific examples
    ABDUCTIVE = auto()       # Best explanation for observations
    ANALOGICAL = auto()      # Transferred from another domain
    CAUSAL = auto()          # Established through causal chain
    COMPUTATIONAL = auto()   # Computed deterministically (math, code execution)
    OBSERVATIONAL = auto()   # Directly observed / retrieved from source
    TESTIMONIAL = auto()     # Reported by another agent/source
    CREATIVE = auto()        # Synthesized creatively (may not correspond to reality)
    SPECULATIVE = auto()     # Hypothetical / counterfactual


_EPISTEMIC_STRENGTH: Dict[EpistemicType, float] = {
    EpistemicType.DEDUCTIVE: 1.0,
    EpistemicType.COMPUTATIONAL: 0.99,
    EpistemicType.OBSERVATIONAL: 0.85,
    EpistemicType.CAUSAL: 0.80,
    EpistemicType.INDUCTIVE: 0.70,
    EpistemicType.TESTIMONIAL: 0.60,
    EpistemicType.ABDUCTIVE: 0.55,
    EpistemicType.ANALOGICAL: 0.50,
    EpistemicType.CREATIVE: 0.30,
    EpistemicType.SPECULATIVE: 0.20,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CognitiveOrigin — Which lobe/mode/process produced this
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CognitiveOrigin:
    """Tracks exactly how a piece of knowledge was produced.

    This is the 'birth certificate' for every NRSIData value that
    comes from an AGI cognitive engine.
    """

    lobe: Optional[LobeType] = None
    epistemic_type: EpistemicType = EpistemicType.OBSERVATIONAL
    mode_spectrum: str = ""           # DETERMINISTIC, CREATIVE, etc.
    engine_name: str = ""             # "logic_engine", "causal_reasoner", etc.
    method_name: str = ""             # "forward_chain", "compute", etc.
    reasoning_depth: int = 0          # How many inference steps
    evidence_count: int = 0           # How many pieces of evidence support this
    timestamp: float = field(default_factory=time.time)

    def strength_factor(self) -> float:
        """How much should we weight this origin when combining knowledge?"""
        return _EPISTEMIC_STRENGTH.get(self.epistemic_type, 0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TemporalScope — When knowledge is valid
#    (Defined before ReasoningProvenance so forward reference resolves.)
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalValidity(Enum):
    """How stable this knowledge is over time."""

    ETERNAL = auto()       # Mathematical truths, logical axioms
    STABLE = auto()        # Physical constants, well-established science
    DURABLE = auto()       # Institutional facts (Paris is capital of France)
    CURRENT = auto()       # True now but subject to change (current president)
    HISTORICAL = auto()    # Was true, may not be now
    EPHEMERAL = auto()     # True only briefly (current weather, stock price)
    PROJECTED = auto()     # Expected to be true in the future


_STALENESS_HALF_LIVES: Dict[TemporalValidity, float] = {
    TemporalValidity.DURABLE: 365.25 * 86400,   # ~1 year
    TemporalValidity.CURRENT: 30.0 * 86400,     # 30 days
    TemporalValidity.HISTORICAL: 3650 * 86400,  # ~10 years (slow decay)
    TemporalValidity.EPHEMERAL: 3600.0,          # 1 hour
    TemporalValidity.PROJECTED: 7.0 * 86400,    # 1 week
}

_LN2 = math.log(2)


@dataclass(frozen=True)
class TemporalScope:
    """When is this knowledge valid?"""

    validity: TemporalValidity = TemporalValidity.CURRENT
    valid_from: Optional[float] = None     # Unix timestamp
    valid_until: Optional[float] = None    # Unix timestamp, None = indefinite
    last_verified: Optional[float] = None  # When was this last checked?

    def is_current(self, now: Optional[float] = None) -> bool:
        """Is this knowledge currently valid?"""
        now = now if now is not None else time.time()

        if self.validity in (TemporalValidity.ETERNAL, TemporalValidity.STABLE):
            return True

        if self.valid_until is not None and now > self.valid_until:
            return False

        if self.valid_from is not None and now < self.valid_from:
            return False

        if self.validity == TemporalValidity.PROJECTED:
            if self.valid_from is not None and now < self.valid_from:
                return True
            return self.valid_until is None or now <= self.valid_until

        return True

    def staleness_factor(self, now: Optional[float] = None) -> float:
        """0.0 = fresh, 1.0 = completely stale.  Affects confidence."""
        now = now if now is not None else time.time()

        if self.validity in (TemporalValidity.ETERNAL, TemporalValidity.STABLE):
            return 0.0

        if self.valid_until is not None and now > self.valid_until:
            return 1.0

        ref_time = self.last_verified or self.valid_from
        if ref_time is None:
            return 0.0

        age_seconds = max(0.0, now - ref_time)
        half_life = _STALENESS_HALF_LIVES.get(self.validity, 30.0 * 86400)
        return min(1.0, 1.0 - math.exp(-_LN2 * age_seconds / half_life))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ReasoningProvenance — Extended provenance for AGI
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ReasoningProvenance(ProvenanceEntry):
    """Extended provenance that records cognitive reasoning details.

    Inherits from ProvenanceEntry so it's compatible with existing
    NRSIData provenance chains, but adds reasoning-specific fields.
    """

    cognitive_origin: Optional[CognitiveOrigin] = None
    evidence_chain: Tuple[str, ...] = ()        # IDs of supporting evidence
    reasoning_steps: Tuple[str, ...] = ()       # Human-readable reasoning trace
    premises: Tuple[str, ...] = ()              # What was assumed
    alternatives_considered: int = 0             # How many alternatives examined
    contradiction_checked: bool = False          # Was contradiction detection run?
    counterfactual: bool = False                 # Is this a hypothetical?
    temporal_scope: Optional[TemporalScope] = None


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ClaimLifecycle — Knowledge evolution
# ═══════════════════════════════════════════════════════════════════════════════

class ClaimStatus(IntEnum):
    """Lifecycle stage of a knowledge claim."""

    EXTRACTED = 0       # Just extracted from text, unvalidated
    VALIDATED = 1       # Passed at least one validation
    CORROBORATED = 2    # Confirmed by multiple independent sources
    ESTABLISHED = 3     # High confidence, widely supported
    CHALLENGED = 4      # Previously established but now contradicted
    DEPRECATED = 5      # Superseded by newer knowledge
    RETRACTED = 6       # Explicitly determined to be false


_ALLOWED_TRANSITIONS: Dict[ClaimStatus, FrozenSet[ClaimStatus]] = {
    ClaimStatus.EXTRACTED: frozenset({ClaimStatus.VALIDATED, ClaimStatus.RETRACTED}),
    ClaimStatus.VALIDATED: frozenset({
        ClaimStatus.CORROBORATED, ClaimStatus.CHALLENGED, ClaimStatus.RETRACTED,
    }),
    ClaimStatus.CORROBORATED: frozenset({
        ClaimStatus.ESTABLISHED, ClaimStatus.CHALLENGED, ClaimStatus.RETRACTED,
    }),
    ClaimStatus.ESTABLISHED: frozenset({
        ClaimStatus.CHALLENGED, ClaimStatus.DEPRECATED, ClaimStatus.RETRACTED,
    }),
    ClaimStatus.CHALLENGED: frozenset({
        ClaimStatus.ESTABLISHED, ClaimStatus.DEPRECATED, ClaimStatus.RETRACTED,
    }),
    ClaimStatus.DEPRECATED: frozenset({ClaimStatus.RETRACTED}),
    ClaimStatus.RETRACTED: frozenset(),
}


@dataclass
class ClaimRecord:
    """A knowledge claim with full lifecycle tracking."""

    claim_id: str
    content: str
    status: ClaimStatus = ClaimStatus.EXTRACTED
    epistemic_type: EpistemicType = EpistemicType.OBSERVATIONAL
    confidence: float = 0.0
    temporal_scope: Optional[TemporalScope] = None
    cognitive_origin: Optional[CognitiveOrigin] = None
    corroboration_count: int = 0
    challenge_count: int = 0
    sources: List[str] = field(default_factory=list)
    history: List[Tuple[float, ClaimStatus, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def advance(self, new_status: ClaimStatus, reason: str) -> None:
        """Move to next lifecycle stage with audit.

        Raises ValueError if the transition is not allowed by the
        lifecycle state machine.
        """
        allowed = _ALLOWED_TRANSITIONS.get(self.status, frozenset())
        if new_status not in allowed:
            raise ValueError(
                f"Cannot transition {self.status.name} → {new_status.name}. "
                f"Allowed: {', '.join(s.name for s in sorted(allowed))}."
            )
        now = time.time()
        self.history.append((now, new_status, reason))
        self.status = new_status
        self.updated_at = now

    def challenge(self, reason: str, source: str = "") -> None:
        """Challenge this claim."""
        self.challenge_count += 1
        if source:
            self.sources.append(f"challenge:{source}")
        if self.status in (
            ClaimStatus.VALIDATED,
            ClaimStatus.CORROBORATED,
            ClaimStatus.ESTABLISHED,
        ):
            self.advance(ClaimStatus.CHALLENGED, reason)
        else:
            now = time.time()
            self.history.append((now, self.status, f"challenge noted: {reason}"))
            self.updated_at = now

    def corroborate(self, source: str, confidence: float) -> None:
        """Add corroborating evidence."""
        self.corroboration_count += 1
        if source:
            self.sources.append(source)
        self.confidence = max(self.confidence, confidence)
        now = time.time()
        self.updated_at = now
        self.history.append(
            (now, self.status, f"corroborated by {source} (conf={confidence:.4f})")
        )

        if (
            self.status == ClaimStatus.VALIDATED
            and self.corroboration_count >= 2
        ):
            self.advance(ClaimStatus.CORROBORATED, "multi-source corroboration")
        elif (
            self.status == ClaimStatus.CORROBORATED
            and self.corroboration_count >= 5
            and self.confidence >= 0.90
        ):
            self.advance(ClaimStatus.ESTABLISHED, "high-confidence corroboration")

    def __repr__(self) -> str:
        return (
            f"ClaimRecord({self.claim_id!r}, status={self.status.name}, "
            f"conf={self.confidence:.2f}, "
            f"corr={self.corroboration_count}, chal={self.challenge_count})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Composition Operators — Combining NRSIData
# ═══════════════════════════════════════════════════════════════════════════════

class EpistemicOps:
    """Operations for combining, deriving, and resolving epistemic data.

    These are the operators the AGI engines need:
      combine  — merge two knowledge items into a stronger one
      derive   — create new knowledge from reasoning over inputs
      resolve  — handle conflicting knowledge
      propagate — compute confidence through a reasoning chain

    All methods are pure functions with no side effects.
    """

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_epistemic_type(data: NRSIData) -> EpistemicType:
        if isinstance(data, EpistemicNRSIData):
            return data.epistemic_type
        return EpistemicType.OBSERVATIONAL

    @staticmethod
    def _extract_strength(data: NRSIData) -> float:
        et = EpistemicOps._extract_epistemic_type(data)
        return _EPISTEMIC_STRENGTH.get(et, 0.5)

    @staticmethod
    def _extract_origin(data: NRSIData) -> Optional[CognitiveOrigin]:
        if isinstance(data, EpistemicNRSIData):
            return data.cognitive_origin
        return None

    @staticmethod
    def _evidence_count(data: NRSIData) -> int:
        origin = EpistemicOps._extract_origin(data)
        if origin is not None:
            return origin.evidence_count
        return len(data.provenance)

    # ── combine ───────────────────────────────────────────────────────

    @staticmethod
    def combine(
        a: NRSIData,
        b: NRSIData,
        method: str = "conservative",
    ) -> NRSIData:
        """Combine two NRSIData values.

        Methods
        -------
        conservative — min confidence, lowest shared trust
        optimistic   — max confidence, highest trust
        bayesian     — P(A∧B) = P(A)·P(B)  for independent facts
        corroborative — confidence increases when sources agree
        """
        if method == "conservative":
            confidence = min(a.confidence, b.confidence)
            trust = min(a.trust_level, b.trust_level, key=lambda t: t.value)
        elif method == "optimistic":
            confidence = max(a.confidence, b.confidence)
            trust = max(a.trust_level, b.trust_level, key=lambda t: t.value)
        elif method == "bayesian":
            confidence = a.confidence * b.confidence
            trust = min(a.trust_level, b.trust_level, key=lambda t: t.value)
        elif method == "corroborative":
            confidence = 1.0 - (1.0 - a.confidence) * (1.0 - b.confidence)
            trust = max(a.trust_level, b.trust_level, key=lambda t: t.value)
        else:
            raise ValueError(f"Unknown combine method: {method!r}")

        confidence = Confidence.validate(confidence)

        merged_provenance = list(a.provenance) + list(b.provenance)
        merged_provenance.append(ProvenanceEntry(
            timestamp=time.time(),
            action="combined",
            confidence=confidence,
            reason=f"method={method} from [{a.id[:8]}] + [{b.id[:8]}]",
        ))

        winner = a if a.confidence >= b.confidence else b
        merged_meta = {**getattr(a, "_metadata", {}), **getattr(b, "_metadata", {})}

        return NRSIData(
            value=winner.value,
            trust_level=trust,
            confidence=confidence,
            provenance=merged_provenance,
            metadata=merged_meta,
        )

    # ── derive ────────────────────────────────────────────────────────

    @staticmethod
    def derive(
        inputs: List[NRSIData],
        derivation: str,
        epistemic_type: EpistemicType = EpistemicType.DEDUCTIVE,
        reasoning_steps: Optional[List[str]] = None,
    ) -> NRSIData:
        """Create new knowledge derived from multiple inputs.

        Confidence propagation per epistemic type:
          DEDUCTIVE     — min of input confidences (chain = weakest link)
          COMPUTATIONAL — same as deductive
          INDUCTIVE     — geometric mean with corroboration boost
          CAUSAL        — product of edge strengths
          ANALOGICAL    — structural_score * min(inputs)
          Others        — arithmetic mean
        """
        if not inputs:
            return NRSIData(derivation, TrustLevel.RAW, Confidence.NONE)

        confidences = [inp.confidence for inp in inputs]

        if epistemic_type in (EpistemicType.DEDUCTIVE, EpistemicType.COMPUTATIONAL):
            derived_confidence = min(confidences)

        elif epistemic_type == EpistemicType.INDUCTIVE:
            if any(c <= 0 for c in confidences):
                derived_confidence = 0.0
            else:
                geo = math.exp(
                    sum(math.log(c) for c in confidences) / len(confidences)
                )
                boost = min(1.0, 1.0 + 0.05 * (len(inputs) - 1))
                derived_confidence = min(1.0, geo * boost)

        elif epistemic_type == EpistemicType.CAUSAL:
            derived_confidence = 1.0
            for c in confidences:
                derived_confidence *= c

        elif epistemic_type == EpistemicType.ANALOGICAL:
            structural_score = sum(confidences) / len(confidences)
            derived_confidence = structural_score * min(confidences)

        else:
            derived_confidence = sum(confidences) / len(confidences)

        derived_confidence = Confidence.validate(
            max(0.0, min(1.0, derived_confidence))
        )

        trust_values = [inp.trust_level for inp in inputs]
        derived_trust = min(trust_values, key=lambda t: t.value)

        merged_prov: List[ProvenanceEntry] = []
        for inp in inputs:
            merged_prov.extend(inp.provenance)

        merged_prov.append(ReasoningProvenance(
            timestamp=time.time(),
            action="derived",
            confidence=derived_confidence,
            reason=derivation,
            cognitive_origin=CognitiveOrigin(epistemic_type=epistemic_type),
            evidence_chain=tuple(inp.id for inp in inputs),
            reasoning_steps=tuple(reasoning_steps or []),
        ))

        return NRSIData(
            value=derivation,
            trust_level=derived_trust,
            confidence=derived_confidence,
            provenance=merged_prov,
        )

    # ── resolve ───────────────────────────────────────────────────────

    @staticmethod
    def resolve_conflict(
        a: NRSIData,
        b: NRSIData,
    ) -> Tuple[NRSIData, str]:
        """Resolve conflicting knowledge claims.

        Resolution priority:
          1. Higher trust level wins
          2. Higher epistemic strength wins (deductive > inductive)
          3. Higher confidence wins
          4. More recent wins (temporal / provenance timestamp)
          5. More evidence wins

        Returns (winner, explanation).
        """
        reasons: List[str] = []

        # 1 — trust level
        if a.trust_level.value != b.trust_level.value:
            winner = a if a.trust_level.value > b.trust_level.value else b
            reasons.append(
                f"trust {winner.trust_level.name} > "
                f"{(b if winner is a else a).trust_level.name}"
            )
            return winner, "; ".join(reasons)

        # 2 — epistemic strength
        str_a = EpistemicOps._extract_strength(a)
        str_b = EpistemicOps._extract_strength(b)
        if abs(str_a - str_b) > 1e-6:
            winner = a if str_a > str_b else b
            loser = b if winner is a else a
            reasons.append(
                f"epistemic strength "
                f"{EpistemicOps._extract_epistemic_type(winner).name}"
                f"({EpistemicOps._extract_strength(winner):.2f}) > "
                f"{EpistemicOps._extract_epistemic_type(loser).name}"
                f"({EpistemicOps._extract_strength(loser):.2f})"
            )
            return winner, "; ".join(reasons)

        # 3 — confidence
        if abs(a.confidence - b.confidence) > 1e-6:
            winner = a if a.confidence > b.confidence else b
            reasons.append(
                f"confidence {winner.confidence:.4f} > "
                f"{(b if winner is a else a).confidence:.4f}"
            )
            return winner, "; ".join(reasons)

        # 4 — recency (latest provenance timestamp)
        ts_a = a.provenance[-1].timestamp if a.provenance else 0.0
        ts_b = b.provenance[-1].timestamp if b.provenance else 0.0
        if abs(ts_a - ts_b) > 0.001:
            winner = a if ts_a > ts_b else b
            reasons.append("more recent provenance")
            return winner, "; ".join(reasons)

        # 5 — evidence count
        ev_a = EpistemicOps._evidence_count(a)
        ev_b = EpistemicOps._evidence_count(b)
        if ev_a != ev_b:
            winner = a if ev_a > ev_b else b
            reasons.append(f"more evidence ({max(ev_a, ev_b)} vs {min(ev_a, ev_b)})")
            return winner, "; ".join(reasons)

        return a, "tie — first argument wins by convention"

    # ── propagate ─────────────────────────────────────────────────────

    @staticmethod
    def propagate_confidence(
        chain: List[float],
        epistemic_type: EpistemicType,
    ) -> float:
        """Compute final confidence through a multi-step reasoning chain.

        DEDUCTIVE      — min(chain)            (weakest link)
        COMPUTATIONAL  — min(chain)            (deterministic)
        CAUSAL         — product(chain)        (multiplicative decay)
        INDUCTIVE      — geometric_mean(chain) (moderate decay)
        ANALOGICAL     — mean(chain) * 0.7     (heavy discount)
        SPECULATIVE    — mean(chain) * 0.3     (very heavy discount)
        Others         — geometric_mean(chain)
        """
        if not chain:
            return 0.0

        chain = [Confidence.validate(c) for c in chain]

        if epistemic_type in (EpistemicType.DEDUCTIVE, EpistemicType.COMPUTATIONAL):
            return min(chain)

        if epistemic_type == EpistemicType.CAUSAL:
            result = 1.0
            for c in chain:
                result *= c
            return result

        if epistemic_type == EpistemicType.INDUCTIVE:
            if any(c <= 0 for c in chain):
                return 0.0
            return math.exp(sum(math.log(c) for c in chain) / len(chain))

        if epistemic_type == EpistemicType.ANALOGICAL:
            return (sum(chain) / len(chain)) * 0.7

        if epistemic_type == EpistemicType.SPECULATIVE:
            return (sum(chain) / len(chain)) * 0.3

        # Fallback — geometric mean
        if any(c <= 0 for c in chain):
            return 0.0
        return math.exp(sum(math.log(c) for c in chain) / len(chain))


# ═══════════════════════════════════════════════════════════════════════════════
# 7. GoalContract / PlanContract — First-class goals and plans
# ═══════════════════════════════════════════════════════════════════════════════

class GoalStatus(Enum):
    PROPOSED = auto()
    ACCEPTED = auto()
    ACTIVE = auto()
    COMPLETED = auto()
    FAILED = auto()
    ABANDONED = auto()


@dataclass
class GoalContract:
    """A first-class goal in NRSI with acceptance criteria.

    Goals are trust-typed: the system commits to goals at a trust level,
    meaning the goal itself has been validated as sensible.
    """

    goal_id: str
    description: str
    status: GoalStatus = GoalStatus.PROPOSED
    acceptance_criteria: List[str] = field(default_factory=list)
    subgoals: List[str] = field(default_factory=list)
    evidence_required: List[str] = field(default_factory=list)
    trust_level: TrustLevel = TrustLevel.RAW
    cognitive_origin: Optional[CognitiveOrigin] = None
    created_at: float = field(default_factory=time.time)
    history: List[Tuple[float, GoalStatus, str]] = field(default_factory=list)

    def accept(self, reason: str) -> None:
        """Accept a proposed goal after validation."""
        if self.status != GoalStatus.PROPOSED:
            raise ValueError(
                f"Can only accept PROPOSED goals, current: {self.status.name}"
            )
        now = time.time()
        self.history.append((now, GoalStatus.ACCEPTED, reason))
        self.status = GoalStatus.ACCEPTED

    def activate(self) -> None:
        """Transition an accepted goal to active processing."""
        if self.status != GoalStatus.ACCEPTED:
            raise ValueError(
                f"Can only activate ACCEPTED goals, current: {self.status.name}"
            )
        now = time.time()
        self.history.append((now, GoalStatus.ACTIVE, "activated"))
        self.status = GoalStatus.ACTIVE

    def complete(self, evidence: List[str]) -> None:
        """Complete an active goal with supporting evidence."""
        if self.status != GoalStatus.ACTIVE:
            raise ValueError(
                f"Can only complete ACTIVE goals, current: {self.status.name}"
            )
        missing = [c for c in self.evidence_required if c not in evidence]
        if missing:
            raise ValueError(f"Missing required evidence: {missing}")
        now = time.time()
        self.history.append(
            (now, GoalStatus.COMPLETED, f"evidence: {', '.join(evidence)}")
        )
        self.status = GoalStatus.COMPLETED

    def fail(self, reason: str) -> None:
        """Mark an active goal as failed."""
        if self.status not in (GoalStatus.ACTIVE, GoalStatus.ACCEPTED):
            raise ValueError(
                f"Can only fail ACTIVE/ACCEPTED goals, current: {self.status.name}"
            )
        now = time.time()
        self.history.append((now, GoalStatus.FAILED, reason))
        self.status = GoalStatus.FAILED

    def abandon(self, reason: str) -> None:
        """Abandon a goal that is no longer relevant."""
        if self.status in (GoalStatus.COMPLETED, GoalStatus.FAILED):
            raise ValueError(f"Cannot abandon {self.status.name} goal")
        now = time.time()
        self.history.append((now, GoalStatus.ABANDONED, reason))
        self.status = GoalStatus.ABANDONED

    def __repr__(self) -> str:
        return (
            f"GoalContract({self.goal_id!r}, "
            f"status={self.status.name}, "
            f"trust={self.trust_level.name})"
        )


@dataclass
class PlanStep:
    """A single step in a PlanContract."""

    step_id: str
    description: str
    assigned_lobe: Optional[LobeType] = None
    epistemic_type: EpistemicType = EpistemicType.OBSERVATIONAL
    status: GoalStatus = GoalStatus.PROPOSED
    result: Optional[NRSIData] = None
    depends_on: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        lobe = self.assigned_lobe.value if self.assigned_lobe else "none"
        return (
            f"PlanStep({self.step_id!r}, "
            f"status={self.status.name}, lobe={lobe})"
        )


@dataclass
class PlanContract:
    """A first-class plan in NRSI.

    Plans decompose goals into steps, each step assigned to a lobe.
    The plan itself is trust-typed: a VALIDATED plan has been checked
    for coherence and feasibility.
    """

    plan_id: str
    goal_id: str
    steps: List[PlanStep] = field(default_factory=list)
    status: GoalStatus = GoalStatus.PROPOSED
    trust_level: TrustLevel = TrustLevel.RAW
    coherence_score: float = 0.0
    estimated_confidence: float = 0.0
    actual_confidence: float = 0.0
    history: List[Tuple[float, GoalStatus, str]] = field(default_factory=list)

    def validate_coherence(self) -> bool:
        """Check that step dependencies form a valid DAG and are satisfiable.

        Returns True if coherent, False if there are dangling refs or cycles.
        """
        step_ids = {s.step_id for s in self.steps}
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    self.coherence_score = 0.0
                    return False

        # Cycle detection via topological-sort attempt (Kahn's algorithm)
        in_degree: Dict[str, int] = {s.step_id: 0 for s in self.steps}
        adj: Dict[str, List[str]] = {s.step_id: [] for s in self.steps}
        for step in self.steps:
            for dep in step.depends_on:
                adj[dep].append(step.step_id)
                in_degree[step.step_id] += 1

        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        visited = 0
        while queue:
            node = queue.pop(0)
            visited += 1
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited != len(self.steps):
            self.coherence_score = 0.0
            return False

        self.coherence_score = 1.0
        return True

    def execute_step(self, step_idx: int, result: NRSIData) -> None:
        """Record the result of executing a plan step."""
        if step_idx < 0 or step_idx >= len(self.steps):
            raise IndexError(f"Step index {step_idx} out of range")
        step = self.steps[step_idx]
        step.result = result
        step.status = GoalStatus.COMPLETED
        now = time.time()
        self.history.append(
            (now, GoalStatus.ACTIVE, f"step {step.step_id} completed")
        )

        completed = sum(1 for s in self.steps if s.status == GoalStatus.COMPLETED)
        self.actual_confidence = completed / len(self.steps) if self.steps else 0.0

        if completed == len(self.steps):
            self.status = GoalStatus.COMPLETED
            self.history.append((now, GoalStatus.COMPLETED, "all steps done"))

    def __repr__(self) -> str:
        done = sum(1 for s in self.steps if s.status == GoalStatus.COMPLETED)
        return (
            f"PlanContract({self.plan_id!r}, "
            f"goal={self.goal_id!r}, "
            f"steps={done}/{len(self.steps)})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. EpistemicNRSIData — Extended NRSIData with full epistemic metadata
# ═══════════════════════════════════════════════════════════════════════════════

class EpistemicNRSIData(NRSIData[T]):
    """NRSIData with full epistemic tracking.

    Drop-in replacement that adds:
      cognitive_origin — which lobe/engine/mode produced this
      epistemic_type  — how we know it (deductive, inductive, …)
      temporal_scope  — when is this valid
      claim_status    — lifecycle stage

    Fully compatible with existing NRSIData consumers — they see
    the same trust/confidence/provenance interface.
    """

    __slots__ = (
        "_epistemic_type",
        "_cognitive_origin",
        "_temporal_scope",
        "_claim_status",
    )

    def __init__(
        self,
        value: T,
        trust_level: TrustLevel,
        confidence: float,
        epistemic_type: EpistemicType = EpistemicType.OBSERVATIONAL,
        cognitive_origin: Optional[CognitiveOrigin] = None,
        temporal_scope: Optional[TemporalScope] = None,
        claim_status: ClaimStatus = ClaimStatus.EXTRACTED,
        provenance: Optional[List[ProvenanceEntry]] = None,
        metadata: Optional[dict] = None,
    ):
        super().__init__(
            value=value,
            trust_level=trust_level,
            confidence=confidence,
            provenance=provenance,
            metadata=metadata,
        )
        self._epistemic_type = epistemic_type
        self._cognitive_origin = cognitive_origin
        self._temporal_scope = temporal_scope
        self._claim_status = claim_status

    # ── properties ────────────────────────────────────────────────────

    @property
    def epistemic_type(self) -> EpistemicType:
        return self._epistemic_type

    @property
    def cognitive_origin(self) -> Optional[CognitiveOrigin]:
        return self._cognitive_origin

    @property
    def temporal_scope(self) -> Optional[TemporalScope]:
        return self._temporal_scope

    @property
    def claim_status(self) -> ClaimStatus:
        return self._claim_status

    @property
    def effective_confidence(self) -> float:
        """Confidence adjusted for epistemic type and temporal staleness.

        A deductive proof with 0.95 confidence stays at 0.95.
        A speculative claim with 0.95 confidence drops to 0.95 * 0.2 = 0.19.
        An old claim's confidence decays based on staleness.
        """
        if self._cognitive_origin is not None:
            factor = self._cognitive_origin.strength_factor()
        else:
            factor = _EPISTEMIC_STRENGTH.get(self._epistemic_type, 0.5)

        base = self._confidence * factor

        if self._temporal_scope is not None:
            staleness = self._temporal_scope.staleness_factor()
            base *= (1.0 - staleness)

        return max(0.0, min(1.0, base))

    # ── trust operations (preserve epistemic metadata) ────────────────

    def _clone_with(self, **overrides: Any) -> EpistemicNRSIData[T]:
        """Internal: build a copy with field overrides."""
        result = EpistemicNRSIData.__new__(EpistemicNRSIData)
        result._value = overrides.get("value", self._value)
        result._trust_level = overrides.get("trust_level", self._trust_level)
        result._confidence = overrides.get("confidence", self._confidence)
        result._provenance = overrides.get("provenance", list(self._provenance))
        result._id = overrides.get("id", self._id)
        result._created_at = overrides.get("created_at", self._created_at)
        result._metadata = overrides.get("metadata", dict(self._metadata))
        result._epistemic_type = overrides.get("epistemic_type", self._epistemic_type)
        result._cognitive_origin = overrides.get(
            "cognitive_origin", self._cognitive_origin
        )
        result._temporal_scope = overrides.get(
            "temporal_scope", self._temporal_scope
        )
        result._claim_status = overrides.get("claim_status", self._claim_status)
        return result

    def elevate(
        self,
        to_level: TrustLevel,
        confidence: float,
        gate_name: str,
        reason: Optional[str] = None,
    ) -> EpistemicNRSIData[T]:
        """Elevate trust level while preserving epistemic metadata."""
        if to_level <= self._trust_level:
            raise TrustError(
                expected_trust=to_level.name,
                actual_trust=self._trust_level.name,
                operation="elevate",
                suggestion=(
                    f"Data is already at {self._trust_level.name}. "
                    f"Cannot elevate to same or lower level {to_level.name}."
                ),
            )

        confidence = Confidence.validate(confidence)
        new_prov = list(self._provenance)
        new_prov.append(ProvenanceEntry(
            timestamp=time.time(),
            action="elevated",
            gate_name=gate_name,
            from_trust=self._trust_level,
            to_trust=to_level,
            confidence=confidence,
            reason=reason,
        ))

        return self._clone_with(
            trust_level=to_level,
            confidence=Confidence.combine(self._confidence, confidence),
            provenance=new_prov,
        )

    def downgrade(
        self,
        to_level: TrustLevel,
        reason: str,
        actor: str,
    ) -> EpistemicNRSIData[T]:
        """Downgrade trust while preserving epistemic metadata."""
        if to_level >= self._trust_level:
            raise TrustError(
                expected_trust=to_level.name,
                actual_trust=self._trust_level.name,
                operation="downgrade",
                suggestion="Downgrade must go to a lower trust level.",
            )

        new_prov = list(self._provenance)
        new_prov.append(ProvenanceEntry(
            timestamp=time.time(),
            action="downgraded",
            from_trust=self._trust_level,
            to_trust=to_level,
            confidence=self._confidence,
            reason=reason,
            actor=actor,
        ))

        return self._clone_with(
            trust_level=to_level,
            provenance=new_prov,
        )

    # ── representation ────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"{self._trust_level.name.lower()}"
            f"[{type(self._value).__name__}]"
            f"(conf={self._confidence:.2f}, "
            f"eff={self.effective_confidence:.2f}, "
            f"type={self._epistemic_type.name.lower()})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Constructor functions for extended epistemic types
# ═══════════════════════════════════════════════════════════════════════════════

def deductive(
    value: T,
    confidence: float,
    proof_chain: Optional[List[str]] = None,
    gate_name: str = "logic_gate",
    **kw: Any,
) -> EpistemicNRSIData[T]:
    """Create deductively proven knowledge."""
    origin = CognitiveOrigin(
        lobe=LobeType.LOGICAL,
        epistemic_type=EpistemicType.DEDUCTIVE,
        mode_spectrum="DETERMINISTIC",
        engine_name=kw.pop("engine_name", "logic_engine"),
        method_name=kw.pop("method_name", "proof"),
        evidence_count=len(proof_chain) if proof_chain else 0,
    )
    return EpistemicNRSIData(
        value=value,
        trust_level=TrustLevel.VALIDATED,
        confidence=confidence,
        epistemic_type=EpistemicType.DEDUCTIVE,
        cognitive_origin=origin,
        temporal_scope=TemporalScope(validity=TemporalValidity.ETERNAL),
        claim_status=ClaimStatus.VALIDATED,
        metadata={
            "gate_name": gate_name,
            "proof_chain": proof_chain or [],
            **kw,
        },
    )


def computed(
    value: T,
    confidence: float = 0.99,
    gate_name: str = "computation_gate",
    **kw: Any,
) -> EpistemicNRSIData[T]:
    """Create computationally derived knowledge."""
    origin = CognitiveOrigin(
        lobe=LobeType.MATHEMATICAL,
        epistemic_type=EpistemicType.COMPUTATIONAL,
        mode_spectrum="DETERMINISTIC",
        engine_name=kw.pop("engine_name", "compute_engine"),
        method_name=kw.pop("method_name", "evaluate"),
    )
    return EpistemicNRSIData(
        value=value,
        trust_level=TrustLevel.VALIDATED,
        confidence=confidence,
        epistemic_type=EpistemicType.COMPUTATIONAL,
        cognitive_origin=origin,
        temporal_scope=TemporalScope(validity=TemporalValidity.ETERNAL),
        claim_status=ClaimStatus.VALIDATED,
        metadata={"gate_name": gate_name, **kw},
    )


def causal(
    value: T,
    confidence: float,
    chain_length: int = 0,
    gate_name: str = "causal_gate",
    **kw: Any,
) -> EpistemicNRSIData[T]:
    """Create causally established knowledge."""
    origin = CognitiveOrigin(
        lobe=LobeType.TEMPORAL,
        epistemic_type=EpistemicType.CAUSAL,
        mode_spectrum="DETERMINISTIC",
        engine_name=kw.pop("engine_name", "causal_reasoner"),
        method_name=kw.pop("method_name", "trace_chain"),
        reasoning_depth=chain_length,
    )
    return EpistemicNRSIData(
        value=value,
        trust_level=TrustLevel.VALIDATED,
        confidence=confidence,
        epistemic_type=EpistemicType.CAUSAL,
        cognitive_origin=origin,
        claim_status=ClaimStatus.VALIDATED,
        metadata={"gate_name": gate_name, "chain_length": chain_length, **kw},
    )


def analogical(
    value: T,
    confidence: float,
    source_domain: str = "",
    target_domain: str = "",
    gate_name: str = "analogy_gate",
    **kw: Any,
) -> EpistemicNRSIData[T]:
    """Create analogically transferred knowledge."""
    origin = CognitiveOrigin(
        lobe=LobeType.CREATIVE,
        epistemic_type=EpistemicType.ANALOGICAL,
        mode_spectrum="CREATIVE",
        engine_name=kw.pop("engine_name", "analogy_engine"),
        method_name=kw.pop("method_name", "cross_map"),
    )
    return EpistemicNRSIData(
        value=value,
        trust_level=TrustLevel.RAW,
        confidence=confidence,
        epistemic_type=EpistemicType.ANALOGICAL,
        cognitive_origin=origin,
        claim_status=ClaimStatus.EXTRACTED,
        metadata={
            "gate_name": gate_name,
            "source_domain": source_domain,
            "target_domain": target_domain,
            **kw,
        },
    )


def speculative(
    value: T,
    confidence: float,
    hypothesis: str = "",
    gate_name: str = "speculation",
    **kw: Any,
) -> EpistemicNRSIData[T]:
    """Create speculative/counterfactual knowledge."""
    origin = CognitiveOrigin(
        lobe=LobeType.CREATIVE,
        epistemic_type=EpistemicType.SPECULATIVE,
        mode_spectrum="CREATIVE",
        engine_name=kw.pop("engine_name", "speculation_engine"),
        method_name=kw.pop("method_name", "hypothesize"),
    )
    return EpistemicNRSIData(
        value=value,
        trust_level=TrustLevel.RAW,
        confidence=confidence,
        epistemic_type=EpistemicType.SPECULATIVE,
        cognitive_origin=origin,
        claim_status=ClaimStatus.EXTRACTED,
        metadata={"gate_name": gate_name, "hypothesis": hypothesis, **kw},
    )


def observed(
    value: T,
    confidence: float,
    source: str = "",
    gate_name: str = "observation_gate",
    **kw: Any,
) -> EpistemicNRSIData[T]:
    """Create observational knowledge (directly observed / retrieved)."""
    origin = CognitiveOrigin(
        lobe=LobeType.LINGUISTIC,
        epistemic_type=EpistemicType.OBSERVATIONAL,
        engine_name=kw.pop("engine_name", "retrieval"),
        method_name=kw.pop("method_name", "observe"),
    )
    return EpistemicNRSIData(
        value=value,
        trust_level=TrustLevel.RAW,
        confidence=confidence,
        epistemic_type=EpistemicType.OBSERVATIONAL,
        cognitive_origin=origin,
        temporal_scope=TemporalScope(
            validity=TemporalValidity.CURRENT,
            last_verified=time.time(),
        ),
        claim_status=ClaimStatus.EXTRACTED,
        metadata={"gate_name": gate_name, "source": source, **kw},
    )


def inductive(
    value: T,
    confidence: float,
    sample_count: int = 0,
    gate_name: str = "induction_gate",
    **kw: Any,
) -> EpistemicNRSIData[T]:
    """Create inductively generalized knowledge."""
    origin = CognitiveOrigin(
        lobe=LobeType.LOGICAL,
        epistemic_type=EpistemicType.INDUCTIVE,
        mode_spectrum="DETERMINISTIC",
        engine_name=kw.pop("engine_name", "induction_engine"),
        method_name=kw.pop("method_name", "generalize"),
        evidence_count=sample_count,
    )
    return EpistemicNRSIData(
        value=value,
        trust_level=TrustLevel.RAW,
        confidence=confidence,
        epistemic_type=EpistemicType.INDUCTIVE,
        cognitive_origin=origin,
        claim_status=ClaimStatus.EXTRACTED,
        metadata={
            "gate_name": gate_name,
            "sample_count": sample_count,
            **kw,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Module exports
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "EpistemicType",
    "TemporalValidity",
    "ClaimStatus",
    "GoalStatus",
    # Dataclasses
    "CognitiveOrigin",
    "ReasoningProvenance",
    "TemporalScope",
    "ClaimRecord",
    "GoalContract",
    "PlanContract",
    "PlanStep",
    # Operators
    "EpistemicOps",
    # Extended NRSIData
    "EpistemicNRSIData",
    # Constructors
    "deductive",
    "computed",
    "causal",
    "analogical",
    "speculative",
    "observed",
    "inductive",
]
