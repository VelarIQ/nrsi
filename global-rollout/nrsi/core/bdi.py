"""NRSI BDI Architecture — Beliefs, Desires, Intentions.

The Belief-Desire-Intention model separates three cognitive attitudes:

  Belief     — What the Impulse thinks is true (doxastic commitment)
  Desire     — What the Impulse wants (motivational state, soft objective)
  Intention  — What the Impulse is committed to doing (persistent plan commitment)

Key properties:
  - Beliefs can be uncertain; desires have utility; intentions have commitment strength
  - Intentions persist across replanning unless explicitly dropped
  - Desires can conflict; the deliberation process resolves them
  - Beliefs constrain which intentions are rational
  - Intentions generate subgoals that become new desires

This goes beyond GoalContract by adding motivational structure.

Patent-covered: NRSI BDI Cognitive Architecture, VelarIQ.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
)

from nrsi.core.errors import NRSIError


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Belief — Doxastic commitment
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Belief:
    """What the agent takes to be true.

    Attributes:
        belief_id:      Unique identifier.
        content:        Propositional content (human-readable).
        confidence:     Subjective probability, 0.0-1.0.
        epistemic_type: Origin kind ("deductive", "observational", …).
        domain:         Knowledge domain this belief belongs to.
        grounding:      Evidence or justification strings.
        believed_since: Unix timestamp when the belief was adopted.
    """

    belief_id: str
    content: str
    confidence: float = 0.5
    epistemic_type: str = "observational"
    domain: str = ""
    grounding: List[str] = field(default_factory=list)
    believed_since: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.belief_id:
            raise ValueError("belief_id must be non-empty")
        self.confidence = float(self.confidence)
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )

    @property
    def is_confident(self) -> bool:
        """True when confidence is above the rationality floor (0.5)."""
        return self.confidence > 0.5

    def __repr__(self) -> str:
        return (
            f"Belief({self.belief_id!r}, "
            f"conf={self.confidence:.2f}, "
            f"domain={self.domain!r})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Desire — Motivational state
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Desire:
    """What the agent wants to achieve.

    Desires are *soft* objectives: having conflicting desires is allowed.
    The deliberation process decides which desires become intentions.

    Attributes:
        desire_id:      Unique identifier.
        description:    Human-readable goal description.
        utility:        Expected value if achieved, 0.0-1.0.
        priority:       Ordinal priority (lower = more important).
        achievable:     True/False/None (unknown).
        conflicts_with: Desire IDs that are mutually exclusive.
        persistent:     If True, desire survives deliberation even when
                        not immediately selected.
    """

    desire_id: str
    description: str
    utility: float = 0.5
    priority: int = 0
    achievable: Optional[bool] = None
    conflicts_with: List[str] = field(default_factory=list)
    persistent: bool = True

    def __post_init__(self) -> None:
        if not self.desire_id:
            raise ValueError("desire_id must be non-empty")
        self.utility = float(self.utility)
        if self.utility < 0.0 or self.utility > 1.0:
            raise ValueError(
                f"utility must be in [0.0, 1.0], got {self.utility}"
            )

    @property
    def is_achievable(self) -> bool:
        """True when achievability is confirmed or unknown (optimistic)."""
        return self.achievable is not False

    @property
    def rank_score(self) -> float:
        """Composite score for deliberation ranking (higher = better)."""
        priority_factor = 1.0 / (1.0 + self.priority)
        return self.utility * priority_factor

    def __repr__(self) -> str:
        return (
            f"Desire({self.desire_id!r}, "
            f"utility={self.utility:.2f}, pri={self.priority})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Intention — Persistent plan commitment
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Intention:
    """What the agent is committed to doing.

    Intentions differ from desires in two critical ways:
      1. They carry a *commitment* — the agent will pursue them across
         replanning cycles unless explicitly dropped.
      2. They have a *belief basis* — a set of beliefs that must remain
         held for the intention to stay rational.

    Attributes:
        intention_id:        Unique identifier.
        description:         Human-readable action description.
        commitment_strength: How strongly committed, 0.0-1.0.
        plan_id:             ID linking to a PlanContract (may be empty).
        belief_basis:        Belief IDs that justify this intention.
        adopted_at:          Unix timestamp of adoption.
        drop_conditions:     Human-readable conditions that invalidate
                             the intention.
    """

    intention_id: str
    description: str
    commitment_strength: float = 0.7
    plan_id: str = ""
    belief_basis: List[str] = field(default_factory=list)
    adopted_at: float = field(default_factory=time.time)
    drop_conditions: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.intention_id:
            raise ValueError("intention_id must be non-empty")
        self.commitment_strength = float(self.commitment_strength)
        if self.commitment_strength < 0.0 or self.commitment_strength > 1.0:
            raise ValueError(
                f"commitment_strength must be in [0.0, 1.0], "
                f"got {self.commitment_strength}"
            )

    def should_drop(self, beliefs: Dict[str, Belief]) -> bool:
        """True when supporting beliefs no longer sustain this intention.

        An intention should be dropped when any of its required beliefs
        either no longer exists or has dropped below the confidence floor.
        """
        if not self.belief_basis:
            return False

        confidence_floor = 0.3
        for bid in self.belief_basis:
            belief = beliefs.get(bid)
            if belief is None:
                return True
            if belief.confidence < confidence_floor:
                return True
        return False

    def reconsider(self, new_beliefs: Dict[str, Belief]) -> bool:
        """True if new evidence warrants reconsidering this intention.

        Reconsideration is triggered when:
          - Any belief in the basis has changed significantly
          - The average basis confidence drops below commitment strength
        """
        if not self.belief_basis:
            return False

        present = [
            new_beliefs[bid]
            for bid in self.belief_basis
            if bid in new_beliefs
        ]
        if len(present) < len(self.belief_basis):
            return True

        avg_confidence = sum(b.confidence for b in present) / len(present)
        return avg_confidence < self.commitment_strength

    def __repr__(self) -> str:
        return (
            f"Intention({self.intention_id!r}, "
            f"strength={self.commitment_strength:.2f}, "
            f"plan={self.plan_id!r})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DeliberationResult — Output of the deliberation process
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DeliberationResult:
    """What the deliberation cycle decided.

    Attributes:
        chosen_intentions:       Newly committed intentions.
        dropped_desires:         Desires that were discarded (conflicts,
                                 unachievable, low rank).
        reason:                  Human-readable summary of the decision.
        alternatives_considered: How many desire combinations were evaluated.
    """

    chosen_intentions: Tuple[Intention, ...]
    dropped_desires: Tuple[Desire, ...]
    reason: str
    alternatives_considered: int

    def __repr__(self) -> str:
        return (
            f"DeliberationResult(chosen={len(self.chosen_intentions)}, "
            f"dropped={len(self.dropped_desires)}, "
            f"alts={self.alternatives_considered})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. RationalityConstraint — Consistency checks for BDI state
# ═══════════════════════════════════════════════════════════════════════════════

class RationalityViolation(Enum):
    """Categories of BDI rationality failure."""

    INTENTION_UNSUPPORTED = auto()
    DESIRE_CONTRADICTION = auto()
    INTENTION_CONFLICT = auto()
    BELIEF_MISSING = auto()


@dataclass(frozen=True)
class RationalityReport:
    """Result of a rationality check."""

    violations: Tuple[Tuple[RationalityViolation, str], ...]
    is_rational: bool

    def __repr__(self) -> str:
        return (
            f"RationalityReport(rational={self.is_rational}, "
            f"violations={len(self.violations)})"
        )


class RationalityConstraint:
    """Enforces BDI rationality invariants.

    Three constraints are checked:
      1. Intentions must be believed possible — every intention needs at
         least one supporting belief with adequate confidence.
      2. Desires should not be contradictory — mutual conflicts among
         active desires are flagged (but not forbidden).
      3. Committed intentions take priority — new desires cannot override
         a standing intention unless the intention is explicitly dropped.
    """

    BELIEF_CONFIDENCE_FLOOR: float = 0.3

    @staticmethod
    def check(
        beliefs: Dict[str, Belief],
        desires: Dict[str, Desire],
        intentions: Dict[str, Intention],
    ) -> RationalityReport:
        """Full rationality audit of the current BDI state."""
        violations: List[Tuple[RationalityViolation, str]] = []

        for iid, intention in intentions.items():
            if not intention.belief_basis:
                continue
            supported = False
            for bid in intention.belief_basis:
                belief = beliefs.get(bid)
                if (
                    belief is not None
                    and belief.confidence >= RationalityConstraint.BELIEF_CONFIDENCE_FLOOR
                ):
                    supported = True
                    break
            if not supported:
                violations.append((
                    RationalityViolation.INTENTION_UNSUPPORTED,
                    f"Intention {iid!r} has no supporting belief above "
                    f"confidence floor "
                    f"({RationalityConstraint.BELIEF_CONFIDENCE_FLOOR})",
                ))

        desire_ids = set(desires.keys())
        checked: Set[Tuple[str, str]] = set()
        for did, desire in desires.items():
            for cid in desire.conflicts_with:
                if cid in desire_ids:
                    pair = (min(did, cid), max(did, cid))
                    if pair not in checked:
                        checked.add(pair)
                        violations.append((
                            RationalityViolation.DESIRE_CONTRADICTION,
                            f"Desires {did!r} and {cid!r} conflict",
                        ))

        intention_domains: Dict[str, List[str]] = {}
        for iid, intention in intentions.items():
            for bid in intention.belief_basis:
                belief = beliefs.get(bid)
                if belief and belief.domain:
                    intention_domains.setdefault(belief.domain, []).append(iid)
        for domain, iids in intention_domains.items():
            if len(iids) > 1:
                violations.append((
                    RationalityViolation.INTENTION_CONFLICT,
                    f"Multiple intentions in domain {domain!r}: "
                    f"{', '.join(iids)}",
                ))

        return RationalityReport(
            violations=tuple(violations),
            is_rational=len(violations) == 0,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. BDIState — Main cognitive state container
# ═══════════════════════════════════════════════════════════════════════════════

class BDIState:
    """Unified Belief-Desire-Intention state with deliberation.

    Provides the full BDI cycle:
      1. Beliefs are updated from perception / inference.
      2. Desires are adopted or dropped based on beliefs.
      3. Deliberation selects a non-conflicting intention set.
      4. Intentions persist across cycles until dropped.
      5. Means-end reasoning decomposes intentions into sub-desires.
    """

    __slots__ = ("_beliefs", "_desires", "_intentions", "_history")

    def __init__(self) -> None:
        self._beliefs: Dict[str, Belief] = {}
        self._desires: Dict[str, Desire] = {}
        self._intentions: Dict[str, Intention] = {}
        self._history: List[Tuple[float, str, str]] = []

    # ── accessors ──────────────────────────────────────────────────────

    @property
    def beliefs(self) -> Dict[str, Belief]:
        return dict(self._beliefs)

    @property
    def desires(self) -> Dict[str, Desire]:
        return dict(self._desires)

    @property
    def intentions(self) -> Dict[str, Intention]:
        return dict(self._intentions)

    # ── belief management ──────────────────────────────────────────────

    def update_belief(self, belief: Belief) -> None:
        """Add or update a belief, then filter intentions that may collapse."""
        self._beliefs[belief.belief_id] = belief
        self._log("belief_updated", belief.belief_id)

    def remove_belief(self, belief_id: str, reason: str = "") -> None:
        """Remove a belief and auto-filter dependent intentions."""
        if belief_id in self._beliefs:
            del self._beliefs[belief_id]
            self._log("belief_removed", f"{belief_id}: {reason}")
            self.filter_intentions(self._beliefs)

    # ── desire management ──────────────────────────────────────────────

    def adopt_desire(self, desire: Desire) -> None:
        """Add a desire to the active set."""
        self._desires[desire.desire_id] = desire
        self._log("desire_adopted", desire.desire_id)

    def drop_desire(self, desire_id: str, reason: str = "") -> None:
        """Remove a desire."""
        if desire_id in self._desires:
            del self._desires[desire_id]
            self._log("desire_dropped", f"{desire_id}: {reason}")

    # ── intention management ───────────────────────────────────────────

    def commit_intention(self, intention: Intention) -> None:
        """Add an intention to the committed set."""
        self._intentions[intention.intention_id] = intention
        self._log("intention_committed", intention.intention_id)

    def drop_intention(self, intention_id: str, reason: str = "") -> None:
        """Explicitly drop a committed intention."""
        if intention_id in self._intentions:
            del self._intentions[intention_id]
            self._log("intention_dropped", f"{intention_id}: {reason}")

    # ── deliberation ───────────────────────────────────────────────────

    def deliberate(self) -> DeliberationResult:
        """Choose intentions from desires given current beliefs.

        Algorithm:
          1. Filter to achievable desires.
          2. Rank by composite score (utility × priority factor).
          3. Greedily select non-conflicting desires.
          4. Convert selections to intentions.
        """
        candidates = [
            d for d in self._desires.values() if d.is_achievable
        ]
        candidates.sort(key=lambda d: d.rank_score, reverse=True)

        selected: List[Desire] = []
        selected_ids: Set[str] = set()
        blocked_ids: Set[str] = set()
        dropped: List[Desire] = []
        alternatives = len(candidates)

        for desire in candidates:
            if desire.desire_id in blocked_ids:
                dropped.append(desire)
                continue

            if any(cid in selected_ids for cid in desire.conflicts_with):
                dropped.append(desire)
                continue

            selected.append(desire)
            selected_ids.add(desire.desire_id)

            for cid in desire.conflicts_with:
                blocked_ids.add(cid)

        for desire in candidates:
            if (
                desire.desire_id not in selected_ids
                and desire not in dropped
            ):
                dropped.append(desire)

        new_intentions: List[Intention] = []
        for desire in selected:
            if any(
                i.description == desire.description
                for i in self._intentions.values()
            ):
                continue

            intention = Intention(
                intention_id=f"int_{desire.desire_id}_{uuid.uuid4().hex[:8]}",
                description=desire.description,
                commitment_strength=desire.utility,
                belief_basis=self._relevant_beliefs(desire),
                adopted_at=time.time(),
            )
            self._intentions[intention.intention_id] = intention
            new_intentions.append(intention)

        reasons: List[str] = []
        if new_intentions:
            reasons.append(
                f"committed {len(new_intentions)} new intention(s)"
            )
        if dropped:
            reasons.append(f"dropped {len(dropped)} desire(s)")

        return DeliberationResult(
            chosen_intentions=tuple(new_intentions),
            dropped_desires=tuple(dropped),
            reason="; ".join(reasons) or "no changes",
            alternatives_considered=alternatives,
        )

    def _relevant_beliefs(self, desire: Desire) -> List[str]:
        """Find belief IDs relevant to a desire (same domain or keyword)."""
        result: List[str] = []
        desc_lower = desire.description.lower()
        for bid, belief in self._beliefs.items():
            if belief.domain and belief.domain.lower() in desc_lower:
                result.append(bid)
            elif any(
                word in belief.content.lower()
                for word in desc_lower.split()
                if len(word) > 3
            ):
                result.append(bid)
        return result

    # ── means-end reasoning ────────────────────────────────────────────

    def means_end_reasoning(self, intention: Intention) -> List[Desire]:
        """Decompose an intention into sub-goal desires.

        Default implementation generates one maintenance desire per
        supporting belief.  Override for domain-specific decomposition.
        """
        subgoals: List[Desire] = []
        for idx, bid in enumerate(intention.belief_basis):
            belief = self._beliefs.get(bid)
            if belief is None:
                continue
            subgoals.append(Desire(
                desire_id=f"{intention.intention_id}_sub_{idx}",
                description=f"Maintain belief: {belief.content}",
                utility=intention.commitment_strength * 0.8,
                priority=idx,
                achievable=True,
                persistent=False,
            ))
        return subgoals

    # ── intention filtering ────────────────────────────────────────────

    def filter_intentions(self, new_beliefs: Dict[str, Belief]) -> List[str]:
        """Drop intentions whose belief basis has collapsed.

        Returns a list of dropped intention IDs.
        """
        to_drop: List[str] = []
        for iid, intention in list(self._intentions.items()):
            if intention.should_drop(new_beliefs):
                to_drop.append(iid)
                del self._intentions[iid]
                self._log(
                    "intention_auto_dropped",
                    f"{iid}: belief basis collapsed",
                )
        return to_drop

    # ── consistency check ──────────────────────────────────────────────

    def consistency_check(self) -> RationalityReport:
        """Full rationality audit of current BDI state."""
        return RationalityConstraint.check(
            self._beliefs, self._desires, self._intentions,
        )

    # ── internal ───────────────────────────────────────────────────────

    def _log(self, action: str, detail: str) -> None:
        self._history.append((time.time(), action, detail))

    @property
    def history(self) -> List[Tuple[float, str, str]]:
        return list(self._history)

    # ── dunder ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"BDIState(beliefs={len(self._beliefs)}, "
            f"desires={len(self._desires)}, "
            f"intentions={len(self._intentions)})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Module Exports
# ═══════════════════════════════════════════════════════════════════════════════

BDIAgent = None  # backward-compat alias set below


class BDIImpulse:
    """Pipeline facade providing the ``update(query, domain, confidence)`` API
    that ``nrsi.core.nrs._process_inner`` expects.

    Each BDIImpulse is an autonomous reasoning unit (Impulse) within a
    Construct that maintains its own beliefs and desires.
    """

    def __init__(self, state: Optional[BDIState] = None) -> None:
        self._state = state or BDIState()
        self._interaction_count = 0

    def update(
        self,
        *,
        query: str,
        domain: str = "general",
        confidence: float = 0.5,
    ) -> None:
        """Integrate a pipeline interaction into the BDI state."""
        self._interaction_count += 1
        bid = f"b-{self._interaction_count}"

        belief = Belief(
            belief_id=bid,
            content=query[:256],
            confidence=max(0.0, min(1.0, confidence)),
            domain=domain,
            epistemic_type="observational",
        )
        self._state.update_belief(belief)

        if confidence < 0.5:
            did = f"d-{self._interaction_count}"
            desire = Desire(
                desire_id=did,
                description=f"Improve understanding of: {query[:64]}",
                utility=1.0 - confidence,
                domain=domain,
            )
            self._state.add_desire(desire)

    @property
    def state(self) -> BDIState:
        return self._state


__all__ = [
    "Belief",
    "Desire",
    "Intention",
    "DeliberationResult",
    "RationalityViolation",
    "RationalityReport",
    "RationalityConstraint",
    "BDIState",
    "BDIImpulse",
    "BDIAgent",  # backward-compat alias
]

BDIAgent = BDIImpulse  # backward-compat alias
