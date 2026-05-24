"""NRSI Affective Types — Emotional Influence on Reasoning.

Emotions aren't noise — they're adaptive signals that should influence
cognitive policy. These types make emotional state a first-class,
auditable part of reasoning.

  AffectiveState   — Current emotional state (valence, arousal, dominance)
  MoodPolicy       — Rules that change reasoning based on emotional state
  AffectGateRule   — "Under high threat, require CERTIFIED for medical claims"
  EmotionalContext  — Emotional metadata attached to NRSIData
  AffectiveShift   — Audit record of emotion-driven policy changes

Patent-covered: NRSI Affective Cognition System, VelarIQ.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

from nrsi.core.types import TrustLevel


# ═══════════════════════════════════════════════════════════════════════════════
# 1. AffectiveDimension — The five emotional axes
# ═══════════════════════════════════════════════════════════════════════════════

class AffectiveDimension(Enum):
    """Five-axis emotional model (extended PAD + stress + curiosity).

    Each dimension is a continuous float in [0.0, 1.0]:
      VALENCE    0.0 = maximally negative, 1.0 = maximally positive
      AROUSAL    0.0 = calm / torpid,      1.0 = excited / agitated
      DOMINANCE  0.0 = overwhelmed,        1.0 = fully in control
      STRESS     0.0 = relaxed,            1.0 = extreme stress
      CURIOSITY  0.0 = disinterest,        1.0 = fascination
    """

    VALENCE   = auto()
    AROUSAL   = auto()
    DOMINANCE = auto()
    STRESS    = auto()
    CURIOSITY = auto()


# ── Mood label lookup ─────────────────────────────────────────────────────────
# (valence_hi, arousal_hi, dominance_hi) → label
# Checked in order; first match wins.

_MOOD_RULES: List[Tuple[Optional[bool], Optional[bool], Optional[bool], float, float, str]] = [
    # (valence_hi?, arousal_hi?, dominance_hi?, stress_floor, curiosity_floor, label)
    (None,  None,  None,  0.8, 0.0, "overwhelmed"),
    (None,  None,  None,  0.6, 0.0, "stressed"),
    (None,  None,  None,  0.0, 0.8, "fascinated"),
    (None,  None,  None,  0.0, 0.6, "curious"),
    (False, True,  False, 0.0, 0.0, "anxious"),
    (False, True,  True,  0.0, 0.0, "angry"),
    (False, False, False, 0.0, 0.0, "depressed"),
    (False, False, True,  0.0, 0.0, "melancholic"),
    (True,  True,  True,  0.0, 0.0, "elated"),
    (True,  True,  False, 0.0, 0.0, "excited"),
    (True,  False, True,  0.0, 0.0, "content"),
    (True,  False, False, 0.0, 0.0, "calm"),
]


def _matches_mood_rule(
    valence: float,
    arousal: float,
    dominance: float,
    stress: float,
    curiosity: float,
    rule: Tuple[Optional[bool], Optional[bool], Optional[bool], float, float, str],
) -> bool:
    v_hi, a_hi, d_hi, stress_floor, curiosity_floor = rule[:5]
    if stress < stress_floor:
        return False
    if curiosity < curiosity_floor:
        return False
    mid = 0.5
    if v_hi is not None and (valence >= mid) != v_hi:
        return False
    if a_hi is not None and (arousal >= mid) != a_hi:
        return False
    if d_hi is not None and (dominance >= mid) != d_hi:
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 2. AffectiveState — Current emotional snapshot
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AffectiveState:
    """Five-float emotional state vector.

    All dimensions are clamped to [0.0, 1.0].  The default is a neutral
    midpoint across all axes.
    """

    valence: float = 0.5
    arousal: float = 0.5
    dominance: float = 0.5
    stress: float = 0.2
    curiosity: float = 0.5

    def __post_init__(self) -> None:
        self.valence = self._clamp(self.valence)
        self.arousal = self._clamp(self.arousal)
        self.dominance = self._clamp(self.dominance)
        self.stress = self._clamp(self.stress)
        self.curiosity = self._clamp(self.curiosity)

    @staticmethod
    def _clamp(v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    # ── dimensional access ─────────────────────────────────────────────

    def get_dimension(self, dim: AffectiveDimension) -> float:
        """Read a single dimension by enum."""
        return {
            AffectiveDimension.VALENCE: self.valence,
            AffectiveDimension.AROUSAL: self.arousal,
            AffectiveDimension.DOMINANCE: self.dominance,
            AffectiveDimension.STRESS: self.stress,
            AffectiveDimension.CURIOSITY: self.curiosity,
        }[dim]

    def set_dimension(self, dim: AffectiveDimension, value: float) -> AffectiveState:
        """Return a new state with one dimension changed."""
        values = self.as_dict()
        values[dim] = self._clamp(value)
        return AffectiveState(
            valence=values[AffectiveDimension.VALENCE],
            arousal=values[AffectiveDimension.AROUSAL],
            dominance=values[AffectiveDimension.DOMINANCE],
            stress=values[AffectiveDimension.STRESS],
            curiosity=values[AffectiveDimension.CURIOSITY],
        )

    def as_dict(self) -> Dict[AffectiveDimension, float]:
        """All dimensions as an ordered dict."""
        return {
            AffectiveDimension.VALENCE: self.valence,
            AffectiveDimension.AROUSAL: self.arousal,
            AffectiveDimension.DOMINANCE: self.dominance,
            AffectiveDimension.STRESS: self.stress,
            AffectiveDimension.CURIOSITY: self.curiosity,
        }

    # ── semantic queries ───────────────────────────────────────────────

    def is_positive(self) -> bool:
        """True when valence is above the midpoint."""
        return self.valence >= 0.5

    def is_high_arousal(self) -> bool:
        """True when arousal exceeds 0.6."""
        return self.arousal >= 0.6

    def is_stressed(self) -> bool:
        """True when stress exceeds 0.6."""
        return self.stress >= 0.6

    def is_curious(self) -> bool:
        """True when curiosity exceeds 0.6."""
        return self.curiosity >= 0.6

    def dominant_affect(self) -> AffectiveDimension:
        """The dimension with the largest deviation from neutral (0.5)."""
        dims = self.as_dict()
        return max(dims, key=lambda d: abs(dims[d] - 0.5))

    def mood_label(self) -> str:
        """Human-readable mood based on the current state vector."""
        for rule in _MOOD_RULES:
            if _matches_mood_rule(
                self.valence, self.arousal, self.dominance,
                self.stress, self.curiosity, rule,
            ):
                return rule[5]
        return "neutral"

    # ── arithmetic ─────────────────────────────────────────────────────

    def shift(
        self,
        valence_delta: float = 0.0,
        arousal_delta: float = 0.0,
        dominance_delta: float = 0.0,
        stress_delta: float = 0.0,
        curiosity_delta: float = 0.0,
    ) -> AffectiveState:
        """Return a new state with deltas applied (clamped)."""
        return AffectiveState(
            valence=self.valence + valence_delta,
            arousal=self.arousal + arousal_delta,
            dominance=self.dominance + dominance_delta,
            stress=self.stress + stress_delta,
            curiosity=self.curiosity + curiosity_delta,
        )

    def distance(self, other: AffectiveState) -> float:
        """Euclidean distance between two affective states."""
        d = self.as_dict()
        o = other.as_dict()
        return sum((d[k] - o[k]) ** 2 for k in d) ** 0.5

    # ── representation ─────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"AffectiveState(v={self.valence:.2f}, a={self.arousal:.2f}, "
            f"d={self.dominance:.2f}, s={self.stress:.2f}, "
            f"c={self.curiosity:.2f}, mood={self.mood_label()!r})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MoodPolicy — Affect-driven reasoning adjustments
# ═══════════════════════════════════════════════════════════════════════════════

_CONDITION_OPS: Dict[str, Callable[[float, float], bool]] = {
    "gt":  lambda v, t: v > t,
    "lt":  lambda v, t: v < t,
    "gte": lambda v, t: v >= t,
    "lte": lambda v, t: v <= t,
}

@dataclass
class MoodPolicy:
    """A rule that adjusts reasoning behaviour based on emotional state.

    Attributes:
        policy_id:    Unique identifier.
        conditions:   Per-dimension thresholds.  Each entry is
                      (dimension, operator, threshold) where operator is
                      one of "gt", "lt", "gte", "lte".
        actions:      Human-readable adjustments to apply when conditions
                      match (e.g. "require_higher_trust",
                      "boost_exploration", "suppress_speculation").
        domain_scope: Optional domain restriction (e.g. "medical").
        description:  Explanation for audit logs.
    """

    policy_id: str
    conditions: List[Tuple[AffectiveDimension, str, float]]
    actions: List[str]
    domain_scope: Optional[str] = None
    description: str = ""

    def evaluate(self, state: AffectiveState) -> bool:
        """True when all conditions are satisfied by *state*."""
        for dim, op, threshold in self.conditions:
            cmp = _CONDITION_OPS.get(op)
            if cmp is None:
                raise ValueError(f"Unknown operator {op!r}")
            if not cmp(state.get_dimension(dim), threshold):
                return False
        return True

    def __repr__(self) -> str:
        conds = ", ".join(
            f"{d.name.lower()}{o}{t:.2f}" for d, o, t in self.conditions
        )
        return f"MoodPolicy({self.policy_id!r}, [{conds}] → {self.actions})"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. AffectGateRule — Trust gating by emotional state
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AffectGateRule:
    """Raise the trust floor for a domain when an affect dimension fires.

    Example: "When arousal > 0.8, require CERTIFIED for medical claims."

    Attributes:
        rule_id:           Unique identifier.
        dimension:         Which affective axis to test.
        threshold:         The trigger value.
        above:             True → triggers when dimension > threshold;
                           False → triggers when dimension < threshold.
        trust_requirement: The minimum TrustLevel to enforce.
        domain:            Knowledge domain this gate applies to.
        explanation:       Audit-friendly reason.
    """

    rule_id: str
    dimension: AffectiveDimension
    threshold: float
    above: bool = True
    trust_requirement: TrustLevel = TrustLevel.CERTIFIED
    domain: str = ""
    explanation: str = ""

    @property
    def condition_label(self) -> str:
        op = ">" if self.above else "<"
        return f"{self.dimension.name.lower()} {op} {self.threshold:.2f}"

    def matches(self, state: AffectiveState) -> bool:
        """True when the affective condition is met."""
        value = state.get_dimension(self.dimension)
        if self.above:
            return value > self.threshold
        return value < self.threshold

    def __repr__(self) -> str:
        return (
            f"AffectGateRule({self.rule_id!r}, "
            f"{self.condition_label} → {self.trust_requirement.name}, "
            f"domain={self.domain!r})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. EmotionalContext — Affect metadata for NRSIData
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EmotionalContext:
    """Emotional metadata intended to be attached to NRSIData via its
    metadata dict.

    Attributes:
        affect_at_creation:            State when the data was produced.
        affect_influence_on_confidence: Multiplier applied to confidence
                                        due to emotional state (1.0 = none).
        affect_policies_applied:       IDs of MoodPolicy rules that fired.
    """

    affect_at_creation: AffectiveState
    affect_influence_on_confidence: float = 1.0
    affect_policies_applied: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage in NRSIData.metadata."""
        return {
            "affect_at_creation": {
                d.name: self.affect_at_creation.get_dimension(d)
                for d in AffectiveDimension
            },
            "affect_influence_on_confidence": self.affect_influence_on_confidence,
            "affect_policies_applied": list(self.affect_policies_applied),
        }

    def __repr__(self) -> str:
        return (
            f"EmotionalContext(mood={self.affect_at_creation.mood_label()!r}, "
            f"influence={self.affect_influence_on_confidence:.2f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. AffectiveShift — Audit record of state transitions
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AffectiveShift:
    """Immutable record of an emotional state transition.

    Attributes:
        old_state:            State before the shift.
        new_state:            State after the shift.
        trigger:              What caused the change (query, evidence, …).
        policy_changes_made:  Policies that fired as a result.
        timestamp:            Unix timestamp.
    """

    old_state: AffectiveState
    new_state: AffectiveState
    trigger: str
    policy_changes_made: Tuple[str, ...]
    timestamp: float

    @property
    def magnitude(self) -> float:
        """Euclidean distance of the shift."""
        return self.old_state.distance(self.new_state)

    def __repr__(self) -> str:
        return (
            f"AffectiveShift({self.old_state.mood_label()!r} → "
            f"{self.new_state.mood_label()!r}, "
            f"Δ={self.magnitude:.3f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. AffectiveMonitor — Live tracking, policy application, shift logging
# ═══════════════════════════════════════════════════════════════════════════════

class AffectiveMonitor:
    """Tracks affective state over time, applies mood policies, and
    records every shift for auditability.
    """

    __slots__ = ("_state", "_policies", "_gate_rules", "_shifts")

    def __init__(
        self,
        initial: Optional[AffectiveState] = None,
        policies: Optional[List[MoodPolicy]] = None,
        gate_rules: Optional[List[AffectGateRule]] = None,
    ) -> None:
        self._state: AffectiveState = initial or AffectiveState()
        self._policies: List[MoodPolicy] = list(policies or [])
        self._gate_rules: List[AffectGateRule] = list(gate_rules or [])
        self._shifts: List[AffectiveShift] = []

    # ── state updates ──────────────────────────────────────────────────

    def update(
        self,
        percept: str,
        *,
        valence_delta: float = 0.0,
        arousal_delta: float = 0.0,
        dominance_delta: float = 0.0,
        stress_delta: float = 0.0,
        curiosity_delta: float = 0.0,
    ) -> AffectiveShift:
        """Apply dimension deltas triggered by *percept* and log the shift."""
        old = self._state
        new = old.shift(
            valence_delta=valence_delta,
            arousal_delta=arousal_delta,
            dominance_delta=dominance_delta,
            stress_delta=stress_delta,
            curiosity_delta=curiosity_delta,
        )
        self._state = new

        fired = [
            p.policy_id for p in self._policies if p.evaluate(new)
        ]
        shift = AffectiveShift(
            old_state=old,
            new_state=new,
            trigger=percept,
            policy_changes_made=tuple(fired),
            timestamp=time.time(),
        )
        self._shifts.append(shift)
        return shift

    # ── policy queries ─────────────────────────────────────────────────

    def apply_policies(
        self, trust_level: TrustLevel, domain: str,
    ) -> List[str]:
        """Check all policies and gate rules; return required adjustments.

        Returns a list of human-readable adjustment strings.
        """
        adjustments: List[str] = []

        for policy in self._policies:
            if policy.domain_scope and policy.domain_scope != domain:
                continue
            if policy.evaluate(self._state):
                adjustments.extend(policy.actions)

        for rule in self._gate_rules:
            if rule.domain and rule.domain != domain:
                continue
            if rule.matches(self._state):
                if trust_level < rule.trust_requirement:
                    adjustments.append(
                        f"gate:{rule.rule_id} requires "
                        f"{rule.trust_requirement.name} "
                        f"(current={trust_level.name}) — "
                        f"{rule.explanation}"
                    )

        return adjustments

    def active_gate_rules(self) -> List[AffectGateRule]:
        """Gate rules whose affect condition is currently satisfied."""
        return [r for r in self._gate_rules if r.matches(self._state)]

    # ── accessors ──────────────────────────────────────────────────────

    def get_current_state(self) -> AffectiveState:
        """Snapshot of current affective state."""
        return self._state

    def history(self, n: int = 10) -> List[AffectiveShift]:
        """Last *n* affective shifts (most recent last)."""
        return list(self._shifts[-n:])

    def emotional_context(self) -> EmotionalContext:
        """Build an EmotionalContext for attachment to NRSIData."""
        fired = [p.policy_id for p in self._policies if p.evaluate(self._state)]
        influence = 1.0
        if self._state.is_stressed():
            influence *= 0.85
        if not self._state.is_positive():
            influence *= 0.90
        return EmotionalContext(
            affect_at_creation=self._state,
            affect_influence_on_confidence=influence,
            affect_policies_applied=fired,
        )

    # ── mutation ───────────────────────────────────────────────────────

    def add_policy(self, policy: MoodPolicy) -> None:
        """Register an additional mood policy."""
        self._policies.append(policy)

    def add_gate_rule(self, rule: AffectGateRule) -> None:
        """Register an additional gate rule."""
        self._gate_rules.append(rule)

    # ── dunder ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"AffectiveMonitor(state={self._state!r}, "
            f"policies={len(self._policies)}, "
            f"shifts={len(self._shifts)})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Module Exports
# ═══════════════════════════════════════════════════════════════════════════════

class AffectiveInfluencer:
    """Pipeline facade providing the ``evaluate(query, emotional_ctx=)`` API
    that ``nrsi.core.nrs._process_inner`` expects.

    Wraps an ``AffectiveMonitor`` and exposes a simplified return object.
    """

    def __init__(self, monitor: Optional[AffectiveMonitor] = None) -> None:
        self._monitor = monitor or AffectiveMonitor()

    def evaluate(
        self,
        query: str,
        *,
        emotional_ctx: Optional[Dict[str, Any]] = None,
    ) -> "AffectiveEvaluation":
        """Evaluate affective influence on the current query."""
        ctx = emotional_ctx or {}
        threat = ctx.get("threat_level", 0.0)
        valence_raw = ctx.get("valence", 0.5)

        valence_delta = 0.0
        if isinstance(valence_raw, (int, float)):
            valence_delta = (float(valence_raw) - 0.5) * 0.2
        stress_delta = min(0.3, float(threat) * 0.4) if threat else 0.0

        shift = self._monitor.update(
            percept=query[:128],
            valence_delta=valence_delta,
            stress_delta=stress_delta,
        )

        state = self._monitor.get_current_state()
        influence = 1.0
        if state.is_stressed():
            influence *= 0.85
        if not state.is_positive():
            influence *= 0.90

        return AffectiveEvaluation(
            valence=state.valence,
            arousal=state.arousal,
            influence_weight=influence,
            mood=state.mood_label(),
        )

    @property
    def monitor(self) -> AffectiveMonitor:
        return self._monitor


@dataclass
class AffectiveEvaluation:
    """Result from ``AffectiveInfluencer.evaluate()``."""
    valence: float = 0.5
    arousal: float = 0.5
    influence_weight: float = 1.0
    mood: str = "neutral"


__all__ = [
    "AffectiveDimension",
    "AffectiveState",
    "MoodPolicy",
    "AffectGateRule",
    "EmotionalContext",
    "AffectiveShift",
    "AffectiveMonitor",
    "AffectiveInfluencer",
    "AffectiveEvaluation",
]
