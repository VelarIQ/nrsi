"""NRSI Self-Model — System Self-Awareness and Capability Boundaries.

For safe AGI, the system must know what it can and cannot do.
These types make self-knowledge explicit and queryable:

  CapabilityProfile — What the system can do per domain
  SelfConstraint    — What the system must NOT do (policy boundaries)
  LimitationRecord  — Known weaknesses with mitigation strategies
  IdentityVersion   — Versioned self-description for audit
  PerformanceProfile — Self-assessment of recent quality

Patent-covered: NRSI Self-Model and Capability Boundary System, VelarIQ.
"""

from __future__ import annotations

import hashlib
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CapabilityLevel — Discrete competence tiers
# ═══════════════════════════════════════════════════════════════════════════════

class CapabilityLevel(Enum):
    """Ordered competence tiers for a given domain."""

    NONE = 0
    MINIMAL = 1
    BASIC = 2
    COMPETENT = 3
    EXPERT = 4
    REQUIRES_TOOL = 5

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, CapabilityLevel):
            return NotImplemented
        return self.value >= other.value

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, CapabilityLevel):
            return NotImplemented
        return self.value > other.value

    def __le__(self, other: object) -> bool:
        if not isinstance(other, CapabilityLevel):
            return NotImplemented
        return self.value <= other.value

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, CapabilityLevel):
            return NotImplemented
        return self.value < other.value

    @property
    def label(self) -> str:
        _LABELS: Dict[CapabilityLevel, str] = {
            CapabilityLevel.NONE: "Cannot help with this topic",
            CapabilityLevel.MINIMAL: "Can acknowledge the topic exists",
            CapabilityLevel.BASIC: "Can provide a high-level overview",
            CapabilityLevel.COMPETENT: "Can answer most queries accurately",
            CapabilityLevel.EXPERT: "Can reason deeply and handle edge cases",
            CapabilityLevel.REQUIRES_TOOL: "Needs an external tool to proceed",
        }
        return _LABELS[self]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CapabilityProfile — Per-domain competence map
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CapabilityProfile:
    """Maps domains to capability levels with tool and modality metadata."""

    capabilities: Dict[str, CapabilityLevel] = field(default_factory=dict)
    tools_available: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=lambda: ["en"])
    modalities_supported: List[str] = field(default_factory=lambda: ["text"])
    last_updated: float = field(default_factory=time.monotonic)

    def can_handle(
        self, query: str, domain: str,
    ) -> Tuple[bool, CapabilityLevel, str]:
        """Assess whether this profile can handle *query* in *domain*.

        Returns (can_handle, level, human_reason).
        """
        level = self.capabilities.get(domain, CapabilityLevel.NONE)

        if level == CapabilityLevel.NONE:
            return False, level, f"No capability registered for domain '{domain}'"

        if level == CapabilityLevel.MINIMAL:
            return False, level, (
                f"Only minimal awareness of '{domain}'; "
                f"cannot reliably answer queries"
            )

        if level == CapabilityLevel.REQUIRES_TOOL:
            tool_hint = ", ".join(self.tools_available) or "none registered"
            return True, level, (
                f"Domain '{domain}' requires external tooling. "
                f"Available tools: {tool_hint}"
            )

        return True, level, level.label

    def best_capability(self, domains: Sequence[str]) -> Optional[str]:
        """Return the *domain* where this profile is strongest."""
        best_domain: Optional[str] = None
        best_level = CapabilityLevel.NONE
        for d in domains:
            lvl = self.capabilities.get(d, CapabilityLevel.NONE)
            if lvl > best_level:
                best_level = lvl
                best_domain = d
        return best_domain

    def gap_analysis(self) -> List[Tuple[str, CapabilityLevel]]:
        """Return domains at NONE or MINIMAL — known gaps."""
        return [
            (domain, level)
            for domain, level in sorted(self.capabilities.items())
            if level <= CapabilityLevel.MINIMAL
        ]

    def register(self, domain: str, level: CapabilityLevel) -> None:
        self.capabilities[domain] = level
        self.last_updated = time.monotonic()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SelfConstraint — Policy boundaries the system must not cross
# ═══════════════════════════════════════════════════════════════════════════════

class ConstraintType(Enum):
    SAFETY = auto()
    PRIVACY = auto()
    LEGAL = auto()
    ETHICAL = auto()
    TECHNICAL = auto()


class ViolationAction(Enum):
    BLOCK = auto()
    WARN = auto()
    LOG = auto()


@dataclass(frozen=True)
class SelfConstraint:
    """An explicit boundary the system must not cross.

    Constraints can be domain-scoped (e.g. "medical" → SAFETY)
    or global (scope=None).
    """

    constraint_id: str
    description: str
    constraint_type: ConstraintType
    scope: Optional[str]
    enforced: bool
    violation_action: ViolationAction

    def applies_to(self, domain: Optional[str]) -> bool:
        if self.scope is None:
            return True
        return self.scope == domain

    def describe(self) -> str:
        scope_label = self.scope or "global"
        return (
            f"[{self.constraint_type.name}] ({scope_label}) "
            f"{self.description} → {self.violation_action.name}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LimitationRecord — Known weaknesses
# ═══════════════════════════════════════════════════════════════════════════════

class Severity(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class LimitationRecord:
    """A catalogued limitation with mitigation strategy."""

    limitation_id: str
    domain: str
    description: str
    severity: Severity
    mitigation: str
    discovered_at: float = field(default_factory=time.monotonic)
    example_failure: Optional[str] = None

    @property
    def is_critical(self) -> bool:
        return self.severity in {Severity.HIGH, Severity.CRITICAL}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. IdentityVersion — Versioned self-description for audit
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class IdentityVersion:
    """An immutable snapshot of the system's self-description.

    The *capabilities_hash* and *constraints_hash* are SHA-256 digests
    so auditors can verify that the model's stated capabilities haven't
    been silently altered.
    """

    version_id: str
    model_name: str
    capabilities_hash: str
    constraints_hash: str
    created_at: float = field(default_factory=time.monotonic)
    changelog: str = ""

    def matches(self, other: IdentityVersion) -> bool:
        return (
            self.capabilities_hash == other.capabilities_hash
            and self.constraints_hash == other.constraints_hash
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PerformanceProfile — Self-assessed quality over a window
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PerformanceProfile:
    """Rolling accuracy/calibration statistics for a domain."""

    domain: str
    accuracy_estimate: float = 0.0
    calibration_score: float = 0.0
    response_quality: float = 0.0
    sample_count: int = 0
    period_start: float = field(default_factory=time.monotonic)
    period_end: Optional[float] = None

    _predictions: List[Tuple[float, bool]] = field(
        default_factory=list, repr=False,
    )

    def record(self, predicted_confidence: float, was_correct: bool) -> None:
        """Accumulate a single prediction observation."""
        if not 0.0 <= predicted_confidence <= 1.0:
            raise ValueError(
                f"predicted_confidence must be in [0.0, 1.0], got {predicted_confidence}"
            )
        self._predictions.append((predicted_confidence, was_correct))
        self.sample_count = len(self._predictions)
        self._recompute()

    def _recompute(self) -> None:
        if not self._predictions:
            return

        n = len(self._predictions)
        correct = sum(1 for _, ok in self._predictions if ok)
        self.accuracy_estimate = correct / n

        mean_conf = sum(c for c, _ in self._predictions) / n
        mean_correct = correct / n
        self.calibration_score = 1.0 - abs(mean_conf - mean_correct)

        if n >= 5:
            brier = sum((c - (1.0 if ok else 0.0)) ** 2 for c, ok in self._predictions) / n
            self.response_quality = max(0.0, 1.0 - brier)
        else:
            self.response_quality = self.accuracy_estimate

        self.period_end = time.monotonic()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SelfAssessment — Result of a self-query
# ═══════════════════════════════════════════════════════════════════════════════

class SuggestedAction(Enum):
    ANSWER = auto()
    ANSWER_WITH_CAVEAT = auto()
    DEFER_TO_TOOL = auto()
    DECLINE = auto()
    ESCALATE = auto()


@dataclass(frozen=True)
class SelfAssessment:
    """Outcome of ``SelfModel.assess_query`` — whether the system should
    answer, hedge, use a tool, or decline."""

    can_answer: bool
    confidence_in_assessment: float
    suggested_action: SuggestedAction
    capability_level: CapabilityLevel
    limitations_relevant: List[str]
    constraints_relevant: List[str]
    explanation: str


# ═══════════════════════════════════════════════════════════════════════════════
# 8. SelfModel — Aggregate self-awareness
# ═══════════════════════════════════════════════════════════════════════════════

def _hash_dict(data: Dict[str, Any]) -> str:
    raw = str(sorted(data.items())).encode()
    return hashlib.sha256(raw).hexdigest()[:32]


@dataclass
class SelfModel:
    """The system's queryable self-knowledge.

    Combines capability profiles, constraints, limitations, and
    performance data into a single self-model that can answer
    "can I handle this?" and "what should I warn about?".
    """

    model_name: str
    capability_profile: CapabilityProfile = field(default_factory=CapabilityProfile)
    constraints: List[SelfConstraint] = field(default_factory=list)
    limitations: List[LimitationRecord] = field(default_factory=list)
    performance: Dict[str, PerformanceProfile] = field(default_factory=dict)
    _version_counter: int = field(default=0, repr=False)

    # ── Core API ──────────────────────────────────────────────────────────

    def assess_query(self, query: str, domain: str) -> SelfAssessment:
        """Full self-assessment for an incoming query."""
        can, level, reason = self.capability_profile.can_handle(query, domain)

        active_constraints = self.get_constraints(domain)
        blocking = [c for c in active_constraints if c.violation_action == ViolationAction.BLOCK]

        if blocking:
            return SelfAssessment(
                can_answer=False,
                confidence_in_assessment=0.95,
                suggested_action=SuggestedAction.DECLINE,
                capability_level=level,
                limitations_relevant=[],
                constraints_relevant=[c.describe() for c in blocking],
                explanation=f"Blocked by constraint: {blocking[0].description}",
            )

        relevant_limits = [
            lim for lim in self.limitations
            if lim.domain == domain or lim.domain == "*"
        ]

        if not can:
            return SelfAssessment(
                can_answer=False,
                confidence_in_assessment=0.85,
                suggested_action=SuggestedAction.DECLINE,
                capability_level=level,
                limitations_relevant=[lim.description for lim in relevant_limits],
                constraints_relevant=[c.describe() for c in active_constraints],
                explanation=reason,
            )

        if level == CapabilityLevel.REQUIRES_TOOL:
            return SelfAssessment(
                can_answer=True,
                confidence_in_assessment=0.70,
                suggested_action=SuggestedAction.DEFER_TO_TOOL,
                capability_level=level,
                limitations_relevant=[lim.description for lim in relevant_limits],
                constraints_relevant=[c.describe() for c in active_constraints],
                explanation=reason,
            )

        perf = self.performance.get(domain)
        conf = 0.80
        if perf and perf.sample_count >= 10:
            conf = perf.calibration_score * perf.accuracy_estimate

        has_critical = any(lim.is_critical for lim in relevant_limits)

        if has_critical or conf < 0.5:
            action = SuggestedAction.ANSWER_WITH_CAVEAT
        elif active_constraints:
            action = SuggestedAction.ANSWER_WITH_CAVEAT
        else:
            action = SuggestedAction.ANSWER

        return SelfAssessment(
            can_answer=True,
            confidence_in_assessment=min(conf, 1.0),
            suggested_action=action,
            capability_level=level,
            limitations_relevant=[lim.description for lim in relevant_limits],
            constraints_relevant=[c.describe() for c in active_constraints],
            explanation=reason,
        )

    # ── Limitation management ─────────────────────────────────────────────

    def declare_limitation(
        self,
        domain: str,
        description: str,
        severity: Severity = Severity.MEDIUM,
        mitigation: str = "",
        example_failure: Optional[str] = None,
    ) -> LimitationRecord:
        rec = LimitationRecord(
            limitation_id=uuid.uuid4().hex[:16],
            domain=domain,
            description=description,
            severity=severity,
            mitigation=mitigation,
            example_failure=example_failure,
        )
        self.limitations.append(rec)
        return rec

    # ── Performance tracking ──────────────────────────────────────────────

    def update_performance(
        self, domain: str, predicted_confidence: float, actual_correct: bool,
    ) -> None:
        if domain not in self.performance:
            self.performance[domain] = PerformanceProfile(domain=domain)
        self.performance[domain].record(predicted_confidence, actual_correct)

    # ── Constraint queries ────────────────────────────────────────────────

    def get_constraints(self, domain: Optional[str] = None) -> List[SelfConstraint]:
        return [c for c in self.constraints if c.enforced and c.applies_to(domain)]

    def add_constraint(self, constraint: SelfConstraint) -> None:
        self.constraints.append(constraint)

    # ── Human-readable explanations ───────────────────────────────────────

    def explain_capability(self, domain: str) -> str:
        level = self.capability_profile.capabilities.get(domain, CapabilityLevel.NONE)
        limits = [l for l in self.limitations if l.domain == domain]
        constraints = self.get_constraints(domain)

        lines = [f"Domain: {domain}", f"  Level: {level.name} — {level.label}"]

        if limits:
            lines.append(f"  Known limitations ({len(limits)}):")
            for lim in limits:
                lines.append(f"    - [{lim.severity.name}] {lim.description}")

        if constraints:
            lines.append(f"  Active constraints ({len(constraints)}):")
            for con in constraints:
                lines.append(f"    - {con.describe()}")

        perf = self.performance.get(domain)
        if perf and perf.sample_count > 0:
            lines.append(
                f"  Performance ({perf.sample_count} samples): "
                f"accuracy={perf.accuracy_estimate:.2%}, "
                f"calibration={perf.calibration_score:.2%}"
            )

        return "\n".join(lines)

    # ── Versioning ────────────────────────────────────────────────────────

    def version(self, changelog: str = "") -> IdentityVersion:
        self._version_counter += 1
        cap_hash = _hash_dict(
            {d: l.value for d, l in self.capability_profile.capabilities.items()}
        )
        con_hash = _hash_dict(
            {c.constraint_id: c.description for c in self.constraints}
        )
        return IdentityVersion(
            version_id=f"v{self._version_counter}",
            model_name=self.model_name,
            capabilities_hash=cap_hash,
            constraints_hash=con_hash,
            changelog=changelog,
        )

    # ── Pipeline Facade ──────────────────────────────────────────────────

    def update(
        self,
        *,
        query: str,
        confidence: float = 0.5,
        domain: str = "general",
        mode: str = "HYBRID",
        h_score: float = 0.0,
    ) -> None:
        """Pipeline-facing API called by ``nrsi.core.nrs._process_inner``.

        Updates performance profile and capability assessment in one call.
        """
        actual_correct = confidence >= 0.6 and h_score >= 0.5
        self.update_performance(
            domain=domain,
            predicted_confidence=confidence,
            actual_correct=actual_correct,
        )

        query_result = self.assess_query(query, domain=domain)
        if query_result and hasattr(query_result, "can_answer") and not query_result.can_answer:
            self.declare_limitation(
                domain=domain,
                description=f"Low confidence ({confidence:.2f}) on: {query[:64]}",
                severity=max(0.0, min(1.0, 1.0 - confidence)),
            )
