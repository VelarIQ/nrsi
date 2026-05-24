"""NRSI Uncertainty Decomposition — Beyond Single Confidence Scores.

A single float confidence conflates fundamentally different kinds of
uncertainty.  These types decompose uncertainty into orthogonal dimensions:

  Aleatoric   — inherent randomness (can't be reduced with more data)
  Epistemic   — lack of knowledge (CAN be reduced with more data)
  Ambiguity   — multiple valid interpretations
  Vagueness   — imprecise/borderline applicability
  Conflict    — strong contradictory evidence
  Ignorance   — complete absence of evidence

Each dimension is 0.0–1.0.  Together they form an UncertaintyProfile
that tells you not just "how uncertain" but "WHY uncertain" — which
determines the right intervention (disambiguate vs fetch vs negotiate).

Patent-covered: NRSI Uncertainty Decomposition System, VelarIQ.
"""

from __future__ import annotations

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
    TypeVar,
    Union,
)

from nrsi.core.types import (
    Confidence,
    NRSIData,
    ProvenanceEntry,
    TrustLevel,
    raw,
)
from nrsi.core.validation import (
    ValidationGate,
    ValidationResult,
    Validator,
)
from nrsi.core.errors import NRSIError


T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════════
# Errors
# ═══════════════════════════════════════════════════════════════════════════════

class UncertaintyError(NRSIError):
    """Raised when an uncertainty operation violates constraints."""

    def __init__(
        self,
        operation: str,
        reason: str,
        suggestion: Optional[str] = None,
    ):
        self.operation = operation
        self.reason = reason
        msg = f"Uncertainty error during '{operation}': {reason}"
        super().__init__(msg, suggestion)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. UncertaintyDimension — the six orthogonal axes
# ═══════════════════════════════════════════════════════════════════════════════

class UncertaintyDimension(Enum):
    """The six orthogonal dimensions of uncertainty."""

    ALEATORIC = auto()
    EPISTEMIC = auto()
    AMBIGUITY = auto()
    VAGUENESS = auto()
    CONFLICT  = auto()
    IGNORANCE = auto()


_INTERVENTION_MAP: Dict[UncertaintyDimension, str] = {
    UncertaintyDimension.ALEATORIC: "accept_irreducible",
    UncertaintyDimension.EPISTEMIC: "gather_more_data",
    UncertaintyDimension.AMBIGUITY: "disambiguate",
    UncertaintyDimension.VAGUENESS: "request_precision",
    UncertaintyDimension.CONFLICT:  "resolve_contradiction",
    UncertaintyDimension.IGNORANCE: "acknowledge_and_flag",
}

_RESOLVABLE: frozenset[UncertaintyDimension] = frozenset({
    UncertaintyDimension.EPISTEMIC,
    UncertaintyDimension.AMBIGUITY,
    UncertaintyDimension.VAGUENESS,
    UncertaintyDimension.CONFLICT,
})


# ═══════════════════════════════════════════════════════════════════════════════
# 2. UncertaintyProfile — the full 6-dimensional decomposition
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UncertaintyProfile:
    """Six-dimensional uncertainty decomposition.

    Each dimension is a float in [0.0, 1.0].  Methods answer:
      - What is the dominant source of uncertainty?
      - Is this resolvable (i.e. not purely aleatoric)?
      - What intervention is needed?
      - What scalar confidence does this map to?
    """

    aleatoric: float = 0.0
    epistemic: float = 0.0
    ambiguity: float = 0.0
    vagueness: float = 0.0
    conflict: float = 0.0
    ignorance: float = 0.0

    def __post_init__(self) -> None:
        self.aleatoric = _clamp01(self.aleatoric)
        self.epistemic = _clamp01(self.epistemic)
        self.ambiguity = _clamp01(self.ambiguity)
        self.vagueness = _clamp01(self.vagueness)
        self.conflict  = _clamp01(self.conflict)
        self.ignorance = _clamp01(self.ignorance)

    # ── Query Methods ─────────────────────────────────────────────────────

    def as_dict(self) -> Dict[UncertaintyDimension, float]:
        return {
            UncertaintyDimension.ALEATORIC: self.aleatoric,
            UncertaintyDimension.EPISTEMIC: self.epistemic,
            UncertaintyDimension.AMBIGUITY: self.ambiguity,
            UncertaintyDimension.VAGUENESS: self.vagueness,
            UncertaintyDimension.CONFLICT:  self.conflict,
            UncertaintyDimension.IGNORANCE: self.ignorance,
        }

    def dominant_dimension(self) -> UncertaintyDimension:
        """Return the dimension with the highest uncertainty."""
        d = self.as_dict()
        return max(d, key=lambda k: d[k])

    def total_uncertainty(self) -> float:
        """Sum of all dimensions (can exceed 1.0 — not a probability)."""
        return (
            self.aleatoric + self.epistemic + self.ambiguity
            + self.vagueness + self.conflict + self.ignorance
        )

    def mean_uncertainty(self) -> float:
        return self.total_uncertainty() / 6.0

    def intervention_needed(self) -> str:
        """Return the recommended intervention for the dominant dimension."""
        return _INTERVENTION_MAP[self.dominant_dimension()]

    def is_resolvable(self) -> bool:
        """Whether the dominant uncertainty can be reduced.

        Aleatoric and ignorance uncertainty cannot be directly reduced
        by more data or analysis.  Epistemic, ambiguity, vagueness, and
        conflict can be.
        """
        dom = self.dominant_dimension()
        resolvable_total = (
            self.epistemic + self.ambiguity + self.vagueness + self.conflict
        )
        irreducible_total = self.aleatoric + self.ignorance
        return resolvable_total > irreducible_total

    def to_confidence(self) -> float:
        """Convert back to a scalar confidence (1.0 - mean_uncertainty).

        This is a lossy projection — the whole point of UncertaintyProfile
        is to avoid collapsing to a single number.  Use only when a scalar
        is required by a downstream interface.
        """
        return _clamp01(1.0 - self.mean_uncertainty())

    @staticmethod
    def from_confidence(
        conf: float,
        primary_type: UncertaintyDimension = UncertaintyDimension.EPISTEMIC,
    ) -> UncertaintyProfile:
        """Heuristic decomposition from a scalar confidence.

        Assigns the bulk of (1 - conf) to *primary_type* with small
        baseline contributions to other dimensions.
        """
        conf = Confidence.validate(conf)
        total_unc = 1.0 - conf
        baseline = total_unc * 0.05
        primary_share = total_unc * 0.75

        profile = UncertaintyProfile(
            aleatoric=baseline,
            epistemic=baseline,
            ambiguity=baseline,
            vagueness=baseline,
            conflict=baseline,
            ignorance=baseline,
        )
        if primary_type == UncertaintyDimension.ALEATORIC:
            profile.aleatoric = _clamp01(primary_share)
        elif primary_type == UncertaintyDimension.EPISTEMIC:
            profile.epistemic = _clamp01(primary_share)
        elif primary_type == UncertaintyDimension.AMBIGUITY:
            profile.ambiguity = _clamp01(primary_share)
        elif primary_type == UncertaintyDimension.VAGUENESS:
            profile.vagueness = _clamp01(primary_share)
        elif primary_type == UncertaintyDimension.CONFLICT:
            profile.conflict = _clamp01(primary_share)
        elif primary_type == UncertaintyDimension.IGNORANCE:
            profile.ignorance = _clamp01(primary_share)
        return profile

    def __repr__(self) -> str:
        dom = self.dominant_dimension()
        return (
            f"UncertaintyProfile(dominant={dom.name}, "
            f"total={self.total_uncertainty():.3f}, "
            f"conf≈{self.to_confidence():.3f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ConflictEvidence — record of contradictory evidence
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ConflictEvidence:
    """Record of two sources providing contradictory evidence."""

    conflict_id: str
    source_a: str
    source_b: str
    claim_a: Any
    claim_b: Any
    overlap_domain: str
    severity: float
    detected_at: float

    def __init__(
        self,
        source_a: str,
        source_b: str,
        claim_a: Any,
        claim_b: Any,
        overlap_domain: str,
        severity: float,
        conflict_id: Optional[str] = None,
        detected_at: Optional[float] = None,
    ) -> None:
        object.__setattr__(self, "conflict_id", conflict_id or str(uuid.uuid4()))
        object.__setattr__(self, "source_a", source_a)
        object.__setattr__(self, "source_b", source_b)
        object.__setattr__(self, "claim_a", claim_a)
        object.__setattr__(self, "claim_b", claim_b)
        object.__setattr__(self, "overlap_domain", overlap_domain)
        object.__setattr__(self, "severity", max(0.0, min(1.0, severity)))
        object.__setattr__(self, "detected_at", detected_at if detected_at is not None else time.time())

    @property
    def is_severe(self) -> bool:
        return self.severity >= 0.7

    def __repr__(self) -> str:
        return (
            f"ConflictEvidence({self.source_a!r} vs {self.source_b!r}, "
            f"domain={self.overlap_domain!r}, severity={self.severity:.2f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. AmbiguitySet — multiple valid interpretations
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Interpretation:
    """A single interpretation with its probability."""

    label: str
    content: Any
    probability: float

    def __post_init__(self) -> None:
        self.probability = _clamp01(self.probability)


@dataclass
class AmbiguitySet:
    """A set of possible interpretations for an ambiguous input.

    If the system has selected one, `selected` is set; otherwise it's
    None and the ambiguity is unresolved.
    """

    interpretations: List[Interpretation] = field(default_factory=list)
    selected: Optional[int] = None
    disambiguation_method: str = ""

    def add(self, label: str, content: Any, probability: float) -> None:
        self.interpretations.append(Interpretation(label, content, probability))

    def select(self, index: int, method: str = "") -> Interpretation:
        if index < 0 or index >= len(self.interpretations):
            raise UncertaintyError(
                operation="select_interpretation",
                reason=f"Index {index} out of range (0–{len(self.interpretations) - 1})",
            )
        self.selected = index
        self.disambiguation_method = method
        return self.interpretations[index]

    @property
    def is_resolved(self) -> bool:
        return self.selected is not None

    @property
    def selected_interpretation(self) -> Optional[Interpretation]:
        if self.selected is not None:
            return self.interpretations[self.selected]
        return None

    @property
    def entropy(self) -> float:
        """Shannon entropy of the interpretation distribution (nats)."""
        total = sum(i.probability for i in self.interpretations)
        if total <= 0.0 or len(self.interpretations) <= 1:
            return 0.0
        h = 0.0
        for interp in self.interpretations:
            p = interp.probability / total
            if p > 0.0:
                h -= p * math.log(p)
        return h

    def normalize(self) -> None:
        """Rescale probabilities so they sum to 1.0."""
        total = sum(i.probability for i in self.interpretations)
        if total <= 0.0:
            return
        for interp in self.interpretations:
            interp.probability /= total

    def __len__(self) -> int:
        return len(self.interpretations)

    def __repr__(self) -> str:
        status = f"selected={self.selected}" if self.is_resolved else "unresolved"
        return f"AmbiguitySet(n={len(self.interpretations)}, {status})"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. UncertaintyPropagation — static methods for chain/merge/branch/reduce
# ═══════════════════════════════════════════════════════════════════════════════

class UncertaintyPropagation:
    """Static methods for propagating UncertaintyProfile through
    reasoning chains, merges, branches, and evidence incorporation.
    """

    @staticmethod
    def chain(profiles: Sequence[UncertaintyProfile]) -> UncertaintyProfile:
        """Propagate uncertainty through a sequential chain.

        In a chain A → B → C, uncertainty accumulates.  We take the
        element-wise maximum across all profiles (conservative).
        """
        if not profiles:
            return UncertaintyProfile()
        return UncertaintyProfile(
            aleatoric=max(p.aleatoric for p in profiles),
            epistemic=max(p.epistemic for p in profiles),
            ambiguity=max(p.ambiguity for p in profiles),
            vagueness=max(p.vagueness for p in profiles),
            conflict=max(p.conflict for p in profiles),
            ignorance=max(p.ignorance for p in profiles),
        )

    @staticmethod
    def merge(profiles: Sequence[UncertaintyProfile]) -> UncertaintyProfile:
        """Merge uncertainty from multiple parallel sources.

        Parallel evidence should *reduce* epistemic/ignorance uncertainty
        (more data = less ignorance) while preserving aleatoric.  We take
        the element-wise mean, which is less conservative than chain().
        """
        if not profiles:
            return UncertaintyProfile()
        n = len(profiles)
        return UncertaintyProfile(
            aleatoric=max(p.aleatoric for p in profiles),
            epistemic=sum(p.epistemic for p in profiles) / n,
            ambiguity=sum(p.ambiguity for p in profiles) / n,
            vagueness=sum(p.vagueness for p in profiles) / n,
            conflict=max(p.conflict for p in profiles),
            ignorance=min(p.ignorance for p in profiles),
        )

    @staticmethod
    def branch(profile: UncertaintyProfile, n_branches: int) -> List[UncertaintyProfile]:
        """Create *n_branches* copies of *profile* for parallel exploration.

        Each branch inherits the parent's uncertainty plus a small
        ambiguity increase (branching itself introduces interpretation variance).
        """
        if n_branches < 1:
            raise UncertaintyError(
                operation="branch", reason=f"n_branches must be >= 1, got {n_branches}",
            )
        per_branch_ambiguity_bump = 0.02 * math.log(n_branches + 1)
        return [
            UncertaintyProfile(
                aleatoric=profile.aleatoric,
                epistemic=profile.epistemic,
                ambiguity=_clamp01(profile.ambiguity + per_branch_ambiguity_bump),
                vagueness=profile.vagueness,
                conflict=profile.conflict,
                ignorance=profile.ignorance,
            )
            for _ in range(n_branches)
        ]

    @staticmethod
    def reduce(
        profile: UncertaintyProfile,
        new_evidence_type: UncertaintyDimension,
        reduction: float = 0.3,
    ) -> UncertaintyProfile:
        """Reduce a specific dimension based on newly acquired evidence.

        Aleatoric uncertainty is not reducible — a reduction request for
        aleatoric is silently capped to 0.
        """
        d = profile.as_dict()
        if new_evidence_type == UncertaintyDimension.ALEATORIC:
            return UncertaintyProfile(**{dim.name.lower(): v for dim, v in d.items()})

        new_vals: Dict[str, float] = {}
        for dim, v in d.items():
            key = dim.name.lower()
            if dim == new_evidence_type:
                new_vals[key] = _clamp01(v - reduction)
            else:
                new_vals[key] = v
        return UncertaintyProfile(**new_vals)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. UncertaintyGate — ValidationGate that inspects uncertainty profiles
# ═══════════════════════════════════════════════════════════════════════════════

class _UncertaintyValidator(Validator):
    """Internal validator that checks an UncertaintyProfile against thresholds."""

    name = "uncertainty_validator"

    def __init__(
        self,
        max_conflict: float,
        max_ignorance: float,
        max_total: float,
    ) -> None:
        self._max_conflict = max_conflict
        self._max_ignorance = max_ignorance
        self._max_total = max_total

    def validate(self, data: Any, context: Any = None) -> ValidationResult:
        profile: Optional[UncertaintyProfile] = None
        if isinstance(data, UncertaintyProfile):
            profile = data
        elif isinstance(data, NRSIData) and isinstance(data.value, UncertaintyProfile):
            profile = data.value
        elif isinstance(data, dict) and "aleatoric" in data:
            try:
                profile = UncertaintyProfile(**{
                    k: float(v) for k, v in data.items()
                    if k in ("aleatoric", "epistemic", "ambiguity", "vagueness", "conflict", "ignorance")
                })
            except (TypeError, ValueError):
                pass

        if profile is None:
            return ValidationResult(
                passed=False,
                confidence=Confidence.NONE,
                validator_name=self.name,
                details="Cannot extract UncertaintyProfile from data",
            )

        failures: List[str] = []
        if profile.conflict > self._max_conflict:
            failures.append(
                f"conflict={profile.conflict:.3f} exceeds max {self._max_conflict:.3f}"
            )
        if profile.ignorance > self._max_ignorance:
            failures.append(
                f"ignorance={profile.ignorance:.3f} exceeds max {self._max_ignorance:.3f}"
            )
        if profile.total_uncertainty() > self._max_total:
            failures.append(
                f"total={profile.total_uncertainty():.3f} exceeds max {self._max_total:.3f}"
            )

        if failures:
            return ValidationResult(
                passed=False,
                confidence=_clamp01(profile.to_confidence()),
                validator_name=self.name,
                details="; ".join(failures),
            )

        detail_parts = [f"profile OK (dominant={profile.dominant_dimension().name})"]
        if profile.ambiguity > 0.3:
            detail_parts.append(f"disambiguation note: ambiguity={profile.ambiguity:.3f}")

        return ValidationResult(
            passed=True,
            confidence=_clamp01(max(profile.to_confidence(), Confidence.MEDIUM)),
            validator_name=self.name,
            details="; ".join(detail_parts),
        )


class UncertaintyGate(ValidationGate):
    """Validation gate that checks uncertainty profiles.

    Blocks data with high conflict or high ignorance.
    Passes through with a disambiguation note when ambiguity is elevated.

    Usage::

        gate = UncertaintyGate(max_conflict=0.5, max_ignorance=0.7)
        gate.process(raw(some_uncertainty_profile))
    """

    def __init__(
        self,
        name: str = "uncertainty_gate",
        max_conflict: float = 0.5,
        max_ignorance: float = 0.7,
        max_total: float = 3.0,
        confidence_threshold: float = Confidence.MEDIUM,
        target_trust: TrustLevel = TrustLevel.VALIDATED,
    ) -> None:
        self._max_conflict = max_conflict
        self._max_ignorance = max_ignorance
        self._max_total = max_total
        super().__init__(
            name=name,
            confidence_threshold=confidence_threshold,
            validators=[_UncertaintyValidator(max_conflict, max_ignorance, max_total)],
            target_trust=target_trust,
            require_all=True,
            audit=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _clamp01(v: float) -> float:
    """Clamp a float to [0.0, 1.0]."""
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Facade — API expected by nrsi.core.nrs._process_inner
# ═══════════════════════════════════════════════════════════════════════════════

class UncertaintyDecomposer:
    """Facade providing the ``decompose(query, domain=)`` API that
    ``nrsi.core.nrs._process_inner`` expects.

    Wraps ``UncertaintyProfile.from_confidence`` with heuristic
    domain-based adjustments.
    """

    _DOMAIN_PROFILES: Dict[str, UncertaintyDimension] = {
        "medical": UncertaintyDimension.EPISTEMIC,
        "financial": UncertaintyDimension.CONFLICT,
        "legal": UncertaintyDimension.AMBIGUITY,
        "science": UncertaintyDimension.EPISTEMIC,
        "engineering": UncertaintyDimension.ALEATORIC,
    }

    def decompose(
        self,
        query: str,
        *,
        domain: str = "general",
        confidence: float = 0.5,
    ) -> UncertaintyProfile:
        """Decompose scalar uncertainty into a 6-dimensional profile."""
        primary = self._DOMAIN_PROFILES.get(
            domain, UncertaintyDimension.EPISTEMIC
        )
        profile = UncertaintyProfile.from_confidence(confidence, primary)

        q_lower = query.lower()
        if "?" in q_lower:
            profile.ambiguity = _clamp01(profile.ambiguity + 0.1)
        if any(w in q_lower for w in ("maybe", "possibly", "uncertain", "might")):
            profile.vagueness = _clamp01(profile.vagueness + 0.15)

        return profile
