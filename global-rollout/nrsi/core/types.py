"""
NRSI Core Type System

The trust type hierarchy is the beating heart of NRSI.

    raw[T]       → Unvalidated data. Cannot be used in trusted computation.
    validated[T] → Passed at least one validation gate.
    trusted[T]   → Passed all required gates with sufficient confidence.
    certified[T] → Trusted + governance policy approved.

Trust flows UP only:  raw → validated → trusted → certified
Downgrade requires explicit justification and audit.

Every piece of data in NRSI carries:
  - A trust level (raw/validated/trusted/certified)
  - A confidence score (0.0 to 1.0)
  - A provenance chain (how it got its trust)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
)

from nrsi.core.errors import TrustError, ConfidenceError


# ── Confidence ───────────────────────────────────────────────────────────────

class Confidence:
    """
    Confidence levels for NRSI data.

    Confidence is a float in [0.0, 1.0] representing how certain
    the system is about a piece of data. Named levels provide
    semantic meaning.
    """

    ABSOLUTE  = 1.0     # Mathematical/logical certainty
    VERY_HIGH = 0.99    # Multiple validation sources agree
    HIGH      = 0.95    # Single strong validation
    MEDIUM    = 0.80    # Needs additional verification
    LOW       = 0.50    # Insufficient for production use
    NONE      = 0.0     # No validation performed

    @staticmethod
    def validate(value: float) -> float:
        """Ensure confidence is within valid range."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"Confidence must be numeric, got {type(value).__name__}")
        value = float(value)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {value}")
        return value

    @staticmethod
    def combine(*values: float) -> float:
        """
        Combine multiple confidence scores.
        Uses conservative strategy: min of all values.
        A chain is only as confident as its weakest link.
        """
        if not values:
            return Confidence.NONE
        return min(Confidence.validate(v) for v in values)

    @staticmethod
    def label(value: float) -> str:
        """Get human-readable label for a confidence value."""
        value = Confidence.validate(value)
        if value >= 1.0:
            return "absolute"
        elif value >= 0.99:
            return "very_high"
        elif value >= 0.95:
            return "high"
        elif value >= 0.80:
            return "medium"
        elif value >= 0.50:
            return "low"
        else:
            return "none"


# ── Trust Levels ─────────────────────────────────────────────────────────────

class TrustLevel(Enum):
    """
    The four trust levels in NRSI, ordered from lowest to highest.
    Trust only flows upward. Downgrade is explicit and audited.
    """

    RAW       = 0   # Unvalidated — cannot be used in trusted computation
    VALIDATED = 1   # Passed at least one validation gate
    TRUSTED   = 2   # Passed all required gates with sufficient confidence
    CERTIFIED = 3   # Trusted + governance policy approved

    def __ge__(self, other: TrustLevel) -> bool:
        if not isinstance(other, TrustLevel):
            return NotImplemented
        return self.value >= other.value

    def __gt__(self, other: TrustLevel) -> bool:
        if not isinstance(other, TrustLevel):
            return NotImplemented
        return self.value > other.value

    def __le__(self, other: TrustLevel) -> bool:
        if not isinstance(other, TrustLevel):
            return NotImplemented
        return self.value <= other.value

    def __lt__(self, other: TrustLevel) -> bool:
        if not isinstance(other, TrustLevel):
            return NotImplemented
        return self.value < other.value


# ── Provenance ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProvenanceEntry:
    """A single entry in a data's provenance chain."""

    timestamp: float
    action: str                          # e.g. "created", "validated", "elevated", "downgraded"
    gate_name: Optional[str] = None      # Which gate processed it
    from_trust: Optional[TrustLevel] = None
    to_trust: Optional[TrustLevel] = None
    confidence: Optional[float] = None
    reason: Optional[str] = None
    actor: Optional[str] = None          # Who/what performed the action

    def __str__(self) -> str:
        parts = [f"[{self.action}]"]
        if self.gate_name:
            parts.append(f"gate={self.gate_name}")
        if self.from_trust and self.to_trust:
            parts.append(f"{self.from_trust.name}→{self.to_trust.name}")
        if self.confidence is not None:
            parts.append(f"confidence={self.confidence:.4f}")
        if self.reason:
            parts.append(f"reason={self.reason}")
        return " ".join(parts)


# ── NRSIData: The Core Wrapper ───────────────────────────────────────────────

T = TypeVar("T")


class NRSIData(Generic[T]):
    """
    The fundamental data wrapper in NRSI.

    Every piece of data in the system is wrapped in NRSIData,
    which carries trust level, confidence, and full provenance.

    You don't create NRSIData directly — use raw(), validated(),
    trusted(), or certified() constructors.
    """

    __slots__ = (
        "_value", "_trust_level", "_confidence", "_provenance",
        "_id", "_created_at", "_metadata",
    )

    def __init__(
        self,
        value: T,
        trust_level: TrustLevel,
        confidence: float,
        provenance: Optional[List[ProvenanceEntry]] = None,
        metadata: Optional[dict] = None,
    ):
        self._value = value
        self._trust_level = trust_level
        self._confidence = Confidence.validate(confidence)
        self._provenance = list(provenance) if provenance else []
        self._id = str(uuid.uuid4())
        self._created_at = time.time()
        self._metadata = metadata or {}

        # Record creation in provenance
        self._provenance.append(ProvenanceEntry(
            timestamp=self._created_at,
            action="created",
            to_trust=trust_level,
            confidence=confidence,
        ))

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def value(self) -> T:
        """The wrapped value."""
        return self._value

    @property
    def trust_level(self) -> TrustLevel:
        """Current trust level."""
        return self._trust_level

    @property
    def confidence(self) -> float:
        """Current confidence score."""
        return self._confidence

    @property
    def confidence_label(self) -> str:
        """Human-readable confidence label."""
        return Confidence.label(self._confidence)

    @property
    def provenance(self) -> List[ProvenanceEntry]:
        """Full provenance chain (immutable copy)."""
        return list(self._provenance)

    @property
    def id(self) -> str:
        """Unique identifier for this data instance."""
        return self._id

    @property
    def is_raw(self) -> bool:
        return self._trust_level == TrustLevel.RAW

    @property
    def is_validated(self) -> bool:
        return self._trust_level >= TrustLevel.VALIDATED

    @property
    def is_trusted(self) -> bool:
        return self._trust_level >= TrustLevel.TRUSTED

    @property
    def is_certified(self) -> bool:
        return self._trust_level >= TrustLevel.CERTIFIED

    # ── Trust Elevation ──────────────────────────────────────────────────

    def elevate(
        self,
        to_level: TrustLevel,
        confidence: float,
        gate_name: str,
        reason: Optional[str] = None,
    ) -> NRSIData[T]:
        """
        Elevate trust level. Returns a NEW NRSIData instance.
        Trust can only go up, never down (use downgrade() for that).
        """
        if to_level <= self._trust_level:
            raise TrustError(
                expected_trust=to_level.name,
                actual_trust=self._trust_level.name,
                operation="elevate",
                suggestion=f"Data is already at {self._trust_level.name}. "
                           f"Cannot elevate to same or lower level {to_level.name}.",
            )

        confidence = Confidence.validate(confidence)
        new_provenance = list(self._provenance)
        new_provenance.append(ProvenanceEntry(
            timestamp=time.time(),
            action="elevated",
            gate_name=gate_name,
            from_trust=self._trust_level,
            to_trust=to_level,
            confidence=confidence,
            reason=reason,
        ))

        result = NRSIData.__new__(NRSIData)
        result._value = self._value
        result._trust_level = to_level
        result._confidence = Confidence.combine(self._confidence, confidence)
        result._provenance = new_provenance
        result._id = self._id  # Same data, elevated
        result._created_at = self._created_at
        result._metadata = dict(self._metadata)
        return result

    def downgrade(
        self,
        to_level: TrustLevel,
        reason: str,
        actor: str,
    ) -> NRSIData[T]:
        """
        Explicitly downgrade trust. Requires justification and actor.
        This is auditable and intentional — never silent.
        """
        if to_level >= self._trust_level:
            raise TrustError(
                expected_trust=to_level.name,
                actual_trust=self._trust_level.name,
                operation="downgrade",
                suggestion="Downgrade must go to a lower trust level.",
            )

        new_provenance = list(self._provenance)
        new_provenance.append(ProvenanceEntry(
            timestamp=time.time(),
            action="downgraded",
            from_trust=self._trust_level,
            to_trust=to_level,
            confidence=self._confidence,
            reason=reason,
            actor=actor,
        ))

        result = NRSIData.__new__(NRSIData)
        result._value = self._value
        result._trust_level = to_level
        result._confidence = self._confidence
        result._provenance = new_provenance
        result._id = self._id
        result._created_at = self._created_at
        result._metadata = dict(self._metadata)
        return result

    # ── Trust Chain ──────────────────────────────────────────────────────

    def trust_chain(self) -> str:
        """Human-readable trust chain for debugging and audit."""
        lines = [f"Trust chain for {type(self._value).__name__} (id: {self._id[:8]}...)"]
        lines.append(f"  Current: {self._trust_level.name} (confidence: {self._confidence:.4f})")
        lines.append("  History:")
        for entry in self._provenance:
            lines.append(f"    {entry}")
        return "\n".join(lines)

    # ── Guards ───────────────────────────────────────────────────────────

    def require_trust(self, minimum: TrustLevel, operation: str = "access") -> T:
        """
        Access the value only if trust level is sufficient.
        This is the enforcement mechanism — raw data cannot
        sneak into trusted computation.
        """
        if self._trust_level < minimum:
            raise TrustError(
                expected_trust=minimum.name,
                actual_trust=self._trust_level.name,
                operation=operation,
            )
        return self._value

    def require_confidence(self, minimum: float, context: str = "") -> T:
        """Access the value only if confidence is sufficient."""
        minimum = Confidence.validate(minimum)
        if self._confidence < minimum:
            raise ConfidenceError(
                actual=self._confidence,
                required=minimum,
                context=context,
            )
        return self._value

    # ── Representation ───────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"{self._trust_level.name.lower()}[{type(self._value).__name__}]"
            f"(confidence={self._confidence:.2f})"
        )

    def __str__(self) -> str:
        return repr(self)


# ── Constructor Functions ────────────────────────────────────────────────────
# These are the public API. Developers use these, not NRSIData directly.

def raw(value: T, metadata: Optional[dict] = None) -> NRSIData[T]:
    """
    Wrap a value as raw (unvalidated) data.
    This is the entry point for all external data into NRSI.
    Raw data MUST pass through a validation gate before use.
    """
    return NRSIData(
        value=value,
        trust_level=TrustLevel.RAW,
        confidence=Confidence.NONE,
        metadata=metadata,
    )


def validated(
    value: T,
    confidence: float,
    gate_name: str,
    metadata: Optional[dict] = None,
) -> NRSIData[T]:
    """Create validated data (passed at least one gate)."""
    return NRSIData(
        value=value,
        trust_level=TrustLevel.VALIDATED,
        confidence=confidence,
        provenance=[ProvenanceEntry(
            timestamp=time.time(),
            action="validated",
            gate_name=gate_name,
            to_trust=TrustLevel.VALIDATED,
            confidence=confidence,
        )],
        metadata=metadata,
    )


def trusted(
    value: T,
    confidence: float,
    gate_name: str,
    metadata: Optional[dict] = None,
) -> NRSIData[T]:
    """Create trusted data (passed all required gates)."""
    return NRSIData(
        value=value,
        trust_level=TrustLevel.TRUSTED,
        confidence=confidence,
        provenance=[ProvenanceEntry(
            timestamp=time.time(),
            action="trusted",
            gate_name=gate_name,
            to_trust=TrustLevel.TRUSTED,
            confidence=confidence,
        )],
        metadata=metadata,
    )


def certified(
    value: T,
    confidence: float,
    gate_name: str,
    policy_name: str,
    metadata: Optional[dict] = None,
) -> NRSIData[T]:
    """Create certified data (trusted + governance approved)."""
    return NRSIData(
        value=value,
        trust_level=TrustLevel.CERTIFIED,
        confidence=confidence,
        provenance=[ProvenanceEntry(
            timestamp=time.time(),
            action="certified",
            gate_name=gate_name,
            to_trust=TrustLevel.CERTIFIED,
            confidence=confidence,
            reason=f"Certified under policy '{policy_name}'",
        )],
        metadata=metadata,
    )


# ── Type Aliases for Annotations ─────────────────────────────────────────────
# These let developers write type hints like: def process(data: Trusted[str]) -> Certified[str]

Raw = NRSIData        # raw[T] in type hints
Validated = NRSIData  # validated[T] in type hints
Trusted = NRSIData    # trusted[T] in type hints
Certified = NRSIData  # certified[T] in type hints
