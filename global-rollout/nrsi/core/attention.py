"""NRSI Attention & Focus — Cognitive Resource Allocation Primitives.

The brain can't process everything at once. Attention selects what matters.
These types make attention allocation a first-class, auditable operation.

Primitives:
  FocusToken        — A single item in the attention field with salience score
  AttentionFrame    — The current working set of active focus tokens (capacity-limited)
  SalienceMap       — Maps claims/hypotheses to salience scores with decay
  ResourceBudget    — Typed compute/time/depth budgets for reasoning tasks
  AttentionPolicy   — Rules for what gets attended, what gets suppressed
  CompetitionResult — Outcome when multiple items compete for attention
  AttentionShift    — Audit trail for changes in attentional focus
  AttentionGate     — ValidationGate that only passes data currently attended

Patent-covered: NRSI Cognitive Attention & Focus System, VelarIQ.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
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
    GateResult,
)
from nrsi.core.errors import NRSIError, ValidationError


T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════════
# Errors
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionError(NRSIError):
    """Raised when an attention operation violates invariants."""

    def __init__(
        self,
        operation: str,
        reason: str,
        suggestion: Optional[str] = None,
    ):
        self.operation = operation
        self.reason = reason
        msg = f"Attention error during '{operation}': {reason}"
        super().__init__(msg, suggestion)


class BudgetExhaustedError(NRSIError):
    """Raised when a resource budget has been fully consumed."""

    def __init__(
        self,
        resource: str,
        consumed: float,
        maximum: float,
        suggestion: Optional[str] = None,
    ):
        self.resource = resource
        self.consumed = consumed
        self.maximum = maximum
        msg = (
            f"Resource budget exhausted: {resource} "
            f"(consumed {consumed:.1f} / max {maximum:.1f})"
        )
        if not suggestion:
            suggestion = "Reduce task scope or allocate a larger budget"
        super().__init__(msg, suggestion)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. EvictionPolicy — how to decide what leaves the attention frame
# ═══════════════════════════════════════════════════════════════════════════════

class EvictionStrategy(Enum):
    """Strategy for evicting tokens when the attention frame is full."""

    LOWEST_SALIENCE = auto()
    OLDEST_ACCESS   = auto()
    OLDEST_CREATED  = auto()
    LEAST_ACCESSED  = auto()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FocusToken — a single item in the attention field
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FocusToken:
    """A single item held in working attention with a salience score.

    Salience decays over time unless the token is re-accessed.
    """

    item_id: str
    content: Any
    salience: float
    source_lobe: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 1
    decay_rate: float = 0.05

    def __post_init__(self) -> None:
        self.salience = _clamp01(self.salience)
        if self.decay_rate < 0.0:
            raise ValueError(f"decay_rate must be >= 0, got {self.decay_rate}")

    def touch(self, boost: float = 0.0) -> None:
        """Record an access and optionally boost salience."""
        self.last_accessed = time.time()
        self.access_count += 1
        if boost > 0.0:
            self.salience = _clamp01(self.salience + boost)

    def apply_decay(self, elapsed_ms: float) -> None:
        """Exponential salience decay based on elapsed time."""
        if elapsed_ms <= 0.0:
            return
        elapsed_s = elapsed_ms / 1000.0
        self.salience = _clamp01(
            self.salience * math.exp(-self.decay_rate * elapsed_s)
        )

    @property
    def age_ms(self) -> float:
        return (time.time() - self.created_at) * 1000.0

    @property
    def idle_ms(self) -> float:
        return (time.time() - self.last_accessed) * 1000.0

    def __repr__(self) -> str:
        return (
            f"FocusToken(id={self.item_id!r}, salience={self.salience:.3f}, "
            f"lobe={self.source_lobe!r}, accesses={self.access_count})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. AttentionFrame — capacity-limited working attention set
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionFrame:
    """The current working set of active focus tokens.

    Capacity defaults to 7 (Miller's Law: 7 ± 2 working memory slots).
    When full, new tokens trigger eviction of the lowest-salience item.
    Every mutation is tracked for audit.
    """

    __slots__ = (
        "_tokens", "_capacity", "_eviction_strategy",
        "_total_switches", "_history",
    )

    def __init__(
        self,
        capacity: int = 7,
        eviction_strategy: EvictionStrategy = EvictionStrategy.LOWEST_SALIENCE,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._tokens: Dict[str, FocusToken] = {}
        self._capacity = capacity
        self._eviction_strategy = eviction_strategy
        self._total_switches: int = 0
        self._history: List[AttentionShift] = []

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return len(self._tokens)

    @property
    def is_full(self) -> bool:
        return len(self._tokens) >= self._capacity

    @property
    def total_switches(self) -> int:
        return self._total_switches

    @property
    def active_ids(self) -> List[str]:
        return sorted(self._tokens, key=lambda k: self._tokens[k].salience, reverse=True)

    @property
    def history(self) -> List[AttentionShift]:
        return list(self._history)

    # ── Core Operations ───────────────────────────────────────────────────

    def attend(self, token: FocusToken) -> Optional[FocusToken]:
        """Add a token to the frame; evict the weakest if at capacity.

        Returns the evicted token (if any), or ``None``.
        """
        if token.item_id in self._tokens:
            existing = self._tokens[token.item_id]
            existing.touch(boost=token.salience - existing.salience)
            return None

        evicted: Optional[FocusToken] = None
        if self.is_full:
            evicted = self._pick_eviction()
            if evicted is not None:
                if token.salience <= evicted.salience:
                    return None
                del self._tokens[evicted.item_id]

        self._tokens[token.item_id] = token
        self._total_switches += 1
        self._record_shift(
            removed=[evicted] if evicted else [],
            added=[token],
            trigger="attend",
        )
        return evicted

    def suppress(self, item_id: str, reason: str = "") -> Optional[FocusToken]:
        """Explicitly remove a token from focus with an audit reason."""
        token = self._tokens.pop(item_id, None)
        if token is not None:
            self._total_switches += 1
            self._record_shift(
                removed=[token],
                added=[],
                trigger=f"suppress({reason})" if reason else "suppress",
            )
        return token

    def boost(self, item_id: str, amount: float) -> None:
        """Increase salience for a token already in focus."""
        token = self._tokens.get(item_id)
        if token is None:
            raise AttentionError(
                operation="boost",
                reason=f"Token {item_id!r} is not in the attention frame",
                suggestion="Call attend() first or check is_attended()",
            )
        token.touch(boost=amount)

    def decay_all(self, elapsed_ms: float) -> List[FocusToken]:
        """Apply exponential decay to every token.  Returns any that
        dropped below a negligible salience threshold (1e-4) and were evicted.
        """
        evicted: List[FocusToken] = []
        for token in list(self._tokens.values()):
            token.apply_decay(elapsed_ms)
            if token.salience < 1e-4:
                del self._tokens[token.item_id]
                evicted.append(token)
        if evicted:
            self._total_switches += len(evicted)
            self._record_shift(removed=evicted, added=[], trigger="decay")
        return evicted

    def top_k(self, k: int) -> List[FocusToken]:
        """Return the *k* highest-salience tokens, sorted descending."""
        ordered = sorted(self._tokens.values(), key=lambda t: t.salience, reverse=True)
        return ordered[:k]

    def is_attended(self, item_id: str) -> bool:
        return item_id in self._tokens

    def get(self, item_id: str) -> Optional[FocusToken]:
        return self._tokens.get(item_id)

    def clear(self) -> List[FocusToken]:
        """Remove all tokens. Returns previously held tokens."""
        removed = list(self._tokens.values())
        self._tokens.clear()
        if removed:
            self._total_switches += len(removed)
            self._record_shift(removed=removed, added=[], trigger="clear")
        return removed

    def snapshot(self) -> List[FocusToken]:
        """Return all tokens ordered by descending salience (read-only)."""
        return sorted(self._tokens.values(), key=lambda t: t.salience, reverse=True)

    # ── Internals ─────────────────────────────────────────────────────────

    def _pick_eviction(self) -> Optional[FocusToken]:
        if not self._tokens:
            return None
        tokens = list(self._tokens.values())
        if self._eviction_strategy == EvictionStrategy.LOWEST_SALIENCE:
            return min(tokens, key=lambda t: t.salience)
        elif self._eviction_strategy == EvictionStrategy.OLDEST_ACCESS:
            return min(tokens, key=lambda t: t.last_accessed)
        elif self._eviction_strategy == EvictionStrategy.OLDEST_CREATED:
            return min(tokens, key=lambda t: t.created_at)
        elif self._eviction_strategy == EvictionStrategy.LEAST_ACCESSED:
            return min(tokens, key=lambda t: t.access_count)
        return min(tokens, key=lambda t: t.salience)

    def _record_shift(
        self,
        removed: List[FocusToken],
        added: List[FocusToken],
        trigger: str,
    ) -> None:
        self._history.append(AttentionShift(
            from_focus=[t.item_id for t in removed],
            to_focus=[t.item_id for t in added],
            trigger=trigger,
            timestamp=time.time(),
        ))

    def __len__(self) -> int:
        return len(self._tokens)

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._tokens

    def __repr__(self) -> str:
        ids = ", ".join(self.active_ids[:5])
        extra = f"…+{len(self._tokens) - 5}" if len(self._tokens) > 5 else ""
        return (
            f"AttentionFrame(size={self.size}/{self._capacity}, "
            f"tokens=[{ids}{extra}])"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SalienceMap — maps keys to salience scores
# ═══════════════════════════════════════════════════════════════════════════════

class SalienceMap:
    """Maps string keys (claims, hypotheses, lobe IDs) to salience scores.

    Supports bulk decay, normalisation, and top-k queries.
    """

    __slots__ = ("_map",)

    def __init__(self, initial: Optional[Dict[str, float]] = None) -> None:
        self._map: Dict[str, float] = {}
        if initial:
            for k, v in initial.items():
                self._map[k] = _clamp01(v)

    def set(self, key: str, salience: float) -> None:
        self._map[key] = _clamp01(salience)

    def get(self, key: str, default: float = 0.0) -> float:
        return self._map.get(key, default)

    def update(self, key: str, delta: float) -> float:
        """Add *delta* to existing salience (clamped to [0,1])."""
        current = self._map.get(key, 0.0)
        new_val = _clamp01(current + delta)
        self._map[key] = new_val
        return new_val

    def decay(self, rate: float, elapsed_ms: float) -> None:
        """Apply exponential decay to all entries."""
        if elapsed_ms <= 0.0:
            return
        elapsed_s = elapsed_ms / 1000.0
        factor = math.exp(-rate * elapsed_s)
        for k in self._map:
            self._map[k] = _clamp01(self._map[k] * factor)

    def normalize(self) -> None:
        """Rescale so the maximum salience is 1.0 (preserves ratios)."""
        if not self._map:
            return
        peak = max(self._map.values())
        if peak <= 0.0:
            return
        for k in self._map:
            self._map[k] /= peak

    def top_k(self, k: int) -> List[Tuple[str, float]]:
        ordered = sorted(self._map.items(), key=lambda kv: kv[1], reverse=True)
        return ordered[:k]

    def prune(self, threshold: float = 1e-4) -> int:
        """Remove entries below *threshold*. Returns count removed."""
        to_remove = [k for k, v in self._map.items() if v < threshold]
        for k in to_remove:
            del self._map[k]
        return len(to_remove)

    def keys(self) -> List[str]:
        return list(self._map.keys())

    def items(self) -> List[Tuple[str, float]]:
        return list(self._map.items())

    def __len__(self) -> int:
        return len(self._map)

    def __contains__(self, key: str) -> bool:
        return key in self._map

    def __repr__(self) -> str:
        top3 = self.top_k(3)
        entries = ", ".join(f"{k}={v:.3f}" for k, v in top3)
        return f"SalienceMap(n={len(self._map)}, top=[{entries}])"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ResourceBudget — typed compute/time/depth budgets
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ResourceBudget:
    """Budget envelope for a reasoning task.

    Tracks consumption across multiple resource dimensions and raises
    ``BudgetExhaustedError`` when any dimension is exceeded.
    """

    max_time_ms: float = 10_000.0
    max_depth: int = 20
    max_tokens: int = 131072
    max_lobe_activations: int = 200

    _consumed_time_ms: float = field(default=0.0, repr=False, init=False)
    _consumed_depth: int = field(default=0, repr=False, init=False)
    _consumed_tokens: int = field(default=0, repr=False, init=False)
    _consumed_lobe_activations: int = field(default=0, repr=False, init=False)

    # ── Query ─────────────────────────────────────────────────────────────

    @property
    def remaining_time_ms(self) -> float:
        return max(0.0, self.max_time_ms - self._consumed_time_ms)

    @property
    def remaining_depth(self) -> int:
        return max(0, self.max_depth - self._consumed_depth)

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self._consumed_tokens)

    @property
    def remaining_lobe_activations(self) -> int:
        return max(0, self.max_lobe_activations - self._consumed_lobe_activations)

    def is_exhausted(self) -> bool:
        return (
            self._consumed_time_ms >= self.max_time_ms
            or self._consumed_depth >= self.max_depth
            or self._consumed_tokens >= self.max_tokens
            or self._consumed_lobe_activations >= self.max_lobe_activations
        )

    def utilisation(self) -> Dict[str, float]:
        """Return utilisation fraction for each dimension."""
        def _frac(consumed: float, maximum: float) -> float:
            return consumed / maximum if maximum > 0 else 0.0
        return {
            "time_ms": _frac(self._consumed_time_ms, self.max_time_ms),
            "depth": _frac(float(self._consumed_depth), float(self.max_depth)),
            "tokens": _frac(float(self._consumed_tokens), float(self.max_tokens)),
            "lobe_activations": _frac(
                float(self._consumed_lobe_activations),
                float(self.max_lobe_activations),
            ),
        }

    # ── Consume ───────────────────────────────────────────────────────────

    def consume(self, resource: str, amount: Union[int, float]) -> None:
        """Consume *amount* of *resource*, raising if exhausted."""
        if resource == "time_ms":
            self._consumed_time_ms += float(amount)
            if self._consumed_time_ms >= self.max_time_ms:
                raise BudgetExhaustedError(
                    "time_ms", self._consumed_time_ms, self.max_time_ms,
                )
        elif resource == "depth":
            self._consumed_depth += int(amount)
            if self._consumed_depth >= self.max_depth:
                raise BudgetExhaustedError(
                    "depth", float(self._consumed_depth), float(self.max_depth),
                )
        elif resource == "tokens":
            self._consumed_tokens += int(amount)
            if self._consumed_tokens >= self.max_tokens:
                raise BudgetExhaustedError(
                    "tokens", float(self._consumed_tokens), float(self.max_tokens),
                )
        elif resource == "lobe_activations":
            self._consumed_lobe_activations += int(amount)
            if self._consumed_lobe_activations >= self.max_lobe_activations:
                raise BudgetExhaustedError(
                    "lobe_activations",
                    float(self._consumed_lobe_activations),
                    float(self.max_lobe_activations),
                )
        else:
            raise AttentionError(
                operation="consume",
                reason=f"Unknown resource dimension: {resource!r}",
                suggestion="Use one of: time_ms, depth, tokens, lobe_activations",
            )

    def consume_time(self, ms: float) -> None:
        self.consume("time_ms", ms)

    def consume_depth(self, levels: int = 1) -> None:
        self.consume("depth", levels)

    def consume_tokens(self, count: int) -> None:
        self.consume("tokens", count)

    def consume_lobe_activation(self, count: int = 1) -> None:
        self.consume("lobe_activations", count)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. AttentionPolicy — declarative rules for what gets attended
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AttentionPolicy:
    """Declarative configuration for attention behaviour.

    Controls which tokens are admitted, how quickly they decay, and
    which lobes/domains receive priority or suppression.
    """

    min_salience_threshold: float = 0.1
    max_capacity: int = 7
    decay_rate: float = 0.05
    boost_on_access: float = 0.05
    priority_lobes: FrozenSet[str] = field(default_factory=frozenset)
    suppress_domains: FrozenSet[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.min_salience_threshold < 0.0 or self.min_salience_threshold > 1.0:
            raise ValueError("min_salience_threshold must be in [0.0, 1.0]")
        if self.max_capacity < 1:
            raise ValueError("max_capacity must be >= 1")
        if self.decay_rate < 0.0:
            raise ValueError("decay_rate must be >= 0")

    def admits(self, token: FocusToken) -> bool:
        """Return whether *token* would be admitted under this policy."""
        if token.salience < self.min_salience_threshold:
            return False
        if self.suppress_domains:
            if token.source_lobe in self.suppress_domains:
                return False
        return True

    def effective_salience(self, token: FocusToken) -> float:
        """Adjust salience based on priority/suppress rules."""
        s = token.salience
        if token.source_lobe in self.priority_lobes:
            s = _clamp01(s * 1.25)
        if token.source_lobe in self.suppress_domains:
            s = _clamp01(s * 0.5)
        return s


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CompetitionResult — outcome when tokens compete for attention
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CompetitionResult:
    """Outcome of a competition for attention between multiple tokens."""

    winner: FocusToken
    losers: Tuple[FocusToken, ...]
    reason: str
    salience_delta: float

    @property
    def margin(self) -> float:
        """Salience difference between winner and best loser."""
        if not self.losers:
            return self.winner.salience
        return self.winner.salience - max(t.salience for t in self.losers)

    def __repr__(self) -> str:
        return (
            f"CompetitionResult(winner={self.winner.item_id!r}, "
            f"losers={len(self.losers)}, delta={self.salience_delta:.3f})"
        )


def compete(candidates: Sequence[FocusToken]) -> CompetitionResult:
    """Run a winner-take-all competition among candidate tokens."""
    if not candidates:
        raise AttentionError(
            operation="compete",
            reason="No candidates provided for competition",
        )
    ordered = sorted(candidates, key=lambda t: t.salience, reverse=True)
    winner = ordered[0]
    losers = tuple(ordered[1:])
    best_loser_s = losers[0].salience if losers else 0.0
    return CompetitionResult(
        winner=winner,
        losers=losers,
        reason="highest_salience",
        salience_delta=winner.salience - best_loser_s,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. AttentionShift — audit trail
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AttentionShift:
    """Immutable record of a change in attentional focus."""

    from_focus: Tuple[str, ...]
    to_focus: Tuple[str, ...]
    trigger: str
    timestamp: float

    def __init__(
        self,
        from_focus: Union[List[str], Tuple[str, ...]],
        to_focus: Union[List[str], Tuple[str, ...]],
        trigger: str,
        timestamp: Optional[float] = None,
    ) -> None:
        object.__setattr__(self, "from_focus", tuple(from_focus))
        object.__setattr__(self, "to_focus", tuple(to_focus))
        object.__setattr__(self, "trigger", trigger)
        object.__setattr__(self, "timestamp", timestamp if timestamp is not None else time.time())

    @property
    def net_change(self) -> int:
        return len(self.to_focus) - len(self.from_focus)

    def __repr__(self) -> str:
        return (
            f"AttentionShift(removed={len(self.from_focus)}, "
            f"added={len(self.to_focus)}, trigger={self.trigger!r})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 9. AttentionGate — ValidationGate subclass gating on attention state
# ═══════════════════════════════════════════════════════════════════════════════

class _AttendedValidator(Validator):
    """Internal validator that checks whether data is currently attended."""

    name = "attended_validator"

    def __init__(self, frame: AttentionFrame) -> None:
        self._frame = frame

    def validate(self, data: Any, context: Any = None) -> ValidationResult:
        item_id: Optional[str] = None
        if isinstance(data, NRSIData):
            item_id = data.id
        elif isinstance(data, dict):
            item_id = data.get("item_id") or data.get("id")
        elif isinstance(data, str):
            item_id = data
        elif hasattr(data, "item_id"):
            item_id = getattr(data, "item_id")

        if item_id is None:
            return ValidationResult(
                passed=False,
                confidence=Confidence.NONE,
                validator_name=self.name,
                details="Cannot extract item_id from data",
            )

        if self._frame.is_attended(item_id):
            token = self._frame.get(item_id)
            salience = token.salience if token else 0.0
            return ValidationResult(
                passed=True,
                confidence=_clamp01(max(salience, Confidence.MEDIUM)),
                validator_name=self.name,
                details=f"Item {item_id!r} is attended (salience={salience:.3f})",
            )

        return ValidationResult(
            passed=False,
            confidence=Confidence.NONE,
            validator_name=self.name,
            details=f"Item {item_id!r} is not in the attention frame",
        )


class AttentionGate(ValidationGate):
    """Validation gate that only passes data currently in the attention frame.

    Usage::

        frame = AttentionFrame(capacity=7)
        gate = AttentionGate(frame, name="focus_gate")
        result = gate.process(some_data)  # raises if not attended
    """

    def __init__(
        self,
        frame: AttentionFrame,
        name: str = "attention_gate",
        confidence_threshold: float = Confidence.MEDIUM,
        target_trust: TrustLevel = TrustLevel.VALIDATED,
    ) -> None:
        self._attention_frame = frame
        super().__init__(
            name=name,
            confidence_threshold=confidence_threshold,
            validators=[_AttendedValidator(frame)],
            target_trust=target_trust,
            require_all=True,
            audit=True,
        )

    @property
    def frame(self) -> AttentionFrame:
        return self._attention_frame


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

@dataclass
class AttentionFocus:
    """Result returned by AttentionController.focus()."""
    target: str = ""
    salience: float = 0.5
    suppressed_lobes: List[str] = field(default_factory=list)
    frame_snapshot: Optional[List[FocusToken]] = None


class AttentionController:
    """Facade wrapping AttentionFrame + SalienceMap for the NRS pipeline.

    Provides the ``focus(query, domain, mode)`` API that
    ``nrsi.core.nrs._process_inner`` expects.
    """

    def __init__(
        self,
        capacity: int = 7,
        policy: Optional[AttentionPolicy] = None,
    ) -> None:
        self._policy = policy or AttentionPolicy(max_capacity=capacity)
        self._frame = AttentionFrame(
            capacity=self._policy.max_capacity,
        )
        self._salience_map = SalienceMap()

    def focus(
        self,
        query: str,
        *,
        domain: str = "general",
        mode: str = "HYBRID",
    ) -> AttentionFocus:
        """Determine attentional focus for *query* and return an AttentionFocus."""
        token = FocusToken(
            item_id=f"q:{query[:64]}",
            content=query,
            salience=0.8,
            source_lobe=domain,
        )
        if self._policy.admits(token):
            self._frame.attend(token)

        suppressed: List[str] = []
        if self._policy.suppress_domains:
            suppressed = list(self._policy.suppress_domains)

        return AttentionFocus(
            target=query[:128],
            salience=token.salience,
            suppressed_lobes=suppressed,
            frame_snapshot=self._frame.snapshot(),
        )

    @property
    def frame(self) -> AttentionFrame:
        return self._frame
