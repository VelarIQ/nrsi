"""NRSI Belief Revision — Principled Knowledge Update Under Contradiction.

When new evidence contradicts existing beliefs, which beliefs should change?
This is the belief revision problem (AGM theory). These types provide:

  BeliefBase      — Set of beliefs with entrenchment ordering
  BeliefDependency — DAG tracking which beliefs support which
  Contraction     — Remove a belief while minimizing collateral damage
  Revision        — Add new belief, contracting contradictions first
  Entrenchment    — How resistant a belief is to removal (0.0-1.0)
  RevisionResult  — What changed, what was retracted, audit trail

AGM Postulates implemented:
  Success: After revision by P, P is believed
  Inclusion: Revision doesn't add anything beyond P and consequences
  Vacuity: If ¬P not believed, revision = expansion
  Consistency: Revision is consistent if P is consistent
  Extensionality: Logically equivalent P and Q produce same revision

Patent-covered: NRSI Belief Revision System, VelarIQ.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

from nrsi.core.errors import NRSIError, AuditRequiredError


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Entrenchment — Resistance to removal
# ═══════════════════════════════════════════════════════════════════════════════

class Entrenchment(IntEnum):
    """How resistant a belief is to removal during contraction.

    Higher values mean harder to retract.  AXIOMATIC beliefs are protected
    by default and require explicit policy override to remove.

    Ordering is total:  SPECULATIVE < HYPOTHETICAL < INFERRED < EMPIRICAL < AXIOMATIC
    """

    SPECULATIVE  = 0
    HYPOTHETICAL = 1
    INFERRED     = 2
    EMPIRICAL    = 3
    AXIOMATIC    = 4


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BeliefEntry — A single belief in the base
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BeliefEntry:
    """A belief with entrenchment, dependency links, and provenance.

    Attributes:
        belief_id:     Unique identifier.
        content:       Human-readable proposition.
        entrenchment:  How resistant to removal (see Entrenchment enum).
        depends_on:    Belief IDs this belief was derived from.
        supports:      Belief IDs this belief provides evidence for.
        confidence:    Subjective probability, 0.0-1.0.
        epistemic_type: Origin kind (e.g. "deductive", "observational").
        added_at:      Unix timestamp when the belief entered the base.
        source:        Free-text provenance (agent, gate, document, …).
        contradicts:   Belief IDs this belief explicitly negates.
    """

    belief_id: str
    content: str
    entrenchment: Entrenchment = Entrenchment.HYPOTHETICAL
    depends_on: List[str] = field(default_factory=list)
    supports: List[str] = field(default_factory=list)
    confidence: float = 0.5
    epistemic_type: str = "observational"
    added_at: float = field(default_factory=time.time)
    source: str = ""
    contradicts: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.belief_id:
            raise ValueError("belief_id must be non-empty")
        if not isinstance(self.confidence, (int, float)):
            raise TypeError(
                f"confidence must be numeric, got {type(self.confidence).__name__}"
            )
        self.confidence = float(self.confidence)
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )

    def __repr__(self) -> str:
        return (
            f"BeliefEntry({self.belief_id!r}, "
            f"entrenchment={self.entrenchment.name}, "
            f"confidence={self.confidence:.2f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Audit & Policy Types
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AuditEntry:
    """Immutable audit record for a belief-base operation."""

    timestamp: float
    action: str
    belief_id: str
    reason: str
    detail: str = ""

    def __str__(self) -> str:
        parts = [f"[{self.action}] {self.belief_id}"]
        if self.reason:
            parts.append(f"reason={self.reason}")
        if self.detail:
            parts.append(self.detail)
        return " | ".join(parts)


@dataclass(frozen=True)
class RevisionPolicy:
    """Configuration governing belief revision behaviour.

    Attributes:
        max_cascade_depth:          Stop cascade propagation after this many
                                    waves of transitive removal.
        protect_entrenchment_above: Beliefs at or above this level are
                                    immune to contraction unless policy is
                                    relaxed.
        require_audit:              If True, operations without a reason
                                    string raise AuditRequiredError.
        allow_axiom_revision:       If False (default), AXIOMATIC beliefs
                                    are completely immutable.
    """

    max_cascade_depth: int = 10
    protect_entrenchment_above: Entrenchment = Entrenchment.EMPIRICAL
    require_audit: bool = True
    allow_axiom_revision: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Result Types
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ContractionResult:
    """Outcome of a belief contraction (removal) operation."""

    removed: Tuple[BeliefEntry, ...]
    retained: Tuple[BeliefEntry, ...]
    cascade_depth: int
    audit: Tuple[AuditEntry, ...]

    @property
    def removed_ids(self) -> List[str]:
        return [b.belief_id for b in self.removed]

    @property
    def retained_ids(self) -> List[str]:
        return [b.belief_id for b in self.retained]

    def __repr__(self) -> str:
        return (
            f"ContractionResult(removed={len(self.removed)}, "
            f"retained={len(self.retained)}, cascade_depth={self.cascade_depth})"
        )


@dataclass(frozen=True)
class RevisionResult:
    """Outcome of a belief revision (contraction + expansion)."""

    added: Tuple[BeliefEntry, ...]
    removed: Tuple[BeliefEntry, ...]
    retained: Tuple[BeliefEntry, ...]
    consistency_maintained: bool
    audit: Tuple[AuditEntry, ...]

    @property
    def added_ids(self) -> List[str]:
        return [b.belief_id for b in self.added]

    @property
    def removed_ids(self) -> List[str]:
        return [b.belief_id for b in self.removed]

    def __repr__(self) -> str:
        return (
            f"RevisionResult(added={len(self.added)}, removed={len(self.removed)}, "
            f"retained={len(self.retained)}, consistent={self.consistency_maintained})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BeliefDependency — DAG of belief-to-belief links
# ═══════════════════════════════════════════════════════════════════════════════

class BeliefDependency:
    """Directed acyclic graph of belief dependencies.

    An edge A → B means "belief A depends on belief B" (B is a foundation
    of A).  Transitive queries let the revision engine discover the full
    impact zone of removing a single belief.
    """

    __slots__ = ("_forward", "_reverse")

    def __init__(self) -> None:
        self._forward: Dict[str, Set[str]] = {}
        self._reverse: Dict[str, Set[str]] = {}

    # ── mutation ───────────────────────────────────────────────────────

    def add(self, belief_id: str, depends_on: Optional[List[str]] = None) -> None:
        """Register *belief_id* and its declared dependencies."""
        deps = set(depends_on) if depends_on else set()
        self._forward.setdefault(belief_id, set()).update(deps)
        self._reverse.setdefault(belief_id, set())
        for dep in deps:
            self._reverse.setdefault(dep, set()).add(belief_id)

    def remove(self, belief_id: str) -> None:
        """Remove *belief_id* and all edges touching it."""
        for dep in self._forward.pop(belief_id, set()):
            self._reverse.get(dep, set()).discard(belief_id)
        for dependent in self._reverse.pop(belief_id, set()):
            self._forward.get(dependent, set()).discard(belief_id)

    # ── direct queries ─────────────────────────────────────────────────

    def direct_dependents(self, belief_id: str) -> Set[str]:
        """Beliefs that directly depend on *belief_id*."""
        return set(self._reverse.get(belief_id, set()))

    def direct_foundations(self, belief_id: str) -> Set[str]:
        """Beliefs that *belief_id* directly depends on."""
        return set(self._forward.get(belief_id, set()))

    # ── transitive queries ─────────────────────────────────────────────

    def dependents(self, belief_id: str) -> Set[str]:
        """Transitive closure: everything that depends on *belief_id*."""
        result: Set[str] = set()
        queue: deque[str] = deque(self._reverse.get(belief_id, set()))
        while queue:
            current = queue.popleft()
            if current not in result:
                result.add(current)
                queue.extend(self._reverse.get(current, set()) - result)
        return result

    def foundations(self, belief_id: str) -> Set[str]:
        """Transitive closure: everything *belief_id* rests on."""
        result: Set[str] = set()
        queue: deque[str] = deque(self._forward.get(belief_id, set()))
        while queue:
            current = queue.popleft()
            if current not in result:
                result.add(current)
                queue.extend(self._forward.get(current, set()) - result)
        return result

    def is_foundational(self, belief_id: str) -> bool:
        """True if *belief_id* has no dependencies (it is a root node)."""
        return len(self._forward.get(belief_id, set())) == 0

    # ── structural queries ─────────────────────────────────────────────

    def topological_sort(self) -> List[str]:
        """Return belief IDs in dependency-safe order (foundations first).

        Raises ValueError if the graph contains a cycle.
        """
        all_ids = set(self._forward.keys()) | set(self._reverse.keys())
        in_degree: Dict[str, int] = {
            bid: len(self._forward.get(bid, set())) for bid in all_ids
        }

        queue: deque[str] = deque(
            sorted(bid for bid, deg in in_degree.items() if deg == 0)
        )
        result: List[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for dependent in sorted(self._reverse.get(node, set())):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(all_ids):
            raise ValueError(
                f"Belief dependency graph contains a cycle "
                f"(sorted {len(result)} of {len(all_ids)} nodes)"
            )
        return result

    def has_cycle(self) -> bool:
        """True if the dependency graph contains a cycle."""
        try:
            self.topological_sort()
            return False
        except ValueError:
            return True

    # ── dunder ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(set(self._forward.keys()) | set(self._reverse.keys()))

    def __contains__(self, belief_id: str) -> bool:
        return belief_id in self._forward or belief_id in self._reverse

    def __repr__(self) -> str:
        return f"BeliefDependency(nodes={len(self)})"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. BeliefBase — Ordered set of beliefs with AGM revision
# ═══════════════════════════════════════════════════════════════════════════════

class BeliefBase:
    """Ordered set of beliefs with AGM-compliant contraction and revision.

    Beliefs are ordered by entrenchment (lowest → easiest to retract).
    Contraction removes beliefs while minimizing collateral damage by
    preferring to retract lower-entrenched beliefs first.  Revision adds
    new evidence by first contracting any contradictions, then expanding.

    Thread-safety: none — callers must synchronize externally.
    """

    __slots__ = ("_beliefs", "_deps", "_policy", "_audit")

    def __init__(self, policy: Optional[RevisionPolicy] = None) -> None:
        self._beliefs: Dict[str, BeliefEntry] = {}
        self._deps: BeliefDependency = BeliefDependency()
        self._policy: RevisionPolicy = policy or RevisionPolicy()
        self._audit: List[AuditEntry] = []

    # ── properties ─────────────────────────────────────────────────────

    @property
    def policy(self) -> RevisionPolicy:
        return self._policy

    @property
    def audit_log(self) -> List[AuditEntry]:
        return list(self._audit)

    @property
    def dependency_graph(self) -> BeliefDependency:
        return self._deps

    # ── internal helpers ───────────────────────────────────────────────

    def _record(
        self, action: str, belief_id: str, reason: str, detail: str = "",
    ) -> AuditEntry:
        entry = AuditEntry(
            timestamp=time.time(),
            action=action,
            belief_id=belief_id,
            reason=reason,
            detail=detail,
        )
        self._audit.append(entry)
        return entry

    def _check_protected(self, entry: BeliefEntry) -> None:
        """Raise if *entry* is protected by policy."""
        if (
            entry.entrenchment == Entrenchment.AXIOMATIC
            and not self._policy.allow_axiom_revision
        ):
            raise NRSIError(
                f"Cannot remove axiomatic belief {entry.belief_id!r}",
                suggestion="Set allow_axiom_revision=True in RevisionPolicy",
            )
        if entry.entrenchment >= self._policy.protect_entrenchment_above:
            raise NRSIError(
                f"Belief {entry.belief_id!r} ({entry.entrenchment.name}) is at or "
                f"above protection threshold "
                f"({self._policy.protect_entrenchment_above.name})",
                suggestion="Lower protect_entrenchment_above in RevisionPolicy",
            )

    def _compute_cascade(
        self, belief_id: str,
    ) -> Tuple[List[str], int]:
        """Compute the minimal set of belief IDs to remove and cascade depth.

        A dependent belief is removed only when ALL of its in-base
        foundations land in the removal set (no remaining support).
        """
        to_remove: Set[str] = {belief_id}
        depth = 0
        changed = True

        while changed:
            changed = False
            wave: Set[str] = set()
            for bid, entry in self._beliefs.items():
                if bid in to_remove or not entry.depends_on:
                    continue
                active_deps = [
                    d for d in entry.depends_on if d in self._beliefs
                ]
                if active_deps and all(d in to_remove for d in active_deps):
                    wave.add(bid)

            if wave:
                to_remove |= wave
                depth += 1
                changed = True
                if depth >= self._policy.max_cascade_depth:
                    break

        return sorted(to_remove), depth

    def _find_conflicts(self, new_belief: BeliefEntry) -> List[BeliefEntry]:
        """Existing beliefs that *new_belief* contradicts (both directions)."""
        conflicts: List[BeliefEntry] = []
        seen: Set[str] = set()

        for cid in new_belief.contradicts:
            if cid in self._beliefs and cid not in seen:
                conflicts.append(self._beliefs[cid])
                seen.add(cid)

        for entry in self._beliefs.values():
            if (
                new_belief.belief_id in entry.contradicts
                and entry.belief_id not in seen
            ):
                conflicts.append(entry)
                seen.add(entry.belief_id)

        return conflicts

    # ── expansion ──────────────────────────────────────────────────────

    def add(self, entry: BeliefEntry) -> None:
        """Add a belief (expansion — no contradiction checking).

        Raises ValueError if the belief ID already exists.
        Use revise() to add evidence that may contradict existing beliefs.
        """
        if entry.belief_id in self._beliefs:
            raise ValueError(
                f"Belief {entry.belief_id!r} already exists; "
                f"use revise() to update under contradiction"
            )
        self._beliefs[entry.belief_id] = entry
        self._deps.add(entry.belief_id, entry.depends_on)

        for dep_id in entry.depends_on:
            if dep_id in self._beliefs:
                dep = self._beliefs[dep_id]
                if entry.belief_id not in dep.supports:
                    dep.supports.append(entry.belief_id)

        self._record(
            "added", entry.belief_id, "expansion",
            detail=f"entrenchment={entry.entrenchment.name}",
        )

    # ── contraction ────────────────────────────────────────────────────

    def minimal_contraction_set(self, belief_id: str) -> List[str]:
        """Smallest set of belief IDs whose removal is forced by removing
        *belief_id* (includes *belief_id* itself).

        A dependent is included only when ALL its in-base foundations are
        in the removal set.
        """
        if belief_id not in self._beliefs:
            raise KeyError(f"Belief {belief_id!r} not found")
        ids, _ = self._compute_cascade(belief_id)
        return ids

    def contract(self, belief_id: str, reason: str) -> ContractionResult:
        """Remove *belief_id* with minimal collateral damage.

        Protected beliefs (at or above the policy entrenchment threshold,
        or AXIOMATIC when axiom revision is disabled) cannot be contracted.
        Protected beliefs in the cascade survive instead of blocking the
        entire operation.
        """
        if self._policy.require_audit and not reason:
            raise AuditRequiredError(
                "contract", suggestion="Provide a non-empty reason string",
            )
        if belief_id not in self._beliefs:
            raise KeyError(f"Belief {belief_id!r} not found")

        self._check_protected(self._beliefs[belief_id])

        cascade_ids, raw_depth = self._compute_cascade(belief_id)

        final_removal: List[str] = []
        for rid in cascade_ids:
            entry = self._beliefs[rid]
            if rid != belief_id:
                try:
                    self._check_protected(entry)
                except NRSIError:
                    self._record(
                        "cascade_skipped", rid, reason,
                        detail=f"protected at {entry.entrenchment.name}",
                    )
                    continue
            final_removal.append(rid)

        removed_entries: List[BeliefEntry] = []
        audit_entries: List[AuditEntry] = []

        for rid in final_removal:
            entry = self._beliefs.pop(rid)
            self._deps.remove(rid)

            for other in self._beliefs.values():
                if rid in other.supports:
                    other.supports = [s for s in other.supports if s != rid]
                if rid in other.depends_on:
                    other.depends_on = [d for d in other.depends_on if d != rid]
                if rid in other.contradicts:
                    other.contradicts = [c for c in other.contradicts if c != rid]

            removed_entries.append(entry)
            audit_entries.append(self._record(
                "contracted", rid, reason,
                detail="primary" if rid == belief_id else f"cascade(depth≤{raw_depth})",
            ))

        retained = tuple(self._beliefs.values())
        return ContractionResult(
            removed=tuple(removed_entries),
            retained=retained,
            cascade_depth=raw_depth,
            audit=tuple(audit_entries),
        )

    # ── revision (AGM: contraction then expansion) ─────────────────────

    def revise(self, new_belief: BeliefEntry, reason: str) -> RevisionResult:
        """Add *new_belief*, first contracting any contradictions.

        Implements the AGM postulates:
          Success    — *new_belief* is in the base after revision.
          Vacuity    — If nothing contradicts, this is simple expansion.
          Consistency — The result is consistent if *new_belief* is consistent.
        """
        if self._policy.require_audit and not reason:
            raise AuditRequiredError(
                "revise", suggestion="Provide a non-empty reason string",
            )

        conflicts = self._find_conflicts(new_belief)

        if not conflicts:
            self.add(new_belief)
            ae = self._record(
                "revised", new_belief.belief_id, reason,
                detail="vacuity — no contradictions",
            )
            retained = tuple(self._beliefs.values())
            return RevisionResult(
                added=(new_belief,),
                removed=(),
                retained=retained,
                consistency_maintained=True,
                audit=(ae,),
            )

        conflicts.sort(key=lambda e: e.entrenchment)

        all_removed: List[BeliefEntry] = []
        all_audit: List[AuditEntry] = []

        for conflict in conflicts:
            if conflict.belief_id not in self._beliefs:
                continue
            try:
                cr = self.contract(
                    conflict.belief_id,
                    reason=f"contradicted by {new_belief.belief_id}",
                )
                all_removed.extend(cr.removed)
                all_audit.extend(cr.audit)
            except NRSIError:
                all_audit.append(self._record(
                    "revision_blocked", conflict.belief_id, reason,
                    detail=f"protected at {conflict.entrenchment.name}",
                ))

        if new_belief.belief_id in self._beliefs:
            self._beliefs.pop(new_belief.belief_id)
            self._deps.remove(new_belief.belief_id)

        self.add(new_belief)
        all_audit.append(self._record(
            "revised", new_belief.belief_id, reason,
            detail=f"contracted {len(all_removed)} beliefs",
        ))

        consistency = len(self.contradictions()) == 0
        retained = tuple(self._beliefs.values())
        return RevisionResult(
            added=(new_belief,),
            removed=tuple(all_removed),
            retained=retained,
            consistency_maintained=consistency,
            audit=tuple(all_audit),
        )

    # ── queries ────────────────────────────────────────────────────────

    def query(self, content: str) -> Optional[BeliefEntry]:
        """Find a belief by exact content match."""
        for entry in self._beliefs.values():
            if entry.content == content:
                return entry
        return None

    def get(self, belief_id: str) -> Optional[BeliefEntry]:
        """Retrieve a belief by ID, or None."""
        return self._beliefs.get(belief_id)

    def contradictions(self) -> List[Tuple[BeliefEntry, BeliefEntry]]:
        """All pairs of co-existing beliefs that explicitly contradict."""
        pairs: List[Tuple[BeliefEntry, BeliefEntry]] = []
        seen: Set[Tuple[str, str]] = set()
        for entry in self._beliefs.values():
            for cid in entry.contradicts:
                if cid in self._beliefs:
                    pair_key = (min(entry.belief_id, cid), max(entry.belief_id, cid))
                    if pair_key not in seen:
                        seen.add(pair_key)
                        pairs.append((entry, self._beliefs[cid]))
        return pairs

    def by_entrenchment(self, level: Entrenchment) -> List[BeliefEntry]:
        """All beliefs at a specific entrenchment level."""
        return [e for e in self._beliefs.values() if e.entrenchment == level]

    def ordered(self) -> List[BeliefEntry]:
        """All beliefs ordered by entrenchment (lowest first)."""
        return sorted(self._beliefs.values(), key=lambda e: (e.entrenchment, e.belief_id))

    def all_beliefs(self) -> List[BeliefEntry]:
        """Snapshot of every belief currently in the base."""
        return list(self._beliefs.values())

    # ── dunder ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._beliefs)

    def __contains__(self, belief_id: str) -> bool:
        return belief_id in self._beliefs

    def __iter__(self) -> Iterator[BeliefEntry]:
        return iter(self._beliefs.values())

    def __repr__(self) -> str:
        return f"BeliefBase(size={len(self._beliefs)}, policy={self._policy!r})"


# ═══════════════════════════════════════════════════════════════════════════════
# Module Exports
# ═══════════════════════════════════════════════════════════════════════════════

class BeliefRevisionEngine:
    """Pipeline facade providing the ``update(claim, confidence, domain)`` API
    that ``nrsi.core.nrs._process_inner`` expects.

    Wraps a ``BeliefBase`` and performs auto-revision when a new claim
    arrives, returning a summary result.
    """

    def __init__(self, base: Optional[BeliefBase] = None) -> None:
        self._base = base or BeliefBase(
            policy=RevisionPolicy(require_audit=False)
        )
        self._interaction_count = 0

    def update(
        self,
        *,
        claim: str,
        confidence: float = 0.5,
        domain: str = "general",
    ) -> "BeliefUpdateResult":
        """Integrate *claim* into the belief base and return a summary."""
        self._interaction_count += 1
        bid = f"auto-{self._interaction_count}"

        entrenchment = Entrenchment.SPECULATIVE
        if confidence >= 0.85:
            entrenchment = Entrenchment.EMPIRICAL
        elif confidence >= 0.6:
            entrenchment = Entrenchment.INFERRED
        elif confidence >= 0.4:
            entrenchment = Entrenchment.HYPOTHETICAL

        entry = BeliefEntry(
            belief_id=bid,
            content=claim[:512],
            entrenchment=entrenchment,
            confidence=max(0.0, min(1.0, confidence)),
            epistemic_type="observational",
            source=f"pipeline-{domain}",
        )

        existing = self._base.query(claim[:512])
        was_revised = False

        if existing:
            try:
                entry.contradicts = [existing.belief_id]
                result = self._base.revise(entry, reason=f"updated from pipeline ({domain})")
                was_revised = len(result.removed) > 0
            except Exception:
                pass
        else:
            try:
                self._base.add(entry)
            except ValueError:
                pass

        contradictions = self._base.contradictions()
        consistency = 1.0 if not contradictions else max(0.0, 1.0 - len(contradictions) * 0.1)

        return BeliefUpdateResult(
            was_revised=was_revised,
            consistency_score=consistency,
            belief_count=len(self._base),
        )

    @property
    def base(self) -> BeliefBase:
        return self._base


@dataclass
class BeliefUpdateResult:
    """Result from ``BeliefRevisionEngine.update()``."""
    was_revised: bool = False
    consistency_score: float = 1.0
    belief_count: int = 0


__all__ = [
    "Entrenchment",
    "BeliefEntry",
    "AuditEntry",
    "RevisionPolicy",
    "ContractionResult",
    "RevisionResult",
    "BeliefDependency",
    "BeliefBase",
    "BeliefRevisionEngine",
    "BeliefUpdateResult",
]
