"""NRSI Normative Types — Ethical and Policy Reasoning Primitives.

Deontic logic for AGI: what SHOULD be done, what's PERMITTED, what's FORBIDDEN.

  Obligation   — Impulse MUST do X (or face violation)
  Permission   — Impulse MAY do X
  Prohibition  — Impulse MUST NOT do X
  Norm         — A normative rule with conditions, scope, and precedence
  NormConflict — When norms contradict, how to resolve
  HarmAssessment — Structured harm evaluation for decisions

Patent-covered: NRSI Normative Reasoning System, VelarIQ.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Deontic primitives
# ═══════════════════════════════════════════════════════════════════════════════

class DeonticType(Enum):
    """Modal operators from deontic logic."""

    OBLIGATION = auto()    # Impulse MUST perform the action
    PERMISSION = auto()    # Impulse MAY perform the action
    PROHIBITION = auto()   # Impulse MUST NOT perform the action
    EXEMPTION = auto()     # Impulse is released from a standing obligation
    POWER = auto()         # Authority to create, modify, or revoke norms


class NormScope(Enum):
    """Spatial / contextual reach of a norm."""

    GLOBAL = auto()         # Applies to all agents everywhere
    DOMAIN = auto()         # Applies within a named knowledge domain
    SESSION = auto()        # Applies within a single user session
    QUERY = auto()          # Applies to the current query only
    IMPULSE_SPECIFIC = auto()  # Applies to one named Impulse instance


class ConflictType(Enum):
    """How two norms contradict each other."""

    OBLIGATION_PROHIBITION = auto()  # Must do X vs must not do X
    COMPETING_OBLIGATIONS = auto()   # Must do X vs must do Y (mutually exclusive)
    PERMISSION_PROHIBITION = auto()  # May do X vs must not do X
    SCOPE_OVERLAP = auto()           # Same action, different scopes disagree
    TEMPORAL_OVERLAP = auto()        # Norms valid in overlapping windows disagree


class ResolutionStrategy(Enum):
    """How a norm conflict is resolved."""

    PRIORITY = auto()      # Higher-priority norm wins
    SPECIFICITY = auto()   # More specific scope wins
    RECENCY = auto()       # More recently enacted norm wins
    AUTHORITY = auto()     # Norm from higher-authority enactor wins
    PROHIBITION_WINS = auto()  # Safety default: prohibition overrides permission


class VerdictType(Enum):
    """Outcome of checking an action against a norm set."""

    PERMITTED = auto()
    OBLIGATED = auto()
    PROHIBITED = auto()
    CONFLICTED = auto()


class HarmCategory(Enum):
    """Broad categories of potential harm."""

    PHYSICAL = auto()
    PSYCHOLOGICAL = auto()
    FINANCIAL = auto()
    REPUTATIONAL = auto()
    PRIVACY = auto()
    DISCRIMINATION = auto()
    MISINFORMATION = auto()
    ENVIRONMENTAL = auto()
    AUTONOMY = auto()


class EthicalFramework(Enum):
    """Major ethical traditions used for multi-lens evaluation."""

    DEONTOLOGICAL = auto()     # Rule-based (Kantian)
    CONSEQUENTIALIST = auto()  # Outcome-based (utilitarian)
    VIRTUE_ETHICS = auto()     # Character-based (Aristotelian)
    CARE_ETHICS = auto()       # Relationship-based
    RIGHTS_BASED = auto()      # Individual rights as constraints


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Norm — a single normative rule
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Norm:
    """A normative rule with conditions, scope, and precedence.

    ``condition`` is a callable predicate that receives an arbitrary
    context dict and returns True when this norm is in force.
    """

    norm_id: str
    deontic_type: DeonticType
    description: str
    condition: Callable[[Dict[str, Any]], bool] = field(repr=False)
    scope: NormScope = NormScope.GLOBAL
    priority: int = 0
    domain: Optional[str] = None
    action: Optional[str] = None
    enacted_by: Optional[str] = None
    active: bool = True
    valid_from: Optional[float] = None
    valid_until: Optional[float] = None

    # ── helpers ────────────────────────────────────────────────────────────

    def applies_to(self, context: Dict[str, Any]) -> bool:
        """Return True when this norm is in force for *context*."""
        if not self.active:
            return False
        now = time.time()
        if self.valid_from is not None and now < self.valid_from:
            return False
        if self.valid_until is not None and now > self.valid_until:
            return False
        try:
            return self.condition(context)
        except Exception:
            return False

    def is_obligation(self) -> bool:
        return self.deontic_type is DeonticType.OBLIGATION

    def is_prohibition(self) -> bool:
        return self.deontic_type is DeonticType.PROHIBITION

    def is_permission(self) -> bool:
        return self.deontic_type is DeonticType.PERMISSION


# ═══════════════════════════════════════════════════════════════════════════════
# 3. NormConflict — when two norms disagree
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class NormConflict:
    """A detected conflict between two norms."""

    norm_a: Norm
    norm_b: Norm
    conflict_type: ConflictType
    resolution_strategy: ResolutionStrategy = ResolutionStrategy.PRIORITY

    @property
    def explanation(self) -> str:
        return (
            f"Conflict ({self.conflict_type.name}): "
            f"norm '{self.norm_a.norm_id}' ({self.norm_a.deontic_type.name}) "
            f"vs norm '{self.norm_b.norm_id}' ({self.norm_b.deontic_type.name}). "
            f"Suggested resolution: {self.resolution_strategy.name}."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. NormVerdict — result of checking an action against norms
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NormVerdict:
    """Structured outcome of a norm check on a proposed action."""

    action: str
    verdict: VerdictType
    applicable_norms: List[Norm] = field(default_factory=list)
    explanation: str = ""
    overridden_norms: List[Norm] = field(default_factory=list)

    @property
    def is_allowed(self) -> bool:
        return self.verdict in (VerdictType.PERMITTED, VerdictType.OBLIGATED)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. NormSet — collection with conflict resolution
# ═══════════════════════════════════════════════════════════════════════════════

class NormSet:
    """Manages a set of ``Norm`` instances and resolves conflicts.

    Norms are stored by id.  ``check_action`` evaluates deontic status
    by collecting applicable norms and applying priority ordering.
    """

    def __init__(self) -> None:
        self._norms: Dict[str, Norm] = {}
        self._removal_log: List[Tuple[str, str, float]] = []

    # ── mutation ───────────────────────────────────────────────────────────

    def add_norm(self, norm: Norm) -> None:
        self._norms[norm.norm_id] = norm

    def remove_norm(self, norm_id: str, reason: str) -> None:
        if norm_id in self._norms:
            self._removal_log.append((norm_id, reason, time.time()))
            del self._norms[norm_id]

    # ── querying ───────────────────────────────────────────────────────────

    @property
    def norms(self) -> List[Norm]:
        return list(self._norms.values())

    def applicable_norms(self, context: Dict[str, Any]) -> List[Norm]:
        """Return norms that fire for *context*, ordered by priority desc."""
        matching = [n for n in self._norms.values() if n.applies_to(context)]
        matching.sort(key=lambda n: n.priority, reverse=True)
        return matching

    def check_action(
        self, action: str, context: Dict[str, Any],
    ) -> NormVerdict:
        """Evaluate deontic status of *action* under active norms."""
        applicable = self.applicable_norms(context)
        if not applicable:
            return NormVerdict(
                action=action,
                verdict=VerdictType.PERMITTED,
                explanation="No applicable norms — default permit.",
            )

        obligations: List[Norm] = []
        prohibitions: List[Norm] = []
        permissions: List[Norm] = []

        for norm in applicable:
            if norm.deontic_type is DeonticType.OBLIGATION:
                obligations.append(norm)
            elif norm.deontic_type is DeonticType.PROHIBITION:
                prohibitions.append(norm)
            elif norm.deontic_type is DeonticType.PERMISSION:
                permissions.append(norm)

        # Detect direct conflict: obligation + prohibition on same action
        if obligations and prohibitions:
            top_obligation = obligations[0]
            top_prohibition = prohibitions[0]
            if top_prohibition.priority >= top_obligation.priority:
                return NormVerdict(
                    action=action,
                    verdict=VerdictType.PROHIBITED,
                    applicable_norms=applicable,
                    explanation=(
                        f"Prohibition '{top_prohibition.norm_id}' "
                        f"(priority {top_prohibition.priority}) overrides obligation "
                        f"'{top_obligation.norm_id}' (priority {top_obligation.priority})."
                    ),
                    overridden_norms=obligations,
                )
            return NormVerdict(
                action=action,
                verdict=VerdictType.CONFLICTED,
                applicable_norms=applicable,
                explanation=(
                    f"Conflict: obligation '{top_obligation.norm_id}' vs "
                    f"prohibition '{top_prohibition.norm_id}' — "
                    f"obligation has higher priority but prohibition exists."
                ),
            )

        if prohibitions:
            return NormVerdict(
                action=action,
                verdict=VerdictType.PROHIBITED,
                applicable_norms=prohibitions,
                explanation=f"Prohibited by '{prohibitions[0].norm_id}'.",
                overridden_norms=permissions,
            )

        if obligations:
            return NormVerdict(
                action=action,
                verdict=VerdictType.OBLIGATED,
                applicable_norms=obligations,
                explanation=f"Obligated by '{obligations[0].norm_id}'.",
            )

        return NormVerdict(
            action=action,
            verdict=VerdictType.PERMITTED,
            applicable_norms=permissions,
            explanation="Permitted (no prohibition or obligation in scope).",
        )

    def conflicts(self) -> List[NormConflict]:
        """Detect pairwise conflicts across all stored norms."""
        result: List[NormConflict] = []
        norms = list(self._norms.values())
        for i, a in enumerate(norms):
            for b in norms[i + 1:]:
                ctype = _detect_conflict_type(a, b)
                if ctype is not None:
                    result.append(NormConflict(
                        norm_a=a,
                        norm_b=b,
                        conflict_type=ctype,
                        resolution_strategy=_suggest_resolution(a, b),
                    ))
        return result


def _detect_conflict_type(a: Norm, b: Norm) -> Optional[ConflictType]:
    """Heuristic pair-wise conflict detection."""
    pair = frozenset((a.deontic_type, b.deontic_type))

    if pair == frozenset((DeonticType.OBLIGATION, DeonticType.PROHIBITION)):
        return ConflictType.OBLIGATION_PROHIBITION

    if (a.deontic_type is DeonticType.OBLIGATION
            and b.deontic_type is DeonticType.OBLIGATION
            and a.domain == b.domain):
        return ConflictType.COMPETING_OBLIGATIONS

    if pair == frozenset((DeonticType.PERMISSION, DeonticType.PROHIBITION)):
        return ConflictType.PERMISSION_PROHIBITION

    return None


def _suggest_resolution(a: Norm, b: Norm) -> ResolutionStrategy:
    """Pick a default resolution strategy from norm metadata."""
    if a.priority != b.priority:
        return ResolutionStrategy.PRIORITY
    if a.scope != b.scope:
        return ResolutionStrategy.SPECIFICITY
    a_from = a.valid_from or 0.0
    b_from = b.valid_from or 0.0
    if a_from != b_from:
        return ResolutionStrategy.RECENCY
    return ResolutionStrategy.PROHIBITION_WINS


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Harm assessment
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StakeholderImpact:
    """Impact assessment for one stakeholder group."""

    stakeholder_group: str
    positive_impacts: List[str] = field(default_factory=list)
    negative_impacts: List[str] = field(default_factory=list)

    @property
    def net_assessment(self) -> str:
        p = len(self.positive_impacts)
        n = len(self.negative_impacts)
        if p > n:
            return "net_positive"
        if n > p:
            return "net_negative"
        return "balanced"


@dataclass
class HarmAssessment:
    """Structured harm evaluation for a proposed action."""

    action_description: str
    affected_parties: List[str] = field(default_factory=list)
    harm_categories: Dict[HarmCategory, float] = field(default_factory=dict)
    mitigations: List[str] = field(default_factory=list)
    justification: str = ""
    stakeholder_impacts: List[StakeholderImpact] = field(default_factory=list)

    def __post_init__(self) -> None:
        for cat, sev in self.harm_categories.items():
            if not (0.0 <= sev <= 1.0):
                raise ValueError(
                    f"Severity for {cat.name} must be in [0.0, 1.0], got {sev}"
                )

    @property
    def total_severity(self) -> float:
        """Mean severity across all harm categories (0.0 if none)."""
        if not self.harm_categories:
            return 0.0
        return sum(self.harm_categories.values()) / len(self.harm_categories)

    @property
    def max_severity(self) -> float:
        if not self.harm_categories:
            return 0.0
        return max(self.harm_categories.values())

    @property
    def acceptable(self) -> bool:
        """Acceptable when no single category exceeds 0.7 and mean < 0.5."""
        return self.max_severity <= 0.7 and self.total_severity < 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# 7. NormativeReasoner — orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

_ALWAYS_TRUE: Callable[[Dict[str, Any]], bool] = lambda _ctx: True


class NormativeReasoner:
    """High-level API for deontic evaluation, harm assessment, and conflict
    resolution over a ``NormSet``.
    """

    def __init__(self, norms: Optional[NormSet] = None) -> None:
        self._norms = norms or NormSet()

    @property
    def norm_set(self) -> NormSet:
        return self._norms

    # ── core operations ────────────────────────────────────────────────────

    def evaluate_action(
        self,
        action: str,
        context: Dict[str, Any],
        norms: Optional[NormSet] = None,
    ) -> NormVerdict:
        """Check deontic status of *action* under optional override norm set."""
        ns = norms if norms is not None else self._norms
        return ns.check_action(action, context)

    def assess_harm(
        self,
        action: str,
        stakeholders: Sequence[StakeholderImpact],
        harm_scores: Optional[Dict[HarmCategory, float]] = None,
        mitigations: Optional[List[str]] = None,
    ) -> HarmAssessment:
        """Build a ``HarmAssessment`` from stakeholder impacts."""
        affected = [s.stakeholder_group for s in stakeholders]
        cats: Dict[HarmCategory, float] = harm_scores or {}

        if not cats and stakeholders:
            neg_count = sum(len(s.negative_impacts) for s in stakeholders)
            if neg_count > 0:
                severity = min(1.0, neg_count * 0.15)
                cats[HarmCategory.PSYCHOLOGICAL] = severity

        return HarmAssessment(
            action_description=action,
            affected_parties=affected,
            harm_categories=cats,
            mitigations=mitigations or [],
            stakeholder_impacts=list(stakeholders),
        )

    def resolve_conflict(
        self,
        norm_a: Norm,
        norm_b: Norm,
        context: Dict[str, Any],
    ) -> Norm:
        """Return the winning norm when two conflict in *context*."""
        strategy = _suggest_resolution(norm_a, norm_b)

        if strategy is ResolutionStrategy.PRIORITY:
            return norm_a if norm_a.priority >= norm_b.priority else norm_b

        if strategy is ResolutionStrategy.SPECIFICITY:
            scope_rank = {
                NormScope.QUERY: 5,
                NormScope.IMPULSE_SPECIFIC: 4,
                NormScope.SESSION: 3,
                NormScope.DOMAIN: 2,
                NormScope.GLOBAL: 1,
            }
            a_rank = scope_rank.get(norm_a.scope, 0)
            b_rank = scope_rank.get(norm_b.scope, 0)
            return norm_a if a_rank >= b_rank else norm_b

        if strategy is ResolutionStrategy.RECENCY:
            a_from = norm_a.valid_from or 0.0
            b_from = norm_b.valid_from or 0.0
            return norm_a if a_from >= b_from else norm_b

        # PROHIBITION_WINS or AUTHORITY — safety default
        if norm_a.is_prohibition():
            return norm_a
        if norm_b.is_prohibition():
            return norm_b
        return norm_a if norm_a.priority >= norm_b.priority else norm_b

    def check(
        self,
        query: str,
        *,
        domain: str = "general",
        confidence: float = 1.0,
        claim_type: str = "",
        contains_pii: bool = False,
        epistemic: str = "",
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> NormVerdict:
        """Pipeline-facing alias for ``evaluate_action``.

        Builds a context dict from *query*, *domain*, and optional enrichment
        fields, then delegates to ``evaluate_action``.  Returns a
        ``NormVerdict`` whose ``is_allowed`` / ``verdict`` fields let the
        pipeline read ``violations`` and ``is_compliant``.
        """
        ctx: Dict[str, Any] = {
            "query": query,
            "domain": domain,
            "confidence": confidence,
            "claim_type": claim_type,
            "contains_pii": contains_pii,
            "epistemic": epistemic,
        }
        if extra_context:
            ctx.update(extra_context)
        verdict = self.evaluate_action(action=query, context=ctx)

        _active_violations: List[str] = []
        q_lower = query.lower()
        for n in verdict.applicable_norms:
            if n.deontic_type is not DeonticType.PROHIBITION:
                continue
            nid = n.norm_id.lower()
            triggered = False
            if "medical" in nid and domain in ("medical", "health"):
                if claim_type == "factual" or confidence < 0.8:
                    triggered = True
            elif "legal" in nid and domain == "legal":
                if claim_type == "factual" or confidence < 0.8:
                    triggered = True
            elif "financial" in nid and domain == "financial":
                triggered = True
            elif ("pii" in nid or "personal_data" in nid or "privacy" in nid) and contains_pii:
                triggered = True
            elif "discriminat" in nid:
                _disc_terms = {"racist", "sexist", "discriminat", "slur", "hate"}
                if any(t in q_lower for t in _disc_terms):
                    triggered = True
            elif "speculative" in nid and confidence < 0.5 and claim_type == "factual":
                triggered = True
            elif "harm" in nid:
                _harm_terms = {
                    "hurt", "kill", "weapon", "bomb", "poison", "attack",
                    "hack", "hacking", "exploit", "malware", "phishing",
                    "ddos", "ransomware", "brute force", "crack password",
                    "steal data", "bypass security", "break into",
                    "how to hack", "how do i hack", "how can i hack",
                }
                if any(t in q_lower for t in _harm_terms):
                    triggered = True
            if triggered:
                _active_violations.append(n.description)

        verdict.violations = _active_violations  # type: ignore[attr-defined]
        _compliant = len(_active_violations) == 0
        verdict.is_compliant = _compliant  # type: ignore[attr-defined]
        if not _compliant:
            verdict.verdict = VerdictType.PROHIBITED
        return verdict

    def explain_decision(self, verdict: NormVerdict) -> str:
        """Human-readable explanation of a ``NormVerdict``."""
        lines: List[str] = [
            f"Action: {verdict.action}",
            f"Verdict: {verdict.verdict.name}",
        ]
        if verdict.explanation:
            lines.append(f"Reason: {verdict.explanation}")
        if verdict.applicable_norms:
            lines.append("Applicable norms:")
            for n in verdict.applicable_norms:
                lines.append(
                    f"  [{n.deontic_type.name}] {n.norm_id} "
                    f"(priority={n.priority}, scope={n.scope.name}): "
                    f"{n.description}"
                )
        if verdict.overridden_norms:
            lines.append("Overridden norms:")
            for n in verdict.overridden_norms:
                lines.append(f"  {n.norm_id}: {n.description}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Default norms — safety baseline
# ═══════════════════════════════════════════════════════════════════════════════

def default_safety_norms() -> NormSet:
    """A starter ``NormSet`` encoding fundamental safety prohibitions."""
    ns = NormSet()

    ns.add_norm(Norm(
        norm_id="safety.no_physical_harm",
        deontic_type=DeonticType.PROHIBITION,
        description="Must not recommend actions that cause physical harm.",
        condition=_ALWAYS_TRUE,
        scope=NormScope.GLOBAL,
        priority=1000,
        domain="safety",
        enacted_by="system",
    ))

    ns.add_norm(Norm(
        norm_id="safety.no_discrimination",
        deontic_type=DeonticType.PROHIBITION,
        description="Must not produce discriminatory outputs.",
        condition=_ALWAYS_TRUE,
        scope=NormScope.GLOBAL,
        priority=1000,
        domain="safety",
        enacted_by="system",
    ))

    ns.add_norm(Norm(
        norm_id="safety.no_deception",
        deontic_type=DeonticType.PROHIBITION,
        description="Must not deliberately deceive users.",
        condition=_ALWAYS_TRUE,
        scope=NormScope.GLOBAL,
        priority=950,
        domain="safety",
        enacted_by="system",
    ))

    ns.add_norm(Norm(
        norm_id="safety.explain_uncertainty",
        deontic_type=DeonticType.OBLIGATION,
        description="Must disclose uncertainty when confidence < 0.8.",
        condition=lambda ctx: ctx.get("confidence", 1.0) < 0.8,
        scope=NormScope.GLOBAL,
        priority=800,
        domain="transparency",
        enacted_by="system",
    ))

    ns.add_norm(Norm(
        norm_id="safety.cite_sources",
        deontic_type=DeonticType.OBLIGATION,
        description="Must cite sources for factual claims.",
        condition=lambda ctx: ctx.get("claim_type") == "factual",
        scope=NormScope.GLOBAL,
        priority=700,
        domain="transparency",
        enacted_by="system",
    ))

    ns.add_norm(Norm(
        norm_id="safety.privacy_protection",
        deontic_type=DeonticType.PROHIBITION,
        description="Must not disclose private personal information.",
        condition=lambda ctx: ctx.get("contains_pii", False),
        scope=NormScope.GLOBAL,
        priority=950,
        domain="privacy",
        enacted_by="system",
    ))

    return ns
