"""NRSI Inter-Impulse Trust — Impulse Trust and Negotiation within Constructs.

When multiple NRS Impulses (autonomous reasoning units) communicate within
or across Constructs (organizational containers), they need formal trust:
  Who do I trust for what? How much? Based on what evidence?
  What tasks can I delegate? What commitments have been made?

  ImpulseReputation  — Trust profile for a known Impulse
  DelegationContract — Formal task delegation with acceptance criteria
  NegotiationState   — State machine for multi-round negotiation
  TrustPolicy        — Rules for extending/revoking trust
  ImpulseRole        — Typed roles in multi-Impulse collaboration

Terminology:
  Construct — organizational container / ecosystem for NRS intelligence.
  Impulse   — individual autonomous reasoning unit within a Construct.

Patent-covered: NRSI Inter-Impulse Trust System, VelarIQ.
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
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ImpulseId — Node / Impulse identity
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ImpulseId:
    """Immutable identifier for an NRS Impulse (autonomous reasoning unit)."""

    node_id: str
    display_name: str
    public_key_hash: str
    first_seen: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImpulseId):
            return NotImplemented
        return self.node_id == other.node_id

    def __hash__(self) -> int:
        return hash(self.node_id)

    def short(self) -> str:
        return f"{self.display_name}({self.node_id[:8]})"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TrustDimension — Multi-axis trust
# ═══════════════════════════════════════════════════════════════════════════════

class TrustDimension(Enum):
    """Axes of trust between Impulses (Mayer-Davis-Schoorman model)."""

    COMPETENCE = auto()
    INTEGRITY = auto()
    BENEVOLENCE = auto()
    RELIABILITY = auto()
    PREDICTABILITY = auto()


_DEFAULT_WEIGHTS: Dict[TrustDimension, float] = {
    TrustDimension.COMPETENCE: 0.30,
    TrustDimension.INTEGRITY: 0.25,
    TrustDimension.BENEVOLENCE: 0.10,
    TrustDimension.RELIABILITY: 0.20,
    TrustDimension.PREDICTABILITY: 0.15,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ImpulseReputation — Trust profile for a known Impulse
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ImpulseReputation:
    """Per-Impulse trust scores across multiple dimensions.

    Trust is updated via Bayesian-style weighted increments:
    on success the score moves toward 1.0 proportional to (1 − current),
    on failure it moves toward 0.0 proportional to current.
    """

    impulse_id: ImpulseId
    trust_scores: Dict[TrustDimension, float] = field(
        default_factory=lambda: {d: 0.5 for d in TrustDimension},
    )
    interaction_count: int = 0
    successful_interactions: int = 0
    failed_interactions: int = 0
    domains_trusted: List[str] = field(default_factory=list)
    last_interaction: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for dim, score in self.trust_scores.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Trust score for {dim.name} must be in [0.0, 1.0], got {score}"
                )

    # ── Aggregation ───────────────────────────────────────────────────────

    def overall_trust(
        self, weights: Optional[Dict[TrustDimension, float]] = None,
    ) -> float:
        """Weighted average trust across all dimensions."""
        w = weights or _DEFAULT_WEIGHTS
        total_w = sum(w.get(d, 0.0) for d in TrustDimension)
        if total_w == 0.0:
            return 0.0
        return sum(
            self.trust_scores.get(d, 0.0) * w.get(d, 0.0)
            for d in TrustDimension
        ) / total_w

    def trust_for_domain(self, domain: str) -> float:
        """Return overall trust if *domain* is in the trusted set, else 0."""
        if domain in self.domains_trusted:
            return self.overall_trust()
        return 0.0

    # ── Update ────────────────────────────────────────────────────────────

    def update(
        self,
        success: bool,
        domain: Optional[str] = None,
        dimensions: Optional[Sequence[TrustDimension]] = None,
        learning_rate: float = 0.10,
    ) -> None:
        """Bayesian-style trust update after an interaction.

        ``learning_rate`` controls how much a single observation shifts
        the score.  Higher rates make the system more reactive.
        """
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError(f"learning_rate must be in (0.0, 1.0], got {learning_rate}")

        dims = dimensions or list(TrustDimension)
        for d in dims:
            cur = self.trust_scores.get(d, 0.5)
            if success:
                self.trust_scores[d] = cur + learning_rate * (1.0 - cur)
            else:
                self.trust_scores[d] = cur - learning_rate * cur

        self.interaction_count += 1
        if success:
            self.successful_interactions += 1
        else:
            self.failed_interactions += 1

        if domain and domain not in self.domains_trusted and success:
            self.domains_trusted.append(domain)

        self.last_interaction = time.monotonic()

    # ── Query ─────────────────────────────────────────────────────────────

    def is_trusted_for(self, task_type: str, min_trust: float = 0.6) -> bool:
        return self.trust_for_domain(task_type) >= min_trust

    def success_rate(self) -> float:
        if self.interaction_count == 0:
            return 0.0
        return self.successful_interactions / self.interaction_count

    def summary(self) -> Dict[str, Any]:
        return {
            "impulse": self.impulse_id.short(),
            "overall_trust": round(self.overall_trust(), 4),
            "interactions": self.interaction_count,
            "success_rate": round(self.success_rate(), 4),
            "domains": self.domains_trusted,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DelegationContract — Formal task delegation
# ═══════════════════════════════════════════════════════════════════════════════

class ContractStatus(Enum):
    PROPOSED = auto()
    ACCEPTED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    REJECTED = auto()


_TERMINAL_STATES: FrozenSet[ContractStatus] = frozenset({
    ContractStatus.COMPLETED,
    ContractStatus.FAILED,
    ContractStatus.REJECTED,
})


@dataclass
class DelegationContract:
    """A typed contract between a delegator and a delegate.

    The contract encodes *what* is being delegated, *what* trust is
    required per dimension, acceptance criteria, and a deadline.
    State transitions follow PROPOSED → ACCEPTED → IN_PROGRESS →
    COMPLETED|FAILED, with REJECTED as an early exit.
    """

    contract_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    delegator: ImpulseId = field(default_factory=lambda: ImpulseId("", "", ""))
    delegate: ImpulseId = field(default_factory=lambda: ImpulseId("", "", ""))
    task_description: str = ""
    acceptance_criteria: str = ""
    deadline: Optional[float] = None
    trust_required: Dict[TrustDimension, float] = field(default_factory=dict)

    status: ContractStatus = ContractStatus.PROPOSED
    created_at: float = field(default_factory=time.monotonic)
    completed_at: Optional[float] = None
    evidence: Optional[str] = None
    rejection_reason: Optional[str] = None
    failure_reason: Optional[str] = None

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_STATES

    def _require_status(self, *expected: ContractStatus) -> None:
        if self.status not in expected:
            names = ", ".join(s.name for s in expected)
            raise ValueError(
                f"Contract {self.contract_id} is {self.status.name}; "
                f"expected one of: {names}"
            )

    def accept(self) -> None:
        self._require_status(ContractStatus.PROPOSED)
        self.status = ContractStatus.ACCEPTED

    def start(self) -> None:
        self._require_status(ContractStatus.ACCEPTED)
        self.status = ContractStatus.IN_PROGRESS

    def reject(self, reason: str) -> None:
        self._require_status(ContractStatus.PROPOSED)
        self.status = ContractStatus.REJECTED
        self.rejection_reason = reason
        self.completed_at = time.monotonic()

    def complete(self, evidence: str) -> None:
        self._require_status(ContractStatus.IN_PROGRESS, ContractStatus.ACCEPTED)
        self.status = ContractStatus.COMPLETED
        self.evidence = evidence
        self.completed_at = time.monotonic()

    def fail(self, reason: str) -> None:
        self._require_status(ContractStatus.IN_PROGRESS, ContractStatus.ACCEPTED)
        self.status = ContractStatus.FAILED
        self.failure_reason = reason
        self.completed_at = time.monotonic()

    def meets_trust_requirements(self, reputation: ImpulseReputation) -> bool:
        """Check whether *reputation* satisfies every trust threshold."""
        for dim, required in self.trust_required.items():
            actual = reputation.trust_scores.get(dim, 0.0)
            if actual < required:
                return False
        return True

    def is_overdue(self) -> bool:
        if self.deadline is None or self.is_terminal:
            return False
        return time.monotonic() > self.deadline


# ═══════════════════════════════════════════════════════════════════════════════
# 5. NegotiationState — Multi-round negotiation state machine
# ═══════════════════════════════════════════════════════════════════════════════

class NegotiationStatus(Enum):
    OPEN = auto()
    PROPOSAL = auto()
    COUNTER = auto()
    AGREED = auto()
    DEADLOCKED = auto()
    WITHDRAWN = auto()


@dataclass(frozen=True)
class Offer:
    """A single negotiation offer from one party."""

    impulse_id: ImpulseId
    content: Dict[str, Any]
    round_number: int
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class NegotiationState:
    """State machine for multi-round negotiation between Impulses.

    Tracks parties, offers, rounds, and terminal states (AGREED,
    DEADLOCKED, WITHDRAWN).  Each action validates that the
    requesting Impulse is a party and the negotiation is still open.
    """

    negotiation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parties: List[ImpulseId] = field(default_factory=list)
    topic: str = ""
    status: NegotiationStatus = NegotiationStatus.OPEN
    offers: List[Offer] = field(default_factory=list)
    current_round: int = 0
    max_rounds: int = 10
    agreements: Dict[str, Any] = field(default_factory=dict)
    withdrawal_reason: Optional[str] = None

    def _require_party(self, impulse_id: ImpulseId) -> None:
        if impulse_id not in self.parties:
            raise ValueError(
                f"Impulse {impulse_id.short()} is not a party to "
                f"negotiation {self.negotiation_id}"
            )

    def _require_open(self) -> None:
        terminal = {NegotiationStatus.AGREED, NegotiationStatus.DEADLOCKED, NegotiationStatus.WITHDRAWN}
        if self.status in terminal:
            raise ValueError(
                f"Negotiation {self.negotiation_id} is already {self.status.name}"
            )

    def propose(self, impulse_id: ImpulseId, offer_content: Dict[str, Any]) -> Offer:
        self._require_party(impulse_id)
        self._require_open()

        self.current_round += 1
        if self.current_round > self.max_rounds:
            self.status = NegotiationStatus.DEADLOCKED
            raise ValueError(
                f"Max rounds ({self.max_rounds}) exceeded — DEADLOCKED"
            )

        offer = Offer(
            impulse_id=impulse_id,
            content=offer_content,
            round_number=self.current_round,
        )
        self.offers.append(offer)
        self.status = NegotiationStatus.PROPOSAL
        return offer

    def counter(self, impulse_id: ImpulseId, counter_content: Dict[str, Any]) -> Offer:
        self._require_party(impulse_id)
        self._require_open()

        if not self.offers:
            raise ValueError("Cannot counter without a prior offer")

        self.current_round += 1
        if self.current_round > self.max_rounds:
            self.status = NegotiationStatus.DEADLOCKED
            raise ValueError(
                f"Max rounds ({self.max_rounds}) exceeded — DEADLOCKED"
            )

        offer = Offer(
            impulse_id=impulse_id,
            content=counter_content,
            round_number=self.current_round,
        )
        self.offers.append(offer)
        self.status = NegotiationStatus.COUNTER
        return offer

    def accept(self, impulse_id: ImpulseId) -> None:
        self._require_party(impulse_id)
        self._require_open()

        if not self.offers:
            raise ValueError("Cannot accept — no offers on the table")

        last_offer = self.offers[-1]
        if last_offer.impulse_id == impulse_id:
            raise ValueError("Cannot accept your own offer")

        self.status = NegotiationStatus.AGREED
        self.agreements = dict(last_offer.content)

    def withdraw(self, impulse_id: ImpulseId, reason: str) -> None:
        self._require_party(impulse_id)
        self._require_open()
        self.status = NegotiationStatus.WITHDRAWN
        self.withdrawal_reason = reason

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            NegotiationStatus.AGREED,
            NegotiationStatus.DEADLOCKED,
            NegotiationStatus.WITHDRAWN,
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "negotiation_id": self.negotiation_id,
            "topic": self.topic,
            "status": self.status.name,
            "parties": [p.short() for p in self.parties],
            "rounds": self.current_round,
            "offers": len(self.offers),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TrustPolicy — Rules for extending/revoking trust
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrustPolicy:
    """System-wide parameters governing trust dynamics.

    ``trust_decay_rate`` is applied per time-unit to inactive agents.
    ``require_evidence_for_trust_above`` forces audit evidence before
    trust exceeds the threshold.
    """

    min_trust_for_delegation: float = 0.6
    trust_decay_rate: float = 0.01
    trust_recovery_rate: float = 0.05
    require_evidence_for_trust_above: float = 0.9
    blacklist: Set[str] = field(default_factory=set)

    def is_blacklisted(self, impulse_id: ImpulseId) -> bool:
        return impulse_id.node_id in self.blacklist

    def can_delegate(self, reputation: ImpulseReputation) -> bool:
        if self.is_blacklisted(reputation.impulse_id):
            return False
        return reputation.overall_trust() >= self.min_trust_for_delegation

    def apply_decay(self, reputation: ImpulseReputation, elapsed_units: float) -> None:
        """Decay trust scores for an inactive Impulse."""
        decay = self.trust_decay_rate * elapsed_units
        for dim in TrustDimension:
            cur = reputation.trust_scores.get(dim, 0.5)
            reputation.trust_scores[dim] = max(0.0, cur - decay * cur)

    def apply_recovery(
        self, reputation: ImpulseReputation, elapsed_units: float,
    ) -> None:
        """Recover trust toward baseline after rehabilitation."""
        baseline = 0.5
        rate = self.trust_recovery_rate * elapsed_units
        for dim in TrustDimension:
            cur = reputation.trust_scores.get(dim, 0.0)
            if cur < baseline:
                reputation.trust_scores[dim] = min(baseline, cur + rate * (baseline - cur))

    def requires_evidence(self, trust_score: float) -> bool:
        return trust_score >= self.require_evidence_for_trust_above

    def blacklist_impulse(self, impulse_id: ImpulseId, reason: str = "") -> None:
        self.blacklist.add(impulse_id.node_id)

    def unblacklist_impulse(self, impulse_id: ImpulseId) -> None:
        self.blacklist.discard(impulse_id.node_id)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ImpulseRole — Typed roles in multi-Impulse collaboration
# ═══════════════════════════════════════════════════════════════════════════════

class ImpulseRole(Enum):
    """Named roles for coalition participants."""

    LEADER = auto()
    PEER = auto()
    SPECIALIST = auto()
    VALIDATOR = auto()
    OBSERVER = auto()

    @property
    def can_delegate(self) -> bool:
        return self in {ImpulseRole.LEADER, ImpulseRole.PEER}

    @property
    def can_validate(self) -> bool:
        return self in {ImpulseRole.VALIDATOR, ImpulseRole.LEADER}

    @property
    def is_passive(self) -> bool:
        return self == ImpulseRole.OBSERVER


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Coalition — Group of Impulses with a shared goal (a Construct)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Coalition:
    """A named group of Impulses collaborating toward a shared goal within a Construct.

    The *trust_matrix* maps (node_id, node_id) pairs to a trust float,
    capturing the pairwise trust landscape within the Construct.
    """

    coalition_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    members: Dict[ImpulseId, ImpulseRole] = field(default_factory=dict)
    shared_goal: str = ""
    trust_matrix: Dict[Tuple[str, str], float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)

    def add_member(self, impulse: ImpulseId, role: ImpulseRole) -> None:
        self.members[impulse] = role
        for existing in self.members:
            if existing != impulse:
                pair_a = (existing.node_id, impulse.node_id)
                pair_b = (impulse.node_id, existing.node_id)
                self.trust_matrix.setdefault(pair_a, 0.5)
                self.trust_matrix.setdefault(pair_b, 0.5)

    def remove_member(self, impulse: ImpulseId) -> None:
        self.members.pop(impulse, None)
        to_remove = [
            key for key in self.trust_matrix
            if impulse.node_id in key
        ]
        for key in to_remove:
            del self.trust_matrix[key]

    def get_role(self, impulse: ImpulseId) -> Optional[ImpulseRole]:
        return self.members.get(impulse)

    def pairwise_trust(self, a: ImpulseId, b: ImpulseId) -> float:
        return self.trust_matrix.get((a.node_id, b.node_id), 0.0)

    def update_pairwise_trust(
        self, a: ImpulseId, b: ImpulseId, score: float,
    ) -> None:
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Trust score must be in [0.0, 1.0], got {score}")
        self.trust_matrix[(a.node_id, b.node_id)] = score

    def leaders(self) -> List[ImpulseId]:
        return [a for a, r in self.members.items() if r == ImpulseRole.LEADER]

    def validators(self) -> List[ImpulseId]:
        return [a for a, r in self.members.items() if r == ImpulseRole.VALIDATOR]

    def specialists(self) -> List[ImpulseId]:
        return [a for a, r in self.members.items() if r == ImpulseRole.SPECIALIST]

    def mean_trust(self) -> float:
        scores = list(self.trust_matrix.values())
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def summary(self) -> Dict[str, Any]:
        return {
            "coalition_id": self.coalition_id,
            "shared_goal": self.shared_goal,
            "member_count": len(self.members),
            "roles": {a.short(): r.name for a, r in self.members.items()},
            "mean_trust": round(self.mean_trust(), 4),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Facade — API expected by nrsi.core.nrs._process_inner
# ═══════════════════════════════════════════════════════════════════════════════

class InterImpulseTrustManager:
    """Facade providing the ``record(instance_id, confidence, h_score, domain)``
    API that ``nrsi.core.nrs._process_inner`` expects.

    Maintains per-Impulse reputation records and exposes them for pipeline audit.
    """

    def __init__(self) -> None:
        self._reputations: Dict[str, ImpulseReputation] = {}

    def record(
        self,
        *,
        instance_id: str,
        confidence: float = 0.5,
        h_score: float = 0.0,
        domain: str = "general",
    ) -> None:
        """Record an interaction outcome for *instance_id*."""
        rep = self._reputations.get(instance_id)
        if rep is None:
            impulse = ImpulseId(
                node_id=instance_id,
                display_name=instance_id,
                public_key_hash="",
            )
            rep = ImpulseReputation(impulse_id=impulse, domains_trusted=[domain])
            self._reputations[instance_id] = rep

        success = ((confidence + h_score) / 2.0) >= 0.5
        rep.update(success=success, domain=domain)

        if domain and domain not in rep.domains_trusted:
            if rep.overall_trust() >= 0.6:
                rep.domains_trusted.append(domain)

    def get_reputation(self, instance_id: str) -> Optional[ImpulseReputation]:
        return self._reputations.get(instance_id)

    @property
    def all_reputations(self) -> Dict[str, ImpulseReputation]:
        return dict(self._reputations)
