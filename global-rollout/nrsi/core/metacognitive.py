"""NRSI Metacognitive Types — Knowing What You Know and Don't Know.

Second-order knowledge: reasoning about your own reasoning.

The ModeVector has a numeric 'metacognitive' dimension, but that's a
control knob, not structured metaknowledge.  These types express:
  - What the system knows it knows (grounded knowledge)
  - What it knows it doesn't know (identified gaps)
  - What it doesn't know it doesn't know (blind spots, discovered post-hoc)
  - How calibrated its confidence is (meta-confidence)
  - What its competence boundaries are (domain scope)

Every judgment and calibration record feeds the audit trail, so the system
can explain not just *what* it answered but *how certain it was that it
could answer*, and whether that certainty was historically warranted.

Patent-covered: NRSI Metacognitive Self-Assessment System, VelarIQ.
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
    TypeVar,
    Union,
)

from nrsi.core.types import Confidence, NRSIData, TrustLevel
from nrsi.core.errors import NRSIError


T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════════
# Errors
# ═══════════════════════════════════════════════════════════════════════════════

class MetacognitiveError(NRSIError):
    """Raised when a metacognitive operation encounters an inconsistency."""

    def __init__(
        self,
        operation: str,
        reason: str,
        suggestion: Optional[str] = None,
    ):
        self.operation = operation
        self.reason = reason
        msg = f"Metacognitive error during '{operation}': {reason}"
        super().__init__(msg, suggestion)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MetaKnowledgeState — what the system knows about what it knows
# ═══════════════════════════════════════════════════════════════════════════════

class MetaKnowledgeState(Enum):
    """Second-order knowledge status for a claim or assertion.

    Ordered from most grounded to least tractable:
      KNOWN           — verified claim with supporting evidence
      BELIEVED        — held with reasonable confidence, not yet verified
      SUSPENDED       — previously believed, now under active review
      UNKNOWN_KNOWN   — the system knows this is a gap it needs to fill
      UNKNOWN_UNKNOWN — blind spot, discovered only in retrospect
    """

    KNOWN           = auto()
    BELIEVED        = auto()
    SUSPENDED       = auto()
    UNKNOWN_KNOWN   = auto()
    UNKNOWN_UNKNOWN = auto()

    @property
    def is_grounded(self) -> bool:
        return self == MetaKnowledgeState.KNOWN

    @property
    def is_held(self) -> bool:
        return self in (MetaKnowledgeState.KNOWN, MetaKnowledgeState.BELIEVED)

    @property
    def is_gap(self) -> bool:
        return self in (
            MetaKnowledgeState.UNKNOWN_KNOWN,
            MetaKnowledgeState.UNKNOWN_UNKNOWN,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. KnowledgeAssertion — a claim with metacognitive annotation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KnowledgeAssertion:
    """A single knowledge claim annotated with its metacognitive status."""

    claim_id: str
    content: Any
    meta_state: MetaKnowledgeState
    grounding: List[str] = field(default_factory=list)
    confidence: float = 0.5
    domain_scope: str = "general"
    created_at: float = field(default_factory=time.time)
    last_reviewed: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.confidence = Confidence.validate(self.confidence)

    def review(self, new_state: MetaKnowledgeState, reason: str = "") -> None:
        """Transition the assertion to a new meta-state."""
        self.meta_state = new_state
        self.last_reviewed = time.time()

    def add_evidence(self, evidence_id: str) -> None:
        if evidence_id not in self.grounding:
            self.grounding.append(evidence_id)

    @property
    def evidence_count(self) -> int:
        return len(self.grounding)

    @property
    def is_actionable(self) -> bool:
        """Whether this assertion can be used in downstream reasoning."""
        return self.meta_state in (
            MetaKnowledgeState.KNOWN,
            MetaKnowledgeState.BELIEVED,
        ) and self.confidence >= 0.3

    def __repr__(self) -> str:
        return (
            f"KnowledgeAssertion(id={self.claim_id!r}, "
            f"state={self.meta_state.name}, "
            f"conf={self.confidence:.2f}, "
            f"evidence={self.evidence_count})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. OpenQuestion — an identified knowledge gap
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OpenQuestion:
    """A question the system knows it needs answered."""

    question_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: str = ""
    why_it_matters: str = ""
    priority: float = 0.5
    domain: str = "general"
    blocking_goals: List[str] = field(default_factory=list)
    asked_at: float = field(default_factory=time.time)
    attempts: int = 0
    resolved: bool = False
    resolution: Optional[str] = None

    def __post_init__(self) -> None:
        self.priority = max(0.0, min(1.0, self.priority))

    def attempt(self) -> None:
        """Record an attempt to answer this question."""
        self.attempts += 1

    def resolve(self, answer: str) -> None:
        """Mark the question as resolved."""
        self.resolved = True
        self.resolution = answer

    @property
    def is_blocking(self) -> bool:
        return len(self.blocking_goals) > 0 and not self.resolved

    @property
    def age_seconds(self) -> float:
        return time.time() - self.asked_at

    def __repr__(self) -> str:
        status = "resolved" if self.resolved else f"open(attempts={self.attempts})"
        return (
            f"OpenQuestion({self.question!r}, "
            f"priority={self.priority:.2f}, {status})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CompetenceScope — domain competence self-assessment
# ═══════════════════════════════════════════════════════════════════════════════

class CompetenceLevel(Enum):
    """Capability level within a domain."""

    NONE       = 0
    BASIC      = 1
    COMPETENT  = 2
    EXPERT     = 3

    def __ge__(self, other: CompetenceLevel) -> bool:
        if not isinstance(other, CompetenceLevel):
            return NotImplemented
        return self.value >= other.value

    def __gt__(self, other: CompetenceLevel) -> bool:
        if not isinstance(other, CompetenceLevel):
            return NotImplemented
        return self.value > other.value

    def __le__(self, other: CompetenceLevel) -> bool:
        if not isinstance(other, CompetenceLevel):
            return NotImplemented
        return self.value <= other.value

    def __lt__(self, other: CompetenceLevel) -> bool:
        if not isinstance(other, CompetenceLevel):
            return NotImplemented
        return self.value < other.value


@dataclass
class CompetenceScope:
    """Self-assessed capability level for a specific domain."""

    domain: str
    level: CompetenceLevel
    known_limitations: List[str] = field(default_factory=list)
    last_calibrated: float = field(default_factory=time.time)

    def recalibrate(self, new_level: CompetenceLevel, limitations: Optional[List[str]] = None) -> None:
        self.level = new_level
        if limitations is not None:
            self.known_limitations = list(limitations)
        self.last_calibrated = time.time()

    @property
    def age_since_calibration_s(self) -> float:
        return time.time() - self.last_calibrated

    def __repr__(self) -> str:
        return (
            f"CompetenceScope(domain={self.domain!r}, "
            f"level={self.level.name}, "
            f"limitations={len(self.known_limitations)})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CalibrationRecord / CalibrationScore — tracking predictive calibration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CalibrationRecord:
    """A single prediction-vs-outcome observation for calibration tracking."""

    record_id: str
    predicted_confidence: float
    actual_outcome: bool
    domain: str
    timestamp: float

    def __init__(
        self,
        predicted_confidence: float,
        actual_outcome: bool,
        domain: str = "general",
        record_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        object.__setattr__(self, "record_id", record_id or str(uuid.uuid4()))
        object.__setattr__(self, "predicted_confidence", Confidence.validate(predicted_confidence))
        object.__setattr__(self, "actual_outcome", actual_outcome)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "timestamp", timestamp if timestamp is not None else time.time())

    @property
    def error(self) -> float:
        """Absolute error: |predicted - actual|."""
        actual = 1.0 if self.actual_outcome else 0.0
        return abs(self.predicted_confidence - actual)

    @property
    def squared_error(self) -> float:
        actual = 1.0 if self.actual_outcome else 0.0
        return (self.predicted_confidence - actual) ** 2


@dataclass(frozen=True)
class CalibrationScore:
    """Aggregate calibration metrics computed from a set of records."""

    overconfidence_rate: float
    underconfidence_rate: float
    brier_score: float
    ece: float
    sample_count: int

    @property
    def is_well_calibrated(self) -> bool:
        return self.ece < 0.1 and self.sample_count >= 20

    @property
    def dominant_bias(self) -> str:
        if self.overconfidence_rate > self.underconfidence_rate + 0.05:
            return "overconfident"
        elif self.underconfidence_rate > self.overconfidence_rate + 0.05:
            return "underconfident"
        return "balanced"

    def __repr__(self) -> str:
        return (
            f"CalibrationScore(brier={self.brier_score:.4f}, "
            f"ece={self.ece:.4f}, n={self.sample_count}, "
            f"bias={self.dominant_bias})"
        )


def compute_calibration(records: Sequence[CalibrationRecord], n_bins: int = 10) -> CalibrationScore:
    """Compute calibration metrics from a sequence of records.

    Uses binned Expected Calibration Error (ECE) with *n_bins* equal-width
    bins over the [0, 1] confidence interval.
    """
    if not records:
        return CalibrationScore(
            overconfidence_rate=0.0,
            underconfidence_rate=0.0,
            brier_score=0.0,
            ece=0.0,
            sample_count=0,
        )

    overconfident = 0
    underconfident = 0
    brier_sum = 0.0
    n = len(records)

    bins: List[List[CalibrationRecord]] = [[] for _ in range(n_bins)]
    for rec in records:
        brier_sum += rec.squared_error
        actual_val = 1.0 if rec.actual_outcome else 0.0
        if rec.predicted_confidence > actual_val + 0.1:
            overconfident += 1
        elif rec.predicted_confidence < actual_val - 0.1:
            underconfident += 1
        bin_idx = min(int(rec.predicted_confidence * n_bins), n_bins - 1)
        bins[bin_idx].append(rec)

    # ECE: weighted average of |accuracy - confidence| per bin
    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(r.predicted_confidence for r in b) / len(b)
        avg_acc = sum(1.0 if r.actual_outcome else 0.0 for r in b) / len(b)
        ece += (len(b) / n) * abs(avg_acc - avg_conf)

    return CalibrationScore(
        overconfidence_rate=overconfident / n,
        underconfidence_rate=underconfident / n,
        brier_score=brier_sum / n,
        ece=ece,
        sample_count=n,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MetacognitiveJudgment — can-I-answer-this assessment
# ═══════════════════════════════════════════════════════════════════════════════

class JudgmentType(Enum):
    """Possible metacognitive judgments about a query."""

    CAN_ANSWER     = auto()
    CANNOT_ANSWER  = auto()
    UNCERTAIN      = auto()
    NEED_MORE_INFO = auto()
    OUT_OF_SCOPE   = auto()


@dataclass(frozen=True)
class MetacognitiveJudgment:
    """The system's assessment of whether it can answer a query."""

    query: str
    judgment_type: JudgmentType
    justification: str
    confidence_in_judgment: float
    domain: str = "general"
    timestamp: float = field(default_factory=time.time)

    def __init__(
        self,
        query: str,
        judgment_type: JudgmentType,
        justification: str,
        confidence_in_judgment: float,
        domain: str = "general",
        timestamp: Optional[float] = None,
    ) -> None:
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "judgment_type", judgment_type)
        object.__setattr__(self, "justification", justification)
        object.__setattr__(self, "confidence_in_judgment", Confidence.validate(confidence_in_judgment))
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "timestamp", timestamp if timestamp is not None else time.time())

    @property
    def should_attempt(self) -> bool:
        return self.judgment_type in (
            JudgmentType.CAN_ANSWER,
            JudgmentType.UNCERTAIN,
        ) and self.confidence_in_judgment >= 0.3

    @property
    def needs_escalation(self) -> bool:
        return self.judgment_type in (
            JudgmentType.CANNOT_ANSWER,
            JudgmentType.OUT_OF_SCOPE,
        )

    def __repr__(self) -> str:
        return (
            f"MetacognitiveJudgment({self.judgment_type.name}, "
            f"conf={self.confidence_in_judgment:.2f}, "
            f"domain={self.domain!r})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. BlindSpotRecord — discovered unknown-unknowns
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BlindSpotRecord:
    """A record of a previously unknown gap discovered after the fact."""

    spot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    discovered_at: float = field(default_factory=time.time)
    domain: str = "general"
    description: str = ""
    how_discovered: str = ""
    impact_assessment: str = ""
    severity: float = 0.5
    mitigated: bool = False

    def __post_init__(self) -> None:
        self.severity = max(0.0, min(1.0, self.severity))

    def mitigate(self, mitigation: str) -> None:
        self.mitigated = True
        self.impact_assessment = f"{self.impact_assessment} [Mitigated: {mitigation}]"

    def __repr__(self) -> str:
        status = "mitigated" if self.mitigated else f"severity={self.severity:.2f}"
        return (
            f"BlindSpotRecord(domain={self.domain!r}, "
            f"{status}, {self.description!r:.40})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MetacognitiveMonitor — central tracker for metacognitive state
# ═══════════════════════════════════════════════════════════════════════════════

class MetacognitiveMonitor:
    """Tracks metacognitive state: assertions, open questions, competence
    scopes, calibration records, and blind spots.

    Provides the core `assess`, `calibrate`, and introspection methods
    that the rest of the system calls to ask "can I handle this?".
    """

    __slots__ = (
        "_assertions", "_questions", "_competences",
        "_calibration_records", "_blind_spots", "_judgments",
    )

    def __init__(self) -> None:
        self._assertions: Dict[str, KnowledgeAssertion] = {}
        self._questions: Dict[str, OpenQuestion] = {}
        self._competences: Dict[str, CompetenceScope] = {}
        self._calibration_records: List[CalibrationRecord] = []
        self._blind_spots: List[BlindSpotRecord] = []
        self._judgments: List[MetacognitiveJudgment] = []

    # ── Assertions ────────────────────────────────────────────────────────

    def register_assertion(self, assertion: KnowledgeAssertion) -> None:
        self._assertions[assertion.claim_id] = assertion

    def get_assertion(self, claim_id: str) -> Optional[KnowledgeAssertion]:
        return self._assertions.get(claim_id)

    def assertions_by_state(self, state: MetaKnowledgeState) -> List[KnowledgeAssertion]:
        return [a for a in self._assertions.values() if a.meta_state == state]

    def assertions_for_domain(self, domain: str) -> List[KnowledgeAssertion]:
        return [a for a in self._assertions.values() if a.domain_scope == domain]

    # ── Open Questions ────────────────────────────────────────────────────

    def ask(self, question: OpenQuestion) -> None:
        self._questions[question.question_id] = question

    def get_open_questions(self, domain: Optional[str] = None) -> List[OpenQuestion]:
        qs = [q for q in self._questions.values() if not q.resolved]
        if domain:
            qs = [q for q in qs if q.domain == domain]
        return sorted(qs, key=lambda q: q.priority, reverse=True)

    def resolve_question(self, question_id: str, answer: str) -> None:
        q = self._questions.get(question_id)
        if q is None:
            raise MetacognitiveError(
                operation="resolve_question",
                reason=f"Question {question_id!r} not found",
            )
        q.resolve(answer)

    # ── Competence ────────────────────────────────────────────────────────

    def register_competence(self, scope: CompetenceScope) -> None:
        self._competences[scope.domain] = scope

    def competence_check(self, domain: str) -> CompetenceScope:
        """Retrieve the competence scope for *domain*, or a NONE-level default."""
        if domain in self._competences:
            return self._competences[domain]
        return CompetenceScope(
            domain=domain,
            level=CompetenceLevel.NONE,
            known_limitations=["No competence assessment available"],
        )

    # ── Calibration ───────────────────────────────────────────────────────

    def record_prediction(
        self,
        predicted_confidence: float,
        actual_outcome: bool,
        domain: str = "general",
    ) -> CalibrationRecord:
        rec = CalibrationRecord(
            predicted_confidence=predicted_confidence,
            actual_outcome=actual_outcome,
            domain=domain,
        )
        self._calibration_records.append(rec)
        return rec

    def calibrate(
        self,
        predictions: Optional[Sequence[CalibrationRecord]] = None,
        domain: Optional[str] = None,
        n_bins: int = 10,
    ) -> CalibrationScore:
        """Compute calibration score from stored records or provided ones."""
        records: Sequence[CalibrationRecord]
        if predictions is not None:
            records = predictions
        else:
            records = self._calibration_records
        if domain:
            records = [r for r in records if r.domain == domain]
        return compute_calibration(records, n_bins=n_bins)

    # ── Blind Spots ───────────────────────────────────────────────────────

    def report_blind_spot(
        self,
        domain: str,
        description: str,
        how_discovered: str,
        impact_assessment: str = "",
        severity: float = 0.5,
    ) -> BlindSpotRecord:
        spot = BlindSpotRecord(
            domain=domain,
            description=description,
            how_discovered=how_discovered,
            impact_assessment=impact_assessment,
            severity=severity,
        )
        self._blind_spots.append(spot)
        return spot

    def get_blind_spots(self, domain: Optional[str] = None, unmitigated_only: bool = False) -> List[BlindSpotRecord]:
        spots = list(self._blind_spots)
        if domain:
            spots = [s for s in spots if s.domain == domain]
        if unmitigated_only:
            spots = [s for s in spots if not s.mitigated]
        return sorted(spots, key=lambda s: s.severity, reverse=True)

    # ── Assessment (the core metacognitive act) ───────────────────────────

    def assess(self, query: str, domain: str = "general") -> MetacognitiveJudgment:
        """Produce a metacognitive judgment about whether the system can
        handle *query* in the given *domain*.

        This is a heuristic assessment based on registered competences,
        calibration history, blind spots, and open questions.
        """
        scope = self.competence_check(domain)
        cal = self.calibrate(domain=domain)
        blind_spots = self.get_blind_spots(domain=domain, unmitigated_only=True)
        open_qs = self.get_open_questions(domain=domain)

        if scope.level == CompetenceLevel.NONE:
            judgment = MetacognitiveJudgment(
                query=query,
                judgment_type=JudgmentType.OUT_OF_SCOPE,
                justification=f"No registered competence for domain {domain!r}",
                confidence_in_judgment=0.8,
                domain=domain,
            )
            self._judgments.append(judgment)
            return judgment

        if scope.level == CompetenceLevel.BASIC and len(blind_spots) > 2:
            judgment = MetacognitiveJudgment(
                query=query,
                judgment_type=JudgmentType.CANNOT_ANSWER,
                justification=(
                    f"BASIC competence in {domain!r} with "
                    f"{len(blind_spots)} unmitigated blind spots"
                ),
                confidence_in_judgment=0.6,
                domain=domain,
            )
            self._judgments.append(judgment)
            return judgment

        conf_in_judgment = 0.5
        if cal.sample_count >= 20 and cal.is_well_calibrated:
            conf_in_judgment = 0.85
        elif cal.sample_count >= 10:
            conf_in_judgment = 0.65

        if scope.level >= CompetenceLevel.COMPETENT:
            j_type = JudgmentType.CAN_ANSWER
            justification = (
                f"{scope.level.name} competence in {domain!r}, "
                f"calibration ECE={cal.ece:.3f}"
            )
        else:
            j_type = JudgmentType.UNCERTAIN
            justification = (
                f"{scope.level.name} competence in {domain!r}, "
                f"{len(open_qs)} open questions"
            )

        if blind_spots:
            j_type = JudgmentType.UNCERTAIN
            justification += f"; {len(blind_spots)} blind spots"
            conf_in_judgment *= 0.8

        judgment = MetacognitiveJudgment(
            query=query,
            judgment_type=j_type,
            justification=justification,
            confidence_in_judgment=min(1.0, conf_in_judgment),
            domain=domain,
        )
        self._judgments.append(judgment)
        return judgment

    # ── Introspection ─────────────────────────────────────────────────────

    @property
    def total_assertions(self) -> int:
        return len(self._assertions)

    @property
    def total_open_questions(self) -> int:
        return sum(1 for q in self._questions.values() if not q.resolved)

    @property
    def total_blind_spots(self) -> int:
        return len(self._blind_spots)

    @property
    def total_calibration_records(self) -> int:
        return len(self._calibration_records)

    def evaluate(
        self,
        *,
        query: str,
        answer: str = "",
        confidence: float = 0.5,
        domain: str = "general",
        h_score: float = 0.0,
    ) -> "MetacognitiveEvaluation":
        """Pipeline-facing API for cognitive post-processing.

        Combines ``assess`` + calibration + gap detection into a single
        return object that ``nrsi.core.nrs._process_inner`` expects.
        """
        judgment = self.assess(query, domain=domain)
        cal = self.calibrate(domain=domain)
        open_qs = self.get_open_questions(domain=domain)
        blind_spots = self.get_blind_spots(domain=domain, unmitigated_only=True)

        should_revise = (
            confidence < 0.5
            or h_score < 0.7
            or judgment.judgment_type in (JudgmentType.CANNOT_ANSWER, JudgmentType.OUT_OF_SCOPE)
        )
        detected_gaps = [q.question for q in open_qs[:5]]
        if blind_spots:
            detected_gaps.extend(s.description for s in blind_spots[:3])

        return MetacognitiveEvaluation(
            calibration_score=1.0 - cal.ece if cal.sample_count > 0 else 0.5,
            should_revise=should_revise,
            detected_gaps=detected_gaps,
            judgment=judgment,
            confidence_in_answer=confidence,
        )

    def summary(self) -> Dict[str, Any]:
        """Return a dict summary of the metacognitive state."""
        cal = self.calibrate()
        return {
            "assertions": self.total_assertions,
            "open_questions": self.total_open_questions,
            "blind_spots": self.total_blind_spots,
            "calibration_records": self.total_calibration_records,
            "calibration": {
                "brier_score": cal.brier_score,
                "ece": cal.ece,
                "bias": cal.dominant_bias,
            },
            "competence_domains": list(self._competences.keys()),
        }

    def __repr__(self) -> str:
        return (
            f"MetacognitiveMonitor(assertions={self.total_assertions}, "
            f"questions={self.total_open_questions}, "
            f"blind_spots={self.total_blind_spots})"
        )


@dataclass
class MetacognitiveEvaluation:
    """Result from ``MetacognitiveMonitor.evaluate()``."""
    calibration_score: float = 0.5
    should_revise: bool = False
    detected_gaps: List[str] = field(default_factory=list)
    judgment: Optional[MetacognitiveJudgment] = None
    confidence_in_answer: float = 0.5
