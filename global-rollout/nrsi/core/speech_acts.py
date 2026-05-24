"""NRSI Speech Act Types — Communicative Intent Classification.

Utterances aren't just text — they're moves in a dialogue protocol.
These types classify communicative intent so the system can respond
appropriately (assertions need verification, commands need execution,
questions need answers, promises create obligations).

Based on Searle's taxonomy with FIPA-ACL alignment:
  Assertive  — commits speaker to truth of proposition
  Directive  — attempts to get hearer to do something
  Commissive — commits speaker to future action
  Expressive — expresses psychological state
  Declarative — brings about state of affairs by utterance

Patent-covered: NRSI Speech Act Classification System, VelarIQ.
"""

from __future__ import annotations

import re
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
# 1. SpeechActType — Communicative intent taxonomy
# ═══════════════════════════════════════════════════════════════════════════════

class SpeechActType(Enum):
    """Taxonomy of communicative intents (Searle + FIPA-ACL)."""

    # Assertives — commit speaker to truth
    ASSERT = auto()
    INFORM = auto()
    CONFIRM = auto()
    DENY = auto()

    # Directives — attempt to get hearer to act
    QUESTION = auto()
    REQUEST = auto()
    COMMAND = auto()
    SUGGEST = auto()

    # Commissives — commit speaker to future action
    PROMISE = auto()
    WARN = auto()

    # Expressives — psychological state
    THANK = auto()
    APOLOGIZE = auto()
    GREET = auto()
    FAREWELL = auto()

    # Declaratives — change state of affairs by utterance
    DECLARE = auto()
    PERMIT = auto()
    FORBID = auto()

    @property
    def category(self) -> str:
        _CATEGORY_MAP: Dict[SpeechActType, str] = {
            SpeechActType.ASSERT: "assertive",
            SpeechActType.INFORM: "assertive",
            SpeechActType.CONFIRM: "assertive",
            SpeechActType.DENY: "assertive",
            SpeechActType.QUESTION: "directive",
            SpeechActType.REQUEST: "directive",
            SpeechActType.COMMAND: "directive",
            SpeechActType.SUGGEST: "directive",
            SpeechActType.PROMISE: "commissive",
            SpeechActType.WARN: "commissive",
            SpeechActType.THANK: "expressive",
            SpeechActType.APOLOGIZE: "expressive",
            SpeechActType.GREET: "expressive",
            SpeechActType.FAREWELL: "expressive",
            SpeechActType.DECLARE: "declarative",
            SpeechActType.PERMIT: "declarative",
            SpeechActType.FORBID: "declarative",
        }
        return _CATEGORY_MAP[self]

    @property
    def creates_obligation(self) -> bool:
        return self in {
            SpeechActType.PROMISE,
            SpeechActType.REQUEST,
            SpeechActType.COMMAND,
            SpeechActType.PERMIT,
            SpeechActType.FORBID,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SpeechAct — A single communicative event
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SpeechAct:
    """An utterance classified by communicative intent.

    ``illocutionary_force`` captures conviction strength: 1.0 for a firm
    command, 0.3 for a tentative suggestion.
    """

    act_type: SpeechActType
    speaker: str
    hearer: str
    propositional_content: str
    illocutionary_force: float
    act_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if not 0.0 <= self.illocutionary_force <= 1.0:
            raise ValueError(
                f"illocutionary_force must be in [0.0, 1.0], "
                f"got {self.illocutionary_force}"
            )

    @property
    def category(self) -> str:
        return self.act_type.category

    @property
    def creates_obligation(self) -> bool:
        return self.act_type.creates_obligation


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FelicityCondition — What must hold for a speech act to be valid
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FelicityCondition:
    """Austin/Searle felicity conditions for speech act validity.

    A promise is *infelicitous* if the speaker cannot possibly fulfill it.
    A command is infelicitous if the speaker has no authority.
    """

    act_type: SpeechActType
    precondition: str
    sincerity: str
    essential: str

    def check(self, speaker_authority: bool, speaker_can_fulfill: bool) -> bool:
        if self.act_type in {SpeechActType.COMMAND, SpeechActType.FORBID, SpeechActType.PERMIT}:
            return speaker_authority
        if self.act_type == SpeechActType.PROMISE:
            return speaker_can_fulfill
        return True


DEFAULT_FELICITY: Dict[SpeechActType, FelicityCondition] = {
    SpeechActType.ASSERT: FelicityCondition(
        act_type=SpeechActType.ASSERT,
        precondition="Speaker has evidence for proposition",
        sincerity="Speaker believes proposition is true",
        essential="Counts as commitment to truth of proposition",
    ),
    SpeechActType.QUESTION: FelicityCondition(
        act_type=SpeechActType.QUESTION,
        precondition="Speaker does not already know the answer",
        sincerity="Speaker wants to know the answer",
        essential="Counts as an attempt to elicit information",
    ),
    SpeechActType.COMMAND: FelicityCondition(
        act_type=SpeechActType.COMMAND,
        precondition="Speaker has authority over hearer",
        sincerity="Speaker wants the action performed",
        essential="Counts as an attempt to get hearer to act",
    ),
    SpeechActType.PROMISE: FelicityCondition(
        act_type=SpeechActType.PROMISE,
        precondition="Speaker is able to perform the promised action",
        sincerity="Speaker intends to perform the action",
        essential="Counts as commitment to future action",
    ),
    SpeechActType.WARN: FelicityCondition(
        act_type=SpeechActType.WARN,
        precondition="Speaker believes event is not in hearer's interest",
        sincerity="Speaker believes the warning is warranted",
        essential="Counts as notification of potential harm",
    ),
    SpeechActType.PERMIT: FelicityCondition(
        act_type=SpeechActType.PERMIT,
        precondition="Speaker has authority to grant permission",
        sincerity="Speaker is willing to allow the action",
        essential="Counts as removal of a prior prohibition",
    ),
    SpeechActType.FORBID: FelicityCondition(
        act_type=SpeechActType.FORBID,
        precondition="Speaker has authority to restrict action",
        sincerity="Speaker does not want the action performed",
        essential="Counts as prohibition of an action",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SpeechActClassifier — Pattern-based intent detection
# ═══════════════════════════════════════════════════════════════════════════════

_QUESTION_PATTERN = re.compile(
    r"(?i)^(what|who|where|when|why|how|which|is|are|was|were|do|does|did|can|could|"
    r"would|should|shall|will|may|might|have|has|had)\b|"
    r"\?\s*$"
)
_COMMAND_PATTERN = re.compile(
    r"(?i)^(do|run|execute|close|open|stop|start|delete|remove|create|build|deploy|"
    r"send|fetch|get|put|set|clear|reset|abort|shutdown|restart|enable|disable)\b"
)
_PROMISE_PATTERN = re.compile(
    r"(?i)\b(i will|i shall|i commit|i promise|i guarantee|i pledge|we will|we shall)\b"
)
_WARN_PATTERN = re.compile(
    r"(?i)\b(danger|caution|risk|warning|beware|alert|hazard|threat|unsafe|critical)\b"
)
_SUGGEST_PATTERN = re.compile(
    r"(?i)\b(maybe|perhaps|consider|you could|you might|suggestion|recommend|"
    r"how about|what if|it might be)\b"
)
_GREET_PATTERN = re.compile(
    r"(?i)^(hello|hi|hey|greetings|good morning|good afternoon|good evening)\b"
)
_FAREWELL_PATTERN = re.compile(
    r"(?i)^(goodbye|bye|farewell|see you|take care|later|goodnight)\b"
)
_THANK_PATTERN = re.compile(r"(?i)\b(thank|thanks|appreciate|grateful)\b")
_APOLOGIZE_PATTERN = re.compile(r"(?i)\b(sorry|apolog|my bad|pardon)\b")
_DENY_PATTERN = re.compile(
    r"(?i)^(no|nope|negative|i disagree|that is wrong|incorrect|denied)\b"
)
_CONFIRM_PATTERN = re.compile(
    r"(?i)^(yes|yep|correct|confirmed|affirmative|agreed|exactly|right)\b"
)


class SpeechActClassifier:
    """Classify raw text into speech acts using surface-pattern heuristics.

    Priority order mirrors Searle's specificity hierarchy — more specific
    intents (warn, promise) override general ones (assert).
    """

    _RULES: Sequence[Tuple[re.Pattern[str], SpeechActType, float]] = (
        (_WARN_PATTERN, SpeechActType.WARN, 0.80),
        (_PROMISE_PATTERN, SpeechActType.PROMISE, 0.75),
        (_COMMAND_PATTERN, SpeechActType.COMMAND, 0.70),
        (_QUESTION_PATTERN, SpeechActType.QUESTION, 0.70),
        (_SUGGEST_PATTERN, SpeechActType.SUGGEST, 0.65),
        (_GREET_PATTERN, SpeechActType.GREET, 0.90),
        (_FAREWELL_PATTERN, SpeechActType.FAREWELL, 0.90),
        (_THANK_PATTERN, SpeechActType.THANK, 0.85),
        (_APOLOGIZE_PATTERN, SpeechActType.APOLOGIZE, 0.80),
        (_DENY_PATTERN, SpeechActType.DENY, 0.75),
        (_CONFIRM_PATTERN, SpeechActType.CONFIRM, 0.80),
    )

    @staticmethod
    def classify(
        text: str,
        speaker: str = "system",
        hearer: str = "user",
        context: Optional[Dict[str, Any]] = None,
    ) -> SpeechAct:
        """Classify *text* into a ``SpeechAct``.

        Applies patterns in priority order; falls back to ASSERT for
        declarative statements that match no specific intent.
        """
        stripped = text.strip()
        if not stripped:
            return SpeechAct(
                act_type=SpeechActType.ASSERT,
                speaker=speaker,
                hearer=hearer,
                propositional_content="",
                illocutionary_force=0.0,
            )

        for pattern, act_type, base_force in SpeechActClassifier._RULES:
            if pattern.search(stripped):
                return SpeechAct(
                    act_type=act_type,
                    speaker=speaker,
                    hearer=hearer,
                    propositional_content=stripped,
                    illocutionary_force=base_force,
                )

        return SpeechAct(
            act_type=SpeechActType.ASSERT,
            speaker=speaker,
            hearer=hearer,
            propositional_content=stripped,
            illocutionary_force=0.60,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DialogueCommitment — Obligations created by speech acts
# ═══════════════════════════════════════════════════════════════════════════════

class CommitmentStatus(Enum):
    ACTIVE = auto()
    FULFILLED = auto()
    VIOLATED = auto()
    WITHDRAWN = auto()


@dataclass
class DialogueCommitment:
    """A social commitment created by a speech act.

    When agent A promises X, an ACTIVE commitment is recorded.
    It transitions to FULFILLED when evidence is provided, VIOLATED
    after deadline, or WITHDRAWN if retracted.
    """

    agent_id: str
    commitment_content: str
    created_by_act: SpeechAct
    status: CommitmentStatus = CommitmentStatus.ACTIVE
    commitment_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    created_at: float = field(default_factory=time.monotonic)
    resolved_at: Optional[float] = None

    def fulfill(self) -> None:
        if self.status != CommitmentStatus.ACTIVE:
            raise ValueError(
                f"Cannot fulfill commitment in state {self.status.name}"
            )
        self.status = CommitmentStatus.FULFILLED
        self.resolved_at = time.monotonic()

    def violate(self) -> None:
        if self.status != CommitmentStatus.ACTIVE:
            raise ValueError(
                f"Cannot violate commitment in state {self.status.name}"
            )
        self.status = CommitmentStatus.VIOLATED
        self.resolved_at = time.monotonic()

    def withdraw(self) -> None:
        if self.status != CommitmentStatus.ACTIVE:
            raise ValueError(
                f"Cannot withdraw commitment in state {self.status.name}"
            )
        self.status = CommitmentStatus.WITHDRAWN
        self.resolved_at = time.monotonic()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DialogueState — Running state of a conversation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DialogueState:
    """Track commitments, turn count, and topic across a dialogue.

    ``process_act`` is the main entry point — it records the act,
    updates topic tracking, and materialises any new commitments.
    """

    dialogue_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    commitments: Dict[str, List[DialogueCommitment]] = field(default_factory=dict)
    turn_count: int = 0
    current_topic: Optional[str] = None
    history: List[SpeechAct] = field(default_factory=list)

    def process_act(self, act: SpeechAct) -> Optional[DialogueCommitment]:
        """Ingest a speech act, update state, and return any new commitment."""
        self.history.append(act)
        self.turn_count += 1
        self.current_topic = act.propositional_content[:120] or self.current_topic

        commitment: Optional[DialogueCommitment] = None

        if act.act_type == SpeechActType.PROMISE:
            commitment = DialogueCommitment(
                agent_id=act.speaker,
                commitment_content=act.propositional_content,
                created_by_act=act,
            )
            self.commitments.setdefault(act.speaker, []).append(commitment)

        elif act.act_type == SpeechActType.COMMAND:
            commitment = DialogueCommitment(
                agent_id=act.hearer,
                commitment_content=act.propositional_content,
                created_by_act=act,
            )
            self.commitments.setdefault(act.hearer, []).append(commitment)

        elif act.act_type == SpeechActType.REQUEST:
            commitment = DialogueCommitment(
                agent_id=act.hearer,
                commitment_content=act.propositional_content,
                created_by_act=act,
            )
            self.commitments.setdefault(act.hearer, []).append(commitment)

        elif act.act_type == SpeechActType.PERMIT:
            commitment = DialogueCommitment(
                agent_id=act.speaker,
                commitment_content=f"PERMIT: {act.propositional_content}",
                created_by_act=act,
            )
            self.commitments.setdefault(act.speaker, []).append(commitment)

        elif act.act_type == SpeechActType.FORBID:
            commitment = DialogueCommitment(
                agent_id=act.speaker,
                commitment_content=f"FORBID: {act.propositional_content}",
                created_by_act=act,
            )
            self.commitments.setdefault(act.speaker, []).append(commitment)

        return commitment

    def check_commitments(self, agent_id: str) -> List[DialogueCommitment]:
        """Return all ACTIVE commitments for *agent_id*."""
        return [
            c
            for c in self.commitments.get(agent_id, [])
            if c.status == CommitmentStatus.ACTIVE
        ]

    def violated_commitments(self) -> List[DialogueCommitment]:
        """Return all VIOLATED commitments across every agent."""
        result: List[DialogueCommitment] = []
        for agent_commits in self.commitments.values():
            result.extend(c for c in agent_commits if c.status == CommitmentStatus.VIOLATED)
        return result

    def all_commitments(self) -> List[DialogueCommitment]:
        result: List[DialogueCommitment] = []
        for agent_commits in self.commitments.values():
            result.extend(agent_commits)
        return result

    def summary(self) -> Dict[str, Any]:
        all_c = self.all_commitments()
        return {
            "dialogue_id": self.dialogue_id,
            "turn_count": self.turn_count,
            "current_topic": self.current_topic,
            "total_commitments": len(all_c),
            "active": sum(1 for c in all_c if c.status == CommitmentStatus.ACTIVE),
            "fulfilled": sum(1 for c in all_c if c.status == CommitmentStatus.FULFILLED),
            "violated": sum(1 for c in all_c if c.status == CommitmentStatus.VIOLATED),
            "withdrawn": sum(1 for c in all_c if c.status == CommitmentStatus.WITHDRAWN),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DialogueProtocol — Allowed act sequences
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DialogueProtocol:
    """A finite protocol defining legal act-type transitions.

    ``transitions`` maps each act type to the set of act types that
    may legally follow it.  ``validate_sequence`` checks a full trace.
    """

    protocol_name: str
    transitions: Dict[SpeechActType, FrozenSet[SpeechActType]]
    description: str = ""

    def is_valid_transition(self, prev: SpeechActType, next_act: SpeechActType) -> bool:
        allowed = self.transitions.get(prev)
        if allowed is None:
            return True
        return next_act in allowed

    def validate_sequence(self, acts: Sequence[SpeechAct]) -> List[Tuple[int, SpeechAct, SpeechAct, str]]:
        """Return a list of (index, prev_act, bad_act, reason) for violations."""
        violations: List[Tuple[int, SpeechAct, SpeechAct, str]] = []
        for i in range(1, len(acts)):
            prev, cur = acts[i - 1], acts[i]
            if not self.is_valid_transition(prev.act_type, cur.act_type):
                allowed = self.transitions.get(prev.act_type, frozenset())
                violations.append((
                    i,
                    prev,
                    cur,
                    f"{cur.act_type.name} cannot follow {prev.act_type.name}; "
                    f"allowed: {', '.join(a.name for a in allowed)}",
                ))
        return violations


FIPA_QUERY_PROTOCOL = DialogueProtocol(
    protocol_name="fipa-query",
    description="FIPA Query Interaction Protocol — question/answer with optional refusal",
    transitions={
        SpeechActType.QUESTION: frozenset({
            SpeechActType.INFORM,
            SpeechActType.CONFIRM,
            SpeechActType.DENY,
            SpeechActType.WARN,
        }),
        SpeechActType.INFORM: frozenset({
            SpeechActType.CONFIRM,
            SpeechActType.DENY,
            SpeechActType.QUESTION,
            SpeechActType.THANK,
        }),
    },
)

FIPA_REQUEST_PROTOCOL = DialogueProtocol(
    protocol_name="fipa-request",
    description="FIPA Request Interaction Protocol — request/accept/refuse/report",
    transitions={
        SpeechActType.REQUEST: frozenset({
            SpeechActType.PROMISE,
            SpeechActType.DENY,
            SpeechActType.SUGGEST,
        }),
        SpeechActType.COMMAND: frozenset({
            SpeechActType.PROMISE,
            SpeechActType.DENY,
            SpeechActType.INFORM,
        }),
        SpeechActType.PROMISE: frozenset({
            SpeechActType.INFORM,
            SpeechActType.WARN,
            SpeechActType.APOLOGIZE,
        }),
    },
)
