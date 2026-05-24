"""
NRSI Cognitive Mode Control — Adaptive Multi-Dimensional Mode System.

Replaces the primitive 2-position DETERMINISTIC/HYBRID switch with a
continuous 7-dimension mode vector driven by the nervous system, domain
constraints, user preferences, and historical performance.

Every query produces a ModeDecision containing:
  - ModeVector: 7 floats (analytical, creative, factual, critical,
    empathetic, exploratory, metacognitive) each 0.0-1.0
  - Subsystem directives: which neurons fire, which lobes activate,
    what TVS mode to use, how the generative engine should shape tone
  - Optional multi-pass chain for T4 queries
  - Adversarial review flag for creative output with factual claims

Patent-covered: NRSI Adaptive Cognitive Mode Selection, VelarIQ.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── High-Risk Domains ────────────────────────────────────────────────────────

HIGH_RISK_DOMAINS: FrozenSet[str] = frozenset({
    "medical", "legal", "financial", "safety", "pharmaceutical",
})

# ── Mode Spectrum ────────────────────────────────────────────────────────────

class ModeSpectrum(str, Enum):
    DETERMINISTIC = "DETERMINISTIC"
    ANALYTICAL = "ANALYTICAL"
    CREATIVE = "CREATIVE"
    EMPATHETIC = "EMPATHETIC"
    EXPLORATORY = "EXPLORATORY"
    CRITICAL = "CRITICAL"
    METACOGNITIVE = "METACOGNITIVE"
    HYBRID = "HYBRID"

_MODE_TO_LEGACY = {
    ModeSpectrum.DETERMINISTIC: "DETERMINISTIC",
    ModeSpectrum.ANALYTICAL: "DETERMINISTIC",
    ModeSpectrum.CREATIVE: "PROBABILISTIC",
    ModeSpectrum.EMPATHETIC: "HYBRID",
    ModeSpectrum.EXPLORATORY: "PROBABILISTIC",
    ModeSpectrum.CRITICAL: "DETERMINISTIC",
    ModeSpectrum.METACOGNITIVE: "HYBRID",
    ModeSpectrum.HYBRID: "HYBRID",
}


# ── Public 3-Mode Contract ──────────────────────────────────────────────────
#
# The internal CognitiveModeController operates over a 7-D ModeVector. The
# user-facing API exposes exactly three modes plus an auto-selector. This is
# the single, stable contract product/UI/SDKs target. Anything else (the
# 7-D vector, the lobe weights, etc.) is implementation detail.
#
#   deterministic  — analytical + factual locked high; creative=0.
#                    Used for math, code, citations, regulated domains.
#   hybrid         — grounded in facts but creative. factual stays high
#                    so claims remain checkable, creative is moderate so
#                    phrasing/structure can vary. Adversarial fact-check
#                    is enabled.
#   creative       — creative + exploratory high; factual remains a floor
#                    (we don't want fiction that contradicts known facts);
#                    adversarial check still runs on factual claims.
#   auto           — let CognitiveModeController.classify() decide based on
#                    query content, domain, and user history.

PUBLIC_MODES: FrozenSet[str] = frozenset({"deterministic", "hybrid", "creative", "auto"})


def normalize_public_mode(value: Optional[str]) -> str:
    """Coerce arbitrary input to a canonical public-mode string.

    Accepts case-insensitive aliases. Anything unrecognised → ``"auto"`` so
    callers can pass through user input without having to validate first.
    """
    if not value:
        return "auto"
    v = str(value).strip().lower()
    aliases = {
        "deterministic": "deterministic",
        "det": "deterministic",
        "factual": "deterministic",
        "analytical": "deterministic",
        "precise": "deterministic",
        "hybrid": "hybrid",
        "balanced": "hybrid",
        "grounded": "hybrid",
        "grounded_creative": "hybrid",
        "creative": "creative",
        "probabilistic": "creative",
        "exploratory": "creative",
        "free": "creative",
        "auto": "auto",
        "default": "auto",
    }
    return aliases.get(v, "auto")


def vector_for_public_mode(mode: str) -> "ModeVector":
    """Map a public mode string to a deterministic ModeVector preset.

    Used by the CognitiveModeController when ``mode_override`` is one of the
    three public modes, and by the inference dispatch layer to score worker
    selection without touching the controller.
    """
    canonical = normalize_public_mode(mode)
    if canonical == "deterministic":
        return ModeVector(
            analytical=0.95,
            factual=1.0,
            critical=0.6,
            creative=0.0,
            exploratory=0.0,
            empathetic=0.05,
            metacognitive=0.2,
        )
    if canonical == "hybrid":
        return ModeVector(
            analytical=0.7,
            factual=0.85,
            critical=0.5,
            creative=0.5,
            exploratory=0.4,
            empathetic=0.25,
            metacognitive=0.35,
        )
    if canonical == "creative":
        return ModeVector(
            analytical=0.4,
            factual=0.55,
            critical=0.35,
            creative=0.95,
            exploratory=0.8,
            empathetic=0.4,
            metacognitive=0.4,
        )
    return ModeVector(analytical=0.5, factual=0.5)


# ── Mode Vector (7-Dimensional) ─────────────────────────────────────────────

@dataclass
class ModeVector:
    """Continuous 7-dimension cognitive mode representation.

    Each dimension is 0.0-1.0. The dominant dimension determines
    the primary ModeSpectrum enum, but all subsystems receive
    the full vector for nuanced behavior selection.
    """
    analytical: float = 0.5
    creative: float = 0.0
    factual: float = 0.5
    critical: float = 0.0
    empathetic: float = 0.0
    exploratory: float = 0.0
    metacognitive: float = 0.0

    def __post_init__(self):
        for attr in self._fields():
            setattr(self, attr, max(0.0, min(1.0, getattr(self, attr))))

    @staticmethod
    def _fields() -> Tuple[str, ...]:
        return (
            "analytical", "creative", "factual", "critical",
            "empathetic", "exploratory", "metacognitive",
        )

    def primary_mode(self) -> ModeSpectrum:
        vals = {
            ModeSpectrum.ANALYTICAL: self.analytical,
            ModeSpectrum.CREATIVE: self.creative,
            ModeSpectrum.DETERMINISTIC: self.factual,
            ModeSpectrum.CRITICAL: self.critical,
            ModeSpectrum.EMPATHETIC: self.empathetic,
            ModeSpectrum.EXPLORATORY: self.exploratory,
            ModeSpectrum.METACOGNITIVE: self.metacognitive,
        }
        best = max(vals, key=vals.get)
        top_val = vals[best]
        second_val = sorted(vals.values(), reverse=True)[1]
        if top_val - second_val < 0.08:
            return ModeSpectrum.HYBRID
        return best

    @property
    def legacy_mode(self) -> str:
        return _MODE_TO_LEGACY.get(self.primary_mode(), "HYBRID")

    def blend(self, other: "ModeVector", alpha: float) -> "ModeVector":
        """Interpolate: result = (1-alpha)*self + alpha*other."""
        alpha = max(0.0, min(1.0, alpha))
        kw = {}
        for f in self._fields():
            kw[f] = (1 - alpha) * getattr(self, f) + alpha * getattr(other, f)
        return ModeVector(**kw)

    def normalize(self) -> "ModeVector":
        total = sum(getattr(self, f) for f in self._fields())
        if total < 1e-9:
            return ModeVector()
        kw = {f: getattr(self, f) / total for f in self._fields()}
        return ModeVector(**kw)

    def distance(self, other: "ModeVector") -> float:
        return math.sqrt(sum(
            (getattr(self, f) - getattr(other, f)) ** 2
            for f in self._fields()
        ))

    def to_dict(self) -> Dict[str, float]:
        return {f: round(getattr(self, f), 4) for f in self._fields()}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "ModeVector":
        return cls(**{f: d.get(f, 0.0) for f in cls._fields()})

    @classmethod
    def from_query_type(cls, query_type: str) -> "ModeVector":
        presets = {
            "factual": cls(analytical=0.7, factual=0.9, critical=0.3),
            "procedural": cls(analytical=0.8, factual=0.7, metacognitive=0.3),
            "creative": cls(creative=0.9, exploratory=0.6, empathetic=0.3),
            "social": cls(empathetic=0.8, creative=0.3, metacognitive=0.2),
            "meta": cls(metacognitive=0.9, analytical=0.5, critical=0.4),
            "philosophical": cls(exploratory=0.7, analytical=0.6, metacognitive=0.5, critical=0.3),
        }
        return presets.get(query_type, cls(analytical=0.5, factual=0.5))


# ── Neuron Strategy ──────────────────────────────────────────────────────────

@dataclass
class NeuronStrategy:
    """Directs how BinaryNeuronBank activates for a given mode."""
    use_stochastic: bool = False
    temperature: float = 0.0
    top_k_multiplier: float = 1.0
    noise_sigma: float = 0.0


# ── Tone Directive ───────────────────────────────────────────────────────────

@dataclass
class ToneDirective:
    """Instructs the generative engine on response style."""
    tone: str = "neutral"
    emotional_coloring: str = "none"
    verbosity: str = "normal"
    formality: float = 0.5
    reading_level: float = 0.5   # 0=child-simple, 1=expert-academic
    personality_traits: Dict[str, float] = field(default_factory=dict)


# ── Mode Decision ────────────────────────────────────────────────────────────

@dataclass
class ModeDecision:
    """Complete output of CognitiveModeController.classify()."""
    vector: ModeVector = field(default_factory=ModeVector)
    primary_mode: ModeSpectrum = ModeSpectrum.HYBRID
    legacy_mode: str = "HYBRID"
    decision_reason: str = ""
    domain_constraint: str = ""
    tvs_mode: str = "HYBRID"
    lobe_weights: List[Tuple[str, float]] = field(default_factory=list)
    neuron_strategy: NeuronStrategy = field(default_factory=NeuronStrategy)
    multi_pass_chain: List[ModeSpectrum] = field(default_factory=list)
    tone_directive: ToneDirective = field(default_factory=ToneDirective)
    section_labels: List[Dict[str, str]] = field(default_factory=list)
    do_adversarial_check: bool = False


# ── Adversarial Review Result ────────────────────────────────────────────────

@dataclass
class AdversarialResult:
    flagged_claims: List[str] = field(default_factory=list)
    corrected_claims: Dict[str, str] = field(default_factory=dict)
    overall_factual_score: float = 1.0
    recommendation: str = "pass"


# ── User Mode Profile ────────────────────────────────────────────────────────

@dataclass
class UserModeProfile:
    user_id: str = ""
    preferred_vector: ModeVector = field(default_factory=ModeVector)
    domain_preferences: Dict[str, Dict[str, float]] = field(default_factory=dict)
    interaction_count: int = 0
    last_updated: float = 0.0
    sophistication_ema: float = 0.5


# ── Keyword Signals ──────────────────────────────────────────────────────────

_CREATIVE_SIGNALS = frozenset({
    # Core creative
    "write", "compose", "create", "imagine", "story", "poem", "fiction",
    "invent", "brainstorm", "design", "dream", "novel", "fantasy",
    "generate", "art", "musical", "lyric", "narrative", "script",
    # Media generation triggers
    "draw", "paint", "sketch", "picture", "image", "video", "audio",
    "song", "music", "animation", "render", "illustration", "visualize",
    "portrait", "landscape", "cinematic", "soundtrack", "voiceover",
    "remix", "mashup", "collage", "storyboard", "scene",
    # Stylistic
    "aesthetic", "abstract", "surreal", "photorealistic", "impressionist",
    "minimalist", "futuristic", "retro", "cartoon", "watercolor",
})
_FACTUAL_SIGNALS = frozenset({
    # Core factual
    "define", "what is", "fact", "true", "false", "exactly", "precisely",
    "calculate", "how many", "when did", "where is", "who is", "date",
    "number", "statistic", "data", "evidence", "source", "reference",
    # Hypothesis / prediction / diagnosis — require deterministic reasoning
    "hypothesis", "hypothesize", "predict", "forecast", "diagnose",
    "cure", "treatment", "prognosis", "etiology", "pathogenesis",
    "biomarker", "clinical", "trial", "mechanism",
    # Business / planning
    "plan", "strategy", "budget", "schedule", "revenue", "profit",
    "roi", "projection", "business", "market", "valuation", "pipeline",
    "roadmap", "milestone", "kpi", "metric", "benchmark",
    # Scientific
    "prove", "theorem", "axiom", "formula", "equation", "derive",
    "quantify", "measure", "replicate", "experiment", "variable",
    "control", "sample", "correlation", "causation",
})
_ANALYTICAL_SIGNALS = frozenset({
    "analyze", "compare", "contrast", "evaluate", "assess", "examine",
    "investigate", "study", "review", "critique", "break down", "dissect",
    "benchmark", "audit", "profile", "survey", "inspect", "diagnose",
    "root cause", "regression", "trend", "optimize", "efficiency",
})
_EMPATHETIC_SIGNALS = frozenset({
    "feel", "feeling", "sad", "happy", "anxious", "worried", "scared",
    "upset", "depressed", "lonely", "help me", "advice", "support",
    "comfort", "understand", "relationship", "grief", "loss",
    "overwhelmed", "stressed", "burnout", "grateful", "hopeful",
    "insecure", "frustrated", "confused", "afraid", "encourage",
})
_EXPLORATORY_SIGNALS = frozenset({
    "explore", "what if", "hypothetical", "speculate", "wonder",
    "possibilities", "alternatives", "might", "could", "suppose",
    "thought experiment", "consider", "brainstorm", "scenario",
    "envision", "futuristic", "utopia", "dystopia", "wildcard",
    "moonshot", "disruptive", "paradigm", "reimagine", "prototype",
})
_CRITICAL_SIGNALS = frozenset({
    "argue", "counter", "flaw", "weakness", "problem", "issue",
    "wrong", "incorrect", "disagree", "challenge", "debate", "refute",
    "disprove", "fallacy", "bias", "contradiction", "inconsistent",
    "loophole", "oversight", "risk", "vulnerability", "limitation",
})
_META_SIGNALS = frozenset({
    "think about", "reasoning", "logic", "how do you know", "meta",
    "self-reflect", "confidence", "certain", "uncertain", "assumption",
    "methodology", "approach", "strategy", "framework",
    "epistemology", "heuristic", "calibration", "introspect",
    "decision process", "rationale", "justification", "first principles",
})
_PHILOSOPHICAL_SIGNALS = frozenset({
    "meaning", "purpose", "existence", "consciousness", "morality",
    "ethics", "philosophy", "philosophical", "metaphysics", "ontology",
    "epistemology", "virtue", "justice", "freedom", "soul", "truth",
    "reality", "perception", "identity", "nature",
    "determinism", "free will", "utilitarianism", "deontology",
    "nihilism", "stoicism", "existentialism", "phenomenology",
})


# ── Cognitive Mode Controller ────────────────────────────────────────────────

class CognitiveModeController:
    """Adaptive multi-dimensional mode classification.

    Combines nervous system signals, domain constraints, query
    complexity, user preferences, and historical performance into
    a ModeDecision that drives every downstream subsystem.
    """

    def __init__(self, redis_client=None):
        self._redis = redis_client
        self._mode_history: List[Dict[str, Any]] = []
        self._stats = {
            "classifications": 0,
            "mode_counts": {m.value: 0 for m in ModeSpectrum},
            "adversarial_checks": 0,
            "multi_pass_triggered": 0,
        }
        self._user_store = UserModeProfileStore(redis_client)
        self._domain_history = DomainModeHistory(redis_client)

    def classify(
        self,
        query: str,
        domain: str = "",
        percept=None,
        emotional_ctx: Optional[Dict[str, Any]] = None,
        mode_override: str = "",
        user_id: str = "",
        tier: str = "T1",
        motor_plan=None,
    ) -> ModeDecision:
        self._stats["classifications"] += 1
        emotional_ctx = emotional_ctx or {}
        reasons: List[str] = []

        # Classify query type from keywords first — they are the strongest
        # signal because the user's words are explicit intent.  The nervous
        # system percept is a secondary hint only.
        ql = query.lower()
        _ql_words = set(ql.split())
        _kw_creative = len(_ql_words & _CREATIVE_SIGNALS)
        _kw_factual = len(_ql_words & _FACTUAL_SIGNALS)
        _kw_analytical = len(_ql_words & _ANALYTICAL_SIGNALS)
        _kw_empathetic = len(_ql_words & _EMPATHETIC_SIGNALS)
        _kw_exploratory = len(_ql_words & _EXPLORATORY_SIGNALS)
        _kw_critical = len(_ql_words & _CRITICAL_SIGNALS)
        _kw_meta = len(_ql_words & _META_SIGNALS)
        _kw_philo = len(_ql_words & _PHILOSOPHICAL_SIGNALS)

        _kw_best = max(
            ("creative", _kw_creative),
            ("philosophical", _kw_philo),
            ("meta", _kw_meta),
            ("social", _kw_empathetic),
            ("exploratory", _kw_exploratory),
            ("analytical", _kw_analytical),
            ("critical", _kw_critical),
            ("factual", _kw_factual),
            key=lambda x: x[1],
        )
        if _kw_best[1] > 0:
            qt = _kw_best[0]
        else:
            qt = getattr(percept, "query_type", "") if percept else ""
            if not qt:
                if any(p in ql for p in ("what is", "who is", "where is", "when did", "how many", "define", "calculate")):
                    qt = "factual"
                elif any(p in ql for p in ("why", "how does", "explain", "analyze", "compare")):
                    qt = "procedural"
                else:
                    qt = "factual"
        vector = ModeVector.from_query_type(qt)
        reasons.append(f"query_type={qt}")

        # Explicit override
        if mode_override:
            vector = self._apply_override(mode_override, vector)
            reasons.append(f"override={mode_override}")

        # Keyword signal detection
        vector = self._keyword_signals(query, vector)

        # Nervous system influence
        if percept or emotional_ctx:
            vector = self._nervous_influence(percept, emotional_ctx, vector)
            reasons.append("nervous_system")

        # Domain constraint
        domain_constraint = ""
        if domain and domain.lower() in HIGH_RISK_DOMAINS:
            vector = self._domain_constraint(domain, vector)
            domain_constraint = f"{domain}:clamped"
            reasons.append(f"domain_constraint={domain}")

        # Complexity adjustment
        vector = self._complexity_adjustment(tier, vector)

        # User preference blend
        user_profile = self._user_store.get_profile(user_id) if user_id else None
        if user_profile and user_profile.interaction_count > 5:
            pref = user_profile.preferred_vector
            if domain and domain in user_profile.domain_preferences:
                pref = ModeVector.from_dict(user_profile.domain_preferences[domain])
            vector = self._user_preference_blend(pref, vector)
            reasons.append("user_prefs")

        # Domain history adjustment
        optimal = self._domain_history.optimal_vector(domain, qt)
        if optimal:
            vector = self._domain_history_adjustment(optimal, vector)
            reasons.append("domain_history")

        primary = vector.primary_mode()
        self._stats["mode_counts"][primary.value] = (
            self._stats["mode_counts"].get(primary.value, 0) + 1
        )

        # Build subsystem directives
        tvs_mode = self._select_tvs_mode(vector)
        lobe_weights = self._select_lobes(vector)
        neuron_strategy = self._select_neuron_strategy(vector)
        multi_pass = self._should_multi_pass(vector, tier)
        if multi_pass:
            self._stats["multi_pass_triggered"] += 1
        do_adversarial = self._should_adversarial_check(vector)
        if do_adversarial:
            self._stats["adversarial_checks"] += 1
        _user_soph = getattr(percept, "user_sophistication", 0.5) if percept else 0.5
        _profile_ema = user_profile.sophistication_ema if user_profile else 0.5
        _blended_soph = _user_soph * 0.6 + _profile_ema * 0.4
        tone = self._build_tone_directive(vector, emotional_ctx, motor_plan, _blended_soph)

        return ModeDecision(
            vector=vector,
            primary_mode=primary,
            legacy_mode=vector.legacy_mode,
            decision_reason="; ".join(reasons),
            domain_constraint=domain_constraint,
            tvs_mode=tvs_mode,
            lobe_weights=lobe_weights,
            neuron_strategy=neuron_strategy,
            multi_pass_chain=multi_pass,
            tone_directive=tone,
            do_adversarial_check=do_adversarial,
        )

    # ── Internal classifiers ─────────────────────────────────────────

    def _apply_override(self, override: str, vector: ModeVector) -> ModeVector:
        # Public 3-mode contract takes precedence and uses the canonical
        # presets defined alongside ``PUBLIC_MODES``. Anything else falls
        # back to the legacy upper-case spectrum overrides.
        if normalize_public_mode(override) in {"deterministic", "hybrid", "creative"}:
            return vector_for_public_mode(override)
        upper = override.upper()
        overrides = {
            "DETERMINISTIC": ModeVector(analytical=0.8, factual=1.0, critical=0.4),
            "PROBABILISTIC": ModeVector(creative=0.8, exploratory=0.6, empathetic=0.3),
            "CREATIVE": ModeVector(creative=0.9, exploratory=0.7, empathetic=0.2),
            "ANALYTICAL": ModeVector(analytical=0.9, factual=0.7, critical=0.5),
            "HYBRID": ModeVector(analytical=0.5, factual=0.5, creative=0.3),
        }
        if upper in overrides:
            return overrides[upper]
        return vector

    def _keyword_signals(self, query: str, vector: ModeVector) -> ModeVector:
        ql = query.lower()
        words = set(ql.split())
        boosts = {
            "creative": len(words & _CREATIVE_SIGNALS) * 0.15,
            "factual": len(words & _FACTUAL_SIGNALS) * 0.10,
            "analytical": len(words & _ANALYTICAL_SIGNALS) * 0.10,
            "empathetic": len(words & _EMPATHETIC_SIGNALS) * 0.12,
            "exploratory": len(words & _EXPLORATORY_SIGNALS) * 0.10,
            "critical": len(words & _CRITICAL_SIGNALS) * 0.10,
            "metacognitive": len(words & _META_SIGNALS) * 0.10,
        }
        # Two-word phrase matching
        for phrase in ("what if", "how do you know", "think about",
                       "break down", "what is", "who is", "where is",
                       "when did", "how many", "help me",
                       "thought experiment"):
            if phrase in ql:
                if phrase in ("what if", "thought experiment"):
                    boosts["exploratory"] = boosts.get("exploratory", 0) + 0.15
                elif phrase in ("how do you know", "think about"):
                    boosts["metacognitive"] = boosts.get("metacognitive", 0) + 0.15
                elif phrase == "help me":
                    boosts["empathetic"] = boosts.get("empathetic", 0) + 0.15
                elif phrase == "break down":
                    boosts["analytical"] = boosts.get("analytical", 0) + 0.15
                else:
                    boosts["factual"] = boosts.get("factual", 0) + 0.10

        kw = {}
        for f in ModeVector._fields():
            kw[f] = min(1.0, getattr(vector, f) + boosts.get(f, 0))
        return ModeVector(**kw)

    def _nervous_influence(
        self, percept, emotional_ctx: Dict[str, Any], vector: ModeVector,
    ) -> ModeVector:
        threat = emotional_ctx.get("threat_level", 0)
        valence = emotional_ctx.get("valence", 0)
        kw = vector.to_dict()

        if threat > 0.5:
            kw["analytical"] = min(1.0, kw["analytical"] + threat * 0.4)
            kw["factual"] = min(1.0, kw["factual"] + threat * 0.3)
            kw["creative"] = max(0.0, kw["creative"] - threat * 0.5)
            kw["exploratory"] = max(0.0, kw["exploratory"] - threat * 0.3)

        if valence > 0.3:
            kw["creative"] = min(1.0, kw["creative"] + valence * 0.2)
            kw["exploratory"] = min(1.0, kw["exploratory"] + valence * 0.15)
        elif valence < -0.3:
            kw["empathetic"] = min(1.0, kw["empathetic"] + abs(valence) * 0.3)

        if percept:
            qt = getattr(percept, "query_type", "")
            if qt == "social":
                kw["empathetic"] = min(1.0, kw["empathetic"] + 0.3)
            elif qt == "meta":
                kw["metacognitive"] = min(1.0, kw["metacognitive"] + 0.3)
            elif qt == "creative":
                kw["creative"] = min(1.0, kw["creative"] + 0.25)

            emotion = getattr(percept, "emotional_tone", None)
            if emotion:
                emo_val = emotion.value if hasattr(emotion, "value") else str(emotion)
                if emo_val in ("anxious", "frustrated", "sad"):
                    kw["empathetic"] = min(1.0, kw["empathetic"] + 0.25)
                elif emo_val == "curious":
                    kw["exploratory"] = min(1.0, kw["exploratory"] + 0.2)
                elif emo_val == "creative":
                    kw["creative"] = min(1.0, kw["creative"] + 0.2)
                elif emo_val == "analytical":
                    kw["analytical"] = min(1.0, kw["analytical"] + 0.2)

        return ModeVector.from_dict(kw)

    def _domain_constraint(self, domain: str, vector: ModeVector) -> ModeVector:
        kw = vector.to_dict()
        kw["creative"] = min(kw["creative"], 0.3)
        kw["exploratory"] = min(kw["exploratory"], 0.4)
        kw["analytical"] = max(kw["analytical"], 0.7)
        kw["factual"] = max(kw["factual"], 0.7)
        kw["critical"] = max(kw["critical"], 0.4)
        return ModeVector.from_dict(kw)

    def _complexity_adjustment(self, tier: str, vector: ModeVector) -> ModeVector:
        kw = vector.to_dict()
        if tier in ("T0", "T1"):
            kw["factual"] = min(1.0, kw["factual"] + 0.15)
            kw["creative"] = max(0.0, kw["creative"] - 0.1)
            kw["metacognitive"] = max(0.0, kw["metacognitive"] - 0.1)
        elif tier == "T4":
            kw["metacognitive"] = min(1.0, kw["metacognitive"] + 0.2)
            kw["exploratory"] = min(1.0, kw["exploratory"] + 0.15)
            kw["critical"] = min(1.0, kw["critical"] + 0.1)
        return ModeVector.from_dict(kw)

    def _user_preference_blend(
        self, user_pref: ModeVector, vector: ModeVector,
    ) -> ModeVector:
        return vector.blend(user_pref, alpha=0.3)

    def _domain_history_adjustment(
        self, optimal: ModeVector, vector: ModeVector,
    ) -> ModeVector:
        return vector.blend(optimal, alpha=0.2)

    # ── Subsystem selectors ──────────────────────────────────────────

    def _select_tvs_mode(self, vector: ModeVector) -> str:
        if vector.creative > 0.6:
            return "CREATIVE"
        if vector.factual > 0.6:
            return "DETERMINISTIC"
        return "HYBRID"

    def _select_lobes(self, vector: ModeVector) -> List[Tuple[str, float]]:
        mapping = {
            "linguistic": 0.3 + vector.factual * 0.3,
            "logical": 0.2 + vector.analytical * 0.4,
            "mathematical": vector.analytical * 0.3,
            "spatial": vector.exploratory * 0.3,
            "temporal": 0.1 + vector.metacognitive * 0.3,
            "creative": vector.creative * 0.5,
            "causal": vector.analytical * 0.35 + vector.factual * 0.15,
            "analogical": vector.creative * 0.3 + vector.exploratory * 0.2,
            "planning": vector.analytical * 0.2 + vector.metacognitive * 0.3,
            "memory": 0.20 + vector.factual * 0.25,
            "metacognitive": vector.metacognitive * 0.4 + vector.analytical * 0.1,
        }
        return sorted(
            [(lobe, round(w, 3)) for lobe, w in mapping.items() if w > 0.15],
            key=lambda x: x[1],
            reverse=True,
        )

    def _select_neuron_strategy(self, vector: ModeVector) -> NeuronStrategy:
        if vector.creative > 0.5 or vector.exploratory > 0.5:
            temp = max(vector.creative, vector.exploratory) * 1.5
            return NeuronStrategy(
                use_stochastic=True,
                temperature=min(temp, 2.0),
                top_k_multiplier=1.0 + vector.creative * 0.5,
                noise_sigma=vector.creative * 0.3,
            )
        return NeuronStrategy()

    def _should_multi_pass(
        self, vector: ModeVector, tier: str,
    ) -> List[ModeSpectrum]:
        if tier != "T4":
            return []
        if vector.exploratory < 0.4 and vector.metacognitive < 0.4:
            return []
        chain: List[ModeSpectrum] = []
        if vector.exploratory > 0.3:
            chain.append(ModeSpectrum.EXPLORATORY)
        chain.append(ModeSpectrum.ANALYTICAL)
        if vector.critical > 0.3:
            chain.append(ModeSpectrum.CRITICAL)
        return chain

    def _should_adversarial_check(self, vector: ModeVector) -> bool:
        return vector.creative > 0.5 and vector.factual > 0.3

    def _build_tone_directive(
        self,
        vector: ModeVector,
        emotional_ctx: Dict[str, Any],
        motor_plan=None,
        user_sophistication: float = 0.5,
    ) -> ToneDirective:
        tone = "neutral"
        emotional_coloring = "none"
        verbosity = "normal"
        formality = 0.5
        reading_level = max(0.0, min(1.0, user_sophistication))

        if motor_plan:
            tone = getattr(motor_plan, "tone", tone)
            emotional_coloring = getattr(motor_plan, "emotional_coloring", emotional_coloring)
            verbosity = getattr(motor_plan, "verbosity", verbosity)
        else:
            threat = emotional_ctx.get("threat_level", 0)
            valence = emotional_ctx.get("valence", 0)
            if threat > 0.5:
                tone = "careful"
            elif vector.creative > 0.6:
                tone = "expressive"
            elif vector.empathetic > 0.5:
                tone = "warm"
            elif vector.analytical > 0.7:
                tone = "precise"
            elif vector.critical > 0.5:
                tone = "careful"

            if valence < -0.3:
                emotional_coloring = "reassuring"
            elif valence > 0.5:
                emotional_coloring = "encouraging"

            if vector.metacognitive > 0.5 or vector.analytical > 0.7:
                verbosity = "detailed"
            elif vector.factual > 0.8 and vector.creative < 0.2:
                verbosity = "concise"

        formality = 0.3 + vector.analytical * 0.3 + vector.factual * 0.2 - vector.creative * 0.2
        formality = max(0.0, min(1.0, formality))

        # Adjust verbosity by sophistication: low → brief, high → richer detail
        if reading_level < 0.25 and verbosity == "detailed":
            verbosity = "moderate"
        elif reading_level < 0.15:
            verbosity = "brief"

        traits: Dict[str, float] = {
            "humor": max(0.0, vector.creative * 0.4 + vector.empathetic * 0.2 - vector.critical * 0.3),
            "empathy": vector.empathetic * 0.8 + vector.creative * 0.1,
            "enthusiasm": max(0.0, vector.exploratory * 0.5 + vector.creative * 0.3),
            "precision": vector.analytical * 0.5 + vector.factual * 0.3,
            "curiosity": vector.exploratory * 0.6 + vector.metacognitive * 0.3,
        }

        return ToneDirective(
            tone=tone,
            emotional_coloring=emotional_coloring,
            verbosity=verbosity,
            formality=formality,
            reading_level=reading_level,
            personality_traits=traits,
        )

    # ── Outcome Recording ────────────────────────────────────────────

    def record_outcome(
        self,
        vector: ModeVector,
        tvs_score: float,
        domain: str,
        query_type: str = "",
        user_id: str = "",
        user_sophistication: float = 0.5,
    ) -> None:
        self._mode_history.append({
            "vector": vector.to_dict(),
            "tvs_score": tvs_score,
            "domain": domain,
            "query_type": query_type,
            "user_id": user_id,
            "timestamp": time.time(),
        })
        if len(self._mode_history) > 1000:
            self._mode_history = self._mode_history[-500:]

        self._domain_history.record(domain, query_type, vector, tvs_score)

        if user_id:
            self._user_store.update_profile(
                user_id, vector, tvs_score, domain,
                user_sophistication=user_sophistication)

    @property
    def stats(self) -> Dict[str, Any]:
        return {**self._stats, "history_size": len(self._mode_history)}


# ── Adversarial Mode Reviewer ────────────────────────────────────────────────

_ENTITY_RE = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
_DATE_RE = re.compile(
    r'\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|'
    r'July|August|September|October|November|December)\s+\d{4}|\d{4})\b'
)
_NUMERIC_RE = re.compile(
    r'\b(\d[\d,]*\.?\d*)\s*(?:%|km|mi|m/s|mph|kg|lb|°[CF]|MW|GW|Hz|'
    r'billion|million|trillion|meters|kilometres|kilometers|miles)\b',
    re.IGNORECASE,
)

_FILLER = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "has",
    "have", "had", "do", "does", "did", "will", "can", "may", "in", "of",
    "to", "for", "on", "with", "at", "by", "from", "as", "or", "and",
    "but", "not", "it", "its", "this", "that", "so", "if",
})


class AdversarialModeReviewer:
    """Deterministic fact-check pass for creative output.

    When the mode controller flags do_adversarial_check, this reviewer
    re-extracts factual claims from creative text and verifies them
    against VLT persistent memory and the causal graph.
    """

    def review(
        self,
        response_text: str,
        vlt=None,
        causal_graph=None,
    ) -> AdversarialResult:
        claims = self._extract_factual_from_creative(response_text)
        if not claims:
            return AdversarialResult()

        flagged: List[str] = []
        corrected: Dict[str, str] = {}
        confirmed = 0

        for claim in claims:
            verified, score, note = self._verify_claim(claim, vlt, causal_graph)
            if verified:
                confirmed += 1
            else:
                flagged.append(claim)
                if note:
                    corrected[claim] = note

        total = len(claims)
        factual_score = confirmed / total if total > 0 else 1.0

        if factual_score >= 0.8:
            recommendation = "pass"
        elif factual_score >= 0.5:
            recommendation = "warn"
        else:
            recommendation = "block"

        return AdversarialResult(
            flagged_claims=flagged,
            corrected_claims=corrected,
            overall_factual_score=factual_score,
            recommendation=recommendation,
        )

    def _extract_factual_from_creative(self, text: str) -> List[str]:
        claims: List[str] = []
        seen: Set[str] = set()

        for m in _ENTITY_RE.finditer(text):
            entity = m.group(1)
            key = entity.lower()
            if len(entity) > 2 and key not in _FILLER and key not in seen:
                seen.add(key)
                claims.append(entity)

        for m in _DATE_RE.finditer(text):
            d = m.group(1)
            if d not in seen:
                seen.add(d)
                claims.append(d)

        for m in _NUMERIC_RE.finditer(text):
            n = m.group(0)
            if n not in seen:
                seen.add(n)
                claims.append(n)

        return claims

    def _verify_claim(
        self, claim: str, vlt, causal_graph,
    ) -> Tuple[bool, float, str]:
        if vlt is None:
            return False, 0.0, "no_vlt_available"

        try:
            results = vlt.search(query=claim, min_confidence=0.5) if hasattr(vlt, "search") else []
            if not results:
                return False, 0.3, "no_matching_claims_in_vlt"

            claim_lower = claim.lower()
            for item in results[:5]:
                val = str(getattr(item, "value", "")).lower()
                if claim_lower in val or val in claim_lower:
                    return True, 0.9, ""
                tokens_claim = {w for w in claim_lower.split() if w not in _FILLER}
                tokens_source = set(val.split())
                if tokens_claim and tokens_claim.issubset(tokens_source):
                    return True, 0.8, ""
        except Exception as exc:
            logger.warning("_verify_claim VLT search error (fail-closed): %s", exc)
            return False, 0.3, "vlt_search_error"

        return False, 0.2, "not_found_in_vlt"


# ── User Mode Profile Store (Redis-backed) ──────────────────────────────────

class UserModeProfileStore:
    """Persists per-user mode preferences to Redis with EMA updates."""

    _PREFIX = "mode_profile:"

    def __init__(self, redis_client=None):
        self._redis = redis_client
        self._local: Dict[str, UserModeProfile] = {}

    def get_profile(self, user_id: str) -> Optional[UserModeProfile]:
        if not user_id:
            return None

        if user_id in self._local:
            return self._local[user_id]

        if self._redis:
            try:
                raw = self._redis.get(f"{self._PREFIX}{user_id}")
                if raw:
                    data = json.loads(raw)
                    profile = UserModeProfile(
                        user_id=user_id,
                        preferred_vector=ModeVector.from_dict(data.get("vector", {})),
                        domain_preferences=data.get("domain_preferences", {}),
                        interaction_count=data.get("interaction_count", 0),
                        last_updated=data.get("last_updated", 0),
                        sophistication_ema=data.get("sophistication_ema", 0.5),
                    )
                    self._local[user_id] = profile
                    return profile
            except Exception as exc:
                logger.debug("Redis profile load failed for %s: %s", user_id, exc)

        return None

    def update_profile(
        self,
        user_id: str,
        used_vector: ModeVector,
        tvs_score: float,
        domain: str = "",
        user_sophistication: float = 0.5,
    ) -> None:
        if not user_id:
            return

        profile = self.get_profile(user_id) or UserModeProfile(user_id=user_id)
        alpha = 0.1 * max(0.5, tvs_score)
        profile.preferred_vector = profile.preferred_vector.blend(used_vector, alpha)
        profile.interaction_count += 1
        profile.last_updated = time.time()

        soph_alpha = 0.15
        profile.sophistication_ema = (
            (1 - soph_alpha) * profile.sophistication_ema
            + soph_alpha * user_sophistication)

        if domain:
            existing = profile.domain_preferences.get(domain, {})
            existing_vec = ModeVector.from_dict(existing) if existing else ModeVector()
            blended = existing_vec.blend(used_vector, alpha)
            profile.domain_preferences[domain] = blended.to_dict()

        self._local[user_id] = profile

        if self._redis:
            try:
                data = {
                    "vector": profile.preferred_vector.to_dict(),
                    "domain_preferences": profile.domain_preferences,
                    "interaction_count": profile.interaction_count,
                    "last_updated": profile.last_updated,
                    "sophistication_ema": profile.sophistication_ema,
                }
                self._redis.set(
                    f"{self._PREFIX}{user_id}",
                    json.dumps(data),
                    ex=86400 * 90,
                )
            except Exception as exc:
                logger.debug("Redis profile save failed for %s: %s", user_id, exc)

    def clear_profile(self, user_id: str) -> None:
        self._local.pop(user_id, None)
        if self._redis:
            try:
                self._redis.delete(f"{self._PREFIX}{user_id}")
            except Exception as exc:
                logger.debug("Redis profile delete failed for %s: %s", user_id, exc)


# ── Domain Mode History (Redis-backed) ───────────────────────────────────────

class DomainModeHistory:
    """Tracks domain+query_type -> mode performance over time.

    Uses Redis sorted sets scored by TVS to find historically
    optimal mode vectors per domain/query_type combination.
    """

    _PREFIX = "mode_history:"
    _MAX_PER_KEY = 100

    def __init__(self, redis_client=None):
        self._redis = redis_client
        self._local: Dict[str, List[Dict[str, Any]]] = {}

    def record(
        self,
        domain: str,
        query_type: str,
        vector: ModeVector,
        tvs_score: float,
    ) -> None:
        key = f"{domain or 'general'}:{query_type or 'unknown'}"
        entry = {
            "vector": vector.to_dict(),
            "tvs_score": tvs_score,
            "timestamp": time.time(),
        }

        if key not in self._local:
            self._local[key] = []
        self._local[key].append(entry)
        if len(self._local[key]) > self._MAX_PER_KEY:
            self._local[key] = self._local[key][-self._MAX_PER_KEY:]

        if self._redis:
            try:
                member = json.dumps(entry)
                self._redis.zadd(f"{self._PREFIX}{key}", {member: tvs_score})
                self._redis.zremrangebyrank(
                    f"{self._PREFIX}{key}", 0, -(self._MAX_PER_KEY + 1),
                )
            except Exception as exc:
                logger.debug("Redis mode history record failed for %s: %s", key, exc)

    def optimal_vector(
        self, domain: str, query_type: str,
    ) -> Optional[ModeVector]:
        key = f"{domain or 'general'}:{query_type or 'unknown'}"

        entries: List[Dict[str, Any]] = []

        if self._redis:
            try:
                raw_list = self._redis.zrevrange(
                    f"{self._PREFIX}{key}", 0, 19, withscores=True,
                )
                for raw_member, score in raw_list:
                    data = json.loads(raw_member)
                    entries.append(data)
            except Exception as exc:
                logger.debug("Redis mode history read failed for %s: %s", key, exc)

        if not entries and key in self._local:
            sorted_local = sorted(
                self._local[key], key=lambda x: x.get("tvs_score", 0), reverse=True,
            )
            top_20_pct = max(1, len(sorted_local) // 5)
            entries = sorted_local[:top_20_pct]

        if not entries:
            return None

        avg = {f: 0.0 for f in ModeVector._fields()}
        for e in entries:
            vec = e.get("vector", {})
            for f in ModeVector._fields():
                avg[f] += vec.get(f, 0.0)
        n = len(entries)
        for f in ModeVector._fields():
            avg[f] /= n

        return ModeVector.from_dict(avg)

    def get_stats(self, domain: str) -> Dict[str, Any]:
        prefix = f"{domain}:"
        stats: Dict[str, Any] = {"domain": domain, "query_types": {}}
        for key, entries in self._local.items():
            if key.startswith(prefix):
                qt = key.split(":", 1)[1] if ":" in key else "unknown"
                scores = [e.get("tvs_score", 0) for e in entries]
                stats["query_types"][qt] = {
                    "count": len(entries),
                    "avg_tvs": sum(scores) / len(scores) if scores else 0,
                    "max_tvs": max(scores) if scores else 0,
                }
        return stats


# ── Mode Shift Event (for streaming) ────────────────────────────────────────

@dataclass
class ModeShiftEvent:
    """Emitted during mid-stream mode re-evaluation."""
    section_text: str = ""
    section_mode: ModeSpectrum = ModeSpectrum.HYBRID
    mode_vector: Dict[str, float] = field(default_factory=dict)
    is_mode_shift: bool = False
    shift_reason: str = ""
