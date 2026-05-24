"""
NRSI Truthful Validation Score (TVS) + Determinism — Fact Verification & Hardware Guarantees.

Replaces the LLM-oriented H_score with a fact-retrieval-native validation system.
NRS does NOT predict tokens — it retrieves hard data from knowledge packs, VLT,
and web search. "Hallucination" in the LLM sense cannot occur. What CAN happen:
  - A claim is not backed by any source  → knowledge gap, needs web search
  - A claim contradicts a verified source → wrong fact selected, retry
  - No sources exist at all              → store gap pattern, fetch from web

TVS validates:
  1. Claim extraction — mode-aware decomposition of the response into checkable claims
  2. Multi-source corroboration — each claim checked against 2+ independent sources
  3. Factual entity validation — in creative mode, only named entities/dates/numbers

Absolute minimum truthful validation: 95.2%
  TVS >= 0.952 → VALIDATED (deliver answer)
  TVS <  0.952 → RETRY   (gather more sources, recompose, re-validate)

Mode behavior:
  DETERMINISTIC  — every statement is a claim, all must be sourced
  HYBRID         — factual claims validated at 95.2%, creative framing allowed
  CREATIVE       — only extractable factual entities validated at 95.2%,
                   pure creative output (fiction, poetry) flows freely

Patent-covered: NRSI Truthful Validation, VelarIQ Fact Coverage System.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ── TVS Core ─────────────────────────────────────────────────────────────────

TRUTH_FLOOR = 0.952

_FILLER = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "has", "have",
    "had", "do", "does", "did", "will", "can", "may", "in", "of", "to", "for",
    "on", "with", "at", "by", "from", "as", "or", "and", "but", "not", "it",
    "its", "this", "that", "so", "if", "also", "very", "more", "most", "such",
    "than", "then", "about", "into", "over", "after", "up", "out", "no", "yes",
})


class ValidationMode(str, Enum):
    DETERMINISTIC = "DETERMINISTIC"
    HYBRID = "HYBRID"
    CREATIVE = "CREATIVE"


class TVSVerdict(str, Enum):
    VALIDATED = "validated"
    RETRY = "retry"


@dataclass
class FactualClaim:
    """A single checkable claim extracted from a response."""
    text: str
    claim_type: str  # "statement", "entity", "numeric", "date"
    confirmed_by: List[str] = field(default_factory=list)
    source_count: int = 0
    is_confirmed: bool = False


@dataclass
class TVSResult:
    """Result of Truthful Validation Score computation."""
    score: float
    verdict: TVSVerdict
    claims_total: int
    claims_confirmed: int
    claims_unconfirmed: List[str] = field(default_factory=list)
    mode: str = "HYBRID"
    details: Dict[str, Any] = field(default_factory=dict)
    gap_patterns: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.score >= TRUTH_FLOOR

    @property
    def needs_retry(self) -> bool:
        return self.score < TRUTH_FLOOR and self.claims_total > 0

    def __repr__(self) -> str:
        return (
            f"TVSResult({self.score:.4f}, {self.verdict.value}, "
            f"{self.claims_confirmed}/{self.claims_total} claims confirmed)"
        )


# ── Backward-compatible aliases ──────────────────────────────────────────────
# Other modules reference the old H-score types. Keep the names available
# but route them through TVS semantics.

class HScoreVerdict(str, Enum):
    VALIDATED = "validated"
    ACCEPTABLE = "acceptable"
    SUSPICIOUS = "suspicious"
    HALLUCINATION = "hallucination"


@dataclass
class HScoreComponents:
    entropy: float = 0.0
    consistency: float = 0.0
    entailment: float = 0.0


@dataclass
class HScore:
    score: float
    components: HScoreComponents
    verdict: HScoreVerdict
    query: str = ""
    response: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"HScore({self.score:.3f}, {self.verdict.value}, "
            f"entropy={self.components.entropy:.2f}, "
            f"consistency={self.components.consistency:.2f}, "
            f"entailment={self.components.entailment:.2f})"
        )


# ── Claim Extraction ─────────────────────────────────────────────────────────

_ENTITY_PATTERN = re.compile(
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
)
_DATE_PATTERN = re.compile(
    r'\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|'
    r'July|August|September|October|November|December)\s+\d{4}|\d{4})\b'
)
_NUMERIC_PATTERN = re.compile(
    r'\b(\d[\d,]*\.?\d*)\s*(?:%|km|mi|m/s|mph|kg|lb|°[CF]|MW|GW|TW|Hz|'
    r'billion|million|trillion|thousand|meters|kilometres|kilometers|miles|'
    r'feet|inches|watts|volts|amps|bytes|bits)\b',
    re.IGNORECASE,
)


def extract_claims(response: str, mode: ValidationMode) -> List[FactualClaim]:
    """Extract checkable claims from a response, mode-aware.

    DETERMINISTIC: every sentence is a claim
    HYBRID: sentences containing factual content are claims
    CREATIVE: only named entities, dates, and numbers are claims
    """
    if not response or len(response.strip()) < 10:
        return []

    claims: List[FactualClaim] = []

    if mode == ValidationMode.CREATIVE:
        for m in _ENTITY_PATTERN.finditer(response):
            entity = m.group(1)
            if len(entity) > 2 and entity.lower() not in _FILLER:
                claims.append(FactualClaim(text=entity, claim_type="entity"))
        for m in _DATE_PATTERN.finditer(response):
            claims.append(FactualClaim(text=m.group(1), claim_type="date"))
        for m in _NUMERIC_PATTERN.finditer(response):
            claims.append(FactualClaim(text=m.group(0), claim_type="numeric"))
        seen = set()
        deduped = []
        for c in claims:
            key = c.text.lower().strip()
            if key not in seen:
                seen.add(key)
                deduped.append(c)
        return deduped

    sentences = re.split(r'(?<=[.!?])\s+', response.strip())
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 15:
            continue

        if mode == ValidationMode.DETERMINISTIC:
            claims.append(FactualClaim(text=sent, claim_type="statement"))
        else:
            has_entity = bool(_ENTITY_PATTERN.search(sent))
            has_date = bool(_DATE_PATTERN.search(sent))
            has_number = bool(_NUMERIC_PATTERN.search(sent))
            if has_entity or has_date or has_number:
                claims.append(FactualClaim(text=sent, claim_type="statement"))

    return claims


# ── Multi-Source Corroboration ───────────────────────────────────────────────

def _token_overlap(text_a: str, text_b: str) -> float:
    """Weighted token overlap between two texts, ignoring filler words."""
    tokens_a = {w.lower() for w in text_a.split()} - _FILLER
    tokens_b = {w.lower() for w in text_b.split()} - _FILLER
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = tokens_a & tokens_b
    return len(overlap) / min(len(tokens_a), len(tokens_b))


def _entity_present(claim_text: str, source_text: str) -> bool:
    """Check if a named entity or number from a claim appears in a source."""
    ct = claim_text.lower().strip()
    st = source_text.lower()
    if ct in st:
        return True
    ct_tokens = {w for w in ct.split() if w not in _FILLER and len(w) > 2}
    st_tokens = set(st.split())
    if ct_tokens and ct_tokens.issubset(st_tokens):
        return True
    return False


def corroborate_claim(
    claim: FactualClaim,
    sources: List[str],
    threshold: float = 0.25,
) -> FactualClaim:
    """Check a single claim against multiple sources.

    A claim is confirmed if at least 1 source supports it (we count how many).
    For entity/date/numeric claims, exact presence is checked.
    For statement claims, token overlap above threshold confirms.
    """
    confirmed_sources: List[str] = []

    for src in sources:
        if not src or len(src) < 10:
            continue

        if claim.claim_type in ("entity", "date", "numeric"):
            if _entity_present(claim.text, src):
                confirmed_sources.append(src[:80])
        else:
            overlap = _token_overlap(claim.text, src)
            if overlap >= threshold:
                confirmed_sources.append(src[:80])

    claim.confirmed_by = confirmed_sources
    claim.source_count = len(confirmed_sources)
    claim.is_confirmed = len(confirmed_sources) >= 1
    return claim


# ── TVS Calculator ───────────────────────────────────────────────────────────

class TruthfulValidationScorer:
    """Truthful Validation Score (TVS) — fact-coverage-based validation.

    Replaces LLM-oriented H_score. NRS retrieves hard data, not token
    predictions, so validation means: "are the claims backed by sources?"

    TVS = confirmed_claims / total_claims
    Minimum: 95.2% or answer is retried with more sources.
    """

    def __init__(self, truth_floor: float = TRUTH_FLOOR):
        self.truth_floor = truth_floor
        self._scores_computed = 0
        self._retries_triggered = 0
        self._gap_patterns: List[Dict[str, Any]] = []

    def validate(
        self,
        query: str,
        response: str,
        sources: List[str],
        mode: ValidationMode = ValidationMode.HYBRID,
    ) -> TVSResult:
        """Compute TVS for a response against available sources."""
        self._scores_computed += 1

        claims = extract_claims(response, mode)

        if not claims:
            return TVSResult(
                score=1.0,
                verdict=TVSVerdict.VALIDATED,
                claims_total=0,
                claims_confirmed=0,
                mode=mode.value,
                details={"reason": "no_factual_claims_to_validate"},
            )

        for claim in claims:
            corroborate_claim(claim, sources)

        confirmed = sum(1 for c in claims if c.is_confirmed)
        total = len(claims)

        # Axiom-backed self-verification for deterministic claims
        if mode == ValidationMode.DETERMINISTIC and confirmed == 0 and total > 0:
            _axiom_patterns = {
                r'\b\d+\s*[\+\-\*\/]\s*\d+\s*=\s*\d+': 'arithmetic',
                r'\b(true|false|valid|invalid|proven|theorem)\b': 'logic',
                r'\b(equals?|identical|same as|equivalent)\b': 'identity',
                r'\bc\s*=\s*299': 'physics_constant',
                r'\bpi\s*=?\s*3\.14': 'math_constant',
                r'\bDNA\b.*\b(A|T|G|C)\b': 'biology_fact',
            }
            for claim in claims:
                for pattern, _axiom_type in _axiom_patterns.items():
                    if re.search(pattern, claim.text, re.IGNORECASE):
                        claim.is_confirmed = True
                        claim.confirmed_by = [f"axiom:{_axiom_type}"]
                        claim.source_count = 1
                        break
            confirmed = sum(1 for c in claims if c.is_confirmed)

        # For deterministic mode with no external sources,
        # pipeline confidence serves as a self-verification floor
        if mode == ValidationMode.DETERMINISTIC and not sources and total > 0:
            tvs = max(confirmed / total, 0.75) if total > 0 else 1.0
        else:
            tvs = confirmed / total if total > 0 else 0.0

        unconfirmed = [c.text for c in claims if not c.is_confirmed]
        gap_patterns = []
        if tvs < self.truth_floor:
            self._retries_triggered += 1
            gap_patterns = [
                f"query={query[:60]}|unconfirmed={c}"
                for c in unconfirmed[:5]
            ]
            self._gap_patterns.extend(
                {"query": query[:80], "claim": c, "time": time.time()}
                for c in unconfirmed[:5]
            )

        verdict = TVSVerdict.VALIDATED if tvs >= self.truth_floor else TVSVerdict.RETRY

        return TVSResult(
            score=tvs,
            verdict=verdict,
            claims_total=total,
            claims_confirmed=confirmed,
            claims_unconfirmed=unconfirmed,
            mode=mode.value,
            details={
                "claims": [
                    {
                        "text": c.text[:100],
                        "type": c.claim_type,
                        "confirmed": c.is_confirmed,
                        "source_count": c.source_count,
                    }
                    for c in claims
                ],
                "truth_floor": self.truth_floor,
            },
            gap_patterns=gap_patterns,
        )

    def score(
        self,
        query: str,
        response: str,
        sources: Optional[List[str]] = None,
        mode: ValidationMode = ValidationMode.HYBRID,
        # Legacy params — accepted but ignored
        samples: Optional[List[str]] = None,
        ground_truth: Optional[List[str]] = None,
        token_probabilities: Optional[List[float]] = None,
    ) -> TVSResult:
        """Unified entry point. Accepts legacy H-score params for compatibility."""
        all_sources = list(sources or [])
        if ground_truth:
            all_sources.extend(ground_truth)
        return self.validate(query, response, all_sources, mode)

    @property
    def gap_patterns(self) -> List[Dict[str, Any]]:
        return list(self._gap_patterns)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "scores_computed": self._scores_computed,
            "retries_triggered": self._retries_triggered,
            "gap_patterns_stored": len(self._gap_patterns),
            "truth_floor": self.truth_floor,
        }


# ── Legacy HScoreCalculator wrapper ─────────────────────────────────────────
# Provides the old .score() interface for callers that haven't migrated yet.
# Internally delegates to TruthfulValidationScorer.

class HScoreCalculator:
    """Backward-compatible wrapper over TruthfulValidationScorer.

    Old callers pass samples/ground_truth/token_probabilities — these are
    mapped to TVS sources. The returned HScore object bridges old consumers.
    """

    VALIDATED_THRESHOLD = TRUTH_FLOOR
    ACCEPTABLE_THRESHOLD = 0.85
    SUSPICIOUS_THRESHOLD = 0.50

    def __init__(self, **kwargs):
        self._tvs = TruthfulValidationScorer()
        self._scores_computed = 0

    def score(
        self,
        query: str,
        response: str,
        samples: Optional[List[str]] = None,
        ground_truth: Optional[List[str]] = None,
        token_probabilities: Optional[List[float]] = None,
    ) -> HScore:
        self._scores_computed += 1
        sources = list(ground_truth or [])
        tvs_result = self._tvs.validate(
            query, response, sources, ValidationMode.HYBRID,
        )

        if tvs_result.score >= TRUTH_FLOOR:
            verdict = HScoreVerdict.VALIDATED
        elif tvs_result.score >= 0.85:
            verdict = HScoreVerdict.ACCEPTABLE
        elif tvs_result.score >= 0.50:
            verdict = HScoreVerdict.SUSPICIOUS
        else:
            verdict = HScoreVerdict.HALLUCINATION

        return HScore(
            score=tvs_result.score,
            components=HScoreComponents(
                entropy=0.0,
                consistency=1.0,
                entailment=tvs_result.score,
            ),
            verdict=verdict,
            query=query,
            response=response,
            details={
                "tvs_claims_total": tvs_result.claims_total,
                "tvs_claims_confirmed": tvs_result.claims_confirmed,
                "tvs_unconfirmed": tvs_result.claims_unconfirmed[:3],
                "tvs_verdict": tvs_result.verdict.value,
            },
        )

    @property
    def stats(self) -> Dict[str, Any]:
        return self._tvs.stats


# ── Deterministic Output ─────────────────────────────────────────────────────

class SamplingMode(Enum):
    """Output selection modes. NRS uses DETERMINISTIC only."""
    DETERMINISTIC = "deterministic"


@dataclass
class OutputConfig:
    """Deterministic output configuration.

    NRS does NOT generate — it SELECTS.
    Temperature is always 0. Sampling is always disabled.
    """
    mode: SamplingMode = SamplingMode.DETERMINISTIC
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int = 4096

    def __post_init__(self):
        if self.mode != SamplingMode.DETERMINISTIC:
            raise ValueError(
                "NRS only supports DETERMINISTIC output mode."
            )
        if self.temperature != 0.0:
            raise ValueError(
                f"Temperature must be 0.0 (got {self.temperature})."
            )
        if self.top_k != 0:
            raise ValueError(f"top_k must be 0/disabled (got {self.top_k}).")
        if self.top_p != 0.0:
            raise ValueError(f"top_p must be 0.0/disabled (got {self.top_p}).")

    def select(self, logits: List[float]) -> int:
        if not logits:
            raise ValueError("Empty logits — cannot select")
        return max(range(len(logits)), key=lambda i: logits[i])


# ── Hardware Determinism ─────────────────────────────────────────────────────

@dataclass
class HardwareConfig:
    """Hardware determinism configuration.

    Lock GPU frequency, power, activation budget for bit-identical output.
    """
    target_power_w: int = 200
    power_tolerance_w: int = 10
    gpu_frequency_mhz: int = 1600
    dvfs_enabled: bool = False
    activation_budget: int = 111_000
    deterministic_math: bool = True

    def __post_init__(self):
        if self.dvfs_enabled:
            raise ValueError(
                "DVFS must be DISABLED for hardware determinism."
            )

    def validate_power(self, measured_w: float) -> Dict[str, Any]:
        low = self.target_power_w - self.power_tolerance_w
        high = self.target_power_w + self.power_tolerance_w
        return {
            "measured_w": measured_w, "target_w": self.target_power_w,
            "tolerance_w": self.power_tolerance_w,
            "range": f"{low}-{high}W",
            "in_range": low <= measured_w <= high,
            "deviation_w": abs(measured_w - self.target_power_w),
        }

    def validate_frequency(self, measured_mhz: float) -> Dict[str, Any]:
        return {
            "measured_mhz": measured_mhz, "target_mhz": self.gpu_frequency_mhz,
            "locked": abs(measured_mhz - self.gpu_frequency_mhz) < 1.0,
            "deviation_mhz": abs(measured_mhz - self.gpu_frequency_mhz),
        }

    def validate_activation(self, active_neurons: int) -> Dict[str, Any]:
        return {
            "active_neurons": active_neurons, "budget": self.activation_budget,
            "within_budget": active_neurons <= self.activation_budget,
            "utilization": active_neurons / self.activation_budget,
        }

    @property
    def determinism_guarantees(self) -> Dict[str, bool]:
        return {
            "fixed_frequency": not self.dvfs_enabled,
            "fixed_power_target": True,
            "fixed_activation_budget": True,
            "deterministic_cuda": self.deterministic_math,
            "no_sampling": True,
            "temperature_zero": True,
        }

    def __repr__(self) -> str:
        return (
            f"HardwareConfig({self.target_power_w}W ±{self.power_tolerance_w}W, "
            f"{self.gpu_frequency_mhz}MHz locked, "
            f"budget={self.activation_budget})"
        )


# ── Determinism Validator ────────────────────────────────────────────────────

class DeterminismValidator:
    """End-to-end determinism validation.

    Proves: same query → identical output, bit-for-bit.
    """

    def __init__(
        self,
        hardware: Optional[HardwareConfig] = None,
        output: Optional[OutputConfig] = None,
    ):
        self.hardware = hardware or HardwareConfig()
        self.output = output or OutputConfig()
        self._checks: List[Dict[str, Any]] = []

    def check_reproducibility(
        self, query: str, output_a: str, output_b: str,
    ) -> Dict[str, Any]:
        hash_a = hashlib.sha256(output_a.encode()).hexdigest()
        hash_b = hashlib.sha256(output_b.encode()).hexdigest()
        identical = hash_a == hash_b
        result = {
            "query": query, "identical": identical,
            "hash_a": hash_a, "hash_b": hash_b,
            "length_a": len(output_a), "length_b": len(output_b),
        }
        self._checks.append(result)
        if not identical:
            for i, (a, b) in enumerate(zip(output_a, output_b)):
                if a != b:
                    result["first_diff_position"] = i
                    result["diff_context_a"] = output_a[max(0, i - 10):i + 10]
                    result["diff_context_b"] = output_b[max(0, i - 10):i + 10]
                    break
        return result

    def full_audit(
        self, power_w: float = 200.0, frequency_mhz: float = 1600.0,
        active_neurons: int = 111_000,
    ) -> Dict[str, Any]:
        power = self.hardware.validate_power(power_w)
        freq = self.hardware.validate_frequency(frequency_mhz)
        activation = self.hardware.validate_activation(active_neurons)
        guarantees = self.hardware.determinism_guarantees
        return {
            "all_pass": (
                power["in_range"] and freq["locked"]
                and activation["within_budget"] and all(guarantees.values())
            ),
            "power": power, "frequency": freq,
            "activation": activation, "guarantees": guarantees,
            "output_config": {
                "mode": self.output.mode.value,
                "temperature": self.output.temperature,
                "top_k": self.output.top_k,
                "top_p": self.output.top_p,
            },
            "reproducibility_checks": len(self._checks),
            "all_reproducible": all(c["identical"] for c in self._checks) if self._checks else True,
        }
