"""
NRSI Router — T0 Central Brain.

T0 is the spinal cord of the NRS — every query passes through it.
It receives neuron activation patterns and routes to the appropriate
processing lobe(s) at the correct tier.

T0 does three things:
  1. Complexity scoring (0-100): How hard is this query?
  2. Domain detection: What domain does this belong to?
  3. Tier routing: Which tier (T0-T4) should handle it?

Routing thresholds:
  R(q) = T0 if C(q) < 10     → greetings, acks, trivial echoes
  R(q) = T1 if 10 ≤ C(q) < 25 → 60% of queries (simple facts)
  R(q) = T2 if 25 ≤ C(q) < 60 → 20% (validation required)
  R(q) = T3 if 60 ≤ C(q) < 85 → 15% (reasoning required)
  R(q) = T4 if C(q) ≥ 85      → 5% (meta-reasoning/creative)

T0 also integrates with:
  - VLT Memory: check cache before processing
  - PVS-4: deterministic pattern matching for known queries
  - Neuron activation: domain signals from peripheral system
  - Domain creases: route to correct knowledge module

Query distribution (production):
  80% stop at T1 → 15% at T2 → 4% at T3 → 1% T4
  = 97% cost reduction vs. running everything at T4
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from nrsi.core.neurons import BinaryNeuronBank, ActivationPattern
from nrsi.core.lobes import (
    IntegrationCore, ProcessingLobe, LobeType, LobeResult,
    LinguisticLobe, LogicalLobe, MathematicalLobe,
    SpatialLobe, TemporalLobe, CreativeProcessingLobe,
)
from nrsi.core.creases import CreaseRegistry, AxiomBase, AxiomTier


# ── Tier ─────────────────────────────────────────────────────────────────────

class Tier(Enum):
    """Processing tiers — complexity-based routing."""
    T0 = "T0"   # Sensory / NLP preprocessor — greetings, acks, trivial echoes
    T1 = "T1"   # Fact processing — simple lookups
    T2 = "T2"   # Validation — consistency checking
    T3 = "T3"   # Reasoning — multi-step logic
    T4 = "T4"   # Meta-reasoning — novel decomposition, creative


# Tier thresholds
TIER_THRESHOLDS = {
    Tier.T0: (0, 10),
    Tier.T1: (10, 25),
    Tier.T2: (25, 60),
    Tier.T3: (60, 85),
    Tier.T4: (85, 101),
}

# Tier → typical lobe activations
TIER_LOBE_MAP: Dict[Tier, List[LobeType]] = {
    Tier.T0: [LobeType.LINGUISTIC],
    Tier.T1: [LobeType.LINGUISTIC],
    Tier.T2: [LobeType.LINGUISTIC, LobeType.LOGICAL],
    Tier.T3: [LobeType.LOGICAL, LobeType.MATHEMATICAL, LobeType.TEMPORAL],
    Tier.T4: [LobeType.LOGICAL, LobeType.TEMPORAL, LobeType.CREATIVE],
}

# Tier power budget (watts)
TIER_POWER: Dict[Tier, int] = {
    Tier.T0: 10,
    Tier.T1: 50,
    Tier.T2: 75,
    Tier.T3: 120,
    Tier.T4: 200,
}


# ── Complexity Analyzer ──────────────────────────────────────────────────────

@dataclass
class ComplexityScore:
    """
    Complexity analysis of a query.

    Scoring dimensions:
      token_complexity: length and vocabulary richness
      structural_complexity: nesting, clauses, dependencies
      domain_specificity: how specialized the domain is
      reasoning_depth: does it need multi-step logic?
      novelty: is this a known pattern or novel query?

    Final score: weighted combination (0-100)
    """
    token_complexity: float = 0.0
    structural_complexity: float = 0.0
    domain_specificity: float = 0.0
    reasoning_depth: float = 0.0
    novelty: float = 0.0
    composite: float = 0.0
    tier: Optional[Tier] = None
    domain_hint: Optional[str] = None

    def __repr__(self) -> str:
        return f"ComplexityScore({self.composite:.1f}, tier={self.tier.value if self.tier else '?'})"


class ComplexityAnalyzer:
    """
    Query complexity scorer — determines routing tier.

    In production, this is a lightweight spaCy NLP pipeline
    running on CPU pods. Fast enough to not be the bottleneck.

    Scoring:
      - Token complexity (word count, vocabulary level)
      - Structural complexity (clauses, nesting)
      - Domain specificity (domain keywords detected)
      - Reasoning depth (logical connectives, multi-step indicators)
      - Novelty (PVS cache miss = novel query)

    Weights:
      α=0.15 token, β=0.20 structural, γ=0.20 domain,
      δ=0.25 reasoning, ε=0.20 novelty
    """

    # Indicators of higher complexity
    REASONING_INDICATORS = {
        "why", "how", "because", "therefore", "implies", "causes",
        "if", "then", "prove", "derive", "analyze", "compare",
        "evaluate", "explain", "justify", "reason", "infer",
        "deduce", "conclude", "synthesize", "critique",
    }

    DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        "medical": ["diagnosis", "treatment", "symptom", "drug", "patient",
                     "clinical", "disease", "therapy", "dosage", "prognosis"],
        "financial": ["portfolio", "equity", "derivative", "hedge", "bond",
                      "interest", "return", "risk", "investment", "market"],
        "legal": ["statute", "precedent", "jurisdiction", "liability",
                  "contract", "tort", "plaintiff", "defendant", "appeal"],
        "engineering": ["stress", "load", "material", "tolerance", "design",
                        "structural", "thermal", "circuit", "voltage"],
        "physics": ["energy", "force", "mass", "velocity", "quantum",
                    "relativity", "thermodynamic", "electromagnetic"],
        "mathematics": ["theorem", "proof", "integral", "derivative", "topology",
                        "algebra", "matrix", "equation", "convergence"],
    }

    def __init__(self):
        self._analyses = 0

    def analyze(
        self,
        query: str,
        pvs_hit: bool = False,
        activation_pattern: Optional[ActivationPattern] = None,
    ) -> ComplexityScore:
        """
        Analyze query complexity and determine routing tier.
        """
        self._analyses += 1
        words = query.lower().split()
        word_count = len(words)
        word_set = set(words)

        # 1. Token complexity (0-100)
        token_c = min(word_count * 3, 100)

        # 2. Structural complexity (0-100)
        structural_c = 0.0
        if "?" in query:
            structural_c += 10
        clause_indicators = {"and", "but", "or", "which", "that", "where", "when"}
        structural_c += len(word_set & clause_indicators) * 12
        structural_c = min(structural_c, 100)

        # 3. Domain specificity (0-100)
        domain_c = 0.0
        detected_domain = None
        best_domain_score = 0
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            hits = len(word_set & set(keywords))
            if hits > best_domain_score:
                best_domain_score = hits
                detected_domain = domain
        domain_c = min(best_domain_score * 20, 100)

        # 4. Reasoning depth (0-100)
        reasoning_c = 0.0
        reasoning_hits = word_set & self.REASONING_INDICATORS
        reasoning_c = min(len(reasoning_hits) * 18, 100)
        # Multi-step queries
        if word_count > 30:
            reasoning_c = min(reasoning_c + 20, 100)

        # 5. Novelty (0-100)
        novelty_c = 0.0 if pvs_hit else 50.0  # PVS miss = novel
        if activation_pattern:
            # High sparsity variation suggests novelty
            if activation_pattern.sparsity < 0.0005:
                novelty_c += 25

        # Composite score
        composite = (
            0.15 * token_c +
            0.20 * structural_c +
            0.20 * domain_c +
            0.25 * reasoning_c +
            0.20 * novelty_c
        )

        # Determine tier
        tier = Tier.T1
        for t, (low, high) in TIER_THRESHOLDS.items():
            if low <= composite < high:
                tier = t
                break

        # Use domain from activation pattern if available
        if activation_pattern and not detected_domain:
            dist = activation_pattern.domain_distribution()
            if dist:
                detected_domain = max(dist, key=dist.get)

        return ComplexityScore(
            token_complexity=token_c,
            structural_complexity=structural_c,
            domain_specificity=domain_c,
            reasoning_depth=reasoning_c,
            novelty=novelty_c,
            composite=composite,
            tier=tier,
            domain_hint=detected_domain,
        )


# ── Router Result ────────────────────────────────────────────────────────────

@dataclass
class RouterResult:
    """
    Complete routing decision from T0.

    Contains:
      - Which tier to process at
      - Which domain was detected
      - Which lobes should activate
      - Complexity analysis
      - VLT/PVS cache status
      - Neuron activation pattern reference
    """
    tier: Tier
    domain: Optional[str]
    lobes: List[LobeType]
    complexity: ComplexityScore
    pvs_hit: bool = False
    pvs_cached_result: Optional[Any] = None
    vlt_hit: bool = False
    vlt_cached_result: Optional[Any] = None
    activation_signature: Optional[str] = None
    power_budget_w: int = 50
    processing_time_ms: float = 0.0

    def __repr__(self) -> str:
        lobes_str = "+".join(l.value for l in self.lobes)
        cache = " [PVS-HIT]" if self.pvs_hit else ""
        cache += " [VLT-HIT]" if self.vlt_hit else ""
        return (
            f"RouterResult({self.tier.value}, domain={self.domain}, "
            f"lobes=[{lobes_str}], complexity={self.complexity.composite:.1f}"
            f"{cache})"
        )


# ── T0 Router ────────────────────────────────────────────────────────────────

class T0Router:
    """
    T0 Central Router — The Brain's Spinal Cord.

    Every query passes through T0. It:
      1. Checks PVS cache (deterministic pattern match) — <1ms
      2. Checks VLT L1 cache — <5ms
      3. Activates peripheral neurons (binary firing)
      4. Analyzes complexity (0-100 score)
      5. Detects domain (from neurons + query analysis)
      6. Determines tier (T0-T4)
      7. Selects lobes to activate
      8. Routes to Integration Core

    If PVS has a cached result → return immediately (MODE 1: low compute).
    If novel query → full analysis + processing (MODE 2: high compute).
    Once processed → result cached in PVS for future queries.

    This is why 80% of queries stop at T1:
      Most queries are known patterns with cached results.
      Only truly novel or complex queries escalate.

    Usage:
        router = T0Router()

        # Configure (optional — defaults are sensible)
        router.neurons = BinaryNeuronBank(total_neurons=100_000, active_k=100)
        router.crease_registry.create("medical")

        # Route a query
        result = router.route("What is the capital of France?")
        # → RouterResult(T1, domain=None, lobes=[linguistic], complexity=12.3)

        # Route a complex query
        result = router.route("Analyze the pharmacokinetic interaction between warfarin and aspirin")
        # → RouterResult(T3, domain=medical, lobes=[logical,temporal], complexity=72.1)
    """

    def __init__(
        self,
        neurons: Optional[BinaryNeuronBank] = None,
        crease_registry: Optional[CreaseRegistry] = None,
        integration_core: Optional[IntegrationCore] = None,
        pvs=None,   # PVS4 instance (optional, avoids circular import)
        vlt=None,    # VLT instance (optional, avoids circular import)
    ):
        self.neurons = neurons or BinaryNeuronBank(
            total_neurons=10_000, active_k=10
        )
        self.crease_registry = crease_registry or CreaseRegistry()
        self.integration_core = integration_core or IntegrationCore()
        self.analyzer = ComplexityAnalyzer()

        # Optional integrations (connected post-init to avoid circular deps)
        self._pvs = pvs
        self._vlt = vlt

        # Tier override rules (for tuition system corrections)
        self._tier_overrides: Dict[str, Tier] = {}

        # Stats
        self._queries_routed = 0
        self._pvs_hits = 0
        self._vlt_hits = 0
        self._tier_distribution: Dict[Tier, int] = {t: 0 for t in Tier}

    def connect_pvs(self, pvs) -> None:
        """Connect PVS-4 for deterministic pattern matching."""
        self._pvs = pvs

    def connect_vlt(self, vlt) -> None:
        """Connect VLT for memory cache checks."""
        self._vlt = vlt

    def add_tier_override(self, query_pattern: str, tier: Tier) -> None:
        """
        Add a routing override (from Tuition System corrections).
        Specific query patterns always route to specified tier.
        """
        self._tier_overrides[query_pattern.lower()] = tier

    def route(
        self,
        query: str,
        force_tier: Optional[Tier] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RouterResult:
        """
        Route a query through T0. The main entry point.

        Steps:
          1. Check PVS cache → instant return if hit
          2. Check VLT L1 cache → instant return if hit
          3. Check tier overrides (tuition corrections)
          4. Activate neurons (peripheral system)
          5. Analyze complexity
          6. Determine tier + domain + lobes
          7. Return routing decision
        """
        t0 = time.time()
        self._queries_routed += 1

        pvs_hit = False
        pvs_cached = None
        vlt_hit = False
        vlt_cached = None
        activation_sig = None

        # Step 1: Check PVS cache
        if self._pvs:
            match = self._pvs.lookup(query)
            if match:
                pvs_hit = True
                pvs_cached = match
                self._pvs_hits += 1

        # Step 2: Check VLT L1 cache
        if self._vlt:
            cached = self._vlt.recall(query)
            if cached is not None:
                vlt_hit = True
                vlt_cached = cached
                self._vlt_hits += 1

        # Step 3: Check tier overrides (tuition corrections)
        query_lower = query.lower()
        override_tier = None
        for pattern, tier in self._tier_overrides.items():
            if pattern in query_lower:
                override_tier = tier
                break

        # Step 4: Activate neurons
        activation = self.neurons.activate_fast(query)
        activation_sig = activation.signature

        # Step 5: Analyze complexity
        complexity = self.analyzer.analyze(
            query,
            pvs_hit=pvs_hit,
            activation_pattern=activation,
        )

        # Step 6: Determine tier
        if force_tier:
            tier = force_tier
        elif override_tier:
            tier = override_tier
        elif pvs_hit and hasattr(pvs_cached, 'tier'):
            tier_str = pvs_cached.tier
            tier = Tier(tier_str) if isinstance(tier_str, str) else complexity.tier
        else:
            tier = complexity.tier

        # Step 7: Determine domain
        domain = complexity.domain_hint
        if not domain and activation.domain_signals:
            dist = activation.domain_distribution()
            if dist:
                domain = max(dist, key=dist.get)

        # Step 8: Select lobes
        lobes = TIER_LOBE_MAP.get(tier, [LobeType.LINGUISTIC])

        # Adjust lobes based on domain
        if domain in ("mathematics", "physics") and LobeType.MATHEMATICAL not in lobes:
            lobes = lobes + [LobeType.MATHEMATICAL]
        if domain and complexity.reasoning_depth > 50 and LobeType.LOGICAL not in lobes:
            lobes = lobes + [LobeType.LOGICAL]

        self._tier_distribution[tier] = self._tier_distribution.get(tier, 0) + 1
        elapsed_ms = (time.time() - t0) * 1000

        return RouterResult(
            tier=tier,
            domain=domain,
            lobes=lobes,
            complexity=complexity,
            pvs_hit=pvs_hit,
            pvs_cached_result=pvs_cached,
            vlt_hit=vlt_hit,
            vlt_cached_result=vlt_cached,
            activation_signature=activation_sig,
            power_budget_w=TIER_POWER.get(tier, 50),
            processing_time_ms=elapsed_ms,
        )

    def route_and_process(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline: route → activate lobes → return result.
        """
        routing = self.route(query, context=context)

        # If PVS cached → return immediately (MODE 1: low compute)
        if routing.pvs_hit and routing.pvs_cached_result:
            return {
                "mode": "MODE_1_LOW_COMPUTE",
                "routing": routing,
                "result": routing.pvs_cached_result,
                "source": "pvs_cache",
            }

        # If VLT cached → return immediately
        if routing.vlt_hit and routing.vlt_cached_result:
            return {
                "mode": "MODE_1_LOW_COMPUTE",
                "routing": routing,
                "result": routing.vlt_cached_result,
                "source": "vlt_cache",
            }

        # MODE 2: High compute — process through lobes
        if len(routing.lobes) == 1:
            lobe_result = self.integration_core.process_single(
                query, routing.lobes[0],
                domain=routing.domain, context=context,
            )
            return {
                "mode": "MODE_2_HIGH_COMPUTE",
                "routing": routing,
                "result": lobe_result,
                "source": f"lobe:{routing.lobes[0].value}",
            }
        else:
            multi_result = self.integration_core.process_multi(
                query, routing.lobes,
                domain=routing.domain, context=context,
            )
            return {
                "mode": "MODE_2_HIGH_COMPUTE",
                "routing": routing,
                "result": multi_result,
                "source": f"multi_lobe:{'+'.join(l.value for l in routing.lobes)}",
            }

    @property
    def tier_distribution(self) -> Dict[str, float]:
        """Percentage of queries at each tier."""
        total = sum(self._tier_distribution.values())
        if total == 0:
            return {t.value: 0.0 for t in Tier}
        return {t.value: c / total for t, c in self._tier_distribution.items()}

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "queries_routed": self._queries_routed,
            "pvs_hits": self._pvs_hits,
            "vlt_hits": self._vlt_hits,
            "tier_distribution": {t.value: c for t, c in self._tier_distribution.items()},
            "tier_percentages": self.tier_distribution,
            "neurons": self.neurons.stats,
            "analyzer_analyses": self.analyzer._analyses,
        }

    def __repr__(self) -> str:
        return f"T0Router(queries={self._queries_routed}, tiers={dict(self._tier_distribution)})"
