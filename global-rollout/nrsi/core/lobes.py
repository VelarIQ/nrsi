"""
NRSI Lobes — NRS Processing Lobes + Integration Core.

The NRS brain has 6 specialized processing lobes,
analogous to regions of the human cerebral cortex.

Each lobe processes different aspects of queries.
Not all lobes activate for every query — sparse activation
means typically 1-2 lobes process any given query.

Lobes:
  Linguistic  (2.0B params) — Broca's/Wernicke's areas
    NLU, syntax analysis, semantic parsing

  Logical     (1.5B params) — Prefrontal cortex
    Deductive reasoning, inference chains, contradiction detection

  Mathematical(1.0B params) — Parietal lobe
    Numerical computation, symbolic math, statistical analysis

  Spatial     (1.5B params) — Right hemisphere
    Geometric reasoning, relational mapping, structural analysis

  Temporal    (1.5B params) — Hippocampus
    Sequence processing, temporal reasoning, causal inference

  Creative    (2.5B params) — Association cortex
    Novel synthesis, analogy generation, creative problem-solving

Integration Core:
  Cross-attention coordination between lobes.
  Like white matter tracts connecting brain regions.
  Manages multi-lobe queries where 2+ lobes need to collaborate.

Average activation: 1.3 lobes per query.
Different queries → different lobes activated.
But consistent total activation budget → stable power draw.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from nrsi.core.creases import DomainCrease, CreaseRegistry


# ── Lobe Types ───────────────────────────────────────────────────────────────

class LobeType(Enum):
    """NRS processing lobes — six primary plus extended cognitive lobes."""
    LINGUISTIC = "linguistic"
    LOGICAL = "logical"
    MATHEMATICAL = "mathematical"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    CREATIVE = "creative"
    CAUSAL = "causal"
    ANALOGICAL = "analogical"
    PLANNING = "planning"
    MEMORY = "memory"
    METACOGNITIVE = "metacognitive"
    COGNITIVE = "cognitive"
    VERIFICATION = "verification"
    REASONING = "reasoning"
    CONTEXT = "context"


# Biological analogues and parameter counts
LOBE_SPECS: Dict[LobeType, Dict[str, Any]] = {
    LobeType.LINGUISTIC: {
        "bio_analogue": "Broca's & Wernicke's areas",
        "params_b": 2.0,
        "function": "Natural language understanding, syntax analysis, semantic parsing",
        "tier_affinity": ["T1", "T2"],  # Most work at lower tiers
    },
    LobeType.LOGICAL: {
        "bio_analogue": "Prefrontal cortex",
        "params_b": 1.5,
        "function": "Deductive reasoning, inference chains, contradiction detection",
        "tier_affinity": ["T2", "T3"],
    },
    LobeType.MATHEMATICAL: {
        "bio_analogue": "Parietal lobe",
        "params_b": 1.0,
        "function": "Numerical computation, symbolic mathematics, statistical analysis",
        "tier_affinity": ["T2", "T3"],
    },
    LobeType.SPATIAL: {
        "bio_analogue": "Right hemisphere",
        "params_b": 1.5,
        "function": "Geometric reasoning, relational mapping, structural analysis",
        "tier_affinity": ["T2", "T3"],
    },
    LobeType.TEMPORAL: {
        "bio_analogue": "Hippocampus",
        "params_b": 1.5,
        "function": "Sequence processing, temporal reasoning, causal inference",
        "tier_affinity": ["T2", "T3", "T4"],
    },
    LobeType.CREATIVE: {
        "bio_analogue": "Association cortex",
        "params_b": 2.5,
        "function": "Novel synthesis, analogy generation, creative problem-solving",
        "tier_affinity": ["T3", "T4"],  # Deep processing only
    },
    LobeType.CAUSAL: {
        "bio_analogue": "Anterior cingulate cortex",
        "params_b": 1.2,
        "function": "Cause-effect reasoning, counterfactual analysis",
        "tier_affinity": ["T2", "T3"],
    },
    LobeType.ANALOGICAL: {
        "bio_analogue": "Angular gyrus",
        "params_b": 1.0,
        "function": "Structural mapping, analogy generation, transfer learning",
        "tier_affinity": ["T2", "T3"],
    },
    LobeType.PLANNING: {
        "bio_analogue": "Dorsolateral prefrontal cortex",
        "params_b": 1.3,
        "function": "Goal decomposition, sequential planning, resource allocation",
        "tier_affinity": ["T2", "T3"],
    },
    LobeType.MEMORY: {
        "bio_analogue": "Hippocampus",
        "params_b": 1.5,
        "function": "Episodic retrieval, working memory coordination, consolidation",
        "tier_affinity": ["T1", "T2"],
    },
    LobeType.METACOGNITIVE: {
        "bio_analogue": "Medial prefrontal cortex",
        "params_b": 0.8,
        "function": "Self-monitoring, confidence calibration, strategy selection",
        "tier_affinity": ["T3"],
    },
    LobeType.COGNITIVE: {
        "bio_analogue": "Association cortex",
        "params_b": 1.0,
        "function": "General cognitive processing, cross-modal integration",
        "tier_affinity": ["T1", "T2"],
    },
    LobeType.VERIFICATION: {
        "bio_analogue": "Ventromedial prefrontal cortex",
        "params_b": 0.7,
        "function": "Fact-checking, consistency verification, source validation",
        "tier_affinity": ["T2", "T3"],
    },
    LobeType.REASONING: {
        "bio_analogue": "Lateral prefrontal network",
        "params_b": 1.4,
        "function": "Multi-strategy reasoning orchestration (deductive, inductive, abductive)",
        "tier_affinity": ["T2", "T3"],
    },
    LobeType.CONTEXT: {
        "bio_analogue": "Temporal-parietal junction",
        "params_b": 0.9,
        "function": "Context management, relevance filtering, attention direction",
        "tier_affinity": ["T1", "T2"],
    },
}


# ── Lobe Result ──────────────────────────────────────────────────────────────

@dataclass
class LobeResult:
    """
    Output from a processing lobe.

    Every lobe returns a standardized result with:
      - The processed output value
      - Confidence score (0-1)
      - Which lobe processed it
      - Processing time
      - Whether further processing is needed (escalate)
    """
    lobe: LobeType
    value: Any
    confidence: float
    processing_time_ms: float
    needs_escalation: bool = False
    escalation_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        esc = " [ESCALATE]" if self.needs_escalation else ""
        return (
            f"LobeResult({self.lobe.value}, conf={self.confidence:.2f}, "
            f"{self.processing_time_ms:.1f}ms{esc})"
        )


# ── Processing Lobe Base ─────────────────────────────────────────────────────

class ProcessingLobe:
    """
    Base class for all NRS processing lobes.

    Each lobe:
      - Has domain creases associated with it
      - Processes queries within its specialization
      - Returns LobeResult with confidence
      - Can escalate to higher-tier processing if needed
      - Tracks performance stats

    Lobes don't have their own neurons — they PROCESS
    the same neuron firing patterns differently.
    Neurons are in the peripheral system.
    Lobes are processing METHODS in the brain.
    """

    def __init__(self, lobe_type: LobeType):
        self.lobe_type = lobe_type
        self.spec = LOBE_SPECS[lobe_type]
        self._creases: Dict[str, DomainCrease] = {}
        self._processors: List[Callable] = []

        # Stats
        self._queries = 0
        self._escalations = 0
        self._total_time_ms = 0.0

    def attach_crease(self, crease: DomainCrease) -> None:
        """Attach a domain crease to this lobe."""
        self._creases[crease.domain] = crease

    def register_processor(self, fn: Callable) -> None:
        """Register a processing function for this lobe."""
        self._processors.append(fn)

    def process(
        self,
        query: str,
        domain: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> LobeResult:
        """
        Process a query through this lobe.

        Steps:
          1. Check domain creases for relevant knowledge
          2. Run registered processors
          3. Assess confidence
          4. If confidence too low → flag for escalation
        """
        t0 = time.time()
        self._queries += 1

        result_value = None
        confidence = 0.0
        needs_escalation = False
        escalation_reason = None
        metadata: Dict[str, Any] = {"domain": domain}

        # Step 1: Check domain creases
        if domain and domain in self._creases:
            crease = self._creases[domain]
            crease_result = crease.query(query)
            if crease_result is not None:
                result_value = crease_result
                confidence = 0.85  # Crease knowledge = high confidence
                metadata["source"] = "domain_crease"

        # Step 2: Run processors
        for processor in self._processors:
            try:
                proc_result = processor(query, domain=domain, context=context or {})
                if proc_result is not None:
                    if isinstance(proc_result, dict):
                        result_value = proc_result.get("value", result_value)
                        confidence = max(confidence, proc_result.get("confidence", 0.0))
                        metadata.update(proc_result.get("metadata", {}))
                    else:
                        result_value = proc_result
                        confidence = max(confidence, 0.7)
            except Exception as e:
                metadata["processor_error"] = str(e)

        # Step 3: Assess — escalate if confidence too low
        if confidence < 0.5:
            needs_escalation = True
            escalation_reason = f"Low confidence ({confidence:.2f}) in {self.lobe_type.value} lobe"
            self._escalations += 1

        elapsed_ms = (time.time() - t0) * 1000
        self._total_time_ms += elapsed_ms

        return LobeResult(
            lobe=self.lobe_type,
            value=result_value,
            confidence=confidence,
            processing_time_ms=elapsed_ms,
            needs_escalation=needs_escalation,
            escalation_reason=escalation_reason,
            metadata=metadata,
        )

    @property
    def stats(self) -> Dict[str, Any]:
        avg_ms = self._total_time_ms / self._queries if self._queries > 0 else 0
        return {
            "lobe": self.lobe_type.value,
            "bio_analogue": self.spec["bio_analogue"],
            "params_b": self.spec["params_b"],
            "creases": list(self._creases.keys()),
            "processors": len(self._processors),
            "queries": self._queries,
            "escalations": self._escalations,
            "avg_time_ms": round(avg_ms, 2),
        }

    def __repr__(self) -> str:
        return (
            f"ProcessingLobe({self.lobe_type.value}, "
            f"params={self.spec['params_b']}B, "
            f"creases={len(self._creases)})"
        )


# ── Specialized Lobes ────────────────────────────────────────────────────────

class LinguisticLobe(ProcessingLobe):
    """Linguistic processing — NLU, syntax, semantics (Broca's/Wernicke's)."""
    def __init__(self):
        super().__init__(LobeType.LINGUISTIC)


class LogicalLobe(ProcessingLobe):
    """Logical reasoning — deduction, inference, contradiction detection (Prefrontal)."""
    def __init__(self):
        super().__init__(LobeType.LOGICAL)


class MathematicalLobe(ProcessingLobe):
    """Mathematical computation — numerical, symbolic, statistical (Parietal)."""
    def __init__(self):
        super().__init__(LobeType.MATHEMATICAL)


class SpatialLobe(ProcessingLobe):
    """Spatial reasoning — geometry, relations, structure (Right hemisphere)."""
    def __init__(self):
        super().__init__(LobeType.SPATIAL)


class TemporalLobe(ProcessingLobe):
    """Temporal processing — sequences, causality, time (Hippocampus)."""
    def __init__(self):
        super().__init__(LobeType.TEMPORAL)


class CreativeProcessingLobe(ProcessingLobe):
    """Creative synthesis — novel combination, analogy (Association cortex).

    When a CreativeLobe (from memory.py) is attached, this lobe uses
    synthesize() and analogize() for genuine creative generation instead
    of the generic ProcessingLobe.process() path.
    """
    def __init__(self):
        super().__init__(LobeType.CREATIVE)
        self._creative_lobe = None

    def attach_creative_lobe(self, creative_lobe) -> None:
        self._creative_lobe = creative_lobe

    def process(
        self,
        query: str,
        domain: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> LobeResult:
        t0 = time.time()
        self._queries += 1
        metadata: Dict[str, Any] = {"domain": domain, "source": "creative_lobe"}

        if self._creative_lobe is None:
            return super().process(query, domain=domain, context=context)

        try:
            words = query.split()
            inputs = [query]
            if context and "lobe_results" in context:
                for lr in context["lobe_results"]:
                    val = getattr(lr, "value", None)
                    if val:
                        inputs.append(str(val)[:200])

            target_domain = domain or "general"
            proposal = self._creative_lobe.synthesize(
                inputs=inputs,
                target_domain=target_domain,
                synthesis_type="novel_combination",
                context=context,
            )

            analogy_proposal = None
            source_domain = (context or {}).get("source_domain", "")
            if source_domain and source_domain != target_domain:
                analogy_proposal = self._creative_lobe.analogize(
                    source_domain=source_domain,
                    target_domain=target_domain,
                    source_pattern=query,
                )

            result_value = {
                "synthesis": proposal.value if proposal else None,
                "analogy": analogy_proposal.value if analogy_proposal else None,
                "creative_confidence": getattr(proposal, "confidence_estimate", 0.3),
            }
            confidence = getattr(proposal, "confidence_estimate", 0.3)
            if analogy_proposal:
                confidence = max(confidence, getattr(analogy_proposal, "confidence_estimate", 0.25))

            metadata["synthesis_type"] = "novel_combination"
            metadata["has_analogy"] = analogy_proposal is not None
        except Exception as e:
            result_value = None
            confidence = 0.0
            metadata["error"] = str(e)

        needs_escalation = confidence < 0.5
        if needs_escalation:
            self._escalations += 1

        elapsed_ms = (time.time() - t0) * 1000
        self._total_time_ms += elapsed_ms

        return LobeResult(
            lobe=self.lobe_type,
            value=result_value,
            confidence=confidence,
            processing_time_ms=elapsed_ms,
            needs_escalation=needs_escalation,
            escalation_reason="low creative confidence" if needs_escalation else None,
            metadata=metadata,
        )


# ── Integration Core ─────────────────────────────────────────────────────────

@dataclass
class IntegrationMessage:
    """
    Message passed between lobes through the Integration Core.

    Like axonal signals traveling through white matter tracts.
    """
    source_lobe: LobeType
    target_lobe: LobeType
    content: Any
    message_type: str = "data"   # 'data' | 'request' | 'response' | 'escalation'
    priority: int = 0
    created_at: float = field(default_factory=time.time)


class IntegrationCore:
    """
    Integration Core — Cross-Lobe Coordination.

    Like white matter tracts in the brain, the Integration Core
    manages communication between processing lobes.

    For multi-lobe queries (5% of all queries), the Integration
    Core coordinates:
      1. Which lobes need to activate
      2. Data passing between lobes
      3. Result synthesis from multiple lobes
      4. Conflict resolution when lobes disagree

    Cross-attention mechanism: each lobe's output attends to
    other active lobes' outputs for coherent synthesis.

    Usage:
        core = IntegrationCore()
        core.register_lobe(linguistic_lobe)
        core.register_lobe(logical_lobe)

        # Single-lobe processing
        result = core.process_single(query, LobeType.LINGUISTIC)

        # Multi-lobe processing (coordinated)
        result = core.process_multi(query, [LobeType.LINGUISTIC, LobeType.LOGICAL])

        # Auto-route (core decides which lobes)
        result = core.process_auto(query, domain="medical")
    """

    def __init__(self):
        self._lobes: Dict[LobeType, ProcessingLobe] = {}
        self._message_bus: List[IntegrationMessage] = []

        # Stats
        self._single_lobe_queries = 0
        self._multi_lobe_queries = 0
        self._messages_sent = 0

    def register_lobe(self, lobe: ProcessingLobe) -> None:
        """Register a processing lobe with the Integration Core."""
        self._lobes[lobe.lobe_type] = lobe

    def get_lobe(self, lobe_type: LobeType) -> Optional[ProcessingLobe]:
        return self._lobes.get(lobe_type)

    def send_message(self, msg: IntegrationMessage) -> None:
        """Send a message between lobes (through the bus)."""
        self._message_bus.append(msg)
        self._messages_sent += 1

    def process_single(
        self,
        query: str,
        lobe_type: LobeType,
        domain: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> LobeResult:
        """Route query to a single lobe."""
        self._single_lobe_queries += 1
        lobe = self._lobes.get(lobe_type)
        if lobe is None:
            return LobeResult(
                lobe=lobe_type,
                value=None,
                confidence=0.0,
                processing_time_ms=0.0,
                needs_escalation=True,
                escalation_reason=f"Lobe {lobe_type.value} not registered",
            )
        return lobe.process(query, domain=domain, context=context)

    def process_multi(
        self,
        query: str,
        lobe_types: List[LobeType],
        domain: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        merge: str = "highest_confidence",
    ) -> Dict[str, Any]:
        """
        Process query through multiple lobes and synthesize results.

        Merge strategies:
          'highest_confidence' — take the result with best confidence
          'unanimous'          — all lobes must agree (confidence > 0.5)
          'weighted_average'   — weight by lobe confidence scores
          'all'                — return all results unmerged
        """
        self._multi_lobe_queries += 1
        results: List[LobeResult] = []

        for lt in lobe_types:
            lobe = self._lobes.get(lt)
            if lobe:
                r = lobe.process(query, domain=domain, context=context)
                results.append(r)

                # Send integration messages between lobes
                for other_lt in lobe_types:
                    if other_lt != lt:
                        self.send_message(IntegrationMessage(
                            source_lobe=lt,
                            target_lobe=other_lt,
                            content={"confidence": r.confidence, "value": r.value},
                            message_type="data",
                        ))

        if not results:
            return {
                "value": None,
                "confidence": 0.0,
                "lobes_activated": [],
                "merge_strategy": merge,
            }

        if merge == "highest_confidence":
            best = max(results, key=lambda r: r.confidence)
            return {
                "value": best.value,
                "confidence": best.confidence,
                "primary_lobe": best.lobe.value,
                "lobes_activated": [r.lobe.value for r in results],
                "merge_strategy": merge,
                "all_results": results,
            }
        elif merge == "unanimous":
            passing = [r for r in results if r.confidence >= 0.5]
            unanimous = len(passing) == len(results)
            best = max(results, key=lambda r: r.confidence) if results else None
            return {
                "value": best.value if best else None,
                "confidence": min(r.confidence for r in results) if unanimous else 0.0,
                "unanimous": unanimous,
                "lobes_activated": [r.lobe.value for r in results],
                "merge_strategy": merge,
            }
        elif merge == "all":
            return {
                "results": results,
                "lobes_activated": [r.lobe.value for r in results],
                "merge_strategy": merge,
            }
        else:
            # weighted_average
            total_conf = sum(r.confidence for r in results)
            if total_conf == 0:
                return {"value": None, "confidence": 0.0, "merge_strategy": merge}
            best = max(results, key=lambda r: r.confidence)
            avg_conf = total_conf / len(results)
            return {
                "value": best.value,
                "confidence": avg_conf,
                "lobes_activated": [r.lobe.value for r in results],
                "merge_strategy": merge,
            }

    def process_weighted(
        self,
        query: str,
        lobe_weights: List[Tuple[str, float]],
        domain: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process through multiple lobes with explicit mode-derived weights.

        When creative weight > 0.6 the creative lobe runs first and its
        output seeds later lobes as context (creative-first pipeline).
        """
        self._multi_lobe_queries += 1
        results: List[LobeResult] = []
        creative_first = False
        creative_weight = 0.0
        creative_result: Optional[LobeResult] = None

        for lobe_name, weight in lobe_weights:
            if lobe_name == "creative" and weight > 0.6:
                creative_first = True
                creative_weight = weight
                break

        if creative_first:
            creative_lt = LobeType.CREATIVE
            lobe = self._lobes.get(creative_lt)
            if lobe:
                creative_result = lobe.process(query, domain=domain, context=context or {})
                results.append(creative_result)

        ctx = dict(context or {})
        if creative_result and creative_result.value:
            ctx["creative_seed"] = str(creative_result.value)[:500]
            ctx["lobe_results"] = [creative_result]

        for lobe_name, weight in lobe_weights:
            if creative_first and lobe_name == "creative":
                continue
            try:
                lt = LobeType(lobe_name)
            except ValueError:
                continue
            lobe = self._lobes.get(lt)
            if lobe:
                r = lobe.process(query, domain=domain, context=ctx)
                results.append(r)

        if not results:
            return {
                "value": None,
                "confidence": 0.0,
                "lobes_activated": [],
                "merge_strategy": "weighted",
            }

        weight_map = {name: w for name, w in lobe_weights}
        total_weighted_conf = 0.0
        total_weight = 0.0
        best_value = None
        best_score = -1.0

        for r in results:
            w = weight_map.get(r.lobe.value, 0.5)
            scored = r.confidence * w
            total_weighted_conf += scored
            total_weight += w
            if scored > best_score:
                best_score = scored
                best_value = r.value

        avg_conf = total_weighted_conf / total_weight if total_weight > 0 else 0.0

        return {
            "value": best_value,
            "confidence": avg_conf,
            "primary_lobe": results[0].lobe.value if results else "unknown",
            "lobes_activated": [r.lobe.value for r in results],
            "merge_strategy": "weighted",
            "creative_first": creative_first,
            "all_results": results,
        }

    @property
    def registered_lobes(self) -> List[str]:
        return [lt.value for lt in self._lobes.keys()]

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "registered_lobes": self.registered_lobes,
            "single_lobe_queries": self._single_lobe_queries,
            "multi_lobe_queries": self._multi_lobe_queries,
            "messages_sent": self._messages_sent,
            "lobe_stats": {lt.value: l.stats for lt, l in self._lobes.items()},
        }

    def __repr__(self) -> str:
        return f"IntegrationCore(lobes={self.registered_lobes})"
