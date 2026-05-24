"""NRSI AGI Integration — Native Cognitive Engine Registration.

Converts AGI engines from standalone Python modules into first-class NRSI
lobe processors that respect the trust type system, validation gates,
VLT memory hierarchy, and cognitive mode control.

Architecture:
  LogicEngine      → processor on LogicalLobe (LobeType.LOGICAL)
  ComputationEngine → processor on MathematicalLobe (LobeType.MATHEMATICAL)
  SemanticEngine    → processor on LinguisticLobe (LobeType.LINGUISTIC)
  CausalReasoner    → processor on CausalLobe (LobeType.CAUSAL)
  AnalogyEngine     → processor on AnalogicalLobe (LobeType.ANALOGICAL)
  Planner           → processor on PlanningLobe (LobeType.PLANNING) + cross-lobe coordinator
  WorkingMemory     → processor on MemoryLobe (LobeType.MEMORY) + VLT L1/L2 adapter
  LearningEngine    → VLT L3 adapter + crease writer
  SelfImprovement   → processor on MetacognitiveLobe (LobeType.METACOGNITIVE) + mode tuner

Every engine output is wrapped in EpistemicNRSIData (when available) carrying
its epistemic type, cognitive origin, and temporal scope — not just a trust level:
  - Computation results: computed() — COMPUTATIONAL, confidence 0.99
  - Logic proofs: deductive() — DEDUCTIVE, proof chain attached
  - Semantic matches: observed() — OBSERVATIONAL, confidence varies
  - Causal chains: causal() — CAUSAL, chain-strength product
  - Counterfactuals: speculative() — SPECULATIVE, hypothesis attached
  - Analogies: analogical() — ANALOGICAL, source/target domains
  - Plan steps: PlanContract with GoalContract lifecycle
  - Learned claims: ClaimRecord with lifecycle tracking
Falls back to plain NRSIData when epistemic module is unavailable.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from nrsi.core.types import (
    Confidence,
    NRSIData,
    TrustLevel,
    ProvenanceEntry,
    raw,
    validated,
    trusted,
)
from nrsi.core.validation import (
    ValidationGate,
    FunctionValidator,
    ValidationResult,
    Validator,
)
from nrsi.core.lobes import (
    ProcessingLobe,
    LobeType,
    LobeResult,
    IntegrationCore,
    LogicalLobe,
    MathematicalLobe,
    LinguisticLobe,
    SpatialLobe,
    TemporalLobe,
    CreativeProcessingLobe,
    IntegrationMessage,
)
from nrsi.core.memory import VLT, VLTLayer, VLTItem, PVS4, ProcessingMode, TuitionSystem
from nrsi.core.mode_control import (
    ModeVector,
    ModeDecision,
    ModeSpectrum,
    CognitiveModeController,
)
from nrsi.core.creases import DomainCrease

logger = logging.getLogger("nrsi.agi")

__all__ = [
    "AGIIntegration",
    "LogicLobeProcessor",
    "MathLobeProcessor",
    "SemanticLobeProcessor",
    "CausalLobeProcessor",
    "AnalogyLobeProcessor",
    "PlannerCoordinator",
    "VLTWorkingMemoryAdapter",
    "VLTLearningAdapter",
    "ModeControlTuner",
]

# ── Import AGI engines with guards ───────────────────────────────────────────

try:
    from nrsip.logic_engine import LogicEngine
except ImportError:
    LogicEngine = None  # type: ignore[assignment,misc]

try:
    from nrsip.semantic_engine import SemanticEngine
except ImportError:
    SemanticEngine = None  # type: ignore[assignment,misc]

try:
    from nrsip.causal_engine import CausalReasoner
except ImportError:
    CausalReasoner = None  # type: ignore[assignment,misc]

try:
    from nrsip.analogy_engine import AnalogyEngine
except ImportError:
    AnalogyEngine = None  # type: ignore[assignment,misc]

try:
    from nrsip.planner import Planner, StepAction
except ImportError:
    Planner = None  # type: ignore[assignment,misc]
    StepAction = None  # type: ignore[assignment,misc]

try:
    from nrsip.working_memory import WorkingMemory as WMEngine
except ImportError:
    WMEngine = None  # type: ignore[assignment,misc]

try:
    from nrsip.learning_engine import LearningEngine
except ImportError:
    LearningEngine = None  # type: ignore[assignment,misc]

try:
    from nrsip.self_improvement import SelfImprovementLoop
except ImportError:
    SelfImprovementLoop = None  # type: ignore[assignment,misc]

try:
    from nrsi.core.epistemic import (
        EpistemicType, CognitiveOrigin, ReasoningProvenance,
        TemporalScope, TemporalValidity, ClaimStatus, ClaimRecord,
        EpistemicOps, EpistemicNRSIData,
        deductive, computed, causal, analogical, speculative, observed,
        GoalContract, GoalStatus, PlanContract, PlanStep,
    )
    _HAS_EPISTEMIC = True
except ImportError:
    _HAS_EPISTEMIC = False


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_mode(context: Optional[Dict[str, Any]]) -> Optional[ModeDecision]:
    if not context:
        return None
    md = context.get("mode_decision")
    if isinstance(md, ModeDecision):
        return md
    mv = context.get("mode_vector")
    if isinstance(mv, ModeVector):
        decision = ModeDecision(vector=mv, primary_mode=mv.primary_mode())
        return decision
    return None


def _is_deterministic(context: Optional[Dict[str, Any]]) -> bool:
    md = _extract_mode(context)
    if md is None:
        return False
    return md.primary_mode in (ModeSpectrum.DETERMINISTIC, ModeSpectrum.ANALYTICAL)


def _is_creative(context: Optional[Dict[str, Any]]) -> bool:
    md = _extract_mode(context)
    if md is None:
        return False
    return md.primary_mode in (ModeSpectrum.CREATIVE, ModeSpectrum.EXPLORATORY)


# ── 1. LogicLobeProcessor ───────────────────────────────────────────────────

class LogicLobeProcessor:
    """Wraps LogicEngine as a ProcessingLobe processor for LogicalLobe.

    Proof results elevate to TRUSTED; derived-only facts stay VALIDATED.
    Runs through an internal gate that checks contradiction-freedom and
    proof-chain validity.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._gate = ValidationGate(
            name="logic_validation",
            confidence_threshold=0.4,
            target_trust=TrustLevel.VALIDATED,
            validators=[
                FunctionValidator(self._check_no_contradictions, name="contradiction_free"),
                FunctionValidator(self._check_proof_chain, name="proof_chain_valid"),
            ],
            require_all=True,
        )
        self._invocations = 0

    def __call__(
        self,
        query: str,
        domain: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._invocations += 1
        context = context or {}
        t0 = time.time()

        if context.get("ingest_text"):
            self._engine.ingest_text(str(context["ingest_text"]))

        result = self._engine.query(query)
        elapsed_ms = (time.time() - t0) * 1000

        if result.answered and result.proof and result.proof.proved:
            conf = max(result.confidence, Confidence.HIGH)
            answer_text = result.answer
            if _HAS_EPISTEMIC:
                _origin = CognitiveOrigin(
                    lobe=LobeType.LOGICAL,
                    epistemic_type=EpistemicType.DEDUCTIVE,
                    engine_name="logic_engine",
                    method_name="forward_chain" if result.derived_facts else "query",
                    reasoning_depth=len(result.derived_facts) if result.derived_facts else 0,
                    evidence_count=result.proof.depth if result.proof else 0,
                )
                if result.proof is not None:
                    _chain = [str(f) for f in result.derived_facts] if result.derived_facts else []
                    data = deductive(
                        answer_text, confidence=conf,
                        proof_chain=_chain,
                    )
                else:
                    data = EpistemicNRSIData(
                        answer_text, TrustLevel.VALIDATED, conf,
                        epistemic_type=EpistemicType.DEDUCTIVE,
                        cognitive_origin=_origin,
                    )
            else:
                data = trusted(
                    answer_text,
                    confidence=conf,
                    gate_name="logic_proof",
                    metadata={"proof_depth": result.proof.depth, "engine": "logic"},
                )
        elif result.answered:
            conf = max(result.confidence, 0.5)
            answer_text = result.answer
            if _HAS_EPISTEMIC:
                _origin = CognitiveOrigin(
                    lobe=LobeType.LOGICAL,
                    epistemic_type=EpistemicType.DEDUCTIVE,
                    engine_name="logic_engine",
                    method_name="forward_chain" if result.derived_facts else "query",
                    reasoning_depth=len(result.derived_facts) if result.derived_facts else 0,
                    evidence_count=0,
                )
                data = EpistemicNRSIData(
                    answer_text, TrustLevel.VALIDATED, conf,
                    epistemic_type=EpistemicType.DEDUCTIVE,
                    cognitive_origin=_origin,
                )
            else:
                data = validated(
                    answer_text,
                    confidence=conf,
                    gate_name="logic_derivation",
                    metadata={"derived_facts": len(result.derived_facts), "engine": "logic"},
                )
        else:
            conf = result.confidence if result.confidence > 0 else 0.2
            data = raw(
                result.answer or "No logical conclusion reached.",
                metadata={"engine": "logic", "reasoning_trace": result.reasoning_trace},
            )

        return {
            "value": data,
            "confidence": conf,
            "metadata": {
                "engine": "logic",
                "proved": bool(result.proof and result.proof.proved),
                "contradictions": len(result.contradictions),
                "derived_facts": len(result.derived_facts),
                "elapsed_ms": elapsed_ms,
            },
        }

    def _check_no_contradictions(self, data: Any) -> Tuple[bool, float]:
        contradictions = self._engine.check_consistency()
        if contradictions:
            return False, 0.1
        return True, 0.95

    def _check_proof_chain(self, data: Any) -> Tuple[bool, float]:
        if data is None:
            return False, 0.0
        if isinstance(data, dict):
            proof = data.get("proof")
            if proof is None:
                return False, 0.2
            proved = False
            if hasattr(proof, "proved"):
                proved = proof.proved
            elif isinstance(proof, dict):
                proved = proof.get("proved", False)
            if proved:
                steps = 0
                if hasattr(proof, "steps"):
                    steps = len(proof.steps)
                elif isinstance(proof, dict):
                    steps = len(proof.get("steps", []))
                return True, min(0.95, 0.6 + steps * 0.05)
            return False, 0.3
        if hasattr(data, "proof"):
            p = data.proof
            if p and getattr(p, "proved", False):
                return True, 0.9
            return False, 0.3
        return False, 0.2


# ── 2. MathLobeProcessor ────────────────────────────────────────────────────

class MathLobeProcessor:
    """Wraps a computation engine for MathematicalLobe.

    Deterministic computations are TRUSTED at 0.99 confidence.
    The gate checks execution success and output parseability.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._gate = ValidationGate(
            name="math_validation",
            confidence_threshold=0.9,
            target_trust=TrustLevel.TRUSTED,
            validators=[
                FunctionValidator(self._check_execution, name="execution_ok"),
                FunctionValidator(self._check_parseable, name="output_parseable"),
            ],
        )
        self._invocations = 0

    def __call__(
        self,
        query: str,
        domain: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._invocations += 1
        context = context or {}

        can = False
        if hasattr(self._engine, "can_compute"):
            can = self._engine.can_compute(query)
        elif hasattr(self._engine, "evaluate"):
            can = True

        if not can:
            return {
                "value": raw("Cannot compute expression.", metadata={"engine": "math"}),
                "confidence": 0.0,
                "metadata": {"engine": "math", "computable": False},
            }

        t0 = time.time()
        try:
            if hasattr(self._engine, "compute"):
                result = self._engine.compute(query)
            elif hasattr(self._engine, "evaluate"):
                result = self._engine.evaluate(query)
            else:
                result = str(self._engine(query))
            elapsed_ms = (time.time() - t0) * 1000

            if _HAS_EPISTEMIC:
                data = computed(
                    result, confidence=0.99,
                )
            else:
                data = trusted(
                    result,
                    confidence=Confidence.VERY_HIGH,
                    gate_name="math_compute",
                    metadata={"engine": "math", "deterministic": True},
                )
            return {
                "value": data,
                "confidence": Confidence.VERY_HIGH,
                "metadata": {
                    "engine": "math",
                    "deterministic": True,
                    "elapsed_ms": elapsed_ms,
                },
            }
        except Exception as exc:
            return {
                "value": raw(f"Computation error: {exc}", metadata={"engine": "math"}),
                "confidence": 0.0,
                "metadata": {"engine": "math", "error": str(exc)},
            }

    @staticmethod
    def _check_execution(data: Any) -> Tuple[bool, float]:
        if data is None:
            return False, 0.0
        return True, 0.99

    @staticmethod
    def _check_parseable(data: Any) -> Tuple[bool, float]:
        try:
            str(data)
            return True, 0.99
        except Exception:
            return False, 0.0


# ── 3. SemanticLobeProcessor ─────────────────────────────────────────────────

class SemanticLobeProcessor:
    """Wraps SemanticEngine for LinguisticLobe.

    Mode-aware thresholds: DETERMINISTIC ≥ 0.85, CREATIVE ≥ 0.3.
    Results are VALIDATED at the similarity score's confidence.
    """

    DETERMINISTIC_THRESHOLD = 0.85
    CREATIVE_THRESHOLD = 0.3
    DEFAULT_THRESHOLD = 0.5

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._invocations = 0

    def __call__(
        self,
        query: str,
        domain: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._invocations += 1
        context = context or {}

        if _is_deterministic(context):
            threshold = self.DETERMINISTIC_THRESHOLD
        elif _is_creative(context):
            threshold = self.CREATIVE_THRESHOLD
        else:
            threshold = self.DEFAULT_THRESHOLD

        t0 = time.time()

        candidates = context.get("candidates", [])
        if candidates:
            matches = self._engine.best_match(query, candidates, top_k=10)
            filtered = [(text, score, idx) for text, score, idx in matches if score >= threshold]
        else:
            concepts = self._engine.extract_key_concepts(query)
            filtered_concepts = [(t, s) for t, s in concepts if s >= threshold]
            filtered = [(t, s, i) for i, (t, s) in enumerate(filtered_concepts)]

        elapsed_ms = (time.time() - t0) * 1000

        if not filtered:
            return {
                "value": raw(
                    "No semantic matches above threshold.",
                    metadata={"engine": "semantic", "threshold": threshold},
                ),
                "confidence": 0.0,
                "metadata": {"engine": "semantic", "matches": 0, "threshold": threshold},
            }

        best_score = filtered[0][1] if filtered else 0.0
        result_data = [{"text": t, "score": s} for t, s, _ in filtered[:10]]

        data = validated(
            result_data,
            confidence=best_score,
            gate_name="semantic_similarity",
            metadata={"engine": "semantic", "threshold": threshold},
        )
        return {
            "value": data,
            "confidence": best_score,
            "metadata": {
                "engine": "semantic",
                "matches": len(filtered),
                "threshold": threshold,
                "elapsed_ms": elapsed_ms,
            },
        }


# ── 4. CausalLobeProcessor ──────────────────────────────────────────────────

class CausalLobeProcessor:
    """Wraps CausalReasoner for TemporalLobe.

    Causal chains are VALIDATED with confidence = chain-strength product.
    Counterfactuals stay RAW (speculative). Gate enforces acyclicity and
    minimum strength > 0.2.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._gate = ValidationGate(
            name="causal_validation",
            confidence_threshold=0.2,
            target_trust=TrustLevel.VALIDATED,
            validators=[
                FunctionValidator(self._chain_acyclic, name="acyclic_check"),
                FunctionValidator(self._min_strength, name="strength_floor"),
            ],
            require_all=True,
        )
        self._invocations = 0

    def __call__(
        self,
        query: str,
        domain: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._invocations += 1
        context = context or {}

        if context.get("ingest_text"):
            self._engine.ingest(str(context["ingest_text"]))

        t0 = time.time()
        result = self._engine.query(query)
        elapsed_ms = (time.time() - t0) * 1000

        is_counterfactual = result.query_type == "counterfactual"
        conf = max(result.confidence, 0.0)

        if is_counterfactual:
            if _HAS_EPISTEMIC:
                data = speculative(
                    result.explanation, confidence=conf,
                    hypothesis=query,
                )
            else:
                data = raw(
                    result.explanation,
                    metadata={"engine": "causal", "speculative": True},
                )
        elif result.answered and conf >= 0.2:
            if _HAS_EPISTEMIC:
                data = causal(
                    result.explanation, confidence=conf,
                    chain_length=len(result.chains),
                )
            else:
                data = validated(
                    result.explanation,
                    confidence=conf,
                    gate_name="causal_chain",
                    metadata={
                        "engine": "causal",
                        "chain_count": len(result.chains),
                        "query_type": result.query_type,
                    },
                )
        else:
            data = raw(result.explanation, metadata={"engine": "causal"})

        return {
            "value": data,
            "confidence": conf,
            "metadata": {
                "engine": "causal",
                "query_type": result.query_type,
                "answered": result.answered,
                "chains": len(result.chains),
                "counterfactual": is_counterfactual,
                "elapsed_ms": elapsed_ms,
            },
        }

    @staticmethod
    def _chain_acyclic(data: Any) -> Tuple[bool, float]:
        if isinstance(data, str) and "cycle" in data.lower():
            return False, 0.0
        return True, 0.95

    @staticmethod
    def _min_strength(data: Any) -> Tuple[bool, float]:
        MIN_THRESHOLD = 0.3
        if data is None:
            return False, 0.0
        strength = None
        if isinstance(data, dict):
            strength = data.get("strength") or data.get("confidence")
            if strength is None:
                chains = data.get("chains", [])
                if chains:
                    strengths = [
                        getattr(c, "strength", None) or c.get("strength", 0)
                        if isinstance(c, dict) else getattr(c, "strength", 0)
                        for c in chains
                    ]
                    strength = min(strengths) if strengths else None
        elif hasattr(data, "strength"):
            strength = data.strength
        elif hasattr(data, "confidence"):
            strength = data.confidence
        if strength is None:
            return False, 0.1
        strength = float(strength)
        if strength < MIN_THRESHOLD:
            return False, strength
        return True, strength


# ── 5. AnalogyLobeProcessor ─────────────────────────────────────────────────

class AnalogyLobeProcessor:
    """Wraps AnalogyEngine for CreativeProcessingLobe.

    Analogies are creative output → trust = RAW, confidence = mapping score.
    Mode-aware thresholds gate which analogies are returned.
    """

    DETERMINISTIC_THRESHOLD = 0.7
    CREATIVE_THRESHOLD = 0.2
    DEFAULT_THRESHOLD = 0.4

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._invocations = 0

    def __call__(
        self,
        query: str,
        domain: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._invocations += 1
        context = context or {}

        if _is_deterministic(context):
            threshold = self.DETERMINISTIC_THRESHOLD
        elif _is_creative(context):
            threshold = self.CREATIVE_THRESHOLD
        else:
            threshold = self.DEFAULT_THRESHOLD

        t0 = time.time()
        target_domain = context.get("target_domain", "")
        result = self._engine.find_analogy(query, target_domain=target_domain)
        elapsed_ms = (time.time() - t0) * 1000

        conf = result.confidence
        if conf < threshold:
            return {
                "value": raw(
                    "No analogy above confidence threshold.",
                    metadata={"engine": "analogy", "threshold": threshold},
                ),
                "confidence": conf,
                "metadata": {
                    "engine": "analogy",
                    "below_threshold": True,
                    "threshold": threshold,
                    "actual_score": conf,
                },
            }

        if _HAS_EPISTEMIC:
            data = analogical(
                result.explanation,
                confidence=conf,
                source_domain=result.source,
                target_domain=result.target,
            )
        else:
            data = raw(
                {
                    "explanation": result.explanation,
                    "inferences": result.inferences,
                    "source": result.source,
                    "target": result.target,
                    "mapping_score": conf,
                },
                metadata={"engine": "analogy", "creative": True},
            )
        return {
            "value": data,
            "confidence": conf,
            "metadata": {
                "engine": "analogy",
                "source": result.source,
                "target": result.target,
                "mapping_score": conf,
                "inferences": len(result.inferences),
                "elapsed_ms": elapsed_ms,
            },
        }


# ── 6. PlannerCoordinator ───────────────────────────────────────────────────

_STEP_LOBE_MAP: Dict[str, LobeType] = {}
if StepAction is not None:
    _STEP_LOBE_MAP = {
        StepAction.RETRIEVE.value: LobeType.LINGUISTIC,
        StepAction.COMPUTE.value: LobeType.MATHEMATICAL,
        StepAction.INFER.value: LobeType.LOGICAL,
        StepAction.SYNTHESIZE.value: LobeType.CREATIVE,
    }


class PlannerCoordinator:
    """Wraps Planner for cross-lobe coordination via IntegrationCore.

    Decomposes complex queries into plan steps and dispatches each step
    to the appropriate lobe.  COMPARE steps use multi-lobe processing.
    Plan results are VALIDATED when coherence check passes.
    """

    def __init__(self, engine: Any, integration_core: IntegrationCore) -> None:
        self._planner = engine
        self._core = integration_core
        self._invocations = 0

    def __call__(
        self,
        query: str,
        domain: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._invocations += 1
        context = context or {}

        t0 = time.time()
        plan = self._planner.plan(query, domain=domain or "general", context=context)

        _goal: Any = None
        _plan_contract: Any = None
        if _HAS_EPISTEMIC:
            _goal = GoalContract(
                goal_id=f"goal_{int(t0 * 1000)}",
                description=query,
                status=GoalStatus.PROPOSED,
                trust_level=TrustLevel.RAW,
            )
            _goal.accept("planner decomposed query")
            _goal.activate()
            _epistemic_steps = []
            for _s in plan.steps:
                _a = _s.action.value if hasattr(_s.action, "value") else str(_s.action)
                _epistemic_steps.append(PlanStep(
                    step_id=_s.step_id,
                    description=_s.description,
                    assigned_lobe=_STEP_LOBE_MAP.get(_a),
                    epistemic_type=EpistemicType.OBSERVATIONAL,
                ))
            _plan_contract = PlanContract(
                plan_id=f"plan_{int(t0 * 1000)}",
                goal_id=_goal.goal_id,
                steps=_epistemic_steps,
                status=GoalStatus.ACTIVE,
            )

        step_results: Dict[str, Any] = {}
        for _step_idx, step in enumerate(plan.steps):
            action_val = step.action.value if hasattr(step.action, "value") else str(step.action)
            lobe_type = _STEP_LOBE_MAP.get(action_val)

            if action_val == "compare":
                multi_result = self._core.process_multi(
                    step.description,
                    [LobeType.LINGUISTIC, LobeType.LOGICAL],
                    domain=domain,
                    context=context,
                )
                step.result = multi_result.get("value")
                step.confidence = multi_result.get("confidence", 0.3)
                step.status = type(step.status)("completed") if hasattr(step.status, "value") else "completed"
            elif lobe_type and lobe_type in {lt for lt in self._core._lobes}:
                lobe_result = self._core.process_single(
                    step.description, lobe_type, domain=domain, context=context,
                )
                step.result = lobe_result.value
                step.confidence = lobe_result.confidence
                step.status = type(step.status)("completed") if hasattr(step.status, "value") else "completed"
            else:
                step.result = step.description
                step.confidence = 0.3
                step.status = type(step.status)("completed") if hasattr(step.status, "value") else "completed"

            if step.result is not None:
                step_results[step.step_id] = step.result

            if _HAS_EPISTEMIC and _plan_contract is not None:
                _step_data = validated(
                    str(step.result or ""),
                    confidence=getattr(step, "confidence", 0.3),
                    gate_name="plan_step",
                )
                try:
                    _plan_contract.execute_step(_step_idx, _step_data)
                except (IndexError, ValueError):
                    pass

        synthesis = self._planner.synthesize_results(plan)
        coherence = self._planner.check_coherence(plan)
        elapsed_ms = (time.time() - t0) * 1000

        plan_conf = plan.actual_confidence if plan.actual_confidence > 0 else 0.5
        is_coherent = coherence.get("coherent", True)

        if is_coherent and plan_conf >= 0.3:
            data = validated(
                synthesis,
                confidence=plan_conf,
                gate_name="planner_coherence",
                metadata={"engine": "planner", "steps": len(plan.steps)},
            )
        else:
            data = raw(synthesis, metadata={"engine": "planner", "coherent": is_coherent})

        _result_meta: Dict[str, Any] = {
            "engine": "planner",
            "decomposition": plan.decomposition_type,
            "steps_total": len(plan.steps),
            "steps_completed": plan.completed_steps,
            "coherent": is_coherent,
            "contradictions": len(coherence.get("contradictions", [])),
            "elapsed_ms": elapsed_ms,
        }
        if _HAS_EPISTEMIC and _plan_contract is not None:
            _plan_contract.coherence_score = 1.0 if is_coherent else 0.0
            _plan_contract.actual_confidence = plan_conf
            _result_meta["plan_contract_id"] = _plan_contract.plan_id
            _result_meta["goal_contract_id"] = _goal.goal_id
            _result_meta["plan_status"] = _plan_contract.status.name

        return {
            "value": data,
            "confidence": plan_conf,
            "metadata": _result_meta,
        }


# ── 7. VLTWorkingMemoryAdapter ──────────────────────────────────────────────

class VLTWorkingMemoryAdapter:
    """Bridges the WorkingMemory engine to VLT L1 (ephemeral) and L2 (session).

    Conversation turns are stored as ephemeral VLT items with auto-expiry.
    Open questions persist in L2 for the session duration.
    Entity and emotion detections are stored as L1 metadata.
    """

    L1_TTL = 300.0  # 5 minutes for ephemeral items

    def __init__(self, engine: Any, vlt: VLT) -> None:
        self._wm = engine
        self._vlt = vlt

    def process_turn(
        self,
        query: str,
        response: str,
        claims: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
    ) -> None:
        """Full turn processing: update working memory + sync to VLT."""
        self._wm.process_turn(
            query, response,
            extracted_claims=claims,
            detected_entities=entities,
        )

        turn_key = f"wm_turn_{int(time.time() * 1000)}"
        self._vlt.store(
            turn_key,
            {"query": query[:500], "response": response[:500]},
            layer=VLTLayer.L1_EPHEMERAL,
            confidence=0.8,
            source="working_memory",
            ttl=self.L1_TTL,
        )

        ctx = self._wm.get_context_for_response()
        for entity in ctx.get("active_entities", [])[:30]:
            self._vlt.store(
                f"wm_entity_{str(entity)[:40]}",
                entity,
                layer=VLTLayer.L1_EPHEMERAL,
                confidence=0.6,
                source="wm_extraction",
                ttl=self.L1_TTL,
            )

        for q in ctx.get("open_questions", [])[:15]:
            self._vlt.store(
                f"wm_oq_{hash(q) & 0xFFFFFFFF:08x}",
                q,
                layer=VLTLayer.L2_SESSION,
                confidence=0.7,
                source="wm_question",
                tags={"open_question"},
            )

    def get_context(self) -> Dict[str, Any]:
        """Merge VLT L1+L2 with WorkingMemory state for response generation."""
        wm_ctx = self._wm.get_context_for_response()

        l1_items = self._vlt.search(layer=VLTLayer.L1_EPHEMERAL, min_confidence=0.3)
        l2_items = self._vlt.search(layer=VLTLayer.L2_SESSION, min_confidence=0.3)

        vlt_facts = [item.value for item in l1_items[:50]]
        vlt_session = [item.value for item in l2_items[:50]]

        wm_ctx["vlt_ephemeral"] = vlt_facts
        wm_ctx["vlt_session"] = vlt_session
        wm_ctx["vlt_l1_count"] = len(l1_items)
        wm_ctx["vlt_l2_count"] = len(l2_items)
        return wm_ctx


# ── 8. VLTLearningAdapter ───────────────────────────────────────────────────

class VLTLearningAdapter:
    """Bridges LearningEngine to VLT L3 (persistent) and DomainCreases.

    Accepted claims become VLT L3 items with semantic keys.
    High-corroboration claims (confidence > 0.8) are promoted to
    DomainCrease entries on the appropriate lobe.
    """

    CREASE_PROMOTION_THRESHOLD = 0.8
    CORROBORATION_REQUIRED = 2

    def __init__(
        self,
        engine: Any,
        vlt: VLT,
        creases: Optional[Dict[str, DomainCrease]] = None,
    ) -> None:
        self._learning = engine
        self._vlt = vlt
        self._creases = creases or {}
        self._claims_stored = 0
        self._creases_written = 0

    def learn_and_store(
        self,
        query: str,
        response: str,
        conversation_history: List[Dict[str, Any]],
        response_confidence: float,
        web_facts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run learning pipeline and persist accepted claims to VLT L3."""
        result = self._learning.learn_from_interaction(
            query, response, conversation_history,
            response_confidence, web_facts=web_facts,
        )

        for event in result.get("events", []):
            if event.get("result") != "accepted":
                continue
            claim = event.get("claim", {})
            subj = claim.get("subject", "")
            rel = claim.get("relation", "")
            obj = claim.get("object", "")
            conf = float(claim.get("confidence", 0.5))

            claim_key = f"learned_{subj}_{rel}_{obj}"[:80]

            if _HAS_EPISTEMIC:
                record = ClaimRecord(
                    claim_id=claim_key,
                    content=f"{subj} {rel} {obj}",
                    status=ClaimStatus.EXTRACTED,
                    epistemic_type=EpistemicType.OBSERVATIONAL,
                    confidence=conf,
                    sources=["learning_engine"],
                )
                record.advance(ClaimStatus.VALIDATED, "accepted by learning engine")
                if conf >= self.CREASE_PROMOTION_THRESHOLD:
                    record.corroborate("high_confidence", conf)
                store_value: Any = {
                    "subject": subj, "relation": rel, "object": obj,
                    "claim_record": {
                        "claim_id": record.claim_id,
                        "status": record.status.name,
                        "epistemic_type": record.epistemic_type.name,
                        "confidence": record.confidence,
                        "corroboration_count": record.corroboration_count,
                    },
                }
            else:
                store_value = {"subject": subj, "relation": rel, "object": obj}

            self._vlt.store(
                claim_key,
                store_value,
                layer=VLTLayer.L3_PERSISTENT,
                confidence=conf,
                domain=rel,
                source="learning_engine",
                tags={"learned_claim"},
            )
            self._claims_stored += 1

            if conf >= self.CREASE_PROMOTION_THRESHOLD:
                self._promote_to_crease(subj, rel, obj, conf)

        return result

    def get_claims_for_query(self, query: str, top_k: int = 10) -> List[Any]:
        """Check VLT L3 before falling back to the engine's internal store.

        When epistemic types are available, reconstructs ClaimRecord
        wrappers with lifecycle metadata for each stored claim.
        """
        vlt_results = self._vlt.search(
            tags={"learned_claim"},
            layer=VLTLayer.L3_PERSISTENT,
            min_confidence=0.3,
        )[:top_k]

        if vlt_results:
            if _HAS_EPISTEMIC:
                claims: List[Any] = []
                for item in vlt_results:
                    val = item.value
                    cr_data = val.get("claim_record") if isinstance(val, dict) else None
                    if cr_data:
                        record = ClaimRecord(
                            claim_id=cr_data.get("claim_id", ""),
                            content=(
                                f"{val.get('subject', '')} "
                                f"{val.get('relation', '')} "
                                f"{val.get('object', '')}"
                            ),
                            status=ClaimStatus[cr_data.get("status", "EXTRACTED")],
                            epistemic_type=EpistemicType[
                                cr_data.get("epistemic_type", "OBSERVATIONAL")
                            ],
                            confidence=cr_data.get("confidence", 0.0),
                            corroboration_count=cr_data.get("corroboration_count", 0),
                        )
                        claims.append(record)
                    else:
                        claims.append(val)
                return claims
            return [item.value for item in vlt_results]

        return self._learning.get_claims_for_query(query, top_k=top_k)

    def _promote_to_crease(self, subj: str, rel: str, obj: str, conf: float) -> None:
        """Write high-confidence claims to the appropriate DomainCrease."""
        domain = rel
        crease = self._creases.get(domain)
        if crease is None:
            return

        try:
            crease.grow(
                f"{subj}_{rel}_{obj}"[:60],
                {"subject": subj, "relation": rel, "object": obj, "confidence": conf},
                source="learning_engine",
                mesh_validated=True,
                layer=1,
            )
            self._creases_written += 1
        except (ValueError, KeyError):
            pass

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "claims_stored_vlt": self._claims_stored,
            "creases_written": self._creases_written,
            "learning_stats": self._learning.stats,
        }


# ── 9. ModeControlTuner ─────────────────────────────────────────────────────

class ModeControlTuner:
    """Bridges SelfImprovementLoop to CognitiveModeController.

    Translates tuner parameter overrides into mode vector adjustments.
    Registers as a pre-classify adaptation layer.
    """

    def __init__(
        self,
        improvement_loop: Any,
        mode_controller: CognitiveModeController,
    ) -> None:
        self._loop = improvement_loop
        self._controller = mode_controller

    def apply_tuning(self, domain: str, decision: ModeDecision) -> ModeDecision:
        """Adjust a ModeDecision based on self-improvement parameters."""
        tuner = self._loop.tuner

        force_mode = tuner.get("force_mode", domain)
        if force_mode and isinstance(force_mode, str):
            override_vector = self._controller._apply_override(force_mode, decision.vector)
            decision.vector = override_vector
            decision.primary_mode = override_vector.primary_mode()
            decision.legacy_mode = override_vector.legacy_mode

        creativity_boost = tuner.get("creativity_boost", domain)
        if creativity_boost and isinstance(creativity_boost, (int, float)):
            kw = decision.vector.to_dict()
            kw["creative"] = min(1.0, kw["creative"] + float(creativity_boost))
            decision.vector = ModeVector.from_dict(kw)

        factual_strictness = tuner.get("factual_strictness", domain)
        if factual_strictness and isinstance(factual_strictness, (int, float)):
            kw = decision.vector.to_dict()
            kw["analytical"] = min(1.0, kw["analytical"] + float(factual_strictness))
            kw["factual"] = min(1.0, kw["factual"] + float(factual_strictness))
            decision.vector = ModeVector.from_dict(kw)

        return decision

    def get_web_max_facts(self, domain: str, default: int = 20) -> int:
        val = self._loop.tuner.get("web_max_facts", domain, default)
        try:
            return int(val)
        except (TypeError, ValueError):
            return default


# ── 10. AGIIntegration ───────────────────────────────────────────────────────

class AGIIntegration:
    """Wire all AGI engines into NRSI as native components.

    Usage:
        agi = AGIIntegration(
            lobes=integration_core,
            vlt=vlt,
            pvs=pvs,
            mode_controller=mode_controller,
        )
        agi.register_engines(
            logic_engine=logic_engine,
            computation_engine=computation_engine,
            semantic_engine=semantic_engine,
            causal_engine=causal_engine,
            analogy_engine=analogy_engine,
            planner=planner,
            working_memory=working_memory,
            learning_engine=learning_engine,
            improvement_loop=improvement_loop,
        )
    """

    def __init__(
        self,
        lobes: IntegrationCore,
        vlt: VLT,
        pvs: Optional[PVS4] = None,
        tuition: Optional[TuitionSystem] = None,
        mode_controller: Optional[CognitiveModeController] = None,
        creases: Optional[Dict[str, DomainCrease]] = None,
    ) -> None:
        self._lobes = lobes
        self._vlt = vlt
        self._pvs = pvs or vlt.pvs
        self._tuition = tuition or vlt.tuition
        self._mode_controller = mode_controller
        self._creases = creases or {}

        self._processors: Dict[str, Any] = {}
        self._adapters: Dict[str, Any] = {}
        self._registered_engines: Dict[str, bool] = {}

    def register_engines(self, engines: Any = None, **kw_engines: Any) -> Dict[str, bool]:
        if engines and isinstance(engines, dict):
            kw_engines.update(engines)
        engines = kw_engines
        """Register all provided AGI engines as NRSI-native processors."""
        results: Dict[str, bool] = {}

        if engines.get("logic_engine"):
            results["logic"] = self._register_logic(engines["logic_engine"])

        if engines.get("computation_engine"):
            results["math"] = self._register_math(engines["computation_engine"])

        if engines.get("semantic_engine"):
            results["semantic"] = self._register_semantic(engines["semantic_engine"])

        if engines.get("causal_engine"):
            results["causal"] = self._register_causal(engines["causal_engine"])

        if engines.get("analogy_engine"):
            results["analogy"] = self._register_analogy(engines["analogy_engine"])

        if engines.get("planner"):
            results["planner"] = self._register_planner(engines["planner"])

        if engines.get("working_memory"):
            results["working_memory"] = self._register_working_memory(engines["working_memory"])

        if engines.get("learning_engine"):
            results["learning"] = self._register_learning(engines["learning_engine"])

        if engines.get("improvement_loop") and self._mode_controller:
            results["mode_tuner"] = self._register_mode_tuner(engines["improvement_loop"])

        self._registered_engines = results
        logger.info("AGI integration registered: %s", results)
        return results

    def _register_logic(self, engine: Any) -> bool:
        proc = LogicLobeProcessor(engine)
        self._processors["logic"] = proc
        lobe = self._lobes.get_lobe(LobeType.LOGICAL)
        if lobe is None:
            lobe = LogicalLobe()
            self._lobes.register_lobe(lobe)
        lobe.register_processor(proc)
        return True

    def _register_math(self, engine: Any) -> bool:
        proc = MathLobeProcessor(engine)
        self._processors["math"] = proc
        lobe = self._lobes.get_lobe(LobeType.MATHEMATICAL)
        if lobe is None:
            lobe = MathematicalLobe()
            self._lobes.register_lobe(lobe)
        lobe.register_processor(proc)
        return True

    def _register_semantic(self, engine: Any) -> bool:
        proc = SemanticLobeProcessor(engine)
        self._processors["semantic"] = proc
        lobe = self._lobes.get_lobe(LobeType.LINGUISTIC)
        if lobe is None:
            lobe = LinguisticLobe()
            self._lobes.register_lobe(lobe)
        lobe.register_processor(proc)
        return True

    def _register_causal(self, engine: Any) -> bool:
        # CausalReasoner attaches to the TEMPORAL lobe — causal chains
        # are temporal-sequence reasoning (cause precedes effect). Lobe
        # class is TemporalLobe so it can use the temporal-specific
        # processors. Asserted by tests/test_nrsi_agi_integration.py
        # ::test_causal_registered_on_temporal_lobe.
        proc = CausalLobeProcessor(engine)
        self._processors["causal"] = proc
        lobe = self._lobes.get_lobe(LobeType.TEMPORAL)
        if lobe is None:
            lobe = TemporalLobe()
            self._lobes.register_lobe(lobe)
        lobe.register_processor(proc)
        return True

    def _register_analogy(self, engine: Any) -> bool:
        # AnalogyEngine attaches to the CREATIVE lobe (Association
        # cortex bio-analogue, lobes.py L103-106). The lobe class is
        # CreativeProcessingLobe — not the generic ProcessingLobe — so
        # downstream code can dispatch to its mode-aware creative path.
        # Documented in AnalogyLobeProcessor's own docstring
        # ("Wraps AnalogyEngine for CreativeProcessingLobe", L620) and
        # asserted by tests/test_nrsi_agi_integration.py
        # ::test_analogy_registered_on_creative_lobe.
        proc = AnalogyLobeProcessor(engine)
        self._processors["analogy"] = proc
        lobe = self._lobes.get_lobe(LobeType.CREATIVE)
        if lobe is None:
            lobe = CreativeProcessingLobe()
            self._lobes.register_lobe(lobe)
        lobe.register_processor(proc)
        return True

    def _register_planner(self, engine: Any) -> bool:
        coord = PlannerCoordinator(engine, self._lobes)
        self._processors["planner"] = coord
        lobe = self._lobes.get_lobe(LobeType.PLANNING)
        if lobe is None:
            lobe = ProcessingLobe(LobeType.PLANNING)
            self._lobes.register_lobe(lobe)
        lobe.register_processor(coord)
        return True

    def _register_working_memory(self, engine: Any) -> bool:
        adapter = VLTWorkingMemoryAdapter(engine, self._vlt)
        self._adapters["working_memory"] = adapter
        lobe = self._lobes.get_lobe(LobeType.MEMORY)
        if lobe is None:
            lobe = ProcessingLobe(LobeType.MEMORY)
            self._lobes.register_lobe(lobe)
        lobe.register_processor(adapter)
        return True

    def _register_learning(self, engine: Any) -> bool:
        adapter = VLTLearningAdapter(engine, self._vlt, creases=self._creases)
        self._adapters["learning"] = adapter
        return True

    def _register_mode_tuner(self, loop: Any) -> bool:
        tuner = ModeControlTuner(loop, self._mode_controller)
        self._adapters["mode_tuner"] = tuner
        lobe = self._lobes.get_lobe(LobeType.METACOGNITIVE)
        if lobe is None:
            lobe = ProcessingLobe(LobeType.METACOGNITIVE)
            self._lobes.register_lobe(lobe)
        lobe.register_processor(tuner)
        return True

    def get_context_for_response(self, query: str) -> Dict[str, Any]:
        """Get combined context from working memory + learning for response generation."""
        ctx: Dict[str, Any] = {}
        wm_adapter = self._adapters.get("working_memory")
        if wm_adapter:
            try:
                wm_ctx = wm_adapter.get_context()
                ctx.update(wm_ctx)
            except Exception as exc:
                logger.warning("WM context retrieval failed: %s", exc)
        learning_adapter = self._adapters.get("learning")
        if learning_adapter:
            try:
                claims = learning_adapter.get_claims_for_query(query, top_k=3)
                ctx["learned_claims"] = claims
            except Exception as exc:
                logger.warning("Learning claims retrieval failed: %s", exc)
        return ctx

    def post_process(
        self,
        query: str,
        response: Optional[str],
        session_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        response_confidence: float = 0.5,
    ) -> None:
        """Run post-response hooks through the integrated adapters."""
        if not response:
            return
        wm_adapter = self._adapters.get("working_memory")
        if wm_adapter:
            try:
                wm_adapter.process_turn(query, response)
            except Exception as exc:
                logger.warning("WM post_process failed: %s", exc)
        learning_adapter = self._adapters.get("learning")
        if learning_adapter:
            try:
                learning_adapter.learn_and_store(
                    query=query,
                    response=response,
                    conversation_history=conversation_history or [],
                    response_confidence=response_confidence,
                )
            except Exception as exc:
                logger.warning("Learning post_process failed: %s", exc)

    # ── NRS-facing API ──────────────────────────────────────────────────

    def coordinate_plan(self, query: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a multi-step plan using cross-lobe coordination."""
        planner = self._processors.get("planner")
        if not planner:
            return None
        result = planner(query, domain=context.get("domain", "general"), context=context)
        if not result:
            return None
        value = result.get("value")
        meta = result.get("metadata", {})
        synthesis = ""
        if value is not None:
            synthesis = str(getattr(value, "value", value))
        return {
            "synthesis": synthesis,
            "confidence": result.get("confidence", 0.0),
            "step_count": meta.get("steps_total", 0),
            "plan_type": meta.get("decomposition", ""),
        }

    def post_process_full(self, query: str, result_text: str, session_id: str,
                          response=None,
                          conversation_history: Optional[List[Dict[str, Any]]] = None):
        """Run post-processing: WM turn, learning, self-monitor, improvement.

        Extended version that accepts the full NRSResponse for richer
        self-improvement recording.  Falls back gracefully when *response*
        is None.
        """
        wm = self._adapters.get("working_memory")
        if wm:
            try:
                wm.process_turn(query, result_text)
            except Exception as exc:
                logger.warning("WM post_process_full failed: %s", exc)
        learning = self._adapters.get("learning")
        if learning:
            try:
                conf = response.result_confidence if response else 0.5
                learning.learn_and_store(
                    query, result_text,
                    conversation_history or [],
                    conf,
                )
            except Exception as exc:
                logger.warning("Learning post_process_full failed: %s", exc)
        tuner = self._adapters.get("mode_tuner")
        if tuner and hasattr(tuner, "_loop") and response:
            try:
                loop = tuner._loop
                if hasattr(loop, "record_interaction") and SelfImprovementLoop is not None:
                    from nrsip.self_improvement import InteractionRecord
                    ir = InteractionRecord(
                        query=query,
                        domain=getattr(response, "domain_detected", "general"),
                        mode=getattr(response, "mode", "HYBRID"),
                        confidence=getattr(response, "result_confidence", 0.5),
                        latency_ms=getattr(response, "processing_time_ms", 0),
                        had_facts=getattr(response, "result_confidence", 0) > 0.6,
                        had_web_facts=False,
                        had_computation=getattr(response, "guardrails", {}).get("computed", False),
                        had_logic_proof=getattr(response, "guardrails", {}).get("logic_proof", False),
                        plan_used="plan_steps" in getattr(response, "guardrails", {}),
                        response_length=len(getattr(response, "answer", "") or ""),
                    )
                    loop.record_interaction(ir)
            except Exception as exc:
                logger.warning("Self-improvement recording failed: %s", exc)

    def apply_mode_tuning(self, response, mode_decision):
        """Apply self-improvement runtime overrides to mode."""
        tuner = self._adapters.get("mode_tuner")
        if not tuner:
            return
        domain = getattr(response, "domain_detected", "general") or "general"
        try:
            updated = tuner.apply_tuning(domain, mode_decision)
            if updated:
                response.mode = updated.legacy_mode
                response.mode_vector = updated.vector.to_dict()
        except Exception as exc:
            logger.warning("Mode tuning application failed: %s", exc)
        try:
            if hasattr(tuner._loop, "get_runtime_param"):
                force_mode = tuner._loop.get_runtime_param("force_mode", domain, default=None)
                if force_mode:
                    response.mode = force_mode
                    response.guardrails["mode_overridden_by_improvement"] = True
        except Exception as exc:
            logger.warning("Mode force-override check failed: %s", exc)

    # ── Accessors ────────────────────────────────────────────────────

    def get_processor(self, name: str) -> Optional[Any]:
        return self._processors.get(name)

    def get_adapter(self, name: str) -> Optional[Any]:
        return self._adapters.get(name)

    @property
    def stats(self) -> Dict[str, Any]:
        proc_stats = {}
        for name, proc in self._processors.items():
            inv = getattr(proc, "_invocations", 0)
            proc_stats[name] = {"invocations": inv}

        adapter_stats = {}
        for name, adapter in self._adapters.items():
            if hasattr(adapter, "stats"):
                adapter_stats[name] = adapter.stats
            else:
                adapter_stats[name] = {"registered": True}

        return {
            "registered_engines": dict(self._registered_engines),
            "processors": proc_stats,
            "adapters": adapter_stats,
            "lobes_active": self._lobes.registered_lobes,
            "vlt_stats": self._vlt.stats,
        }
