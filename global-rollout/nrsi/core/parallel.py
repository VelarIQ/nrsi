"""
NRSI Parallel Pathways

The brain doesn't process in a single stream. It runs thousands of
parallel pathways simultaneously:

  Visual cortex: edge detection, color, motion, depth — all at once
  Auditory cortex: pitch, timing, location — all at once
  Decision: multiple hypotheses compete simultaneously

The strongest pathway wins. Weaker pathways get laterally inhibited.
This is how the brain achieves both speed and robustness:
  - Speed: don't wait for serial processing
  - Robustness: if one pathway fails, others still produce results
  - Accuracy: multiple independent assessments → consensus

In PRISM terms:
  - Run multiple validation approaches concurrently on the same claim
  - First approach to reach high confidence can inhibit the rest
  - Or require consensus across N approaches before accepting

Merge strategies (how parallel results combine):

  WINNER     — Highest confidence wins, others discarded
               Brain analog: winner-take-all lateral inhibition
               PRISM use: fastest confident tier result accepted

  CONSENSUS  — All pathways must agree above threshold
               Brain analog: population coding convergence
               PRISM use: high-stakes claims need multiple validators

  WEIGHTED   — Combine results weighted by confidence
               Brain analog: Bayesian integration across sensory modalities
               PRISM use: ensemble validation for medium-confidence claims

  ALL        — Return all results, let caller decide
               Brain analog: no integration (raw parallel output)
               PRISM use: debugging, audit, comparison

Every parallel execution is tracked. You can always see which pathways
ran, which were inhibited, which won, and why.
"""

from __future__ import annotations

import time
import concurrent.futures
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import threading

try:
    import torch
    _TORCH_GPU = torch.cuda.is_available()
except ImportError:
    _TORCH_GPU = False
    torch = None  # type: ignore[assignment]


# ── Merge Strategy ───────────────────────────────────────────────────────────

class MergeStrategy(Enum):
    """How to combine results from parallel pathways."""
    WINNER    = auto()   # Highest confidence wins
    CONSENSUS = auto()   # All must agree
    WEIGHTED  = auto()   # Confidence-weighted combination
    ALL       = auto()   # Return everything


# ── Pathway Result ───────────────────────────────────────────────────────────

@dataclass
class PathwayResult:
    """Result from a single parallel pathway."""

    pathway_name: str
    output: Any
    confidence: float
    elapsed_ms: float
    success: bool
    error: Optional[str] = None
    inhibited: bool = False
    inhibited_by: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        if self.inhibited:
            return f"  ⊖ {self.pathway_name}: INHIBITED by {self.inhibited_by}"
        status = "✓" if self.success else "✗"
        return f"  {status} {self.pathway_name}: conf={self.confidence:.4f} ({self.elapsed_ms:.1f}ms)"


# ── Parallel Result ──────────────────────────────────────────────────────────

@dataclass
class ParallelResult:
    """Combined result from parallel pathway execution."""

    merged_output: Any
    merged_confidence: float
    strategy_used: MergeStrategy
    winner: Optional[str]
    pathway_results: List[PathwayResult]
    total_elapsed_ms: float
    backend: str = "cpu"

    @property
    def pathways_run(self) -> int:
        return sum(1 for r in self.pathway_results if not r.inhibited and r.success)

    @property
    def pathways_inhibited(self) -> int:
        return sum(1 for r in self.pathway_results if r.inhibited)

    @property
    def summary(self) -> str:
        lines = [
            f"Parallel [{self.strategy_used.name}] ({self.backend}): "
            f"{self.pathways_run} ran, {self.pathways_inhibited} inhibited, "
            f"{self.total_elapsed_ms:.1f}ms total",
        ]
        if self.winner:
            lines.append(f"  Winner: {self.winner} (conf={self.merged_confidence:.4f})")
        for r in self.pathway_results:
            lines.append(str(r))
        return "\n".join(lines)


# ── Pathway ──────────────────────────────────────────────────────────────────

class Pathway:
    """
    A single processing stream within a parallel layer.

    Each pathway has:
    - A name (e.g., "neuron_validation", "semantic_check", "knowledge_lookup")
    - A processor function: data → output dict with "confidence" key
    - An optional weight (for WEIGHTED merge)
    - Stats tracking

    The processor MUST return a dict with at least a "confidence" key.
    """

    def __init__(
        self,
        name: str,
        processor: Callable[[Any], Any],
        weight: float = 1.0,
        description: Optional[str] = None,
    ):
        self.name = name
        self.processor = processor
        self.weight = weight
        self.description = description

        # Stats
        self._runs = 0
        self._total_ms = 0.0
        self._failures = 0
        self._inhibitions = 0

    def execute(self, data: Any) -> PathwayResult:
        """Execute this pathway and return result."""
        start = time.time()
        try:
            output = self.processor(data)
            elapsed = (time.time() - start) * 1000
            self._runs += 1
            self._total_ms += elapsed

            # Extract confidence
            confidence = 0.0
            if isinstance(output, dict):
                confidence = output.get("confidence", 0.0)
            elif isinstance(output, (int, float)):
                confidence = float(output)

            return PathwayResult(
                pathway_name=self.name,
                output=output,
                confidence=confidence,
                elapsed_ms=elapsed,
                success=True,
            )
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            self._failures += 1
            self._total_ms += elapsed
            return PathwayResult(
                pathway_name=self.name,
                output=None,
                confidence=0.0,
                elapsed_ms=elapsed,
                success=False,
                error=str(e),
            )

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "runs": self._runs,
            "failures": self._failures,
            "inhibitions": self._inhibitions,
            "avg_ms": self._total_ms / self._runs if self._runs > 0 else 0.0,
        }

    def __repr__(self) -> str:
        return f"Pathway('{self.name}', weight={self.weight})"


# ── Parallel Layer ───────────────────────────────────────────────────────────

class ParallelLayer:
    """
    A processing layer that runs multiple pathways concurrently.

    This is the NRSI primitive for parallel processing. It replaces
    a single-processor Layer when you need multiple simultaneous
    validation approaches.

    Usage:
        parallel = ParallelLayer(
            name="multi_validation",
            pathways=[
                Pathway("neuron", neuron_check),
                Pathway("semantic", semantic_check),
                Pathway("knowledge", knowledge_check),
            ],
            merge=MergeStrategy.WINNER,
        )

        result = parallel.process(claim_data)
        # result.winner = "neuron" (if it had highest confidence)
        # result.merged_confidence = 0.97

    With lateral inhibition:
        parallel = ParallelLayer(
            name="multi_validation",
            pathways=[...],
            merge=MergeStrategy.WINNER,
            lateral_inhibition=True,
            inhibition_threshold=0.95,
        )
        # If "neuron" returns 0.97 first, "semantic" and "knowledge"
        # get laterally inhibited — don't even finish processing.

    Thread safety:
        Pathways execute in a ThreadPoolExecutor. Each pathway is
        independent. No shared mutable state between pathways.
    """

    def __init__(
        self,
        name: str,
        pathways: Optional[List[Pathway]] = None,
        merge: MergeStrategy = MergeStrategy.WINNER,
        consensus_threshold: float = 0.8,
        lateral_inhibition: bool = False,
        inhibition_threshold: float = 0.95,
        max_workers: Optional[int] = None,
        description: Optional[str] = None,
    ):
        self.name = name
        self._pathways: List[Pathway] = list(pathways or [])
        self.merge_strategy = merge
        self.consensus_threshold = consensus_threshold
        self.lateral_inhibition = lateral_inhibition
        self.inhibition_threshold = inhibition_threshold
        self.max_workers = max_workers
        self.description = description

        # Track lateral inhibition events
        self._inhibition_events: List[Dict[str, Any]] = []

    def add_pathway(self, pathway: Pathway) -> ParallelLayer:
        """Add a pathway."""
        self._pathways.append(pathway)
        return self

    def process(self, data: Any, *, force_cpu: bool = False) -> ParallelResult:
        """
        Execute all pathways concurrently and merge results.

        Uses CUDA streams when a torch-capable GPU is detected, falling
        back to ThreadPoolExecutor on CPU.  Set *force_cpu* to bypass
        GPU dispatch (useful for testing / deterministic audit runs).

        If lateral_inhibition is enabled and a pathway exceeds the
        inhibition_threshold, remaining pathways are cancelled/marked
        as inhibited.
        """
        start = time.time()

        if not self._pathways:
            return ParallelResult(
                merged_output=data,
                merged_confidence=0.0,
                strategy_used=self.merge_strategy,
                winner=None,
                pathway_results=[],
                total_elapsed_ms=0.0,
            )

        use_gpu = _TORCH_GPU and not force_cpu
        results: List[PathwayResult] = []

        if self.lateral_inhibition:
            if use_gpu:
                results = self._execute_gpu_with_inhibition(data)
            else:
                results = self._execute_with_inhibition(data)
        else:
            if use_gpu:
                results = self._execute_gpu_parallel(data)
            else:
                results = self._execute_parallel(data)

        merged_output, merged_confidence, winner = self._merge(results)

        total_ms = (time.time() - start) * 1000
        backend = "cuda" if use_gpu else "cpu"

        return ParallelResult(
            merged_output=merged_output,
            merged_confidence=merged_confidence,
            strategy_used=self.merge_strategy,
            winner=winner,
            pathway_results=results,
            total_elapsed_ms=total_ms,
            backend=backend,
        )

    def _execute_parallel(self, data: Any) -> List[PathwayResult]:
        """Execute all pathways concurrently without inhibition."""
        results: List[PathwayResult] = []
        workers = self.max_workers or len(self._pathways)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(p.execute, data): p
                for p in self._pathways
            }
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        return results

    def _execute_with_inhibition(self, data: Any) -> List[PathwayResult]:
        """
        Execute pathways with lateral inhibition.

        As soon as one pathway exceeds the inhibition threshold,
        remaining pathways are marked as inhibited. In a real
        GPU-backed system, this would cancel their execution.
        In the Python DSL, we use threading with early exit.
        """
        results: List[PathwayResult] = []
        inhibitor: Optional[str] = None
        workers = self.max_workers or len(self._pathways)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(p.execute, data): p
                for p in self._pathways
            }

            for future in concurrent.futures.as_completed(futures):
                pathway = futures[future]
                result = future.result()

                if inhibitor is not None:
                    # This pathway was laterally inhibited
                    result = PathwayResult(
                        pathway_name=pathway.name,
                        output=result.output,
                        confidence=result.confidence,
                        elapsed_ms=result.elapsed_ms,
                        success=result.success,
                        inhibited=True,
                        inhibited_by=inhibitor,
                    )
                    pathway._inhibitions += 1
                    results.append(result)
                    continue

                results.append(result)

                # Check if this result should laterally inhibit others
                if (result.success
                    and result.confidence >= self.inhibition_threshold):
                    inhibitor = pathway.name
                    self._inhibition_events.append({
                        "inhibitor": pathway.name,
                        "confidence": result.confidence,
                        "threshold": self.inhibition_threshold,
                        "timestamp": time.time(),
                    })

        return results

    # ── GPU (CUDA stream) execution paths ─────────────────────────────────

    def _execute_gpu_parallel(self, data: Any) -> List[PathwayResult]:
        """Execute all pathways on dedicated CUDA streams without inhibition."""
        streams = [torch.cuda.Stream() for _ in self._pathways]
        results: List[Optional[PathwayResult]] = [None] * len(self._pathways)

        def _run(idx: int, pathway: Pathway, stream: torch.cuda.Stream) -> None:
            with torch.cuda.stream(stream):
                results[idx] = pathway.execute(data)
            stream.synchronize()

        threads = [
            threading.Thread(target=_run, args=(i, p, s))
            for i, (p, s) in enumerate(zip(self._pathways, streams))
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        return [r for r in results if r is not None]

    def _execute_gpu_with_inhibition(self, data: Any) -> List[PathwayResult]:
        """Execute pathways on CUDA streams with lateral inhibition.

        Each pathway runs on its own stream + thread.  A shared
        ``threading.Event`` acts as the inhibition signal: when one
        pathway's confidence exceeds the threshold, the event is set
        and subsequently completing pathways are marked inhibited.
        """
        streams = [torch.cuda.Stream() for _ in self._pathways]
        results: List[Optional[PathwayResult]] = [None] * len(self._pathways)
        inhibit_signal = threading.Event()
        inhibitor_ref: List[Optional[str]] = [None]
        lock = threading.Lock()

        def _run(idx: int, pathway: Pathway, stream: torch.cuda.Stream) -> None:
            if inhibit_signal.is_set():
                pathway._inhibitions += 1
                results[idx] = PathwayResult(
                    pathway_name=pathway.name,
                    output=None,
                    confidence=0.0,
                    elapsed_ms=0.0,
                    success=False,
                    inhibited=True,
                    inhibited_by=inhibitor_ref[0],
                )
                return

            with torch.cuda.stream(stream):
                result = pathway.execute(data)
            stream.synchronize()

            with lock:
                if inhibit_signal.is_set():
                    pathway._inhibitions += 1
                    results[idx] = PathwayResult(
                        pathway_name=pathway.name,
                        output=result.output,
                        confidence=result.confidence,
                        elapsed_ms=result.elapsed_ms,
                        success=result.success,
                        inhibited=True,
                        inhibited_by=inhibitor_ref[0],
                    )
                    return

                results[idx] = result

                if (result.success
                        and result.confidence >= self.inhibition_threshold):
                    inhibitor_ref[0] = pathway.name
                    inhibit_signal.set()
                    self._inhibition_events.append({
                        "inhibitor": pathway.name,
                        "confidence": result.confidence,
                        "threshold": self.inhibition_threshold,
                        "timestamp": time.time(),
                    })

        threads = [
            threading.Thread(target=_run, args=(i, p, s))
            for i, (p, s) in enumerate(zip(self._pathways, streams))
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        return [r for r in results if r is not None]

    def _merge(
        self, results: List[PathwayResult]
    ) -> Tuple[Any, float, Optional[str]]:
        """Merge pathway results according to strategy."""

        # Filter to successful, non-inhibited results
        valid = [r for r in results if r.success and not r.inhibited]

        if not valid:
            return None, 0.0, None

        if self.merge_strategy == MergeStrategy.WINNER:
            return self._merge_winner(valid)
        elif self.merge_strategy == MergeStrategy.CONSENSUS:
            return self._merge_consensus(valid)
        elif self.merge_strategy == MergeStrategy.WEIGHTED:
            return self._merge_weighted(valid)
        elif self.merge_strategy == MergeStrategy.ALL:
            return self._merge_all(valid)
        else:
            return self._merge_winner(valid)

    def _merge_winner(
        self, results: List[PathwayResult]
    ) -> Tuple[Any, float, Optional[str]]:
        """Highest confidence wins."""
        best = max(results, key=lambda r: r.confidence)
        return best.output, best.confidence, best.pathway_name

    def _merge_consensus(
        self, results: List[PathwayResult]
    ) -> Tuple[Any, float, Optional[str]]:
        """All must agree above threshold."""
        all_above = all(r.confidence >= self.consensus_threshold for r in results)
        if all_above:
            # Consensus achieved — use highest confidence result
            best = max(results, key=lambda r: r.confidence)
            avg_conf = sum(r.confidence for r in results) / len(results)
            return best.output, avg_conf, f"consensus({len(results)})"
        else:
            # No consensus — return lowest confidence to signal uncertainty
            worst = min(results, key=lambda r: r.confidence)
            avg_conf = sum(r.confidence for r in results) / len(results)
            return worst.output, avg_conf, None

    def _merge_weighted(
        self, results: List[PathwayResult]
    ) -> Tuple[Any, float, Optional[str]]:
        """Confidence-weighted combination."""
        # Get pathway weights
        pathway_weights = {p.name: p.weight for p in self._pathways}

        total_weight = 0.0
        weighted_conf = 0.0
        best_result = None
        best_score = -1.0

        for r in results:
            w = pathway_weights.get(r.pathway_name, 1.0)
            score = r.confidence * w
            weighted_conf += score
            total_weight += w
            if score > best_score:
                best_score = score
                best_result = r

        avg_conf = weighted_conf / total_weight if total_weight > 0 else 0.0
        winner = best_result.pathway_name if best_result else None
        output = best_result.output if best_result else None
        return output, avg_conf, winner

    def _merge_all(
        self, results: List[PathwayResult]
    ) -> Tuple[Any, float, Optional[str]]:
        """Return all results."""
        outputs = {r.pathway_name: r.output for r in results}
        avg_conf = sum(r.confidence for r in results) / len(results)
        return outputs, avg_conf, "all"

    @property
    def pathways(self) -> List[Pathway]:
        return list(self._pathways)

    @property
    def inhibition_events(self) -> List[Dict[str, Any]]:
        return list(self._inhibition_events)

    @property
    def gpu_available(self) -> bool:
        return _TORCH_GPU

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "pathway_count": len(self._pathways),
            "merge_strategy": self.merge_strategy.name,
            "lateral_inhibition": self.lateral_inhibition,
            "inhibition_events": len(self._inhibition_events),
            "gpu_available": self.gpu_available,
            "pathways": [p.stats for p in self._pathways],
        }

    def __repr__(self) -> str:
        names = [p.name for p in self._pathways]
        return f"ParallelLayer('{self.name}', pathways={names}, merge={self.merge_strategy.name})"
