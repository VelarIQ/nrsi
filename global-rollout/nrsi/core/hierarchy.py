"""
NRSI Hierarchical Layers — Bidirectional

The neuromorphic processing hierarchy supports full bidirectional
data flow between layers. Like the brain:

  - Feedforward (upward): sensory → pattern → reasoning → validation
  - Feedback (downward): validation → reasoning → pattern → sensory
  - Lateral: layer-to-layer at the same level

This is how real neuromorphic processing works. Higher layers
refine lower layers. Validation feeds corrections back down.
Reasoning adjusts pattern recognition. The system converges
on truth through cycles, not a single pass.

The only rule: ALL data flow is tracked and auditable.
Trust metadata travels with the data in every direction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

from nrsi.core.types import NRSIData, TrustLevel, Confidence, raw
from nrsi.core.errors import LayerViolationError, NRSIError
from nrsi.core.signals import InhibitoryNetwork, InhibitionEvent, SignalType


T = TypeVar("T")


# ── Flow Direction ───────────────────────────────────────────────────────────

class FlowDirection(Enum):
    """Direction of data flow between layers."""
    FEEDFORWARD = auto()   # Lower → Higher (upward)
    FEEDBACK    = auto()   # Higher → Lower (downward)
    LATERAL     = auto()   # Same level


# ── Layer ────────────────────────────────────────────────────────────────────

class Layer:
    """
    A processing layer in the neuromorphic hierarchy.

    Each layer has:
    - A name and level (position in hierarchy)
    - A feedforward processor (handles data flowing up)
    - An optional feedback processor (handles data flowing down)
    - An optional lateral processor (handles data from same level)
    - Persistent state that survives across processing cycles
    """

    def __init__(
        self,
        name: str,
        level: int,
        processor: Optional[Callable] = None,
        feedback_processor: Optional[Callable] = None,
        lateral_processor: Optional[Callable] = None,
        description: Optional[str] = None,
    ):
        if level < 0:
            raise ValueError(f"Layer level must be non-negative, got {level}")

        self.name = name
        self.level = level
        self.description = description

        self._processor = processor
        self._feedback_processor = feedback_processor
        self._lateral_processor = lateral_processor

        # Internal state — persists across cycles
        self._state: Dict[str, Any] = {}

        # Stats
        self._feedforward_count = 0
        self._feedback_count = 0
        self._lateral_count = 0
        self._total_time_ms = 0.0

    def process(self, data: Any, context: Optional[Any] = None) -> Any:
        """Process data in the feedforward direction (upward)."""
        return self._execute(self._processor, data, context, "feedforward")

    def process_feedback(self, data: Any, context: Optional[Any] = None) -> Any:
        """Process feedback data from higher layers (downward)."""
        if self._feedback_processor:
            return self._execute(self._feedback_processor, data, context, "feedback")
        return data

    def process_lateral(self, data: Any, context: Optional[Any] = None) -> Any:
        """Process data from a same-level layer."""
        if self._lateral_processor:
            return self._execute(self._lateral_processor, data, context, "lateral")
        return data

    def _execute(
        self, processor: Optional[Callable], data: Any,
        context: Optional[Any], direction: str,
    ) -> Any:
        start = time.time()

        if processor:
            result = processor(data, context) if context is not None else processor(data)
        else:
            result = data

        elapsed = (time.time() - start) * 1000
        self._total_time_ms += elapsed

        if direction == "feedforward":
            self._feedforward_count += 1
        elif direction == "feedback":
            self._feedback_count += 1
        elif direction == "lateral":
            self._lateral_count += 1

        return result

    # ── State Management ─────────────────────────────────────────────────

    def set_state(self, key: str, value: Any) -> None:
        """Store state that persists across processing cycles."""
        self._state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Retrieve persisted state."""
        return self._state.get(key, default)

    def clear_state(self) -> None:
        self._state.clear()

    @property
    def state(self) -> Dict[str, Any]:
        return dict(self._state)

    # ── Processor Setters (for chaining) ─────────────────────────────────

    def set_processor(self, fn: Callable) -> Layer:
        self._processor = fn
        return self

    def set_feedback_processor(self, fn: Callable) -> Layer:
        self._feedback_processor = fn
        return self

    def set_lateral_processor(self, fn: Callable) -> Layer:
        self._lateral_processor = fn
        return self

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._feedforward_count + self._feedback_count + self._lateral_count
        return {
            "name": self.name,
            "level": self.level,
            "feedforward": self._feedforward_count,
            "feedback": self._feedback_count,
            "lateral": self._lateral_count,
            "total": total,
            "avg_time_ms": self._total_time_ms / total if total > 0 else 0.0,
        }

    def __repr__(self) -> str:
        return f"Layer(name='{self.name}', level={self.level})"

    def __lt__(self, other: Layer) -> bool:
        return self.level < other.level


# ── Flow Record ──────────────────────────────────────────────────────────────

@dataclass
class FlowRecord:
    """Records a single data movement between layers."""

    cycle: int
    direction: FlowDirection
    source_name: str
    source_level: int
    target_name: str
    target_level: int
    input_data: Any
    output_data: Any
    elapsed_ms: float
    success: bool
    error: Optional[str] = None
    inhibited: bool = False
    inhibition_reason: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        if self.inhibited:
            return (
                f"⊖ [INHIBITED  ] "
                f"{self.target_name}(L{self.target_level}) "
                f"— {self.inhibition_reason}"
            )
        arrows = {
            FlowDirection.FEEDFORWARD: "↑",
            FlowDirection.FEEDBACK: "↓",
            FlowDirection.LATERAL: "↔",
        }
        arrow = arrows[self.direction]
        status = "✓" if self.success else "✗"
        return (
            f"{status} [{self.direction.name:11}] "
            f"{self.source_name}(L{self.source_level}) {arrow} "
            f"{self.target_name}(L{self.target_level}) "
            f"({self.elapsed_ms:.1f}ms)"
        )


# ── Pipeline Result ──────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Result of processing through the bidirectional hierarchy."""

    success: bool
    final_output: Any
    flow_records: List[FlowRecord]
    total_cycles: int
    total_elapsed_ms: float
    converged: bool = False
    convergence_reason: Optional[str] = None

    @property
    def feedforward_count(self) -> int:
        return sum(1 for r in self.flow_records if r.direction == FlowDirection.FEEDFORWARD and not r.inhibited)

    @property
    def feedback_count(self) -> int:
        return sum(1 for r in self.flow_records if r.direction == FlowDirection.FEEDBACK and not r.inhibited)

    @property
    def inhibited_count(self) -> int:
        return sum(1 for r in self.flow_records if r.inhibited)

    @property
    def summary(self) -> str:
        status = "CONVERGED" if self.converged else ("COMPLETED" if self.success else "FAILED")
        lines = [
            f"Pipeline {status}: {self.total_cycles} cycle(s)",
            f"  Flows: {self.feedforward_count} feedforward ↑, {self.feedback_count} feedback ↓, {self.inhibited_count} inhibited ⊖",
            f"  Total time: {self.total_elapsed_ms:.1f}ms",
        ]
        if self.convergence_reason:
            lines.append(f"  Convergence: {self.convergence_reason}")

        # Show records grouped by cycle
        cycles_shown = set()
        for r in self.flow_records:
            if r.cycle not in cycles_shown:
                lines.append(f"  ── Cycle {r.cycle} ──")
                cycles_shown.add(r.cycle)
            lines.append(f"    {r}")
        return "\n".join(lines)


# ── Bidirectional Pipeline ───────────────────────────────────────────────────

class HierarchicalPipeline:
    """
    Bidirectional processing pipeline with inhibition.

    Three modes:

    1. SINGLE — Feedforward only, one pass up
       pipeline.process(data, mode="single")

    2. CYCLE — Feedforward up, then feedback down (one cycle)
       pipeline.process(data, mode="cycle")

    3. CONVERGE — Repeat cycles until output stabilizes
       pipeline.process(data, mode="converge", max_cycles=10)

    Inhibition:
    Attach an InhibitoryNetwork to the pipeline. Before each layer
    executes, the network checks all rules. If a rule fires targeting
    the next layer, that layer is skipped. This is how PRISM T0-T4
    tier validation becomes fast — high confidence at T1 means T2-T4
    are inhibited and never execute.
    """

    def __init__(
        self,
        layers: Optional[List[Layer]] = None,
        name: str = "pipeline",
        inhibition: Optional[InhibitoryNetwork] = None,
    ):
        self.name = name
        self._layers: List[Layer] = []
        self._layers_by_name: Dict[str, Layer] = {}
        self._inhibition = inhibition or InhibitoryNetwork()

        if layers:
            for l in sorted(layers, key=lambda x: x.level):
                self.add_layer(l)

    def add_layer(self, layer_obj: Layer) -> HierarchicalPipeline:
        """Add a layer. Layers are sorted by level automatically."""
        if layer_obj.name in self._layers_by_name:
            raise NRSIError(
                f"Layer '{layer_obj.name}' already exists",
                suggestion="Use a unique name for each layer",
            )
        self._layers.append(layer_obj)
        self._layers.sort(key=lambda l: l.level)
        self._layers_by_name[layer_obj.name] = layer_obj
        return self

    def get_layer(self, name: str) -> Layer:
        if name not in self._layers_by_name:
            raise NRSIError(
                f"Layer '{name}' not found",
                suggestion=f"Available: {', '.join(self._layers_by_name.keys())}",
            )
        return self._layers_by_name[name]

    @property
    def inhibition(self) -> InhibitoryNetwork:
        """Access the inhibitory network."""
        return self._inhibition

    def set_inhibition(self, network: InhibitoryNetwork) -> HierarchicalPipeline:
        """Set the inhibitory network."""
        self._inhibition = network
        return self

    # ── Processing ───────────────────────────────────────────────────────

    def process(
        self,
        data: Any,
        context: Optional[Any] = None,
        mode: str = "single",
        max_cycles: int = 5,
        convergence_fn: Optional[Callable[[Any, Any], bool]] = None,
    ) -> PipelineResult:
        if not self._layers:
            raise NRSIError("Pipeline has no layers")

        if mode == "single":
            return self._single_pass(data, context)
        elif mode == "cycle":
            return self._full_cycle(data, context)
        elif mode == "converge":
            return self._convergent(data, context, max_cycles, convergence_fn)
        else:
            raise NRSIError(f"Unknown mode: '{mode}'", suggestion="Use 'single', 'cycle', or 'converge'")

    def _single_pass(self, data: Any, context: Optional[Any]) -> PipelineResult:
        """Feedforward only, with inhibition."""
        start = time.time()
        records: List[FlowRecord] = []
        current = data
        inhibited_set: Set[str] = set()

        self._inhibition.reset()

        for i, lyr in enumerate(self._layers):
            # Check inhibition
            if lyr.name in inhibited_set:
                records.append(FlowRecord(
                    cycle=0, direction=FlowDirection.FEEDFORWARD,
                    source_name=self._layers[i - 1].name if i > 0 else "input",
                    source_level=self._layers[i - 1].level if i > 0 else -1,
                    target_name=lyr.name, target_level=lyr.level,
                    input_data=current, output_data=current,
                    elapsed_ms=0.0, success=True,
                    inhibited=True, inhibition_reason=self._get_inhibition_reason(lyr.name),
                ))
                continue  # Skip this layer

            record = self._exec(lyr, current, context, FlowDirection.FEEDFORWARD,
                                self._layers[i - 1] if i > 0 else None, cycle=0)
            records.append(record)
            if not record.success:
                return PipelineResult(False, None, records, 0, (time.time() - start) * 1000)
            current = record.output_data

            # Evaluate inhibition after this layer
            newly_inhibited = self._inhibition.evaluate(lyr.name, current, cycle=0)
            inhibited_set.update(newly_inhibited)

        return PipelineResult(True, current, records, 1, (time.time() - start) * 1000)

    def _full_cycle(self, data: Any, context: Optional[Any]) -> PipelineResult:
        """One feedforward up, one feedback down, with inhibition."""
        start = time.time()
        records: List[FlowRecord] = []
        current = data
        inhibited_set: Set[str] = set()

        self._inhibition.reset()

        # Feedforward: bottom → top
        for i, lyr in enumerate(self._layers):
            if lyr.name in inhibited_set:
                records.append(self._inhibited_record(
                    lyr, self._layers[i - 1] if i > 0 else None,
                    current, FlowDirection.FEEDFORWARD, 0,
                ))
                continue

            record = self._exec(lyr, current, context, FlowDirection.FEEDFORWARD,
                                self._layers[i - 1] if i > 0 else None, cycle=0)
            records.append(record)
            if not record.success:
                return PipelineResult(False, None, records, 0, (time.time() - start) * 1000)
            current = record.output_data

            newly_inhibited = self._inhibition.evaluate(lyr.name, current, cycle=0)
            inhibited_set.update(newly_inhibited)

        # Feedback: top → bottom (inhibition resets for feedback pass)
        fb_inhibited: Set[str] = set()
        for i in range(len(self._layers) - 1, -1, -1):
            lyr = self._layers[i]
            src = self._layers[i + 1] if i < len(self._layers) - 1 else None

            if lyr.name in fb_inhibited:
                records.append(self._inhibited_record(
                    lyr, src, current, FlowDirection.FEEDBACK, 0,
                ))
                continue

            record = self._exec(lyr, current, context, FlowDirection.FEEDBACK, src, cycle=0)
            records.append(record)
            if not record.success:
                return PipelineResult(False, None, records, 1, (time.time() - start) * 1000)
            current = record.output_data

        return PipelineResult(True, current, records, 1, (time.time() - start) * 1000)

    def _convergent(
        self, data: Any, context: Optional[Any],
        max_cycles: int, convergence_fn: Optional[Callable],
    ) -> PipelineResult:
        """Repeat feedforward+feedback until output stabilizes, with inhibition."""
        start = time.time()
        all_records: List[FlowRecord] = []
        current = data
        previous = None
        converged = False
        reason = None
        cycle = 0

        self._inhibition.reset()

        for cycle in range(max_cycles):
            inhibited_set: Set[str] = set()
            self._inhibition.reset_cycle(cycle)

            # Feedforward
            ff = current
            for i, lyr in enumerate(self._layers):
                if lyr.name in inhibited_set:
                    all_records.append(self._inhibited_record(
                        lyr, self._layers[i - 1] if i > 0 else None,
                        ff, FlowDirection.FEEDFORWARD, cycle,
                    ))
                    continue

                rec = self._exec(lyr, ff, context, FlowDirection.FEEDFORWARD,
                                 self._layers[i - 1] if i > 0 else None, cycle=cycle)
                all_records.append(rec)
                if not rec.success:
                    return PipelineResult(False, None, all_records, cycle + 1, (time.time() - start) * 1000)
                ff = rec.output_data

                newly_inhibited = self._inhibition.evaluate(lyr.name, ff, cycle=cycle)
                inhibited_set.update(newly_inhibited)

            # Feedback
            fb = ff
            for i in range(len(self._layers) - 1, -1, -1):
                lyr = self._layers[i]
                src = self._layers[i + 1] if i < len(self._layers) - 1 else None
                rec = self._exec(lyr, fb, context, FlowDirection.FEEDBACK, src, cycle=cycle)
                all_records.append(rec)
                if not rec.success:
                    return PipelineResult(False, None, all_records, cycle + 1, (time.time() - start) * 1000)
                fb = rec.output_data

            current = fb

            # Check convergence
            if previous is not None:
                if convergence_fn:
                    if convergence_fn(previous, current):
                        converged = True
                        reason = f"Converged at cycle {cycle + 1}"
                        break
                else:
                    if self._equal(previous, current):
                        converged = True
                        reason = f"Output stabilized at cycle {cycle + 1}"
                        break

            previous = current

        if not converged:
            reason = f"Max cycles ({max_cycles}) reached"

        return PipelineResult(
            success=True, final_output=current, flow_records=all_records,
            total_cycles=cycle + 1, total_elapsed_ms=(time.time() - start) * 1000,
            converged=converged, convergence_reason=reason,
        )

    # ── Execution Helper ─────────────────────────────────────────────────

    def _inhibited_record(
        self, lyr: Layer, source: Optional[Layer],
        data: Any, direction: FlowDirection, cycle: int,
    ) -> FlowRecord:
        """Create a flow record for an inhibited (skipped) layer."""
        reason = self._get_inhibition_reason(lyr.name)
        return FlowRecord(
            cycle=cycle, direction=direction,
            source_name=source.name if source else "input",
            source_level=source.level if source else -1,
            target_name=lyr.name, target_level=lyr.level,
            input_data=data, output_data=data,
            elapsed_ms=0.0, success=True,
            inhibited=True, inhibition_reason=reason,
        )

    def _get_inhibition_reason(self, layer_name: str) -> str:
        """Get the reason a layer was inhibited from the most recent event."""
        for event in reversed(self._inhibition.events):
            if layer_name in event.target_layers:
                return f"{event.rule_name}: {event.reason}"
        return "inhibited"

    def _exec(
        self, lyr: Layer, data: Any, context: Optional[Any],
        direction: FlowDirection, source: Optional[Layer], cycle: int,
    ) -> FlowRecord:
        start = time.time()
        try:
            if direction == FlowDirection.FEEDFORWARD:
                output = lyr.process(data, context)
            elif direction == FlowDirection.FEEDBACK:
                output = lyr.process_feedback(data, context)
            else:
                output = lyr.process_lateral(data, context)

            return FlowRecord(
                cycle=cycle, direction=direction,
                source_name=source.name if source else "input",
                source_level=source.level if source else -1,
                target_name=lyr.name, target_level=lyr.level,
                input_data=data, output_data=output,
                elapsed_ms=(time.time() - start) * 1000, success=True,
            )
        except Exception as e:
            return FlowRecord(
                cycle=cycle, direction=direction,
                source_name=source.name if source else "input",
                source_level=source.level if source else -1,
                target_name=lyr.name, target_level=lyr.level,
                input_data=data, output_data=None,
                elapsed_ms=(time.time() - start) * 1000, success=False, error=str(e),
            )

    @staticmethod
    def _equal(a: Any, b: Any) -> bool:
        try:
            return a == b
        except Exception:
            return str(a) == str(b)

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def layers(self) -> List[Layer]:
        return list(self._layers)

    @property
    def depth(self) -> int:
        return len(self._layers)

    def __repr__(self) -> str:
        layer_names = " ⇄ ".join(f"{l.name}(L{l.level})" for l in self._layers)
        return f"HierarchicalPipeline({layer_names})"


# ── Convenience Decorator ────────────────────────────────────────────────────

def layer(
    name: str,
    level: int,
    description: Optional[str] = None,
    feedback: Optional[Callable] = None,
    lateral: Optional[Callable] = None,
):
    """
    Decorator to create a Layer from a function.

    Usage:
        @layer("sensory", level=0)
        def process_input(data):
            return clean(data)

        @layer("reasoning", level=2, feedback=refine_from_above)
        def analyze(data):
            return reason_about(data)
    """
    def decorator(fn: Callable) -> Layer:
        l = Layer(
            name=name, level=level, processor=fn,
            feedback_processor=feedback,
            lateral_processor=lateral,
            description=description,
        )
        l.__name__ = fn.__name__
        l.__doc__ = fn.__doc__
        return l
    return decorator
