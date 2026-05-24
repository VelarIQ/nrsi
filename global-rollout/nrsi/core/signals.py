"""
NRSI Signal System — Excitatory & Inhibitory

The brain has two fundamental signal types:

  EXCITATORY (glutamate) — "process this, amplify this, propagate this"
  INHIBITORY (GABA)      — "suppress this, block this, don't propagate"

This is not just a brain feature. It's a performance architecture.
When T1 neuron validation returns 0.95 confidence on "the sky is blue,"
inhibition kills T2-T4 instantly. You just saved 60-620ms of GPU time.

For a "this drug cures cancer" claim, no inhibition fires —
the full T0→T4 chain runs because nothing is confident enough to suppress.

Inhibition types (all biological analogs):

  CONFIDENCE  — Auto-inhibit when confidence exceeds threshold
                Brain analog: lateral inhibition (winner suppresses losers)

  RULE        — Explicit condition → suppress target layers
                Brain analog: feedforward inhibition (interneuron gating)

  FEEDBACK    — Higher layer sends inhibitory signal downward
                Brain analog: feedback inhibition (cortical top-down)

  LATERAL     — Same-level pathway suppresses weaker pathways
                Brain analog: lateral inhibition (contrast enhancement)

Every inhibition event is recorded. Full audit trail.
You can always ask "why was this layer skipped?" and get a
traceable answer back to the rule, condition, and data that triggered it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ── Signal Types ─────────────────────────────────────────────────────────────

class SignalType(Enum):
    """The two fundamental signal types in neuromorphic processing."""
    EXCITATORY = auto()   # Amplify, propagate, process
    INHIBITORY = auto()   # Suppress, block, skip


class InhibitionType(Enum):
    """How the inhibition was triggered."""
    CONFIDENCE = auto()   # Confidence threshold exceeded → skip higher tiers
    RULE       = auto()   # Explicit condition matched
    FEEDBACK   = auto()   # Higher layer sent inhibitory signal down
    LATERAL    = auto()   # Same-level pathway suppression


# ── Signal ───────────────────────────────────────────────────────────────────

@dataclass
class Signal:
    """
    A signal flowing through the neuromorphic hierarchy.

    Every piece of data in NRSI is wrapped in a Signal that carries:
    - The data itself
    - Signal type (excitatory or inhibitory)
    - Strength (0.0 to 1.0) — how strongly this signal should affect processing
    - Source — which layer/component produced this signal
    - Metadata — any additional context
    """

    data: Any
    signal_type: SignalType = SignalType.EXCITATORY
    strength: float = 1.0
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def is_excitatory(self) -> bool:
        return self.signal_type == SignalType.EXCITATORY

    @property
    def is_inhibitory(self) -> bool:
        return self.signal_type == SignalType.INHIBITORY

    def inhibit(self, reason: str, strength: float = 1.0, source: Optional[str] = None) -> Signal:
        """Create an inhibitory version of this signal."""
        return Signal(
            data=self.data,
            signal_type=SignalType.INHIBITORY,
            strength=strength,
            source=source or self.source,
            metadata={**self.metadata, "inhibition_reason": reason},
        )

    def excite(self, strength: float = 1.0, source: Optional[str] = None) -> Signal:
        """Create an excitatory version of this signal."""
        return Signal(
            data=self.data,
            signal_type=SignalType.EXCITATORY,
            strength=strength,
            source=source or self.source,
            metadata=dict(self.metadata),
        )

    def __repr__(self) -> str:
        t = "⊕" if self.is_excitatory else "⊖"
        return f"Signal({t} strength={self.strength:.2f}, source={self.source})"


# ── Inhibition Event ─────────────────────────────────────────────────────────

@dataclass
class InhibitionEvent:
    """
    Records a single inhibition event — a layer was suppressed.
    Full audit trail: what fired, why, what was suppressed, when.
    """

    rule_name: str
    inhibition_type: InhibitionType
    source_layer: str
    target_layers: List[str]
    reason: str
    trigger_data: Any
    trigger_confidence: Optional[float] = None
    strength: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        targets = ", ".join(self.target_layers)
        return (
            f"⊖ INHIBIT [{self.inhibition_type.name}] "
            f"{self.source_layer} → suppressed [{targets}]: {self.reason}"
        )


# ── Inhibition Rule ──────────────────────────────────────────────────────────

class InhibitionRule:
    """
    A rule that, when its condition is met, suppresses target layers.

    This is the core primitive. Everything else builds on this.

    Usage:
        # Skip T2-T4 when T1 is already confident
        rule = InhibitionRule(
            name="high_confidence_early_exit",
            condition=lambda data, ctx: data.get("confidence", 0) > 0.95,
            targets=["T2_slm", "T3_llm", "T4_elm"],
            inhibition_type=InhibitionType.CONFIDENCE,
            reason="Confidence {confidence:.4f} exceeds 0.95 — higher tiers unnecessary",
        )

        # Suppress parallel pathway when primary pathway is stronger
        rule = InhibitionRule(
            name="lateral_winner_takes_all",
            condition=lambda data, ctx: ctx.get("primary_confidence", 0) > ctx.get("secondary_confidence", 0),
            targets=["secondary_pathway"],
            inhibition_type=InhibitionType.LATERAL,
            reason="Primary pathway dominates with higher confidence",
        )
    """

    def __init__(
        self,
        name: str,
        condition: Callable[[Any, Dict[str, Any]], bool],
        targets: List[str],
        inhibition_type: InhibitionType = InhibitionType.RULE,
        reason: str = "",
        strength: float = 1.0,
    ):
        self.name = name
        self.condition = condition
        self.targets = targets
        self.inhibition_type = inhibition_type
        self.reason_template = reason
        self.strength = strength

        # Stats
        self._fires = 0
        self._checks = 0

    def evaluate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Optional[InhibitionEvent]:
        """
        Check if this rule fires against the given data.

        Returns InhibitionEvent if inhibition triggered, None otherwise.
        """
        ctx = context or {}
        self._checks += 1

        try:
            should_inhibit = self.condition(data, ctx)
        except Exception:
            # Rule evaluation failure = no inhibition (fail-open for safety)
            return None

        if not should_inhibit:
            return None

        self._fires += 1

        # Format reason with available data
        reason = self.reason_template
        try:
            if isinstance(data, dict):
                reason = reason.format(**data, **ctx)
            else:
                reason = reason.format(data=data, **ctx)
        except (KeyError, IndexError, ValueError):
            pass  # Use raw template if formatting fails

        # Extract confidence if data is a dict
        confidence = None
        if isinstance(data, dict):
            confidence = data.get("confidence")

        return InhibitionEvent(
            rule_name=self.name,
            inhibition_type=self.inhibition_type,
            source_layer=ctx.get("current_layer", "unknown"),
            target_layers=list(self.targets),
            reason=reason,
            trigger_data=data,
            trigger_confidence=confidence,
            strength=self.strength,
        )

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.inhibition_type.name,
            "checks": self._checks,
            "fires": self._fires,
            "fire_rate": self._fires / self._checks if self._checks > 0 else 0,
            "targets": self.targets,
        }

    def __repr__(self) -> str:
        return f"InhibitionRule(name='{self.name}', targets={self.targets}, fires={self._fires})"


# ── Inhibitory Network ───────────────────────────────────────────────────────

class InhibitoryNetwork:
    """
    Manages all inhibition rules and evaluates them during pipeline processing.

    The InhibitoryNetwork is attached to a HierarchicalPipeline.
    Before each layer executes, the network checks all rules against
    the current data. If any rule fires targeting the next layer,
    that layer is skipped.

    This is the mechanism that makes PRISM tier validation fast:
    - T1 neuron validation returns 0.95 confidence
    - InhibitoryNetwork has a CONFIDENCE rule: >0.95 → skip T2,T3,T4
    - Rule fires → T2,T3,T4 inhibited → pipeline skips to output
    - Total time: T0 + T1 = 3-7ms instead of 260-630ms

    Usage:
        network = InhibitoryNetwork()

        network.add_rule(InhibitionRule(
            name="tier_skip",
            condition=lambda d, c: d.get("confidence", 0) > 0.95,
            targets=["T2_slm", "T3_llm", "T4_elm"],
            inhibition_type=InhibitionType.CONFIDENCE,
        ))

        # During pipeline processing:
        inhibited = network.evaluate("T1_neuron", output_data, context)
        # inhibited = {"T2_slm", "T3_llm", "T4_elm"} if confidence > 0.95
    """

    def __init__(self):
        self._rules: List[InhibitionRule] = []
        self._events: List[InhibitionEvent] = []
        self._inhibited_layers: Dict[int, Set[str]] = {}  # cycle → inhibited layer names

    def add_rule(self, rule: InhibitionRule) -> InhibitoryNetwork:
        """Add an inhibition rule."""
        self._rules.append(rule)
        return self

    def add_rules(self, *rules: InhibitionRule) -> InhibitoryNetwork:
        """Add multiple inhibition rules."""
        for r in rules:
            self._rules.append(r)
        return self

    def evaluate(
        self,
        source_layer: str,
        data: Any,
        context: Optional[Dict[str, Any]] = None,
        cycle: int = 0,
    ) -> Set[str]:
        """
        Evaluate all rules after a layer produces output.

        Returns set of layer names that should be inhibited (skipped).
        """
        ctx = context or {}
        ctx["current_layer"] = source_layer
        newly_inhibited: Set[str] = set()

        for rule in self._rules:
            event = rule.evaluate(data, ctx)
            if event is not None:
                event.source_layer = source_layer
                self._events.append(event)
                newly_inhibited.update(event.target_layers)

        # Track per-cycle
        if cycle not in self._inhibited_layers:
            self._inhibited_layers[cycle] = set()
        self._inhibited_layers[cycle].update(newly_inhibited)

        return newly_inhibited

    def is_inhibited(self, layer_name: str, cycle: int = 0) -> bool:
        """Check if a layer is currently inhibited for a given cycle."""
        return layer_name in self._inhibited_layers.get(cycle, set())

    def reset_cycle(self, cycle: int) -> None:
        """Reset inhibition state for a new cycle."""
        self._inhibited_layers[cycle] = set()

    def reset(self) -> None:
        """Reset all inhibition state (but keep rules)."""
        self._events.clear()
        self._inhibited_layers.clear()

    @property
    def events(self) -> List[InhibitionEvent]:
        """All inhibition events recorded."""
        return list(self._events)

    @property
    def rules(self) -> List[InhibitionRule]:
        return list(self._rules)

    @property
    def stats(self) -> Dict[str, Any]:
        total_inhibitions = len(self._events)
        layers_saved = set()
        for e in self._events:
            layers_saved.update(e.target_layers)

        return {
            "total_rules": len(self._rules),
            "total_inhibition_events": total_inhibitions,
            "unique_layers_inhibited": list(layers_saved),
            "rules": [r.stats for r in self._rules],
        }

    def __repr__(self) -> str:
        return f"InhibitoryNetwork(rules={len(self._rules)}, events={len(self._events)})"


# ── Convenience: Common inhibition rules ─────────────────────────────────────

def confidence_inhibition(
    threshold: float,
    targets: List[str],
    name: Optional[str] = None,
    strength: float = 1.0,
) -> InhibitionRule:
    """
    Create a confidence-based inhibition rule.

    When output confidence exceeds threshold, suppress target layers.
    This is the PRISM tier-skip optimization.

    Usage:
        rule = confidence_inhibition(0.95, ["T2_slm", "T3_llm", "T4_elm"])
    """
    return InhibitionRule(
        name=name or f"confidence_>{threshold}",
        condition=lambda d, c: (
            d.get("confidence", 0) > threshold
            if isinstance(d, dict)
            else False
        ),
        targets=targets,
        inhibition_type=InhibitionType.CONFIDENCE,
        reason=f"Confidence exceeds {threshold} — higher tiers suppressed",
        strength=strength,
    )


def contradiction_inhibition(
    targets: List[str],
    name: Optional[str] = None,
) -> InhibitionRule:
    """
    Inhibit when contradictions are detected.

    If a layer detects contradictions in the data, suppress the
    target layers to prevent contradictory information from propagating.
    """
    return InhibitionRule(
        name=name or "contradiction_suppression",
        condition=lambda d, c: (
            d.get("contradictions", 0) > 0
            if isinstance(d, dict)
            else False
        ),
        targets=targets,
        inhibition_type=InhibitionType.RULE,
        reason="Contradictions detected — suppressing propagation",
        strength=1.0,
    )


def low_confidence_inhibition(
    threshold: float,
    targets: List[str],
    name: Optional[str] = None,
) -> InhibitionRule:
    """
    Inhibit when confidence is TOO LOW.

    If early processing shows the data is garbage, don't waste
    GPU cycles on higher tiers. Kill it early.
    """
    return InhibitionRule(
        name=name or f"low_confidence_<{threshold}",
        condition=lambda d, c: (
            d.get("confidence", 1.0) < threshold
            if isinstance(d, dict)
            else False
        ),
        targets=targets,
        inhibition_type=InhibitionType.RULE,
        reason=f"Confidence below {threshold} — data quality too low for higher processing",
        strength=1.0,
    )
