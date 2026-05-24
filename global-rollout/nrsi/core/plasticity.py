"""
NRSI Plasticity — Adaptive Behavior Without Retraining

The brain rewires itself in real-time:

  HEBBIAN — "neurons that fire together wire together"
            Successful pathways strengthen, unused ones weaken.

  HOMEOSTATIC — maintain baseline activity levels.
                If a layer is over-firing, its threshold rises.
                If under-firing, threshold drops.

  STRUCTURAL — create new connections, prune dead ones.
               Successful processing routes get reinforced.

In NRSI, plasticity means:
  - Confidence thresholds adapt: a gate that consistently sees
    high-confidence data raises its bar. One that starves lowers it.
  - Pathway weights shift: a parallel pathway that keeps winning
    gets higher weight. One that keeps losing gets downweighted.
  - Inhibition thresholds tune: if inhibition fires too aggressively
    (skipping tiers that later proved necessary), the threshold rises.
  - Layer routing changes: frequently used processing paths
    get priority. Rarely used paths get pruned.

This is NOT retraining. No gradient descent. No backpropagation.
It's online adaptation — the system tunes itself based on outcomes
while running. Every adaptation is logged and reversible.

For AGI:
  Plasticity is how AGI learns from experience without stopping.
  Every query processed makes the system slightly better at
  processing similar queries in the future.
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import cupy as _cp
    _GPU_AVAILABLE = True
except ImportError:
    _GPU_AVAILABLE = False
    _cp = None

try:
    import torch as _torch
    _TORCH_GPU = _torch.cuda.is_available()
except ImportError:
    _torch = None
    _TORCH_GPU = False


def _xp(use_gpu: bool = True):
    if use_gpu and _GPU_AVAILABLE:
        return _cp
    return np


# ── Plasticity Type ──────────────────────────────────────────────────────────

class PlasticityType(Enum):
    """How the adaptation occurred."""
    HEBBIAN      = auto()   # Success strengthens, failure weakens
    HOMEOSTATIC  = auto()   # Maintain baseline activity
    STRUCTURAL   = auto()   # Create/prune connections


# ── Adaptation Event ─────────────────────────────────────────────────────────

@dataclass
class AdaptationEvent:
    """Records a single plasticity event — something changed."""

    plasticity_type: PlasticityType
    target: str
    parameter: str
    old_value: float
    new_value: float
    reason: str
    timestamp: float = field(default_factory=time.time)

    @property
    def delta(self) -> float:
        return self.new_value - self.old_value

    def __str__(self) -> str:
        direction = "↑" if self.delta > 0 else "↓"
        return (
            f"  ⚡ [{self.plasticity_type.name:12}] {self.target}.{self.parameter}: "
            f"{self.old_value:.4f} → {self.new_value:.4f} ({direction}{abs(self.delta):.4f}) "
            f"— {self.reason}"
        )


# ── Adaptive Threshold ───────────────────────────────────────────────────────

class AdaptiveThreshold:
    """
    A threshold that adapts based on outcomes.

    Tracks a running window of results and adjusts the threshold
    to maintain a target pass rate.

    Usage:
        threshold = AdaptiveThreshold(
            name="t1_confidence",
            initial=0.85,
            target_pass_rate=0.80,  # Want 80% of items to pass
            learning_rate=0.01,
        )

        # Record outcomes
        threshold.record(passed=True, value=0.92)
        threshold.record(passed=True, value=0.88)
        threshold.record(passed=False, value=0.70)

        # Threshold auto-adjusts to achieve target pass rate
        current = threshold.value  # May have shifted from 0.85
    """

    def __init__(
        self,
        name: str,
        initial: float = 0.5,
        target_pass_rate: float = 0.8,
        learning_rate: float = 0.01,
        min_value: float = 0.0,
        max_value: float = 1.0,
        window_size: int = 100,
    ):
        self.name = name
        self._value = initial
        self._initial = initial
        self.target_pass_rate = target_pass_rate
        self.learning_rate = learning_rate
        self.min_value = min_value
        self.max_value = max_value
        self.window_size = window_size

        self._history: List[bool] = []  # Pass/fail history
        self._values: List[float] = []  # Input values
        self._adaptations: List[AdaptationEvent] = []

    @property
    def value(self) -> float:
        return self._value

    @property
    def current_pass_rate(self) -> float:
        if not self._history:
            return 0.0
        window = self._history[-self.window_size:]
        return sum(window) / len(window)

    def record(self, passed: bool, value: float) -> Optional[AdaptationEvent]:
        """
        Record an outcome and potentially adapt.

        Returns AdaptationEvent if threshold changed.
        """
        self._history.append(passed)
        self._values.append(value)

        # Only adapt after enough samples
        if len(self._history) < 10:
            return None

        current_rate = self.current_pass_rate
        rate_error = current_rate - self.target_pass_rate

        # If pass rate is too high → raise threshold (be stricter)
        # If pass rate is too low → lower threshold (be more lenient)
        if abs(rate_error) > 0.05:  # 5% deadband
            old = self._value
            adjustment = rate_error * self.learning_rate
            self._value = max(
                self.min_value,
                min(self.max_value, self._value + adjustment),
            )

            if abs(self._value - old) > 0.0001:
                event = AdaptationEvent(
                    plasticity_type=PlasticityType.HOMEOSTATIC,
                    target=self.name,
                    parameter="threshold",
                    old_value=old,
                    new_value=self._value,
                    reason=f"pass_rate={current_rate:.2%}, target={self.target_pass_rate:.2%}",
                )
                self._adaptations.append(event)
                return event

        return None

    def check(self, value: float) -> bool:
        """Check if a value passes the current threshold."""
        return value >= self._value

    def reset(self) -> None:
        """Reset to initial value."""
        self._value = self._initial
        self._history.clear()
        self._values.clear()

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "current_value": self._value,
            "initial_value": self._initial,
            "drift": self._value - self._initial,
            "current_pass_rate": self.current_pass_rate,
            "target_pass_rate": self.target_pass_rate,
            "samples": len(self._history),
            "adaptations": len(self._adaptations),
        }

    def __repr__(self) -> str:
        return f"AdaptiveThreshold('{self.name}', value={self._value:.4f}, drift={self._value - self._initial:+.4f})"


# ── Adaptive Weight ──────────────────────────────────────────────────────────

class AdaptiveWeight:
    """
    A weight that strengthens or weakens based on success/failure.

    Hebbian learning: success → strengthen, failure → weaken.
    Stores the underlying value as a torch scalar tensor on GPU
    when CUDA is available, falling back to a plain float.

    Usage:
        weight = AdaptiveWeight("neuron_pathway", initial=1.0)
        weight.reinforce(success=True)   # Weight increases
        weight.reinforce(success=True)   # Weight increases more
        weight.reinforce(success=False)  # Weight decreases
    """

    def __init__(
        self,
        name: str,
        initial: float = 1.0,
        learning_rate: float = 0.05,
        min_weight: float = 0.1,
        max_weight: float = 5.0,
        decay_rate: float = 0.001,
        use_gpu: bool = True,
    ):
        self.name = name
        self._initial = initial
        self.learning_rate = learning_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.decay_rate = decay_rate
        self._use_gpu = use_gpu and _TORCH_GPU

        if self._use_gpu:
            self._t_value = _torch.tensor(
                initial, dtype=_torch.float32, device="cuda",
            )
        else:
            self._t_value = None
            self._value = initial

        self._successes = 0
        self._failures = 0
        self._adaptations: List[AdaptationEvent] = []

    @property
    def value(self) -> float:
        if self._use_gpu:
            return self._t_value.item()
        return self._value

    @value.setter
    def value(self, v: float) -> None:
        if self._use_gpu:
            self._t_value.fill_(v)
        else:
            self._value = v

    def reinforce(self, success: bool, magnitude: float = 1.0) -> AdaptationEvent:
        """
        Hebbian reinforcement.

        success=True → strengthen (weight increases)
        success=False → weaken (weight decreases)
        magnitude → how strongly to adjust (0.0-1.0)
        """
        old = self.value

        if success:
            self._successes += 1
            delta = self.learning_rate * magnitude
            self.value = min(self.max_weight, old + delta)
        else:
            self._failures += 1
            delta = self.learning_rate * magnitude
            self.value = max(self.min_weight, old - delta)

        event = AdaptationEvent(
            plasticity_type=PlasticityType.HEBBIAN,
            target=self.name,
            parameter="weight",
            old_value=old,
            new_value=self.value,
            reason=f"{'success' if success else 'failure'} (magnitude={magnitude:.2f})",
        )
        self._adaptations.append(event)
        return event

    def apply_decay(self) -> Optional[AdaptationEvent]:
        """
        Natural decay — weights drift toward initial value over time.
        Like synaptic decay without stimulation.
        """
        cur = self.value
        if abs(cur - self._initial) < 0.001:
            return None

        old = cur
        direction = -1.0 if cur > self._initial else 1.0
        cur += direction * self.decay_rate

        if direction > 0 and cur > self._initial:
            cur = self._initial
        elif direction < 0 and cur < self._initial:
            cur = self._initial

        if abs(cur - old) > 0.0001:
            self.value = cur
            event = AdaptationEvent(
                plasticity_type=PlasticityType.HEBBIAN,
                target=self.name,
                parameter="weight",
                old_value=old,
                new_value=cur,
                reason="natural decay toward baseline",
            )
            self._adaptations.append(event)
            return event
        return None

    @property
    def success_rate(self) -> float:
        total = self._successes + self._failures
        return self._successes / total if total > 0 else 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        v = self.value
        return {
            "name": self.name,
            "current_weight": v,
            "initial_weight": self._initial,
            "drift": v - self._initial,
            "successes": self._successes,
            "failures": self._failures,
            "success_rate": self.success_rate,
            "adaptations": len(self._adaptations),
            "gpu_backed": self._use_gpu,
        }

    def __repr__(self) -> str:
        v = self.value
        return f"AdaptiveWeight('{self.name}', value={v:.4f}, drift={v - self._initial:+.4f})"


# ── Plasticity Manager ───────────────────────────────────────────────────────

class PlasticityManager:
    """
    Manages all adaptive components in the system.

    Tracks thresholds, weights, and routing decisions.
    Provides a single interface for recording outcomes and
    querying current adaptive state.

    When GPU is available, batch operations use CuPy/torch
    vectorized paths for simultaneous weight updates.

    Usage:
        plasticity = PlasticityManager()

        # Register adaptive components
        plasticity.add_threshold("t1_gate", initial=0.85)
        plasticity.add_weight("neuron_pathway", initial=1.0)

        # Record outcomes
        plasticity.record_outcome("t1_gate", passed=True, value=0.92)
        plasticity.reinforce_weight("neuron_pathway", success=True)

        # Batch: reinforce many weights at once (GPU-accelerated)
        plasticity.reinforce_batch([
            ("pathway_a", True, 0.8),
            ("pathway_b", False, 0.5),
            ("pathway_c", True, 1.0),
        ])

        # Query current state
        plasticity.get_threshold("t1_gate")  # May have shifted
        plasticity.get_weight("neuron_pathway")  # May have strengthened
    """

    def __init__(self, use_gpu: bool = True):
        self._thresholds: Dict[str, AdaptiveThreshold] = {}
        self._weights: Dict[str, AdaptiveWeight] = {}
        self._all_events: List[AdaptationEvent] = []
        self._use_gpu = use_gpu and (_GPU_AVAILABLE or _TORCH_GPU)

    def add_threshold(
        self,
        name: str,
        initial: float = 0.5,
        target_pass_rate: float = 0.8,
        learning_rate: float = 0.01,
        **kwargs,
    ) -> AdaptiveThreshold:
        """Register an adaptive threshold."""
        t = AdaptiveThreshold(
            name=name,
            initial=initial,
            target_pass_rate=target_pass_rate,
            learning_rate=learning_rate,
            **kwargs,
        )
        self._thresholds[name] = t
        return t

    def add_weight(
        self,
        name: str,
        initial: float = 1.0,
        learning_rate: float = 0.05,
        **kwargs,
    ) -> AdaptiveWeight:
        """Register an adaptive weight."""
        w = AdaptiveWeight(
            name=name,
            initial=initial,
            learning_rate=learning_rate,
            use_gpu=self._use_gpu,
            **kwargs,
        )
        self._weights[name] = w
        return w

    def record_outcome(
        self, threshold_name: str, passed: bool, value: float,
    ) -> Optional[AdaptationEvent]:
        """Record outcome for a threshold and let it adapt."""
        t = self._thresholds.get(threshold_name)
        if t is None:
            return None
        event = t.record(passed, value)
        if event:
            self._all_events.append(event)
        return event

    def reinforce_weight(
        self, weight_name: str, success: bool, magnitude: float = 1.0,
    ) -> Optional[AdaptationEvent]:
        """Reinforce a weight based on outcome."""
        w = self._weights.get(weight_name)
        if w is None:
            return None
        event = w.reinforce(success, magnitude)
        self._all_events.append(event)
        return event

    def reinforce_batch(
        self,
        updates: List[Tuple[str, bool, float]],
    ) -> List[AdaptationEvent]:
        """Batch-reinforce multiple weights in a single vectorised pass.

        Each element is (weight_name, success, magnitude).
        When GPU is available the deltas are computed as a single
        CuPy/torch vector operation then scattered back to individual
        weights. On CPU it falls back to sequential reinforcement.

        Returns the list of AdaptationEvent objects generated.
        """
        if not updates:
            return []

        ordered_names: List[str] = []
        ordered_success: List[bool] = []
        ordered_mag: List[float] = []
        weights_list: List[AdaptiveWeight] = []

        for name, success, mag in updates:
            w = self._weights.get(name)
            if w is None:
                continue
            ordered_names.append(name)
            ordered_success.append(success)
            ordered_mag.append(mag)
            weights_list.append(w)

        if not weights_list:
            return []

        n = len(weights_list)

        if self._use_gpu and _GPU_AVAILABLE and n >= 4:
            xp = _xp(True)
            old_vals = xp.array(
                [w.value for w in weights_list], dtype=xp.float32,
            )
            lrs = xp.array(
                [w.learning_rate for w in weights_list], dtype=xp.float32,
            )
            mags = xp.array(ordered_mag, dtype=xp.float32)
            signs = xp.array(
                [1.0 if s else -1.0 for s in ordered_success],
                dtype=xp.float32,
            )
            mins = xp.array(
                [w.min_weight for w in weights_list], dtype=xp.float32,
            )
            maxs = xp.array(
                [w.max_weight for w in weights_list], dtype=xp.float32,
            )
            new_vals = old_vals + signs * lrs * mags
            new_vals = xp.clip(new_vals, mins, maxs)

            old_list = old_vals.get().tolist()
            new_list = new_vals.get().tolist()

            events: List[AdaptationEvent] = []
            for i, w in enumerate(weights_list):
                w.value = new_list[i]
                if ordered_success[i]:
                    w._successes += 1
                else:
                    w._failures += 1
                ev = AdaptationEvent(
                    plasticity_type=PlasticityType.HEBBIAN,
                    target=w.name,
                    parameter="weight",
                    old_value=old_list[i],
                    new_value=new_list[i],
                    reason=f"batch {'success' if ordered_success[i] else 'failure'} (magnitude={ordered_mag[i]:.2f})",
                )
                w._adaptations.append(ev)
                events.append(ev)
            self._all_events.extend(events)
            return events

        events = []
        for w, success, mag in zip(weights_list, ordered_success, ordered_mag):
            ev = w.reinforce(success, mag)
            events.append(ev)
        self._all_events.extend(events)
        return events

    def get_threshold(self, name: str) -> Optional[float]:
        t = self._thresholds.get(name)
        return t.value if t else None

    def get_weight(self, name: str) -> Optional[float]:
        w = self._weights.get(name)
        return w.value if w else None

    def check_threshold(self, name: str, value: float) -> Optional[bool]:
        t = self._thresholds.get(name)
        return t.check(value) if t else None

    def decay_all_weights(self) -> List[AdaptationEvent]:
        """Apply natural decay to all weights.

        When GPU is available and there are enough weights, the decay
        is computed as a single vectorised pass.
        """
        weight_list = list(self._weights.values())
        n = len(weight_list)

        if self._use_gpu and _GPU_AVAILABLE and n >= 4:
            xp = _xp(True)
            vals = xp.array([w.value for w in weight_list], dtype=xp.float32)
            initials = xp.array([w._initial for w in weight_list], dtype=xp.float32)
            rates = xp.array([w.decay_rate for w in weight_list], dtype=xp.float32)

            diff = vals - initials
            mask = xp.abs(diff) >= 0.001
            direction = xp.where(diff > 0, -1.0, 1.0)
            new_vals = vals + direction * rates

            overshoot_pos = (direction > 0) & (new_vals > initials)
            overshoot_neg = (direction < 0) & (new_vals < initials)
            new_vals = xp.where(overshoot_pos | overshoot_neg, initials, new_vals)
            new_vals = xp.where(mask, new_vals, vals)

            changed = xp.abs(new_vals - vals) > 0.0001
            old_arr = vals.get().tolist()
            new_arr = new_vals.get().tolist()
            changed_arr = changed.get().tolist()

            events: List[AdaptationEvent] = []
            for i, w in enumerate(weight_list):
                if changed_arr[i]:
                    w.value = new_arr[i]
                    ev = AdaptationEvent(
                        plasticity_type=PlasticityType.HEBBIAN,
                        target=w.name,
                        parameter="weight",
                        old_value=old_arr[i],
                        new_value=new_arr[i],
                        reason="natural decay toward baseline",
                    )
                    w._adaptations.append(ev)
                    events.append(ev)
            self._all_events.extend(events)
            return events

        events = []
        for w in weight_list:
            e = w.apply_decay()
            if e:
                events.append(e)
                self._all_events.append(e)
        return events

    @property
    def events(self) -> List[AdaptationEvent]:
        return list(self._all_events)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "thresholds": {n: t.stats for n, t in self._thresholds.items()},
            "weights": {n: w.stats for n, w in self._weights.items()},
            "total_adaptations": len(self._all_events),
            "gpu_enabled": self._use_gpu,
        }

    def __repr__(self) -> str:
        return (
            f"PlasticityManager("
            f"{len(self._thresholds)} thresholds, "
            f"{len(self._weights)} weights, "
            f"{len(self._all_events)} events)"
        )
