"""
NRSI validation gates.

This module provides the lightweight gate/runtime primitives that the
services and the transpiler expect at import time:

- `ValidationResult`
- `GateResult`
- `Validator`
- `FunctionValidator`
- `ValidationGate`
- `gate(...)`
"""

from __future__ import annotations

import abc
import functools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from nrsi.core.errors import ConfidenceError, ValidationError
from nrsi.core.types import Confidence, NRSIData, TrustLevel, raw


@dataclass
class ValidationResult:
    """Outcome of one validator."""

    passed: bool
    confidence: float
    validator_name: str
    details: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.confidence = Confidence.validate(self.confidence)


@dataclass
class GateResult:
    """Aggregate result for a full gate execution."""

    gate_name: str
    passed: bool
    confidence: float
    results: List[ValidationResult]
    elapsed_ms: float
    input_data: Any = None

    @property
    def failed_validators(self) -> List[ValidationResult]:
        return [result for result in self.results if not result.passed]

    @property
    def summary(self) -> str:
        total = len(self.results)
        passed = sum(1 for result in self.results if result.passed)
        failed = ", ".join(result.validator_name for result in self.failed_validators)
        base = (
            f"Gate '{self.gate_name}' "
            f"{'PASSED' if self.passed else 'FAILED'}: "
            f"{passed}/{total} validators passed, "
            f"confidence={self.confidence:.4f}, "
            f"elapsed={self.elapsed_ms:.2f}ms"
        )
        return f"{base}, failed=[{failed}]" if failed else base


class Validator(abc.ABC):
    """Abstract validator contract."""

    name: str = "validator"

    @abc.abstractmethod
    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        raise NotImplementedError


class FunctionValidator(Validator):
    """Adapter for plain Python callables."""

    def __init__(self, fn: Callable[..., Any], name: Optional[str] = None):
        self._fn = fn
        self.name = name or getattr(fn, "__name__", "function_validator")

    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        try:
            raw_result = self._fn(data, context)
        except TypeError:
            raw_result = self._fn(data)
        except Exception as exc:
            return ValidationResult(
                passed=False,
                confidence=Confidence.NONE,
                validator_name=self.name,
                details=f"Validator raised exception: {exc}",
            )

        if isinstance(raw_result, ValidationResult):
            if not raw_result.validator_name:
                raw_result.validator_name = self.name
            return raw_result

        if isinstance(raw_result, tuple):
            if len(raw_result) == 3:
                passed, confidence, details = raw_result
                return ValidationResult(
                    passed=bool(passed),
                    confidence=float(confidence),
                    validator_name=self.name,
                    details=None if details is None else str(details),
                )
            if len(raw_result) == 2:
                passed, confidence = raw_result
                return ValidationResult(
                    passed=bool(passed),
                    confidence=float(confidence),
                    validator_name=self.name,
                )

        if isinstance(raw_result, float):
            return ValidationResult(
                passed=raw_result > 0,
                confidence=abs(raw_result),
                validator_name=self.name,
            )

        return ValidationResult(
            passed=bool(raw_result),
            confidence=Confidence.HIGH if raw_result else Confidence.NONE,
            validator_name=self.name,
        )


class ValidationGate:
    """Run validators and elevate trust on success."""

    def __init__(
        self,
        name: Optional[str] = None,
        confidence_threshold: float = Confidence.HIGH,
        validators: Optional[List[Validator]] = None,
        target_trust: TrustLevel = TrustLevel.TRUSTED,
        require_all: bool = True,
        audit: bool = True,
    ) -> None:
        self._name = name or f"gate_{id(self):x}"
        self._confidence_threshold = Confidence.validate(confidence_threshold)
        self._validators: List[Validator] = list(validators or [])
        self._target_trust = target_trust
        self._require_all = require_all
        self._audit = audit
        self._last_gate_result: Optional[GateResult] = None
        self._total_processed = 0
        self._total_passed = 0
        self._total_failed = 0
        self._confidence_sum = 0.0

    def __call__(self, fn: Callable[..., Any]) -> Callable[..., NRSIData]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> NRSIData:
            return self.process(fn(*args, **kwargs))

        return wrapper

    @property
    def name(self) -> str:
        return self._name

    @property
    def last_gate_result(self) -> Optional[GateResult]:
        return self._last_gate_result

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "name": self._name,
            "total_processed": self._total_processed,
            "total_passed": self._total_passed,
            "total_failed": self._total_failed,
            "pass_rate": self._total_passed / max(self._total_processed, 1),
            "avg_confidence": self._confidence_sum / max(self._total_passed, 1),
        }

    def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> NRSIData:
        start = time.monotonic()
        self._total_processed += 1

        # Unwrap NRSIData so validators receive raw values (str/dict/etc.).
        # Validators are content-shape predicates and shouldn't have to
        # know about the trust system. Audit/gate result still records
        # the original wrapper so provenance is preserved end-to-end.
        validator_input = data.value if isinstance(data, NRSIData) else data
        results = [validator.validate(validator_input, context) for validator in self._validators]
        passed = all(result.passed for result in results) if self._require_all else any(
            result.passed for result in results
        )
        confidence = (
            sum(result.confidence for result in results) / len(results) if results else Confidence.ABSOLUTE
        )
        gate_result = GateResult(
            gate_name=self._name,
            passed=passed,
            confidence=confidence,
            results=results,
            elapsed_ms=(time.monotonic() - start) * 1000,
            input_data=data if self._audit else None,
        )
        self._last_gate_result = gate_result

        if not passed:
            self._total_failed += 1
            failed = ", ".join(result.validator_name for result in gate_result.failed_validators)
            raise ValidationError(
                gate_name=self._name,
                reason=f"Validation failed: {failed or 'gate rejected input'}",
                confidence=confidence,
                required_confidence=self._confidence_threshold,
                suggestion="Ensure the input satisfies every required validator.",
            )

        if confidence < self._confidence_threshold:
            self._total_failed += 1
            raise ConfidenceError(
                actual=confidence,
                required=self._confidence_threshold,
                context=f"gate={self._name}",
                suggestion="Add stronger validators or lower the gate threshold intentionally.",
            )

        self._total_passed += 1
        self._confidence_sum += confidence

        if isinstance(data, NRSIData):
            if data.trust_level >= self._target_trust:
                return data
            return data.elevate(
                to_level=self._target_trust,
                confidence=confidence,
                gate_name=self._name,
                reason=gate_result.summary,
            )

        wrapped = raw(data)
        if self._target_trust == TrustLevel.RAW:
            return wrapped
        return wrapped.elevate(
            to_level=self._target_trust,
            confidence=confidence,
            gate_name=self._name,
            reason=gate_result.summary,
        )


def gate(
    name: Optional[str] = None,
    confidence_threshold: float = Confidence.HIGH,
    validators: Optional[List[Validator]] = None,
    target_trust: TrustLevel = TrustLevel.TRUSTED,
    require_all: bool = True,
    audit: bool = True,
) -> ValidationGate:
    """Convenience constructor matching transpiler/runtime expectations."""

    return ValidationGate(
        name=name,
        confidence_threshold=confidence_threshold,
        validators=validators,
        target_trust=target_trust,
        require_all=require_all,
        audit=audit,
    )
