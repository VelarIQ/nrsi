"""
NRSI Core Exceptions

Every error in NRSI follows the same philosophy:
1. What went wrong — clear description
2. Why it's wrong — reference to the architectural principle
3. How to fix it — concrete suggestion
4. Context — the specific values/types involved
"""

from __future__ import annotations
from typing import Any, Optional


class NRSIError(Exception):
    """Base exception for all NRSI errors."""

    def __init__(self, message: str, suggestion: Optional[str] = None):
        self.suggestion = suggestion
        if suggestion:
            message = f"{message}\n  → Suggestion: {suggestion}"
        super().__init__(message)


# ── Trust & Validation Errors ────────────────────────────────────────────────

class ValidationError(NRSIError):
    """Raised when data fails validation through a gate."""

    def __init__(
        self,
        gate_name: str,
        reason: str,
        confidence: Optional[float] = None,
        required_confidence: Optional[float] = None,
        suggestion: Optional[str] = None,
    ):
        self.gate_name = gate_name
        self.reason = reason
        self.confidence = confidence
        self.required_confidence = required_confidence

        parts = [f"Validation failed at gate '{gate_name}': {reason}"]
        if confidence is not None and required_confidence is not None:
            parts.append(
                f"  Confidence: {confidence:.4f} (required: {required_confidence:.4f})"
            )
        super().__init__("\n".join(parts), suggestion)


class TrustError(NRSIError):
    """Raised when trust level is insufficient for an operation."""

    def __init__(
        self,
        expected_trust: str,
        actual_trust: str,
        operation: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        self.expected_trust = expected_trust
        self.actual_trust = actual_trust
        self.operation = operation

        msg = f"Trust level insufficient: expected '{expected_trust}', got '{actual_trust}'"
        if operation:
            msg = f"Cannot perform '{operation}': {msg}"
        if not suggestion:
            suggestion = f"Pass data through a validation gate to elevate trust from '{actual_trust}' to '{expected_trust}'"
        super().__init__(msg, suggestion)


class ConfidenceError(NRSIError):
    """Raised when confidence score is below the required threshold."""

    def __init__(
        self,
        actual: float,
        required: float,
        context: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        self.actual = actual
        self.required = required

        msg = f"Confidence {actual:.4f} is below required threshold {required:.4f}"
        if context:
            msg = f"{msg} ({context})"
        if not suggestion:
            suggestion = "Add additional validation layers or lower the confidence threshold"
        super().__init__(msg, suggestion)


# ── Hierarchy Errors ─────────────────────────────────────────────────────────

class LayerViolationError(NRSIError):
    """Raised when data flow violates hierarchical layer rules."""

    def __init__(
        self,
        source_layer: int,
        target_layer: int,
        source_name: Optional[str] = None,
        target_name: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        self.source_layer = source_layer
        self.target_layer = target_layer

        src = f"layer {source_layer}" + (f" ({source_name})" if source_name else "")
        tgt = f"layer {target_layer}" + (f" ({target_name})" if target_name else "")

        if target_layer < source_layer:
            direction = "downward"
            if not suggestion:
                suggestion = "Data flows upward through the hierarchy. Use a channel for explicit downward communication."
        elif target_layer > source_layer + 1:
            direction = "skip-layer"
            if not suggestion:
                suggestion = f"Data must flow through each layer sequentially. Define intermediate layers between {src} and {tgt}."
        else:
            direction = "invalid"

        msg = f"Illegal {direction} data flow from {src} to {tgt}"
        super().__init__(msg, suggestion)


# ── Knowledge Errors ─────────────────────────────────────────────────────────

class KnowledgeMutationError(NRSIError):
    """Raised when attempting to mutate immutable knowledge."""

    def __init__(
        self,
        pattern_name: str,
        fact_name: str,
        suggestion: Optional[str] = None,
    ):
        self.pattern_name = pattern_name
        self.fact_name = fact_name

        msg = f"Cannot mutate fact '{fact_name}' in immutable knowledge pattern '{pattern_name}'"
        if not suggestion:
            suggestion = f"Use {pattern_name}.evolve(version='x.y.z') to create a new version with the updated fact"
        super().__init__(msg, suggestion)


class KnowledgeNotFoundError(NRSIError):
    """Raised when a referenced knowledge fact doesn't exist."""

    def __init__(
        self,
        pattern_name: str,
        fact_name: str,
        suggestion: Optional[str] = None,
    ):
        msg = f"Fact '{fact_name}' not found in knowledge pattern '{pattern_name}'"
        if not suggestion:
            suggestion = f"Register the fact first: {pattern_name}.register(fact='{fact_name}', ...)"
        super().__init__(msg, suggestion)


# ── Governance Errors ────────────────────────────────────────────────────────

class GovernanceViolationError(NRSIError):
    """Raised when an operation violates a governance policy."""

    def __init__(
        self,
        policy_name: str,
        standard: str,
        violation: str,
        suggestion: Optional[str] = None,
    ):
        self.policy_name = policy_name
        self.standard = standard
        self.violation = violation

        msg = f"Governance violation [{standard}] in policy '{policy_name}': {violation}"
        super().__init__(msg, suggestion)


class AuditRequiredError(NRSIError):
    """Raised when an operation requires an audit trail but none exists."""

    def __init__(
        self,
        operation: str,
        suggestion: Optional[str] = None,
    ):
        msg = f"Operation '{operation}' requires an audit trail"
        if not suggestion:
            suggestion = "Enable audit logging or wrap the operation in an audited context"
        super().__init__(msg, suggestion)


# ── Communication Errors ─────────────────────────────────────────────────────

class ChannelError(NRSIError):
    """Raised when NRSIP channel communication fails."""

    def __init__(
        self,
        channel_name: str,
        reason: str,
        suggestion: Optional[str] = None,
    ):
        self.channel_name = channel_name
        msg = f"Channel '{channel_name}' error: {reason}"
        super().__init__(msg, suggestion)
