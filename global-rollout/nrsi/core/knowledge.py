"""
NRSI Knowledge Patterns

Knowledge in NRSI is persistent, versioned, and never silently degraded.
You can evolve knowledge (create new versions) and deprecate facts,
but you cannot mutate or delete without an explicit, auditable action.

Forgetting is intentional. Silence is not an option.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from nrsi.core.types import Confidence, NRSIData, TrustLevel
from nrsi.core.errors import (
    KnowledgeMutationError,
    KnowledgeNotFoundError,
    NRSIError,
)


# ── Knowledge Fact ───────────────────────────────────────────────────────────

@dataclass
class KnowledgeFact:
    """A single piece of knowledge within a pattern."""

    name: str
    value: Any
    confidence: float
    source: str
    unit: Optional[str] = None
    validated_by: Optional[str] = None     # Gate name that validated this
    registered_at: float = field(default_factory=time.time)
    deprecated: bool = False
    deprecated_at: Optional[float] = None
    deprecated_by: Optional[str] = None
    deprecation_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.confidence = Confidence.validate(self.confidence)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "confidence": self.confidence,
            "confidence_label": Confidence.label(self.confidence),
            "source": self.source,
            "unit": self.unit,
            "validated_by": self.validated_by,
            "registered_at": self.registered_at,
            "deprecated": self.deprecated,
            "deprecated_at": self.deprecated_at,
            "deprecated_by": self.deprecated_by,
            "deprecation_reason": self.deprecation_reason,
        }


# ── Knowledge Pattern ────────────────────────────────────────────────────────

class KnowledgePattern:
    """
    A collection of related knowledge facts within a domain.

    Knowledge patterns are:
    - Versioned (semantic versioning)
    - Immutable once registered (evolve to create new versions)
    - Auditable (every change is recorded)
    - Persistent (designed to survive process restarts)
    """

    def __init__(
        self,
        domain: str,
        version: str = "1.0.0",
        immutable: bool = True,
        description: Optional[str] = None,
        parent_version: Optional[str] = None,
    ):
        self.domain = domain
        self.version = version
        self.immutable = immutable
        self.description = description
        self.parent_version = parent_version
        self.created_at = time.time()

        self._facts: Dict[str, KnowledgeFact] = {}
        self._audit_log: List[Dict[str, Any]] = []

        self._log("created", f"Knowledge pattern '{domain}' v{version} created")

    # ── Register Facts ───────────────────────────────────────────────────

    def register(
        self,
        fact: str,
        value: Any,
        confidence: float,
        source: str,
        unit: Optional[str] = None,
        validated_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeFact:
        """
        Register a new knowledge fact.

        Once registered in an immutable pattern, facts cannot be changed.
        Use evolve() to create a new version with updated facts.
        """
        if fact in self._facts:
            existing = self._facts[fact]
            if existing.deprecated:
                # Allow re-registering deprecated facts
                pass
            elif self.immutable:
                raise KnowledgeMutationError(
                    pattern_name=f"{self.domain}@{self.version}",
                    fact_name=fact,
                )
            # If not immutable, allow overwrite (mutable patterns)

        kf = KnowledgeFact(
            name=fact,
            value=value,
            confidence=confidence,
            source=source,
            unit=unit,
            validated_by=validated_by,
            metadata=metadata or {},
        )
        self._facts[fact] = kf
        self._log("registered", f"Fact '{fact}' registered", fact_name=fact)
        return kf

    # ── Access Facts ─────────────────────────────────────────────────────

    def get(self, fact: str, include_deprecated: bool = False) -> KnowledgeFact:
        """Retrieve a knowledge fact by name."""
        if fact not in self._facts:
            raise KnowledgeNotFoundError(
                pattern_name=f"{self.domain}@{self.version}",
                fact_name=fact,
            )
        kf = self._facts[fact]
        if kf.deprecated and not include_deprecated:
            raise KnowledgeNotFoundError(
                pattern_name=f"{self.domain}@{self.version}",
                fact_name=fact,
                suggestion=f"Fact '{fact}' is deprecated: {kf.deprecation_reason}",
            )
        return kf

    def get_value(self, fact: str) -> Any:
        """Quick access to a fact's value."""
        return self.get(fact).value

    def get_as_nrsi(self, fact: str) -> NRSIData:
        """Get a fact wrapped as trusted NRSIData."""
        kf = self.get(fact)
        return NRSIData(
            value=kf.value,
            trust_level=TrustLevel.TRUSTED,
            confidence=kf.confidence,
            metadata={
                "source": kf.source,
                "domain": self.domain,
                "version": self.version,
                "fact": kf.name,
            },
        )

    def has(self, fact: str) -> bool:
        """Check if a fact exists (excluding deprecated)."""
        return fact in self._facts and not self._facts[fact].deprecated

    @property
    def facts(self) -> Dict[str, KnowledgeFact]:
        """All active (non-deprecated) facts."""
        return {
            name: kf for name, kf in self._facts.items()
            if not kf.deprecated
        }

    @property
    def all_facts(self) -> Dict[str, KnowledgeFact]:
        """All facts including deprecated."""
        return dict(self._facts)

    # ── Deprecation ──────────────────────────────────────────────────────

    def deprecate(
        self,
        fact: str,
        reason: str,
        deprecated_by: str,
    ) -> None:
        """
        Deprecate a fact. The fact remains in the pattern
        but is no longer returned by default queries.
        This is the only way to "remove" knowledge — explicit, audited, justified.
        """
        if fact not in self._facts:
            raise KnowledgeNotFoundError(
                pattern_name=f"{self.domain}@{self.version}",
                fact_name=fact,
            )

        kf = self._facts[fact]
        kf.deprecated = True
        kf.deprecated_at = time.time()
        kf.deprecated_by = deprecated_by
        kf.deprecation_reason = reason

        self._log(
            "deprecated",
            f"Fact '{fact}' deprecated: {reason}",
            fact_name=fact,
            actor=deprecated_by,
        )

    # ── Evolution ────────────────────────────────────────────────────────

    def evolve(
        self,
        version: str,
        description: Optional[str] = None,
    ) -> KnowledgePattern:
        """
        Create a new version of this pattern.

        The new version inherits all active facts from the current version.
        You can then register new facts or deprecate inherited ones.
        This is copy-on-write semantics for knowledge.
        """
        new_pattern = KnowledgePattern(
            domain=self.domain,
            version=version,
            immutable=self.immutable,
            description=description or self.description,
            parent_version=self.version,
        )

        # Copy all active facts
        for name, kf in self._facts.items():
            if not kf.deprecated:
                new_pattern._facts[name] = KnowledgeFact(
                    name=kf.name,
                    value=copy.deepcopy(kf.value),
                    confidence=kf.confidence,
                    source=kf.source,
                    unit=kf.unit,
                    validated_by=kf.validated_by,
                    registered_at=kf.registered_at,
                    metadata=dict(kf.metadata),
                )

        new_pattern._log(
            "evolved",
            f"Evolved from v{self.version} with {len(new_pattern._facts)} inherited facts",
        )
        return new_pattern

    # ── Audit ────────────────────────────────────────────────────────────

    def _log(
        self,
        action: str,
        message: str,
        fact_name: Optional[str] = None,
        actor: Optional[str] = None,
    ) -> None:
        self._audit_log.append({
            "timestamp": time.time(),
            "action": action,
            "message": message,
            "domain": self.domain,
            "version": self.version,
            "fact_name": fact_name,
            "actor": actor,
        })

    @property
    def audit_log(self) -> List[Dict[str, Any]]:
        """Full audit log for this pattern."""
        return list(self._audit_log)

    # ── Persistence ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to dictionary for persistence."""
        return {
            "domain": self.domain,
            "version": self.version,
            "immutable": self.immutable,
            "description": self.description,
            "parent_version": self.parent_version,
            "created_at": self.created_at,
            "facts": {name: kf.to_dict() for name, kf in self._facts.items()},
            "audit_log": self._audit_log,
        }

    def save(self, path: str) -> None:
        """Save to JSON file."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> KnowledgePattern:
        """Load from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        pattern = cls(
            domain=data["domain"],
            version=data["version"],
            immutable=data.get("immutable", True),
            description=data.get("description"),
            parent_version=data.get("parent_version"),
        )
        pattern.created_at = data.get("created_at", time.time())

        for name, fact_data in data.get("facts", {}).items():
            kf = KnowledgeFact(
                name=fact_data["name"],
                value=fact_data["value"],
                confidence=fact_data["confidence"],
                source=fact_data["source"],
                unit=fact_data.get("unit"),
                validated_by=fact_data.get("validated_by"),
                registered_at=fact_data.get("registered_at", time.time()),
                deprecated=fact_data.get("deprecated", False),
                deprecated_at=fact_data.get("deprecated_at"),
                deprecated_by=fact_data.get("deprecated_by"),
                deprecation_reason=fact_data.get("deprecation_reason"),
            )
            pattern._facts[name] = kf

        return pattern

    # ── Representation ───────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"KnowledgePattern(domain='{self.domain}', version='{self.version}', "
            f"facts={len(self.facts)}, immutable={self.immutable})"
        )

    def __str__(self) -> str:
        lines = [
            f"Knowledge: {self.domain} v{self.version}",
            f"  Facts: {len(self.facts)} active, {len(self._facts) - len(self.facts)} deprecated",
        ]
        for name, kf in self.facts.items():
            conf = Confidence.label(kf.confidence)
            lines.append(f"  - {name} = {kf.value} (confidence: {conf}, source: {kf.source})")
        return "\n".join(lines)


# ── Knowledge Base ───────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    A collection of knowledge patterns.

    The knowledge base manages multiple patterns across domains
    and provides unified querying across all knowledge.
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self._patterns: Dict[str, Dict[str, KnowledgePattern]] = {}
        # Keyed by domain -> version -> pattern

    def add(self, pattern: KnowledgePattern) -> None:
        """Add a knowledge pattern to the base."""
        if pattern.domain not in self._patterns:
            self._patterns[pattern.domain] = {}
        self._patterns[pattern.domain][pattern.version] = pattern

    def get_pattern(
        self,
        domain: str,
        version: Optional[str] = None,
    ) -> KnowledgePattern:
        """
        Get a pattern by domain and optionally version.
        If no version specified, returns the latest.
        """
        if domain not in self._patterns:
            raise NRSIError(
                f"Knowledge domain '{domain}' not found",
                suggestion=f"Available domains: {', '.join(self._patterns.keys())}",
            )

        versions = self._patterns[domain]
        if version:
            if version not in versions:
                raise NRSIError(
                    f"Version '{version}' not found for domain '{domain}'",
                    suggestion=f"Available versions: {', '.join(versions.keys())}",
                )
            return versions[version]
        else:
            # Return latest version (simple string sort — works for semver)
            latest = sorted(versions.keys())[-1]
            return versions[latest]

    def query(self, domain: str, fact: str, version: Optional[str] = None) -> Any:
        """Quick query: get a fact value from a domain."""
        pattern = self.get_pattern(domain, version)
        return pattern.get_value(fact)

    @property
    def domains(self) -> List[str]:
        return list(self._patterns.keys())

    @property
    def stats(self) -> Dict[str, Any]:
        total_facts = 0
        for domain_versions in self._patterns.values():
            for pattern in domain_versions.values():
                total_facts += len(pattern.facts)
        return {
            "name": self.name,
            "domains": len(self._patterns),
            "total_patterns": sum(len(v) for v in self._patterns.values()),
            "total_active_facts": total_facts,
        }

    def __repr__(self) -> str:
        return f"KnowledgeBase(name='{self.name}', domains={self.domains})"
