"""
NRSI Creases — Domain Knowledge Modules + Axiom Tiers.

Creases are specialized neural pathways — modular knowledge
structures analogous to cortical folds in the human brain.

Brain analogy:
  Baby brain: smooth, few creases → limited specialization
  Adult brain: highly folded → deep domain expertise
  More creases = more specialized knowledge packed in

Each crease encapsulates:
  1. Domain-specific validated facts
  2. Domain-specific inference patterns
  3. Domain-specific validation criteria
  4. Domain-specific terminology mappings

Creases are TRAINED ONCE and LOCKED. Never retrained.
New knowledge enters through the Symbiotic Mesh as
validated additions — the crease structure grows but
existing validated pathways are never modified.

Axiom Tiers (Ground Truth Hierarchy):
  T0 Axioms: Mathematical/logical truths (immutable forever)
    - 2+2=4, law of non-contradiction, modus ponens
  T1 Validated Facts: Verified against authoritative sources
    - Speed of light, boiling point of water, Paris is capital of France
  Dynamic Knowledge: Current state, subject to update
    - Stock prices, weather, election results

Everything validates against this hierarchy.
T0 axioms can never be contradicted.
T1 facts update only through Symbiotic Mesh validation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Set


# ── Axiom Tiers ──────────────────────────────────────────────────────────────

class AxiomTier(IntEnum):
    """
    Ground truth hierarchy. Higher tier = more immutable.

    T0: Mathematical/logical axioms — NEVER change.
        Cannot be contradicted by any source.
        Examples: 2+2=4, A ∧ ¬A = False, modus ponens.

    T1: Validated facts — change only through Symbiotic Mesh.
        Verified against authoritative external sources.
        Examples: speed of light, chemical formulas.

    T2: Domain knowledge — domain-crease level facts.
        Trained once per crease, locked after training.
        Examples: drug interactions, tax code rules.

    T3: Dynamic knowledge — current state of the world.
        Updated through continuous Mesh validation.
        Examples: stock prices, election results.
    """
    T0_AXIOM = 0       # Mathematical/logical truth — immutable forever
    T1_FACT = 1         # Validated fact — immutable without Mesh override
    T2_DOMAIN = 2       # Domain crease knowledge — trained once, locked
    T3_DYNAMIC = 3      # Dynamic — updated through Mesh


@dataclass
class Axiom:
    """
    A single axiom or validated fact in the ground truth hierarchy.

    key: Unique identifier
    value: The truth content
    tier: Which axiom tier (T0-T3)
    domain: Optional domain association
    source: Where this axiom came from
    immutable: Whether it can ever be changed (T0 = always True)
    validated_at: When it was last validated
    """
    key: str
    value: Any
    tier: AxiomTier
    domain: Optional[str] = None
    source: str = "system"
    immutable: bool = False
    validated_at: float = field(default_factory=time.time)

    def __post_init__(self):
        # T0 axioms are always immutable
        if self.tier == AxiomTier.T0_AXIOM:
            self.immutable = True

    def __repr__(self) -> str:
        lock = "🔒" if self.immutable else "🔓"
        return f"Axiom({lock} {self.tier.name}: '{self.key}')"


class AxiomBase:
    """
    The ground truth database — axioms and validated facts.

    Hierarchical validation: any new knowledge must be
    consistent with ALL higher-tier axioms.

    T3 cannot contradict T2.
    T2 cannot contradict T1.
    T1 cannot contradict T0.
    T0 is absolute truth.

    Usage:
        base = AxiomBase()

        # T0: Mathematical axioms (immutable forever)
        base.add_axiom("addition_commutativity", "a + b = b + a",
                        AxiomTier.T0_AXIOM, source="mathematics")

        # T1: Validated facts
        base.add_axiom("speed_of_light", 299_792_458,
                        AxiomTier.T1_FACT, domain="physics", source="NIST")

        # Check consistency
        base.is_consistent("speed_of_light", 300_000_000)  # False — contradicts T1

        # T0 can never be overwritten
        base.add_axiom("addition_commutativity", "wrong", AxiomTier.T0_AXIOM)
        # → raises ValueError
    """

    def __init__(self):
        self._axioms: Dict[str, Axiom] = {}
        self._by_tier: Dict[AxiomTier, Dict[str, Axiom]] = {
            t: {} for t in AxiomTier
        }
        self._by_domain: Dict[str, Dict[str, Axiom]] = {}

    def add_axiom(
        self,
        key: str,
        value: Any,
        tier: AxiomTier,
        domain: Optional[str] = None,
        source: str = "system",
    ) -> Axiom:
        """
        Add an axiom to the ground truth base.

        T0 axioms cannot be overwritten once set.
        T1 facts cannot be overwritten without explicit force.
        """
        existing = self._axioms.get(key)
        if existing and existing.immutable:
            raise ValueError(
                f"Cannot overwrite immutable axiom '{key}' "
                f"(tier={existing.tier.name}). "
                f"T0 axioms are absolute truth."
            )

        axiom = Axiom(
            key=key,
            value=value,
            tier=tier,
            domain=domain,
            source=source,
        )
        self._axioms[key] = axiom
        self._by_tier[tier][key] = axiom

        if domain:
            if domain not in self._by_domain:
                self._by_domain[domain] = {}
            self._by_domain[domain][key] = axiom

        return axiom

    def get(self, key: str) -> Optional[Axiom]:
        return self._axioms.get(key)

    def lookup(self, query: str, domain: Optional[str] = None) -> Optional[str]:
        """Search axioms for one relevant to *query* by keyword overlap.

        Requires at least 2 content-word overlaps to avoid spurious matches.
        Domain-scoped axioms are preferred when *domain* is provided.
        """
        _stop = {"the", "what", "how", "why", "who", "when", "where", "which",
                 "does", "can", "will", "for", "and", "are", "was", "has",
                 "been", "this", "that", "with", "from", "about", "into"}
        q_terms = {w.lower() for w in query.split() if len(w) > 2 and w.lower() not in _stop}
        if len(q_terms) < 1:
            return None

        best_score, best_axiom = 0, None
        for axiom in self._axioms.values():
            val_str = str(axiom.value).lower()
            key_str = axiom.key.lower().replace("_", " ").replace(".", " ")
            combined = val_str + " " + key_str
            a_terms = {w for w in combined.split() if len(w) > 2 and w not in _stop}
            overlap = len(q_terms & a_terms)
            if domain and axiom.domain == domain:
                overlap += 1
            if overlap > best_score:
                best_score = overlap
                best_axiom = axiom

        if best_score >= 2 and best_axiom:
            return str(best_axiom.value)
        return None

    def get_tier(self, tier: AxiomTier) -> Dict[str, Axiom]:
        return dict(self._by_tier[tier])

    def get_domain(self, domain: str) -> Dict[str, Axiom]:
        return dict(self._by_domain.get(domain, {}))

    def is_consistent(self, key: str, proposed_value: Any) -> bool:
        """
        Check if a proposed value is consistent with existing axioms.

        If the key exists at a higher tier, the proposed value
        must match the existing value.
        """
        existing = self._axioms.get(key)
        if existing is None:
            return True  # No conflict
        return existing.value == proposed_value

    def validate_against_axioms(
        self,
        key: str,
        value: Any,
        tier: AxiomTier,
    ) -> Dict[str, Any]:
        """
        Validate a proposed fact against all higher-tier axioms.

        Returns validation result with details.
        """
        result = {
            "key": key,
            "proposed_tier": tier.name,
            "consistent": True,
            "conflicts": [],
        }

        existing = self._axioms.get(key)
        if existing:
            if existing.tier.value < tier.value:
                # Higher tier exists — must match
                if existing.value != value:
                    result["consistent"] = False
                    result["conflicts"].append({
                        "existing_tier": existing.tier.name,
                        "existing_value": existing.value,
                        "proposed_value": value,
                    })
            elif existing.immutable:
                if existing.value != value:
                    result["consistent"] = False
                    result["conflicts"].append({
                        "existing_tier": existing.tier.name,
                        "reason": "immutable axiom cannot be overwritten",
                    })

        return result

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total": len(self._axioms),
            "by_tier": {t.name: len(a) for t, a in self._by_tier.items()},
            "domains": list(self._by_domain.keys()),
            "immutable": sum(1 for a in self._axioms.values() if a.immutable),
        }

    def __repr__(self) -> str:
        counts = ", ".join(f"{t.name}={len(a)}" for t, a in self._by_tier.items())
        return f"AxiomBase({counts})"


# ── Domain Crease ────────────────────────────────────────────────────────────

class CreaseState(Enum):
    """Lifecycle state of a domain crease."""
    TRAINING = "training"       # Being trained on domain data
    VALIDATING = "validating"   # Post-training validation (99.9% threshold)
    LOCKED = "locked"           # Production — no gradient updates
    GROWING = "growing"         # Locked but accepting Mesh-validated additions
    RETIRED = "retired"         # Superseded by newer crease version


@dataclass
class CreaseLayer:
    """
    Internal layer within a domain crease.

    Like cortical layers (1-6) in the brain, each crease
    has internal layers organizing different knowledge types.

    Layer 1: Core facts (most fundamental domain knowledge)
    Layer 2: Inference patterns (how facts relate)
    Layer 3: Validation criteria (domain-specific checks)
    Layer 4: Terminology (domain jargon, aliases, mappings)
    """
    layer_id: int
    name: str
    facts: Dict[str, Any] = field(default_factory=dict)
    fact_count: int = 0

    def add(self, key: str, value: Any) -> None:
        self.facts[key] = value
        self.fact_count = len(self.facts)

    def get(self, key: str) -> Optional[Any]:
        return self.facts.get(key)

    def __repr__(self) -> str:
        return f"CreaseLayer({self.layer_id}: {self.name}, facts={self.fact_count})"


class DomainCrease:
    """
    A Domain Crease — Modular Knowledge Module.

    Analogous to cortical folds in the human brain.
    Trained once on validated domain data, then LOCKED.
    Never retrained. Never modified via gradient descent.

    Internal structure (4 layers):
      Layer 1 — Core Facts: fundamental domain knowledge
      Layer 2 — Inference Patterns: how facts relate, cause/effect
      Layer 3 — Validation Criteria: domain-specific checks
      Layer 4 — Terminology: jargon, aliases, mappings

    Training protocol:
      1. Curate validated dataset from authoritative sources
      2. Train crease parameters (supervised learning)
      3. Validate against held-out test set (99.9% accuracy threshold)
      4. LOCK crease — no further gradient updates
      5. Enable Symbiotic Mesh integration for validated additions

    After locking, the crease can still GROW through Mesh-validated
    additions, but existing pathways are never modified.

    Usage:
        crease = DomainCrease("medical")

        # Add knowledge during training
        crease.add_fact("aspirin_mechanism", "COX-1/COX-2 inhibition",
                         layer=1)
        crease.add_inference("nsaid_gi_risk",
                              "NSAIDs increase GI bleeding risk",
                              premises=["COX-1 inhibition", "gastric protection"])
        crease.add_validation_rule("drug_interaction_check",
                                     lambda x: x.get("contraindications") is not None)
        crease.add_terminology("MI", "Myocardial Infarction")

        # Lock after training
        crease.lock()

        # After locking — can add Mesh-validated knowledge
        crease.grow("new_drug_data", {...}, source="fda_feed", mesh_validated=True)
    """

    def __init__(self, domain: str, version: str = "1.0"):
        self.domain = domain
        self.version = version
        self.state = CreaseState.TRAINING
        self.created_at = time.time()
        self.locked_at: Optional[float] = None
        self.training_accuracy: Optional[float] = None

        # Internal layers
        self.layers: Dict[int, CreaseLayer] = {
            1: CreaseLayer(1, "Core Facts"),
            2: CreaseLayer(2, "Inference Patterns"),
            3: CreaseLayer(3, "Validation Criteria"),
            4: CreaseLayer(4, "Terminology"),
        }

        # Validation rules (callables)
        self._validators: Dict[str, Callable] = {}

        # Growth log (post-lock additions via Mesh)
        self._growth_log: List[Dict[str, Any]] = []

        # Stats
        self._total_facts = 0
        self._queries_served = 0

    def add_fact(self, key: str, value: Any, layer: int = 1) -> None:
        """Add a fact during training phase."""
        if self.state == CreaseState.LOCKED or self.state == CreaseState.GROWING:
            raise ValueError(
                f"Crease '{self.domain}' is locked. "
                f"Use grow() for Mesh-validated additions."
            )
        self.layers[layer].add(key, value)
        self._total_facts += 1

    def add_inference(
        self,
        key: str,
        conclusion: str,
        premises: Optional[List[str]] = None,
    ) -> None:
        """Add an inference pattern (Layer 2)."""
        self.layers[2].add(key, {
            "conclusion": conclusion,
            "premises": premises or [],
        })
        self._total_facts += 1

    def add_validation_rule(self, key: str, rule: Callable) -> None:
        """Add a domain-specific validation rule (Layer 3)."""
        self._validators[key] = rule
        self.layers[3].add(key, f"validator:{key}")

    def add_terminology(self, abbreviation: str, full_term: str) -> None:
        """Add terminology mapping (Layer 4)."""
        self.layers[4].add(abbreviation, full_term)

    def lock(self, training_accuracy: float = 0.999) -> None:
        """
        Lock the crease. No more gradient updates. Ever.

        Requires training accuracy ≥ 99.9%.
        After locking, crease moves to GROWING state
        (accepts Mesh-validated additions only).
        """
        if training_accuracy < 0.999:
            raise ValueError(
                f"Training accuracy {training_accuracy:.4f} below "
                f"99.9% threshold. Cannot lock crease."
            )
        self.state = CreaseState.LOCKED
        self.locked_at = time.time()
        self.training_accuracy = training_accuracy
        # Immediately transition to GROWING (accepts Mesh additions)
        self.state = CreaseState.GROWING

    def grow(
        self,
        key: str,
        value: Any,
        source: str,
        mesh_validated: bool = False,
        layer: int = 1,
    ) -> None:
        """
        Add Mesh-validated knowledge to a locked crease.

        MUST be mesh_validated=True. Unvalidated additions
        are rejected — this is how we prevent hallucinations.
        """
        if self.state not in (CreaseState.GROWING, CreaseState.LOCKED):
            raise ValueError(
                f"Crease '{self.domain}' is in state {self.state.value}. "
                f"Only locked/growing creases accept Mesh additions."
            )
        if not mesh_validated:
            raise ValueError(
                "Cannot add unvalidated knowledge to a locked crease. "
                "All post-lock additions MUST be Mesh-validated. "
                "Unvalidated additions = potential hallucinations."
            )
        self.layers[layer].add(key, value)
        self._total_facts += 1
        self._growth_log.append({
            "key": key,
            "source": source,
            "layer": layer,
            "added_at": time.time(),
        })

    def query(self, key: str) -> Optional[Any]:
        """Look up a fact across all layers."""
        self._queries_served += 1
        for layer in self.layers.values():
            result = layer.get(key)
            if result is not None:
                return result
        return None

    def search(self, query: str, min_relevance: float = 0.3) -> List[tuple]:
        """Fuzzy keyword search across all facts in the crease.

        Returns list of (key, value, relevance_score) tuples sorted by relevance.
        """
        self._queries_served += 1
        _stop = {"the", "what", "how", "why", "who", "when", "where", "which",
                 "is", "are", "does", "can", "will", "for", "and", "this",
                 "that", "with", "from", "about", "into", "should", "take", "a", "an"}
        q_terms = {w.lower().strip("?!.,") for w in query.split()
                   if w.lower().strip("?!.,") not in _stop and len(w.strip("?!.,")) > 1}
        if not q_terms:
            return []

        results: list = []
        for layer in self.layers.values():
            for key, value in layer.facts.items():
                val_str = str(value)
                key_terms = {w.lower() for w in key.replace("_", " ").split() if len(w) > 1}
                val_terms = {w.lower() for w in val_str.split() if len(w) > 1} - _stop
                matched = q_terms & (key_terms | val_terms)
                if not matched:
                    continue
                relevance = len(matched) / max(len(q_terms), 1)
                if relevance >= min_relevance:
                    results.append((key, val_str, round(relevance, 3)))

        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def validate_domain(self, data: Dict[str, Any]) -> Dict[str, bool]:
        """Run all domain-specific validation rules."""
        results = {}
        for name, validator in self._validators.items():
            try:
                results[name] = validator(data)
            except Exception:
                results[name] = False
        return results

    def resolve_terminology(self, term: str) -> Optional[str]:
        """Resolve a domain abbreviation to full term."""
        return self.layers[4].get(term)

    @property
    def total_facts(self) -> int:
        return sum(l.fact_count for l in self.layers.values())

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "version": self.version,
            "state": self.state.value,
            "training_accuracy": self.training_accuracy,
            "layers": {l.name: l.fact_count for l in self.layers.values()},
            "total_facts": self.total_facts,
            "growth_additions": len(self._growth_log),
            "queries_served": self._queries_served,
            "validators": len(self._validators),
        }

    def __repr__(self) -> str:
        return (
            f"DomainCrease('{self.domain}', state={self.state.value}, "
            f"facts={self.total_facts})"
        )


# ── Crease Registry ─────────────────────────────────────────────────────────

class CreaseRegistry:
    """
    Registry of all domain creases in the NRS.

    Manages the collection of domain knowledge modules.
    Like the complete set of cortical folds in the brain.

    Usage:
        registry = CreaseRegistry()

        # Create and register creases
        med = registry.create("medical")
        fin = registry.create("financial")

        # Look up crease by domain
        crease = registry.get("medical")

        # List all domains
        domains = registry.domains  # ["medical", "financial"]
    """

    def __init__(self):
        self._creases: Dict[str, DomainCrease] = {}
        self.axiom_base = AxiomBase()

    def create(self, domain: str, version: str = "1.0") -> DomainCrease:
        """Create and register a new domain crease."""
        crease = DomainCrease(domain, version)
        self._creases[domain] = crease
        return crease

    def get(self, domain: str) -> Optional[DomainCrease]:
        return self._creases.get(domain)

    def query_across_domains(self, key: str) -> Dict[str, Any]:
        """Query a key across all domain creases."""
        results = {}
        for domain, crease in self._creases.items():
            result = crease.query(key)
            if result is not None:
                results[domain] = result
        return results

    @property
    def domains(self) -> List[str]:
        return list(self._creases.keys())

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_creases": len(self._creases),
            "domains": self.domains,
            "creases": {d: c.stats for d, c in self._creases.items()},
            "axiom_base": self.axiom_base.stats,
        }

    def __repr__(self) -> str:
        return f"CreaseRegistry(domains={self.domains})"
