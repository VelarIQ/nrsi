"""NRSI Abstraction Hierarchy — Concept Lattice and Type Lifting.

Knowledge exists at different levels of abstraction:
  GoldenRetriever → Dog → Animal → LivingThing → Entity
  HTTP 404 → Client Error → HTTP Error → Network Error → Error

These types provide:
  Concept          — A node in the abstraction hierarchy
  ConceptRelation  — SubconceptOf, InstanceOf, PartOf, etc.
  ConceptLattice   — The full hierarchy as a DAG
  AbstractionLevel — Named granularity levels
  TypeLifting      — Move a claim from specific to general
  TypeLowering     — Move from general to specific (with caveats)
  DefaultInheritance — Properties inherit down unless overridden

Patent-covered: NRSI Abstraction Lattice System, VelarIQ.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Enums
# ═══════════════════════════════════════════════════════════════════════════════

class ConceptRelation(Enum):
    """Semantic relationship between two concepts."""

    SUBCONCEPT_OF = auto()  # is-a (Dog SUBCONCEPT_OF Animal)
    INSTANCE_OF = auto()    # particular (Fido INSTANCE_OF Dog)
    PART_OF = auto()        # has-a / meronymy
    RELATED_TO = auto()     # loose association
    OPPOSITE_OF = auto()    # antonym
    DERIVED_FROM = auto()   # etymological / causal origin
    EQUIVALENT_TO = auto()  # synonymy


class AbstractionLevel(Enum):
    """Named granularity levels in a concept lattice."""

    INSTANCE = 0    # This specific thing  (Fido)
    SPECIES = 1     # This type of thing   (Golden Retriever)
    GENUS = 2       # This category        (Dog)
    FAMILY = 3      # This broad group     (Mammal)
    DOMAIN = 4      # This field           (Animal)
    UNIVERSAL = 5   # Applies to everything (Entity)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Concept
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Concept:
    """A node in the abstraction hierarchy.

    Multiple inheritance is supported: a concept may have several
    parents (e.g. ``FlyingFish`` is a subconcept of both ``Fish``
    and ``FlyingAnimal``).
    """

    concept_id: str
    name: str
    parent_ids: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    domain: Optional[str] = None
    abstraction_level: AbstractionLevel = AbstractionLevel.GENUS


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Relation record (edge in the lattice)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ConceptEdge:
    """Directed edge: child → parent with a typed relation."""

    child_id: str
    parent_id: str
    relation: ConceptRelation = ConceptRelation.SUBCONCEPT_OF


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ConceptLattice — the full hierarchy
# ═══════════════════════════════════════════════════════════════════════════════

class ConceptLattice:
    """A concept lattice stored as a DAG.

    Edges point child → parent.  The lattice supports multiple
    inheritance, transitive closure queries, and default property
    inheritance.
    """

    def __init__(self) -> None:
        self._concepts: Dict[str, Concept] = {}
        self._edges: List[ConceptEdge] = []
        self._children: Dict[str, List[str]] = {}
        self._parents: Dict[str, List[str]] = {}

    # ── construction ───────────────────────────────────────────────────────

    def add_concept(self, concept: Concept) -> None:
        self._concepts[concept.concept_id] = concept
        self._children.setdefault(concept.concept_id, [])
        self._parents.setdefault(concept.concept_id, [])
        for pid in concept.parent_ids:
            if pid in self._concepts:
                self.add_relation(concept.concept_id, pid, ConceptRelation.SUBCONCEPT_OF)

    def add_relation(
        self,
        child_id: str,
        parent_id: str,
        relation: ConceptRelation = ConceptRelation.SUBCONCEPT_OF,
    ) -> None:
        edge = ConceptEdge(child_id=child_id, parent_id=parent_id, relation=relation)
        self._edges.append(edge)
        self._parents.setdefault(child_id, []).append(parent_id)
        self._children.setdefault(parent_id, []).append(child_id)

    @property
    def concepts(self) -> Dict[str, Concept]:
        return dict(self._concepts)

    @property
    def relations(self) -> List[ConceptEdge]:
        return list(self._edges)

    # ── traversal queries ──────────────────────────────────────────────────

    def ancestors(self, concept_id: str) -> List[str]:
        """All ancestors (parents, grandparents, …) via BFS."""
        result: List[str] = []
        visited: Set[str] = set()
        queue: deque[str] = deque(self._parents.get(concept_id, []))
        while queue:
            cid = queue.popleft()
            if cid in visited:
                continue
            visited.add(cid)
            result.append(cid)
            queue.extend(self._parents.get(cid, []))
        return result

    def descendants(self, concept_id: str) -> List[str]:
        """All descendants (children, grandchildren, …) via BFS."""
        result: List[str] = []
        visited: Set[str] = set()
        queue: deque[str] = deque(self._children.get(concept_id, []))
        while queue:
            cid = queue.popleft()
            if cid in visited:
                continue
            visited.add(cid)
            result.append(cid)
            queue.extend(self._children.get(cid, []))
        return result

    def is_subconcept(self, child_id: str, parent_id: str) -> bool:
        """Transitive is-a check."""
        if child_id == parent_id:
            return True
        return parent_id in self.ancestors(child_id)

    def distance(self, concept_a: str, concept_b: str) -> int:
        """Shortest undirected hop count between two concepts.

        Returns -1 if no path exists.
        """
        if concept_a == concept_b:
            return 0
        visited: Set[str] = {concept_a}
        queue: deque[Tuple[str, int]] = deque([(concept_a, 0)])
        while queue:
            cid, dist = queue.popleft()
            neighbours = (
                self._parents.get(cid, []) + self._children.get(cid, [])
            )
            for nid in neighbours:
                if nid == concept_b:
                    return dist + 1
                if nid not in visited:
                    visited.add(nid)
                    queue.append((nid, dist + 1))
        return -1

    def common_ancestor(self, concept_a: str, concept_b: str) -> Optional[str]:
        """Lowest (most specific) common ancestor of two concepts."""
        anc_a = self.ancestors(concept_a)
        anc_b_set = set(self.ancestors(concept_b))
        for a in anc_a:
            if a in anc_b_set:
                return a
        return None

    def inherited_properties(self, concept_id: str) -> Dict[str, Any]:
        """Merge properties walking up the hierarchy.

        More specific (closer to the concept) values override general ones.
        """
        chain = [concept_id] + self.ancestors(concept_id)
        merged: Dict[str, Any] = {}
        for cid in reversed(chain):
            concept = self._concepts.get(cid)
            if concept:
                merged.update(concept.properties)
        return merged


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TypeLifting / TypeLowering
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LiftedClaim:
    """A claim that has been generalized upward in the lattice."""

    original_concept: str
    lifted_concept: str
    original_claim: str
    lifted_claim: str
    original_confidence: float
    lifted_confidence: float
    generalization_tax: float


@dataclass
class LoweredClaim:
    """A general claim specialized downward — use with caution."""

    original_concept: str
    lowered_concept: str
    original_claim: str
    lowered_claim: str
    original_confidence: float
    lowered_confidence: float
    caveat: str


class TypeLifting:
    """Generalize a claim from a specific concept to a more abstract one.

    Each abstraction-level hop applies a *generalization tax* that
    reduces confidence (what is true of dogs may not be true of all animals).
    """

    GENERALIZATION_TAX_PER_HOP: float = 0.05

    def __init__(self, lattice: ConceptLattice) -> None:
        self._lattice = lattice

    def lift(
        self,
        claim: str,
        from_concept: str,
        to_concept: str,
    ) -> LiftedClaim:
        hops = self._lattice.distance(from_concept, to_concept)
        if hops < 0:
            hops = 1
        tax = min(0.5, hops * self.GENERALIZATION_TAX_PER_HOP)
        original_conf = 1.0
        lifted_conf = max(0.0, original_conf - tax)

        from_name = self._concept_name(from_concept)
        to_name = self._concept_name(to_concept)

        return LiftedClaim(
            original_concept=from_concept,
            lifted_concept=to_concept,
            original_claim=claim,
            lifted_claim=claim.replace(from_name, to_name) if from_name != to_name else claim,
            original_confidence=original_conf,
            lifted_confidence=round(lifted_conf, 4),
            generalization_tax=round(tax, 4),
        )

    def _concept_name(self, concept_id: str) -> str:
        c = self._lattice._concepts.get(concept_id)
        return c.name if c else concept_id


class TypeLowering:
    """Specialize a general claim to a more specific concept.

    Lowering always adds a caveat because general properties may not
    hold for every subclass.
    """

    SPECIALIZATION_PENALTY: float = 0.10

    def __init__(self, lattice: ConceptLattice) -> None:
        self._lattice = lattice

    def lower(
        self,
        claim: str,
        from_concept: str,
        to_concept: str,
    ) -> LoweredClaim:
        hops = self._lattice.distance(from_concept, to_concept)
        if hops < 0:
            hops = 1
        penalty = min(0.5, hops * self.SPECIALIZATION_PENALTY)
        original_conf = 1.0
        lowered_conf = max(0.0, original_conf - penalty)

        from_name = self._concept_name(from_concept)
        to_name = self._concept_name(to_concept)

        return LoweredClaim(
            original_concept=from_concept,
            lowered_concept=to_concept,
            original_claim=claim,
            lowered_claim=claim.replace(from_name, to_name) if from_name != to_name else claim,
            original_confidence=original_conf,
            lowered_confidence=round(lowered_conf, 4),
            caveat=f"Generalized from {from_name}; may not apply to all {to_name}.",
        )

    def _concept_name(self, concept_id: str) -> str:
        c = self._lattice._concepts.get(concept_id)
        return c.name if c else concept_id


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DefaultInheritance
# ═══════════════════════════════════════════════════════════════════════════════

class DefaultInheritance:
    """Walk up the lattice to resolve property values.

    Local overrides shadow inherited values (most-specific wins).
    """

    def __init__(self, lattice: ConceptLattice) -> None:
        self._lattice = lattice

    def resolve(self, concept_id: str, property_name: str) -> Optional[Any]:
        """Walk from *concept_id* upward until *property_name* is found."""
        chain = [concept_id] + self._lattice.ancestors(concept_id)
        for cid in chain:
            concept = self._lattice._concepts.get(cid)
            if concept and property_name in concept.properties:
                return concept.properties[property_name]
        return None

    def overrides(self, concept_id: str) -> Dict[str, Any]:
        """Properties defined directly on *concept_id* that shadow ancestors."""
        concept = self._lattice._concepts.get(concept_id)
        if concept is None:
            return {}
        parent_props = self._lattice.inherited_properties(concept_id)
        own: Dict[str, Any] = {}
        for k, v in concept.properties.items():
            ancestor_val = None
            for pid in concept.parent_ids:
                ancestor_val = self.resolve(pid, k)
                if ancestor_val is not None:
                    break
            if ancestor_val is not None and ancestor_val != v:
                own[k] = v
        return own


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Pre-loaded common hierarchies
# ═══════════════════════════════════════════════════════════════════════════════

def _add_chain(lattice: ConceptLattice, chain: List[Tuple[str, str]], domain: str) -> None:
    """Insert a linear chain of (id, name) tuples as parent→child."""
    prev_id: Optional[str] = None
    for cid, name in chain:
        parents = [prev_id] if prev_id else []
        lattice.add_concept(Concept(
            concept_id=cid,
            name=name,
            parent_ids=parents,
            domain=domain,
        ))
        prev_id = cid


def common_concept_lattice() -> ConceptLattice:
    """A starter lattice with 20+ everyday hierarchies pre-loaded."""
    lat = ConceptLattice()

    _add_chain(lat, [
        ("entity", "Entity"),
        ("living_thing", "Living Thing"),
        ("animal", "Animal"),
        ("mammal", "Mammal"),
        ("dog", "Dog"),
        ("golden_retriever", "Golden Retriever"),
    ], "biology")

    _add_chain(lat, [
        ("entity", "Entity"),
        ("living_thing", "Living Thing"),
        ("animal", "Animal"),
        ("bird", "Bird"),
        ("parrot", "Parrot"),
    ], "biology")

    _add_chain(lat, [
        ("entity", "Entity"),
        ("living_thing", "Living Thing"),
        ("plant", "Plant"),
        ("tree", "Tree"),
        ("oak", "Oak"),
    ], "biology")

    _add_chain(lat, [
        ("entity", "Entity"),
        ("artifact", "Artifact"),
        ("vehicle", "Vehicle"),
        ("car", "Car"),
        ("sedan", "Sedan"),
    ], "transport")

    _add_chain(lat, [
        ("vehicle", "Vehicle"),
        ("truck", "Truck"),
    ], "transport")

    _add_chain(lat, [
        ("vehicle", "Vehicle"),
        ("bicycle", "Bicycle"),
    ], "transport")

    _add_chain(lat, [
        ("entity", "Entity"),
        ("event", "Event"),
        ("error", "Error"),
        ("network_error", "Network Error"),
        ("http_error", "HTTP Error"),
        ("client_error", "Client Error"),
        ("http_404", "HTTP 404"),
    ], "computing")

    _add_chain(lat, [
        ("error", "Error"),
        ("runtime_error", "Runtime Error"),
        ("null_pointer", "Null Pointer"),
    ], "computing")

    _add_chain(lat, [
        ("error", "Error"),
        ("timeout_error", "Timeout Error"),
    ], "computing")

    _add_chain(lat, [
        ("entity", "Entity"),
        ("abstract_concept", "Abstract Concept"),
        ("emotion", "Emotion"),
        ("positive_emotion", "Positive Emotion"),
        ("joy", "Joy"),
    ], "psychology")

    _add_chain(lat, [
        ("emotion", "Emotion"),
        ("negative_emotion", "Negative Emotion"),
        ("anger", "Anger"),
    ], "psychology")

    _add_chain(lat, [
        ("entity", "Entity"),
        ("place", "Place"),
        ("country", "Country"),
    ], "geography")

    _add_chain(lat, [
        ("place", "Place"),
        ("city", "City"),
    ], "geography")

    _add_chain(lat, [
        ("entity", "Entity"),
        ("substance", "Substance"),
        ("chemical", "Chemical"),
        ("acid", "Acid"),
    ], "chemistry")

    _add_chain(lat, [
        ("entity", "Entity"),
        ("abstract_concept", "Abstract Concept"),
        ("disease", "Disease"),
        ("infectious_disease", "Infectious Disease"),
        ("viral_infection", "Viral Infection"),
    ], "medicine")

    _add_chain(lat, [
        ("disease", "Disease"),
        ("chronic_disease", "Chronic Disease"),
        ("diabetes", "Diabetes"),
    ], "medicine")

    _add_chain(lat, [
        ("artifact", "Artifact"),
        ("tool", "Tool"),
        ("software", "Software"),
        ("operating_system", "Operating System"),
    ], "computing")

    _add_chain(lat, [
        ("software", "Software"),
        ("application", "Application"),
        ("web_app", "Web Application"),
    ], "computing")

    _add_chain(lat, [
        ("entity", "Entity"),
        ("quantity", "Quantity"),
        ("currency", "Currency"),
    ], "finance")

    _add_chain(lat, [
        ("artifact", "Artifact"),
        ("food", "Food"),
        ("fruit", "Fruit"),
        ("apple", "Apple"),
    ], "nutrition")

    _add_chain(lat, [
        ("food", "Food"),
        ("grain", "Grain"),
        ("rice", "Rice"),
    ], "nutrition")

    return lat


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Facade — API expected by nrsi.core.nrs._process_inner
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AbstractionClassification:
    """Result from ``AbstractionHierarchy.classify()``."""
    level: int = 0
    category: str = ""
    concept_id: str = ""


class AbstractionHierarchy:
    """Facade providing the ``classify(content, domain)`` API that
    ``nrsi.core.nrs._process_inner`` expects.

    Wraps a ``ConceptLattice`` and looks up the best-matching concept
    for a piece of content.
    """

    def __init__(self, lattice: Optional[ConceptLattice] = None) -> None:
        self._lattice = lattice or common_concept_lattice()

    def classify(
        self,
        *,
        content: str,
        domain: str = "general",
    ) -> AbstractionClassification:
        """Classify *content* within the concept lattice."""
        tokens = content.lower().split()[:20]
        best_concept: Optional[Concept] = None
        best_depth = -1

        for tok in tokens:
            node = self._lattice._concepts.get(tok)
            if node is not None:
                try:
                    depth = len(self._lattice.ancestors(tok))
                except Exception:
                    depth = 0
                if depth > best_depth:
                    best_depth = depth
                    best_concept = node

        if best_concept is None:
            return AbstractionClassification(
                level=AbstractionLevel.UNIVERSAL.value,
                category="general",
            )

        level_val = min(best_depth, AbstractionLevel.UNIVERSAL.value)
        return AbstractionClassification(
            level=level_val,
            category=best_concept.domain or domain,
            concept_id=best_concept.concept_id,
        )

    @property
    def lattice(self) -> ConceptLattice:
        return self._lattice
