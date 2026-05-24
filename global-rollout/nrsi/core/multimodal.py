"""NRSI Cross-Modal Binding — Object Identity Across Modalities.

When processing multimodal input, the same real-world entity may appear
in text ("the car"), audio (engine sound), and video (visual frame).
These types bind cross-modal references to shared identity.

  BindingId        — Unique identity for a real-world entity/event
  ModalReference   — A reference to an entity in a specific modality
  CrossModalBinding — Links references across modalities
  SynchronyConstraint — Temporal alignment between modalities
  FusionResult     — Combined evidence from multiple modalities

Patent-covered: NRSI Cross-Modal Binding System, VelarIQ.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Enums
# ═══════════════════════════════════════════════════════════════════════════════

class ModalityType(Enum):
    """Supported input/output modalities."""

    TEXT = auto()
    AUDIO = auto()
    VIDEO = auto()
    IMAGE = auto()
    SENSOR = auto()
    STRUCTURED_DATA = auto()


class EntityType(Enum):
    """Ontological category of a bound entity."""

    OBJECT = auto()
    EVENT = auto()
    PERSON = auto()
    PLACE = auto()
    CONCEPT = auto()


class AlignmentMethod(Enum):
    """How cross-modal references were linked."""

    COREFERENCE = auto()   # Textual coreference resolution
    TEMPORAL = auto()       # Timestamps align
    SPATIAL = auto()        # Bounding-box / spatial overlap
    SEMANTIC = auto()       # Embedding similarity


class FusionMethod(Enum):
    """Strategy for combining evidence from multiple modalities."""

    EARLY = auto()    # Fuse raw features before inference
    LATE = auto()     # Fuse per-modality decisions
    HYBRID = auto()   # Combine early and late signals


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ModalReference — pointer into a single modality
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModalReference:
    """A reference to an entity within one modality.

    ``content_locator`` is modality-specific:
      - TEXT:   ``{"text": "the car", "char_start": 42, "char_end": 49}``
      - AUDIO:  ``{"start_ms": 1200, "end_ms": 3400}``
      - VIDEO:  ``{"frame": 120, "bbox": [x, y, w, h]}``
      - IMAGE:  ``{"bbox": [x, y, w, h]}``
      - SENSOR: ``{"channel": "lidar", "timestamp_ms": 500}``
      - STRUCTURED_DATA: ``{"table": "orders", "row_id": 7}``
    """

    ref_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    modality: ModalityType = ModalityType.TEXT
    content_locator: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    description: str = ""

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence must be in [0.0, 1.0], got {self.confidence}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BindingId — shared identity token
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BindingId:
    """Stable identity for a real-world entity tracked across modalities."""

    entity_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    entity_type: EntityType = EntityType.OBJECT
    display_name: str = ""
    first_seen_modality: ModalityType = ModalityType.TEXT


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CrossModalBinding — links references together
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CrossModalBinding:
    """Associates ``ModalReference`` instances across modalities under one
    ``BindingId``.
    """

    binding_id: BindingId
    references: List[ModalReference] = field(default_factory=list)
    alignment_confidence: float = 1.0
    method: AlignmentMethod = AlignmentMethod.SEMANTIC
    created_at: float = field(default_factory=time.time)

    # ── mutation ───────────────────────────────────────────────────────────

    def add_reference(self, ref: ModalReference) -> None:
        self.references.append(ref)
        self._recompute_alignment()

    # ── queries ────────────────────────────────────────────────────────────

    def alignment_score(self) -> float:
        """How well the references agree (mean confidence)."""
        if not self.references:
            return 0.0
        return sum(r.confidence for r in self.references) / len(self.references)

    def primary_reference(self) -> Optional[ModalReference]:
        """Reference with the highest confidence."""
        if not self.references:
            return None
        return max(self.references, key=lambda r: r.confidence)

    @property
    def modalities(self) -> Set[ModalityType]:
        return {r.modality for r in self.references}

    @property
    def is_multimodal(self) -> bool:
        return len(self.modalities) > 1

    # ── internals ──────────────────────────────────────────────────────────

    def _recompute_alignment(self) -> None:
        self.alignment_confidence = self.alignment_score()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SynchronyConstraint — temporal/spatial alignment spec
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SynchronyConstraint:
    """Temporal (and optional spatial) alignment between two modality
    streams for a shared binding.
    """

    binding_id: BindingId
    modality_a: ModalityType
    modality_b: ModalityType
    temporal_offset_ms: float = 0.0
    tolerance_ms: float = 100.0
    spatial_overlap: Optional[float] = None

    @property
    def is_synchronized(self) -> bool:
        return abs(self.temporal_offset_ms) <= self.tolerance_ms


# ═══════════════════════════════════════════════════════════════════════════════
# 6. FusionResult — combined evidence
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FusionConflict:
    """Disagreement between modalities on the same entity."""

    modality_a: ModalityType
    modality_b: ModalityType
    description: str


@dataclass
class FusionResult:
    """Outcome of fusing evidence across modalities for one binding."""

    binding_id: BindingId
    fused_confidence: float
    contributing_modalities: List[ModalityType] = field(default_factory=list)
    fusion_method: FusionMethod = FusionMethod.LATE
    conflicts: List[FusionConflict] = field(default_factory=list)

    @property
    def is_consistent(self) -> bool:
        return len(self.conflicts) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MultiModalRegistry — central binding tracker
# ═══════════════════════════════════════════════════════════════════════════════

class MultiModalRegistry:
    """Tracks all ``CrossModalBinding`` instances and provides lookup,
    fusion, and temporal-alignment operations.
    """

    def __init__(self) -> None:
        self._bindings: Dict[str, CrossModalBinding] = {}

    # ── bind ───────────────────────────────────────────────────────────────

    def bind(
        self,
        references: Sequence[ModalReference],
        entity_type: EntityType = EntityType.OBJECT,
        display_name: str = "",
        method: AlignmentMethod = AlignmentMethod.SEMANTIC,
    ) -> CrossModalBinding:
        """Create a new cross-modal binding from a set of references."""
        first_mod = references[0].modality if references else ModalityType.TEXT
        bid = BindingId(
            entity_type=entity_type,
            display_name=display_name,
            first_seen_modality=first_mod,
        )
        binding = CrossModalBinding(
            binding_id=bid,
            references=list(references),
            method=method,
        )
        binding._recompute_alignment()
        self._bindings[bid.entity_id] = binding
        return binding

    # ── lookup ─────────────────────────────────────────────────────────────

    def find_by_entity(self, entity_id: str) -> Optional[CrossModalBinding]:
        return self._bindings.get(entity_id)

    def find_by_modality(
        self,
        modality: ModalityType,
        locator_key: Optional[str] = None,
        locator_value: Optional[Any] = None,
    ) -> List[CrossModalBinding]:
        """Find bindings that include a reference in *modality*.

        Optionally filter by a specific locator field (e.g.
        ``locator_key="text", locator_value="the car"``).
        """
        results: List[CrossModalBinding] = []
        for binding in self._bindings.values():
            for ref in binding.references:
                if ref.modality is not modality:
                    continue
                if locator_key is not None:
                    if ref.content_locator.get(locator_key) != locator_value:
                        continue
                results.append(binding)
                break
        return results

    def all_bindings(self) -> List[CrossModalBinding]:
        return list(self._bindings.values())

    # ── fusion ─────────────────────────────────────────────────────────────

    def fuse(
        self,
        entity_id: str,
        method: FusionMethod = FusionMethod.LATE,
    ) -> Optional[FusionResult]:
        """Combine evidence across modalities for one entity."""
        binding = self._bindings.get(entity_id)
        if binding is None:
            return None

        modalities = sorted(binding.modalities, key=lambda m: m.name)
        confidences = [r.confidence for r in binding.references]
        if not confidences:
            return FusionResult(
                binding_id=binding.binding_id,
                fused_confidence=0.0,
                contributing_modalities=modalities,
                fusion_method=method,
            )

        # Fused confidence: 1 − ∏(1 − cᵢ)  (Noisy-OR combination)
        product = 1.0
        for c in confidences:
            product *= (1.0 - c)
        fused = round(1.0 - product, 6)

        conflicts = self._detect_conflicts(binding)

        return FusionResult(
            binding_id=binding.binding_id,
            fused_confidence=fused,
            contributing_modalities=modalities,
            fusion_method=method,
            conflicts=conflicts,
        )

    # ── temporal alignment ─────────────────────────────────────────────────

    def temporal_align(
        self,
        entity_id_a: str,
        entity_id_b: str,
    ) -> Optional[SynchronyConstraint]:
        """Compute temporal alignment between two bindings.

        Uses the first time-stamped reference in each binding.
        """
        ba = self._bindings.get(entity_id_a)
        bb = self._bindings.get(entity_id_b)
        if ba is None or bb is None:
            return None

        ts_a = self._extract_timestamp(ba)
        ts_b = self._extract_timestamp(bb)
        if ts_a is None or ts_b is None:
            return None

        offset = ts_b - ts_a
        mod_a = ba.references[0].modality if ba.references else ModalityType.TEXT
        mod_b = bb.references[0].modality if bb.references else ModalityType.TEXT

        return SynchronyConstraint(
            binding_id=ba.binding_id,
            modality_a=mod_a,
            modality_b=mod_b,
            temporal_offset_ms=offset,
        )

    # ── internals ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_timestamp(binding: CrossModalBinding) -> Optional[float]:
        for ref in binding.references:
            loc = ref.content_locator
            for key in ("start_ms", "timestamp_ms", "frame"):
                if key in loc:
                    return float(loc[key])
        return None

    @staticmethod
    def _detect_conflicts(binding: CrossModalBinding) -> List[FusionConflict]:
        """Flag references whose confidence diverges significantly."""
        conflicts: List[FusionConflict] = []
        refs = binding.references
        for i, ra in enumerate(refs):
            for rb in refs[i + 1:]:
                if ra.modality == rb.modality:
                    continue
                if abs(ra.confidence - rb.confidence) > 0.4:
                    conflicts.append(FusionConflict(
                        modality_a=ra.modality,
                        modality_b=rb.modality,
                        description=(
                            f"Confidence divergence: "
                            f"{ra.modality.name}={ra.confidence:.2f} vs "
                            f"{rb.modality.name}={rb.confidence:.2f}"
                        ),
                    ))
        return conflicts


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Facade — API expected by nrsi.core.nrs._process_inner
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CrossModalBindResult:
    """Result from ``CrossModalBinder.bind()``."""
    strength: float = 0.0
    bound_modalities: List[str] = field(default_factory=list)
    binding_id: str = ""


class CrossModalBinder:
    """Facade providing the ``bind(modality, content, domain)`` API that
    ``nrsi.core.nrs._process_inner`` expects.

    Wraps ``MultiModalRegistry`` for the simplified pipeline interface.
    """

    def __init__(self, registry: Optional[MultiModalRegistry] = None) -> None:
        self._registry = registry or MultiModalRegistry()

    def bind(
        self,
        *,
        modality: str,
        content: str,
        domain: str = "general",
    ) -> CrossModalBindResult:
        """Create a cross-modal binding from a single modality reference."""
        try:
            mod_type = ModalityType[modality.upper()]
        except KeyError:
            mod_type = ModalityType.TEXT

        ref = ModalReference(
            modality=mod_type,
            content_locator={"text": content[:200], "domain": domain},
            confidence=0.8,
        )
        binding = self._registry.bind(
            references=[ref],
            entity_type=EntityType.CONCEPT,
            display_name=content[:64],
        )
        return CrossModalBindResult(
            strength=binding.alignment_confidence,
            bound_modalities=[r.modality.name for r in binding.references],
            binding_id=binding.binding_id,
        )

    @property
    def registry(self) -> MultiModalRegistry:
        return self._registry
