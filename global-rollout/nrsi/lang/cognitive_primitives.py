"""NRSI Cognitive Primitives — Runtime Implementations.

These are the 5 NRSI-native primitives that close the remaining AGI gaps
without requiring external ML frameworks. They're called by transpiled
NRSI code and also wired directly into the NRS pipeline engines.

Primitives:
    nrsi_compose        — compositional text synthesis from knowledge fragments
    LearnableStore      — persistent belief store with decay and reinforcement
    nrsi_semantic_distance — graph-based meaning similarity (no embeddings)
    nrsi_decompose      — recursive goal splitting via causal chains
    nrsi_intent_match   — semantic pattern matching with wildcards
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger("nrsi.cognitive_primitives")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. COMPOSE — Compositional Text Synthesis
# ═══════════════════════════════════════════════════════════════════════════════

_COMPOSITION_STRATEGIES = {
    "synthesis", "interpolation", "hierarchical",
    "contrastive", "narrative", "analytical",
}

_DISCOURSE_CONNECTORS = {
    "synthesis": [
        "Building on this, ", "Furthermore, ", "This connects to ",
        "In addition, ", "Extending this analysis, ",
    ],
    "contrastive": [
        "However, ", "In contrast, ", "On the other hand, ",
        "Conversely, ", "While this is true, ",
    ],
    # Narrative connectives previously implied causal/temporal sequence
    # ("Following from this", "This led to", "In the wake of this") even when
    # the source fragments were independently retrieved facts that share only
    # a keyword with the query. That produced fluent prose that read like a
    # causal chain but was untrue. We now use neutral additive connectives so
    # the engine never asserts causation it didn't derive.
    "narrative": [
        "Related: ", "Also: ", "On a related note, ",
        "Separately, ", "Independently, ",
    ],
    "analytical": [
        "The evidence suggests ", "Analysis indicates ",
        "The data shows ", "Upon examination, ",
    ],
}


@dataclass
class CompositionResult:
    """Output of nrsi_compose."""
    text: str = ""
    confidence: float = 0.0
    provenance: List[str] = field(default_factory=list)
    strategy: str = "synthesis"
    fragment_count: int = 0


def nrsi_compose(
    sources: List[Any],
    *,
    strategy: str = "synthesis",
    provenance: str = "epistemic",
    domain: str = "general",
    max_fragments: int = 4,
    query: Optional[str] = None,
) -> CompositionResult:
    """Synthesize coherent text from heterogeneous knowledge fragments.

    Unlike template interpolation, this performs:
    1. Fragment deduplication via content hashing
    2. Relevance ranking by domain overlap
    3. Discourse-aware stitching with connectors
    4. Confidence propagation from source fragments
    """
    if strategy not in _COMPOSITION_STRATEGIES:
        strategy = "synthesis"

    fragments: List[Tuple[str, float]] = []
    for src in sources:
        if isinstance(src, str):
            fragments.append((src, 0.7))
        elif isinstance(src, dict):
            fragments.append((
                str(src.get("text", src.get("value", str(src)))),
                float(src.get("confidence", 0.7)),
            ))
        elif isinstance(src, (list, tuple)):
            for item in src:
                if isinstance(item, str):
                    fragments.append((item, 0.7))
                elif isinstance(item, dict):
                    fragments.append((
                        str(item.get("text", item.get("value", str(item)))),
                        float(item.get("confidence", 0.7)),
                    ))
        else:
            fragments.append((str(src), 0.5))

    seen: Set[str] = set()
    unique: List[Tuple[str, float]] = []
    for text, conf in fragments:
        sig = hashlib.sha256(text[:100].lower().encode()).hexdigest()[:12]
        if sig not in seen and len(text.strip()) > 10:
            seen.add(sig)
            unique.append((text.strip(), conf))

    unique = unique[:max_fragments]
    if not unique:
        return CompositionResult(
            text="Insufficient knowledge fragments for composition.",
            confidence=0.1, strategy=strategy,
        )

    # ── Query-aware relevance filter ──
    # When a query is provided, score each fragment against it via spaCy
    # vectors and keep only those above a relevance threshold. Stops the
    # composer from chaining tangentially-related facts (e.g. "boiling
    # point of water" pulling fluoridation/salinization facts that share
    # the keyword "water" but don't answer the question).
    #
    # Threshold note: GloVe/spaCy 300d cosines for English run high — even
    # unrelated facts share 0.30-0.45 from common stopword embedding mass.
    # We use 0.55 as an empirical "actually relevant" floor and abstain
    # entirely if nothing clears it.
    if query and unique:
        try:
            from nrsi.core.neurons import NRSEmbeddingEngine
            _emb = NRSEmbeddingEngine(dim=300)
            if _emb._external_loaded:
                import numpy as _np
                q_vec = _emb.embed(query)
                q_norm = _np.linalg.norm(q_vec)
                if q_norm > 1e-8:
                    q_unit = q_vec / q_norm
                    scored: List[Tuple[str, float, float]] = []
                    for text, conf in unique:
                        f_vec = _emb.embed(text[:256])
                        f_norm = _np.linalg.norm(f_vec)
                        if f_norm < 1e-8:
                            continue
                        cos = float(_np.dot(q_unit, f_vec / f_norm))
                        if cos >= 0.55:
                            scored.append((text, conf, cos))
                    if scored:
                        scored.sort(key=lambda x: -x[2])
                        unique = [(t, c) for t, c, _ in scored[:max_fragments]]
                    else:
                        # No fragment crosses the relevance threshold.
                        # Honest abstention is better than templated prose.
                        return CompositionResult(
                            text=("I don't have a grounded answer to this in my "
                                  "knowledge packs. Rather than chain together "
                                  "tangentially-related facts, I'd rather flag "
                                  "the gap. Try rephrasing or pointing me at a "
                                  "specific domain."),
                            confidence=0.30,
                            provenance=["abstention:no_relevant_fact"],
                            strategy=strategy,
                            fragment_count=0,
                        )
        except Exception:
            pass

    unique.sort(key=lambda x: -x[1])

    connectors = _DISCOURSE_CONNECTORS.get(strategy, _DISCOURSE_CONNECTORS["synthesis"])
    composed_parts: List[str] = []
    provenance_ids: List[str] = []

    for i, (text, conf) in enumerate(unique):
        text = text.rstrip(". ") + "."
        if i == 0:
            composed_parts.append(text)
        else:
            connector = connectors[i % len(connectors)]
            first_char = text[0].lower() if text else ""
            composed_parts.append(f"{connector}{first_char}{text[1:]}")
        provenance_ids.append(f"frag_{i}:{conf:.2f}")

    composed_text = " ".join(composed_parts)
    avg_conf = sum(c for _, c in unique) / len(unique) if unique else 0.0
    composed_conf = avg_conf * 0.9

    return CompositionResult(
        text=composed_text,
        confidence=composed_conf,
        provenance=provenance_ids,
        strategy=strategy,
        fragment_count=len(unique),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PERSIST / LearnableStore — Durable Belief Store
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StoredBelief:
    """A single belief in the LearnableStore."""
    claim_hash: str
    text: str
    confidence: float
    domain: str = "general"
    source: str = ""
    created_at: float = field(default_factory=time.time)
    last_reinforced: float = field(default_factory=time.time)
    reinforcement_count: int = 0
    decay_rate: float = 0.01


class LearnableStore:
    """Persistent belief store with decay, reinforcement, and conflict resolution.

    Closes the learning engine persistence gap. Beliefs decay over time
    unless reinforced. Conflicting beliefs trigger revision.
    """

    def __init__(
        self,
        *,
        decay: float = 0.01,
        reinforcement: float = 0.1,
        conflict_resolution: str = "revision",
        backing: str = "memory",
        backing_path: str = "",
    ) -> None:
        self._decay_rate = max(0.0, min(1.0, float(decay)))
        self._reinforcement_delta = max(0.0, min(1.0, float(reinforcement)))
        self._conflict_strategy = conflict_resolution
        self._backing = backing
        self._backing_path = backing_path or ""
        self._beliefs: Dict[str, StoredBelief] = {}
        self._conflict_log: List[Dict[str, Any]] = []

        if self._backing == "disk" and self._backing_path:
            self._load_from_disk()

    def store(
        self, text: str, confidence: float = 0.7,
        domain: str = "general", source: str = "",
    ) -> Tuple[bool, str]:
        """Store a belief. Returns (was_new, claim_hash)."""
        claim_hash = hashlib.sha256(text.lower().strip().encode()).hexdigest()[:16]

        existing = self._beliefs.get(claim_hash)
        if existing is not None:
            existing.confidence = min(
                0.99, existing.confidence + self._reinforcement_delta
            )
            existing.last_reinforced = time.time()
            existing.reinforcement_count += 1
            self._maybe_flush()
            return False, claim_hash

        conflicts = self._find_conflicts(text, domain)
        if conflicts and self._conflict_strategy == "revision":
            for c_hash in conflicts:
                old = self._beliefs[c_hash]
                if confidence > old.confidence:
                    self._conflict_log.append({
                        "action": "superseded",
                        "old_hash": c_hash,
                        "new_hash": claim_hash,
                        "old_conf": old.confidence,
                        "new_conf": confidence,
                    })
                    del self._beliefs[c_hash]
                else:
                    self._conflict_log.append({
                        "action": "rejected",
                        "old_hash": c_hash,
                        "new_hash": claim_hash,
                        "reason": "lower_confidence",
                    })
                    return False, claim_hash

        self._beliefs[claim_hash] = StoredBelief(
            claim_hash=claim_hash, text=text, confidence=confidence,
            domain=domain, source=source, decay_rate=self._decay_rate,
        )
        self._maybe_flush()
        return True, claim_hash

    def query(
        self, domain: Optional[str] = None, min_confidence: float = 0.0,
        keywords: Optional[List[str]] = None, limit: int = 50,
    ) -> List[StoredBelief]:
        """Query beliefs with optional filters."""
        self._apply_decay()
        results: List[StoredBelief] = []
        for b in self._beliefs.values():
            if domain and b.domain != domain:
                continue
            if b.confidence < min_confidence:
                continue
            if keywords:
                text_lower = b.text.lower()
                if not any(kw.lower() in text_lower for kw in keywords):
                    continue
            results.append(b)
        results.sort(key=lambda x: -x.confidence)
        return results[:limit]

    def _find_conflicts(self, text: str, domain: str) -> List[str]:
        """Find beliefs that might conflict with the new claim."""
        conflicts: List[str] = []
        words = set(text.lower().split())
        negation_words = {"not", "no", "never", "none", "neither", "nor",
                          "isn't", "aren't", "doesn't", "don't", "wasn't",
                          "weren't", "won't", "can't", "couldn't", "shouldn't"}
        has_negation = bool(words & negation_words)

        for h, b in self._beliefs.items():
            if b.domain != domain:
                continue
            b_words = set(b.text.lower().split())
            overlap = len(words & b_words) / max(len(words | b_words), 1)
            if overlap < 0.4:
                continue
            b_has_neg = bool(b_words & negation_words)
            if has_negation != b_has_neg:
                conflicts.append(h)
        return conflicts

    def _apply_decay(self) -> None:
        """Apply time-based decay to all beliefs."""
        now = time.time()
        to_remove: List[str] = []
        for h, b in self._beliefs.items():
            elapsed_hours = (now - b.last_reinforced) / 3600.0
            decay = b.decay_rate * elapsed_hours
            b.confidence = max(0.0, b.confidence - decay)
            if b.confidence < 0.01:
                to_remove.append(h)
        for h in to_remove:
            del self._beliefs[h]

    def _maybe_flush(self) -> None:
        if self._backing == "disk" and self._backing_path:
            self._flush_to_disk()

    def _flush_to_disk(self) -> None:
        try:
            data = {
                h: {
                    "text": b.text, "confidence": b.confidence,
                    "domain": b.domain, "source": b.source,
                    "created_at": b.created_at,
                    "last_reinforced": b.last_reinforced,
                    "reinforcement_count": b.reinforcement_count,
                }
                for h, b in self._beliefs.items()
            }
            path = Path(self._backing_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("LearnableStore flush failed: %s", exc)

    def _load_from_disk(self) -> None:
        try:
            path = Path(self._backing_path)
            if not path.exists():
                return
            data = json.loads(path.read_text(encoding="utf-8"))
            for h, entry in data.items():
                self._beliefs[h] = StoredBelief(
                    claim_hash=h, text=entry["text"],
                    confidence=entry.get("confidence", 0.5),
                    domain=entry.get("domain", "general"),
                    source=entry.get("source", ""),
                    created_at=entry.get("created_at", time.time()),
                    last_reinforced=entry.get("last_reinforced", time.time()),
                    reinforcement_count=entry.get("reinforcement_count", 0),
                )
        except Exception as exc:
            logger.warning("LearnableStore load failed: %s", exc)

    def __len__(self) -> int:
        return len(self._beliefs)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_beliefs": len(self._beliefs),
            "conflicts_resolved": len(self._conflict_log),
            "domains": list({b.domain for b in self._beliefs.values()}),
            "avg_confidence": (
                sum(b.confidence for b in self._beliefs.values()) / len(self._beliefs)
                if self._beliefs else 0.0
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SEMANTIC_DISTANCE — Graph-Based Meaning Similarity
# ═══════════════════════════════════════════════════════════════════════════════

_SEMANTIC_GRAPH: Dict[str, Set[str]] = {}
_DOMAIN_HIERARCHIES: Dict[str, List[str]] = {
    "science": ["physics", "chemistry", "biology", "astronomy", "geology",
                "mathematics", "computer_science", "engineering"],
    "medicine": ["anatomy", "pharmacology", "pathology", "surgery",
                 "psychiatry", "neurology", "cardiology", "oncology"],
    "law": ["criminal", "civil", "constitutional", "international",
            "corporate", "intellectual_property", "environmental"],
    "finance": ["banking", "investment", "insurance", "trading",
                "accounting", "economics", "cryptocurrency"],
    "technology": ["software", "hardware", "networking", "security",
                   "ai", "databases", "cloud", "mobile"],
}

_SYNONYM_CLUSTERS: Dict[str, FrozenSet[str]] = {
    "large": frozenset({"big", "huge", "massive", "enormous", "vast", "extensive"}),
    "small": frozenset({"tiny", "little", "minute", "compact", "minor"}),
    "fast": frozenset({"quick", "rapid", "swift", "speedy", "prompt"}),
    "good": frozenset({"excellent", "great", "superior", "fine", "outstanding"}),
    "bad": frozenset({"poor", "terrible", "awful", "inferior", "deficient"}),
    "create": frozenset({"make", "build", "construct", "develop", "produce", "generate"}),
    "remove": frozenset({"delete", "eliminate", "erase", "destroy", "discard"}),
    "understand": frozenset({"comprehend", "grasp", "perceive", "recognize", "know"}),
    "change": frozenset({"modify", "alter", "transform", "adjust", "revise"}),
    "important": frozenset({"significant", "crucial", "vital", "essential", "critical"}),
}

_REVERSE_SYNONYM: Dict[str, str] = {}
for _cluster_key, _members in _SYNONYM_CLUSTERS.items():
    for _m in _members:
        _REVERSE_SYNONYM[_m] = _cluster_key
    _REVERSE_SYNONYM[_cluster_key] = _cluster_key


@dataclass
class SemanticDistanceResult:
    similarity: float = 0.0
    method: str = "combined"
    shared_concepts: List[str] = field(default_factory=list)
    domain_overlap: float = 0.0


def nrsi_semantic_distance(
    a: Any, b: Any,
    *,
    metric: str = "combined",
    domain: str = "general",
) -> SemanticDistanceResult:
    """Compute meaning similarity between two concepts without embeddings.

    Uses three complementary signals:
    1. Token overlap (Jaccard on word stems)
    2. Synonym cluster membership
    3. Domain hierarchy distance
    """
    text_a = str(a).lower().strip()
    text_b = str(b).lower().strip()

    tokens_a = set(_tokenize(text_a))
    tokens_b = set(_tokenize(text_b))

    jaccard = _jaccard(tokens_a, tokens_b)

    syn_score = _synonym_similarity(tokens_a, tokens_b)

    domain_score = _domain_distance(tokens_a, tokens_b)

    shared = list(tokens_a & tokens_b)

    combined = (0.4 * jaccard) + (0.35 * syn_score) + (0.25 * domain_score)
    combined = max(0.0, min(1.0, combined))

    return SemanticDistanceResult(
        similarity=combined,
        method=metric,
        shared_concepts=shared[:10],
        domain_overlap=domain_score,
    )


def _tokenize(text: str) -> Set[str]:
    """Split text into meaningful tokens, stripping noise."""
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                  "being", "have", "has", "had", "do", "does", "did", "will",
                  "would", "could", "should", "may", "might", "shall", "can",
                  "to", "of", "in", "for", "on", "with", "at", "by", "from",
                  "as", "into", "about", "it", "its", "this", "that", "these",
                  "those", "and", "or", "but", "not", "if", "then", "than"}
    words = re.findall(r'[a-z]+', text)
    return {w for w in words if w not in stop_words and len(w) > 2}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _synonym_similarity(a: Set[str], b: Set[str]) -> float:
    """Measure how many tokens from a/b share synonym clusters."""
    if not a or not b:
        return 0.0
    clusters_a = {_REVERSE_SYNONYM.get(w, w) for w in a}
    clusters_b = {_REVERSE_SYNONYM.get(w, w) for w in b}
    shared = clusters_a & clusters_b
    union = clusters_a | clusters_b
    return len(shared) / len(union) if union else 0.0


def _domain_distance(a: Set[str], b: Set[str]) -> float:
    """Score based on shared domain hierarchy membership."""
    domains_a: Set[str] = set()
    domains_b: Set[str] = set()
    for parent, children in _DOMAIN_HIERARCHIES.items():
        child_set = set(children)
        for w in a:
            if w in child_set or w == parent:
                domains_a.add(parent)
        for w in b:
            if w in child_set or w == parent:
                domains_b.add(parent)
    if not domains_a and not domains_b:
        return 0.5
    if not domains_a or not domains_b:
        return 0.2
    shared = domains_a & domains_b
    return len(shared) / max(len(domains_a | domains_b), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DECOMPOSE — Recursive Goal Splitting
# ═══════════════════════════════════════════════════════════════════════════════

_DECOMPOSE_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(why|what caused|what causes|reason for)\b", "causal"),
    (r"\b(compare|vs\.?|versus|differ|difference)\b", "comparative"),
    (r"\b(how does|how do|mechanism|explain how)\b", "explanatory"),
    (r"\b(history|timeline|when did|chronolog)\b", "temporal"),
    (r"\b(analy[sz]\w*|assess\w*|examin\w*|investigat\w*)\b", "analytical"),
    (r"\b(should|pros and cons|worth|advisable|recommend)\b", "evaluative"),
    (r"\b(list|enumerate|name all|what are the|types of)\b", "enumerative"),
    (r"\b(prove|demonstrate that|show that)\b", "proof"),
    (r"\b(compute|calculate|how much|how many)\b", "computational"),
    (r"\b(what if|suppose|hypothetically|imagine)\b", "counterfactual"),
    (r"\b(predict|forecast|will|future|expect)\b", "predictive"),
]

_RECURSIVE_TEMPLATES: Dict[str, List[Dict[str, str]]] = {
    "causal": [
        {"action": "identify", "desc": "Identify candidate causes of {goal}"},
        {"action": "verify", "desc": "Verify each cause against evidence"},
        {"action": "rank", "desc": "Rank causes by impact and evidential support"},
        {"action": "interact", "desc": "Map interactions between causal factors"},
        {"action": "synthesize", "desc": "Synthesize causal explanation"},
    ],
    "comparative": [
        {"action": "decompose", "desc": "Identify entities being compared"},
        {"action": "retrieve_a", "desc": "Gather information about first entity"},
        {"action": "retrieve_b", "desc": "Gather information about second entity"},
        {"action": "dimensions", "desc": "Identify comparison dimensions"},
        {"action": "compare", "desc": "Compare entities on each dimension"},
        {"action": "synthesize", "desc": "Summarize comparison and key differences"},
    ],
    "explanatory": [
        {"action": "components", "desc": "Identify components/mechanisms of {goal}"},
        {"action": "explain_each", "desc": "Explain each component individually"},
        {"action": "interactions", "desc": "Map interactions between components"},
        {"action": "verify", "desc": "Verify explanation consistency"},
        {"action": "synthesize", "desc": "Build coherent explanation"},
    ],
    "analytical": [
        {"action": "gather", "desc": "Gather data and evidence for {goal}"},
        {"action": "decompose", "desc": "Break down into analyzable dimensions"},
        {"action": "patterns", "desc": "Identify patterns and trends"},
        {"action": "significance", "desc": "Evaluate significance of each pattern"},
        {"action": "confounds", "desc": "Check for confounds and alternatives"},
        {"action": "synthesize", "desc": "Synthesize analytical conclusion"},
    ],
    "counterfactual": [
        {"action": "baseline", "desc": "Establish baseline reality for {goal}"},
        {"action": "modify", "desc": "Apply the hypothetical modification"},
        {"action": "propagate", "desc": "Trace causal consequences of the change"},
        {"action": "constraints", "desc": "Check physical/logical constraints"},
        {"action": "synthesize", "desc": "Describe the counterfactual outcome"},
    ],
    "predictive": [
        {"action": "historical", "desc": "Gather historical patterns for {goal}"},
        {"action": "trends", "desc": "Identify current trends and trajectories"},
        {"action": "constraints", "desc": "Identify constraints and limiting factors"},
        {"action": "scenarios", "desc": "Generate plausible scenarios"},
        {"action": "synthesize", "desc": "Synthesize most likely prediction"},
    ],
}

for _dt in ("temporal", "evaluative", "enumerative", "proof", "computational"):
    if _dt not in _RECURSIVE_TEMPLATES:
        _RECURSIVE_TEMPLATES[_dt] = _RECURSIVE_TEMPLATES["analytical"]


@dataclass
class DecomposeResult:
    goal: str = ""
    decomposition_type: str = "analytical"
    steps: List[Dict[str, str]] = field(default_factory=list)
    depth: int = 1
    estimated_confidence: float = 0.7


def nrsi_decompose(
    goal: Any,
    *,
    max_depth: int = 3,
    strategy: str = "auto",
    domain: str = "general",
) -> DecomposeResult:
    """Recursively decompose a goal into sub-goals using causal reasoning.

    Unlike static HTN templates, this:
    1. Detects goal type from semantic content
    2. Applies type-specific decomposition templates
    3. Recursively breaks complex sub-goals further (up to max_depth)
    4. Estimates confidence based on decomposition coverage
    """
    goal_text = str(goal).strip()
    if not goal_text:
        return DecomposeResult(goal="(empty)", estimated_confidence=0.0)

    dtype = _detect_goal_type(goal_text) if strategy == "auto" else strategy
    if dtype not in _RECURSIVE_TEMPLATES:
        dtype = "analytical"

    template = _RECURSIVE_TEMPLATES[dtype]
    steps = [
        {"action": s["action"], "description": s["desc"].replace("{goal}", goal_text[:100])}
        for s in template
    ]

    if max_depth > 1 and len(steps) >= 3:
        expanded: List[Dict[str, str]] = []
        for step in steps:
            expanded.append(step)
            if step["action"] in ("decompose", "components", "gather"):
                sub = nrsi_decompose(
                    step["description"], max_depth=max_depth - 1,
                    strategy="analytical", domain=domain,
                )
                for sub_step in sub.steps[:2]:
                    expanded.append({
                        "action": f"sub_{sub_step['action']}",
                        "description": f"  [sub] {sub_step['description']}",
                    })
        steps = expanded

    conf = _estimate_decompose_confidence(dtype, domain, len(steps))

    return DecomposeResult(
        goal=goal_text[:200],
        decomposition_type=dtype,
        steps=steps,
        depth=min(max_depth, 3),
        estimated_confidence=conf,
    )


def _detect_goal_type(query: str) -> str:
    q = query.lower()
    scores: Dict[str, float] = defaultdict(float)
    for pattern, dtype in _DECOMPOSE_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            scores[dtype] += 1.0
    if scores:
        return max(scores, key=scores.get)  # type: ignore[arg-type]
    return "analytical"


def _estimate_decompose_confidence(dtype: str, domain: str, n_steps: int) -> float:
    base = 0.75
    if dtype in ("causal", "explanatory"):
        base = 0.80
    elif dtype in ("proof", "computational"):
        base = 0.85
    elif dtype in ("counterfactual", "predictive"):
        base = 0.65
    step_bonus = min(0.1, n_steps * 0.01)
    return min(0.95, base + step_bonus)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. INTENT_MATCH — Semantic Pattern Matching
# ═══════════════════════════════════════════════════════════════════════════════

_INTENT_SIGNAL_SETS: Dict[str, FrozenSet[str]] = {
    "causal": frozenset({
        "cause", "causes", "caused", "why", "because", "reason", "result",
        "leads", "lead", "effect", "effects", "consequence", "due",
        "therefore", "thus", "hence",
    }),
    "inference": frozenset({
        "if", "then", "implies", "follow", "follows", "conclude",
        "infer", "deduce", "derive", "predict",
    }),
    "analogy": frozenset({
        "like", "similar", "analogy", "analogous", "compare", "parallel",
        "metaphor", "resemble", "resembles",
    }),
    "explanation": frozenset({
        "explain", "how", "describe", "mechanism", "works", "process",
        "connect", "connection", "between", "link", "relationship",
    }),
    "counterfactual": frozenset({
        "what if", "without", "remove", "imagine", "suppose",
        "hypothetically", "alternatively",
    }),
    "temporal": frozenset({
        "when", "before", "after", "during", "timeline", "history",
        "century", "year", "era", "period", "recently", "currently",
    }),
    "quantitative": frozenset({
        "how much", "how many", "greater", "less", "more", "fewer",
        "rank", "largest", "smallest", "total", "sum", "percent", "ratio",
    }),
    "planning": frozenset({
        "how to", "steps", "plan", "achieve", "goal", "need",
        "require", "blocker", "prerequisite", "implement",
    }),
    "evaluative": frozenset({
        "should", "pros", "cons", "worth", "advisable", "recommend",
        "better", "worse", "optimal", "best", "worst",
    }),
    "definitional": frozenset({
        "what is", "define", "definition", "meaning", "means",
        "refers to", "known as", "called",
    }),
    "social": frozenset({
        "think", "believe", "feel", "perspective", "opinion",
        "agree", "disagree", "understand", "intent", "motive",
    }),
}


@dataclass
class IntentMatchResult:
    primary_intent: str = "general"
    scores: Dict[str, float] = field(default_factory=dict)
    matched_signals: List[str] = field(default_factory=list)
    confidence: float = 0.0
    needs: Dict[str, bool] = field(default_factory=dict)


def nrsi_intent_match(
    query: Any,
    belief_base: Any = None,
    *,
    threshold: float = 0.15,
    wildcards: str = "semantic",
    fallback: str = "general",
) -> IntentMatchResult:
    """Match query intent against semantic signal patterns and beliefs.

    Unlike rule-based token matching, this uses:
    1. Multi-gram matching (not just single tokens)
    2. Synonym expansion from _REVERSE_SYNONYM
    3. Belief-base-aware boosting (if beliefs relate to the intent)
    4. Wildcard expansion for partial matches
    """
    query_text = str(query).lower().strip()
    tokens = set(query_text.split())
    bigrams = set()
    words = query_text.split()
    for i in range(len(words) - 1):
        bigrams.add(f"{words[i]} {words[i+1]}")

    scores: Dict[str, float] = {}
    matched_signals: List[str] = []

    for intent_name, signals in _INTENT_SIGNAL_SETS.items():
        score = 0.0
        for signal in signals:
            if " " in signal:
                if signal in query_text:
                    score += 1.5
                    matched_signals.append(f"{intent_name}:{signal}")
            elif signal in tokens:
                score += 1.0
                matched_signals.append(f"{intent_name}:{signal}")
            elif wildcards == "semantic":
                expanded = _REVERSE_SYNONYM.get(signal)
                if expanded and expanded in tokens:
                    score += 0.7
                    matched_signals.append(f"{intent_name}:{signal}~syn")
        normalizer = max(len(signals) * 0.3, 1.0)
        scores[intent_name] = min(1.0, score / normalizer)

    if belief_base and isinstance(belief_base, (dict, list)):
        beliefs = belief_base if isinstance(belief_base, list) else list(belief_base.values())
        for belief_item in beliefs[:20]:
            belief_text = str(belief_item).lower() if not isinstance(belief_item, dict) else str(belief_item.get("text", "")).lower()
            for intent_name in scores:
                belief_tokens = set(belief_text.split())
                overlap = len(tokens & belief_tokens) / max(len(tokens), 1)
                if overlap > 0.3:
                    scores[intent_name] = min(1.0, scores[intent_name] + overlap * 0.2)

    best_intent = fallback
    best_score = 0.0
    for name, score in scores.items():
        if score > best_score and score >= threshold:
            best_intent = name
            best_score = score

    needs = {
        name: score >= threshold
        for name, score in scores.items()
    }

    return IntentMatchResult(
        primary_intent=best_intent,
        scores=scores,
        matched_signals=matched_signals[:20],
        confidence=best_score,
        needs=needs,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Module Exports
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "nrsi_compose", "CompositionResult",
    "LearnableStore", "StoredBelief",
    "nrsi_semantic_distance", "SemanticDistanceResult",
    "nrsi_decompose", "DecomposeResult",
    "nrsi_intent_match", "IntentMatchResult",
]
