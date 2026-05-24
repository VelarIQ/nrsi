"""NRSI Explanation Structure Types — Machine-Checkable Reasoning Traces.

Explanations aren't just strings — they're structured DAGs of reasoning
steps that can be verified, compared, and presented at different levels.

  ExplanationNode   — Single step in an explanation
  ExplanationDAG    — Full explanation graph (acyclic)
  ContrastiveExplanation — Why P instead of Q?
  FeatureAttribution — Which inputs most influenced the output
  ExplanationLevel  — Same explanation at different granularity
  TeachingScaffold  — Explanation structured for learning

Patent-covered: NRSI Explanation Graph System, VelarIQ.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Enums
# ═══════════════════════════════════════════════════════════════════════════════

class ExplanationType(Enum):
    """What kind of explanatory step this is."""

    CAUSAL = auto()        # X because Y
    MECHANISTIC = auto()   # X works by Y
    FUNCTIONAL = auto()    # X exists to do Y
    TELEOLOGICAL = auto()  # X's purpose is Y
    CONTRASTIVE = auto()   # X not Y because Z
    STATISTICAL = auto()   # X correlates with Y
    ANALOGICAL = auto()    # X like Y
    DEDUCTIVE = auto()     # X follows from Y by logic


class EdgeRelation(Enum):
    """Semantic relationship along an explanation edge."""

    SUPPORTS = auto()
    BECAUSE = auto()
    DESPITE = auto()
    INSTEAD_OF = auto()
    ANALOGOUS_TO = auto()
    IMPLIES = auto()


class ExplanationLevel(Enum):
    """Granularity at which an explanation is rendered."""

    EXPERT = auto()        # Full formal detail
    INTERMEDIATE = auto()  # Key steps only
    SIMPLE = auto()        # One-sentence summary
    ELI5 = auto()          # Explain like I'm 5


class AttributionMethod(Enum):
    """Algorithm used to compute feature attribution."""

    SHAP = auto()
    LIME = auto()
    ATTENTION = auto()
    GRADIENT = auto()
    COUNTERFACTUAL = auto()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ExplanationNode / ExplanationEdge
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExplanationNode:
    """A single step in a structured explanation."""

    node_id: str
    content: str
    explanation_type: ExplanationType
    premises: List[str] = field(default_factory=list)
    confidence: float = 1.0
    lobe_source: Optional[str] = None
    evidence_ids: List[str] = field(default_factory=list)
    depth_level: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence must be in [0.0, 1.0], got {self.confidence}"
            )


@dataclass(frozen=True)
class ExplanationEdge:
    """Directed edge in an explanation DAG."""

    from_node: str
    to_node: str
    relation: EdgeRelation


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ExplanationDAG — the full reasoning graph
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DiffResult:
    """Difference between two explanation DAGs."""

    only_in_a: List[str] = field(default_factory=list)
    only_in_b: List[str] = field(default_factory=list)
    shared: List[str] = field(default_factory=list)
    structural_diff: str = ""


class ExplanationDAG:
    """Directed acyclic graph of reasoning steps.

    Nodes are ``ExplanationNode`` instances keyed by ``node_id``.
    Edges carry semantic relations (supports, because, despite, …).
    """

    def __init__(
        self,
        query_being_explained: str = "",
        root_node_id: Optional[str] = None,
    ) -> None:
        self.query_being_explained: str = query_being_explained
        self.root_node_id: Optional[str] = root_node_id
        self._nodes: Dict[str, ExplanationNode] = {}
        self._edges: List[ExplanationEdge] = []
        self._adjacency: Dict[str, List[str]] = {}
        self._reverse: Dict[str, List[str]] = {}

    # ── graph construction ─────────────────────────────────────────────────

    def add_node(self, node: ExplanationNode) -> None:
        self._nodes[node.node_id] = node
        self._adjacency.setdefault(node.node_id, [])
        self._reverse.setdefault(node.node_id, [])
        if self.root_node_id is None:
            self.root_node_id = node.node_id

    def add_edge(self, edge: ExplanationEdge) -> None:
        if edge.from_node not in self._nodes:
            raise KeyError(f"Source node '{edge.from_node}' not in DAG")
        if edge.to_node not in self._nodes:
            raise KeyError(f"Target node '{edge.to_node}' not in DAG")
        self._edges.append(edge)
        self._adjacency[edge.from_node].append(edge.to_node)
        self._reverse[edge.to_node].append(edge.from_node)

    @property
    def nodes(self) -> Dict[str, ExplanationNode]:
        return dict(self._nodes)

    @property
    def edges(self) -> List[ExplanationEdge]:
        return list(self._edges)

    # ── structural queries ─────────────────────────────────────────────────

    def verify_acyclic(self) -> bool:
        """Return True if the graph is acyclic (valid DAG)."""
        visited: Set[str] = set()
        in_stack: Set[str] = set()

        def _dfs(nid: str) -> bool:
            visited.add(nid)
            in_stack.add(nid)
            for child in self._adjacency.get(nid, []):
                if child in in_stack:
                    return False
                if child not in visited and not _dfs(child):
                    return False
            in_stack.discard(nid)
            return True

        for nid in self._nodes:
            if nid not in visited:
                if not _dfs(nid):
                    return False
        return True

    def depth(self) -> int:
        """Longest path from root to any leaf."""
        if not self._nodes or self.root_node_id is None:
            return 0

        depths: Dict[str, int] = {}

        topo = self._topological_order()
        for nid in topo:
            parents = self._reverse.get(nid, [])
            if not parents:
                depths[nid] = 0
            else:
                depths[nid] = max(depths.get(p, 0) for p in parents) + 1
        return max(depths.values()) if depths else 0

    def critical_path(self) -> List[ExplanationNode]:
        """Nodes on the longest path from root to any leaf."""
        if not self._nodes or self.root_node_id is None:
            return []

        distances: Dict[str, int] = {}
        predecessors: Dict[str, Optional[str]] = {}
        topo = self._topological_order()

        for nid in topo:
            distances[nid] = 0
            predecessors[nid] = None

        for nid in topo:
            for child in self._adjacency.get(nid, []):
                new_dist = distances[nid] + 1
                if new_dist > distances.get(child, 0):
                    distances[child] = new_dist
                    predecessors[child] = nid

        if not distances:
            return []
        farthest = max(distances, key=lambda k: distances[k])
        path: List[str] = []
        cur: Optional[str] = farthest
        while cur is not None:
            path.append(cur)
            cur = predecessors.get(cur)
        path.reverse()
        return [self._nodes[nid] for nid in path if nid in self._nodes]

    def weakest_link(self) -> Optional[ExplanationNode]:
        """Node with lowest confidence on the critical path."""
        cp = self.critical_path()
        if not cp:
            return None
        return min(cp, key=lambda n: n.confidence)

    def counterfactual_sensitivity(self) -> List[ExplanationNode]:
        """Premises whose removal changes reachability from root to leaves.

        Returns nodes that are single points of failure — articulation
        points along the forward graph.
        """
        if not self._nodes or self.root_node_id is None:
            return []

        leaves = {
            nid for nid, children in self._adjacency.items()
            if not children and nid in self._nodes
        }
        if not leaves:
            return []

        def _reachable_leaves(exclude: Optional[str] = None) -> Set[str]:
            visited: Set[str] = set()
            queue = deque([self.root_node_id])
            while queue:
                nid = queue.popleft()
                if nid is None or nid in visited or nid == exclude:
                    continue
                visited.add(nid)
                for child in self._adjacency.get(nid, []):
                    queue.append(child)
            return visited & leaves

        baseline = _reachable_leaves()
        sensitive: List[ExplanationNode] = []
        for nid in list(self._nodes):
            if nid == self.root_node_id:
                continue
            if _reachable_leaves(exclude=nid) != baseline:
                sensitive.append(self._nodes[nid])
        return sensitive

    # ── rendering ──────────────────────────────────────────────────────────

    def to_natural_language(self, max_depth: Optional[int] = None) -> str:
        """Render the DAG as readable prose."""
        if not self._nodes:
            return "(empty explanation)"

        lines: List[str] = []
        if self.query_being_explained:
            lines.append(f"Q: {self.query_being_explained}")
            lines.append("")

        visited: Set[str] = set()

        def _render(nid: str, depth: int) -> None:
            if nid in visited:
                return
            if max_depth is not None and depth > max_depth:
                return
            visited.add(nid)
            node = self._nodes.get(nid)
            if node is None:
                return
            indent = "  " * depth
            conf = f" [{node.confidence:.0%}]" if node.confidence < 1.0 else ""
            lines.append(f"{indent}• {node.content}{conf}")
            for child in self._adjacency.get(nid, []):
                _render(child, depth + 1)

        if self.root_node_id and self.root_node_id in self._nodes:
            _render(self.root_node_id, 0)
        else:
            for nid in self._nodes:
                _render(nid, 0)

        return "\n".join(lines)

    def to_formal(self, fmt: str = "sexp") -> str:
        """Render as machine-checkable format (S-expression by default)."""
        lines: List[str] = []
        for edge in self._edges:
            src = self._nodes.get(edge.from_node)
            tgt = self._nodes.get(edge.to_node)
            if src and tgt:
                lines.append(
                    f"({edge.relation.name} "
                    f"\"{src.content}\" "
                    f"\"{tgt.content}\")"
                )
        return "\n".join(lines) if lines else "(empty)"

    # ── internals ──────────────────────────────────────────────────────────

    def _topological_order(self) -> List[str]:
        """Kahn's algorithm — returns nodes in topological order."""
        in_degree: Dict[str, int] = {nid: 0 for nid in self._nodes}
        for edge in self._edges:
            in_degree[edge.to_node] = in_degree.get(edge.to_node, 0) + 1

        queue = deque(nid for nid, d in in_degree.items() if d == 0)
        order: List[str] = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for child in self._adjacency.get(nid, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return order


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ContrastiveExplanation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ContrastiveExplanation:
    """Why did P happen instead of Q?"""

    fact: str
    foil: str
    difference_factors: List[str] = field(default_factory=list)
    explanation: str = ""

    def as_sentence(self) -> str:
        factors = ", ".join(self.difference_factors) if self.difference_factors else "unknown factors"
        return (
            f"'{self.fact}' occurred instead of '{self.foil}' "
            f"because of {factors}."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FeatureAttribution
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FeatureAttribution:
    """Which input features most influenced the output."""

    input_features: Dict[str, float]
    output: str
    attribution_method: AttributionMethod = AttributionMethod.SHAP

    @property
    def top_k_features(self) -> List[Tuple[str, float]]:
        """Features sorted by absolute weight descending (top-5 default)."""
        return sorted(
            self.input_features.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )[:5]

    def as_text(self) -> str:
        top = self.top_k_features
        parts = [f"{name} ({weight:+.3f})" for name, weight in top]
        return (
            f"Output '{self.output}' driven by: {', '.join(parts)} "
            f"(method: {self.attribution_method.name})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TeachingScaffold — multi-level presentation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TeachingScaffold:
    """An explanation structured for progressive learning.

    Stores renderings at multiple ``ExplanationLevel`` granularities
    so the UI can let users drill in or zoom out.
    """

    topic: str
    levels: Dict[ExplanationLevel, str] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)
    follow_ups: List[str] = field(default_factory=list)

    def at_level(self, level: ExplanationLevel) -> str:
        return self.levels.get(level, "(not available at this level)")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ExplanationRenderer — render / compare / verify
# ═══════════════════════════════════════════════════════════════════════════════

class ExplanationRenderer:
    """Utility class: render DAGs at requested granularity, diff them,
    and verify structural soundness.
    """

    # ── render ─────────────────────────────────────────────────────────────

    @staticmethod
    def render(dag: ExplanationDAG, level: ExplanationLevel = ExplanationLevel.EXPERT) -> str:
        depth_map = {
            ExplanationLevel.EXPERT: None,
            ExplanationLevel.INTERMEDIATE: 3,
            ExplanationLevel.SIMPLE: 1,
            ExplanationLevel.ELI5: 0,
        }
        max_depth = depth_map.get(level)

        if level is ExplanationLevel.ELI5:
            root = dag._nodes.get(dag.root_node_id or "")
            if root:
                return f"Simply put: {root.content}"
            return "(no explanation available)"

        return dag.to_natural_language(max_depth=max_depth)

    # ── compare ────────────────────────────────────────────────────────────

    @staticmethod
    def compare(dag_a: ExplanationDAG, dag_b: ExplanationDAG) -> DiffResult:
        ids_a = set(dag_a._nodes.keys())
        ids_b = set(dag_b._nodes.keys())

        shared = ids_a & ids_b
        only_a = ids_a - ids_b
        only_b = ids_b - ids_a

        structural = ""
        if only_a or only_b:
            parts: List[str] = []
            if only_a:
                parts.append(f"{len(only_a)} node(s) only in A")
            if only_b:
                parts.append(f"{len(only_b)} node(s) only in B")
            structural = "; ".join(parts)
        else:
            edge_set_a = {(e.from_node, e.to_node, e.relation) for e in dag_a._edges}
            edge_set_b = {(e.from_node, e.to_node, e.relation) for e in dag_b._edges}
            if edge_set_a != edge_set_b:
                structural = "Same nodes but different edge structure"
            else:
                structural = "Structurally identical"

        return DiffResult(
            only_in_a=sorted(only_a),
            only_in_b=sorted(only_b),
            shared=sorted(shared),
            structural_diff=structural,
        )

    # ── verify ─────────────────────────────────────────────────────────────

    @staticmethod
    def verify(dag: ExplanationDAG) -> List[str]:
        """Return a list of structural issues (empty = sound)."""
        issues: List[str] = []

        if not dag._nodes:
            issues.append("DAG is empty (no nodes)")
            return issues

        if dag.root_node_id is None:
            issues.append("No root node set")
        elif dag.root_node_id not in dag._nodes:
            issues.append(f"Root node '{dag.root_node_id}' not found in nodes")

        if not dag.verify_acyclic():
            issues.append("Graph contains a cycle (not a valid DAG)")

        node_ids = set(dag._nodes.keys())
        for edge in dag._edges:
            if edge.from_node not in node_ids:
                issues.append(f"Edge source '{edge.from_node}' not in nodes")
            if edge.to_node not in node_ids:
                issues.append(f"Edge target '{edge.to_node}' not in nodes")

        for node in dag._nodes.values():
            for pid in node.premises:
                if pid not in node_ids:
                    issues.append(
                        f"Node '{node.node_id}' references premise '{pid}' "
                        f"which is not in the DAG"
                    )

        leaves = [
            nid for nid in dag._nodes
            if not dag._adjacency.get(nid)
        ]
        for lid in leaves:
            leaf = dag._nodes[lid]
            if not leaf.evidence_ids and not leaf.premises:
                issues.append(
                    f"Leaf node '{lid}' has no evidence or premises "
                    f"(unsupported conclusion)"
                )

        return issues


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Facade — API expected by nrsi.core.nrs._process_inner
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExplanationResult:
    """Result from ``ExplanationBuilder.build()``."""
    explanation_type: str = ""
    depth: int = 0
    text: str = ""
    dag: Optional[ExplanationDAG] = None


class ExplanationBuilder:
    """Facade providing the ``build(query, answer, mode, domain)`` API
    that ``nrsi.core.nrs._process_inner`` expects.
    """

    _MODE_TYPE_MAP: Dict[str, ExplanationType] = {
        "DETERMINISTIC": ExplanationType.DEDUCTIVE,
        "CREATIVE": ExplanationType.ANALOGICAL,
        "HYBRID": ExplanationType.MECHANISTIC,
        "ANALYTICAL": ExplanationType.CAUSAL,
    }

    def build(
        self,
        *,
        query: str,
        answer: str,
        mode: str = "HYBRID",
        domain: str = "general",
    ) -> ExplanationResult:
        """Build a structured explanation DAG and return a summary."""
        etype = self._MODE_TYPE_MAP.get(mode, ExplanationType.MECHANISTIC)

        dag = ExplanationDAG(query_being_explained=query)

        root = ExplanationNode(
            node_id="root",
            content=f"Answer to: {query[:128]}",
            explanation_type=etype,
            confidence=0.9,
            lobe_source=domain,
            depth_level=0,
        )
        dag.add_node(root)

        evidence_node = ExplanationNode(
            node_id="evidence",
            content=answer[:256],
            explanation_type=etype,
            confidence=0.85,
            lobe_source=domain,
            depth_level=1,
            evidence_ids=["pipeline_result"],
        )
        dag.add_node(evidence_node)
        dag.add_edge(ExplanationEdge(
            from_node="root",
            to_node="evidence",
            relation=EdgeRelation.BECAUSE,
        ))

        return ExplanationResult(
            explanation_type=etype.name,
            depth=dag.depth(),
            text=dag.to_natural_language(max_depth=3),
            dag=dag,
        )
