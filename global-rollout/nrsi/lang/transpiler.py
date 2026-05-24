"""NRSI Transpiler — AST to Python Code Generation.

Converts a type-checked NRSI AST into Python source code that uses
the nrsi.core type system (NRSIData, ValidationGate, ProcessingLobe, etc.).

Each NRSI construct maps to specific Python patterns:
  trust x: validated[string] → x = validated("...", confidence, "gate_name")
  gate verify(d: raw[T]) -> validated[T] → ValidationGate with validators
  lobe logical { processor ... } → ProcessingLobe subclass with register_processor
  norm no_medical → Norm object in NormSet
  belief base facts → BeliefBase with BeliefEntry objects
  fn name(params) -> ret → def name(params) -> ret: (with trust enforcement)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ═══════════════════════════════════════════════════════════════════════════════
# 1. AST Node Definitions
#
# These dataclasses define the abstract syntax tree that the parser produces
# and the transpiler consumes.  They live here (rather than a separate
# ast_nodes module) so the language toolchain stays self-contained until
# a dedicated parser ships.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ASTNode:
    """Base for all AST nodes — carries source location."""

    line: int = 0
    column: int = 0


# ── Type Expressions ─────────────────────────────────────────────────────────

@dataclass
class TypeExpr(ASTNode):
    """Type expression: ``validated[string]``, ``raw[T]``, ``list[int]``."""

    base: str = ""
    params: List[TypeExpr] = field(default_factory=list)

    def __repr__(self) -> str:
        if self.params:
            inner = ", ".join(repr(p) for p in self.params)
            return f"{self.base}[{inner}]"
        return self.base


@dataclass
class Param(ASTNode):
    """Function / gate parameter with optional type annotation."""

    name: str = ""
    type_expr: Optional[TypeExpr] = None
    default: Optional["Expr"] = None


# ── Expressions ──────────────────────────────────────────────────────────────

@dataclass
class Expr(ASTNode):
    """Base for all expressions."""


@dataclass
class LiteralExpr(Expr):
    """String, integer, float, boolean, or none literal."""

    value: Any = None
    literal_type: str = ""


@dataclass
class IdentExpr(Expr):
    """Identifier reference."""

    name: str = ""


@dataclass
class BinOpExpr(Expr):
    """Binary operation: ``left op right``."""

    op: str = ""
    left: Optional[Expr] = None
    right: Optional[Expr] = None


@dataclass
class UnaryExpr(Expr):
    """Unary operation: ``op operand``."""

    op: str = ""
    operand: Optional[Expr] = None


@dataclass
class CallExpr(Expr):
    """Function call: ``callee(args)``."""

    callee: Optional[Expr] = None
    args: List[Expr] = field(default_factory=list)


@dataclass
class FieldAccessExpr(Expr):
    """Field access: ``obj.field_name``."""

    obj: Optional[Expr] = None
    field_name: str = ""


@dataclass
class SliceExpr(Expr):
    """Slice: ``lower:upper``."""
    lower: Optional[Expr] = None
    upper: Optional[Expr] = None
    step: Optional[Expr] = None


@dataclass
class KeywordArg(Expr):
    """Keyword argument: ``name=value``."""
    name: str = ""
    value: Optional[Expr] = None


@dataclass
class IndexExpr(Expr):
    """Index/subscript: ``obj[index]``."""

    obj: Optional[Expr] = None
    index: Optional[Expr] = None


@dataclass
class ListExpr(Expr):
    """List literal: ``[a, b, c]``."""

    elements: List[Expr] = field(default_factory=list)


@dataclass
class DictExpr(Expr):
    """Dict literal: ``{k: v, ...}``."""

    pairs: List[Tuple[Expr, Expr]] = field(default_factory=list)


@dataclass
class MatchArm(ASTNode):
    """Single arm in a match expression."""

    pattern: Optional[Expr] = None
    body: List[ASTNode] = field(default_factory=list)


@dataclass
class MatchExpr(Expr):
    """Match expression: ``match subject { arms }``."""

    subject: Optional[Expr] = None
    arms: List[MatchArm] = field(default_factory=list)


# ── Statements ───────────────────────────────────────────────────────────────

@dataclass
class ReturnStmt(ASTNode):
    """``return expr``."""

    value: Optional[Expr] = None


@dataclass
class LetStmt(ASTNode):
    """``let name: type = expr`` or ``let mut name = expr``."""

    name: str = ""
    type_expr: Optional[TypeExpr] = None
    mutable: bool = False
    value: Optional[Expr] = None


@dataclass
class AssignStmt(ASTNode):
    """``target = value``."""

    target: str = ""
    value: Optional[Expr] = None


@dataclass
class ExprStmt(ASTNode):
    """Expression used as statement."""

    expr: Optional[Expr] = None


@dataclass
class IfStmt(ASTNode):
    """``if cond { then } else { else }``."""

    condition: Optional[Expr] = None
    then_body: List[ASTNode] = field(default_factory=list)
    else_body: List[ASTNode] = field(default_factory=list)


@dataclass
class ForStmt(ASTNode):
    """``for var in iterable { body }``."""

    var: str = ""
    iterable: Optional[Expr] = None
    body: List[ASTNode] = field(default_factory=list)


@dataclass
class Block(ASTNode):
    """Brace-delimited block: ``{ stmts }``."""

    stmts: List[ASTNode] = field(default_factory=list)


@dataclass
class WhileStmt(ASTNode):
    """``while cond { body }``."""

    condition: Optional[Expr] = None
    body: List[ASTNode] = field(default_factory=list)


@dataclass
class TryStmt(ASTNode):
    """``try { body } catch(err) { handler }``."""

    body: List[ASTNode] = field(default_factory=list)
    catch_var: str = ""
    catch_body: List[ASTNode] = field(default_factory=list)
    finally_body: List[ASTNode] = field(default_factory=list)


@dataclass
class BreakStmt(ASTNode):
    """``break``."""


@dataclass
class ContinueStmt(ASTNode):
    """``continue``."""


@dataclass
class PassStmt(ASTNode):
    """``pass``."""


@dataclass
class RaiseStmt(ASTNode):
    """``raise expr``."""

    value: Optional[Expr] = None


@dataclass
class DelStmt(ASTNode):
    """``del target``."""

    target: Expr = None


@dataclass
class AugAssignStmt(ASTNode):
    """``target += value``."""

    target: Optional[Expr] = None
    op: str = ""
    value: Optional[Expr] = None


@dataclass
class ClassDecl(ASTNode):
    """``class Name(Base) { fields; fn methods }``."""

    name: str = ""
    base_classes: List[str] = field(default_factory=list)
    fields: List[Any] = field(default_factory=list)
    methods: List[Any] = field(default_factory=list)


# ── Declarations ─────────────────────────────────────────────────────────────

@dataclass
class TrustDecl(ASTNode):
    """``trust x: validated[string] = "hello" confidence: 0.95``."""

    name: str = ""
    trust_level: str = ""
    inner_type: str = ""
    value: Optional[Expr] = None
    confidence: float = 0.0
    epistemic: str = ""
    temporal_tier: str = ""


@dataclass
class RequireClause(ASTNode):
    """``require confidence >= 0.95`` inside a gate."""

    field: str = ""
    op: str = ""
    value: float = 0.0


@dataclass
class GateDecl(ASTNode):
    """``gate verify(data: raw[string]) -> validated[string] { ... }``."""

    name: str = ""
    params: List[Param] = field(default_factory=list)
    return_type: Optional[TypeExpr] = None
    requires: List[RequireClause] = field(default_factory=list)
    validators: List[str] = field(default_factory=list)
    body: List[ASTNode] = field(default_factory=list)


@dataclass
class ProcessorDecl(ASTNode):
    """``processor forward_chain(q: string) -> validated[LogicResult] { ... }``."""

    name: str = ""
    params: List[Param] = field(default_factory=list)
    return_type: Optional[TypeExpr] = None
    body: List[ASTNode] = field(default_factory=list)


@dataclass
class LobeDecl(ASTNode):
    """``lobe logical { processor ... }``."""

    name: str = ""
    processors: List[ProcessorDecl] = field(default_factory=list)


@dataclass
class NormDecl(ASTNode):
    """``norm no_medical { type: prohibition, scope: domain("medical") }``."""

    name: str = ""
    deontic_type: str = ""
    scope: str = ""
    domain: str = ""
    priority: int = 0
    description: str = ""
    action: str = ""


@dataclass
class AxiomDecl(ASTNode):
    """``axiom "c = 299792458 m/s" tier: T1_FACT``."""

    content: str = ""
    tier: str = ""
    confidence: float = 1.0
    epistemic: str = ""


@dataclass
class BeliefBaseDecl(ASTNode):
    """``belief base physics { axiom ... }``."""

    name: str = ""
    axioms: List[AxiomDecl] = field(default_factory=list)


@dataclass
class FnDecl(ASTNode):
    """``fn process(data: trusted[Claim]) -> certified[Result] { ... }``."""

    name: str = ""
    params: List[Param] = field(default_factory=list)
    return_type: Optional[TypeExpr] = None
    body: List[ASTNode] = field(default_factory=list)
    is_export: bool = False


@dataclass
class ConstDecl(ASTNode):
    """``const PI = 3.14159``."""

    name: str = ""
    type_expr: Optional[TypeExpr] = None
    value: Optional[Expr] = None


@dataclass
class ImportDecl(ASTNode):
    """``import foo`` or ``from foo import bar as baz``."""

    module: str = ""
    names: List[Tuple[str, Optional[str]]] = field(default_factory=list)


@dataclass
class Module(ASTNode):
    """Top-level compilation unit."""

    name: str = ""
    imports: List[ASTNode] = field(default_factory=list)
    declarations: List[ASTNode] = field(default_factory=list)
    source_file: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

class DiagnosticSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


@dataclass
class Diagnostic:
    """A compiler diagnostic (error, warning, info, hint)."""

    severity: DiagnosticSeverity
    message: str
    line: int = 0
    column: int = 0
    filename: str = ""

    def __str__(self) -> str:
        loc = f"{self.filename}:" if self.filename else ""
        return f"{loc}{self.line}:{self.column}: {self.severity.name}: {self.message}"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Transpilation Lookup Tables
# ═══════════════════════════════════════════════════════════════════════════════

TRUST_CONSTRUCTORS: Dict[str, str] = {
    "raw": "raw",
    "validated": "validated",
    "trusted": "trusted",
    "certified": "certified",
}

TRUST_LEVELS: Dict[str, str] = {
    "raw": "TrustLevel.RAW",
    "validated": "TrustLevel.VALIDATED",
    "trusted": "TrustLevel.TRUSTED",
    "certified": "TrustLevel.CERTIFIED",
}

DEONTIC_TYPES: Dict[str, str] = {
    "obligation": "DeonticType.OBLIGATION",
    "permission": "DeonticType.PERMISSION",
    "prohibition": "DeonticType.PROHIBITION",
    "exemption": "DeonticType.EXEMPTION",
}

NORM_SCOPES: Dict[str, str] = {
    "global": "NormScope.GLOBAL",
    "domain": "NormScope.DOMAIN",
    "session": "NormScope.SESSION",
    "query": "NormScope.QUERY",
}

LOBE_CLASSES: Dict[str, Tuple[str, str]] = {
    "linguistic": ("LinguisticLobe", "LobeType.LINGUISTIC"),
    "logical": ("LogicalLobe", "LobeType.LOGICAL"),
    "mathematical": ("MathematicalLobe", "LobeType.MATHEMATICAL"),
    "spatial": ("SpatialLobe", "LobeType.SPATIAL"),
    "temporal": ("TemporalLobe", "LobeType.TEMPORAL"),
    "creative": ("CreativeProcessingLobe", "LobeType.CREATIVE"),
    "causal": ("ProcessingLobe", "LobeType.CAUSAL"),
    "analogical": ("ProcessingLobe", "LobeType.ANALOGICAL"),
    "planning": ("ProcessingLobe", "LobeType.PLANNING"),
    "memory": ("ProcessingLobe", "LobeType.MEMORY"),
    "metacognitive": ("ProcessingLobe", "LobeType.METACOGNITIVE"),
    "cognitive": ("ProcessingLobe", "LobeType.COGNITIVE"),
    "verification": ("ProcessingLobe", "LobeType.VERIFICATION"),
    "reasoning": ("ProcessingLobe", "LobeType.REASONING"),
    "context": ("ProcessingLobe", "LobeType.CONTEXT"),
}

TIER_ENTRENCHMENT: Dict[str, str] = {
    "T0_AXIOM": "Entrenchment.AXIOMATIC",
    "T0_axiom": "Entrenchment.AXIOMATIC",
    "T1_FACT": "Entrenchment.EMPIRICAL",
    "T2_INFERRED": "Entrenchment.INFERRED",
    "T3_HYPOTHETICAL": "Entrenchment.HYPOTHETICAL",
    "T4_SPECULATIVE": "Entrenchment.SPECULATIVE",
    "AXIOMATIC": "Entrenchment.AXIOMATIC",
    "axiomatic": "Entrenchment.AXIOMATIC",
    "EMPIRICAL": "Entrenchment.EMPIRICAL",
    "empirical": "Entrenchment.EMPIRICAL",
    "INFERRED": "Entrenchment.INFERRED",
    "HYPOTHETICAL": "Entrenchment.HYPOTHETICAL",
    "SPECULATIVE": "Entrenchment.SPECULATIVE",
}

_NRSI_IDENT_REMAP: Dict[str, str] = {
    "nil": "None",
}

BINOP_MAP: Dict[str, str] = {
    "+": "+", "-": "-", "*": "*", "/": "/", "%": "%",
    "**": "**",
    "~~": "//",
    "==": "==", "!=": "!=", "<": "<", ">": ">",
    "<=": "<=", ">=": ">=",
    "<<": "<<", ">>": ">>",
    "^": "^",
    "and": "and", "or": "or",
    "&&": "and", "||": "or",
    "in": "in", "not in": "not in",
    "is": "is", "&": "&",
}

NRSI_TYPE_MAP: Dict[str, str] = {
    "string": "str",
    "int": "int",
    "float": "float",
    "bool": "bool",
    "none": "None",
    "list": "list",
    "dict": "dict",
    "any": "Any",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Transpiler
# ═══════════════════════════════════════════════════════════════════════════════

class Transpiler:
    """Convert NRSI AST to Python source code.

    Generated code uses the nrsi.core type system:
      - NRSIData / TrustLevel / raw / validated / trusted / certified
      - ValidationGate for gate declarations
      - ProcessingLobe subclasses for lobe declarations
      - Norm / NormSet for normative declarations
      - BeliefBase / BeliefEntry for belief base declarations

    Usage::

        transpiler = Transpiler()
        python_source = transpiler.transpile(module_ast)
    """

    def __init__(self, type_checked: bool = True) -> None:
        self._type_checked = type_checked
        self._indent = 0
        self._output: List[str] = []
        self._imports_needed: Set[str] = set()
        self._exported_names: List[str] = []
        self._source_map: Dict[int, int] = {}
        self._emit_source_map = False

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def source_map(self) -> Dict[int, int]:
        return dict(self._source_map)

    def format_source_map(self) -> str:
        return json.dumps(self._source_map, sort_keys=True)

    def transpile(self, module: Module) -> str:
        """Transpile a complete NRSI module to Python source."""
        self._indent = 0
        self._output = []
        self._imports_needed = set()
        self._exported_names = []
        self._source_map = {}

        self._scan_imports(module)
        self._emit_header(module)
        self._emit_imports()
        self._emit_line("")

        for imp in getattr(module, "imports", []):
            self._transpile_decl(imp)

        if getattr(module, "imports", []):
            self._emit_line("")

        for decl in module.declarations:
            self._transpile_decl(decl)
            self._emit_line("")

        if self._exported_names:
            names = ", ".join(f'"{n}"' for n in self._exported_names)
            self._emit_line(f"__all__ = [{names}]")
            self._emit_line("")

        if self._emit_source_map:
            self._emit_line(f"# __nrsi_source_map__: {self.format_source_map()}")

        return "\n".join(self._output)

    # ── Import scanning ──────────────────────────────────────────────────

    def _scan_imports(self, module: Module) -> None:
        """Pre-scan declarations to decide which nrsi.core imports to emit.

        Uses class name matching so both transpiler-local and parser-produced
        AST nodes are detected correctly.
        """
        _IMPORT_MAP = {
            "TrustDecl": ("types",),
            "GateDecl": ("types", "validation"),
            "LobeDecl": ("lobes", "types", "validation"),
            "NormDecl": ("normative",),
            "BeliefBaseDecl": ("belief_revision",),
            "FnDecl": ("types",),
            "ComposeDecl": ("cognitive_primitives",),
            "PersistDecl": ("cognitive_primitives",),
            "EnumDecl": ("types", "enum_support"),
            "StructDecl": ("types", "dataclass_support"),
            "ClassDecl": (),
        }
        for decl in module.declarations:
            cls_name = type(decl).__name__
            if cls_name in _IMPORT_MAP:
                self._imports_needed.update(_IMPORT_MAP[cls_name])

    # ── Emit helpers ─────────────────────────────────────────────────────

    def _emit_line(self, line: str, nrsi_line: int = 0) -> None:
        if not line.strip():
            self._output.append("")
        else:
            self._output.append(("    " * self._indent) + line)
        if nrsi_line > 0:
            self._source_map[len(self._output)] = nrsi_line

    def _emit_lines(self, lines: List[str]) -> None:
        for line in lines:
            self._emit_line(line)

    def _indent_up(self) -> None:
        self._indent += 1

    def _indent_down(self) -> None:
        self._indent = max(0, self._indent - 1)

    # ── Header / imports ─────────────────────────────────────────────────

    def _emit_header(self, module: Module) -> None:
        raw_name = module.name or module.source_file or "<nrsi>"
        safe_name = raw_name.split("\n")[0][:80].replace("\\", "\\\\").replace('"""', "'''")
        self._emit_line(f'"""Auto-generated Python from NRSI module: {safe_name}')
        self._emit_line("")
        self._emit_line("Generated by the NRSI transpiler. Do not edit directly.")
        self._emit_line('Trust enforcement is active at runtime."""')
        self._emit_line("")

    def _emit_imports(self) -> None:
        self._emit_line("from __future__ import annotations")
        self._emit_line("")

        stdlib_imports: List[str] = []
        if "enum_support" in self._imports_needed:
            stdlib_imports.append("from enum import Enum")
        if "dataclass_support" in self._imports_needed:
            stdlib_imports.append("from dataclasses import dataclass, field")
            stdlib_imports.append("from typing import Any, Dict, List, Optional, Set, Tuple")
        for line in stdlib_imports:
            self._emit_line(line)
        if stdlib_imports:
            self._emit_line("")

        if "types" in self._imports_needed:
            self._emit_line("from nrsi.core.types import (")
            self._indent_up()
            self._emit_line("NRSIData, TrustLevel, Confidence,")
            self._emit_line("raw, validated, trusted, certified,")
            self._indent_down()
            self._emit_line(")")

        if "validation" in self._imports_needed:
            self._emit_line("from nrsi.core.validation import (")
            self._indent_up()
            self._emit_line("ValidationGate, ValidationResult, FunctionValidator,")
            self._indent_down()
            self._emit_line(")")

        if "lobes" in self._imports_needed:
            self._emit_line("from nrsi.core.lobes import (")
            self._indent_up()
            self._emit_line("ProcessingLobe, LobeType, LobeResult,")
            self._emit_line("LinguisticLobe, LogicalLobe, MathematicalLobe,")
            self._emit_line("SpatialLobe, TemporalLobe, CreativeProcessingLobe,")
            self._indent_down()
            self._emit_line(")")

        if "normative" in self._imports_needed:
            self._emit_line("from nrsi.core.normative import (")
            self._indent_up()
            self._emit_line("Norm, NormSet, DeonticType, NormScope,")
            self._indent_down()
            self._emit_line(")")

        if "belief_revision" in self._imports_needed:
            self._emit_line("from nrsi.core.belief_revision import (")
            self._indent_up()
            self._emit_line("BeliefBase, BeliefEntry, Entrenchment, RevisionPolicy,")
            self._indent_down()
            self._emit_line(")")

        if "cognitive_primitives" in self._imports_needed:
            self._emit_line("from nrsi.lang.cognitive_primitives import (")
            self._indent_up()
            self._emit_line("nrsi_compose, nrsi_semantic_distance,")
            self._emit_line("nrsi_decompose, nrsi_intent_match,")
            self._emit_line("LearnableStore,")
            self._indent_down()
            self._emit_line(")")

    # ── Declaration dispatch ─────────────────────────────────────────────

    def _transpile_decl(self, decl: ASTNode) -> None:
        """Route a top-level declaration to the appropriate handler."""
        _DISPATCH = {
            TrustDecl: self._transpile_trust_decl,
            GateDecl: self._transpile_gate_decl,
            LobeDecl: self._transpile_lobe_decl,
            NormDecl: self._transpile_norm_decl,
            BeliefBaseDecl: self._transpile_belief_base,
            FnDecl: self._transpile_fn,
            ConstDecl: self._transpile_const,
            ImportDecl: self._transpile_import,
            "ComposeDecl": self._transpile_compose_decl,
            "PersistDecl": self._transpile_persist_decl,
            "EnumDecl": self._transpile_enum_decl,
            "StructDecl": self._transpile_struct_decl,
            "ClassDecl": self._transpile_class_decl,
        }
        handler = _DISPATCH.get(type(decl))
        if handler is not None:
            handler(decl)
        elif isinstance(decl, (LetStmt, AssignStmt, ExprStmt)):
            self._transpile_stmt(decl)
        else:
            name_match = type(decl).__name__
            if name_match in ("LetStmt", "AssignStmt", "ExprStmt"):
                self._transpile_stmt(decl)
                return
            for cls, h in _DISPATCH.items():
                key_name = cls if isinstance(cls, str) else cls.__name__
                if key_name == name_match:
                    h(decl)
                    return
            self._emit_line(
                f"pass  # [transpiler] unsupported: {type(decl).__name__}"
            )

    # ── trust ────────────────────────────────────────────────────────────

    def _transpile_trust_decl(self, decl: TrustDecl) -> None:
        """``trust x: validated[string] = "hello" confidence: 0.95``

        Emits::

            x = validated("hello", 0.95, "trust_decl_x")
        """
        constructor = TRUST_CONSTRUCTORS.get(decl.trust_level, "raw")
        value_code = self._transpile_expr(decl.value) if decl.value else "None"
        ann = getattr(decl, "annotations", {}) or {}
        conf_node = ann.get("confidence")
        conf = float(getattr(conf_node, "value", 0)) if conf_node else getattr(decl, "confidence", 0.0) or 0.0
        gate_label = f"trust_decl_{decl.name}"

        if constructor == "raw":
            self._emit_line(f"{decl.name} = raw({value_code})")
        elif constructor == "certified":
            self._emit_line(
                f"{decl.name} = certified("
                f"{value_code}, {conf}, {gate_label!r}, \"nrsi_transpiled\")"
            )
        else:
            self._emit_line(
                f"{decl.name} = {constructor}("
                f"{value_code}, {conf}, {gate_label!r})"
            )

        ep_node = ann.get("epistemic")
        epistemic = getattr(ep_node, "name", None) or getattr(decl, "epistemic", None)
        if epistemic:
            self._emit_line(
                f'{decl.name}._metadata["epistemic"] = {epistemic!r}'
            )
        temporal = getattr(decl, "temporal_tier", None)
        if temporal:
            self._emit_line(
                f'{decl.name}._metadata["temporal_tier"] = {temporal!r}'
            )
        self._exported_names.append(decl.name)

    # ── gate ─────────────────────────────────────────────────────────────

    def _transpile_gate_decl(self, decl) -> None:
        """Transpile gate declaration from either transpiler-native or parser AST.

        All ``require`` conditions are now emitted as runtime assertions
        inside the gate function, not just ``confidence >= X``.  This
        ensures domain guards, epistemic checks, and other predicates
        authored in NRSI are actually enforced.
        """
        gate_var = f"_gate_{decl.name}"
        conf_threshold = 0.95
        validators = []
        extra_require_exprs: list[str] = []

        requires = getattr(decl, "requires", None)
        if requires is not None:
            for req in requires:
                if getattr(req, "field", "") == "confidence":
                    conf_threshold = req.value
            validators = getattr(decl, "validators", []) or []
        else:
            for stmt in (getattr(decl, "body", None) or []):
                stype = type(stmt).__name__
                if stype == "RequireStmt" and getattr(stmt, "condition", None):
                    cond = stmt.condition
                    if (type(cond).__name__ == "BinaryOp"
                            and getattr(cond, "op", "") in (">=", ">")
                            and getattr(getattr(cond, "left", None), "name", "") == "confidence"
                            and getattr(getattr(cond, "right", None), "value", None) is not None):
                            conf_threshold = cond.right.value
                    else:
                        expr_code = self._transpile_expr(cond)
                        if expr_code and expr_code not in ("True", "true", "None"):
                            extra_require_exprs.append(expr_code)
                elif stype == "ValidateStmt":
                    validators.append(getattr(stmt, "validator", ""))

        target_trust = "TrustLevel.VALIDATED"
        ret = getattr(decl, "return_type", None)
        if ret:
            base = getattr(ret, "trust_level", None) or getattr(ret, "base", None)
            if base and base in TRUST_LEVELS:
                target_trust = TRUST_LEVELS[base]

        validator_list = ", ".join(repr(v) for v in validators if v)

        self._emit_line(f"{gate_var} = ValidationGate(")
        self._indent_up()
        self._emit_line(f'name={decl.name!r},')
        self._emit_line(f"confidence_threshold={conf_threshold},")
        self._emit_line(f"validators=[{validator_list}],")
        self._emit_line(f"target_trust={target_trust},")
        self._indent_down()
        self._emit_line(")")
        self._emit_line("")

        params = getattr(decl, "params", []) or []
        param_names = ", ".join(p.name for p in params)
        self._emit_line(f"def {decl.name}({param_names}):")
        self._indent_up()
        if extra_require_exprs:
            first_param = params[0].name if params else "None"
            self._emit_line(f"_data = {first_param}")
            for expr_code in extra_require_exprs:
                safe_code = expr_code.replace("data.", "_data.")
                self._emit_line(
                    f"if not ({safe_code}):"
                )
                self._indent_up()
                self._emit_line(
                    f'raise ValueError("Gate {decl.name!r} require failed: {expr_code}")'
                )
                self._indent_down()
        first_param = params[0].name if params else "None"
        self._emit_line(f"return {gate_var}.process({first_param})")
        self._indent_down()
        self._exported_names.append(decl.name)

    # ── lobe ─────────────────────────────────────────────────────────────

    def _transpile_lobe_decl(self, decl: LobeDecl) -> None:
        """``lobe logical { processor forward_chain(...) { ... } }``

        Emits a ProcessingLobe subclass with registered processors and
        instantiates a module-level singleton.
        """
        lobe_info = LOBE_CLASSES.get(decl.name)
        if lobe_info is None:
            self._emit_line(
                f"# [transpiler] unknown lobe type: {decl.name!r}"
            )
            return

        base_class, _lobe_type = lobe_info
        class_name = f"_{decl.name.capitalize()}LobeExt"

        self._emit_line(f"class {class_name}({base_class}):")
        self._indent_up()
        self._emit_line(
            f'"""Extended {decl.name} lobe with NRSI-defined processors."""'
        )
        self._emit_line("")

        # __init__
        self._emit_line("def __init__(self):")
        self._indent_up()
        if base_class == "ProcessingLobe":
            self._emit_line(f"super().__init__({_lobe_type})")
        else:
            self._emit_line("super().__init__()")
        for proc in decl.processors:
            self._emit_line(f"self.register_processor(self._{proc.name})")
        self._indent_down()
        self._emit_line("")

        # processor methods
        for proc in decl.processors:
            self._transpile_processor(proc)

        self._indent_down()

        instance_name = f"{decl.name}_lobe"
        self._emit_line(f"{instance_name} = {class_name}()")
        self._exported_names.append(instance_name)

    def _transpile_processor(self, proc: ProcessorDecl) -> None:
        """Emit a processor method inside a lobe class body."""
        self._emit_line(
            f"def _{proc.name}(self, query, domain=None, context=None):"
        )
        self._indent_up()

        if proc.body:
            for stmt in proc.body:
                self._transpile_stmt(stmt)
        else:
            self._emit_line("return {")
            self._indent_up()
            self._emit_line('"value": None,')
            self._emit_line('"confidence": 0.5,')
            self._emit_line(f'"metadata": {{"processor": {proc.name!r}}},')
            self._indent_down()
            self._emit_line("}")

        self._indent_down()
        self._emit_line("")

    # ── norm ─────────────────────────────────────────────────────────────

    def _transpile_norm_decl(self, decl) -> None:
        """Transpile norm declaration from either transpiler-native or parser AST.

        The condition expression from the NRSI source is compiled into a
        real Python lambda so that norm enforcement actually evaluates the
        authored predicate at runtime, not a blanket ``True``.
        """
        var_name = f"_norm_{decl.name}"
        deontic_type = getattr(decl, "deontic_type", "prohibition")
        deontic = DEONTIC_TYPES.get(deontic_type, "DeonticType.PROHIBITION")
        scope_val = getattr(decl, "scope", "global")
        scope = NORM_SCOPES.get(scope_val, "NormScope.GLOBAL")
        domain = getattr(decl, "domain", "")
        priority = getattr(decl, "priority", 0)
        description = getattr(decl, "description", decl.name)
        action = getattr(decl, "action", "")

        cond_expr = getattr(decl, "condition", None)
        if cond_expr is not None:
            cond_code = self._transpile_expr(cond_expr)
            if cond_code in ("True", "true"):
                condition = "lambda ctx: True"
            elif cond_code in ("False", "false"):
                condition = "lambda ctx: False"
            else:
                condition = f"lambda ctx: {cond_code}"
        elif domain:
            condition = f'lambda ctx: ctx.get("domain") == {domain!r}'
        else:
            condition = "lambda ctx: True"

        self._emit_line(f"{var_name} = Norm(")
        self._indent_up()
        self._emit_line(f"norm_id={decl.name!r},")
        self._emit_line(f"deontic_type={deontic},")
        self._emit_line(f"description={description!r},")
        self._emit_line(f"condition={condition},")
        self._emit_line(f"scope={scope},")
        self._emit_line(f"priority={priority},")
        if action:
            self._emit_line(f"action={action!r},")
        if domain:
            self._emit_line(f"domain={domain!r},")
        self._indent_down()
        self._emit_line(")")
        self._exported_names.append(var_name)

    # ── belief base ──────────────────────────────────────────────────────

    def _transpile_belief_base(self, decl: BeliefBaseDecl) -> None:
        """``belief base physics { axiom "c = 299792458 m/s" tier: T1_FACT }``

        Emits::

            _belief_base_physics = BeliefBase()
            _belief_base_physics.add(BeliefEntry(belief_id="physics_0", ...))
        """
        var_name = f"_belief_base_{decl.name}"
        self._emit_line(f"{var_name} = BeliefBase()")

        for idx, axiom in enumerate(decl.axioms):
            entry_id = f"{decl.name}_{idx}"
            entrenchment = TIER_ENTRENCHMENT.get(
                axiom.tier, "Entrenchment.HYPOTHETICAL"
            )
            epistemic = getattr(axiom, "epistemic", None) or "observational"
            confidence = getattr(axiom, "confidence", 0.9)

            self._emit_line(f"{var_name}.add(BeliefEntry(")
            self._indent_up()
            self._emit_line(f"belief_id={entry_id!r},")
            self._emit_line(f"content={axiom.content!r},")
            self._emit_line(f"entrenchment={entrenchment},")
            self._emit_line(f"confidence={confidence},")
            self._emit_line(f"epistemic_type={epistemic!r},")
            self._indent_down()
            self._emit_line("))")

        self._exported_names.append(var_name)

    # ── fn ───────────────────────────────────────────────────────────────

    def _transpile_fn(self, decl) -> None:
        """Transpile function declaration from either transpiler-native or parser AST."""
        params = getattr(decl, "params", []) or []
        param_parts: List[str] = []
        for p in params:
            prefix = ""
            if getattr(p, 'is_kw_variadic', False):
                prefix = "**"
            elif getattr(p, 'is_variadic', False):
                prefix = "*"
            default = getattr(p, "default", None)
            if default is not None:
                param_parts.append(f"{prefix}{p.name}={self._transpile_expr(default)}")
            else:
                param_parts.append(f"{prefix}{p.name}")
        param_names = ", ".join(param_parts)
        ret_annotation = ""
        if decl.return_type:
            py_type = self._resolve_type(decl.return_type)
            ret_annotation = f" -> {py_type}"

        for deco in (getattr(decl, 'decorators', None) or []):
            deco_str = self._transpile_expr(deco)
            self._emit_line(f"@{deco_str}")

        async_prefix = "async " if getattr(decl, 'is_async', False) else ""
        self._emit_line(f"{async_prefix}def {decl.name}({param_names}){ret_annotation}:")
        self._indent_up()

        for p in params:
            te = getattr(p, "type_expr", None)
            base = getattr(te, "base", None) or getattr(te, "trust_level", None)
            if base and base in TRUST_LEVELS:
                trust_py = TRUST_LEVELS[base]
                self._emit_line(f"if isinstance({p.name}, NRSIData):")
                self._indent_up()
                self._emit_line(
                    f'{p.name}.require_trust('
                    f'{trust_py}, "{decl.name}.{p.name}")'
                )
                self._indent_down()

        body = getattr(decl, "body", []) or []
        if body:
            for stmt in body:
                self._transpile_stmt(stmt)
        else:
            self._emit_line("pass")

        self._indent_down()

        if getattr(decl, "is_export", False):
            self._exported_names.append(decl.name)

    # ── const ────────────────────────────────────────────────────────────

    def _transpile_const(self, decl: ConstDecl) -> None:
        """``const PI = 3.14159`` → ``PI = 3.14159``."""
        val = self._transpile_expr(decl.value) if decl.value else "None"
        self._emit_line(f"{decl.name} = {val}")
        self._exported_names.append(decl.name)

    # ── compose ──────────────────────────────────────────────────────────

    def _transpile_compose_decl(self, decl) -> None:
        """``compose answer from facts, web { strategy: "synthesis" }``

        Emits::
            answer = nrsi_compose(["facts", "web"], strategy="synthesis", ...)
        """
        sources = getattr(decl, "sources", [])
        src_parts: List[str] = []
        for s in sources:
            name = getattr(s, "name", None)
            if name:
                src_parts.append(repr(name))
            else:
                src_parts.append(self._transpile_expr(s))
        src_list = ", ".join(src_parts)
        ann = getattr(decl, "annotations", {}) or {}
        kw_parts: List[str] = []
        for k, v in ann.items():
            kw_parts.append(f"{k}={self._transpile_expr(v)}")
        kw = ", ".join(kw_parts)
        args = f"[{src_list}]"
        if kw:
            args += f", {kw}"
        self._emit_line(f"{decl.name} = nrsi_compose({args})")

    # ── persist ─────────────────────────────────────────────────────────

    def _transpile_persist_decl(self, decl) -> None:
        """``persist store { decay: 0.01 }``

        Emits::
            store = LearnableStore(decay=0.01, ...)
        """
        config = getattr(decl, "config", {}) or {}
        kw_parts: List[str] = []
        for k, v in config.items():
            kw_parts.append(f"{k}={self._transpile_expr(v)}")
        kw = ", ".join(kw_parts)
        self._emit_line(f"{decl.name} = LearnableStore({kw})")

    # ── import ───────────────────────────────────────────────────────────

    def _transpile_import(self, decl) -> None:
        mod_path = getattr(decl, "module", None) or getattr(decl, "module_path", "") or ""
        _NRSI_MODULE_MAP = {
            "nrsi.stdlib.types": "nrsi.core.types",
            "nrsi.stdlib.validation": "nrsi.core.validation",
            "nrsi.stdlib.normative": "nrsi.core.normative",
            "nrsi.stdlib.lobes": "nrsi.core.lobes",
            "nrsi.stdlib.beliefs": "nrsi.core.belief_revision",
            "nrsi.stdlib.memory": "nrsi.core.memory",
            "nrsi.stdlib.cognitive": "nrsi.lang.cognitive_primitives",
            "types": "nrsi.stdlib.types",
            "gates": "nrsi.stdlib.gates",
            "norms": "nrsi.stdlib.norms",
            "lobes": "nrsi.stdlib.lobes",
            "beliefs": "nrsi.stdlib.beliefs",
            "memory": "nrsi.core.memory",
            "cognitive": "nrsi.lang.cognitive_primitives",
            "validation": "nrsi.core.validation",
            "normative": "nrsi.core.normative",
            "reasoning": "nrsi.stdlib.reasoning",
            "thinking": "nrsi.stdlib.thinking",
            "verification": "nrsi.stdlib.verification",
            "tools": "nrsi.stdlib.tools",
            "context": "nrsi.stdlib.context",
            "cognitive_loop": "nrsi.stdlib.cognitive_loop",
        }
        _STDLIB_REMAP = {
            "hash": "hashlib", "regex": "re", "os_path": "os.path",
            "json_lib": "json", "random_lib": "random",
        }
        py_mod = _NRSI_MODULE_MAP.get(mod_path, mod_path)
        if py_mod.startswith("std/"):
            py_mod = py_mod[4:]
        py_mod = py_mod.replace("/", ".")
        py_mod = _STDLIB_REMAP.get(py_mod, py_mod)
        names = getattr(decl, "names", []) or []
        alias = getattr(decl, "alias", "")
        if py_mod == "logging" and names == ["log"]:
            self._emit_line("import logging")
            self._emit_line("log = logging.getLogger(__name__)")
            return
        if names and len(names) == 1 and names[0] == py_mod:
            self._emit_line(f"import {py_mod}")
            return
        if names:
            parts = []
            for item in names:
                if isinstance(item, tuple):
                    name, al = item
                    parts.append(f"{name} as {al}" if al else name)
                else:
                    parts.append(str(item))
            self._emit_line(f"from {py_mod} import {', '.join(parts)}")
        elif alias:
            self._emit_line(f"import {py_mod} as {alias}")
        else:
            self._emit_line(f"import {py_mod}")

    # ── enum ─────────────────────────────────────────────────────────────

    def _transpile_enum_decl(self, decl) -> None:
        """``enum TrustLevel { RAW, VALIDATED, TRUSTED, CERTIFIED }``

        Emits a Python ``Enum`` subclass::

            class TrustLevel(Enum):
                RAW = "raw"
                VALIDATED = "validated"
        """
        name = getattr(decl, "name", "UnknownEnum")
        variants = getattr(decl, "variants", []) or getattr(decl, "members", [])
        variant_values = getattr(decl, "variant_values", {}) or {}
        self._emit_line(f"class {name}(Enum):")
        self._indent_up()
        if not variants:
            self._emit_line("pass")
        else:
            for v in variants:
                if isinstance(v, str):
                    vname = v
                    if vname in variant_values:
                        vval_expr = variant_values[vname]
                        self._emit_line(f'{vname} = {self._transpile_expr(vval_expr)}')
                        continue
                    vval = v.lower()
                elif hasattr(v, "name"):
                    vname = v.name
                    vval = getattr(v, "value", v.name.lower())
                    if hasattr(vval, "value"):
                        vval = vval.value
                else:
                    vname = str(v)
                    vval = str(v).lower()
                self._emit_line(f'{vname} = {vval!r}')
        self._indent_down()
        self._exported_names.append(name)

    # ── struct ───────────────────────────────────────────────────────────

    def _transpile_struct_decl(self, decl) -> None:
        """``type NRSIData { value: any, confidence: float, metadata: dict }``

        Emits a ``@dataclass`` class::

            @dataclass
            class NRSIData:
                value: Any = None
                confidence: float = 0.0
                metadata: dict = field(default_factory=dict)
        """
        name = getattr(decl, "name", "UnknownStruct")
        fields = getattr(decl, "fields", []) or getattr(decl, "members", [])
        self._emit_line("@dataclass")
        self._emit_line(f"class {name}:")
        self._indent_up()
        if not fields:
            self._emit_line("pass")
        else:
            for f in fields:
                fname = getattr(f, "name", str(f))
                ftype_node = getattr(f, "type_expr", None) or getattr(f, "type", None)
                if ftype_node:
                    py_type = self._resolve_type(ftype_node)
                else:
                    py_type = "Any"

                default = getattr(f, "default", None)
                if default is not None:
                    try:
                        dval = self._transpile_expr(default)
                    except Exception:
                        dval = repr(default)
                    self._emit_line(f"{fname}: {py_type} = {dval}")
                elif py_type in ("dict", "Dict", "Map") or py_type.startswith("Dict[") or py_type.startswith("Map["):
                    self._emit_line(f"{fname}: {py_type} = field(default_factory=dict)")
                elif py_type in ("list", "List") or py_type.startswith("List[") or py_type.startswith("list[") or py_type.startswith("Set["):
                    self._emit_line(f"{fname}: {py_type} = field(default_factory=list)")
                else:
                    self._emit_line(f"{fname}: {py_type} = None")
        self._indent_down()
        self._exported_names.append(name)

    # ── Body helper ────────────────────────────────────────────────────

    def _transpile_body(self, body) -> None:
        """Transpile a list of statements, emitting ``pass`` if empty."""
        if body:
            for s in body:
                self._transpile_stmt(s)
        else:
            self._emit_line("pass")

    # ── Statement transpilation ──────────────────────────────────────────

    def _transpile_stmt(self, stmt) -> None:
        """Convert a statement AST node to Python line(s)."""
        stype = type(stmt).__name__
        nl = getattr(stmt, "line", 0) or 0

        if stype == "ReturnStmt":
            if getattr(stmt, "value", None):
                self._emit_line(
                    f"return {self._transpile_expr(stmt.value)}", nrsi_line=nl
                )
            else:
                self._emit_line("return", nrsi_line=nl)

        elif stype == "LetStmt":
            val = self._transpile_expr(stmt.value) if getattr(stmt, "value", None) else "None"
            self._emit_line(f"{stmt.name} = {val}", nrsi_line=nl)

        elif stype == "AssignStmt":
            val = self._transpile_expr(stmt.value) if getattr(stmt, "value", None) else "None"
            target = getattr(stmt, "target", "x")
            if hasattr(target, "object") or hasattr(target, "obj"):
                target = self._transpile_expr(target)
            elif hasattr(target, "name"):
                target = target.name
            elif not isinstance(target, str):
                target = self._transpile_expr(target)
            self._emit_line(f"{target} = {val}", nrsi_line=nl)

        elif stype == "ExprStmt":
            expr = getattr(stmt, "expr", None)
            if expr:
                lt = getattr(expr, "literal_type", "")
                if lt == "export_list":
                    for n in getattr(expr, "value", []):
                        if n not in self._exported_names:
                            self._exported_names.append(n)
                else:
                    self._emit_line(self._transpile_expr(expr), nrsi_line=nl)

        elif stype in ("IfStmt", "IfExpr"):
            self._transpile_if_stmt(stmt)

        elif stype == "ForStmt":
            self._transpile_for_stmt(stmt)

        elif stype == "Block":
            for s in getattr(stmt, "stmts", []):
                self._transpile_stmt(s)

        elif stype == "RequireStmt":
            cond = getattr(stmt, "condition", None)
            if cond:
                self._emit_line(
                    f"assert {self._transpile_expr(cond)}", nrsi_line=nl
                )

        elif stype == "ValidateStmt":
            vname = getattr(stmt, "validator", "validator")
            self._emit_line(f"{vname}(data)", nrsi_line=nl)

        elif stype == "WhileStmt":
            self._transpile_while_stmt(stmt)

        elif stype == "TryStmt":
            self._transpile_try_stmt(stmt)

        elif stype == "BreakStmt":
            self._emit_line("break", nrsi_line=nl)

        elif stype == "ContinueStmt":
            self._emit_line("continue", nrsi_line=nl)

        elif stype == "PassStmt":
            self._emit_line("pass", nrsi_line=nl)

        elif stype == "RaiseStmt":
            expr_val = getattr(stmt, "expr", None) or getattr(stmt, "value", None)
            if expr_val:
                expr_code = self._transpile_expr(expr_val)
                cause = getattr(stmt, 'cause', None)
                if cause:
                    cause_code = self._transpile_expr(cause)
                    self._emit_line(
                        f"raise {expr_code} from {cause_code}", nrsi_line=nl
                    )
                else:
                    self._emit_line(f"raise {expr_code}", nrsi_line=nl)
            else:
                self._emit_line("raise", nrsi_line=nl)

        elif stype == "DelStmt":
            target = getattr(stmt, "target", None)
            if target:
                self._emit_line(
                    f"del {self._transpile_expr(target)}", nrsi_line=nl
                )
            else:
                self._emit_line("del", nrsi_line=nl)

        elif stype == "ImportDecl":
            self._transpile_import(stmt)

        elif stype == "AugAssignStmt":
            _AUG_OP_MAP = {
                "**=": "**=",
                "~~=": "//=",
                "&=": "&=",
                "|=": "|=",
                "^=": "^=",
                "<<=": "<<=",
                ">>=": ">>=",
            }
            target = getattr(stmt, "target", None)
            if target:
                t_code = self._transpile_expr(target) if hasattr(target, "name") or hasattr(target, "object") else str(target)
            else:
                t_code = "x"
            val = self._transpile_expr(getattr(stmt, "value", None))
            op = getattr(stmt, "op", "+=")
            op = _AUG_OP_MAP.get(op, op)
            self._emit_line(f"{t_code} {op} {val}", nrsi_line=nl)

        elif stype == "MatchExpr":
            self._transpile_match_stmt(stmt)
            return

        elif stype == "WithStmt":
            ctx = self._transpile_expr(stmt.context)
            if stmt.alias:
                self._emit_line(
                    f"with {ctx} as {stmt.alias}:", nrsi_line=nl
                )
            else:
                self._emit_line(f"with {ctx}:", nrsi_line=nl)
            self._indent_up()
            self._transpile_body(stmt.body)
            self._indent_down()
            return

        elif stype == "AssertStmt":
            cond = self._transpile_expr(stmt.condition)
            if stmt.message:
                msg = self._transpile_expr(stmt.message)
                self._emit_line(f"assert {cond}, {msg}", nrsi_line=nl)
            else:
                self._emit_line(f"assert {cond}", nrsi_line=nl)
            return

        elif stype == "GlobalStmt":
            keyword = "nonlocal" if getattr(stmt, 'is_nonlocal', False) else "global"
            self._emit_line(
                f"{keyword} {', '.join(stmt.names)}", nrsi_line=nl
            )
            return

        elif stype == "FnDecl":
            self._transpile_fn(stmt)
            return

        elif stype == "ClassDecl":
            self._transpile_class_decl(stmt)
            return

        elif stype == "ImportDecl":
            self._transpile_import(stmt)
            return

        else:
            self._emit_line(
                f"pass  # [transpiler] unsupported stmt: {stype}",
                nrsi_line=nl,
            )

    def _transpile_if_stmt(self, stmt: IfStmt) -> None:
        nl = getattr(stmt, "line", 0) or 0
        cond = self._transpile_expr(stmt.condition) if stmt.condition else "True"
        self._emit_line(f"if {cond}:", nrsi_line=nl)
        self._indent_up()
        if stmt.then_body:
            for s in stmt.then_body:
                self._transpile_stmt(s)
        else:
            self._emit_line("pass", nrsi_line=nl)
        self._indent_down()

        if stmt.else_body:
            self._emit_line("else:", nrsi_line=nl)
            self._indent_up()
            for s in stmt.else_body:
                self._transpile_stmt(s)
            self._indent_down()

    def _transpile_for_stmt(self, stmt: ForStmt) -> None:
        nl = getattr(stmt, "line", 0) or 0
        iter_code = self._transpile_expr(stmt.iterable) if stmt.iterable else "[]"
        var_name = getattr(stmt, "var", None) or getattr(stmt, "variable", "item")
        self._emit_line(f"for {var_name} in {iter_code}:", nrsi_line=nl)
        self._indent_up()
        if stmt.body:
            for s in stmt.body:
                self._transpile_stmt(s)
        else:
            self._emit_line("pass", nrsi_line=nl)
        self._indent_down()

    def _transpile_match_stmt(self, expr: MatchExpr) -> None:
        """Emit a match expression as an if/elif chain."""
        nl = getattr(expr, "line", 0) or 0
        subj = self._transpile_expr(expr.subject)
        tmp = "_match_subj"
        self._emit_line(f"{tmp} = {subj}", nrsi_line=nl)

        for i, arm in enumerate(expr.arms):
            pat = self._transpile_expr(arm.pattern)
            keyword = "if" if i == 0 else "elif"
            self._emit_line(f"{keyword} {tmp} == {pat}:", nrsi_line=nl)
            self._indent_up()
            if arm.body:
                for s in arm.body:
                    self._transpile_stmt(s)
            else:
                self._emit_line("pass", nrsi_line=nl)
            self._indent_down()

        if expr.arms:
            self._emit_line("else:", nrsi_line=nl)
            self._indent_up()
            self._emit_line("pass", nrsi_line=nl)
            self._indent_down()

    # ── while / try / class transpilation ────────────────────────────────

    def _transpile_while_stmt(self, stmt) -> None:
        nl = getattr(stmt, "line", 0) or 0
        cond = self._transpile_expr(getattr(stmt, "condition", None)) or "True"
        self._emit_line(f"while {cond}:", nrsi_line=nl)
        self._indent_up()
        body = getattr(stmt, "body", [])
        if body:
            for s in body:
                self._transpile_stmt(s)
        else:
            self._emit_line("pass", nrsi_line=nl)
        self._indent_down()

    def _transpile_try_stmt(self, stmt) -> None:
        nl = getattr(stmt, "line", 0) or 0
        self._emit_line("try:", nrsi_line=nl)
        self._indent_up()
        body = getattr(stmt, "body", [])
        if body:
            for s in body:
                self._transpile_stmt(s)
        else:
            self._emit_line("pass", nrsi_line=nl)
        self._indent_down()
        catch_body = getattr(stmt, "catch_body", [])
        catch_var = getattr(stmt, "catch_var", "")
        if catch_body or not getattr(stmt, "finally_body", []):
            exc = f" as {catch_var}" if catch_var else ""
            self._emit_line(f"except Exception{exc}:", nrsi_line=nl)
            self._indent_up()
            if catch_body:
                for s in catch_body:
                    self._transpile_stmt(s)
            else:
                self._emit_line("pass", nrsi_line=nl)
            self._indent_down()
        finally_body = getattr(stmt, "finally_body", [])
        if finally_body:
            self._emit_line("finally:", nrsi_line=nl)
            self._indent_up()
            for s in finally_body:
                self._transpile_stmt(s)
            self._indent_down()

    def _transpile_class_decl(self, decl) -> None:
        name = getattr(decl, "name", "UnknownClass")
        bases = getattr(decl, "base_classes", [])
        fields = getattr(decl, "fields", [])
        methods = getattr(decl, "methods", [])
        for deco in (getattr(decl, 'decorators', None) or []):
            deco_str = self._transpile_expr(deco)
            self._emit_line(f"@{deco_str}")
        base_str = f"({', '.join(bases)})" if bases else ""
        self._emit_line(f"class {name}{base_str}:")
        self._indent_up()
        if not fields and not methods:
            self._emit_line("pass")
        else:
            for f in fields:
                fname = getattr(f, "name", str(f))
                ftype_node = getattr(f, "type_expr", None) or getattr(f, "type", None)
                if ftype_node:
                    py_type = self._resolve_type(ftype_node)
                else:
                    py_type = "Any"
                default = getattr(f, "default", None)
                if default is not None:
                    try:
                        dval = self._transpile_expr(default)
                    except Exception:
                        dval = repr(default)
                    self._emit_line(f"{fname}: {py_type} = {dval}")
                else:
                    self._emit_line(f"{fname}: {py_type} = None")
            if fields:
                self._emit_line("")
            has_explicit_init = False
            has_explicit_dunder_init = False
            for method in methods:
                mtype = type(method).__name__
                if mtype == "FnDecl":
                    mname = getattr(method, "name", "")
                    mparams = getattr(method, "params", []) or []
                    m_has_self = mparams and getattr(mparams[0], "name", "") == "self"
                    if mname == "init" and m_has_self:
                        has_explicit_init = True
                    if mname == "__init__" and m_has_self:
                        has_explicit_dunder_init = True
            if has_explicit_init and not has_explicit_dunder_init:
                self._emit_line("def __init__(self, *_a, **_kw):")
                self._indent_up()
                self._emit_line("self.init(*_a, **_kw)")
                self._indent_down()
            self._emit_line("")
            for method in methods:
                mtype = type(method).__name__
                if mtype == "FnDecl":
                    self._transpile_fn(method)
                    self._emit_line("")
                else:
                    self._transpile_stmt(method)
        self._indent_down()
        self._exported_names.append(name)

    # ── Expression transpilation ─────────────────────────────────────────

    def _transpile_expr(self, expr) -> str:
        """Convert an expression AST node to a Python code string."""
        if expr is None:
            return "None"

        etype = type(expr).__name__

        if etype in ("LiteralExpr", "Literal"):
            value = getattr(expr, "value", None)
            if isinstance(value, str) and value.startswith("\x00f\x00"):
                content = value[3:]
                return f'f"{content}"'
            return self._transpile_literal(expr)
        if etype in ("IdentExpr", "Identifier"):
            name = getattr(expr, "name", "None")
            return _NRSI_IDENT_REMAP.get(name, name)
        if etype in ("BinOpExpr", "BinaryOp"):
            return self._transpile_binop(expr)
        if etype in ("UnaryExpr", "UnaryOp"):
            return self._transpile_unary(expr)
        if etype == "CallExpr":
            return self._transpile_call(expr)
        if etype in ("FieldAccessExpr", "MemberAccess"):
            obj = getattr(expr, "obj", None) or getattr(expr, "object", None)
            field = getattr(expr, "field_name", None) or getattr(expr, "member", "")
            return f"{self._transpile_expr(obj)}.{field}"
        if etype == "IndexExpr":
            obj = getattr(expr, "obj", None) or getattr(expr, "object", None)
            idx = getattr(expr, "index", None)
            if idx and type(idx).__name__ == "SliceExpr":
                lo = self._transpile_expr(idx.lower) if getattr(idx, "lower", None) else ""
                hi = self._transpile_expr(idx.upper) if getattr(idx, "upper", None) else ""
                return f"{self._transpile_expr(obj)}[{lo}:{hi}]"
            return f"{self._transpile_expr(obj)}[{self._transpile_expr(idx)}]"
        if etype == "SliceExpr":
            lo = self._transpile_expr(expr.lower) if getattr(expr, "lower", None) else ""
            hi = self._transpile_expr(expr.upper) if getattr(expr, "upper", None) else ""
            return f"{lo}:{hi}"
        if etype == "KeywordArg":
            return f"{expr.name}={self._transpile_expr(expr.value)}"
        if etype == "ListExpr":
            elems = ", ".join(self._transpile_expr(e) for e in expr.elements)
            return f"[{elems}]"
        if etype == "DictExpr":
            pairs = ", ".join(
                f"{self._transpile_expr(k)}: {self._transpile_expr(v)}"
                for k, v in expr.pairs
            )
            return "{" + pairs + "}"
        if etype == "MatchExpr":
            return self._transpile_match_inline(expr)
        if etype == "SemanticDistanceExpr":
            left = self._transpile_expr(getattr(expr, "left", None))
            right = self._transpile_expr(getattr(expr, "right", None))
            ann = getattr(expr, "annotations", {}) or {}
            kw = ", ".join(f"{k}={self._transpile_expr(v)}" for k, v in ann.items())
            args = f"{left}, {right}"
            if kw:
                args += f", {kw}"
            return f"nrsi_semantic_distance({args})"
        if etype == "DecomposeExpr":
            goal = self._transpile_expr(getattr(expr, "goal", None))
            ann = getattr(expr, "annotations", {}) or {}
            kw = ", ".join(f"{k}={self._transpile_expr(v)}" for k, v in ann.items())
            args = goal
            if kw:
                args += f", {kw}"
            return f"nrsi_decompose({args})"
        if etype == "IntentMatchExpr":
            q = self._transpile_expr(getattr(expr, "query_expr", None))
            bb = self._transpile_expr(getattr(expr, "belief_base", None))
            ann = getattr(expr, "annotations", {}) or {}
            kw = ", ".join(f"{k}={self._transpile_expr(v)}" for k, v in ann.items())
            args = f"{q}, {bb}"
            if kw:
                args += f", {kw}"
            return f"nrsi_intent_match({args})"

        if etype == "FnDecl":
            params = getattr(expr, "params", [])
            body = getattr(expr, "body", [])
            pnames = []
            for p in params:
                pname = getattr(p, "name", str(p))
                pnames.append(pname)
            param_str = ", ".join(pnames)
            if len(body) == 1 and type(body[0]).__name__ == "ReturnStmt":
                ret_val = getattr(body[0], "value", None)
                if ret_val:
                    return f"lambda {param_str}: {self._transpile_expr(ret_val)}"
            fn_name = f"_anon_{id(expr)}"
            el = getattr(expr, "line", 0) or 0
            self._emit_line(f"def {fn_name}({param_str}):", nrsi_line=el)
            self._indent_up()
            if body:
                for s in body:
                    self._transpile_stmt(s)
            else:
                self._emit_line("pass", nrsi_line=el)
            self._indent_down()
            return fn_name

        if etype == "YieldExpr":
            if expr.is_from:
                val = self._transpile_expr(expr.value) if expr.value else ""
                return f"(yield from {val})"
            if expr.value:
                return f"(yield {self._transpile_expr(expr.value)})"
            return "(yield)"

        if etype == "AwaitExpr":
            return f"(await {self._transpile_expr(expr.value)})"

        if etype == "ComprehensionExpr":
            var = expr.variable
            it = self._transpile_expr(expr.iterable)
            cond = ""
            if expr.condition:
                cond = f" if {self._transpile_expr(expr.condition)}"
            if expr.kind == "dict":
                k = self._transpile_expr(expr.key)
                v = self._transpile_expr(expr.value)
                return "{" + f"{k}: {v} for {var} in {it}{cond}" + "}"
            elif expr.kind == "set":
                elem = self._transpile_expr(expr.element)
                return "{" + f"{elem} for {var} in {it}{cond}" + "}"
            elif expr.kind == "generator":
                elem = self._transpile_expr(expr.element)
                return f"({elem} for {var} in {it}{cond})"
            else:  # list
                elem = self._transpile_expr(expr.element)
                return f"[{elem} for {var} in {it}{cond}]"

        if etype == "TernaryExpr":
            then = self._transpile_expr(expr.then_expr)
            cond = self._transpile_expr(expr.condition)
            els = self._transpile_expr(expr.else_expr)
            return f"({then} if {cond} else {els})"

        if etype == "LambdaExpr":
            params = []
            for p in expr.params:
                if hasattr(p, 'default') and p.default is not None:
                    params.append(f"{p.name}={self._transpile_expr(p.default)}")
                else:
                    params.append(p.name)
            body = self._transpile_expr(expr.body)
            return f"(lambda {', '.join(params)}: {body})"

        return f"None  # unsupported expr: {etype}"

    def _transpile_literal(self, expr: LiteralExpr) -> str:
        if expr.literal_type == "string":
            return repr(str(expr.value))
        if expr.literal_type == "int":
            val = str(expr.value)
            if val.startswith(("0x", "0X", "0o", "0O", "0b", "0B")):
                return val
            return str(int(val.replace("_", "")))
        if expr.literal_type == "float":
            return str(float(expr.value))
        if expr.literal_type == "bool":
            return "True" if expr.value else "False"
        if expr.literal_type == "none":
            return "None"
        return repr(expr.value)

    def _transpile_binop(self, expr: BinOpExpr) -> str:
        op = BINOP_MAP.get(expr.op, expr.op)
        if op == "is" and type(expr.right).__name__ in ("UnaryExpr", "UnaryOp"):
            inner = expr.right
            if getattr(inner, "op", "") in ("not", "!"):
                left = self._transpile_expr(expr.left)
                operand = self._transpile_expr(inner.operand)
                return f"({left} is not {operand})"
        left = self._transpile_expr(expr.left)
        right = self._transpile_expr(expr.right)
        return f"({left} {op} {right})"

    def _transpile_unary(self, expr: UnaryExpr) -> str:
        operand = self._transpile_expr(expr.operand)
        if expr.op in ("not", "!"):
            return f"(not {operand})"
        if expr.op == "~":
            return f"(~{operand})"
        return f"({expr.op}{operand})"

    _METHOD_TO_BUILTIN = {"len": "len", "keys": None, "values": None, "items": None}

    def _transpile_call(self, expr: CallExpr) -> str:
        callee = self._transpile_expr(expr.callee)

        if callee.startswith("super."):
            method = callee[6:]
            args_str = ", ".join(self._transpile_expr(a) for a in expr.args)
            return f"super().{method}({args_str})"

        if callee.endswith(".len") and not expr.args:
            obj_code = callee[:-4]
            return f"len({obj_code})"

        if callee.endswith(".slice"):
            obj_code = callee[:-6]
            args = [self._transpile_expr(a) for a in expr.args]
            if len(args) == 2:
                return f"{obj_code}[{args[0]}:{args[1]}]"
            if len(args) == 1:
                return f"{obj_code}[{args[0]}:]"
            return f"{obj_code}[:]"

        if callee.endswith(".sort_by") and len(expr.args) == 1:
            obj_code = callee[:-8]
            arg = self._transpile_expr(expr.args[0])
            return f"{obj_code}.sort(key={arg})"

        _NRSI_METHOD_REMAP = {
            "starts_with": "startswith", "ends_with": "endswith",
            "to_upper": "upper", "to_lower": "lower",
            "trim": "strip", "trim_start": "lstrip", "trim_end": "rstrip",
            "char_at": "__getitem__", "index_of": "index",
            "contains": "__contains__",
            "remove": "pop",
        }
        _remapped_method = None
        for nrsi_m, py_m in _NRSI_METHOD_REMAP.items():
            if callee.endswith("." + nrsi_m):
                callee = callee[:-(len(nrsi_m))] + py_m
                _remapped_method = nrsi_m
                break

        if _remapped_method == "remove" and len(expr.args) == 1:
            obj_code = callee[:-(len("pop") + 1)]
            arg = self._transpile_expr(expr.args[0])
            return f"{obj_code}.pop({arg}, None)"

        if callee == "List":
            elems = ", ".join(self._transpile_expr(a) for a in expr.args)
            return f"[{elems}]"
        if callee in ("__tuple__", "Tuple"):
            elems = ", ".join(self._transpile_expr(a) for a in expr.args)
            return f"({elems},)" if len(expr.args) == 1 else f"({elems})"
        if callee in ("Set", "Set.of"):
            elems = ", ".join(self._transpile_expr(a) for a in expr.args)
            return "{" + elems + "}"
        if callee in ("Set.from", "set", "Set.new"):
            args_str = ", ".join(self._transpile_expr(a) for a in expr.args)
            return f"set({args_str})"
        if callee == "Dict.new" or callee == "dict":
            args_str = ", ".join(self._transpile_expr(a) for a in expr.args)
            return f"dict({args_str})"
        parts: List[str] = []
        for a in expr.args:
            atype = type(a).__name__
            if atype == "SpreadExpr":
                prefix = "**" if a.is_double else "*"
                parts.append(f"{prefix}{self._transpile_expr(a.value)}")
            elif atype == "KeywordArg":
                parts.append(f"{a.name}={self._transpile_expr(a.value)}")
            elif atype in ("BinOpExpr", "BinaryOp") and getattr(a, "op", "") == ":":
                key = getattr(getattr(a, "left", None), "name", self._transpile_expr(a.left))
                val = self._transpile_expr(a.right)
                parts.append(f"{key}={val}")
            else:
                parts.append(self._transpile_expr(a))
        return f"{callee}({', '.join(parts)})"

    def _transpile_match_inline(self, expr: MatchExpr) -> str:
        """Inline a match expression as a chained conditional expression."""
        subj = self._transpile_expr(expr.subject)
        parts: List[str] = []
        for arm in expr.arms:
            pat = self._transpile_expr(arm.pattern)
            body_code = "None"
            if arm.body:
                last = arm.body[-1]
                if isinstance(last, ReturnStmt) and last.value:
                    body_code = self._transpile_expr(last.value)
                elif isinstance(last, ExprStmt) and last.expr:
                    body_code = self._transpile_expr(last.expr)
            parts.append(f"{body_code} if ({subj}) == ({pat})")
        if parts:
            return f"({' else '.join(parts)} else None)"
        return "None"

    # ── Type resolution ──────────────────────────────────────────────────

    def _resolve_type(self, te) -> str:
        """Convert an NRSI TypeExpr to a Python type-annotation string."""
        base = getattr(te, "base", None) or getattr(te, "name", None) or getattr(te, "trust_level", None) or "Any"
        inner_type = getattr(te, "inner_type", None)
        params = getattr(te, "params", None) or getattr(te, "type_args", None) or []

        mapped_base = NRSI_TYPE_MAP.get(base, base)
        if base in TRUST_CONSTRUCTORS:
            if inner_type:
                inner = self._resolve_type(inner_type)
                return f"NRSIData[{inner}]"
            if params:
                inner = self._resolve_type(params[0])
                return f"NRSIData[{inner}]"
            return "NRSIData"
        if params:
            inner = ", ".join(self._resolve_type(p) for p in params)
            return f"{mapped_base}[{inner}]"
        return mapped_base


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Module-level convenience API
# ═══════════════════════════════════════════════════════════════════════════════

def transpile(module: Module, *, type_checked: bool = True) -> str:
    """Transpile an NRSI ``Module`` AST to Python source code."""
    return Transpiler(type_checked=type_checked).transpile(module)


def transpile_declaration(decl: ASTNode) -> str:
    """Transpile a single declaration (for REPL / incremental use)."""
    wrapper = Module(declarations=[decl])
    t = Transpiler()
    t._scan_imports(wrapper)
    t._output = []
    t._transpile_decl(decl)
    return "\n".join(t._output)


__all__ = [
    # AST nodes
    "ASTNode", "TypeExpr", "Param",
    "Expr", "LiteralExpr", "IdentExpr", "BinOpExpr", "UnaryExpr",
    "CallExpr", "FieldAccessExpr", "IndexExpr", "ListExpr", "DictExpr",
    "MatchArm", "MatchExpr",
    "ReturnStmt", "LetStmt", "AssignStmt", "ExprStmt",
    "IfStmt", "ForStmt", "Block",
    "TrustDecl", "RequireClause", "GateDecl", "ProcessorDecl", "LobeDecl",
    "NormDecl", "AxiomDecl", "BeliefBaseDecl", "FnDecl", "ConstDecl",
    "ImportDecl", "Module",
    # Diagnostics
    "DiagnosticSeverity", "Diagnostic",
    # Transpiler
    "Transpiler", "transpile", "transpile_declaration",
    # Lookup tables
    "TRUST_LEVELS", "TRUST_CONSTRUCTORS", "LOBE_CLASSES",
    "DEONTIC_TYPES", "NORM_SCOPES", "TIER_ENTRENCHMENT",
]
