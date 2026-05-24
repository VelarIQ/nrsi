"""NRSI Type Checker — Static Trust and Epistemic Type Enforcement.

Enforces at compile time (AST walk, before transpilation):
1. Trust level flow: raw → validated → trusted → certified (never down without explicit downgrade)
2. Gate requirements: raw data MUST pass through a gate before being used as validated/trusted
3. Epistemic consistency: deductive claims need proof, causal claims need chains
4. Norm compliance: prohibited actions are flagged
5. Type compatibility: function params/returns match declarations
6. Scope rules: variables must be declared before use
7. Belief base consistency: no contradictions within a base
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from nrsi.lang.parser import (
    ASTNode, Module, TrustDecl, GateDecl, LobeDecl, ProcessorDecl,
    NormDecl, BeliefBaseDecl, StructDecl, EnumDecl, FnDecl,
    ImportDecl, LetStmt, ReturnStmt, AssignStmt, ForStmt, ExprStmt,
    RequireStmt, ValidateStmt, AxiomDecl,
    TypeExpr, SimpleType, TrustType, GenericType, UnionType, FunctionType,
    Expr, Literal, Identifier, BinaryOp, UnaryOp, CallExpr,
    MemberAccess, IndexExpr, IfExpr, MatchExpr, Param,
    ComposeDecl, PersistDecl,
    SemanticDistanceExpr, DecomposeExpr, IntentMatchExpr,
    ClassDecl, AugAssignStmt, RaiseStmt, WhileStmt, TryStmt,
    WithStmt, AssertStmt, GlobalStmt, DelStmt, BreakStmt, ContinueStmt,
    PassStmt,
)

logger = logging.getLogger("nrsi.lang.typechecker")


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

class DiagnosticSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


@dataclass
class Diagnostic:
    severity: DiagnosticSeverity
    message: str
    line: int
    column: int
    rule: str
    suggestion: str = ""

    def __str__(self) -> str:
        tag = self.severity.name
        loc = f"{self.line}:{self.column}"
        parts = [f"[{tag}] {loc} ({self.rule}) {self.message}"]
        if self.suggestion:
            parts.append(f"  → {self.suggestion}")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Trust Rank Constants
# ═══════════════════════════════════════════════════════════════════════════════

_TRUST_RANKS: Dict[str, int] = {
    "raw": 0,
    "validated": 1,
    "trusted": 2,
    "certified": 3,
}

_TRUST_NAMES = list(_TRUST_RANKS.keys())

_EPISTEMIC_VALID: Set[str] = {
    "deductive", "inductive", "abductive", "analogical",
    "causal", "computational", "observational", "testimonial",
    "creative", "speculative",
}

_LOBE_EPISTEMIC_AFFINITY: Dict[str, Set[str]] = {
    "logical": {"deductive", "abductive"},
    "mathematical": {"computational", "deductive"},
    "linguistic": {"testimonial", "observational", "inductive"},
    "temporal": {"causal", "observational"},
    "spatial": {"analogical", "observational"},
    "creative": {"creative", "speculative", "analogical"},
}


# ═══════════════════════════════════════════════════════════════════════════════
# NRSIType — Internal Type Representation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NRSIType:
    """Internal representation of an NRSI type for checking."""

    base_name: str
    trust_level: Optional[str] = None
    epistemic: Optional[str] = None
    generic_args: List[NRSIType] = field(default_factory=list)
    is_function: bool = False
    param_types: List[NRSIType] = field(default_factory=list)
    return_type: Optional[NRSIType] = None

    def trust_rank(self) -> int:
        if self.trust_level is None:
            return -1
        return _TRUST_RANKS.get(self.trust_level, -1)

    def is_trust_compatible(self, required: NRSIType) -> bool:
        """Can this type be used where *required* is expected?

        Trust flows up only: a ``trusted[X]`` value satisfies a
        ``validated[X]`` parameter, but not vice-versa.  When neither
        side has a trust annotation the check is vacuously true.
        """
        if required.trust_level is None:
            return True
        if self.trust_level is None:
            return required.trust_level is None
        return self.trust_rank() >= required.trust_rank()

    def is_base_compatible(self, other: NRSIType) -> bool:
        if self.base_name == other.base_name:
            return True
        if self.base_name in ("any", "unknown") or other.base_name in ("any", "unknown"):
            return True
        if self.base_name == "never":
            return True
        return False

    def is_assignable_to(self, target: NRSIType) -> bool:
        """Full compatibility: base + trust + generics."""
        if self.base_name in ("unknown", "any") or target.base_name in ("unknown", "any"):
            return True
        if not self.is_base_compatible(target):
            return False
        if not self.is_trust_compatible(target):
            return False
        if target.generic_args and self.generic_args:
            if len(self.generic_args) != len(target.generic_args):
                return False
            for s_arg, t_arg in zip(self.generic_args, target.generic_args):
                if not s_arg.is_assignable_to(t_arg):
                    return False
        return True

    def display(self) -> str:
        base = self.base_name
        if self.generic_args:
            args = ", ".join(a.display() for a in self.generic_args)
            base = f"{base}<{args}>"
        if self.trust_level:
            base = f"{self.trust_level}[{base}]"
        if self.epistemic:
            base = f"{base}@{self.epistemic}"
        return base

    def __repr__(self) -> str:
        return f"NRSIType({self.display()})"


UNKNOWN_TYPE = NRSIType(base_name="unknown")
VOID_TYPE = NRSIType(base_name="void")
BOOL_TYPE = NRSIType(base_name="bool")
INT_TYPE = NRSIType(base_name="int")
FLOAT_TYPE = NRSIType(base_name="float")
STRING_TYPE = NRSIType(base_name="string")
ANY_TYPE = NRSIType(base_name="any")
NEVER_TYPE = NRSIType(base_name="never")

_BUILTIN_TYPES: Dict[str, NRSIType] = {
    "void": VOID_TYPE,
    "bool": BOOL_TYPE,
    "int": INT_TYPE,
    "float": FLOAT_TYPE,
    "string": STRING_TYPE,
    "any": ANY_TYPE,
    "never": NEVER_TYPE,
}

_NUMERIC_BASES: Set[str] = {"int", "float"}


# ═══════════════════════════════════════════════════════════════════════════════
# Symbol Table / Scope
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Symbol:
    name: str
    nrsi_type: NRSIType
    mutable: bool = False
    declared_line: int = 0
    kind: str = "variable"


class Scope:
    """Lexical scope with parent chain."""

    def __init__(self, parent: Optional[Scope] = None, name: str = "") -> None:
        self.parent = parent
        self.name = name
        self._symbols: Dict[str, Symbol] = {}

    def define(self, name: str, symbol: Symbol) -> Optional[Diagnostic]:
        if name in self._symbols:
            prev = self._symbols[name]
            return Diagnostic(
                severity=DiagnosticSeverity.ERROR,
                message=(
                    f"Symbol '{name}' already defined in scope '{self.name}' "
                    f"(previous declaration at line {prev.declared_line})"
                ),
                line=symbol.declared_line,
                column=0,
                rule="redefinition",
                suggestion=f"Rename one of the '{name}' declarations",
            )
        self._symbols[name] = symbol
        return None

    def lookup(self, name: str) -> Optional[Symbol]:
        if name in self._symbols:
            return self._symbols[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        return None

    def lookup_local(self, name: str) -> Optional[Symbol]:
        return self._symbols.get(name)

    def all_symbols(self) -> Dict[str, Symbol]:
        result: Dict[str, Symbol] = {}
        if self.parent is not None:
            result.update(self.parent.all_symbols())
        result.update(self._symbols)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# TypeChecker — Main AST Walker
# ═══════════════════════════════════════════════════════════════════════════════

class TypeChecker:
    """Walk the AST and enforce NRSI type rules."""

    def __init__(self) -> None:
        self._diagnostics: List[Diagnostic] = []
        self._identifier_bindings: List[Tuple[Identifier, Optional[Symbol]]] = []
        self._global_scope = Scope(name="global")
        self._current_scope = self._global_scope
        self._gates: Dict[str, GateDecl] = {}
        self._gate_signatures: Dict[str, Tuple[NRSIType, NRSIType]] = {}
        self._norms: List[NormDecl] = []
        self._belief_bases: Dict[str, BeliefBaseDecl] = {}
        self._structs: Dict[str, StructDecl] = {}
        self._enums: Dict[str, EnumDecl] = {}
        self._lobes: Dict[str, LobeDecl] = {}
        self._current_fn: Optional[FnDecl] = None
        self._current_fn_return: Optional[NRSIType] = None
        self._validated_symbols: Set[str] = set()
        self._register_builtins()

    # ── Public API ────────────────────────────────────────────────────────

    def check(self, module: Module) -> List[Diagnostic]:
        """Type-check a parsed NRSI module.  Returns diagnostics."""
        self._diagnostics = []
        self._identifier_bindings = []
        self._phase1_register(module)
        self._phase2_check_bodies(module)
        self._phase3_norm_compliance()
        self._phase4_belief_consistency()
        logger.info(
            "type check complete: %d error(s), %d warning(s)",
            sum(1 for d in self._diagnostics if d.severity == DiagnosticSeverity.ERROR),
            sum(1 for d in self._diagnostics if d.severity == DiagnosticSeverity.WARNING),
        )
        return list(self._diagnostics)

    @property
    def identifier_bindings(self) -> List[Tuple[Identifier, Optional[Symbol]]]:
        """Resolved identifier → symbol pairs from the last ``check`` (for LSP tools)."""
        return list(self._identifier_bindings)

    def global_completion_symbols(self) -> List[str]:
        """Symbols visible at module scope after ``check`` (for LSP completion MVP)."""
        return sorted(self._global_scope.all_symbols().keys())

    @property
    def errors(self) -> List[Diagnostic]:
        return [d for d in self._diagnostics if d.severity == DiagnosticSeverity.ERROR]

    @property
    def warnings(self) -> List[Diagnostic]:
        return [d for d in self._diagnostics if d.severity == DiagnosticSeverity.WARNING]

    @property
    def has_errors(self) -> bool:
        return any(d.severity == DiagnosticSeverity.ERROR for d in self._diagnostics)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _emit(
        self,
        severity: DiagnosticSeverity,
        message: str,
        node: ASTNode,
        rule: str,
        suggestion: str = "",
    ) -> None:
        line = getattr(node, "line", 0) or 0
        col = getattr(node, "column", 0) or 0
        diag = Diagnostic(
            severity=severity, message=message,
            line=line, column=col, rule=rule, suggestion=suggestion,
        )
        self._diagnostics.append(diag)
        if severity == DiagnosticSeverity.ERROR:
            logger.debug("ERROR %s:%s %s", line, col, message)

    def _error(self, msg: str, node: ASTNode, rule: str, suggestion: str = "") -> None:
        self._emit(DiagnosticSeverity.ERROR, msg, node, rule, suggestion)

    def _warning(self, msg: str, node: ASTNode, rule: str, suggestion: str = "") -> None:
        self._emit(DiagnosticSeverity.WARNING, msg, node, rule, suggestion)

    def _info(self, msg: str, node: ASTNode, rule: str) -> None:
        self._emit(DiagnosticSeverity.INFO, msg, node, rule)

    def _push_scope(self, name: str) -> Scope:
        child = Scope(parent=self._current_scope, name=name)
        self._current_scope = child
        return child

    def _pop_scope(self) -> None:
        if self._current_scope.parent is not None:
            self._current_scope = self._current_scope.parent

    def _define(self, name: str, symbol: Symbol) -> None:
        diag = self._current_scope.define(name, symbol)
        if diag is not None:
            self._diagnostics.append(diag)

    def _register_builtins(self) -> None:
        _builtin_fn = NRSIType(
            base_name="builtin_fn", is_function=True,
            param_types=[ANY_TYPE], return_type=ANY_TYPE,
        )
        _builtin_type = NRSIType(base_name="builtin_type")
        scope = self._global_scope

        for name, ntype in _BUILTIN_TYPES.items():
            scope._symbols[name] = Symbol(
                name=name, nrsi_type=ntype, kind="type", declared_line=0,
            )

        _BUILTIN_FUNCTIONS = (
            "print", "log", "assert", "len", "type_of",
            "semantic_distance", "decompose", "intent_match",
            "compose", "persist",
            "str", "int", "float", "bool", "list", "dict", "set", "tuple",
            "bytes", "bytearray", "memoryview", "frozenset",
            "range", "enumerate", "zip", "map", "filter", "sorted", "reversed",
            "min", "max", "sum", "abs", "round", "pow", "divmod",
            "isinstance", "issubclass", "hasattr", "getattr", "setattr", "delattr",
            "id", "hash", "repr", "chr", "ord", "hex", "oct", "bin",
            "input", "open", "super", "property", "staticmethod", "classmethod",
            "callable", "iter", "next", "any", "all",
            "to_string", "to_int", "to_float",
            "format", "vars", "dir", "type",
        )
        for op_name in _BUILTIN_FUNCTIONS:
            scope._symbols[op_name] = Symbol(
                name=op_name, nrsi_type=_builtin_fn,
                kind="function", declared_line=0,
            )

        _BUILTIN_TYPE_NAMES = (
            "List", "Dict", "Set", "Map", "Tuple", "Optional", "Any",
            "Union", "Callable", "Iterator", "Generator", "Coroutine",
            "Sequence", "Mapping", "MutableMapping",
            "Type", "ClassVar", "Final",
            "None", "nil", "True", "False",
            "Exception", "ValueError", "TypeError", "KeyError",
            "IndexError", "AttributeError", "RuntimeError",
            "IOError", "OSError", "FileNotFoundError",
            "ImportError", "StopIteration", "NotImplementedError",
            "object", "dataclass", "field",
        )
        for tname in _BUILTIN_TYPE_NAMES:
            if tname not in scope._symbols:
                scope._symbols[tname] = Symbol(
                    name=tname, nrsi_type=_builtin_type,
                    kind="type", declared_line=0,
                )

    # ── Phase 1: Register top-level declarations ─────────────────────────

    def _phase1_register(self, module: Module) -> None:
        for decl in module.declarations:
            if isinstance(decl, ImportDecl):
                self._register_import(decl)
            elif isinstance(decl, StructDecl):
                self._register_struct(decl)
            elif isinstance(decl, EnumDecl):
                self._register_enum(decl)
            elif isinstance(decl, GateDecl):
                self._register_gate(decl)
            elif isinstance(decl, LobeDecl):
                self._register_lobe(decl)
            elif isinstance(decl, NormDecl):
                self._norms.append(decl)
            elif isinstance(decl, BeliefBaseDecl):
                self._belief_bases[decl.name] = decl
            elif isinstance(decl, FnDecl):
                self._register_fn_signature(decl)
            elif isinstance(decl, ProcessorDecl):
                self._register_processor_signature(decl)
            elif isinstance(decl, TrustDecl):
                pass  # checked in phase 2
            elif isinstance(decl, ComposeDecl):
                self._define(decl.name, Symbol(
                    name=decl.name, nrsi_type=ANY_TYPE,
                    kind="compose", declared_line=getattr(decl, "line", 0) or 0,
                ))
            elif isinstance(decl, PersistDecl):
                self._define(decl.name, Symbol(
                    name=decl.name, nrsi_type=NRSIType(base_name="LearnableStore"),
                    kind="persist", declared_line=getattr(decl, "line", 0) or 0,
                ))

    def _register_import(self, decl: ImportDecl) -> None:
        names_to_register = list(decl.names)
        if not names_to_register and decl.module_path:
            top_name = decl.alias or decl.module_path.split(".")[0]
            names_to_register.append(top_name)
        for name in names_to_register:
            self._define(name, Symbol(
                name=name, nrsi_type=ANY_TYPE,
                kind="import", declared_line=getattr(decl, "line", 0) or 0,
            ))

    def _register_struct(self, decl: StructDecl) -> None:
        self._structs[decl.name] = decl
        struct_type = NRSIType(base_name=decl.name)
        self._define(decl.name, Symbol(
            name=decl.name, nrsi_type=struct_type,
            kind="type", declared_line=getattr(decl, "line", 0) or 0,
        ))

    def _register_enum(self, decl: EnumDecl) -> None:
        self._enums[decl.name] = decl
        enum_type = NRSIType(base_name=decl.name)
        self._define(decl.name, Symbol(
            name=decl.name, nrsi_type=enum_type,
            kind="type", declared_line=getattr(decl, "line", 0) or 0,
        ))

    def _register_gate(self, decl: GateDecl) -> None:
        self._gates[decl.name] = decl
        in_type = UNKNOWN_TYPE
        if decl.params:
            in_type = self._resolve_type(decl.params[0].type_expr) if decl.params[0].type_expr else UNKNOWN_TYPE
        out_type = self._resolve_type(decl.return_type) if decl.return_type else UNKNOWN_TYPE
        self._gate_signatures[decl.name] = (in_type, out_type)
        gate_fn_type = NRSIType(
            base_name="gate", is_function=True,
            param_types=[in_type], return_type=out_type,
        )
        self._define(decl.name, Symbol(
            name=decl.name, nrsi_type=gate_fn_type,
            kind="gate", declared_line=getattr(decl, "line", 0) or 0,
        ))

    def _register_lobe(self, decl: LobeDecl) -> None:
        self._lobes[decl.name] = decl
        self._define(decl.name, Symbol(
            name=decl.name, nrsi_type=NRSIType(base_name="lobe"),
            kind="lobe", declared_line=getattr(decl, "line", 0) or 0,
        ))

    def _register_fn_signature(self, decl: FnDecl) -> None:
        param_types = [self._resolve_type(p.type_expr) for p in decl.params]
        ret = self._resolve_type(decl.return_type) if decl.return_type else VOID_TYPE
        fn_type = NRSIType(
            base_name="fn", is_function=True,
            param_types=param_types, return_type=ret,
        )
        self._define(decl.name, Symbol(
            name=decl.name, nrsi_type=fn_type,
            kind="function", declared_line=getattr(decl, "line", 0) or 0,
        ))

    def _register_processor_signature(self, decl: ProcessorDecl) -> None:
        param_types = [self._resolve_type(p.type_expr) for p in decl.params] if decl.params else []
        ret = self._resolve_type(decl.return_type) if decl.return_type else VOID_TYPE
        fn_type = NRSIType(
            base_name="processor", is_function=True,
            param_types=param_types, return_type=ret,
        )
        self._define(decl.name, Symbol(
            name=decl.name, nrsi_type=fn_type,
            kind="function", declared_line=getattr(decl, "line", 0) or 0,
        ))

    # ── Phase 2: Check bodies ────────────────────────────────────────────

    def _phase2_check_bodies(self, module: Module) -> None:
        for decl in module.declarations:
            if isinstance(decl, FnDecl):
                self._check_fn_decl(decl)
            elif isinstance(decl, ProcessorDecl):
                self._check_processor_decl(decl)
            elif isinstance(decl, GateDecl):
                self._check_gate_decl(decl)
            elif isinstance(decl, TrustDecl):
                self._check_trust_decl(decl)
            elif isinstance(decl, LetStmt):
                self._check_let(decl)

    # ── Phase 3: Norm compliance ─────────────────────────────────────────

    def _phase3_norm_compliance(self) -> None:
        for norm in self._norms:
            self._check_norm_compliance(norm)

    # ── Phase 4: Belief base consistency ─────────────────────────────────

    def _phase4_belief_consistency(self) -> None:
        for name, base in self._belief_bases.items():
            self._check_belief_consistency(base)

    # ── Type Resolution ──────────────────────────────────────────────────

    def _resolve_type(self, texpr: Optional[TypeExpr]) -> NRSIType:
        if texpr is None:
            return UNKNOWN_TYPE

        if isinstance(texpr, SimpleType):
            if texpr.name in _BUILTIN_TYPES:
                return NRSIType(base_name=texpr.name)
            if texpr.name in self._structs or texpr.name in self._enums:
                return NRSIType(base_name=texpr.name)
            return NRSIType(base_name=texpr.name)

        if isinstance(texpr, TrustType):
            level = texpr.trust_level
            if level not in _TRUST_RANKS:
                self._error(
                    f"Unknown trust level '{level}' — expected one of {_TRUST_NAMES}",
                    texpr, "invalid_trust_level",
                )
                return UNKNOWN_TYPE
            inner = self._resolve_type(texpr.inner_type)
            return NRSIType(
                base_name=inner.base_name,
                trust_level=level,
                generic_args=inner.generic_args,
                epistemic=inner.epistemic,
            )

        if isinstance(texpr, GenericType):
            base = texpr.name
            args = [self._resolve_type(a) for a in texpr.type_args]
            return NRSIType(base_name=base, generic_args=args)

        if isinstance(texpr, UnionType):
            members = [self._resolve_type(m) for m in texpr.types]
            names = "|".join(m.display() for m in members)
            return NRSIType(base_name=names)

        if isinstance(texpr, FunctionType):
            p_types = [self._resolve_type(p) for p in texpr.param_types]
            r_type = self._resolve_type(texpr.return_type) if texpr.return_type else VOID_TYPE
            return NRSIType(
                base_name="fn", is_function=True,
                param_types=p_types, return_type=r_type,
            )

        return UNKNOWN_TYPE

    # ── Declaration Checks ───────────────────────────────────────────────

    def _check_trust_decl(self, decl: TrustDecl) -> None:
        declared_level = decl.trust_level
        if declared_level not in _TRUST_RANKS:
            self._error(
                f"Unknown trust level '{declared_level}'",
                decl, "invalid_trust_level",
                suggestion=f"Use one of: {', '.join(_TRUST_NAMES)}",
            )
            return

        if decl.value is not None:
            val_type = self._check_expr(decl.value)
            if val_type.trust_level and val_type.trust_level != declared_level:
                val_rank = _TRUST_RANKS.get(val_type.trust_level, -1)
                decl_rank = _TRUST_RANKS[declared_level]
                if val_rank < decl_rank:
                    self._error(
                        f"Cannot declare {declared_level}[{val_type.base_name}] "
                        f"from a value that is {val_type.trust_level}[{val_type.base_name}] "
                        f"— must pass through a gate",
                        decl, "trust_flow",
                        suggestion=(
                            f"Pass the value through a gate that elevates "
                            f"from {val_type.trust_level} to {declared_level}"
                        ),
                    )

    def _check_gate_decl(self, decl: GateDecl) -> None:
        if decl.name not in self._gate_signatures:
            return
        in_type, out_type = self._gate_signatures[decl.name]

        if in_type.trust_level and out_type.trust_level:
            in_rank = in_type.trust_rank()
            out_rank = out_type.trust_rank()
            if out_rank <= in_rank:
                self._error(
                    f"Gate '{decl.name}' output trust ({out_type.trust_level}) "
                    f"must be higher than input trust ({in_type.trust_level})",
                    decl, "gate_elevation",
                    suggestion="A gate must elevate trust — set a higher output trust level",
                )

        if decl.body:
            self._push_scope(f"gate:{decl.name}")
            if decl.params:
                for p in decl.params:
                    p_type = self._resolve_type(p.type_expr) if p.type_expr else UNKNOWN_TYPE
                    self._define(p.name, Symbol(
                        name=p.name, nrsi_type=p_type,
                        declared_line=getattr(decl, "line", 0) or 0,
                    ))
            gate_builtins = {
                "confidence": NRSIType(base_name="float"),
                "trust": NRSIType(base_name="TrustLevel"),
                "epistemic": NRSIType(base_name="EpistemicType"),
                "temporal": NRSIType(base_name="TemporalValidity"),
                "domain": NRSIType(base_name="string"),
            }
            for gname, gtype in gate_builtins.items():
                self._define(gname, Symbol(
                    name=gname, nrsi_type=gtype,
                    declared_line=getattr(decl, "line", 0) or 0,
                ))
            for stmt in decl.body:
                self._check_stmt(stmt)
            self._pop_scope()

    def _check_fn_decl(self, decl: FnDecl) -> None:
        sym = self._current_scope.lookup(decl.name)
        if sym is None:
            return
        fn_type = sym.nrsi_type
        self._current_fn = decl
        self._current_fn_return = fn_type.return_type

        self._push_scope(f"fn:{decl.name}")

        for i, param in enumerate(decl.params):
            p_type = fn_type.param_types[i] if i < len(fn_type.param_types) else UNKNOWN_TYPE
            self._define(param.name, Symbol(
                name=param.name, nrsi_type=p_type,
                mutable=getattr(param, "mutable", False),
                declared_line=getattr(param, "line", 0) or 0,
            ))

        has_return = False
        for stmt in (decl.body or []):
            self._check_stmt(stmt)
            if isinstance(stmt, ReturnStmt):
                has_return = True

        if fn_type.return_type and fn_type.return_type.base_name != "void" and not has_return:
            if decl.body:
                self._warning(
                    f"Function '{decl.name}' declares return type "
                    f"{fn_type.return_type.display()} but may not return a value on all paths",
                    decl, "missing_return",
                    suggestion="Ensure all code paths return a value",
                )

        self._pop_scope()
        self._current_fn = None
        self._current_fn_return = None

    def _check_processor_decl(self, decl: ProcessorDecl) -> None:
        sym = self._current_scope.lookup(decl.name)
        if sym is None:
            return
        fn_type = sym.nrsi_type

        self._push_scope(f"processor:{decl.name}")
        if decl.params:
            for i, param in enumerate(decl.params):
                p_type = fn_type.param_types[i] if i < len(fn_type.param_types) else UNKNOWN_TYPE
                self._define(param.name, Symbol(
                    name=param.name, nrsi_type=p_type,
                    declared_line=getattr(param, "line", 0) or 0,
                ))

        for stmt in (decl.body or []):
            self._check_stmt(stmt)
        self._pop_scope()

    # ── Statement Checks ─────────────────────────────────────────────────

    def _check_stmt(self, stmt: ASTNode) -> None:
        if isinstance(stmt, LetStmt):
            self._check_let(stmt)
        elif isinstance(stmt, AssignStmt):
            self._check_assignment(stmt)
        elif isinstance(stmt, ReturnStmt):
            self._check_return(stmt)
        elif isinstance(stmt, ForStmt):
            self._check_for(stmt)
        elif isinstance(stmt, ExprStmt):
            self._check_expr(stmt.expr)
        elif isinstance(stmt, RequireStmt):
            self._check_require(stmt)
        elif isinstance(stmt, ValidateStmt):
            self._check_validate(stmt)
        elif isinstance(stmt, AugAssignStmt):
            self._check_aug_assign(stmt)
        elif isinstance(stmt, RaiseStmt):
            self._check_raise(stmt)
        elif isinstance(stmt, ClassDecl):
            self._check_class_decl(stmt)
        elif isinstance(stmt, WithStmt):
            if stmt.context:
                self._check_expr(stmt.context)
            if getattr(stmt, 'alias', ''):
                self._define(stmt.alias, Symbol(
                    name=stmt.alias, nrsi_type=ANY_TYPE,
                    mutable=False, declared_line=getattr(stmt, 'line', 0) or 0,
                ))
            for s in (stmt.body or []):
                self._check_stmt(s)
        elif isinstance(stmt, WhileStmt):
            if stmt.condition:
                self._check_expr(stmt.condition)
            for s in (stmt.body or []):
                self._check_stmt(s)
        elif isinstance(stmt, TryStmt):
            for s in (stmt.body or []):
                self._check_stmt(s)
            catch_var = getattr(stmt, 'catch_var', '')
            if catch_var:
                self._define(catch_var, Symbol(
                    name=catch_var, nrsi_type=ANY_TYPE,
                    declared_line=getattr(stmt, 'line', 0) or 0,
                ))
            for s in (getattr(stmt, 'catch_body', None) or []):
                self._check_stmt(s)
            for s in (getattr(stmt, 'finally_body', None) or []):
                self._check_stmt(s)
        elif isinstance(stmt, AssertStmt):
            if stmt.condition:
                self._check_expr(stmt.condition)
        elif isinstance(stmt, ImportDecl):
            self._register_import(stmt)
        elif isinstance(stmt, IfExpr):
            self._check_if_expr(stmt)
        elif isinstance(stmt, FnDecl):
            self._register_fn_signature(stmt)
            self._check_fn_decl(stmt)
        elif isinstance(stmt, (BreakStmt, ContinueStmt, PassStmt)):
            pass  # Control flow — no type implications
        elif isinstance(stmt, GlobalStmt):
            pass  # Global declaration — scope-level, no type check needed
        elif isinstance(stmt, DelStmt):
            if stmt.target:
                self._check_expr(stmt.target)
        else:
            import logging as _tc_log
            _tc_log.getLogger("nrsi.type_checker").debug("Unchecked statement type: %s", type(stmt).__name__)

    def _check_aug_assign(self, stmt: AugAssignStmt) -> None:
        if stmt.target:
            target_name = self._extract_target_name(stmt.target)
            if target_name:
                sym = self._current_scope.lookup(target_name)
                if sym is None:
                    self._error(
                        f"Undefined variable '{target_name}'",
                        stmt, "undefined_variable",
                        suggestion=f"Declare '{target_name}' with 'let' before assigning",
                    )
                elif not sym.mutable:
                    self._error(
                        f"Cannot reassign immutable variable '{target_name}'",
                        stmt, "immutability",
                        suggestion=f"Declare '{target_name}' with 'let mut' to make it mutable",
                    )
            else:
                self._check_expr(stmt.target)
        if stmt.value:
            self._check_expr(stmt.value)

    def _check_raise(self, stmt: RaiseStmt) -> None:
        if stmt.value:
            self._check_expr(stmt.value)
        if stmt.cause:
            self._check_expr(stmt.cause)

    def _check_class_decl(self, decl: ClassDecl) -> None:
        class_type = NRSIType(base_name=decl.name)
        self._define(decl.name, Symbol(
            name=decl.name, nrsi_type=class_type,
            kind="type", declared_line=getattr(decl, "line", 0) or 0,
        ))
        self._push_scope(f"class:{decl.name}")
        self._define("self", Symbol(
            name="self", nrsi_type=class_type, declared_line=getattr(decl, "line", 0) or 0,
        ))
        for method in (decl.methods or []):
            self._register_fn_signature(method)
            self._check_fn_decl(method)
        self._pop_scope()

    def _check_let(self, stmt: LetStmt) -> None:
        declared_type = (
            self._resolve_type(stmt.type_expr) if stmt.type_expr else None
        )
        init_type: Optional[NRSIType] = None
        if stmt.value is not None:
            init_type = self._check_expr(stmt.value)

        if declared_type and init_type:
            if not init_type.is_assignable_to(declared_type):
                if (
                    declared_type.trust_level
                    and init_type.trust_level
                    and init_type.trust_rank() < declared_type.trust_rank()
                ):
                    self._error(
                        f"Cannot assign {init_type.display()} to variable "
                        f"'{stmt.name}' of type {declared_type.display()} "
                        f"— must pass through a gate",
                        stmt, "trust_flow",
                        suggestion=(
                            f"Pass the value through a gate that elevates "
                            f"from {init_type.trust_level} to {declared_type.trust_level}"
                        ),
                    )
                else:
                    self._error(
                        f"Type mismatch: cannot assign {init_type.display()} "
                        f"to variable '{stmt.name}' of type {declared_type.display()}",
                        stmt, "type_mismatch",
                    )

        final_type = declared_type or init_type or UNKNOWN_TYPE
        is_mutable = getattr(stmt, "mutable", False)

        self._define(stmt.name, Symbol(
            name=stmt.name, nrsi_type=final_type,
            mutable=is_mutable,
            declared_line=getattr(stmt, "line", 0) or 0,
        ))

    def _check_assignment(self, stmt: AssignStmt) -> None:
        target_name = self._extract_target_name(stmt.target)
        if target_name is None:
            self._check_expr(stmt.target)
            val_type = self._check_expr(stmt.value)
            return

        sym = self._current_scope.lookup(target_name)
        if sym is None:
            self._error(
                f"Undefined variable '{target_name}'",
                stmt, "undefined_variable",
                suggestion=f"Declare '{target_name}' with 'let' before assigning",
            )
            return

        if not sym.mutable:
            self._error(
                f"Cannot reassign immutable variable '{target_name}'",
                stmt, "immutability",
                suggestion=f"Declare '{target_name}' with 'let mut' to make it mutable",
            )

        val_type = self._check_expr(stmt.value)
        target_type = sym.nrsi_type

        if not val_type.is_assignable_to(target_type):
            if (
                target_type.trust_level
                and val_type.trust_level
                and val_type.trust_rank() < target_type.trust_rank()
            ):
                self._error(
                    f"Cannot assign {val_type.display()} to variable "
                    f"'{target_name}' of type {target_type.display()} "
                    f"— must pass through a gate",
                    stmt, "trust_flow",
                    suggestion=(
                        f"Pass the value through a gate that elevates "
                        f"from {val_type.trust_level} to {target_type.trust_level}"
                    ),
                )
            elif not val_type.is_base_compatible(target_type):
                self._error(
                    f"Type mismatch: cannot assign {val_type.display()} "
                    f"to '{target_name}' of type {target_type.display()}",
                    stmt, "type_mismatch",
                )

    def _check_return(self, stmt: ReturnStmt) -> None:
        if stmt.value is None:
            ret_type = VOID_TYPE
        else:
            ret_type = self._check_expr(stmt.value)

        if self._current_fn_return is not None:
            expected = self._current_fn_return
            if not ret_type.is_assignable_to(expected):
                fn_name = self._current_fn.name if self._current_fn else "<anonymous>"
                if (
                    expected.trust_level
                    and ret_type.trust_level
                    and ret_type.trust_rank() < expected.trust_rank()
                ):
                    self._error(
                        f"Function '{fn_name}' declares return type "
                        f"{expected.display()} but body returns {ret_type.display()}",
                        stmt, "return_trust_mismatch",
                        suggestion=(
                            f"Pass the return value through a gate that elevates "
                            f"from {ret_type.trust_level} to {expected.trust_level}"
                        ),
                    )
                elif expected.base_name != "void":
                    self._error(
                        f"Function '{fn_name}' declares return type "
                        f"{expected.display()} but returns {ret_type.display()}",
                        stmt, "return_type_mismatch",
                    )

    def _check_for(self, stmt: ForStmt) -> None:
        self._push_scope("for")
        iter_type = self._check_expr(stmt.iterable) if stmt.iterable else UNKNOWN_TYPE

        elem_type: NRSIType
        if iter_type.generic_args:
            elem_type = iter_type.generic_args[0]
        else:
            elem_type = UNKNOWN_TYPE

        self._define(stmt.variable, Symbol(
            name=stmt.variable, nrsi_type=elem_type,
            declared_line=getattr(stmt, "line", 0) or 0,
        ))

        for body_stmt in (stmt.body or []):
            self._check_stmt(body_stmt)
        self._pop_scope()

    def _check_require(self, stmt: RequireStmt) -> None:
        if stmt.condition:
            cond_type = self._check_expr(stmt.condition)
            if cond_type.base_name not in ("bool", "any", "unknown"):
                self._warning(
                    f"Require condition has type {cond_type.display()}, expected bool",
                    stmt, "require_type",
                )

    def _check_validate(self, stmt: ValidateStmt) -> None:
        if stmt.validator:
            self._validated_symbols.add(stmt.validator)
        for arg in (stmt.args or []):
            self._check_expr(arg)

    # ── Expression Checks ────────────────────────────────────────────────

    def _check_expr(self, expr: Expr) -> NRSIType:
        if isinstance(expr, Literal):
            return self._check_literal(expr)
        if isinstance(expr, Identifier):
            return self._check_identifier(expr)
        if isinstance(expr, BinaryOp):
            return self._check_binary(expr)
        if isinstance(expr, UnaryOp):
            return self._check_unary(expr)
        if isinstance(expr, CallExpr):
            return self._check_call(expr)
        if isinstance(expr, MemberAccess):
            return self._check_member_access(expr)
        if isinstance(expr, IndexExpr):
            return self._check_index(expr)
        if isinstance(expr, IfExpr):
            return self._check_if_expr(expr)
        if isinstance(expr, MatchExpr):
            return self._check_match_expr(expr)
        etype = type(expr).__name__
        if etype == "TernaryExpr":
            if getattr(expr, 'condition', None):
                self._check_expr(expr.condition)
            t1 = self._check_expr(expr.then_expr) if getattr(expr, 'then_expr', None) else UNKNOWN_TYPE
            t2 = self._check_expr(expr.else_expr) if getattr(expr, 'else_expr', None) else UNKNOWN_TYPE
            return t1 if t1.base_name != "unknown" else t2
        if etype == "ComprehensionExpr":
            if getattr(expr, 'iterable', None):
                self._check_expr(expr.iterable)
            return ANY_TYPE
        if etype in ("YieldExpr", "AwaitExpr"):
            val = getattr(expr, 'value', None)
            if val:
                return self._check_expr(val)
            return ANY_TYPE
        if etype == "LambdaExpr":
            return NRSIType(base_name="callable", is_function=True, return_type=ANY_TYPE)
        if etype == "SpreadExpr":
            val = getattr(expr, 'value', None)
            if val:
                return self._check_expr(val)
            return ANY_TYPE
        if etype in ("ListLiteral", "DictLiteral", "SetLiteral",
                      "TupleLiteral", "FStringExpr"):
            return ANY_TYPE
        return UNKNOWN_TYPE

    def _check_literal(self, lit: Literal) -> NRSIType:
        val = lit.value
        if isinstance(val, bool):
            return BOOL_TYPE
        if isinstance(val, int):
            return INT_TYPE
        if isinstance(val, float):
            return FLOAT_TYPE
        if isinstance(val, str):
            return STRING_TYPE
        return UNKNOWN_TYPE

    def _check_identifier(self, ident: Identifier) -> NRSIType:
        sym = self._current_scope.lookup(ident.name)
        self._identifier_bindings.append((ident, sym))
        if sym is None:
            self._error(
                f"Undefined variable '{ident.name}'",
                ident, "undefined_variable",
                suggestion=f"Declare '{ident.name}' with 'let' before use",
            )
            return UNKNOWN_TYPE
        return sym.nrsi_type

    def _check_binary(self, expr: BinaryOp) -> NRSIType:
        left = self._check_expr(expr.left)
        right = self._check_expr(expr.right)

        comparison_ops = {"==", "!=", "<", ">", "<=", ">="}
        logical_ops = {"and", "or", "&&", "||"}

        if expr.op in comparison_ops:
            return BOOL_TYPE
        if expr.op in logical_ops:
            return BOOL_TYPE

        arithmetic_ops = {"+", "-", "*", "/", "%", "**", "~~", "<<", ">>", "^", "&", "|"}
        if expr.op in arithmetic_ops:
            if left.base_name in _NUMERIC_BASES and right.base_name in _NUMERIC_BASES:
                if left.base_name == "float" or right.base_name == "float":
                    return FLOAT_TYPE
                return INT_TYPE
            if expr.op == "+" and left.base_name == "string":
                return STRING_TYPE

        trust_left = left.trust_rank()
        trust_right = right.trust_rank()
        if trust_left >= 0 and trust_right >= 0:
            min_rank = min(trust_left, trust_right)
            result_trust = _TRUST_NAMES[min_rank] if min_rank < len(_TRUST_NAMES) else None
            return NRSIType(
                base_name=left.base_name,
                trust_level=result_trust,
            )

        return left if left.base_name != "unknown" else right

    def _check_unary(self, expr: UnaryOp) -> NRSIType:
        operand = self._check_expr(expr.operand)
        if expr.op in ("!", "not"):
            return BOOL_TYPE
        if expr.op == "-":
            return operand
        return operand

    def _check_call(self, call: CallExpr) -> NRSIType:
        callee_name = self._extract_expr_name(call.callee)

        if callee_name and callee_name in self._gates:
            return self._check_gate_call(call, callee_name)

        callee_type = self._check_expr(call.callee)

        if not callee_type.is_function:
            if callee_type.base_name not in ("unknown", "any", "builtin_fn", "builtin_type"):
                self._error(
                    f"'{callee_name or callee_type.display()}' is not callable",
                    call, "not_callable",
                )
            arg_types = [self._check_expr(a) for a in call.args]
            return UNKNOWN_TYPE

        arg_types = [self._check_expr(a) for a in call.args]

        if callee_type.base_name == "builtin_fn":
            return callee_type.return_type or ANY_TYPE

        if callee_type.param_types:
            expected_count = len(callee_type.param_types)
            actual_count = len(arg_types)
            if actual_count != expected_count:
                self._error(
                    f"Function '{callee_name or '?'}' expects {expected_count} "
                    f"argument(s) but got {actual_count}",
                    call, "arity_mismatch",
                )
            else:
                for i, (arg_t, param_t) in enumerate(zip(arg_types, callee_type.param_types)):
                    if not arg_t.is_assignable_to(param_t):
                        if (
                            param_t.trust_level
                            and arg_t.trust_level
                            and arg_t.trust_rank() < param_t.trust_rank()
                        ):
                            self._error(
                                f"Function '{callee_name or '?'}' requires "
                                f"{param_t.display()} for parameter {i} but received "
                                f"{arg_t.display()}",
                                call, "gate_required",
                                suggestion=(
                                    f"Pass the argument through a gate to elevate "
                                    f"from {arg_t.trust_level} to {param_t.trust_level}"
                                ),
                            )
                        elif not arg_t.is_base_compatible(param_t):
                            self._error(
                                f"Argument {i} type mismatch: expected "
                                f"{param_t.display()}, got {arg_t.display()}",
                                call, "arg_type_mismatch",
                            )

        return callee_type.return_type or UNKNOWN_TYPE

    def _check_gate_call(self, call: CallExpr, gate_name: str) -> NRSIType:
        """Check a gate invocation — input must be raw-ish, output is elevated."""
        in_type, out_type = self._gate_signatures[gate_name]

        arg_types = [self._check_expr(a) for a in call.args]
        if not arg_types:
            self._error(
                f"Gate '{gate_name}' requires at least one argument",
                call, "gate_arity",
            )
            return out_type

        actual_in = arg_types[0]

        if in_type.trust_level and actual_in.trust_level:
            if actual_in.trust_rank() > in_type.trust_rank():
                self._warning(
                    f"Data passed to gate '{gate_name}' is already "
                    f"{actual_in.trust_level} (gate expects {in_type.trust_level} or lower)",
                    call, "unnecessary_gate",
                    suggestion="The data is already at a higher trust level than the gate input",
                )

        return out_type

    def _check_member_access(self, expr: MemberAccess) -> NRSIType:
        obj_type = self._check_expr(expr.object)

        if obj_type.base_name in self._structs:
            struct = self._structs[obj_type.base_name]
            for f in (struct.fields or []):
                if f.name == expr.member:
                    field_type = self._resolve_type(f.type_expr)
                    if obj_type.trust_level:
                        field_type = NRSIType(
                            base_name=field_type.base_name,
                            trust_level=obj_type.trust_level,
                            generic_args=field_type.generic_args,
                        )
                    return field_type
            self._error(
                f"Struct '{obj_type.base_name}' has no field '{expr.member}'",
                expr, "unknown_field",
            )

        if obj_type.base_name in self._enums:
            return NRSIType(base_name=obj_type.base_name)

        return UNKNOWN_TYPE

    def _check_index(self, expr: IndexExpr) -> NRSIType:
        obj_type = self._check_expr(expr.object)
        idx_type = self._check_expr(expr.index)
        if obj_type.generic_args:
            return obj_type.generic_args[0]
        return UNKNOWN_TYPE

    def _check_if_expr(self, expr: IfExpr) -> NRSIType:
        cond_type = self._check_expr(expr.condition)
        if cond_type.base_name not in ("bool", "any", "unknown"):
            self._warning(
                f"If condition has type {cond_type.display()}, expected bool",
                expr, "condition_type",
            )

        then_type = VOID_TYPE
        for stmt in (expr.then_body or []):
            then_type = self._check_stmt_or_expr(stmt)

        else_type = VOID_TYPE
        if expr.else_body:
            for stmt in expr.else_body:
                else_type = self._check_stmt_or_expr(stmt)

        if then_type.is_assignable_to(else_type):
            return then_type
        if else_type.is_assignable_to(then_type):
            return else_type
        return then_type

    def _check_match_expr(self, expr: MatchExpr) -> NRSIType:
        subject_type = self._check_expr(expr.subject)
        arm_types: List[NRSIType] = []

        for arm in (expr.arms or []):
            self._push_scope("match_arm")
            if hasattr(arm, "body") and arm.body:
                for stmt in arm.body:
                    arm_type = self._check_stmt_or_expr(stmt)
                arm_types.append(arm_type)
            self._pop_scope()

        if arm_types:
            return arm_types[0]
        return VOID_TYPE

    def _check_stmt_or_expr(self, node: ASTNode) -> NRSIType:
        if isinstance(node, (Literal, Identifier, BinaryOp, UnaryOp,
                             CallExpr, MemberAccess, IndexExpr, IfExpr, MatchExpr)):
            return self._check_expr(node)
        if isinstance(node, ExprStmt):
            return self._check_expr(node.expr)
        if isinstance(node, ReturnStmt) and node.value:
            return self._check_expr(node.value)
        self._check_stmt(node)
        return VOID_TYPE

    # ── Norm Compliance ──────────────────────────────────────────────────

    def _check_norm_compliance(self, norm: NormDecl = None) -> None:
        if norm is None:
            for n in self._norms:
                self._check_norm_compliance(n)
            return

        norm_kind = getattr(norm, "kind", None) or getattr(norm, "deontic", None)
        domain = getattr(norm, "domain", None)
        action = getattr(norm, "action", None)
        norm_name = getattr(norm, "name", "unnamed")

        if norm_kind not in ("prohibition", "obligation", "PROHIBITION", "OBLIGATION"):
            return

        is_prohibition = norm_kind in ("prohibition", "PROHIBITION")

        all_syms = self._global_scope.all_symbols()
        for sym_name, sym in all_syms.items():
            if sym.kind != "function":
                continue

            fn_type = sym.nrsi_type
            if not fn_type.is_function:
                continue

            if is_prohibition and domain and action:
                self._check_fn_norm_prohibition(
                    sym_name, fn_type, norm_name, domain, action, norm,
                )

    def _check_fn_norm_prohibition(
        self,
        fn_name: str,
        fn_type: NRSIType,
        norm_name: str,
        domain: str,
        action: str,
        norm_node: ASTNode,
    ) -> None:
        if fn_type.return_type is None:
            return

        ret = fn_type.return_type
        if action in ("output_raw", "output_unvalidated"):
            if ret.trust_level in ("raw", None):
                if domain.lower() in fn_name.lower() or domain == "*":
                    self._error(
                        f"Norm '{norm_name}' prohibits {action} in domain "
                        f"'{domain}', but function '{fn_name}' returns "
                        f"{ret.display()} (unvalidated)",
                        norm_node, "norm_violation",
                        suggestion=(
                            f"Ensure '{fn_name}' returns at least validated data "
                            f"or add a gate before output"
                        ),
                    )

        if action in ("use_raw", "consume_raw"):
            for p in fn_type.param_types:
                if p.trust_level in ("raw", None):
                    if domain.lower() in fn_name.lower() or domain == "*":
                        self._warning(
                            f"Norm '{norm_name}' prohibits {action} in domain "
                            f"'{domain}', but function '{fn_name}' accepts "
                            f"{p.display()} (raw)",
                            norm_node, "norm_violation",
                            suggestion=f"Require at least validated data for '{fn_name}'",
                        )

    # ── Belief Base Consistency ──────────────────────────────────────────

    def _check_belief_consistency(self, base: BeliefBaseDecl) -> None:
        axioms: List[AxiomDecl] = getattr(base, "axioms", []) or []
        if not axioms:
            return

        seen_predicates: Dict[str, List[AxiomDecl]] = {}
        for axiom in axioms:
            pred = getattr(axiom, "predicate", None) or getattr(axiom, "name", "?")
            seen_predicates.setdefault(pred, []).append(axiom)

        for pred, group in seen_predicates.items():
            if len(group) < 2:
                continue

            negations: List[bool] = []
            for ax in group:
                is_neg = getattr(ax, "negated", False)
                negations.append(is_neg)

            has_positive = any(not n for n in negations)
            has_negative = any(n for n in negations)
            if has_positive and has_negative:
                self._error(
                    f"Belief base '{base.name}' contains contradictory axioms "
                    f"for predicate '{pred}' — both asserted and negated",
                    base, "belief_contradiction",
                    suggestion=(
                        f"Remove either the positive or negative axiom for '{pred}', "
                        f"or use a separate belief base for the alternative view"
                    ),
                )

    # ── Epistemic Consistency ────────────────────────────────────────────

    def check_epistemic_claim(
        self,
        claimed: str,
        producing_lobe: Optional[str],
        node: ASTNode,
    ) -> None:
        """Verify that an epistemic claim is consistent with its source lobe."""
        if claimed not in _EPISTEMIC_VALID:
            self._warning(
                f"Unknown epistemic type '{claimed}'",
                node, "unknown_epistemic",
                suggestion=f"Valid epistemic types: {', '.join(sorted(_EPISTEMIC_VALID))}",
            )
            return

        if producing_lobe is None:
            return

        lobe_lower = producing_lobe.lower()
        if lobe_lower not in _LOBE_EPISTEMIC_AFFINITY:
            return

        compatible = _LOBE_EPISTEMIC_AFFINITY[lobe_lower]
        if claimed not in compatible:
            self._warning(
                f"Claim declared as {claimed} but produced by {producing_lobe} lobe "
                f"(expected one of: {', '.join(sorted(compatible))})",
                node, "epistemic_mismatch",
                suggestion=(
                    f"Either change the claim type to match the {producing_lobe} lobe's "
                    f"output, or route through a lobe that produces {claimed} results"
                ),
            )

    # ── Name Extraction Utilities ────────────────────────────────────────

    def _extract_target_name(self, expr: Any) -> Optional[str]:
        if isinstance(expr, Identifier):
            return expr.name
        if isinstance(expr, str):
            return expr
        return None

    def _extract_expr_name(self, expr: Any) -> Optional[str]:
        if isinstance(expr, Identifier):
            return expr.name
        if isinstance(expr, MemberAccess):
            obj_name = self._extract_expr_name(expr.object)
            if obj_name:
                return f"{obj_name}.{expr.member}"
        if isinstance(expr, str):
            return expr
        return None
