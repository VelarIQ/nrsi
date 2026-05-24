"""NRSI Swift Transpiler — AST to Swift Code Generation.

Converts a type-checked NRSI AST into idiomatic Swift source code that uses
generic trust-level wrappers, precondition-based gates, and strongly typed
norm/belief structures.

Each NRSI construct maps to specific Swift patterns:
  trust x: validated[string]  → let x = NRSIData<String>(...)
  gate verify(d: raw[T])      → func verify(...) with precondition checks
  lobe logical { ... }        → class LogicalLobe { ... }
  norm no_medical              → let normNoMedical = NRSINorm(...)
  belief base facts            → let factsBeliefBase = BeliefBase(...)
  fn name(params) -> ret       → func name(params) -> SwiftType { ... }
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Set

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Lookup Tables
# ═══════════════════════════════════════════════════════════════════════════════

SWIFT_TRUST_LEVELS: Dict[str, str] = {
    "raw": ".raw", "validated": ".validated",
    "trusted": ".trusted", "certified": ".certified",
}
SWIFT_DEONTIC: Dict[str, str] = {
    "obligation": ".obligation", "permission": ".permission",
    "prohibition": ".prohibition", "exemption": ".exemption",
}
SWIFT_BINOP: Dict[str, str] = {
    "+": "+", "-": "-", "*": "*", "/": "/", "%": "%",
    "**": "**", "~~": "/",  # NRSI floor div → Swift truncating division (approx.)
    "==": "==", "!=": "!=", "<": "<", ">": ">", "<=": "<=", ">=": ">=",
    "<<": "<<", ">>": ">>", "^": "^", "&": "&", "|": "|",
    "and": "&&", "or": "||", "&&": "&&", "||": "||",
}
SWIFT_TYPE: Dict[str, str] = {
    "string": "String", "int": "Int", "float": "Double", "bool": "Bool",
    "none": "Void", "void": "Void", "list": "Array", "dict": "Dictionary", "any": "Any",
    "String": "String", "Int": "Int", "Float": "Double",
    "Double": "Double", "Bool": "Bool",
}
SWIFT_LOBE: Dict[str, str] = {
    "linguistic": "LinguisticLobe", "logical": "LogicalLobe",
    "mathematical": "MathematicalLobe", "spatial": "SpatialLobe",
    "temporal": "TemporalLobe", "creative": "CreativeLobe",
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Swift Runtime Preamble
# ═══════════════════════════════════════════════════════════════════════════════

_PREAMBLE = '''\
import Foundation

// MARK: - NRSI Trust Type System
enum TrustLevel: Int, Comparable {
    case raw = 0, validated = 1, trusted = 2, certified = 3
    static func < (lhs: TrustLevel, rhs: TrustLevel) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

enum EpistemicType: String, CaseIterable {
    case deductive, inductive, abductive, analogical, causal
    case computational, observational, testimonial, creative, speculative
}

struct NRSIData<T> {
    let value: T
    var trustLevel: TrustLevel
    var confidence: Double
    var epistemic: EpistemicType
    var domain: String
    func requireTrust(_ required: TrustLevel, context: String = "") {
        precondition(trustLevel >= required,
            "Trust violation in \\(context): have \\(trustLevel), need \\(required)")
    }
    func withTrust(_ level: TrustLevel) -> NRSIData<T> {
        var copy = self; copy.trustLevel = level; return copy
    }
}

// MARK: - NRSI Normative Types
enum DeonticType: String {
    case obligation, permission, prohibition, exemption
}
struct NRSINorm {
    let id: String; let deonticType: DeonticType; let scope: String
    let action: String; let priority: Int; let domain: String
}

// MARK: - NRSI Belief System
struct Axiom { let content: String; let tier: String }
struct BeliefBase {
    let name: String; let entrenchment: String; let axioms: [Axiom]
}
'''

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Helpers
# ═══════════════════════════════════════════════════════════════════════════════

_SNAKE_RE = re.compile(r"_([a-z])")


def _camel(name: str) -> str:
    """``snake_case`` → ``camelCase``, preserving leading underscores."""
    if not name or "_" not in name:
        return name
    prefix, stripped = "", name
    while stripped.startswith("_"):
        prefix += "_"; stripped = stripped[1:]
    return (prefix + _SNAKE_RE.sub(lambda m: m.group(1).upper(), stripped)) if stripped else name


def _pascal(name: str) -> str:
    """``snake_case`` → ``PascalCase``."""
    return "".join(p.capitalize() for p in name.split("_") if p)


def _ga(obj: Any, *keys: str, default: Any = None) -> Any:
    """Chain-fallback getattr across multiple attribute names."""
    for k in keys:
        v = getattr(obj, k, None)
        if v is not None:
            return v
    return default


_NRSI_IDENT_REMAP: Dict[str, str] = {
    "nil": "nil",  # Swift keyword — keep
    "true": "true",
    "false": "false",
}


def _swift_module_import_name(path: str) -> str:
    """Derive a Swift `import Foo` module name from an NRSI dotted path."""
    p = path.strip().replace("/", ".").replace("-", "_")
    if not p:
        return "Foundation"
    parts = [x for x in p.split(".") if x]
    if not parts:
        return "Foundation"
    # Use PascalCase of last segment (common Swift module style)
    return _pascal(parts[-1]) if parts[-1] else "Foundation"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SwiftTranspiler
# ═══════════════════════════════════════════════════════════════════════════════

class SwiftTranspiler:
    """Convert NRSI AST to Swift source code.

    Handles AST nodes from both the parser module (``nrsi.lang.parser``)
    and the transpiler module's internal definitions by dispatching on
    ``type(node).__name__``.

    Usage::

        transpiler = SwiftTranspiler()
        swift_source = transpiler.transpile(module_ast)
    """

    def __init__(self, type_checked: bool = True) -> None:
        self._type_checked = type_checked
        self._indent = 0
        self._output: List[str] = []
        self._declared_types: Set[str] = set()

    # ── Public API ───────────────────────────────────────────────────────

    def transpile(self, module: Any) -> str:
        """Transpile a complete NRSI module to Swift source."""
        self._indent = 0
        self._output = []
        self._declared_types = set()
        name = _ga(module, "name", "source_file", default="<nrsi>")
        self._emit(f"// Auto-generated Swift from NRSI source: {name}")
        self._emit("// Generated by the NRSI SwiftTranspiler. Do not edit directly.")
        self._emit("// Trust enforcement is active at runtime via preconditions.")
        self._emit("")
        for line in _PREAMBLE.splitlines():
            self._output.append(line)
        self._emit("")
        self._emit("// MARK: - NRSI module imports")
        for imp in (_ga(module, "imports", default=[]) or []):
            self._tr_import(imp)
        self._emit("// MARK: - Generated NRSI Declarations")
        self._emit("")
        for decl in (_ga(module, "declarations", default=[]) or []):
            self._transpile_decl(decl)
            self._emit("")
        return "\n".join(self._output)

    # ── Emit helpers ─────────────────────────────────────────────────────

    def _emit(self, line: str) -> None:
        self._output.append(("    " * self._indent + line) if line.strip() else "")

    def _indent_up(self) -> None:
        self._indent += 1

    def _indent_down(self) -> None:
        self._indent = max(0, self._indent - 1)

    # ── Declaration dispatch ─────────────────────────────────────────────

    def _transpile_decl(self, decl: Any) -> None:
        dt = type(decl).__name__
        dispatch: Dict[str, Any] = {
            "TrustDecl": self._tr_trust, "GateDecl": self._tr_gate,
            "LobeDecl": self._tr_lobe, "NormDecl": self._tr_norm,
            "BeliefBaseDecl": self._tr_belief, "FnDecl": self._tr_fn,
            "ConstDecl": self._tr_const, "StructDecl": self._tr_struct,
            "EnumDecl": self._tr_enum, "ClassDecl": self._tr_class,
            "ComposeDecl": self._tr_compose, "PersistDecl": self._tr_persist,
        }
        h = dispatch.get(dt)
        if h:
            h(decl)
        elif dt in ("LetStmt", "AssignStmt", "ExprStmt"):
            self._tr_stmt(decl)
        elif dt == "ImportDecl":
            self._tr_import(decl)
        else:
            self._emit(f"// [SwiftTranspiler] unsupported: {type(decl).__name__}")

    # ── trust ────────────────────────────────────────────────────────────

    def _tr_trust(self, d: Any) -> None:
        name = _camel(d.name)
        trust = SWIFT_TRUST_LEVELS.get(getattr(d, "trust_level", "raw"), ".raw")
        val = self._expr(d.value) if getattr(d, "value", None) else '""'
        vtype = _ga(d, "value_type", "inner_type")
        inner = self._resolve_type(vtype) if vtype else "String"
        ann: Dict = getattr(d, "annotations", {}) or {}
        conf = self._ann_float(ann, "confidence", getattr(d, "confidence", 0.0))
        ep = self._ann_str(ann, "epistemic", getattr(d, "epistemic", "") or "observational")
        dom = self._ann_str(ann, "domain", "general")
        self._emit(f"let {name} = NRSIData<{inner}>(")
        self._indent_up()
        self._emit(f"value: {val},")
        self._emit(f"trustLevel: {trust},")
        self._emit(f"confidence: {conf},")
        self._emit(f"epistemic: .{ep},")
        self._emit(f'domain: "{dom}"')
        self._indent_down()
        self._emit(")")

    def _ann_float(self, ann: Dict, key: str, default: Any) -> float:
        if key in ann:
            v = getattr(ann[key], "value", None)
            return float(v) if v is not None else float(default or 0)
        return float(default or 0)

    def _ann_str(self, ann: Dict, key: str, default: str) -> str:
        if key in ann:
            e = ann[key]
            return str(_ga(e, "value", "name", default=default))
        return default

    # ── import ───────────────────────────────────────────────────────────

    def _tr_import(self, decl: Any) -> None:
        """``import path`` / ``from path import a, b`` → Swift ``import Module`` (+ comments)."""
        path = (_ga(decl, "module_path", "module", default="") or "").strip()
        if not path:
            self._emit("// [SwiftTranspiler] empty import")
            return
        if path.startswith("std/"):
            path = path[4:].replace("/", ".")
        names = getattr(decl, "names", None) or []
        alias = getattr(decl, "alias", "") or ""
        self._emit(f"// NRSI import: {path}" + (f" as {alias}" if alias else ""))
        if names:
            self._emit(f"//   symbols: {', '.join(names)} (Swift imports are module-wide)")
        mod = _swift_module_import_name(path)
        self._emit(f"import {mod}")
        if alias:
            self._emit(f"// NOTE: Swift has no import-alias syntax; typealias {alias} = {mod} may be used manually")

    # ── gate ─────────────────────────────────────────────────────────────

    def _tr_gate(self, d: Any) -> None:
        fn = _camel(d.name)
        params = getattr(d, "params", []) or []
        pl = ", ".join(f"{_camel(p.name)}: {self._param_type(p)}" for p in params)
        ret = getattr(d, "return_type", None)
        rt = self._resolve_type(ret) if ret else "NRSIData<String>"
        target = ".validated"
        if ret:
            base = _ga(ret, "trust_level", "base")
            if base and base in SWIFT_TRUST_LEVELS:
                target = SWIFT_TRUST_LEVELS[base]
        conf, validators = self._gate_reqs(d)
        self._emit(
            f"// NRSI trust gate: `{d.name}` confidence>={conf} "
            f"validators={validators!r} outputTrust={target.strip('.')}"
        )
        self._emit(f"func {fn}({pl}) -> {rt} {{")
        self._indent_up()
        fp = _camel(params[0].name) if params else "data"
        self._emit(f'precondition({fp}.confidence >= {conf}, "Gate \'{d.name}\': confidence below threshold")')
        for v in validators:
            if v:
                self._emit(f"{_camel(v)}({fp})")
        has_extra = False
        for stmt in (getattr(d, "body", None) or []):
            if type(stmt).__name__ not in ("RequireStmt", "ValidateStmt"):
                self._tr_stmt(stmt)
                has_extra = True
        if not has_extra:
            self._emit(f"var result = {fp}")
            self._emit(f"result.trustLevel = {target}")
            self._emit("return result")
        self._indent_down()
        self._emit("}")

    def _gate_reqs(self, d: Any):
        conf, vals = 0.95, []  # type: ignore[var-annotated]
        reqs = getattr(d, "requires", None)
        if reqs is not None:
            for r in reqs:
                if getattr(r, "field", "") == "confidence":
                    conf = r.value
            vals = list(getattr(d, "validators", []) or [])
        else:
            for s in (getattr(d, "body", None) or []):
                sn = type(s).__name__
                if sn == "RequireStmt":
                    c = getattr(s, "condition", None)
                    if c and type(c).__name__ in ("BinOpExpr", "BinaryOp"):
                        if (getattr(c, "op", "") in (">=", ">")
                                and getattr(_ga(c, "left"), "name", "") == "confidence"
                                and getattr(_ga(c, "right"), "value", None) is not None):
                            conf = _ga(c, "right").value
                elif sn == "ValidateStmt":
                    vals.append(getattr(s, "validator", ""))
        return conf, vals

    # ── lobe ─────────────────────────────────────────────────────────────

    def _tr_lobe(self, d: Any) -> None:
        cls = SWIFT_LOBE.get(d.name, _pascal(d.name) + "Lobe")
        procs = getattr(d, "processors", []) or []
        creases = getattr(d, "creases", []) or []
        self._emit(f"class {cls} {{")
        self._indent_up()
        for crease in creases:
            cname = _camel(getattr(crease, "name", "unknown"))
            domain = getattr(crease, "domain", "")
            facts = getattr(crease, "facts", []) or []
            self._emit(f"struct {_pascal(cname)}Crease {{")
            self._indent_up()
            self._emit(f'static let domain = "{domain}"')
            if facts:
                items = ", ".join(f'"{f}"' for f in facts)
                self._emit(f"static let facts: [String] = [{items}]")
            self._indent_down()
            self._emit("}")
            self._emit("")
        for proc in procs:
            self._tr_processor(proc)
            self._emit("")
        self._indent_down()
        self._emit("}")
        self._emit("")
        self._emit(f"let {_camel(d.name)}Lobe = {cls}()")

    def _tr_processor(self, p: Any) -> None:
        fn = _camel(p.name)
        params = getattr(p, "params", []) or []
        pl = ", ".join(f"{_camel(pr.name)}: {self._param_type(pr)}" for pr in params)
        rt = self._resolve_type(p.return_type) if getattr(p, "return_type", None) else "NRSIData<String>"
        self._emit(f"func {fn}({pl}) -> {rt} {{")
        self._indent_up()
        body = getattr(p, "body", []) or []
        if body:
            for s in body:
                self._tr_stmt(s)
        else:
            self._emit(f'return NRSIData(value: "", trustLevel: .raw, '
                        f'confidence: 0.5, epistemic: .computational, domain: "{p.name}")')
        self._indent_down()
        self._emit("}")

    # ── compose ──────────────────────────────────────────────────────────

    def _tr_compose(self, d: Any) -> None:
        name = _camel(d.name)
        sources = getattr(d, "sources", []) or []
        src = ", ".join(self._expr(s) for s in sources)
        ann = getattr(d, "annotations", {}) or {}
        kw = ", ".join(f"{k}: {self._expr(v)}" for k, v in ann.items())
        args = f"[{src}]"
        if kw:
            args += f", {kw}"
        self._emit(f"let {name} = NRSICognitive.compose({args})")

    # ── persist ─────────────────────────────────────────────────────────

    def _tr_persist(self, d: Any) -> None:
        name = _camel(d.name)
        config = getattr(d, "config", {}) or {}
        kw = ", ".join(f"{k}: {self._expr(v)}" for k, v in config.items())
        self._emit(f"let {name} = LearnableStore({kw})")

    # ── norm ─────────────────────────────────────────────────────────────

    def _tr_norm(self, d: Any) -> None:
        var = "norm" + _pascal(d.name)
        dt = SWIFT_DEONTIC.get(getattr(d, "deontic_type", "prohibition"), ".prohibition")
        self._emit(f"let {var} = NRSINorm(")
        self._indent_up()
        self._emit(f'id: "{d.name}",')
        self._emit(f"deonticType: {dt},")
        self._emit(f'scope: "{getattr(d, "scope", "global") or "global"}",')
        self._emit(f'action: "{getattr(d, "action", "") or ""}",')
        self._emit(f"priority: {getattr(d, 'priority', 0) or 0},")
        self._emit(f'domain: "{getattr(d, "domain", "") or ""}"')
        self._indent_down()
        self._emit(")")

    # ── belief base ──────────────────────────────────────────────────────

    def _tr_belief(self, d: Any) -> None:
        var = _camel(d.name) + "BeliefBase"
        ent = getattr(d, "entrenchment", "") or "empirical"
        axioms = getattr(d, "axioms", []) or []
        self._emit(f"let {var} = BeliefBase(")
        self._indent_up()
        self._emit(f'name: "{d.name}",')
        self._emit(f'entrenchment: "{ent}",')
        if axioms:
            self._emit("axioms: [")
            self._indent_up()
            for ax in axioms:
                c = str(getattr(ax, "content", "")).replace("\\", "\\\\").replace('"', '\\"')
                t = getattr(ax, "tier", "")
                self._emit(f'Axiom(content: "{c}", tier: "{t}"),')
            self._indent_down()
            self._emit("]")
        else:
            self._emit("axioms: []")
        self._indent_down()
        self._emit(")")

    # ── fn ───────────────────────────────────────────────────────────────

    def _tr_fn(self, d: Any) -> None:
        fn = _camel(d.name)
        params = getattr(d, "params", []) or []
        parts: List[str] = []
        for p in params:
            st = self._param_type(p)
            pn = _camel(p.name)
            if getattr(p, "is_variadic", False):
                st = f"{st}..."
            elif getattr(p, "is_kw_variadic", False):
                st = "[String: Any]"
            if getattr(p, "default", None) is not None:
                parts.append(f"{pn}: {st} = {self._expr(p.default)}")
            else:
                parts.append(f"{pn}: {st}")
        ret_t = ""
        if getattr(d, "return_type", None):
            ret_t = f" -> {self._resolve_type(d.return_type)}"
        elif getattr(d, "is_async", False):
            ret_t = " -> Void"
        async_kw = "async " if getattr(d, "is_async", False) else ""
        self._emit(f"{async_kw}func {fn}({', '.join(parts)}){ret_t} {{")
        self._indent_up()
        if self._type_checked:
            for p in params:
                te = getattr(p, "type_expr", None)
                tl = _ga(te, "trust_level", "base")
                if tl and tl in SWIFT_TRUST_LEVELS:
                    self._emit(f'{_camel(p.name)}.requireTrust({SWIFT_TRUST_LEVELS[tl]}, '
                               f'context: "{d.name}.{p.name}")')
        body = getattr(d, "body", []) or []
        if body:
            for s in body:
                self._tr_stmt(s)
        else:
            self._emit("// empty body")
        self._indent_down()
        self._emit("}")

    # ── const / struct / enum ────────────────────────────────────────────

    def _tr_const(self, d: Any) -> None:
        val = self._expr(d.value) if getattr(d, "value", None) else "nil"
        ta = f": {self._resolve_type(d.type_expr)}" if getattr(d, "type_expr", None) else ""
        self._emit(f"let {_camel(d.name)}{ta} = {val}")

    def _tr_struct(self, d: Any) -> None:
        name = _pascal(d.name)
        self._declared_types.add(name)
        self._emit(f"struct {name} {{")
        self._indent_up()
        for f in (getattr(d, "fields", None) or []):
            ft = self._resolve_type(f.type_expr) if getattr(f, "type_expr", None) else "Any"
            if getattr(f, "default", None) is not None:
                self._emit(f"var {_camel(f.name)}: {ft} = {self._expr(f.default)}")
            else:
                self._emit(f"let {_camel(f.name)}: {ft}")
        self._indent_down()
        self._emit("}")

    def _tr_enum(self, d: Any) -> None:
        name = _pascal(d.name)
        self._declared_types.add(name)
        variants = getattr(d, "variants", None) or getattr(d, "members", []) or []
        variant_values = getattr(d, "variant_values", None) or {}
        string_raw = any(
            isinstance(vex, str) or getattr(vex, "literal_type", "") == "string"
            for vex in (variant_values or {}).values()
        )
        int_raw = any(
            getattr(vex, "literal_type", "") == "int" or isinstance(vex, int)
            for vex in (variant_values or {}).values()
            if not isinstance(vex, bool)
        )
        raw_ann = ""
        if string_raw and not int_raw:
            raw_ann = ": String"
        elif int_raw and not string_raw:
            raw_ann = ": Int"
        self._emit(f"enum {name}{raw_ann} {{")
        self._indent_up()
        if not variants:
            self._emit("// empty enum")
        else:
            for v in variants:
                vname = v if isinstance(v, str) else getattr(v, "name", str(v))
                if vname in variant_values:
                    vex = variant_values[vname]
                    tn = type(vex).__name__
                    if tn in ("LiteralExpr", "Literal") or getattr(vex, "literal_type", None):
                        rhs = self._expr(vex)
                    elif isinstance(vex, str):
                        rhs = '"' + vex.replace("\\", "\\\\").replace('"', '\\"') + '"'
                    elif isinstance(vex, bool):
                        rhs = "true" if vex else "false"
                    else:
                        rhs = str(vex)
                    self._emit(f"case {_camel(vname)} = {rhs}")
                else:
                    self._emit(f"case {_camel(vname)}")
        self._indent_down()
        self._emit("}")

    def _tr_class(self, d: Any) -> None:
        """``class Name(Base) { fields; methods }``"""
        name = _pascal(d.name)
        bases = getattr(d, "base_classes", []) or []
        inh = ""
        if bases:
            inh = ": " + ", ".join(_pascal(b) for b in bases)
        self._emit(f"class {name}{inh} {{")
        self._indent_up()
        for f in getattr(d, "fields", []) or []:
            fn = _camel(getattr(f, "name", "field"))
            ft = self._resolve_type(getattr(f, "type_expr", None)) if getattr(f, "type_expr", None) else "Any"
            if getattr(f, "default", None) is not None:
                self._emit(f"var {fn}: {ft} = {self._expr(f.default)}")
            else:
                self._emit(f"var {fn}: {ft}")
        if getattr(d, "fields", None):
            self._emit("")
        for deco in getattr(d, "decorators", []) or []:
            self._emit(f"// @deprecated attribute: {self._expr(deco)}")
        for m in getattr(d, "methods", []) or []:
            mt = type(m).__name__
            if mt == "FnDecl":
                self._tr_fn(m)
            elif mt == "LetStmt":
                self._tr_stmt(m)
            elif mt in ("AssignStmt", "ExprStmt", "PassStmt"):
                self._tr_stmt(m)
            else:
                self._emit(f"// [SwiftTranspiler] unsupported class member: {mt}")
            self._emit("")
        self._indent_down()
        self._emit("}")

    # ── Statement transpilation ──────────────────────────────────────────

    def _tr_stmt(self, stmt: Any) -> None:
        sn = type(stmt).__name__
        if sn == "ReturnStmt":
            val = getattr(stmt, "value", None)
            self._emit(f"return {self._expr(val)}" if val else "return")
        elif sn == "LetStmt":
            kw = "var" if getattr(stmt, "mutable", False) else "let"
            ta = f": {self._resolve_type(stmt.type_expr)}" if getattr(stmt, "type_expr", None) else ""
            val = getattr(stmt, "value", None)
            if val:
                self._emit(f"{kw} {_camel(stmt.name)}{ta} = {self._expr(val)}")
            elif ta:
                self._emit(f"{kw} {_camel(stmt.name)}{ta}")
            else:
                self._emit(f"{kw} {_camel(stmt.name)}: Any? = nil")
        elif sn == "AssignStmt":
            tgt = getattr(stmt, "target", None)
            if tgt is None:
                return
            ts = _camel(tgt.name) if hasattr(tgt, "name") else (_camel(tgt) if isinstance(tgt, str) else self._expr(tgt))
            self._emit(f"{ts} = {self._expr(getattr(stmt, 'value', None))}")
        elif sn == "ExprStmt":
            e = getattr(stmt, "expr", None)
            if e:
                self._emit(self._expr(e))
        elif sn in ("IfStmt", "IfExpr"):
            self._tr_if(stmt)
        elif sn == "ForStmt":
            self._tr_for(stmt)
        elif sn == "Block":
            for s in getattr(stmt, "stmts", []):
                self._tr_stmt(s)
        elif sn == "RequireStmt":
            c = getattr(stmt, "condition", None)
            if c:
                self._emit(f'precondition({self._expr(c)}, "Requirement failed")')
        elif sn == "ValidateStmt":
            vn = _camel(getattr(stmt, "validator", "validator"))
            args = getattr(stmt, "args", []) or []
            self._emit(f"{vn}({', '.join(self._expr(a) for a in args) if args else 'data'})")
        elif sn == "MatchExpr":
            self._tr_switch(stmt)
        elif sn == "WhileStmt":
            cond = self._expr(getattr(stmt, "condition", None)) if getattr(stmt, "condition", None) else "true"
            self._emit(f"while {cond} {{")
            self._indent_up()
            for s in (getattr(stmt, "body", None) or []):
                self._tr_stmt(s)
            if not getattr(stmt, "body", None):
                self._emit("// empty while")
            self._indent_down()
            self._emit("}")
        elif sn == "TryStmt":
            self._emit("do {")
            self._indent_up()
            fb = getattr(stmt, "finally_body", None) or []
            if fb:
                self._emit("defer {")
                self._indent_up()
                for s in fb:
                    self._tr_stmt(s)
                self._indent_down()
                self._emit("}")
            for s in (getattr(stmt, "body", None) or []):
                self._tr_stmt(s)
            self._indent_down()
            self._emit("} catch {")
            self._indent_up()
            for s in (getattr(stmt, "catch_body", None) or []):
                self._tr_stmt(s)
            self._indent_down()
            self._emit("}")
        elif sn == "BreakStmt":
            self._emit("break")
        elif sn == "ContinueStmt":
            self._emit("continue")
        elif sn == "PassStmt":
            self._emit("// pass")
        elif sn == "RaiseStmt":
            ev = getattr(stmt, "value", None) or getattr(stmt, "expr", None)
            self._emit(f"throw {self._expr(ev)}" if ev else "throw NSError()")
        elif sn == "AugAssignStmt":
            t = self._expr(getattr(stmt, "target", None))
            op = getattr(stmt, "op", "+=")
            v = self._expr(getattr(stmt, "value", None))
            self._emit(f"{t} {op} {v}")
        elif sn == "DelStmt":
            t = getattr(stmt, "target", None)
            self._emit(f"// [SwiftTranspiler] del {self._expr(t) if t else '?'} — map to manual removal")
        elif sn == "ImportDecl":
            self._tr_import(stmt)
        else:
            self._emit(f"// [SwiftTranspiler] unsupported stmt: {sn}")

    def _tr_if(self, s: Any) -> None:
        cond = self._expr(s.condition) if getattr(s, "condition", None) else "true"
        self._emit(f"if {cond} {{")
        self._indent_up()
        tb = getattr(s, "then_body", []) or []
        for st in tb:
            self._tr_stmt(st)
        if not tb:
            self._emit("// empty then")
        self._indent_down()
        eb = getattr(s, "else_body", []) or []
        if eb:
            self._emit("} else {")
            self._indent_up()
            for st in eb:
                self._tr_stmt(st)
            self._indent_down()
        self._emit("}")

    def _tr_for(self, s: Any) -> None:
        var = _camel(getattr(s, "variable", "") or getattr(s, "var", "") or "item")
        it = self._expr(s.iterable) if getattr(s, "iterable", None) else "[]"
        self._emit(f"for {var} in {it} {{")
        self._indent_up()
        body = getattr(s, "body", []) or []
        for st in body:
            self._tr_stmt(st)
        if not body:
            self._emit("// empty loop")
        self._indent_down()
        self._emit("}")

    def _tr_switch(self, e: Any) -> None:
        subj = self._expr(e.subject) if getattr(e, "subject", None) else "nil"
        self._emit(f"switch {subj} {{")
        for arm in (getattr(e, "arms", None) or []):
            pat = self._expr(arm.pattern) if getattr(arm, "pattern", None) else "_"
            self._emit(f"case {pat}:")
            self._indent_up()
            ab = getattr(arm, "body", []) or []
            for s in ab:
                self._tr_stmt(s)
            if not ab:
                self._emit("break")
            self._indent_down()
        self._emit("default:")
        self._indent_up()
        self._emit("break")
        self._indent_down()
        self._emit("}")

    # ── Expression transpilation ─────────────────────────────────────────

    def _expr(self, e: Any) -> str:
        if e is None:
            return "nil"
        en = type(e).__name__
        if en in ("LiteralExpr", "Literal"):
            return self._lit(e)
        if en in ("IdentExpr", "Identifier"):
            raw = getattr(e, "name", "nil")
            return _NRSI_IDENT_REMAP.get(raw, _camel(raw))
        if en in ("BinOpExpr", "BinaryOp"):
            return self._binop_swift(e)
        if en in ("UnaryExpr", "UnaryOp"):
            o = self._expr(getattr(e, "operand", None))
            return f"(!{o})" if e.op in ("not", "!") else f"({e.op}{o})"
        if en == "CallExpr":
            return self._call_swift(e)
        if en in ("FieldAccessExpr", "MemberAccess"):
            obj = _ga(e, "obj", "object")
            mem = _ga(e, "field_name", "member", default="")
            return f"{self._expr(obj)}.{_camel(mem)}"
        if en == "IndexExpr":
            obj = _ga(e, "obj", "object")
            idx = getattr(e, "index", None)
            if idx is not None and type(idx).__name__ == "SliceExpr":
                lo = self._expr(idx.lower) if getattr(idx, "lower", None) else ""
                hi = self._expr(idx.upper) if getattr(idx, "upper", None) else ""
                return f"{self._expr(obj)}[{lo}..<{hi}]" if hi else f"{self._expr(obj)}[{lo}...]"
            return f"{self._expr(obj)}[{self._expr(idx)}]"
        if en == "ListExpr":
            return f"[{', '.join(self._expr(x) for x in getattr(e, 'elements', []))}]"
        if en == "DictExpr":
            ps = ", ".join(f"{self._expr(k)}: {self._expr(v)}" for k, v in (getattr(e, "pairs", None) or []))
            return f"[{ps}]"
        if en in ("IfExpr", "IfStmt"):
            c = self._expr(getattr(e, "condition", None))
            tv = self._body_last(getattr(e, "then_body", None) or [])
            ev = self._body_last(getattr(e, "else_body", None) or [])
            return f"({c} ? {tv} : {ev})"
        if en == "MatchExpr":
            subj = self._expr(getattr(e, "subject", None))
            parts = [f"{subj} == {self._expr(getattr(a, 'pattern', None))} ? "
                     f"{self._body_last(getattr(a, 'body', None) or [])}"
                     for a in (getattr(e, "arms", None) or [])]
            return ("(" + " : ".join(parts) + " : nil)") if parts else "nil"
        if en == "SemanticDistanceExpr":
            l = self._expr(getattr(e, "left", None))
            r = self._expr(getattr(e, "right", None))
            return f"NRSICognitive.semanticDistance({l}, {r})"
        if en == "DecomposeExpr":
            g = self._expr(getattr(e, "goal", None))
            return f"NRSICognitive.decompose({g})"
        if en == "IntentMatchExpr":
            q = self._expr(getattr(e, "query_expr", None))
            bb = self._expr(getattr(e, "belief_base", None))
            return f"NRSICognitive.intentMatch({q}, {bb})"
        if en == "KeywordArg":
            return f"{_camel(e.name)}: {self._expr(e.value)}"
        if en == "SliceExpr":
            lo = self._expr(e.lower) if getattr(e, "lower", None) else ""
            hi = self._expr(e.upper) if getattr(e, "upper", None) else ""
            return f"{lo}..<{hi}" if hi else f"{lo}..."
        if en == "AwaitExpr":
            return f"try await {self._expr(getattr(e, 'value', None))}"
        if en == "TernaryExpr":
            c = self._expr(getattr(e, "condition", None))
            t = self._expr(getattr(e, "then_expr", None))
            el = self._expr(getattr(e, "else_expr", None))
            return f"({c} ? {t} : {el})"
        return f"nil /* unsupported: {en} */"

    def _binop_swift(self, e: Any) -> str:
        op = getattr(e, "op", "")
        left = self._expr(getattr(e, "left", None))
        right = self._expr(getattr(e, "right", None))
        if op == "in":
            return f"({right}).contains({left})"
        if op == "not in":
            return "!" + f"({right}).contains({left})"
        if op == "is":
            inner = getattr(e, "right", None)
            if inner and type(inner).__name__ in ("UnaryExpr", "UnaryOp") and getattr(inner, "op", "") in (
                "not", "!",
            ):
                operand = self._expr(getattr(inner, "operand", None))
                return f"({left} !== {operand})"
            return f"({left} === {right})"
        mapped = SWIFT_BINOP.get(op, op)
        return f"({left} {mapped} {right})"

    def _call_swift(self, e: Any) -> str:
        callee = self._expr(getattr(e, "callee", None))
        parts: List[str] = []
        for a in (getattr(e, "args", None) or []):
            at = type(a).__name__
            if at == "SpreadExpr":
                parts.append(f"/* spread */ {self._expr(getattr(a, 'value', None))}")
            elif at == "KeywordArg":
                parts.append(f"{_camel(a.name)}: {self._expr(a.value)}")
            elif at in ("BinOpExpr", "BinaryOp") and getattr(a, "op", "") == ":":
                key = getattr(getattr(a, "left", None), "name", self._expr(getattr(a, "left", None)))
                parts.append(f"{_camel(key)}: {self._expr(a.right)}")
            else:
                parts.append(self._expr(a))
        return f"{callee}({', '.join(parts)})"

    def _lit(self, e: Any) -> str:
        lt, val = getattr(e, "literal_type", ""), getattr(e, "value", None)
        if lt == "string":
            return '"' + str(val).replace("\\", "\\\\").replace('"', '\\"') + '"'
        if lt == "int":
            return str(int(val))
        if lt == "float":
            return str(float(val))
        if lt == "bool":
            return "true" if val else "false"
        if lt == "none" or val is None:
            return "nil"
        return repr(val)

    def _body_last(self, body: List) -> str:
        if not body:
            return "nil"
        last = body[-1]
        ln = type(last).__name__
        if ln == "ReturnStmt" and getattr(last, "value", None):
            return self._expr(last.value)
        if ln == "ExprStmt" and getattr(last, "expr", None):
            return self._expr(last.expr)
        return "nil"

    # ── Type resolution ──────────────────────────────────────────────────

    def _resolve_type(self, te: Any) -> str:
        if te is None:
            return "Any"
        tn = type(te).__name__
        if tn == "TrustType":
            inner = self._resolve_type(te.inner_type) if getattr(te, "inner_type", None) else "Any"
            return f"NRSIData<{inner}>"
        if tn == "SimpleType":
            return SWIFT_TYPE.get(getattr(te, "name", "Any"), getattr(te, "name", "Any"))
        if tn == "GenericType":
            name, targs = getattr(te, "name", "Any"), getattr(te, "type_args", []) or []
            if name.lower() in ("list", "array"):
                return f"[{self._resolve_type(targs[0])}]" if targs else "[Any]"
            if name.lower() in ("dict", "dictionary"):
                if len(targs) >= 2:
                    return f"[{self._resolve_type(targs[0])}: {self._resolve_type(targs[1])}]"
                return "[String: Any]"
            m = SWIFT_TYPE.get(name, name)
            return f"{m}<{', '.join(self._resolve_type(a) for a in targs)}>" if targs else m
        if tn == "UnionType":
            types = getattr(te, "types", []) or []
            if len(types) == 2:
                r = [self._resolve_type(t) for t in types]
                if "Void" in r:
                    nv = [t for t in r if t != "Void"]
                    return f"{nv[0]}?" if nv else "Any?"
            return "Any"
        if tn == "FunctionType":
            ps = ", ".join(self._resolve_type(p) for p in (getattr(te, "params", None) or []))
            rt = self._resolve_type(getattr(te, "return_type", None)) if getattr(te, "return_type", None) else "Void"
            return f"({ps}) -> {rt}"
        if tn == "TypeExpr":
            base = getattr(te, "base", None) or "Any"
            params = getattr(te, "params", []) or []
            if base in SWIFT_TRUST_LEVELS:
                return f"NRSIData<{self._resolve_type(params[0])}>" if params else "NRSIData<Any>"
            m = SWIFT_TYPE.get(base, base)
            return f"{m}<{', '.join(self._resolve_type(p) for p in params)}>" if params else m
        base = _ga(te, "base", "name", "trust_level")
        if base:
            if base in SWIFT_TRUST_LEVELS:
                inner = getattr(te, "inner_type", None)
                params = getattr(te, "params", None) or getattr(te, "type_args", None) or []
                if inner:
                    return f"NRSIData<{self._resolve_type(inner)}>"
                if params:
                    return f"NRSIData<{self._resolve_type(params[0])}>"
                return "NRSIData<Any>"
            return SWIFT_TYPE.get(base, base)
        return "Any"

    def _param_type(self, p: Any) -> str:
        te = getattr(p, "type_expr", None)
        return self._resolve_type(te) if te else "Any"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Convenience API
# ═══════════════════════════════════════════════════════════════════════════════

def transpile(module: Any, *, type_checked: bool = True) -> str:
    """Transpile an NRSI ``Module`` AST to Swift source code."""
    return SwiftTranspiler(type_checked=type_checked).transpile(module)


__all__ = ["SwiftTranspiler", "transpile"]
