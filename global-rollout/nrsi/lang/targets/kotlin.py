"""NRSI Kotlin Transpiler — AST to Kotlin Code Generation.

Converts a type-checked NRSI AST into idiomatic Kotlin source code that uses
sealed classes for trust levels, data classes for NRSIData, and require()
for gate enforcement.

Each NRSI construct maps to specific Kotlin patterns:
  trust x: validated[string]  → val x = NRSIData(...)
  gate verify(d: raw[T])      → fun verify(...) with require() checks
  lobe logical { ... }        → class LogicalLobe { ... }
  norm no_medical              → val normNoMedical = NRSINorm(...)
  belief base facts            → val factsBeliefBase = BeliefBase(...)
  fn name(params) -> ret       → fun name(params): KotlinType { ... }
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Set


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Lookup Tables
# ═══════════════════════════════════════════════════════════════════════════════

KT_TRUST_LEVELS: Dict[str, str] = {
    "raw": "TrustLevel.RAW", "validated": "TrustLevel.VALIDATED",
    "trusted": "TrustLevel.TRUSTED", "certified": "TrustLevel.CERTIFIED",
}
KT_DEONTIC: Dict[str, str] = {
    "obligation": "DeonticType.OBLIGATION", "permission": "DeonticType.PERMISSION",
    "prohibition": "DeonticType.PROHIBITION", "exemption": "DeonticType.EXEMPTION",
}
KT_BINOP: Dict[str, str] = {
    "+": "+", "-": "-", "*": "*", "/": "/", "%": "%",
    "==": "==", "!=": "!=", "<": "<", ">": ">", "<=": "<=", ">=": ">=",
    "and": "&&", "or": "||", "&&": "&&", "||": "||",
}
KT_TYPE: Dict[str, str] = {
    "string": "String", "int": "Int", "float": "Double", "bool": "Boolean",
    "none": "Unit", "void": "Unit", "list": "List", "dict": "Map", "any": "Any",
    "String": "String", "Int": "Int", "Float": "Double",
    "Double": "Double", "Bool": "Boolean", "Boolean": "Boolean",
}
KT_LOBE: Dict[str, str] = {
    "linguistic": "LinguisticLobe", "logical": "LogicalLobe",
    "mathematical": "MathematicalLobe", "spatial": "SpatialLobe",
    "temporal": "TemporalLobe", "creative": "CreativeLobe",
    "causal": "CausalLobe", "analogical": "AnalogicalLobe",
    "planning": "PlanningLobe", "memory": "MemoryLobe",
    "metacognitive": "MetacognitiveLobe",
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Kotlin Runtime Preamble
# ═══════════════════════════════════════════════════════════════════════════════

_PREAMBLE = '''\
package ai.velariq.nrsi

// NRSI Trust Type System
enum class TrustLevel(val rank: Int) {
    RAW(0), VALIDATED(1), TRUSTED(2), CERTIFIED(3);
    operator fun compareTo(other: TrustLevel): Int = rank.compareTo(other.rank)
}

enum class EpistemicType {
    DEDUCTIVE, INDUCTIVE, ABDUCTIVE, ANALOGICAL, CAUSAL,
    COMPUTATIONAL, OBSERVATIONAL, TESTIMONIAL, CREATIVE, SPECULATIVE
}

enum class DeonticType {
    OBLIGATION, PERMISSION, PROHIBITION, EXEMPTION
}

data class NRSIData<T>(
    val value: T,
    var trustLevel: TrustLevel = TrustLevel.RAW,
    var confidence: Double = 0.0,
    var epistemic: EpistemicType = EpistemicType.OBSERVATIONAL,
    var domain: String = "general"
) {
    fun requireTrust(required: TrustLevel, context: String = "") {
        require(trustLevel.rank >= required.rank) {
            "Trust violation in $context: have $trustLevel, need $required"
        }
    }

    fun withTrust(level: TrustLevel): NRSIData<T> = copy(trustLevel = level)
}

data class NRSINorm(
    val id: String,
    val deonticType: DeonticType,
    val scope: String = "",
    val action: String = "",
    val priority: Int = 0,
    val domain: String = ""
)

data class Axiom(val content: String, val tier: String = "")

data class BeliefBase(
    val name: String,
    val entrenchment: String = "",
    val axioms: List<Axiom> = emptyList()
)

// ---- Generated Declarations ----
'''

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Helpers
# ═══════════════════════════════════════════════════════════════════════════════

_SNAKE_RE = re.compile(r"_([a-z])")


def _camel(name: str) -> str:
    """``snake_case`` -> ``camelCase``."""
    if not name or "_" not in name:
        return name
    prefix, stripped = "", name
    while stripped.startswith("_"):
        prefix += "_"
        stripped = stripped[1:]
    if not stripped:
        return name
    return prefix + _SNAKE_RE.sub(lambda m: m.group(1).upper(), stripped)


def _pascal(name: str) -> str:
    """``snake_case`` -> ``PascalCase``."""
    return "".join(p.capitalize() for p in name.split("_") if p)


def _ga(obj: Any, *keys: str, default: Any = None) -> Any:
    """Chain-fallback getattr across multiple attribute names."""
    for k in keys:
        v = getattr(obj, k, None)
        if v is not None:
            return v
    return default


_NRSI_IDENT_REMAP: Dict[str, str] = {
    "null": "null",
    "true": "true",
    "false": "false",
}


def _kotlin_import_path(path: str) -> str:
    p = path.strip().replace("/", ".").replace("-", "_")
    if p.startswith("std."):
        p = p[4:]
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# 4. KotlinTranspiler
# ═══════════════════════════════════════════════════════════════════════════════

class KotlinTranspiler:
    """Convert NRSI AST to Kotlin source code.

    Handles AST nodes from both the parser module and the transpiler
    module's internal definitions by dispatching on type(node).__name__.
    """

    def __init__(self, type_checked: bool = True) -> None:
        self._type_checked = type_checked
        self._indent = 0
        self._output: List[str] = []
        self._declared_types: Set[str] = set()

    def transpile(self, module: Any) -> str:
        """Transpile a complete NRSI module to Kotlin source."""
        self._indent = 0
        self._output = []
        self._declared_types = set()

        name = _ga(module, "name", "source_file", default="<nrsi>")
        self._emit(f"// Auto-generated Kotlin from NRSI source: {name}")
        self._emit("// Generated by the NRSI KotlinTranspiler. Do not edit directly.")
        self._emit("// Trust enforcement is active at runtime via require().")
        self._emit("")

        for line in _PREAMBLE.splitlines():
            self._output.append(line)
        self._emit("")

        for imp in (_ga(module, "imports", default=[]) or []):
            self._tr_import(imp)

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
            "TrustDecl": self._tr_trust,
            "GateDecl": self._tr_gate,
            "LobeDecl": self._tr_lobe,
            "NormDecl": self._tr_norm,
            "BeliefBaseDecl": self._tr_belief,
            "FnDecl": self._tr_fn,
            "StructDecl": self._tr_struct,
            "EnumDecl": self._tr_enum,
            "ClassDecl": self._tr_class,
            "ComposeDecl": self._tr_compose,
            "PersistDecl": self._tr_persist,
            "ConstDecl": self._tr_const,
            "CreaseDecl": self._tr_crease,
        }
        h = dispatch.get(dt)
        if h:
            h(decl)
        elif dt in ("LetStmt", "AssignStmt", "ExprStmt"):
            self._tr_stmt(decl)
        elif dt == "ImportDecl":
            self._tr_import(decl)
        else:
            self._emit(f"// [KotlinTranspiler] unsupported: {type(decl).__name__}")

    # ── trust ────────────────────────────────────────────────────────────

    def _tr_trust(self, d: Any) -> None:
        name = _camel(d.name)
        trust = KT_TRUST_LEVELS.get(getattr(d, "trust_level", "raw"), "TrustLevel.RAW")
        val = self._expr(d.value) if getattr(d, "value", None) else '""'
        vtype = _ga(d, "value_type", "inner_type")
        inner = self._resolve_type(vtype) if vtype else "String"
        ann: Dict = getattr(d, "annotations", {}) or {}
        conf = self._ann_float(ann, "confidence", getattr(d, "confidence", 0.0))
        ep = self._ann_str(ann, "epistemic", getattr(d, "epistemic", "") or "observational")
        dom = self._ann_str(ann, "domain", "general")
        self._emit(f"val {name} = NRSIData<{inner}>(")
        self._indent_up()
        self._emit(f"value = {val},")
        self._emit(f"trustLevel = {trust},")
        self._emit(f"confidence = {conf},")
        self._emit(f"epistemic = EpistemicType.{ep.upper()},")
        self._emit(f'domain = "{dom}"')
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
        """``import path`` / ``from path import a`` → Kotlin ``import`` lines."""
        path = (_ga(decl, "module_path", "module", default="") or "").strip()
        if not path:
            self._emit("// [KotlinTranspiler] empty import")
            return
        if path.startswith("std/"):
            path = path[4:].replace("/", ".")
        kpath = _kotlin_import_path(path)
        names = getattr(decl, "names", None) or []
        alias = getattr(decl, "alias", "") or ""
        self._emit(f"// NRSI import: {kpath}" + (f" as {alias}" if alias else ""))
        if not names:
            self._emit(f"import {kpath}.*")
        else:
            for n in names:
                self._emit(f"import {kpath}.{n}")
        if alias:
            self._emit(f"// NOTE: Kotlin import alias requires per-declaration `as` syntax: import {kpath}.X as {alias}")

    # ── gate ─────────────────────────────────────────────────────────────

    def _tr_gate(self, d: Any) -> None:
        fn = _camel(d.name)
        params = getattr(d, "params", []) or []
        pl = ", ".join(f"{_camel(p.name)}: {self._param_type(p)}" for p in params)
        ret = getattr(d, "return_type", None)
        rt = self._resolve_type(ret) if ret else "NRSIData<String>"

        target = "TrustLevel.VALIDATED"
        if ret:
            base = _ga(ret, "trust_level", "base")
            if base and base in KT_TRUST_LEVELS:
                target = KT_TRUST_LEVELS[base]

        conf, validators = self._gate_reqs(d)
        self._emit(
            f"// NRSI trust gate: `{d.name}` confidence>={conf} "
            f"validators={validators!r} outputTrust={target}"
        )
        self._emit(f"fun {fn}({pl}): {rt} {{")
        self._indent_up()

        fp = _camel(params[0].name) if params else "data"
        self._emit(f'require({fp}.confidence >= {conf}) {{ "Gate \'{d.name}\': confidence below threshold" }}')

        for v in validators:
            if v:
                self._emit(f"{_camel(v)}({fp})")

        has_extra = False
        for stmt in (getattr(d, "body", None) or []):
            if type(stmt).__name__ not in ("RequireStmt", "ValidateStmt"):
                self._tr_stmt(stmt)
                has_extra = True

        if not has_extra:
            self._emit(f"return {fp}.copy(trustLevel = {target})")

        self._indent_down()
        self._emit("}")

    def _gate_reqs(self, d: Any) -> tuple:
        conf = 0.95
        validators: List[str] = []
        reqs = getattr(d, "requires", None)
        if reqs is not None:
            for r in reqs:
                if getattr(r, "field", "") == "confidence":
                    conf = float(getattr(r, "value", conf))
            validators = list(getattr(d, "validators", []) or [])
            return conf, validators
        for stmt in (getattr(d, "body", None) or []):
            stype = type(stmt).__name__
            if stype == "RequireStmt":
                cond = getattr(stmt, "condition", None)
                if cond and type(cond).__name__ in ("BinaryOp", "BinOpExpr"):
                    left_name = _ga(getattr(cond, "left", None), "name", default="")
                    if left_name == "confidence" and getattr(cond, "op", "") in (">=", ">"):
                        rv = getattr(getattr(cond, "right", None), "value", None)
                        if rv is not None:
                            conf = float(rv)
            elif stype == "ValidateStmt":
                validators.append(getattr(stmt, "validator", ""))
        return conf, validators

    # ── compose ──────────────────────────────────────────────────────────

    def _tr_compose(self, d: Any) -> None:
        name = _camel(d.name)
        sources = getattr(d, "sources", []) or []
        src = ", ".join(self._expr(s) for s in sources)
        ann = getattr(d, "annotations", {}) or {}
        kw = ", ".join(f"{k} = {self._expr(v)}" for k, v in ann.items())
        args = f"listOf({src})"
        if kw:
            args += f", {kw}"
        self._emit(f"val {name} = NRSICognitive.compose({args})")

    # ── persist ─────────────────────────────────────────────────────────

    def _tr_persist(self, d: Any) -> None:
        name = _camel(d.name)
        config = getattr(d, "config", {}) or {}
        kw = ", ".join(f"{k} = {self._expr(v)}" for k, v in config.items())
        self._emit(f"val {name} = LearnableStore({kw})")

    # ── lobe ─────────────────────────────────────────────────────────────

    def _tr_lobe(self, d: Any) -> None:
        cls = KT_LOBE.get(d.name, _pascal(d.name) + "Lobe")
        self._emit(f"class {cls} {{")
        self._indent_up()

        for proc in (getattr(d, "processors", None) or []):
            self._tr_processor(proc)
            self._emit("")

        for crease in (getattr(d, "creases", None) or []):
            self._tr_crease(crease)
            self._emit("")

        self._indent_down()
        self._emit("}")
        self._emit(f"val {_camel(d.name)}Lobe = {cls}()")

    def _tr_processor(self, p: Any) -> None:
        fn = _camel(p.name)
        params = getattr(p, "params", []) or []
        pl = ", ".join(f"{_camel(pr.name)}: {self._param_type(pr)}" for pr in params)
        ret = getattr(p, "return_type", None)
        rt = self._resolve_type(ret) if ret else "NRSIData<String>"

        self._emit(f"fun {fn}({pl}): {rt} {{")
        self._indent_up()
        body = getattr(p, "body", None) or []
        if body:
            for stmt in body:
                self._tr_stmt(stmt)
        else:
            self._emit(f'return NRSIData(value = "", trustLevel = TrustLevel.RAW, '
                        f'confidence = 0.5, epistemic = EpistemicType.COMPUTATIONAL, domain = "{p.name}")')
        self._indent_down()
        self._emit("}")

    # ── norm ─────────────────────────────────────────────────────────────

    def _tr_norm(self, d: Any) -> None:
        var = _camel("norm_" + d.name)
        dt = getattr(d, "deontic_type", "prohibition")
        deontic = KT_DEONTIC.get(dt, "DeonticType.PROHIBITION")
        scope = getattr(d, "scope", "global")
        action = getattr(d, "action", "")
        priority = getattr(d, "priority", 0)
        domain = getattr(d, "domain", "")

        self._emit(f"val {var} = NRSINorm(")
        self._indent_up()
        self._emit(f'id = "{d.name}",')
        self._emit(f"deonticType = {deontic},")
        self._emit(f'scope = "{scope}",')
        self._emit(f'action = "{action}",')
        self._emit(f"priority = {priority},")
        self._emit(f'domain = "{domain}"')
        self._indent_down()
        self._emit(")")

    # ── belief base ──────────────────────────────────────────────────────

    def _tr_belief(self, d: Any) -> None:
        var = _camel(d.name + "_belief_base")
        ent = getattr(d, "entrenchment", "")
        axioms = getattr(d, "axioms", []) or []

        self._emit(f"val {var} = BeliefBase(")
        self._indent_up()
        self._emit(f'name = "{d.name}",')
        self._emit(f'entrenchment = "{ent}",')
        if axioms:
            self._emit("axioms = listOf(")
            self._indent_up()
            for ax in axioms:
                c = getattr(ax, "content", "")
                t = getattr(ax, "tier", "")
                escaped = c.replace("\\", "\\\\").replace('"', '\\"')
                self._emit(f'Axiom(content = "{escaped}", tier = "{t}"),')
            self._indent_down()
            self._emit(")")
        else:
            self._emit("axioms = emptyList()")
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
                parts.append(f"vararg {pn}: {st}")
            elif getattr(p, "is_kw_variadic", False):
                parts.append(f"vararg {pn}: Pair<String, Any>")
            elif getattr(p, "default", None) is not None:
                parts.append(f"{pn}: {st} = {self._expr(p.default)}")
            else:
                parts.append(f"{pn}: {st}")
        pl = ", ".join(parts)
        ret = getattr(d, "return_type", None)
        rt_str = f": {self._resolve_type(ret)}" if ret else (
            ": Unit" if getattr(d, "is_async", False) else ""
        )
        sus = "suspend " if getattr(d, "is_async", False) else ""
        self._emit(f"{sus}fun {fn}({pl}){rt_str} {{")
        self._indent_up()

        body = getattr(d, "body", []) or []
        if body:
            for stmt in body:
                self._tr_stmt(stmt)
        else:
            self._emit("// no body")

        self._indent_down()
        self._emit("}")

    # ── struct / enum ────────────────────────────────────────────────────

    def _tr_struct(self, d: Any) -> None:
        name = _pascal(d.name)
        self._declared_types.add(name)
        fields = getattr(d, "fields", []) or []
        self._emit(f"data class {name}(")
        self._indent_up()
        for i, f in enumerate(fields):
            ft = self._resolve_type(getattr(f, "type_expr", None)) if getattr(f, "type_expr", None) else "Any"
            comma = "," if i < len(fields) - 1 else ""
            self._emit(f"val {_camel(f.name)}: {ft}{comma}")
        self._indent_down()
        self._emit(")")

    def _tr_enum(self, d: Any) -> None:
        name = _pascal(d.name)
        self._declared_types.add(name)
        variants = getattr(d, "variants", []) or getattr(d, "members", []) or []
        variant_values = getattr(d, "variant_values", None) or {}
        self._emit(f"enum class {name} {{")
        self._indent_up()
        if not variants:
            self._emit("// empty")
        else:
            for i, v in enumerate(variants):
                vname = v if isinstance(v, str) else getattr(v, "name", str(v))
                c = "," if i < len(variants) - 1 else ""
                if vname in variant_values:
                    vex = variant_values[vname]
                    rhs = self._enum_variant_value_expr(vex)
                    self._emit(f"// NRSI raw/associated value: {vname.upper()} = {rhs}")
                self._emit(f"{vname.upper()}{c}")
        self._indent_down()
        self._emit("}")

    def _enum_variant_value_expr(self, vex: Any) -> str:
        tn = type(vex).__name__
        if tn in ("LiteralExpr", "Literal") or getattr(vex, "literal_type", None):
            return self._literal(vex)
        if isinstance(vex, str):
            return '"' + vex.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$") + '"'
        return str(vex)

    def _tr_class(self, d: Any) -> None:
        """``class Name(bases) { ... }`` → Kotlin open class with constructor stubs."""
        name = _pascal(d.name)
        bases = getattr(d, "base_classes", []) or []
        hdr = f"open class {name}"
        if bases:
            hdr += " : " + ", ".join(f"{_pascal(b)}()" for b in bases)
        self._emit(f"{hdr} {{")
        self._indent_up()
        for f in getattr(d, "fields", []) or []:
            fn = _camel(getattr(f, "name", "field"))
            ft = self._resolve_type(getattr(f, "type_expr", None)) if getattr(f, "type_expr", None) else "Any"
            if getattr(f, "default", None) is not None:
                self._emit(f"var {fn}: {ft} = {self._expr(f.default)}")
            else:
                self._emit(f"var {fn}: {ft}? = null")
        if getattr(d, "fields", None):
            self._emit("")
        for m in getattr(d, "methods", []) or []:
            mt = type(m).__name__
            if mt == "FnDecl":
                self._tr_fn(m)
            elif mt in ("LetStmt", "AssignStmt", "ExprStmt", "PassStmt"):
                self._tr_stmt(m)
            else:
                self._emit(f"// [KotlinTranspiler] unsupported class member: {mt}")
            self._emit("")
        self._indent_down()
        self._emit("}")

    def _tr_const(self, d: Any) -> None:
        val = self._expr(d.value) if getattr(d, "value", None) else "null"
        ta = f": {self._resolve_type(d.type_expr)}" if getattr(d, "type_expr", None) else ""
        self._emit(f"val {_camel(d.name)}{ta} = {val}")

    def _tr_crease(self, d: Any) -> None:
        domain = getattr(d, "domain", "general")
        facts = getattr(d, "facts", []) or []
        name = _pascal(f"crease_{domain}")
        self._emit(f"object {name} {{")
        self._indent_up()
        self._emit(f'val domain = "{domain}"')
        if facts:
            escaped = [f.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$") for f in facts]
            items = ", ".join(f'"{e}"' for e in escaped)
            self._emit(f"val facts = listOf({items})")
        else:
            self._emit("val facts = emptyList<String>()")
        self._indent_down()
        self._emit("}")

    # ── statements ───────────────────────────────────────────────────────

    def _tr_stmt(self, stmt: Any) -> None:
        stype = type(stmt).__name__

        if stype == "ReturnStmt":
            val = getattr(stmt, "value", None)
            if val:
                self._emit(f"return {self._expr(val)}")
            else:
                self._emit("return")

        elif stype == "LetStmt":
            val = self._expr(stmt.value) if getattr(stmt, "value", None) else "null"
            kw = "var" if getattr(stmt, "mutable", False) else "val"
            te = getattr(stmt, "type_expr", None)
            annotation = f": {self._resolve_type(te)}" if te else ""
            self._emit(f"{kw} {_camel(stmt.name)}{annotation} = {val}")

        elif stype == "AssignStmt":
            target = getattr(stmt, "target", None)
            tname = _camel(_ga(target, "name", default="x") if target and hasattr(target, "name") else str(target))
            val = self._expr(getattr(stmt, "value", None))
            self._emit(f"{tname} = {val}")

        elif stype == "ExprStmt":
            expr = getattr(stmt, "expr", None)
            if expr:
                self._emit(self._expr(expr))

        elif stype in ("IfStmt", "IfExpr"):
            cond = getattr(stmt, "condition", None)
            self._emit(f"if ({self._expr(cond)}) {{")
            self._indent_up()
            for s in (getattr(stmt, "then_body", None) or []):
                self._tr_stmt(s)
            self._indent_down()
            else_body = getattr(stmt, "else_body", None) or []
            if else_body:
                self._emit("} else {")
                self._indent_up()
                for s in else_body:
                    self._tr_stmt(s)
                self._indent_down()
            self._emit("}")

        elif stype == "ForStmt":
            var = _camel(getattr(stmt, "variable", None) or getattr(stmt, "var", None) or "item")
            iterable = self._expr(getattr(stmt, "iterable", None))
            self._emit(f"for ({var} in {iterable}) {{")
            self._indent_up()
            for s in (getattr(stmt, "body", None) or []):
                self._tr_stmt(s)
            self._indent_down()
            self._emit("}")

        elif stype == "RequireStmt":
            cond = getattr(stmt, "condition", None)
            if cond:
                self._emit(f"require({self._expr(cond)})")

        elif stype == "ValidateStmt":
            vname = _camel(getattr(stmt, "validator", "validate"))
            self._emit(f"{vname}(data)")

        elif stype == "MatchExpr":
            self._tr_when(stmt)

        elif stype == "Block":
            for s in (getattr(stmt, "stmts", None) or []):
                self._tr_stmt(s)

        elif stype == "WhileStmt":
            cond = self._expr(getattr(stmt, "condition", None)) if getattr(stmt, "condition", None) else "true"
            self._emit(f"while ({cond}) {{")
            self._indent_up()
            for s in (getattr(stmt, "body", None) or []):
                self._tr_stmt(s)
            self._indent_down()
            self._emit("}")

        elif stype == "TryStmt":
            self._emit("try {")
            self._indent_up()
            for s in (getattr(stmt, "body", None) or []):
                self._tr_stmt(s)
            self._indent_down()
            self._emit("}")
            cb = getattr(stmt, "catch_body", None) or []
            cv = getattr(stmt, "catch_var", "") or "e"
            self._emit(f"catch ({_camel(cv)}: Exception) {{")
            self._indent_up()
            for s in cb:
                self._tr_stmt(s)
            self._indent_down()
            self._emit("}")
            fb = getattr(stmt, "finally_body", None) or []
            if fb:
                self._emit("finally {")
                self._indent_up()
                for s in fb:
                    self._tr_stmt(s)
                self._indent_down()
                self._emit("}")

        elif stype == "BreakStmt":
            self._emit("break")

        elif stype == "ContinueStmt":
            self._emit("continue")

        elif stype == "PassStmt":
            self._emit("// pass")

        elif stype == "RaiseStmt":
            ev = getattr(stmt, "value", None) or getattr(stmt, "expr", None)
            self._emit(f"throw {self._expr(ev)}" if ev else "throw RuntimeException()")

        elif stype == "AugAssignStmt":
            t = self._expr(getattr(stmt, "target", None))
            op = getattr(stmt, "op", "+=")
            v = self._expr(getattr(stmt, "value", None))
            self._emit(f"{t} {op} {v}")

        elif stype == "DelStmt":
            t = getattr(stmt, "target", None)
            self._emit(f"// [KotlinTranspiler] del {self._expr(t) if t else '?'} — not mapped")

        elif stype == "ImportDecl":
            self._tr_import(stmt)

        else:
            self._emit(f"// [KotlinTranspiler] unsupported stmt: {stype}")

    def _tr_when(self, e: Any) -> None:
        subj = self._expr(getattr(e, "subject", None))
        self._emit(f"when ({subj}) {{")
        self._indent_up()
        for arm in (getattr(e, "arms", None) or []):
            pat = self._expr(getattr(arm, "pattern", None))
            ab = getattr(arm, "body", []) or []
            if len(ab) == 1:
                sn = type(ab[0]).__name__
                if sn == "ExprStmt" and getattr(ab[0], "expr", None):
                    self._emit(f"{pat} -> {self._expr(ab[0].expr)}")
                    continue
                if sn == "ReturnStmt" and getattr(ab[0], "value", None):
                    self._emit(f"{pat} -> {self._expr(ab[0].value)}")
                    continue
            self._emit(f"{pat} -> {{")
            self._indent_up()
            for s in ab:
                self._tr_stmt(s)
            if not ab:
                self._emit("Unit")
            self._indent_down()
            self._emit("}")
        self._emit("else -> {}")
        self._indent_down()
        self._emit("}")

    # ── expressions ──────────────────────────────────────────────────────

    def _expr(self, expr: Any) -> str:
        if expr is None:
            return "null"

        etype = type(expr).__name__

        if etype in ("Literal", "LiteralExpr"):
            return self._literal(expr)
        if etype in ("Identifier", "IdentExpr"):
            raw = _ga(expr, "name", default="null")
            return _NRSI_IDENT_REMAP.get(raw, _camel(raw))
        if etype in ("BinaryOp", "BinOpExpr"):
            return self._binop_kt(expr)
        if etype in ("UnaryOp", "UnaryExpr"):
            operand = self._expr(getattr(expr, "operand", None))
            op = getattr(expr, "op", "!")
            if op in ("not", "!"):
                return f"(!{operand})"
            return f"({op}{operand})"
        if etype == "CallExpr":
            return self._call_kt(expr)
        if etype in ("MemberAccess", "FieldAccessExpr"):
            obj = self._expr(_ga(expr, "object", "obj"))
            member = _camel(_ga(expr, "member", "field_name", default=""))
            return f"{obj}.{member}"
        if etype == "IndexExpr":
            obj = self._expr(_ga(expr, "object", "obj"))
            idxn = getattr(expr, "index", None)
            if idxn is not None and type(idxn).__name__ == "SliceExpr":
                lo = self._expr(idxn.lower) if getattr(idxn, "lower", None) else "0"
                hi = self._expr(idxn.upper) if getattr(idxn, "upper", None) else ""
                return f"{obj}.slice({lo} until {hi})" if hi else f"{obj}.drop({lo})"
            return f"{obj}[{self._expr(idxn)}]"

        if etype == "SemanticDistanceExpr":
            l = self._expr(getattr(expr, "left", None))
            r = self._expr(getattr(expr, "right", None))
            return f"NRSICognitive.semanticDistance({l}, {r})"
        if etype == "DecomposeExpr":
            g = self._expr(getattr(expr, "goal", None))
            return f"NRSICognitive.decompose({g})"
        if etype == "IntentMatchExpr":
            q = self._expr(getattr(expr, "query_expr", None))
            bb = self._expr(getattr(expr, "belief_base", None))
            return f"NRSICognitive.intentMatch({q}, {bb})"
        if etype == "ListExpr":
            elements = ", ".join(self._expr(x) for x in getattr(expr, "elements", []))
            return f"listOf({elements})"
        if etype == "DictExpr":
            pairs = ", ".join(
                f"{self._expr(k)} to {self._expr(v)}"
                for k, v in (getattr(expr, "pairs", None) or [])
            )
            return f"mapOf({pairs})"
        if etype in ("IfExpr", "IfStmt"):
            cond = self._expr(getattr(expr, "condition", None))
            tv = self._body_last(getattr(expr, "then_body", None) or [])
            ev = self._body_last(getattr(expr, "else_body", None) or [])
            return f"if ({cond}) {tv} else {ev}"
        if etype == "MatchExpr":
            subj = self._expr(getattr(expr, "subject", None))
            parts = []
            for arm in (getattr(expr, "arms", None) or []):
                pat = self._expr(getattr(arm, "pattern", None))
                val = self._body_last(getattr(arm, "body", None) or [])
                parts.append(f"{pat} -> {val}")
            inner = "; ".join(parts) + "; else -> null" if parts else "else -> null"
            return f"when ({subj}) {{ {inner} }}"
        if etype == "KeywordArg":
            return f"{_camel(expr.name)} = {self._expr(expr.value)}"
        if etype == "SliceExpr":
            lo = self._expr(expr.lower) if getattr(expr, "lower", None) else "0"
            hi = self._expr(expr.upper) if getattr(expr, "upper", None) else ""
            return f"({lo} until {hi})" if hi else f"({lo}..Int.MAX_VALUE)"
        if etype == "AwaitExpr":
            return self._expr(getattr(expr, "value", None))
        if etype == "TernaryExpr":
            c = self._expr(getattr(expr, "condition", None))
            t = self._expr(getattr(expr, "then_expr", None))
            el = self._expr(getattr(expr, "else_expr", None))
            return f"if ({c}) {t} else {el}"
        return f"null /* unsupported: {etype} */"

    def _binop_kt(self, expr: Any) -> str:
        op = getattr(expr, "op", "")
        left = self._expr(getattr(expr, "left", None))
        right = self._expr(getattr(expr, "right", None))
        if op == ":":
            return f"({left} to {right})"
        if op == "in":
            return f"({left} in {right})"
        if op == "not in":
            return f"({left} !in {right})"
        if op == "is":
            inner = getattr(expr, "right", None)
            if inner and type(inner).__name__ in ("UnaryOp", "UnaryExpr") and getattr(inner, "op", "") in (
                "not", "!",
            ):
                operand = self._expr(getattr(inner, "operand", None))
                return f"({left} !is {operand})"
            return f"({left} is {right})"
        if op == "**":
            return f"Math.pow({left}.toDouble(), {right}.toDouble())"
        if op == "~~":
            return f"({left} / {right})"
        if op == "<<":
            return f"({left} shl {right})"
        if op == ">>":
            return f"({left} shr {right})"
        if op == "^":
            return f"({left} xor {right})"
        if op == "&":
            return f"({left} and {right})"
        if op == "|":
            return f"({left} or {right})"
        mapped = KT_BINOP.get(op, op)
        return f"({left} {mapped} {right})"

    def _call_kt(self, expr: Any) -> str:
        callee = self._expr(getattr(expr, "callee", None))
        parts: List[str] = []
        for a in (getattr(expr, "args", None) or []):
            at = type(a).__name__
            if at == "SpreadExpr":
                parts.append(f"*{self._expr(getattr(a, 'value', None))}")
            elif at == "KeywordArg":
                parts.append(f"{_camel(a.name)} = {self._expr(a.value)}")
            elif at in ("BinOpExpr", "BinaryOp") and getattr(a, "op", "") == ":":
                key = getattr(getattr(a, "left", None), "name", self._expr(getattr(a, "left", None)))
                parts.append(f"{_camel(key)} = {self._expr(a.right)}")
            else:
                parts.append(self._expr(a))
        return f"{callee}({', '.join(parts)})"

    def _literal(self, expr: Any) -> str:
        lt = getattr(expr, "literal_type", "")
        val = getattr(expr, "value", None)
        if lt == "string":
            escaped = str(val).replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")
            return f'"{escaped}"'
        if lt == "int":
            return str(int(val))
        if lt == "float":
            return str(float(val))
        if lt == "bool":
            return "true" if val else "false"
        if lt in ("none", "null"):
            return "null"
        return repr(val)

    def _body_last(self, body: list) -> str:
        if not body:
            return "null"
        last = body[-1]
        ln = type(last).__name__
        if ln == "ReturnStmt" and getattr(last, "value", None):
            return self._expr(last.value)
        if ln == "ExprStmt" and getattr(last, "expr", None):
            return self._expr(last.expr)
        return "null"

    # ── type resolution ──────────────────────────────────────────────────

    def _resolve_type(self, te: Any) -> str:
        if te is None:
            return "Any"
        tname = type(te).__name__

        base = _ga(te, "trust_level", "base", "name", default=None)
        inner = _ga(te, "inner_type", default=None)
        params = _ga(te, "type_args", "params", default=None) or []

        if base in ("raw", "validated", "trusted", "certified"):
            inner_t = self._resolve_type(inner) if inner else (
                self._resolve_type(params[0]) if params else "String"
            )
            return f"NRSIData<{inner_t}>"

        if base and base.lower() in ("list",):
            inner_t = self._resolve_type(params[0]) if params else "Any"
            return f"List<{inner_t}>"

        if base and base.lower() in ("dict", "map"):
            k = self._resolve_type(params[0]) if len(params) > 0 else "String"
            v = self._resolve_type(params[1]) if len(params) > 1 else "Any"
            return f"Map<{k}, {v}>"

        if tname == "UnionType":
            types = getattr(te, "types", []) or []
            if len(types) == 2:
                resolved = [self._resolve_type(t) for t in types]
                if "Unit" in resolved:
                    nv = [t for t in resolved if t != "Unit"]
                    return f"{nv[0]}?" if nv else "Any?"
            return "Any"

        if tname == "FunctionType":
            fparams = getattr(te, "params", []) or []
            ps = ", ".join(self._resolve_type(p) for p in fparams)
            fret = self._resolve_type(getattr(te, "return_type", None)) if getattr(te, "return_type", None) else "Unit"
            return f"({ps}) -> {fret}"

        if base:
            return KT_TYPE.get(base, _pascal(base) if base[0].islower() else base)

        return "Any"

    def _param_type(self, p: Any) -> str:
        te = getattr(p, "type_expr", None)
        return self._resolve_type(te) if te else "Any"
