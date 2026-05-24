"""NRSI source formatter — AST to pretty-printed .nrsi text."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, List, Optional

from nrsi.lang.parser import (
    ASTNode,
    AssertStmt,
    AssignStmt,
    AugAssignStmt,
    AwaitExpr,
    AxiomDecl,
    BeliefBaseDecl,
    BinaryOp,
    BreakStmt,
    CallExpr,
    ClassDecl,
    ComprehensionExpr,
    ComposeDecl,
    ContinueStmt,
    CreaseDecl,
    DecomposeExpr,
    DelStmt,
    DictExpr,
    EnumDecl,
    Expr,
    ExprStmt,
    FieldDecl,
    FnDecl,
    ForStmt,
    GateDecl,
    GenericType,
    GlobalStmt,
    Identifier,
    IfExpr,
    ImportDecl,
    IndexExpr,
    IntentMatchExpr,
    LambdaExpr,
    LetStmt,
    Literal,
    LobeDecl,
    MatchArm,
    MatchExpr,
    MemberAccess,
    Module,
    NormDecl,
    Param,
    PassStmt,
    PersistDecl,
    ProcessorDecl,
    RaiseStmt,
    ReturnStmt,
    SemanticDistanceExpr,
    SimpleType,
    SliceExpr,
    SpreadExpr,
    StructDecl,
    TrustDecl,
    TrustType,
    TryStmt,
    TernaryExpr,
    TypeExpr,
    UnaryOp,
    UnionType,
    FunctionType,
    ValidateStmt,
    RequireStmt,
    WhileStmt,
    WithStmt,
    YieldExpr,
    KeywordArg,
)

__all__ = ["Formatter"]


# Binary operator precedence (higher binds tighter). Used for parenthesization.
_PREC_OR = 10
_PREC_AND = 20
_PREC_CMP = 30
_PREC_ADD = 40
_PREC_MUL = 50
_PREC_UNARY = 60

_OP_PREC = {
    "||": _PREC_OR,
    "&&": _PREC_AND,
    "==": _PREC_CMP,
    "!=": _PREC_CMP,
    "<": _PREC_CMP,
    ">": _PREC_CMP,
    "<=": _PREC_CMP,
    ">=": _PREC_CMP,
    "in": _PREC_CMP,
    "not in": _PREC_CMP,
    "is": _PREC_CMP,
    "+": _PREC_ADD,
    "-": _PREC_ADD,
    "|": _PREC_ADD,
    "&": _PREC_ADD,
    "^": _PREC_ADD,
    "<<": _PREC_ADD,
    ">>": _PREC_ADD,
    "*": _PREC_MUL,
    "/": _PREC_MUL,
    "%": _PREC_MUL,
    "**": _PREC_MUL,
    "//": _PREC_MUL,
    ":": 100,  # struct/dict pseudo-op — always tight
}


def _op_precedence(op: str) -> int:
    return _OP_PREC.get(op, _PREC_ADD)


class Formatter:
    """Pretty-print a parsed NRSI ``Module`` AST as source text."""

    INDENT = "    "

    def __init__(self) -> None:
        self._indent_level = 0

    def format(self, module: Module) -> str:
        chunks: List[str] = []
        for imp in module.imports:
            chunks.append(self._format_import(imp))
        for decl in module.declarations:
            chunks.append(self._format_node(decl))
        text = "\n\n".join(c for c in chunks if c)
        if text and not text.endswith("\n"):
            text += "\n"
        return text

    def _i(self) -> str:
        return self.INDENT * self._indent_level

    @contextmanager
    def _with_indent(self, delta: int) -> Iterator[None]:
        self._indent_level += delta
        try:
            yield
        finally:
            self._indent_level -= delta

    def _indent_block_text(self, text: str) -> str:
        pad = self._i()
        return "\n".join(pad + line for line in text.split("\n"))

    def _format_subdecl_at_root(self, node: ASTNode) -> str:
        """Format a nested declaration as if at indent 0 (caller adds outer indent)."""
        saved = self._indent_level
        self._indent_level = 0
        try:
            return self._format_node(node)
        finally:
            self._indent_level = saved

    def _format_node(self, node: Optional[ASTNode]) -> str:
        if node is None:
            return ""
        if isinstance(node, Module):
            return self.format(node)
        if isinstance(node, ImportDecl):
            return self._format_import(node)
        if isinstance(node, TrustDecl):
            return self._format_trust_decl(node)
        if isinstance(node, GateDecl):
            return self._format_gate_decl(node)
        if isinstance(node, LobeDecl):
            return self._format_lobe_decl(node)
        if isinstance(node, ProcessorDecl):
            return self._format_processor_decl(node)
        if isinstance(node, CreaseDecl):
            return self._format_crease_decl(node)
        if isinstance(node, NormDecl):
            return self._format_norm_decl(node)
        if isinstance(node, BeliefBaseDecl):
            return self._format_belief_base_decl(node)
        if isinstance(node, AxiomDecl):
            return self._format_axiom_decl(node)
        if isinstance(node, StructDecl):
            return self._format_struct_decl(node)
        if isinstance(node, EnumDecl):
            return self._format_enum_decl(node)
        if isinstance(node, FnDecl):
            return self._format_fn_decl(node)
        if isinstance(node, ClassDecl):
            return self._format_class_decl(node)
        if isinstance(node, ComposeDecl):
            return self._format_compose_decl(node)
        if isinstance(node, PersistDecl):
            return self._format_persist_decl(node)
        if isinstance(node, LetStmt):
            return self._format_let_stmt(node)
        if isinstance(node, ReturnStmt):
            return self._format_return_stmt(node)
        if isinstance(node, AssignStmt):
            return f"{self._format_expr(node.target)} = {self._format_expr(node.value)}"
        if isinstance(node, AugAssignStmt):
            return f"{self._format_expr(node.target)} {node.op} {self._format_expr(node.value)}"
        if isinstance(node, ForStmt):
            return self._format_for_stmt(node)
        if isinstance(node, WhileStmt):
            return self._format_while_stmt(node)
        if isinstance(node, IfExpr):
            return self._format_if_expr(node)
        if isinstance(node, MatchExpr):
            return self._format_match_expr(node)
        if isinstance(node, TryStmt):
            return self._format_try_stmt(node)
        if isinstance(node, WithStmt):
            return self._format_with_stmt(node)
        if isinstance(node, RaiseStmt):
            return self._format_raise_stmt(node)
        if isinstance(node, ExprStmt):
            return self._format_expr_stmt(node)
        if isinstance(node, RequireStmt):
            return f"require {self._format_expr(node.condition)}"
        if isinstance(node, ValidateStmt):
            args = ", ".join(self._format_expr(a) for a in node.args)
            inner = f"({args})" if args else "()"
            return f"validate {node.validator}{inner}"
        if isinstance(node, BreakStmt):
            return "break"
        if isinstance(node, ContinueStmt):
            return "continue"
        if isinstance(node, PassStmt):
            return "pass"
        if isinstance(node, DelStmt):
            return f"del {self._format_expr(node.target)}"
        if isinstance(node, AssertStmt):
            if node.message is not None:
                return f"assert {self._format_expr(node.condition)}, {self._format_expr(node.message)}"
            return f"assert {self._format_expr(node.condition)}"
        if isinstance(node, GlobalStmt):
            kw = "nonlocal" if node.is_nonlocal else "global"
            return f"{kw} {', '.join(node.names)}"
        if isinstance(node, MatchArm):
            return self._format_match_arm(node)
        return self._fallback(node)

    def _fallback(self, node: Any) -> str:
        return repr(node)

    def _format_import(self, node: ImportDecl) -> str:
        if node.names:
            names = ", ".join(node.names)
            path = node.module_path
            if path.startswith('"') or path.startswith("'") or "." not in path and "/" in path:
                path_q = self._quote_string(path) if not (path.startswith('"') or path.startswith("'")) else path
            else:
                path_q = path
            return f"from {path_q} import {names}"
        if node.alias:
            return f"import {node.module_path} as {node.alias}"
        return f"import {node.module_path}"

    def _format_trust_decl(self, node: TrustDecl) -> str:
        level = node.trust_level
        if node.value_type is not None:
            level += f"[{self._format_type(node.value_type)}]"
        head = f"trust {node.name}: {level}"
        if node.value is not None:
            head += f" = {self._format_expr(node.value)}"
        if node.annotations:
            ann = "\n".join(
                f"{self._i()}{k}: {self._format_expr(v)}"
                for k, v in node.annotations.items()
            )
            head += "\n" + ann
        return head

    def _format_gate_decl(self, node: GateDecl) -> str:
        sig = f"gate {node.name}{self._format_params(node.params)}"
        if node.return_type:
            sig += f" -> {self._format_type(node.return_type)}"
        body = self._format_body(node.body)
        return f"{sig} {body}"

    def _format_lobe_decl(self, node: LobeDecl) -> str:
        lines = [f"lobe {node.name} {{"]
        with self._with_indent(1):
            for p in node.processors:
                lines.append(self._indent_block_text(self._format_subdecl_at_root(p)))
            for c in node.creases:
                lines.append(self._indent_block_text(self._format_subdecl_at_root(c)))
        lines.append(self._i() + "}")
        return "\n".join(lines)

    def _format_processor_decl(self, node: ProcessorDecl) -> str:
        sig = f"processor {node.name}{self._format_params(node.params)}"
        if node.return_type:
            sig += f" -> {self._format_type(node.return_type)}"
        return f"{sig} {self._format_body(node.body)}"

    def _format_crease_decl(self, node: CreaseDecl) -> str:
        if not node.facts:
            return f"crease {node.domain} {{}}"
        lines = [f"crease {node.domain} {{"]
        with self._with_indent(1):
            for i, fact in enumerate(node.facts):
                suf = "," if i < len(node.facts) - 1 else ""
                lines.append(f"{self._i()}{self._quote_string(fact)}{suf}")
        lines.append(self._i() + "}")
        return "\n".join(lines)

    def _format_norm_decl(self, node: NormDecl) -> str:
        inner: List[str] = []
        with self._with_indent(1):
            if node.deontic_type:
                inner.append(f"{self._i()}type: {node.deontic_type}")
            if node.scope:
                sco = f"{self._i()}scope: {node.scope}"
                inner.append(sco)
            if node.condition is not None:
                inner.append(f"{self._i()}condition: {self._format_expr(node.condition)}")
            if node.action:
                act = (
                    self._quote_string(node.action)
                    if " " in node.action or not node.action.replace("_", "").isalnum()
                    else node.action
                )
                inner.append(f"{self._i()}action: {act}")
            if node.priority:
                inner.append(f"{self._i()}priority: {node.priority}")
            if node.domain:
                dom = node.domain
                dom_out = dom if dom.replace("_", "").isalnum() else self._quote_string(dom)
                inner.append(f"{self._i()}domain: {dom_out}")
        body = "\n".join(inner)
        return f"norm {node.name} {{\n{body}\n{self._i()}}}"

    def _format_belief_base_decl(self, node: BeliefBaseDecl) -> str:
        lines = [f"belief base {node.name} {{"]
        with self._with_indent(1):
            if node.entrenchment:
                lines.append(f"{self._i()}entrenchment: {node.entrenchment}")
            for ax in node.axioms:
                lines.append(self._i() + self._format_axiom_decl(ax))
        lines.append(self._i() + "}")
        return "\n".join(lines)

    def _format_axiom_decl(self, node: AxiomDecl) -> str:
        s = f'axiom {self._quote_string(node.content)}'
        if node.tier:
            s += f" tier: {node.tier}"
        return s

    def _format_struct_decl(self, node: StructDecl) -> str:
        lines = [f"type {node.name} {{"]
        with self._with_indent(1):
            for f in node.fields:
                lines.append(f"{self._i()}{self._format_field_decl(f)},")
        lines.append(self._i() + "}")
        return "\n".join(lines)

    def _format_field_decl(self, node: FieldDecl) -> str:
        s = f"{node.name}: {self._format_type(node.type_expr)}"
        if node.default is not None:
            s += f" = {self._format_expr(node.default)}"
        return s

    def _format_enum_decl(self, node: EnumDecl) -> str:
        lines = [f"enum {node.name} {{"]
        with self._with_indent(1):
            for v in node.variants:
                if v in node.variant_values:
                    val = node.variant_values[v]
                    if isinstance(val, Expr):
                        lines.append(f"{self._i()}{v} = {self._format_expr(val)},")
                    else:
                        lines.append(f"{self._i()}{v} = {val!r},")
                else:
                    lines.append(f"{self._i()}{v},")
        lines.append(self._i() + "}")
        return "\n".join(lines)

    def _format_decorators(self, decs: List[Expr]) -> str:
        return "\n".join(f"@{self._format_expr(d)}" for d in decs)

    def _format_fn_decl(self, node: FnDecl) -> str:
        parts: List[str] = []
        if node.decorators:
            parts.append(self._format_decorators(node.decorators))
        prefix = ""
        if node.is_export:
            prefix = "export "
        if node.is_async:
            prefix += "async "
        name = node.name
        if name == "<lambda>":
            sig = f"fn{self._format_params(node.params)}"
        else:
            sig = f"{prefix}fn {name}{self._format_params(node.params)}"
        if node.return_type:
            sig += f" -> {self._format_type(node.return_type)}"
        body = self._format_body(node.body)
        block = f"{sig} {body}"
        if parts:
            return parts[0] + "\n" + block
        return block

    def _format_class_decl(self, node: ClassDecl) -> str:
        parts: List[str] = []
        if node.decorators:
            parts.append(self._format_decorators(node.decorators))
        bases = ""
        if node.base_classes:
            bases = "(" + ", ".join(node.base_classes) + ")"
        lines = [f"class {node.name}{bases} {{"]
        with self._with_indent(1):
            for f in node.fields:
                lines.append(f"{self._i()}{self._format_field_decl(f)},")
            for m in node.methods:
                lines.append(self._indent_block_text(self._format_subdecl_at_root(m)))
        lines.append(self._i() + "}")
        out = "\n".join(lines)
        if parts:
            return parts[0] + "\n" + out
        return out

    def _format_compose_decl(self, node: ComposeDecl) -> str:
        line = f"compose {node.name}"
        if node.sources:
            line += " from " + ", ".join(self._format_expr(s) for s in node.sources)
        ann = self._format_annotation_block(node.annotations)
        if ann:
            line += " " + ann
        return line

    def _format_persist_decl(self, node: PersistDecl) -> str:
        if not node.config:
            return f"persist {node.name}"
        return f"persist {node.name} {self._format_annotation_block(node.config)}"

    def _format_annotation_block(self, data: dict) -> str:
        if not data:
            return ""
        lines = ["{"]
        with self._with_indent(1):
            for k, v in data.items():
                if isinstance(v, Expr):
                    lines.append(f"{self._i()}{k}: {self._format_expr(v)},")
                else:
                    lines.append(f"{self._i()}{k}: {v!r},")
        lines.append(self._i() + "}")
        return "\n".join(lines)

    def _format_let_stmt(self, node: LetStmt) -> str:
        mut = "mut " if node.mutable else ""
        name_part = node.name.strip()
        if "," in name_part and not (name_part.startswith("(") and name_part.endswith(")")):
            parts = [p.strip() for p in name_part.split(",")]
            name_part = "(" + ", ".join(parts) + ")"
        s = f"let {mut}{name_part}"
        if node.type_expr is not None:
            s += f": {self._format_type(node.type_expr)}"
        if node.value is not None:
            s += f" = {self._format_expr(node.value)}"
        return s

    def _format_return_stmt(self, node: ReturnStmt) -> str:
        if node.value is None:
            return "return"
        return f"return {self._format_expr(node.value)}"

    def _format_for_stmt(self, node: ForStmt) -> str:
        return (
            f"for {node.variable} in {self._format_expr(node.iterable)} "
            f"{self._format_body(node.body)}"
        )

    def _format_while_stmt(self, node: WhileStmt) -> str:
        return f"while {self._format_expr(node.condition)} {self._format_body(node.body)}"

    def _format_if_expr(self, node: IfExpr) -> str:
        lines = [f"if {self._format_expr(node.condition)} {self._format_body(node.then_body)}"]
        if node.else_body:
            if len(node.else_body) == 1 and isinstance(node.else_body[0], IfExpr):
                lines.append(f"else {self._format_if_expr_as_tail(node.else_body[0])}")
            else:
                lines.append(f"else {self._format_body(node.else_body)}")
        return "\n".join(lines)

    def _format_if_expr_as_tail(self, node: IfExpr) -> str:
        s = f"if {self._format_expr(node.condition)} {self._format_body(node.then_body)}"
        if node.else_body:
            if len(node.else_body) == 1 and isinstance(node.else_body[0], IfExpr):
                s += f" else {self._format_if_expr_as_tail(node.else_body[0])}"
            else:
                s += f" else {self._format_body(node.else_body)}"
        return s

    def _format_match_expr(self, node: MatchExpr) -> str:
        lines = [f"match {self._format_expr(node.subject)} {{"]
        with self._with_indent(1):
            for arm in node.arms:
                lines.append(self._i() + self._format_match_arm(arm))
        lines.append(self._i() + "}")
        return "\n".join(lines)

    def _format_match_arm(self, node: MatchArm) -> str:
        body = self._format_stmt_block_content(node.body)
        if len(node.body) == 1 and isinstance(node.body[0], ExprStmt):
            return f"{self._format_expr(node.pattern)} => {self._format_expr(node.body[0].expr)}"
        return f"{self._format_expr(node.pattern)} => {{\n{body}\n{self._i()}}}"

    def _format_try_stmt(self, node: TryStmt) -> str:
        parts = [f"try {self._format_body(node.body)}"]
        if node.catch_body:
            cv = f"({node.catch_var})" if node.catch_var else ""
            parts.append(f"catch{cv} {self._format_body(node.catch_body)}")
        if node.finally_body:
            parts.append(f"finally {self._format_body(node.finally_body)}")
        return "\n".join(parts)

    def _format_with_stmt(self, node: WithStmt) -> str:
        ctx = self._format_expr(node.context)
        if node.alias:
            ctx += f" as {node.alias}"
        return f"with {ctx} {self._format_body(node.body)}"

    def _format_raise_stmt(self, node: RaiseStmt) -> str:
        if node.value is None:
            return "raise"
        s = f"raise {self._format_expr(node.value)}"
        if node.cause is not None:
            s += f" from {self._format_expr(node.cause)}"
        return s

    def _format_expr_stmt(self, node: ExprStmt) -> str:
        ex = node.expr
        if isinstance(ex, Literal) and ex.literal_type == "export_list" and isinstance(ex.value, list):
            return "export " + ", ".join(str(x) for x in ex.value)
        return self._format_expr(ex)

    def _format_body(self, stmts: List[ASTNode]) -> str:
        if not stmts:
            return "{}"
        inner = self._format_stmt_block_content(stmts)
        return "{\n" + inner + "\n" + self._i() + "}"

    def _format_stmt_block_content(self, stmts: List[ASTNode]) -> str:
        lines: List[str] = []
        with self._with_indent(1):
            for st in stmts:
                formatted = self._format_node(st)
                for sub in formatted.split("\n"):
                    lines.append(self._i() + sub)
        return "\n".join(lines)

    def _format_params(self, params: List[Param]) -> str:
        parts = []
        for p in params:
            parts.append(self._format_param(p))
        return "(" + ", ".join(parts) + ")"

    def _format_param(self, p: Param) -> str:
        prefix = ""
        if p.is_kw_variadic:
            prefix = "**"
        elif p.is_variadic:
            prefix = "*"
        s = f"{prefix}{p.name}"
        if p.type_expr is not None:
            s += f": {self._format_type(p.type_expr)}"
        if p.default is not None:
            s += f" = {self._format_expr(p.default)}"
        return s

    def _format_type(self, t: Optional[TypeExpr]) -> str:
        if t is None:
            return ""
        if isinstance(t, SimpleType):
            return t.name
        if isinstance(t, TrustType):
            inner = f"[{self._format_type(t.inner_type)}]" if t.inner_type else ""
            return f"{t.trust_level}{inner}"
        if isinstance(t, GenericType):
            args = ", ".join(self._format_type(a) for a in t.type_args)
            return f"{t.name}[{args}]"
        if isinstance(t, UnionType):
            return " | ".join(self._format_type(x) for x in t.types)
        if isinstance(t, FunctionType):
            ps = ", ".join(self._format_type(p) for p in t.params)
            ret = self._format_type(t.return_type) if t.return_type else ""
            return f"({ps}) -> {ret}"
        return self._fallback(t)

    def _format_expr(self, e: Optional[Expr], parent_prec: int = 0) -> str:
        if e is None:
            return ""
        if isinstance(e, Literal):
            return self._format_literal(e)
        if isinstance(e, Identifier):
            return e.name
        if isinstance(e, UnaryOp):
            inner = self._format_expr(e.operand, _PREC_UNARY)
            op = e.op
            if op == "!":
                return f"!{inner}"
            if op == "-":
                return f"-{inner}"
            return f"{op} {inner}"
        if isinstance(e, BinaryOp):
            return self._format_binary(e, parent_prec)
        if isinstance(e, CallExpr):
            return self._format_call(e)
        if isinstance(e, MemberAccess):
            obj = self._format_expr(e.object, _PREC_UNARY + 1)
            return f"{obj}.{e.member}"
        if isinstance(e, IndexExpr):
            return self._format_index(e)
        if isinstance(e, IfExpr):
            sub = self._format_if_expr(e)
            if parent_prec > 0:
                return f"({sub})"
            return sub
        if isinstance(e, MatchExpr):
            sub = self._format_match_expr(e)
            if parent_prec > 0:
                return f"({sub})"
            return sub
        if isinstance(e, DictExpr):
            return self._format_dict_expr(e)
        if isinstance(e, TernaryExpr):
            return self._format_ternary(e, parent_prec)
        if isinstance(e, LambdaExpr):
            return self._format_lambda(e)
        if isinstance(e, ComprehensionExpr):
            return self._format_comprehension(e)
        if isinstance(e, YieldExpr):
            if e.is_from:
                return f"yield from {self._format_expr(e.value)}"
            if e.value is None:
                return "yield"
            return f"yield {self._format_expr(e.value)}"
        if isinstance(e, AwaitExpr):
            return f"await {self._format_expr(e.value, _PREC_UNARY)}"
        if isinstance(e, SpreadExpr):
            p = "**" if e.is_double else "*"
            return f"{p}{self._format_expr(e.value, _PREC_UNARY)}"
        if isinstance(e, KeywordArg):
            return f"{e.name} = {self._format_expr(e.value)}"
        if isinstance(e, SliceExpr):
            return self._format_slice(e)
        if isinstance(e, SemanticDistanceExpr):
            ann = self._format_annotation_block(e.annotations)
            return f"semantic_distance({self._format_expr(e.left)}, {self._format_expr(e.right)}){ann}"
        if isinstance(e, DecomposeExpr):
            ann = self._format_annotation_block(e.annotations)
            return f"decompose({self._format_expr(e.goal)}){ann}"
        if isinstance(e, IntentMatchExpr):
            ann = self._format_annotation_block(e.annotations)
            return (
                f"intent_match({self._format_expr(e.query_expr)}, "
                f"{self._format_expr(e.belief_base)}){ann}"
            )
        if isinstance(e, FnDecl):
            return self._format_fn_decl(e)
        return self._fallback(e)

    def _format_slice(self, s: SliceExpr) -> str:
        lo = "" if s.lower is None else self._format_expr(s.lower)
        hi = "" if s.upper is None else self._format_expr(s.upper)
        st = "" if s.step is None else self._format_expr(s.step)
        if s.step is not None:
            return f"{lo}:{hi}:{st}"
        return f"{lo}:{hi}"

    def _format_index(self, e: IndexExpr) -> str:
        obj = self._format_expr(e.object, _PREC_UNARY + 1)
        if isinstance(e.index, SliceExpr):
            return f"{obj}[{self._format_slice(e.index)}]"
        return f"{obj}[{self._format_expr(e.index)}]"

    def _format_binary(self, e: BinaryOp, parent_prec: int) -> str:
        op = e.op
        prec = _op_precedence(op)
        if op in ("not in", "is", "in"):
            lhs = self._format_expr(e.left, prec)
            rhs = self._format_expr(e.right, prec + 1)
            s = f"{lhs} {op} {rhs}"
        elif op == ":":
            lhs = self._format_expr(e.left, prec)
            rhs = self._format_expr(e.right, prec)
            s = f"{lhs}: {rhs}"
        else:
            lhs = self._format_expr(e.left, prec)
            rhs = self._format_expr(e.right, prec + 1)
            s = f"{lhs} {op} {rhs}"
        if prec < parent_prec:
            return f"({s})"
        return s

    def _format_ternary(self, e: TernaryExpr, parent_prec: int) -> str:
        then_e = self._format_expr(e.then_expr, _PREC_OR - 1)
        cond = self._format_expr(e.condition, _PREC_OR - 1)
        else_e = self._format_expr(e.else_expr, _PREC_OR - 1)
        s = f"{then_e} if {cond} else {else_e}"
        if parent_prec > _PREC_OR - 1:
            return f"({s})"
        return s

    def _format_lambda(self, e: LambdaExpr) -> str:
        params = ", ".join(self._format_param(p) for p in e.params)
        body = self._format_expr(e.body)
        return f"lambda {params}: {body}"

    def _format_comprehension(self, e: ComprehensionExpr) -> str:
        parts = [f"for {e.variable} in {self._format_expr(e.iterable)}"]
        if e.condition is not None:
            parts.append(f"if {self._format_expr(e.condition)}")
        tail = " ".join(parts)
        if e.kind == "list":
            el = self._format_expr(e.element) if e.element else ""
            return f"[{el} {tail}]"
        if e.kind == "set":
            el = self._format_expr(e.element) if e.element else ""
            return f"{{{el} {tail}}}"
        if e.kind == "dict":
            k = self._format_expr(e.key) if e.key else ""
            v = self._format_expr(e.value) if e.value else ""
            return f"{{{k}: {v} {tail}}}"
        el = self._format_expr(e.element) if e.element else ""
        return f"({el} {tail})"

    def _format_dict_expr(self, e: DictExpr) -> str:
        pairs = []
        for item in e.pairs:
            if isinstance(item, tuple) and len(item) == 2:
                k, v = item
                pairs.append(f"{self._format_expr(k)}: {self._format_expr(v)}")
            else:
                pairs.append(self._format_expr(item))
        return "{" + ", ".join(pairs) + "}"

    def _format_call(self, e: CallExpr) -> str:
        callee = e.callee
        if isinstance(callee, Identifier):
            if callee.name == "__tuple__":
                inner = ", ".join(self._format_expr(a) for a in e.args)
                if len(e.args) == 1:
                    return f"({inner},)"
                return f"({inner})"
            if callee.name == "List":
                inner = ", ".join(self._format_expr(a) for a in e.args)
                return f"[{inner}]"
            if callee.name == "Set":
                inner = ", ".join(self._format_expr(a) for a in e.args)
                return f"{{{inner}}}"
        c = self._format_expr(callee, _PREC_UNARY + 1)
        args = ", ".join(self._format_expr(a) for a in e.args)
        return f"{c}({args})"

    def _format_literal(self, e: Literal) -> str:
        lt = e.literal_type
        v = e.value
        if lt == "string":
            return self._quote_string(str(v))
        if lt == "int":
            return str(v)
        if lt == "float":
            s = repr(v)
            if s.endswith(".0") and isinstance(v, float) and v == int(v):
                return str(int(v))
            return s
        if lt == "bool":
            return "true" if v else "false"
        if lt == "none":
            return "nil"
        return repr(v)

    def _quote_string(self, s: str) -> str:
        esc = (
            s.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\t", "\\t")
            .replace("\r", "\\r")
        )
        return f'"{esc}"'
