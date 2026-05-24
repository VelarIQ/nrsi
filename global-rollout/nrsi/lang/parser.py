"""NRSI Parser — Tokens to Abstract Syntax Tree.

Recursive descent parser for NRSI source.  Produces typed AST nodes
for every language construct: trust declarations, gates, lobes,
processors, norms, beliefs, types, functions, and expressions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from nrsi.lang.lexer import Token, TokenType, Lexer

logger = logging.getLogger("nrsi.lang.parser")

__all__ = [
    # AST base
    "ASTNode", "Module",
    # Type expressions
    "TypeExpr", "SimpleType", "TrustType", "GenericType",
    "UnionType", "FunctionType",
    # Declarations
    "TrustDecl", "GateDecl", "RequireStmt", "ValidateStmt",
    "LobeDecl", "ProcessorDecl", "NormDecl", "BeliefBaseDecl",
    "AxiomDecl", "StructDecl", "EnumDecl", "FnDecl",
    "Param", "FieldDecl", "ImportDecl", "CreaseDecl",
    # Expressions & statements
    "Expr", "Literal", "Identifier", "BinaryOp", "UnaryOp",
    "CallExpr", "MemberAccess", "IndexExpr", "SliceExpr", "KeywordArg", "IfExpr",
    "MatchExpr", "MatchArm",     "LetStmt", "ReturnStmt",
    "AssignStmt", "ForStmt", "ExprStmt", "DictExpr",
    "WhileStmt", "TryStmt", "BreakStmt", "ContinueStmt",
    "PassStmt", "RaiseStmt", "DelStmt", "AugAssignStmt", "ClassDecl",
    # Extended constructs
    "WithStmt", "YieldExpr", "AwaitExpr", "AssertStmt", "GlobalStmt",
    "ComprehensionExpr", "TernaryExpr", "LambdaExpr", "SpreadExpr",
    # Cognitive primitives
    "ComposeDecl", "PersistDecl",
    "SemanticDistanceExpr", "DecomposeExpr", "IntentMatchExpr",
    # Parser
    "ParseError", "Parser", "parse",
]


# ═══════════════════════════════════════════════════════════════════════════════
# AST Node Base
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ASTNode:
    """Base for all AST nodes."""
    line: int = 0
    column: int = 0


@dataclass
class Module(ASTNode):
    """Top-level module (one .nrsi file)."""
    name: str = ""
    declarations: List[ASTNode] = field(default_factory=list)
    imports: List[ImportDecl] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Type Expressions
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TypeExpr(ASTNode):
    """Base type expression."""
    pass


@dataclass
class SimpleType(TypeExpr):
    """Simple type name: string, int, float, bool, etc."""
    name: str = ""


@dataclass
class TrustType(TypeExpr):
    """Trust-wrapped type: raw[T], validated[T], trusted[T], certified[T]."""
    trust_level: str = ""
    inner_type: Optional[TypeExpr] = None


@dataclass
class GenericType(TypeExpr):
    """Generic type: List[T], Dict[K, V], etc."""
    name: str = ""
    type_args: List[TypeExpr] = field(default_factory=list)


@dataclass
class UnionType(TypeExpr):
    """Union type: A | B."""
    types: List[TypeExpr] = field(default_factory=list)


@dataclass
class FunctionType(TypeExpr):
    """Function type: (A, B) -> C."""
    params: List[TypeExpr] = field(default_factory=list)
    return_type: Optional[TypeExpr] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Declarations
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrustDecl(ASTNode):
    """trust name: trust_level[Type] = value"""
    name: str = ""
    trust_level: str = ""
    value_type: Optional[TypeExpr] = None
    value: Optional[Expr] = None
    annotations: Dict[str, Expr] = field(default_factory=dict)


@dataclass
class GateDecl(ASTNode):
    """gate name(param: raw[T]) -> validated[T] { require ...; validate ...; }"""
    name: str = ""
    params: List[Param] = field(default_factory=list)
    return_type: Optional[TypeExpr] = None
    body: List[ASTNode] = field(default_factory=list)


@dataclass
class RequireStmt(ASTNode):
    """require confidence >= 0.95"""
    condition: Optional[Expr] = None


@dataclass
class ValidateStmt(ASTNode):
    """validate validator_name(args...)"""
    validator: str = ""
    args: List[Expr] = field(default_factory=list)


@dataclass
class LobeDecl(ASTNode):
    """lobe name { processor ... }"""
    name: str = ""
    processors: List[ProcessorDecl] = field(default_factory=list)
    creases: List[CreaseDecl] = field(default_factory=list)


@dataclass
class ProcessorDecl(ASTNode):
    """processor name(params) -> ReturnType { body }"""
    name: str = ""
    params: List[Param] = field(default_factory=list)
    return_type: Optional[TypeExpr] = None
    body: List[ASTNode] = field(default_factory=list)


@dataclass
class NormDecl(ASTNode):
    """norm name { type: obligation/permission/prohibition, ... }"""
    name: str = ""
    deontic_type: str = ""
    scope: str = ""
    condition: Optional[Expr] = None
    action: str = ""
    priority: int = 0
    domain: str = ""


@dataclass
class BeliefBaseDecl(ASTNode):
    """belief base name { entrenchment: ..., axiom "..." tier: T0, ... }"""
    name: str = ""
    entrenchment: str = ""
    axioms: List[AxiomDecl] = field(default_factory=list)


@dataclass
class AxiomDecl(ASTNode):
    """axiom "content" tier: T0_AXIOM"""
    content: str = ""
    tier: str = ""


@dataclass
class StructDecl(ASTNode):
    """type Name { field: Type, ... }"""
    name: str = ""
    fields: List[FieldDecl] = field(default_factory=list)


@dataclass
class EnumDecl(ASTNode):
    """enum Name { VARIANT_A, VARIANT_B, VARIANT_C = expr, ... }"""
    name: str = ""
    variants: List[str] = field(default_factory=list)
    variant_values: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FnDecl(ASTNode):
    """fn name(params) -> ReturnType { body }"""
    name: str = ""
    params: List[Param] = field(default_factory=list)
    return_type: Optional[TypeExpr] = None
    body: List[ASTNode] = field(default_factory=list)
    is_export: bool = False
    is_async: bool = False
    decorators: List[Expr] = field(default_factory=list)


@dataclass
class Param(ASTNode):
    name: str = ""
    type_expr: Optional[TypeExpr] = None
    default: Optional[Expr] = None
    is_variadic: bool = False
    is_kw_variadic: bool = False


@dataclass
class FieldDecl(ASTNode):
    name: str = ""
    type_expr: Optional[TypeExpr] = None
    default: Optional[Expr] = None


@dataclass
class ImportDecl(ASTNode):
    module_path: str = ""
    names: List[str] = field(default_factory=list)
    alias: str = ""


@dataclass
class CreaseDecl(ASTNode):
    domain: str = ""
    facts: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Expressions & Statements
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Expr(ASTNode):
    pass


@dataclass
class Literal(Expr):
    value: Any = None
    literal_type: str = ""


@dataclass
class Identifier(Expr):
    name: str = ""


@dataclass
class BinaryOp(Expr):
    left: Optional[Expr] = None
    op: str = ""
    right: Optional[Expr] = None


@dataclass
class UnaryOp(Expr):
    op: str = ""
    operand: Optional[Expr] = None


@dataclass
class CallExpr(Expr):
    callee: Optional[Expr] = None
    args: List[Expr] = field(default_factory=list)


@dataclass
class MemberAccess(Expr):
    object: Optional[Expr] = None
    member: str = ""


@dataclass
class IndexExpr(Expr):
    object: Optional[Expr] = None
    index: Optional[Expr] = None


@dataclass
class SliceExpr(Expr):
    lower: Optional[Expr] = None
    upper: Optional[Expr] = None
    step: Optional[Expr] = None


@dataclass
class KeywordArg(Expr):
    name: str = ""
    value: Optional[Expr] = None


@dataclass
class IfExpr(Expr):
    condition: Optional[Expr] = None
    then_body: List[ASTNode] = field(default_factory=list)
    else_body: List[ASTNode] = field(default_factory=list)


@dataclass
class MatchExpr(Expr):
    subject: Optional[Expr] = None
    arms: List[MatchArm] = field(default_factory=list)


@dataclass
class MatchArm(ASTNode):
    pattern: Optional[Expr] = None
    body: List[ASTNode] = field(default_factory=list)


@dataclass
class LetStmt(ASTNode):
    name: str = ""
    type_expr: Optional[TypeExpr] = None
    value: Optional[Expr] = None
    mutable: bool = False


@dataclass
class ReturnStmt(ASTNode):
    value: Optional[Expr] = None


@dataclass
class AssignStmt(ASTNode):
    target: Optional[Expr] = None
    value: Optional[Expr] = None


@dataclass
class ForStmt(ASTNode):
    variable: str = ""
    iterable: Optional[Expr] = None
    body: List[ASTNode] = field(default_factory=list)


@dataclass
class ExprStmt(ASTNode):
    expr: Optional[Expr] = None


@dataclass
class DictExpr(Expr):
    pairs: List[Any] = field(default_factory=list)


@dataclass
class WhileStmt(ASTNode):
    condition: Optional[Expr] = None
    body: List[ASTNode] = field(default_factory=list)


@dataclass
class TryStmt(ASTNode):
    body: List[ASTNode] = field(default_factory=list)
    catch_var: str = ""
    catch_body: List[ASTNode] = field(default_factory=list)
    finally_body: List[ASTNode] = field(default_factory=list)


@dataclass
class BreakStmt(ASTNode):
    pass


@dataclass
class ContinueStmt(ASTNode):
    pass


@dataclass
class PassStmt(ASTNode):
    pass


@dataclass
class RaiseStmt(ASTNode):
    value: Optional[Expr] = None
    cause: Optional[Expr] = None


@dataclass
class DelStmt(ASTNode):
    target: Expr = None


@dataclass
class AugAssignStmt(ASTNode):
    target: Optional[Expr] = None
    op: str = ""
    value: Optional[Expr] = None


@dataclass
class ClassDecl(ASTNode):
    name: str = ""
    base_classes: List[str] = field(default_factory=list)
    fields: List[FieldDecl] = field(default_factory=list)
    methods: List[FnDecl] = field(default_factory=list)
    decorators: List[Expr] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Cognitive Primitive AST Nodes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ComposeDecl(ASTNode):
    """``compose answer from facts, web { strategy: "synthesis" }``"""
    name: str = ""
    sources: List[Expr] = field(default_factory=list)
    annotations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PersistDecl(ASTNode):
    """``persist store { decay: 0.01, reinforcement: 0.1 }``"""
    name: str = ""
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticDistanceExpr(Expr):
    """``semantic_distance(a, b)`` — graph-based meaning similarity."""
    left: Optional[Expr] = None
    right: Optional[Expr] = None
    annotations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecomposeExpr(Expr):
    """``decompose(goal)`` — recursive goal splitting."""
    goal: Optional[Expr] = None
    annotations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntentMatchExpr(Expr):
    """``intent_match(query, beliefs)`` — semantic pattern matching."""
    query_expr: Optional[Expr] = None
    belief_base: Optional[Expr] = None
    annotations: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Extended Language Constructs
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WithStmt(ASTNode):
    """``with expr as name { body }``."""
    context: Optional[Expr] = None
    alias: str = ""
    body: List[ASTNode] = field(default_factory=list)


@dataclass
class YieldExpr(Expr):
    """``yield expr`` or ``yield from expr``."""
    value: Optional[Expr] = None
    is_from: bool = False


@dataclass
class AwaitExpr(Expr):
    """``await expr``."""
    value: Optional[Expr] = None


@dataclass
class AssertStmt(ASTNode):
    """``assert condition, message``."""
    condition: Optional[Expr] = None
    message: Optional[Expr] = None


@dataclass
class GlobalStmt(ASTNode):
    """``global x, y`` or ``nonlocal x, y``."""
    names: List[str] = field(default_factory=list)
    is_nonlocal: bool = False


@dataclass
class ComprehensionExpr(Expr):
    """List/dict/set/generator comprehension."""
    element: Optional[Expr] = None
    key: Optional[Expr] = None
    value: Optional[Expr] = None
    variable: str = ""
    iterable: Optional[Expr] = None
    condition: Optional[Expr] = None
    kind: str = "list"


@dataclass
class TernaryExpr(Expr):
    """``then_expr if cond else else_expr``."""
    condition: Optional[Expr] = None
    then_expr: Optional[Expr] = None
    else_expr: Optional[Expr] = None


@dataclass
class LambdaExpr(Expr):
    """``lambda params: expr``  OR  ``|params| expr``."""
    params: List[Param] = field(default_factory=list)
    body: Optional[Expr] = None


@dataclass
class SpreadExpr(Expr):
    """``*expr`` or ``**expr`` in call/def context."""
    value: Optional[Expr] = None
    is_double: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# Parse Errors
# ═══════════════════════════════════════════════════════════════════════════════

class ParseError(Exception):
    """Raised when the parser encounters invalid syntax."""

    def __init__(self, message: str, token: Optional[Token] = None):
        self.token = token
        loc = f" at line {token.line}:{token.column}" if token else ""
        super().__init__(f"{message}{loc}")


# ═══════════════════════════════════════════════════════════════════════════════
# Parser
# ═══════════════════════════════════════════════════════════════════════════════

_TRUST_KEYWORDS = frozenset({
    TokenType.KW_RAW, TokenType.KW_VALIDATED,
    TokenType.KW_TRUSTED, TokenType.KW_CERTIFIED,
})

_COMPARISON_OPS: Dict[TokenType, str] = {
    TokenType.EQUAL: "==",
    TokenType.NOT_EQUAL:  "!=",
    TokenType.LANGLE:   "<",
    TokenType.RANGLE:   ">",
    TokenType.LTE:  "<=",
    TokenType.GTE:  ">=",
}

_ADDITIVE_OPS = frozenset({
    TokenType.PLUS, TokenType.MINUS,
    TokenType.PIPE, TokenType.AMPERSAND, TokenType.CARET,
    TokenType.LEFT_SHIFT, TokenType.RIGHT_SHIFT,
})
_MULTIPLICATIVE_OPS = frozenset({
    TokenType.STAR,
    TokenType.SLASH,
    TokenType.PERCENT,
    TokenType.POWER,
    TokenType.FLOOR_DIV,
})
_UNARY_PREFIX = frozenset({TokenType.BANG, TokenType.KW_NOT, TokenType.MINUS})

_BODY_SKIP = frozenset({TokenType.NEWLINE, TokenType.SEMICOLON, TokenType.COMMENT, TokenType.DOC_COMMENT})
_STMT_BOUNDARY = frozenset({
    TokenType.NEWLINE, TokenType.SEMICOLON,
    TokenType.RBRACE, TokenType.EOF,
})


class Parser:
    """Recursive descent parser for NRSI.

    Accepts a token list (typically from ``Lexer.tokenize()``) and
    produces a ``Module`` AST that covers the complete NRSI grammar:
    trust declarations, gates, lobes, processors, norms, belief bases,
    struct/enum types, functions, and all expression forms.
    """

    def __init__(self, tokens: List[Token], filename: str = "<stdin>"):
        self._tokens = tokens
        self._pos = 0
        self._filename = filename
        self._paren_depth = 0

    # ── Token navigation ──────────────────────────────────────────────────

    def _peek(self) -> Token:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return self._tokens[-1]

    def _peek_type(self) -> TokenType:
        return self._peek().type

    def _advance(self) -> Token:
        tok = self._peek()
        if tok.type != TokenType.EOF:
            self._pos += 1
        return tok

    def _check(self, *types: TokenType) -> bool:
        return self._peek_type() in types

    def _match(self, *types: TokenType) -> Optional[Token]:
        if self._peek_type() in types:
            return self._advance()
        return None

    def _expect(self, tt: TokenType, message: str = "") -> Token:
        tok = self._peek()
        if tok.type == tt:
            return self._advance()
        msg = message or f"Expected {tt.name}, got {tok.type.name} ({tok.value!r})"
        raise ParseError(msg, tok)

    def _at_end(self) -> bool:
        return self._peek_type() == TokenType.EOF

    def _skip_newlines(self) -> None:
        while self._peek_type() in _BODY_SKIP:
            self._advance()

    def _skip_terminators(self) -> None:
        while self._peek_type() in _BODY_SKIP:
            self._advance()

    def _at_stmt_boundary(self) -> bool:
        return self._peek_type() in _STMT_BOUNDARY

    def _loc(self, tok: Token) -> Dict[str, int]:
        return {"line": tok.line, "column": tok.column}

    # ── Module entry point ────────────────────────────────────────────────

    def parse(self) -> Module:
        """Parse a complete NRSI module."""
        module = Module(name=self._filename)
        while not self._at_end():
            self._skip_newlines()
            if self._at_end():
                break
            try:
                decl = self._declaration()
            except ParseError:
                raise
            if decl is None:
                continue
            if isinstance(decl, ImportDecl):
                module.imports.append(decl)
            else:
                module.declarations.append(decl)
                if isinstance(decl, FnDecl) and decl.is_export:
                    module.exports.append(decl.name)
            self._skip_terminators()
        return module

    def parse_expression(self) -> Expr:
        """Parse a single expression (useful for REPL / testing)."""
        return self._expression()

    # ── Top-level declarations ────────────────────────────────────────────

    def _declaration(self) -> Optional[ASTNode]:
        self._skip_newlines()
        if self._at_end():
            return None

        tt = self._peek_type()

        if tt == TokenType.AT:
            decorators = self._parse_decorators()
            self._skip_newlines()
            tt = self._peek_type()
            if tt == TokenType.KW_ASYNC:
                self._advance()
                self._skip_newlines()
                decl = self._fn_decl()
                decl.is_async = True
                decl.decorators = decorators
                return decl
            elif tt == TokenType.KW_FN:
                decl = self._fn_decl()
                decl.decorators = decorators
                return decl
            elif tt == TokenType.KW_CLASS:
                decl = self._class_decl()
                decl.decorators = decorators
                return decl
            else:
                self._error(f"Expected 'fn' or 'class' after decorator(s), got {tt}")

        if tt == TokenType.KW_MODULE:
            self._advance()
            name_tok = self._any_identifier_or_keyword()
            self._skip_newlines()
            return None

        if tt == TokenType.KW_IMPORT:
            return self._import_decl()
        if tt == TokenType.KW_FROM:
            return self._import_from_decl()
        if tt == TokenType.KW_EXPORT:
            return self._export_decl()
        if tt == TokenType.KW_TRUST:
            return self._trust_decl()
        if tt == TokenType.KW_GATE:
            return self._gate_decl()
        if tt == TokenType.KW_LOBE:
            return self._lobe_decl()
        if tt == TokenType.KW_NORM:
            return self._norm_decl()
        if tt == TokenType.KW_BELIEF:
            return self._belief_base_decl()
        if tt == TokenType.KW_TYPE:
            return self._struct_decl()
        if tt == TokenType.KW_ENUM:
            return self._enum_decl()
        if tt == TokenType.KW_ASYNC:
            self._advance()
            self._skip_newlines()
            decl = self._fn_decl()
            decl.is_async = True
            return decl
        if tt == TokenType.KW_FN:
            return self._fn_decl()
        if tt == TokenType.KW_LET:
            return self._let_stmt()
        if tt == TokenType.KW_COMPOSE:
            return self._compose_decl()
        if tt == TokenType.KW_PERSIST:
            return self._persist_decl()
        if tt == TokenType.KW_CLASS:
            return self._class_decl()

        return self._expr_or_assign_stmt()

    def _parse_decorators(self) -> List[Expr]:
        decorators: List[Expr] = []
        while self._peek_type() == TokenType.AT:
            self._advance()  # consume @
            expr = self._expression()
            decorators.append(expr)
            self._skip_newlines()
        return decorators

    # ── Import ────────────────────────────────────────────────────────────

    def _import_decl(self) -> ImportDecl:
        """``import module.path [as alias]`` or
        ``import name1[ as a1], name2[ as a2] from "path"``"""
        tok = self._expect(TokenType.KW_IMPORT)

        first = self._any_identifier_or_keyword().value
        names: List[str] = [first]
        # Per-name aliases are only meaningful in a comma-separated list
        # whose terminator is `from "..."`. We must NOT consume `as` after
        # the first name unconditionally because `import x as y` (single
        # name + module-level alias) is the canonical Pythonic form and
        # is parsed by the alias-fallthrough block below.
        while self._match(TokenType.COMMA):
            self._skip_newlines()
            names.append(self._any_identifier_or_keyword().value)
            # Optional per-name alias inside the list, e.g.
            # `import a, b as bee from "mod"`. The alias is consumed but
            # not currently propagated — transpilers re-emit using the
            # local binding name.
            if self._match(TokenType.KW_AS):
                self._any_identifier_or_keyword()

        if self._match(TokenType.KW_FROM):
            if self._check(TokenType.STRING):
                path = self._advance().value
            else:
                path = self._dotted_name()
            return ImportDecl(module_path=path, names=names, **self._loc(tok))

        path = ".".join(names)
        alias = ""
        if self._match(TokenType.KW_AS):
            alias = self._any_identifier_or_keyword().value
        return ImportDecl(module_path=path, alias=alias, **self._loc(tok))

    def _import_from_decl(self) -> ImportDecl:
        """``from module.path import name1, name2`` or ``from "path" import ...``"""
        tok = self._expect(TokenType.KW_FROM)
        if self._check(TokenType.STRING):
            path = self._advance().value
        else:
            path = self._dotted_name()
        self._expect(TokenType.KW_IMPORT)
        names: List[str] = [self._any_identifier_or_keyword().value]
        while self._match(TokenType.COMMA):
            self._skip_newlines()
            names.append(self._any_identifier_or_keyword().value)
        return ImportDecl(module_path=path, names=names, **self._loc(tok))

    def _dotted_name(self) -> str:
        parts = [self._any_identifier_or_keyword().value]
        while self._match(TokenType.DOT):
            parts.append(self._any_identifier_or_keyword().value)
        return ".".join(parts)

    # ── Export ────────────────────────────────────────────────────────────

    def _export_decl(self) -> ASTNode:
        """``export fn name(...) { ... }`` or ``export name1, name2, ...``"""
        tok = self._expect(TokenType.KW_EXPORT)
        if self._peek_type() == TokenType.KW_FN:
            decl = self._fn_decl()
            decl.is_export = True
            return decl
        names: List[str] = [self._any_identifier_or_keyword().value]
        while self._match(TokenType.COMMA):
            self._skip_newlines()
            names.append(self._any_identifier_or_keyword().value)
        return ExprStmt(
            expr=Literal(value=names, literal_type="export_list"),
            **self._loc(tok),
        )
        return decl

    # ── Trust declaration ─────────────────────────────────────────────────

    def _trust_decl(self) -> TrustDecl:
        """``trust name: raw[Type] = value [annotations...]``"""
        tok = self._expect(TokenType.KW_TRUST)
        name = self._any_identifier_or_keyword().value
        self._expect(TokenType.COLON)

        level_tok = self._peek()
        if level_tok.type not in _TRUST_KEYWORDS:
            raise ParseError(
                f"Expected trust level (raw/validated/trusted/certified), "
                f"got {level_tok.value!r}",
                level_tok,
            )
        self._advance()
        trust_level = level_tok.value

        value_type: Optional[TypeExpr] = None
        if self._match(TokenType.LBRACKET):
            value_type = self._type_expr()
            self._expect(TokenType.RBRACKET)

        value: Optional[Expr] = None
        if self._match(TokenType.ASSIGN):
            value = self._expression()

        annotations: Dict[str, Expr] = {}
        while not self._at_end():
            saved_pos = self._pos
            self._skip_newlines()
            tok_next = self._peek()
            if tok_next.type == TokenType.IDENTIFIER or tok_next.type.name.startswith("KW_"):
                next_pos = self._pos + 1
                if next_pos < len(self._tokens) and self._tokens[next_pos].type == TokenType.COLON:
                    ann_key = self._advance().value
                    self._expect(TokenType.COLON)
                    annotations[ann_key] = self._expression()
                    continue
            self._pos = saved_pos
            break

        return TrustDecl(
            name=name, trust_level=trust_level, value_type=value_type,
            value=value, annotations=annotations, **self._loc(tok),
        )

    # ── Gate declaration ──────────────────────────────────────────────────

    def _gate_decl(self) -> GateDecl:
        """``gate name(params) -> ReturnType { require ...; validate ...; }``"""
        tok = self._expect(TokenType.KW_GATE)
        name = self._any_identifier_or_keyword().value
        params = self._param_list()
        ret = self._optional_return_type()
        body = self._brace_body(gate_mode=True)
        return GateDecl(
            name=name, params=params, return_type=ret,
            body=body, **self._loc(tok),
        )

    # ── Lobe declaration ──────────────────────────────────────────────────

    def _lobe_decl(self) -> LobeDecl:
        """``lobe name { processor ...; crease ...; }``"""
        tok = self._expect(TokenType.KW_LOBE)
        name = self._any_identifier_or_keyword().value
        self._expect(TokenType.LBRACE)
        self._skip_newlines()

        processors: List[ProcessorDecl] = []
        creases: List[CreaseDecl] = []

        while not self._check(TokenType.RBRACE) and not self._at_end():
            self._skip_newlines()
            if self._check(TokenType.RBRACE):
                break
            if self._check(TokenType.KW_PROCESSOR):
                processors.append(self._processor_decl())
            elif self._check(TokenType.IDENTIFIER):
                creases.append(self._crease_decl())
            else:
                raise ParseError(
                    "Expected 'processor' or 'crease' inside lobe",
                    self._peek(),
                )
            self._skip_terminators()

        self._expect(TokenType.RBRACE)
        return LobeDecl(
            name=name, processors=processors,
            creases=creases, **self._loc(tok),
        )

    def _processor_decl(self) -> ProcessorDecl:
        """``processor name(params) -> ReturnType { body }``"""
        tok = self._expect(TokenType.KW_PROCESSOR)
        name = self._any_identifier_or_keyword().value
        params = self._param_list()
        ret = self._optional_return_type()
        body = self._brace_body()
        return ProcessorDecl(
            name=name, params=params, return_type=ret,
            body=body, **self._loc(tok),
        )

    def _crease_decl(self) -> CreaseDecl:
        """``crease domain { "fact1", "fact2", ... }``"""
        tok = self._any_identifier_or_keyword()
        domain = self._any_identifier_or_keyword().value
        self._expect(TokenType.LBRACE)
        self._skip_newlines()

        facts: List[str] = []
        while not self._check(TokenType.RBRACE) and not self._at_end():
            self._skip_newlines()
            if self._check(TokenType.RBRACE):
                break
            facts.append(self._expect(TokenType.STRING).value)
            self._match(TokenType.COMMA)
            self._skip_terminators()

        self._expect(TokenType.RBRACE)
        return CreaseDecl(domain=domain, facts=facts, **self._loc(tok))

    # ── Norm declaration ──────────────────────────────────────────────────

    def _norm_decl(self) -> NormDecl:
        tok = self._expect(TokenType.KW_NORM)
        name = self._any_identifier_or_keyword().value
        self._expect(TokenType.LBRACE)
        self._skip_newlines()

        deontic_type = ""
        scope = ""
        condition: Optional[Expr] = None
        action = ""
        priority = 0
        domain = ""

        while not self._check(TokenType.RBRACE) and not self._at_end():
            self._skip_newlines()
            if self._check(TokenType.RBRACE):
                break

            key, key_tok = self._norm_field_key()
            self._expect(TokenType.COLON)

            if key == "type":
                deontic_type = self._any_identifier_or_keyword().value
            elif key == "scope":
                scope = self._any_identifier_or_keyword().value
                if self._match(TokenType.LPAREN):
                    if self._check(TokenType.STRING):
                        domain = self._advance().value
                    else:
                        domain = self._any_identifier_or_keyword().value
                    self._expect(TokenType.RPAREN)
            elif key == "condition":
                condition = self._expression()
            elif key == "action":
                if self._check(TokenType.STRING):
                    action = self._advance().value
                else:
                    action = self._any_identifier_or_keyword().value
            elif key == "priority":
                priority = int(self._expect(TokenType.INTEGER).value)
            elif key == "domain":
                if self._check(TokenType.STRING):
                    domain = self._advance().value
                else:
                    domain = self._any_identifier_or_keyword().value
            else:
                logger.warning("Unknown norm field %r at line %d", key, key_tok.line)
                self._expression()
            self._match(TokenType.COMMA)
            self._skip_terminators()

        self._expect(TokenType.RBRACE)
        return NormDecl(
            name=name, deontic_type=deontic_type, scope=scope,
            condition=condition, action=action, priority=priority,
            domain=domain, **self._loc(tok),
        )

    def _any_identifier_or_keyword(self) -> Token:
        """Accept IDENTIFIER or any keyword token as an identifier value."""
        tok = self._peek()
        if tok.type == TokenType.IDENTIFIER or tok.type.name.startswith("KW_"):
            self._advance()
            return tok
        raise ParseError(
            f"Expected IDENTIFIER, got {tok.type.name} ({tok.value!r})", tok
        )

    def _norm_field_key(self) -> tuple:
        """Accept IDENT or the ``type`` keyword as a norm body field name."""
        tok = self._peek()
        if tok.type == TokenType.KW_TYPE:
            self._advance()
            return ("type", tok)
        if tok.type == TokenType.IDENTIFIER:
            self._advance()
            return (tok.value, tok)
        if tok.type.name.startswith("KW_"):
            self._advance()
            return (tok.value, tok)
        raise ParseError("Expected norm field name", tok)

    # ── Belief base declaration ───────────────────────────────────────────

    def _belief_base_decl(self) -> BeliefBaseDecl:
        tok = self._expect(TokenType.KW_BELIEF)
        self._expect(TokenType.KW_BASE)
        name = self._any_identifier_or_keyword().value
        self._expect(TokenType.LBRACE)
        self._skip_newlines()

        entrenchment = ""
        axioms: List[AxiomDecl] = []

        while not self._check(TokenType.RBRACE) and not self._at_end():
            self._skip_newlines()
            if self._check(TokenType.RBRACE):
                break

            if self._check(TokenType.KW_AXIOM):
                axioms.append(self._axiom_decl())
            elif (self._check(TokenType.IDENTIFIER)
                  and self._peek().value == "entrenchment"):
                self._advance()
                self._expect(TokenType.COLON)
                entrenchment = self._any_identifier_or_keyword().value
            else:
                raise ParseError(
                    "Expected 'axiom' or 'entrenchment' in belief base",
                    self._peek(),
                )
            self._skip_terminators()

        self._expect(TokenType.RBRACE)
        return BeliefBaseDecl(
            name=name, entrenchment=entrenchment,
            axioms=axioms, **self._loc(tok),
        )

    def _axiom_decl(self) -> AxiomDecl:
        """``axiom "content" tier: T0_AXIOM``"""
        tok = self._expect(TokenType.KW_AXIOM)
        content = self._expect(TokenType.STRING).value
        tier = ""
        tok_next = self._peek()
        if (tok_next.type == TokenType.KW_TIER
                or (tok_next.type == TokenType.IDENTIFIER and tok_next.value == "tier")):
            self._advance()
            self._expect(TokenType.COLON)
            tier = self._any_identifier_or_keyword().value
        return AxiomDecl(content=content, tier=tier, **self._loc(tok))

    # ── Struct (type) declaration ─────────────────────────────────────────

    def _struct_decl(self) -> StructDecl:
        """``type Name { field: Type, ... }``"""
        tok = self._expect(TokenType.KW_TYPE)
        name = self._any_identifier_or_keyword().value
        self._expect(TokenType.LBRACE)
        self._skip_newlines()

        fields: List[FieldDecl] = []
        while not self._check(TokenType.RBRACE) and not self._at_end():
            self._skip_newlines()
            if self._check(TokenType.RBRACE):
                break
            fields.append(self._field_decl())
            self._match(TokenType.COMMA)
            self._skip_terminators()

        self._expect(TokenType.RBRACE)
        return StructDecl(name=name, fields=fields, **self._loc(tok))

    def _field_decl(self) -> FieldDecl:
        """``name: Type [= default]``"""
        tok = self._any_identifier_or_keyword()
        self._expect(TokenType.COLON)
        type_expr = self._type_expr()
        default: Optional[Expr] = None
        if self._match(TokenType.ASSIGN):
            default = self._expression()
        return FieldDecl(
            name=tok.value, type_expr=type_expr,
            default=default, **self._loc(tok),
        )

    # ── Enum declaration ──────────────────────────────────────────────────

    def _enum_decl(self) -> EnumDecl:
        """``enum Name { VARIANT_A, VARIANT_B = expr, ... }``"""
        tok = self._expect(TokenType.KW_ENUM)
        name = self._any_identifier_or_keyword().value
        self._expect(TokenType.LBRACE)
        self._skip_newlines()

        variants: List[str] = []
        variant_values: Dict[str, Any] = {}
        while not self._check(TokenType.RBRACE) and not self._at_end():
            self._skip_newlines()
            if self._check(TokenType.RBRACE):
                break
            vname = self._any_identifier_or_keyword().value
            variants.append(vname)
            if self._match(TokenType.ASSIGN):
                variant_values[vname] = self._expression()
            self._match(TokenType.COMMA)
            self._skip_terminators()

        self._expect(TokenType.RBRACE)
        return EnumDecl(name=name, variants=variants,
                        variant_values=variant_values, **self._loc(tok))

    # ── Function declaration ──────────────────────────────────────────────

    def _fn_decl(self) -> FnDecl:
        """``fn name(params) -> ReturnType { body }``"""
        tok = self._expect(TokenType.KW_FN)
        name = self._any_identifier_or_keyword().value
        params = self._param_list()
        ret = self._optional_return_type()
        body = self._brace_body()
        return FnDecl(
            name=name, params=params, return_type=ret,
            body=body, is_export=False, **self._loc(tok),
        )

    # ── Shared declaration helpers ────────────────────────────────────────

    def _param_list(self) -> List[Param]:
        """``(name: Type [= default], ...)``"""
        self._expect(TokenType.LPAREN)
        params: List[Param] = []
        self._skip_newlines()
        while not self._check(TokenType.RPAREN) and not self._at_end():
            self._skip_newlines()
            if self._check(TokenType.RPAREN):
                break
            params.append(self._param())
            if not self._match(TokenType.COMMA):
                break
            self._skip_newlines()
        self._skip_newlines()
        self._expect(TokenType.RPAREN)
        return params

    def _param(self) -> Param:
        is_variadic = False
        is_kw_variadic = False
        if self._check(TokenType.POWER):
            self._advance()
            is_kw_variadic = True
        elif self._check(TokenType.STAR):
            self._advance()
            is_variadic = True
        tok = self._any_identifier_or_keyword()
        type_expr: Optional[TypeExpr] = None
        default: Optional[Expr] = None
        if self._match(TokenType.COLON):
            type_expr = self._type_expr()
        if self._match(TokenType.ASSIGN):
            default = self._expression()
        return Param(
            name=tok.value, type_expr=type_expr,
            default=default, is_variadic=is_variadic,
            is_kw_variadic=is_kw_variadic, **self._loc(tok),
        )

    def _optional_return_type(self) -> Optional[TypeExpr]:
        if self._match(TokenType.ARROW):
            return self._type_expr()
        return None

    def _brace_body(self, gate_mode: bool = False) -> List[ASTNode]:
        self._expect(TokenType.LBRACE)
        self._skip_newlines()
        stmts: List[ASTNode] = []
        while not self._check(TokenType.RBRACE) and not self._at_end():
            self._skip_newlines()
            if self._check(TokenType.RBRACE):
                break
            stmts.append(self._statement(gate_mode=gate_mode))
            self._skip_terminators()
        self._expect(TokenType.RBRACE)
        return stmts

    # ── Statements ────────────────────────────────────────────────────────

    def _statement(self, gate_mode: bool = False) -> ASTNode:
        self._skip_newlines()
        tt = self._peek_type()

        if gate_mode and tt == TokenType.KW_REQUIRE:
            return self._require_stmt()
        if gate_mode and tt == TokenType.KW_VALIDATE:
            return self._validate_stmt()
        if tt == TokenType.KW_LET:
            return self._let_stmt()
        if tt == TokenType.KW_RETURN:
            return self._return_stmt()
        if tt == TokenType.KW_FOR:
            return self._for_stmt()
        if tt == TokenType.KW_IF:
            return self._if_stmt()
        if tt == TokenType.KW_MATCH:
            return self._match_stmt()
        if tt == TokenType.KW_WHILE:
            return self._while_stmt()
        if tt == TokenType.KW_WITH:
            return self._with_stmt()
        if tt == TokenType.KW_ASSERT:
            return self._assert_stmt()
        if tt == TokenType.KW_GLOBAL:
            return self._global_stmt(is_nonlocal=False)
        if tt == TokenType.KW_NONLOCAL:
            return self._global_stmt(is_nonlocal=True)
        if tt == TokenType.KW_TRY:
            return self._try_stmt()
        if tt == TokenType.KW_BREAK:
            tok_b = self._advance()
            return BreakStmt(**self._loc(tok_b))
        if tt == TokenType.KW_CONTINUE:
            tok_c = self._advance()
            return ContinueStmt(**self._loc(tok_c))
        if tt == TokenType.KW_PASS:
            tok_p = self._advance()
            return PassStmt(**self._loc(tok_p))
        if tt == TokenType.KW_RAISE:
            return self._raise_stmt()
        if tt == TokenType.KW_DEL:
            return self._del_stmt()
        if tt == TokenType.KW_ASYNC:
            self._advance()
            self._skip_newlines()
            decl = self._fn_decl()
            decl.is_async = True
            return decl
        if tt == TokenType.KW_FN:
            return self._fn_decl()
        if tt == TokenType.KW_IMPORT:
            return self._import_decl()
        if tt == TokenType.KW_FROM:
            return self._import_from_decl()
        if tt == TokenType.KW_CLASS:
            return self._class_decl()
        return self._expr_or_assign_stmt()

    def _require_stmt(self) -> RequireStmt:
        tok = self._expect(TokenType.KW_REQUIRE)
        condition = self._expression()
        return RequireStmt(condition=condition, **self._loc(tok))

    def _validate_stmt(self) -> ValidateStmt:
        tok = self._expect(TokenType.KW_VALIDATE)
        validator = self._any_identifier_or_keyword().value
        args: List[Expr] = []
        if self._match(TokenType.LPAREN):
            self._skip_newlines()
            while not self._check(TokenType.RPAREN) and not self._at_end():
                args.append(self._expression())
                if not self._match(TokenType.COMMA):
                    break
                self._skip_newlines()
            self._expect(TokenType.RPAREN)
        return ValidateStmt(validator=validator, args=args, **self._loc(tok))

    def _let_stmt(self) -> ASTNode:
        tok = self._expect(TokenType.KW_LET)
        mutable = bool(self._match(TokenType.KW_MUT))

        if self._peek_type() == TokenType.LPAREN:
            self._advance()  # consume (
            names: List[str] = [self._any_identifier_or_keyword().value]
            while self._match(TokenType.COMMA):
                if self._peek_type() == TokenType.RPAREN:
                    break
                names.append(self._any_identifier_or_keyword().value)
            self._expect(TokenType.RPAREN)
            self._expect(TokenType.ASSIGN)
            value = self._expression()
            tuple_target = ", ".join(names)
            return LetStmt(
                name=tuple_target, type_expr=None, value=value,
                mutable=mutable, **self._loc(tok),
            )

        name = self._any_identifier_or_keyword().value
        type_expr: Optional[TypeExpr] = None
        if self._match(TokenType.COLON):
            type_expr = self._type_expr()
        value: Optional[Expr] = None
        if self._match(TokenType.ASSIGN):
            value = self._expression()
        return LetStmt(
            name=name, type_expr=type_expr, value=value,
            mutable=mutable, **self._loc(tok),
        )

    def _return_stmt(self) -> ReturnStmt:
        tok = self._expect(TokenType.KW_RETURN)
        value: Optional[Expr] = None
        if not self._at_stmt_boundary():
            value = self._expression()
        return ReturnStmt(value=value, **self._loc(tok))

    def _for_stmt(self) -> ForStmt:
        tok = self._expect(TokenType.KW_FOR)
        first_var = self._any_identifier_or_keyword().value
        if self._match(TokenType.COMMA):
            extra_vars = [self._any_identifier_or_keyword().value]
            while self._match(TokenType.COMMA):
                extra_vars.append(self._any_identifier_or_keyword().value)
            variable = first_var + ", " + ", ".join(extra_vars)
        else:
            variable = first_var
        self._expect(TokenType.KW_IN)
        self._no_struct_literal = True
        iterable = self._expression()
        self._no_struct_literal = False
        body = self._brace_body()
        return ForStmt(
            variable=variable, iterable=iterable,
            body=body, **self._loc(tok),
        )

    def _if_stmt(self) -> IfExpr:
        tok = self._expect(TokenType.KW_IF)
        self._no_struct_literal = True
        condition = self._expression()
        self._no_struct_literal = False
        then_body = self._brace_body()
        else_body: List[ASTNode] = []
        self._skip_newlines()
        if self._match(TokenType.KW_ELSE):
            self._skip_newlines()
            if self._check(TokenType.KW_IF):
                else_body = [self._if_stmt()]
            else:
                else_body = self._brace_body()
        return IfExpr(
            condition=condition, then_body=then_body,
            else_body=else_body, **self._loc(tok),
        )

    def _match_stmt(self) -> MatchExpr:
        tok = self._expect(TokenType.KW_MATCH)
        self._no_struct_literal = True
        subject = self._expression()
        self._no_struct_literal = False
        self._expect(TokenType.LBRACE)
        self._skip_newlines()

        arms: List[MatchArm] = []
        while not self._check(TokenType.RBRACE) and not self._at_end():
            self._skip_newlines()
            if self._check(TokenType.RBRACE):
                break
            arms.append(self._match_arm())
            self._skip_terminators()

        self._expect(TokenType.RBRACE)
        return MatchExpr(subject=subject, arms=arms, **self._loc(tok))

    def _match_arm(self) -> MatchArm:
        tok = self._peek()
        pattern = self._expression()
        if not self._match(TokenType.ARROW):
            self._expect(TokenType.FAT_ARROW, "Expected '->' or '=>' after match pattern")
        self._skip_newlines()
        if self._check(TokenType.LBRACE):
            body = self._brace_body()
        else:
            expr = self._expression()
            body = [ExprStmt(expr=expr, line=expr.line, column=expr.column)]
        return MatchArm(pattern=pattern, body=body, **self._loc(tok))

    _AUG_ASSIGN_OPS = {
        TokenType.PLUS_ASSIGN: "+=",
        TokenType.MINUS_ASSIGN: "-=",
        TokenType.STAR_ASSIGN: "*=",
        TokenType.SLASH_ASSIGN: "/=",
        TokenType.PERCENT_ASSIGN: "%=",
    }

    def _expr_or_assign_stmt(self) -> ASTNode:
        tok = self._peek()
        expr = self._expression()
        if self._match(TokenType.ASSIGN):
            rhs = self._expression()
            return AssignStmt(target=expr, value=rhs, **self._loc(tok))
        if self._match(TokenType.COLON):
            _ann_type = self._type_expr()
            if self._match(TokenType.ASSIGN):
                rhs = self._expression()
                return AssignStmt(target=expr, value=rhs, **self._loc(tok))
            return ExprStmt(expr=expr, **self._loc(tok))
        aug_tok = self._peek()
        if aug_tok.type in self._AUG_ASSIGN_OPS:
            op = self._AUG_ASSIGN_OPS[aug_tok.type]
            self._advance()
            rhs = self._expression()
            return AugAssignStmt(target=expr, op=op, value=rhs, **self._loc(tok))
        return ExprStmt(expr=expr, **self._loc(tok))

    def _while_stmt(self) -> WhileStmt:
        tok = self._expect(TokenType.KW_WHILE)
        self._no_struct_literal = True
        condition = self._expression()
        self._no_struct_literal = False
        body = self._brace_body()
        return WhileStmt(condition=condition, body=body, **self._loc(tok))

    def _try_stmt(self) -> TryStmt:
        tok = self._expect(TokenType.KW_TRY)
        body = self._brace_body()
        catches = []
        finally_body: List[ASTNode] = []
        self._skip_newlines()
        while self._check(TokenType.KW_CATCH):
            self._advance()
            c_var = ""
            if self._match(TokenType.LPAREN):
                c_var = self._any_identifier_or_keyword().value
                self._expect(TokenType.RPAREN)
            elif self._match(TokenType.KW_AS):
                # Pythonic syntax: `catch as e { ... }` — accept and bind `e`.
                c_var = self._any_identifier_or_keyword().value
            elif not self._check(TokenType.LBRACE):
                c_var = self._any_identifier_or_keyword().value
            c_body = self._brace_body()
            catches.append((c_var, c_body))
            self._skip_newlines()
        if self._match(TokenType.KW_FINALLY):
            finally_body = self._brace_body()
        node = TryStmt(body=body, finally_body=finally_body, **self._loc(tok))
        if catches:
            node.catch_var = catches[0][0]
            node.catch_body = catches[0][1]
        return node

    def _raise_stmt(self) -> RaiseStmt:
        tok = self._expect(TokenType.KW_RAISE)
        value: Optional[Expr] = None
        cause: Optional[Expr] = None
        if not self._at_stmt_boundary():
            value = self._expression()
            if self._check(TokenType.KW_FROM):
                self._advance()
                cause = self._expression()
        return RaiseStmt(value=value, cause=cause, **self._loc(tok))

    def _del_stmt(self) -> DelStmt:
        tok = self._expect(TokenType.KW_DEL)
        target = self._expression()
        return DelStmt(target=target, **self._loc(tok))

    def _with_stmt(self) -> WithStmt:
        tok = self._advance()
        self._no_struct_literal = True
        ctx = self._expression()
        self._no_struct_literal = False
        alias = ""
        if self._check(TokenType.KW_AS):
            self._advance()
            alias = self._expect(TokenType.IDENTIFIER, "expected name after 'as'").value
        body = self._brace_body()
        return WithStmt(context=ctx, alias=alias, body=body, **self._loc(tok))

    def _assert_stmt(self) -> AssertStmt:
        tok = self._advance()
        cond = self._expression()
        msg = None
        if self._check(TokenType.COMMA):
            self._advance()
            msg = self._expression()
        return AssertStmt(condition=cond, message=msg, **self._loc(tok))

    def _global_stmt(self, is_nonlocal: bool) -> GlobalStmt:
        tok = self._advance()
        names = [self._expect(TokenType.IDENTIFIER, "expected variable name").value]
        while self._check(TokenType.COMMA):
            self._advance()
            names.append(self._expect(TokenType.IDENTIFIER, "expected variable name").value)
        return GlobalStmt(names=names, is_nonlocal=is_nonlocal, **self._loc(tok))

    def _class_decl(self) -> ClassDecl:
        """``class Name(Base1, Base2) { field: Type, fn method(self) { } }``"""
        tok = self._expect(TokenType.KW_CLASS)
        name = self._any_identifier_or_keyword().value
        base_classes: List[str] = []
        if self._match(TokenType.LPAREN):
            self._skip_newlines()
            while not self._check(TokenType.RPAREN) and not self._at_end():
                base_classes.append(self._any_identifier_or_keyword().value)
                if not self._match(TokenType.COMMA):
                    break
                self._skip_newlines()
            self._expect(TokenType.RPAREN)
        self._expect(TokenType.LBRACE)
        self._skip_newlines()
        fields: List[FieldDecl] = []
        methods: List[FnDecl] = []
        while not self._check(TokenType.RBRACE) and not self._at_end():
            self._skip_newlines()
            if self._check(TokenType.RBRACE):
                break
            if self._check(TokenType.KW_FN):
                methods.append(self._fn_decl())
            elif self._check(TokenType.KW_PASS):
                self._advance()
            elif self._check(TokenType.KW_LET):
                methods.append(self._let_stmt())
            elif (self._check(TokenType.IDENTIFIER)
                  and self._pos + 1 < len(self._tokens)
                  and self._tokens[self._pos + 1].type in (
                      TokenType.ASSIGN, TokenType.PLUS_ASSIGN,
                      TokenType.MINUS_ASSIGN, TokenType.DOT)):
                methods.append(self._expr_or_assign_stmt())
            else:
                fields.append(self._field_decl())
            self._match(TokenType.COMMA)
            self._skip_terminators()
        self._expect(TokenType.RBRACE)
        return ClassDecl(
            name=name, base_classes=base_classes,
            fields=fields, methods=methods,
            **self._loc(tok),
        )

    # ── Expressions ───────────────────────────────────────────────────────
    #
    # Precedence (lowest → highest):
    #   ||  →  &&  →  == != < > <= >=  →  + -  →  * / %  →  ! - (prefix)  →  () . []

    def _expression(self) -> Expr:
        expr = self._or_expr()
        if self._check(TokenType.KW_IF):
            self._advance()
            condition = self._or_expr()
            self._expect(TokenType.KW_ELSE, "expected 'else' in ternary")
            else_expr = self._expression()
            return TernaryExpr(condition=condition, then_expr=expr, else_expr=else_expr)
        return expr

    def _skip_if_in_parens(self) -> None:
        if self._paren_depth > 0:
            self._skip_newlines()

    def _or_expr(self) -> Expr:
        left = self._and_expr()
        while True:
            self._skip_if_in_parens()
            if not (self._check(TokenType.KW_OR) or self._check(TokenType.LOGICAL_OR)):
                break
            op_tok = self._advance()
            self._skip_newlines()
            right = self._and_expr()
            left = BinaryOp(
                left=left, op="||", right=right,
                **self._loc(op_tok),
            )
        return left

    def _and_expr(self) -> Expr:
        left = self._comparison()
        while True:
            self._skip_if_in_parens()
            if not (self._check(TokenType.KW_AND) or self._check(TokenType.LOGICAL_AND)):
                break
            op_tok = self._advance()
            self._skip_newlines()
            right = self._comparison()
            left = BinaryOp(
                left=left, op="&&", right=right,
                **self._loc(op_tok),
            )
        return left

    def _comparison(self) -> Expr:
        left = self._addition()
        while True:
            self._skip_if_in_parens()
            if self._peek_type() in _COMPARISON_OPS:
                op_tok = self._advance()
                right = self._addition()
                left = BinaryOp(
                    left=left, op=_COMPARISON_OPS[op_tok.type],
                    right=right, **self._loc(op_tok),
                )
            elif self._check(TokenType.KW_NOT) and self._pos + 1 < len(self._tokens) and self._tokens[self._pos + 1].type == TokenType.KW_IN:
                op_tok = self._advance()
                self._advance()
                right = self._addition()
                left = BinaryOp(
                    left=left, op="not in", right=right,
                    **self._loc(op_tok),
                )
            elif self._check(TokenType.KW_IN):
                op_tok = self._advance()
                right = self._addition()
                left = BinaryOp(
                    left=left, op="in", right=right,
                    **self._loc(op_tok),
                )
            elif self._check(TokenType.KW_IS):
                op_tok = self._advance()
                right = self._addition()
                left = BinaryOp(
                    left=left, op="is", right=right,
                    **self._loc(op_tok),
                )
            else:
                break
        return left

    def _addition(self) -> Expr:
        left = self._multiplication()
        while True:
            self._skip_if_in_parens()
            if self._peek_type() not in _ADDITIVE_OPS:
                break
            op_tok = self._advance()
            self._skip_newlines()
            right = self._multiplication()
            left = BinaryOp(
                left=left, op=op_tok.value, right=right,
                **self._loc(op_tok),
            )
        return left

    def _multiplication(self) -> Expr:
        left = self._unary()
        while True:
            self._skip_if_in_parens()
            if self._peek_type() not in _MULTIPLICATIVE_OPS:
                break
            op_tok = self._advance()
            self._skip_newlines()
            right = self._unary()
            left = BinaryOp(
                left=left, op=op_tok.value, right=right,
                **self._loc(op_tok),
            )
        return left

    def _unary(self) -> Expr:
        if self._check(TokenType.BANG, TokenType.KW_NOT):
            op_tok = self._advance()
            operand = self._unary()
            return UnaryOp(op="!", operand=operand, **self._loc(op_tok))
        if self._check(TokenType.MINUS):
            op_tok = self._advance()
            operand = self._unary()
            return UnaryOp(op="-", operand=operand, **self._loc(op_tok))
        return self._postfix()

    def _postfix(self) -> Expr:
        expr = self._primary()
        if (isinstance(expr, Identifier) and self._check(TokenType.LBRACE)
                and not getattr(self, '_no_struct_literal', False)):
            expr = self._struct_literal(expr)
        while True:
            if self._check(TokenType.LPAREN):
                expr = self._finish_call(expr)
            elif self._match(TokenType.DOT):
                member = self._any_identifier_or_keyword()
                expr = MemberAccess(
                    object=expr, member=member.value,
                    line=expr.line, column=expr.column,
                )
            elif self._check(TokenType.LBRACKET):
                self._advance()
                if self._check(TokenType.COLON):
                    self._advance()
                    upper = None
                    if not self._check(TokenType.RBRACKET):
                        upper = self._expression()
                    self._expect(TokenType.RBRACKET)
                    expr = IndexExpr(
                        object=expr,
                        index=SliceExpr(lower=None, upper=upper,
                                        line=expr.line, column=expr.column),
                        line=expr.line, column=expr.column,
                    )
                else:
                    index = self._expression()
                    if self._match(TokenType.COLON):
                        upper = None
                        if not self._check(TokenType.RBRACKET):
                            upper = self._expression()
                        self._expect(TokenType.RBRACKET)
                        expr = IndexExpr(
                            object=expr,
                            index=SliceExpr(lower=index, upper=upper,
                                            line=expr.line, column=expr.column),
                            line=expr.line, column=expr.column,
                        )
                    else:
                        self._expect(TokenType.RBRACKET)
                        expr = IndexExpr(
                            object=expr, index=index,
                            line=expr.line, column=expr.column,
                        )
            else:
                break
        return expr

    def _finish_call(self, callee: Expr) -> CallExpr:
        self._expect(TokenType.LPAREN)
        args: List[Expr] = []
        self._skip_newlines()
        while not self._check(TokenType.RPAREN) and not self._at_end():
            self._skip_newlines()
            if self._check(TokenType.RPAREN):
                break
            if (self._check(TokenType.IDENTIFIER)
                    and self._pos + 1 < len(self._tokens)
                    and self._tokens[self._pos + 1].type in (TokenType.ASSIGN, TokenType.COLON)):
                kw_name = self._advance().value
                self._advance()
                kw_val = self._expression()
                args.append(KeywordArg(name=kw_name, value=kw_val,
                                       line=callee.line, column=callee.column))
            elif (self._peek().type.name.startswith("KW_")
                    and self._pos + 1 < len(self._tokens)
                    and self._tokens[self._pos + 1].type in (TokenType.ASSIGN, TokenType.COLON)):
                kw_name = self._advance().value
                self._advance()
                kw_val = self._expression()
                args.append(KeywordArg(name=kw_name, value=kw_val,
                                       line=callee.line, column=callee.column))
            elif self._check(TokenType.POWER):
                self._advance()
                args.append(SpreadExpr(value=self._expression(), is_double=True,
                                       line=callee.line, column=callee.column))
            elif self._check(TokenType.STAR):
                self._advance()
                args.append(SpreadExpr(value=self._expression(), is_double=False,
                                       line=callee.line, column=callee.column))
            else:
                args.append(self._expression())
            if not self._match(TokenType.COMMA):
                break
            self._skip_newlines()
        self._skip_newlines()
        self._expect(TokenType.RPAREN)
        return CallExpr(
            callee=callee, args=args,
            line=callee.line, column=callee.column,
        )

    def _struct_literal(self, name_expr: Identifier) -> CallExpr:
        """``TypeName { field: value, field: value }`` → CallExpr."""
        self._expect(TokenType.LBRACE)
        self._skip_newlines()
        args: List[Expr] = []
        while not self._check(TokenType.RBRACE) and not self._at_end():
            self._skip_newlines()
            if self._check(TokenType.RBRACE):
                break
            key = self._any_identifier_or_keyword().value
            self._expect(TokenType.COLON)
            val = self._expression()
            args.append(BinaryOp(
                left=Identifier(name=key, line=name_expr.line, column=name_expr.column),
                op=":",
                right=val,
                line=name_expr.line, column=name_expr.column,
            ))
            self._match(TokenType.COMMA)
            self._skip_newlines()
        self._expect(TokenType.RBRACE)
        return CallExpr(
            callee=name_expr, args=args,
            line=name_expr.line, column=name_expr.column,
        )

    def _primary(self) -> Expr:
        tok = self._peek()
        tt = tok.type

        if tt == TokenType.INTEGER:
            self._advance()
            return Literal(value=int(tok.value, 0), literal_type="int", **self._loc(tok))

        if tt == TokenType.FLOAT:
            self._advance()
            return Literal(value=float(tok.value), literal_type="float", **self._loc(tok))

        if tt == TokenType.STRING:
            self._advance()
            return Literal(value=tok.value, literal_type="string", **self._loc(tok))

        if tt in (TokenType.KW_TRUE, TokenType.BOOLEAN) and tok.value == "true":
            self._advance()
            return Literal(value=True, literal_type="bool", **self._loc(tok))

        if tt in (TokenType.KW_FALSE, TokenType.BOOLEAN) and tok.value == "false":
            self._advance()
            return Literal(value=False, literal_type="bool", **self._loc(tok))

        if tt in (TokenType.KW_NONE, TokenType.NONE):
            self._advance()
            return Literal(value=None, literal_type="none", **self._loc(tok))

        if tt == TokenType.IDENTIFIER:
            self._advance()
            return Identifier(name=tok.value, **self._loc(tok))

        if tt in (TokenType.KW_SELF, TokenType.KW_SUPER):
            self._advance()
            return Identifier(name=tok.value, **self._loc(tok))

        if tt == TokenType.KW_SEMANTIC_DISTANCE:
            return self._semantic_distance_expr()
        if tt == TokenType.KW_DECOMPOSE:
            return self._decompose_expr()
        if tt == TokenType.KW_INTENT_MATCH:
            return self._intent_match_expr()

        if tt == TokenType.KW_YIELD:
            self._advance()
            is_from = False
            if self._check(TokenType.KW_FROM):
                self._advance()
                is_from = True
            val = None
            if (not self._check(TokenType.NEWLINE)
                    and not self._check(TokenType.RBRACE)
                    and not self._check(TokenType.RPAREN)):
                val = self._expression()
            return YieldExpr(value=val, is_from=is_from, **self._loc(tok))

        if tt == TokenType.KW_AWAIT:
            self._advance()
            return AwaitExpr(value=self._unary(), **self._loc(tok))

        if tt == TokenType.KW_LAMBDA:
            self._advance()
            lam_params: List[Param] = []
            if not self._check(TokenType.COLON):
                while True:
                    is_var = is_kw_var = False
                    if self._check(TokenType.POWER):
                        self._advance(); is_kw_var = True
                    elif self._check(TokenType.STAR):
                        self._advance(); is_var = True
                    p_tok = self._any_identifier_or_keyword()
                    p_default = None
                    if self._match(TokenType.ASSIGN):
                        p_default = self._expression()
                    lam_params.append(Param(
                        name=p_tok.value, default=p_default,
                        is_variadic=is_var, is_kw_variadic=is_kw_var,
                        **self._loc(p_tok),
                    ))
                    if not self._check(TokenType.COMMA):
                        break
                    self._advance()
            self._expect(TokenType.COLON, "expected ':' in lambda")
            lam_body = self._expression()
            return LambdaExpr(params=lam_params, body=lam_body, **self._loc(tok))

        _EXPR_EXCLUDED_KW = frozenset({
            TokenType.KW_IF, TokenType.KW_ELSE,
            TokenType.KW_FOR, TokenType.KW_LET, TokenType.KW_MUT,
            TokenType.KW_RETURN, TokenType.KW_IMPORT,
            TokenType.KW_FROM, TokenType.KW_EXPORT,
            TokenType.KW_TRUE, TokenType.KW_FALSE,
            TokenType.KW_NONE, TokenType.KW_WHILE,
            TokenType.KW_TRY, TokenType.KW_CATCH,
            TokenType.KW_BREAK, TokenType.KW_CONTINUE, TokenType.KW_PASS,
            TokenType.KW_RAISE, TokenType.KW_DEL, TokenType.KW_CLASS,
            TokenType.KW_YIELD, TokenType.KW_AWAIT, TokenType.KW_LAMBDA,
        })
        if tt.name.startswith("KW_") and tt not in _EXPR_EXCLUDED_KW:
            if tt == TokenType.KW_FN and self._pos + 1 < len(self._tokens) and self._tokens[self._pos + 1].type == TokenType.LPAREN:
                return self._anon_fn_expr()
            self._advance()
            return Identifier(name=tok.value, **self._loc(tok))

        if tt == TokenType.LPAREN:
            self._advance()
            self._paren_depth += 1
            self._skip_newlines()
            expr = self._expression()
            if self._check(TokenType.KW_FOR):
                comp = self._parse_comprehension(expr, "generator")
                self._skip_newlines()
                self._expect(TokenType.RPAREN)
                self._paren_depth -= 1
                return comp
            if self._match(TokenType.COMMA):
                self._skip_newlines()
                elements = [expr]
                while not self._check(TokenType.RPAREN) and not self._at_end():
                    self._skip_newlines()
                    if self._check(TokenType.RPAREN):
                        break
                    elements.append(self._expression())
                    if not self._match(TokenType.COMMA):
                        break
                    self._skip_newlines()
                self._skip_newlines()
                self._expect(TokenType.RPAREN)
                self._paren_depth -= 1
                return CallExpr(
                    callee=Identifier(name="__tuple__", **self._loc(tok)),
                    args=elements, **self._loc(tok),
                )
            self._skip_newlines()
            self._expect(TokenType.RPAREN)
            self._paren_depth -= 1
            return expr

        if tt == TokenType.LBRACKET:
            return self._list_literal()

        if tt == TokenType.LBRACE:
            return self._dict_literal()

        if tt == TokenType.KW_IF:
            return self._if_stmt()

        if tt == TokenType.KW_MATCH:
            return self._match_stmt()

        raise ParseError(
            f"Unexpected token {tt.name} ({tok.value!r})", tok,
        )

    # ── Cognitive primitive parsers ─────────────────────────────────────

    def _compose_decl(self) -> ComposeDecl:
        """``compose answer from facts, web { strategy: "synthesis" }``"""
        tok = self._expect(TokenType.KW_COMPOSE)
        name = self._any_identifier_or_keyword().value
        sources: List[Expr] = []
        if self._match(TokenType.KW_FROM):
            sources.append(self._expression())
            while self._match(TokenType.COMMA):
                self._skip_newlines()
                sources.append(self._expression())

        annotations: Dict[str, Any] = {}

        # The parser may greedily consume the annotation block as call
        # args on the last source.  Detect and unwrap that case.
        if sources:
            last = sources[-1]
            if (type(last).__name__ == "CallExpr"
                    and getattr(last, "args", [])
                    and all(
                        type(a).__name__ in ("BinOpExpr", "BinaryOp")
                        and getattr(a, "op", "") == ":"
                        for a in last.args)):
                for a in last.args:
                    key = getattr(getattr(a, "left", None), "name", "")
                    if key:
                        annotations[key] = a.right
                sources[-1] = getattr(last, "callee", last)

        if not annotations:
            annotations = self._parse_annotation_block()

        return ComposeDecl(name=name, sources=sources,
                           annotations=annotations, **self._loc(tok))

    def _persist_decl(self) -> PersistDecl:
        """``persist store { decay: 0.01, backing: "disk" }``"""
        tok = self._expect(TokenType.KW_PERSIST)
        name = self._any_identifier_or_keyword().value
        config = self._parse_annotation_block()
        return PersistDecl(name=name, config=config, **self._loc(tok))

    def _semantic_distance_expr(self) -> SemanticDistanceExpr:
        """``semantic_distance(a, b)``"""
        tok = self._expect(TokenType.KW_SEMANTIC_DISTANCE)
        self._expect(TokenType.LPAREN)
        left = self._expression()
        self._expect(TokenType.COMMA)
        self._skip_newlines()
        right = self._expression()
        self._expect(TokenType.RPAREN)
        annotations = self._parse_annotation_block()
        return SemanticDistanceExpr(left=left, right=right,
                                    annotations=annotations, **self._loc(tok))

    def _decompose_expr(self) -> DecomposeExpr:
        """``decompose(goal)``"""
        tok = self._expect(TokenType.KW_DECOMPOSE)
        self._expect(TokenType.LPAREN)
        goal = self._expression()
        self._expect(TokenType.RPAREN)
        annotations = self._parse_annotation_block()
        return DecomposeExpr(goal=goal, annotations=annotations,
                             **self._loc(tok))

    def _intent_match_expr(self) -> IntentMatchExpr:
        """``intent_match(query, beliefs)``"""
        tok = self._expect(TokenType.KW_INTENT_MATCH)
        self._expect(TokenType.LPAREN)
        query_expr = self._expression()
        self._expect(TokenType.COMMA)
        self._skip_newlines()
        belief_base = self._expression()
        self._expect(TokenType.RPAREN)
        annotations = self._parse_annotation_block()
        return IntentMatchExpr(query_expr=query_expr, belief_base=belief_base,
                               annotations=annotations, **self._loc(tok))

    def _parse_annotation_block(self) -> Dict[str, Any]:
        """Parse optional ``{ key: value, ... }`` annotation block."""
        annotations: Dict[str, Any] = {}
        if not self._check(TokenType.LBRACE):
            return annotations
        self._advance()
        self._skip_newlines()
        while not self._check(TokenType.RBRACE) and not self._at_end():
            key_tok = self._any_identifier_or_keyword()
            self._expect(TokenType.COLON)
            self._skip_newlines()
            val = self._expression()
            annotations[key_tok.value] = val
            self._match(TokenType.COMMA)
            self._skip_newlines()
        self._expect(TokenType.RBRACE)
        return annotations

    def _dict_literal(self) -> Expr:
        """``{key: value, ...}`` or ``{elem, elem, ...}`` (set) — returns DictExpr or CallExpr(Set)."""
        tok = self._expect(TokenType.LBRACE)
        self._skip_newlines()
        if self._check(TokenType.RBRACE):
            self._advance()
            return DictExpr(pairs=[], **self._loc(tok))
        first = self._expression()
        if self._match(TokenType.COLON):
            self._skip_newlines()
            first_val = self._expression()
            if self._check(TokenType.KW_FOR):
                comp = self._parse_comprehension(None, "dict")
                comp.key = first
                comp.value = first_val
                self._skip_newlines()
                self._expect(TokenType.RBRACE)
                return comp
            pairs = [(first, first_val)]
            while self._match(TokenType.COMMA):
                self._skip_newlines()
                if self._check(TokenType.RBRACE):
                    break
                key = self._expression()
                self._expect(TokenType.COLON)
                self._skip_newlines()
                val = self._expression()
                pairs.append((key, val))
            self._skip_newlines()
            self._expect(TokenType.RBRACE)
            return DictExpr(pairs=pairs, **self._loc(tok))
        else:
            if self._check(TokenType.KW_FOR):
                comp = self._parse_comprehension(first, "set")
                self._skip_newlines()
                self._expect(TokenType.RBRACE)
                return comp
            elements = [first]
            while self._match(TokenType.COMMA):
                self._skip_newlines()
                if self._check(TokenType.RBRACE):
                    break
                elements.append(self._expression())
            self._skip_newlines()
            self._expect(TokenType.RBRACE)
            return CallExpr(
                callee=Identifier(name="Set", **self._loc(tok)),
                args=elements, **self._loc(tok),
            )

    def _anon_fn_expr(self) -> Expr:
        """``fn(params) { body }`` — anonymous function expression."""
        tok = self._expect(TokenType.KW_FN)
        params = self._param_list()
        ret = self._optional_return_type()
        body = self._brace_body()
        fn_decl = FnDecl(name="<lambda>", params=params, return_type=ret,
                         body=body, **self._loc(tok))
        return fn_decl

    def _list_literal(self) -> Expr:
        """``[expr, expr, ...]`` or ``[expr for var in iterable if cond]``."""
        tok = self._expect(TokenType.LBRACKET)
        self._skip_newlines()
        if self._check(TokenType.RBRACKET):
            self._advance()
            return CallExpr(
                callee=Identifier(name="List", **self._loc(tok)),
                args=[], **self._loc(tok),
            )
        first = self._expression()
        if self._check(TokenType.KW_FOR):
            comp = self._parse_comprehension(first, "list")
            self._skip_newlines()
            self._expect(TokenType.RBRACKET)
            return comp
        elements: List[Expr] = [first]
        while self._match(TokenType.COMMA):
            self._skip_newlines()
            if self._check(TokenType.RBRACKET):
                break
            elements.append(self._expression())
        self._skip_newlines()
        self._expect(TokenType.RBRACKET)
        return CallExpr(
            callee=Identifier(name="List", **self._loc(tok)),
            args=elements, **self._loc(tok),
        )

    def _parse_comprehension(self, element: Optional[Expr], kind: str) -> ComprehensionExpr:
        """Parse ``for var in iterable [if cond]`` tail of a comprehension."""
        self._advance()  # consume 'for'
        var_name = self._expect(TokenType.IDENTIFIER, "expected variable in comprehension").value
        self._expect(TokenType.KW_IN, "expected 'in' in comprehension")
        iterable = self._or_expr()
        condition = None
        if self._check(TokenType.KW_IF):
            self._advance()
            condition = self._or_expr()
        return ComprehensionExpr(
            element=element, variable=var_name,
            iterable=iterable, condition=condition, kind=kind,
        )

    # ── Type expressions ──────────────────────────────────────────────────

    def _type_expr(self) -> TypeExpr:
        """Parse a type, including union (``A | B``)."""
        left = self._type_primary()
        while self._check(TokenType.PIPE):
            self._advance()
            right = self._type_primary()
            if isinstance(left, UnionType):
                left.types.append(right)
            else:
                left = UnionType(
                    types=[left, right],
                    line=left.line, column=left.column,
                )
        return left

    def _type_primary(self) -> TypeExpr:
        tok = self._peek()

        if tok.type in _TRUST_KEYWORDS:
            return self._parse_trust_type()

        if tok.type == TokenType.LPAREN:
            return self._function_type()

        if self._check(TokenType.IDENTIFIER):
            name_tok = self._advance()
        elif self._peek().type.name.startswith("KW_"):
            name_tok = self._advance()
        else:
            name_tok = self._expect(TokenType.IDENTIFIER, "Expected type name")
        if self._check(TokenType.LBRACKET):
            return self._generic_type(name_tok)
        return SimpleType(name=name_tok.value, **self._loc(name_tok))

    def _parse_trust_type(self) -> TrustType:
        tok = self._advance()
        inner: Optional[TypeExpr] = None
        if self._match(TokenType.LBRACKET):
            inner = self._type_expr()
            self._expect(TokenType.RBRACKET)
        return TrustType(
            trust_level=tok.value, inner_type=inner,
            **self._loc(tok),
        )

    def _generic_type(self, name_tok: Token) -> GenericType:
        self._expect(TokenType.LBRACKET)
        type_args: List[TypeExpr] = []
        self._skip_newlines()
        while not self._check(TokenType.RBRACKET) and not self._at_end():
            self._skip_newlines()
            if self._check(TokenType.RBRACKET):
                break
            type_args.append(self._type_expr())
            if not self._match(TokenType.COMMA):
                break
            self._skip_newlines()
        self._skip_newlines()
        self._expect(TokenType.RBRACKET)
        return GenericType(
            name=name_tok.value, type_args=type_args,
            **self._loc(name_tok),
        )

    def _function_type(self) -> FunctionType:
        """``(ParamType, ...) -> ReturnType``"""
        tok = self._expect(TokenType.LPAREN)
        params: List[TypeExpr] = []
        self._skip_newlines()
        while not self._check(TokenType.RPAREN) and not self._at_end():
            params.append(self._type_expr())
            if not self._match(TokenType.COMMA):
                break
            self._skip_newlines()
        self._expect(TokenType.RPAREN)
        self._expect(TokenType.ARROW)
        return_type = self._type_expr()
        return FunctionType(
            params=params, return_type=return_type,
            **self._loc(tok),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience
# ═══════════════════════════════════════════════════════════════════════════════

def parse(source: str, filename: str = "<stdin>") -> Module:
    """Lex and parse NRSI source text, returning an AST ``Module``."""
    lexer = Lexer(source, filename=filename)
    tokens = lexer.tokenize()
    return Parser(tokens, filename=filename).parse()
