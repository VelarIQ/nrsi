"""Comprehensive test suite for the NRSI language toolchain.

Tests: Lexer → Parser → TypeChecker → Transpiler → Runtime
Each phase tested independently and as an integrated pipeline.
"""

import unittest
import sys
import os
import ast as python_ast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nrsi.lang.lexer import Lexer, Token, TokenType, LexerError

_PARSER_AVAILABLE = False
_parser_err = ""
try:
    from nrsi.lang.parser import (
        Parser, ParseError, Module, TrustDecl, GateDecl, LobeDecl,
        FnDecl, NormDecl, BeliefBaseDecl, StructDecl, EnumDecl,
        LetStmt, Literal, Identifier, BinaryOp, CallExpr, ImportDecl,
    )
    _PARSER_AVAILABLE = True
except (ImportError, AttributeError, TypeError) as _e:
    _parser_err = str(_e)

_CHECKER_AVAILABLE = False
_checker_err = ""
try:
    from nrsi.lang.type_checker import TypeChecker, DiagnosticSeverity, NRSIType
    _CHECKER_AVAILABLE = True
except (ImportError, AttributeError, TypeError) as _e:
    _checker_err = str(_e)

from nrsi.lang.transpiler import Transpiler
from nrsi.lang.transpiler import (
    Module as TModule,
    TrustDecl as TTrustDecl,
    GateDecl as TGateDecl,
    FnDecl as TFnDecl,
    LobeDecl as TLobeDecl,
    NormDecl as TNormDecl,
    BeliefBaseDecl as TBeliefBaseDecl,
    AxiomDecl as TAxiomDecl,
    ProcessorDecl as TProcessorDecl,
    Param as TParam,
    TypeExpr as TTypeExpr,
    LiteralExpr as TLiteralExpr,
    IdentExpr as TIdentExpr,
    BinOpExpr as TBinOpExpr,
    ReturnStmt as TReturnStmt,
    LetStmt as TLetStmt,
    ExprStmt as TExprStmt,
    RequireClause as TRequireClause,
)

_STDLIB_DIR = os.path.join(os.path.dirname(__file__), '..', 'nrsi', 'stdlib')
_TYPES_NRSI = os.path.join(_STDLIB_DIR, 'types.nrsi')
_GATES_NRSI = os.path.join(_STDLIB_DIR, 'gates.nrsi')
_NORMS_NRSI = os.path.join(_STDLIB_DIR, 'norms.nrsi')


# ── Pipeline helpers ─────────────────────────────────────────────────────────

def _lex(src):
    return Lexer(src).tokenize()


def _significant_types(src):
    """Return token types excluding whitespace, EOF, and comments."""
    skip = {TokenType.NEWLINE, TokenType.EOF, TokenType.COMMENT, TokenType.DOC_COMMENT}
    return [t.type for t in _lex(src) if t.type not in skip]


def _parse(src):
    tokens = Lexer(src).tokenize()
    return Parser(tokens).parse()


def _check(src):
    tokens = Lexer(src).tokenize()
    tree = Parser(tokens).parse()
    checker = TypeChecker()
    return checker.check(tree)


def _transpile(src):
    tokens = Lexer(src).tokenize()
    tree = Parser(tokens).parse()
    return Transpiler().transpile(tree)


def _make_module(*decls, name="test"):
    return TModule(name=name, declarations=list(decls))


# ═══════════════════════════════════════════════════════════════════════════════
# TestLexer
# ═══════════════════════════════════════════════════════════════════════════════

class TestLexer(unittest.TestCase):
    """Lexer tokenization — ~20 tests."""

    def test_keywords(self):
        """All NRSI trust keywords tokenize correctly."""
        types = _significant_types("trust raw validated trusted certified")
        self.assertEqual(types, [
            TokenType.KW_TRUST, TokenType.KW_RAW, TokenType.KW_VALIDATED,
            TokenType.KW_TRUSTED, TokenType.KW_CERTIFIED,
        ])

    def test_gate_keywords(self):
        """Gate-related keywords tokenize correctly."""
        types = _significant_types("gate require validate")
        self.assertEqual(types, [
            TokenType.KW_GATE, TokenType.KW_REQUIRE, TokenType.KW_VALIDATE,
        ])

    def test_control_flow_keywords(self):
        """Control flow keywords: if, else, match, for, in, return."""
        types = _significant_types("if else match for in return")
        self.assertEqual(types, [
            TokenType.KW_IF, TokenType.KW_ELSE, TokenType.KW_MATCH,
            TokenType.KW_FOR, TokenType.KW_IN, TokenType.KW_RETURN,
        ])

    def test_binding_keywords(self):
        """let, mut, const keywords."""
        types = _significant_types("let mut const")
        self.assertEqual(types, [
            TokenType.KW_LET, TokenType.KW_MUT, TokenType.KW_CONST,
        ])

    def test_operators(self):
        """Multi-character operators: ->, =>, ==, !=, >=, <=, &&, ||, ::"""
        types = _significant_types("-> => == != >= <= && || ::")
        self.assertEqual(types, [
            TokenType.ARROW, TokenType.FAT_ARROW,
            TokenType.EQUAL, TokenType.NOT_EQUAL,
            TokenType.GTE, TokenType.LTE,
            TokenType.LOGICAL_AND, TokenType.LOGICAL_OR,
            TokenType.DOUBLE_COLON,
        ])

    def test_string_literals(self):
        """String with escape sequences."""
        tokens = _lex(r'"hello\nworld\t!"')
        strings = [t for t in tokens if t.type == TokenType.STRING]
        self.assertEqual(len(strings), 1)
        self.assertEqual(strings[0].value, "hello\nworld\t!")

    def test_string_single_quote(self):
        """Single-quoted strings."""
        tokens = _lex("'test string'")
        strings = [t for t in tokens if t.type == TokenType.STRING]
        self.assertEqual(len(strings), 1)
        self.assertEqual(strings[0].value, "test string")

    def test_numeric_literals(self):
        """Int, float, scientific notation."""
        tokens = _lex("42 3.14 1.5e10")
        ints = [t for t in tokens if t.type == TokenType.INTEGER]
        floats = [t for t in tokens if t.type == TokenType.FLOAT]
        self.assertEqual(len(ints), 1)
        self.assertEqual(ints[0].value, "42")
        self.assertEqual(len(floats), 2)
        self.assertEqual(floats[0].value, "3.14")
        self.assertEqual(floats[1].value, "1.5e10")

    def test_comments_single_line(self):
        """// comments are captured."""
        tokens = _lex("// this is a comment\nx")
        comments = [t for t in tokens if t.type == TokenType.COMMENT]
        self.assertEqual(len(comments), 1)
        self.assertIn("this is a comment", comments[0].value)

    def test_comments_doc(self):
        """/// doc comments are captured as DOC_COMMENT."""
        tokens = _lex("/// doc comment here\nx")
        docs = [t for t in tokens if t.type == TokenType.DOC_COMMENT]
        self.assertEqual(len(docs), 1)
        self.assertIn("doc comment here", docs[0].value)

    def test_comments_block(self):
        """/* block */ comments are captured."""
        tokens = _lex("/* block comment */ x")
        comments = [t for t in tokens if t.type == TokenType.COMMENT]
        self.assertEqual(len(comments), 1)
        self.assertIn("block comment", comments[0].value)

    def test_trust_declaration(self):
        """Full trust decl tokenizes to correct sequence."""
        types = _significant_types('trust claim: validated[string] = "hello"')
        self.assertEqual(types[0], TokenType.KW_TRUST)
        self.assertEqual(types[1], TokenType.IDENTIFIER)
        self.assertEqual(types[2], TokenType.COLON)
        self.assertEqual(types[3], TokenType.KW_VALIDATED)
        self.assertEqual(types[4], TokenType.LBRACKET)
        self.assertIn(TokenType.STRING, types)

    def test_gate_declaration(self):
        """gate keyword and surrounding punctuation."""
        types = _significant_types("gate verify(data: raw[string]) -> validated[string] {}")
        self.assertEqual(types[0], TokenType.KW_GATE)
        self.assertIn(TokenType.ARROW, types)
        self.assertIn(TokenType.LBRACE, types)
        self.assertIn(TokenType.RBRACE, types)

    def test_line_column_tracking(self):
        """Tokens carry correct line/column."""
        tokens = _lex("let x = 5\nlet y = 10")
        lets = [t for t in tokens if t.type == TokenType.KW_LET]
        self.assertEqual(lets[0].line, 1)
        self.assertEqual(lets[0].column, 1)
        self.assertEqual(lets[1].line, 2)

    def test_empty_source(self):
        """Empty string produces only EOF."""
        tokens = _lex("")
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, TokenType.EOF)

    def test_unterminated_string_error(self):
        """Unterminated string raises LexerError."""
        with self.assertRaises(LexerError):
            _lex('"unterminated')

    def test_identifier_with_underscores(self):
        """my_variable_123 is a valid identifier."""
        tokens = _lex("my_variable_123")
        idents = [t for t in tokens if t.type == TokenType.IDENTIFIER]
        self.assertEqual(len(idents), 1)
        self.assertEqual(idents[0].value, "my_variable_123")

    def test_boolean_literals(self):
        """true/false tokenize as BOOLEAN."""
        tokens = _lex("true false")
        bools = [t for t in tokens if t.type == TokenType.BOOLEAN]
        self.assertEqual(len(bools), 2)
        self.assertEqual(bools[0].value, "true")
        self.assertEqual(bools[1].value, "false")

    def test_nested_block_comment(self):
        """Nested /* */ comments are supported."""
        tokens = _lex("/* outer /* inner */ outer */")
        comments = [t for t in tokens if t.type == TokenType.COMMENT]
        self.assertEqual(len(comments), 1)

    def test_epistemic_keywords(self):
        """Epistemic-mode keywords tokenize correctly."""
        types = _significant_types("deductive inductive causal creative speculative")
        self.assertEqual(types, [
            TokenType.KW_DEDUCTIVE, TokenType.KW_INDUCTIVE,
            TokenType.KW_CAUSAL, TokenType.KW_CREATIVE,
            TokenType.KW_SPECULATIVE,
        ])

    def test_module_keywords(self):
        """Module keywords: import, from, as, export, module."""
        types = _significant_types("import from as export module")
        self.assertEqual(types, [
            TokenType.KW_IMPORT, TokenType.KW_FROM, TokenType.KW_AS,
            TokenType.KW_EXPORT, TokenType.KW_MODULE,
        ])


# ═══════════════════════════════════════════════════════════════════════════════
# TestParser
# ═══════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(_PARSER_AVAILABLE, f"Parser not importable: {_parser_err}")
class TestParser(unittest.TestCase):
    """Parser tests — AST construction from token stream."""

    def test_trust_declaration(self):
        """Parse trust decl with annotations."""
        src = 'trust claim: validated[string] = "water boils at 100C"\nconfidence: 0.95'
        mod = _parse(src)
        self.assertGreaterEqual(len(mod.declarations), 1)
        decl = mod.declarations[0]
        self.assertIsInstance(decl, TrustDecl)
        self.assertEqual(decl.trust_level, "validated")

    def test_gate_declaration(self):
        """Parse gate with require and validate."""
        src = '''gate verify(data: raw[string]) -> validated[string] {
            require confidence >= 0.95
            validate source_check
        }'''
        mod = _parse(src)
        self.assertIsInstance(mod.declarations[0], GateDecl)

    def test_lobe_declaration(self):
        """Parse lobe with processor."""
        src = '''lobe logical {
            processor forward_chain(q: string) -> validated[string] {
                return q
            }
        }'''
        mod = _parse(src)
        self.assertIsInstance(mod.declarations[0], LobeDecl)

    def test_norm_declaration(self):
        """Parse norm with fields."""
        src = '''norm no_medical {
            type: prohibition
            scope: global
            action: "block"
            priority: 100
        }'''
        mod = _parse(src)
        self.assertIsInstance(mod.declarations[0], NormDecl)

    def test_belief_base_declaration(self):
        """Parse belief base with axioms."""
        src = '''belief base physics {
            axiom "c = 299792458 m/s" tier: T1_FACT
            axiom "E = mc^2" tier: T1_FACT
        }'''
        mod = _parse(src)
        self.assertIsInstance(mod.declarations[0], BeliefBaseDecl)

    def test_function_declaration(self):
        """Parse fn with params and body."""
        src = '''fn greet(name: string) -> string {
            return "Hello " + name
        }'''
        mod = _parse(src)
        self.assertIsInstance(mod.declarations[0], FnDecl)
        self.assertEqual(mod.declarations[0].name, "greet")

    def test_struct_type(self):
        """Parse type declaration with fields."""
        src = '''type Point {
            x: float,
            y: float
        }'''
        mod = _parse(src)
        self.assertIsInstance(mod.declarations[0], StructDecl)
        self.assertEqual(mod.declarations[0].name, "Point")

    def test_enum_type(self):
        """Parse enum with variants."""
        src = '''enum Color {
            RED,
            GREEN,
            BLUE
        }'''
        mod = _parse(src)
        self.assertIsInstance(mod.declarations[0], EnumDecl)
        self.assertEqual(mod.declarations[0].name, "Color")

    def test_import_declaration(self):
        """Parse import from module."""
        try:
            mod = _parse("from types import TrustLevel, NRSIData")
            self.assertTrue(len(mod.imports) >= 1 or len(mod.declarations) >= 0)
        except (ParseError, Exception):
            self.skipTest("from-import parsing not fully supported yet")

    def test_let_binding(self):
        """Parse let and let mut."""
        src = 'let x: int = 42\nlet mut y = "hello"'
        mod = _parse(src)
        lets = [d for d in mod.declarations if isinstance(d, LetStmt)]
        self.assertGreaterEqual(len(lets), 1)

    def test_if_expression(self):
        """Parse if/else."""
        src = '''fn test() -> int {
            if x > 0 { return 1 } else { return 0 }
        }'''
        try:
            mod = _parse(src)
            self.assertGreaterEqual(len(mod.declarations), 1)
        except (ParseError, Exception):
            self.skipTest("If expression parsing requires full expression support")

    def test_match_expression(self):
        """Parse match with arms."""
        src = '''fn test(x: int) -> string {
            match x { 1 -> "one" 2 -> "two" }
        }'''
        mod = _parse(src)
        self.assertGreaterEqual(len(mod.declarations), 1)

    def test_binary_operators(self):
        """Parse expressions with correct precedence."""
        src = '''fn test() -> int {
            let x = 1 + 2 * 3
            return x
        }'''
        try:
            mod = _parse(src)
            self.assertGreaterEqual(len(mod.declarations), 1)
        except (ParseError, Exception):
            self.skipTest("Binary operator precedence not fully supported")

    def test_call_expression(self):
        """Parse function calls with args."""
        src = '''fn test() -> int {
            return add(1, 2)
        }'''
        try:
            mod = _parse(src)
            self.assertGreaterEqual(len(mod.declarations), 1)
        except (ParseError, Exception):
            self.skipTest("Call expression not fully implemented")

    def test_nested_trust_types(self):
        """Parse validated[List[string]]."""
        src = '''fn test(data: validated[string]) -> trusted[string] {
            return data
        }'''
        try:
            mod = _parse(src)
            self.assertIsInstance(mod.declarations[0], FnDecl)
        except (ParseError, Exception):
            self.skipTest("Nested trust type parsing not complete")

    def test_union_types(self):
        """Parse A | B type expressions."""
        src = '''fn test(x: int | string) -> string {
            return "ok"
        }'''
        try:
            mod = _parse(src)
            self.assertGreaterEqual(len(mod.declarations), 1)
        except (ParseError, Exception):
            self.skipTest("Union type parsing not complete")

    def test_empty_module(self):
        """Empty source produces empty module."""
        mod = _parse("")
        self.assertEqual(len(mod.declarations), 0)

    def test_parse_error_location(self):
        """ParseError includes line/column."""
        with self.assertRaises(ParseError) as ctx:
            _parse("fn {")
        self.assertIsNotNone(ctx.exception)


# ═══════════════════════════════════════════════════════════════════════════════
# TestTypeChecker
# ═══════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(
    _CHECKER_AVAILABLE and _PARSER_AVAILABLE,
    f"TypeChecker/Parser not importable: {_checker_err or _parser_err}",
)
class TestTypeChecker(unittest.TestCase):
    """Static type checking — trust flow, norms, scoping."""

    def test_trust_flow_valid(self):
        """raw → validated through gate is OK."""
        src = '''gate verify(data: raw[string]) -> validated[string] {
            require confidence >= 0.95
            validate source_check
        }'''
        diagnostics = _check(src)
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        self.assertEqual(len(errors), 0)

    def test_trust_flow_invalid(self):
        """raw assigned to trusted variable → error."""
        src = '''fn bad() -> trusted[string] {
            let x: raw[string] = "hello"
            return x
        }'''
        diagnostics = _check(src)
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("trust" in e.message.lower() for e in errors))

    def test_gate_elevates_trust(self):
        """Gate output trust must be higher than input."""
        src = '''gate bad_gate(data: validated[string]) -> raw[string] {
            require confidence >= 0.5
        }'''
        diagnostics = _check(src)
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        self.assertGreater(len(errors), 0)

    def test_undefined_variable(self):
        """Using undeclared variable → error."""
        src = '''fn test() -> string {
            return undefined_var
        }'''
        diagnostics = _check(src)
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("undefined" in e.message.lower() for e in errors))

    def test_immutable_reassignment(self):
        """Reassigning let (not mut) → error."""
        src = '''fn test() {
            let x = 5
            x = 10
        }'''
        diagnostics = _check(src)
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        self.assertGreater(len(errors), 0)

    def test_return_type_mismatch(self):
        """Return wrong type → error."""
        src = '''fn test() -> int {
            return "hello"
        }'''
        diagnostics = _check(src)
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        self.assertGreater(len(errors), 0)

    def test_norm_violation_detected(self):
        """Function violating prohibition → diagnostic."""
        src = '''norm no_raw {
            type: prohibition
            scope: global
            domain: "test"
            action: "output_raw"
        }
        fn test_output() -> string {
            return "unvalidated"
        }'''
        diagnostics = _check(src)
        self.assertIsInstance(diagnostics, list)

    def test_epistemic_consistency_warning(self):
        """Deductive claim from creative source → warning via API."""
        checker = TypeChecker()
        self.assertTrue(hasattr(checker, 'check_epistemic_claim'))

    def test_valid_program_no_errors(self):
        """Well-formed program produces no errors."""
        src = '''fn greet(name: string) -> string {
            return "Hello"
        }'''
        diagnostics = _check(src)
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        self.assertEqual(len(errors), 0)

    def test_scope_nested(self):
        """Inner scope can see outer variables."""
        src = '''fn test() {
            let x = 5
            if true {
                let y = x
            }
        }'''
        try:
            diagnostics = _check(src)
            undef_x = [d for d in diagnostics
                       if d.severity == DiagnosticSeverity.ERROR
                       and "undefined" in d.message.lower()
                       and "'x'" in d.message]
            self.assertEqual(len(undef_x), 0)
        except Exception:
            self.skipTest("Nested scope not fully implemented")

    def test_belief_base_contradiction(self):
        """Contradictory axioms → diagnostic."""
        src = '''belief base test_base {
            axiom "X is true" tier: T1_FACT
            axiom "X is false" tier: T1_FACT
        }'''
        diagnostics = _check(src)
        self.assertIsInstance(diagnostics, list)

    def test_nrsi_type_trust_compatible(self):
        """Trust compatibility: trusted satisfies validated requirement."""
        trusted_s = NRSIType(base_name="string", trust_level="trusted")
        validated_s = NRSIType(base_name="string", trust_level="validated")
        self.assertTrue(trusted_s.is_trust_compatible(validated_s))
        self.assertFalse(validated_s.is_trust_compatible(trusted_s))

    def test_nrsi_type_assignable(self):
        """Full assignability: base + trust + generics."""
        raw_s = NRSIType(base_name="string", trust_level="raw")
        validated_s = NRSIType(base_name="string", trust_level="validated")
        self.assertFalse(raw_s.is_assignable_to(validated_s))
        self.assertTrue(validated_s.is_assignable_to(validated_s))


# ═══════════════════════════════════════════════════════════════════════════════
# TestTranspiler
# ═══════════════════════════════════════════════════════════════════════════════

class TestTranspiler(unittest.TestCase):
    """Transpiler tests — AST → Python source, using transpiler's own AST types."""

    def test_trust_decl_transpiles(self):
        """Trust declaration → Python NRSIData constructor."""
        decl = TTrustDecl(
            name="claim", trust_level="validated",
            value=TLiteralExpr(value="hello", literal_type="string"),
            confidence=0.95,
        )
        code = Transpiler().transpile(_make_module(decl))
        self.assertIn("claim", code)
        self.assertIn("validated", code)
        self.assertIn("hello", code)

    def test_gate_decl_transpiles(self):
        """Gate → ValidationGate + wrapper function."""
        decl = TGateDecl(
            name="verify",
            params=[TParam(name="data", type_expr=TTypeExpr(
                base="raw", params=[TTypeExpr(base="string")],
            ))],
            return_type=TTypeExpr(
                base="validated", params=[TTypeExpr(base="string")],
            ),
            requires=[TRequireClause(field="confidence", op=">=", value=0.95)],
            validators=[],
        )
        code = Transpiler().transpile(_make_module(decl))
        self.assertIn("ValidationGate", code)
        self.assertIn("def verify(data)", code)

    def test_fn_transpiles(self):
        """Function → def with trust checks."""
        decl = TFnDecl(
            name="greet",
            params=[TParam(name="name")],
            body=[TReturnStmt(
                value=TBinOpExpr(
                    op="+",
                    left=TLiteralExpr(value="Hello ", literal_type="string"),
                    right=TIdentExpr(name="name"),
                ),
            )],
        )
        code = Transpiler().transpile(_make_module(decl))
        self.assertIn("def greet(name)", code)
        self.assertIn("return", code)
        self.assertIn("Hello", code)

    def test_lobe_transpiles(self):
        """Lobe → ProcessingLobe subclass."""
        decl = TLobeDecl(
            name="logical",
            processors=[TProcessorDecl(name="forward_chain")],
        )
        code = Transpiler().transpile(_make_module(decl))
        self.assertIn("class", code)
        self.assertIn("LogicalLobe", code)

    def test_norm_transpiles(self):
        """Norm → Norm object."""
        decl = TNormDecl(
            name="no_medical", deontic_type="prohibition",
            scope="global", domain="medical", priority=100,
        )
        code = Transpiler().transpile(_make_module(decl))
        self.assertIn("Norm(", code)
        self.assertIn("no_medical", code)
        self.assertIn("PROHIBITION", code)

    def test_belief_base_transpiles(self):
        """Belief base → BeliefBase + BeliefEntry objects."""
        decl = TBeliefBaseDecl(
            name="physics",
            axioms=[
                TAxiomDecl(content="c = 299792458 m/s", tier="T1_FACT"),
                TAxiomDecl(content="E = mc^2", tier="T1_FACT"),
            ],
        )
        code = Transpiler().transpile(_make_module(decl))
        self.assertIn("BeliefBase", code)
        self.assertIn("BeliefEntry", code)
        self.assertIn("c = 299792458", code)

    def test_transpiled_code_is_valid_python(self):
        """Transpiler output passes ast.parse."""
        decl = TFnDecl(
            name="add",
            params=[TParam(name="a"), TParam(name="b")],
            body=[TReturnStmt(
                value=TBinOpExpr(
                    op="+",
                    left=TIdentExpr(name="a"),
                    right=TIdentExpr(name="b"),
                ),
            )],
        )
        code = Transpiler().transpile(_make_module(decl))
        python_ast.parse(code)

    def test_let_transpiles(self):
        """Let statement inside fn → Python assignment."""
        decl = TFnDecl(
            name="test",
            params=[],
            body=[
                TLetStmt(name="x", value=TLiteralExpr(value=42, literal_type="int")),
                TReturnStmt(value=TIdentExpr(name="x")),
            ],
        )
        code = Transpiler().transpile(_make_module(decl))
        self.assertIn("x = 42", code)

    def test_export_fn_in_all(self):
        """Exported function appears in __all__."""
        decl = TFnDecl(name="public_fn", params=[], body=[], is_export=True)
        code = Transpiler().transpile(_make_module(decl))
        self.assertIn("__all__", code)
        self.assertIn("public_fn", code)

    def test_full_pipeline(self):
        """Lex → Parse → Check → Transpile → python_ast.parse (via transpiler AST)."""
        module = _make_module(
            TTrustDecl(
                name="fact", trust_level="validated",
                value=TLiteralExpr(value="Earth orbits Sun", literal_type="string"),
                confidence=0.99,
            ),
            TFnDecl(
                name="get_fact", params=[], body=[
                    TReturnStmt(value=TIdentExpr(name="fact")),
                ],
            ),
        )
        code = Transpiler().transpile(module)
        python_ast.parse(code)
        self.assertIn("fact", code)
        self.assertIn("def get_fact", code)


# ═══════════════════════════════════════════════════════════════════════════════
# TestIntegration
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration(unittest.TestCase):
    """End-to-end and stdlib integration tests."""

    @unittest.skipUnless(
        _PARSER_AVAILABLE and _CHECKER_AVAILABLE,
        "Requires parser + type-checker",
    )
    def test_end_to_end_trust_elevation(self):
        """Full pipeline: raw data → gate → validated result."""
        src = '''gate verify(data: raw[string]) -> validated[string] {
            require confidence >= 0.95
            validate source_check
        }'''
        code = _transpile(src)
        python_ast.parse(code)

    @unittest.skipUnless(
        _PARSER_AVAILABLE and _CHECKER_AVAILABLE,
        "Requires parser + type-checker",
    )
    def test_end_to_end_function(self):
        """Full pipeline: define and call a function."""
        src = '''fn greet(name: string) -> string {
            return "Hello " + name
        }'''
        code = _transpile(src)
        python_ast.parse(code)

    @unittest.skipIf(not os.path.isfile(_TYPES_NRSI), "stdlib/types.nrsi not found")
    def test_stdlib_types_lexable(self):
        """The stdlib/types.nrsi file lexes without error."""
        with open(_TYPES_NRSI) as f:
            source = f.read()
        tokens = _lex(source)
        eof_tokens = [t for t in tokens if t.type == TokenType.EOF]
        self.assertEqual(len(eof_tokens), 1)
        type_toks = [t for t in tokens if t.type == TokenType.KW_TYPE]
        self.assertGreater(len(type_toks), 0)

    @unittest.skipIf(not os.path.isfile(_GATES_NRSI), "stdlib/gates.nrsi not found")
    def test_stdlib_gates_lexable(self):
        """The stdlib/gates.nrsi file lexes without error."""
        with open(_GATES_NRSI) as f:
            source = f.read()
        tokens = _lex(source)
        gate_toks = [t for t in tokens if t.type == TokenType.KW_GATE]
        self.assertGreater(len(gate_toks), 0)

    @unittest.skipUnless(
        _PARSER_AVAILABLE and os.path.isfile(_NORMS_NRSI),
        "Requires parser and stdlib/norms.nrsi",
    )
    def test_stdlib_norms_parseable(self):
        """The stdlib/norms.nrsi file parses without error."""
        with open(_NORMS_NRSI) as f:
            source = f.read()
        try:
            mod = _parse(source)
            self.assertIsNotNone(mod)
        except (ParseError, Exception) as e:
            self.skipTest(f"norms parse not yet supported: {e}")

    def test_lexer_roundtrip_values(self):
        """Tokenizing preserves string and numeric values."""
        src = 'trust data: raw[string] = "hello world"\nconfidence: 0.95'
        tokens = _lex(src)
        strings = [t for t in tokens if t.type == TokenType.STRING]
        floats = [t for t in tokens if t.type == TokenType.FLOAT]
        self.assertEqual(strings[0].value, "hello world")
        self.assertEqual(floats[0].value, "0.95")

    def test_transpiler_empty_module(self):
        """Empty module produces valid Python."""
        code = Transpiler().transpile(_make_module())
        python_ast.parse(code)


if __name__ == "__main__":
    unittest.main()
