"""Test suite for NRSI CLI, multi-target transpilation, and target registry.

Tests: CLI argument parsing, target registry, Python/Swift/Kotlin output,
stdlib compilation across all targets, and end-to-end pipeline validation.
"""

import ast as python_ast
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nrsi.lang.lexer import Lexer
from nrsi.lang.parser import Parser
from nrsi.lang.targets import get_transpiler, list_targets, REGISTRY


# ── Helpers ──────────────────────────────────────────────────────────────────

_STDLIB_DIR = os.path.join(os.path.dirname(__file__), "..", "nrsi", "stdlib")

_SIMPLE_GATE = """\
gate verify(data: raw[string]) -> validated[string] {
    require confidence >= 0.85
    validate source_check
}
"""

_SIMPLE_FN = """\
fn analyze(query: string) -> validated[string] {
    let raw_data: raw[string] = query
    let verified: validated[string] = verify(raw_data)
    return verified
}
"""

_SIMPLE_NORM = """\
norm no_unvalidated_medical {
    type: prohibition
    scope: domain
    action: "output_raw"
    priority: 100
}
"""

_SIMPLE_BELIEF = """\
belief base physics {
    entrenchment: empirical
    axiom "c = 299792458 m/s" tier: T0_AXIOM
    axiom "E = mc^2" tier: T0_AXIOM
}
"""

_SIMPLE_TRUST = """\
trust claim: validated[string] = "water boils at 100C"
    confidence: 0.95
    epistemic: observational
"""

_FULL_PROGRAM = _SIMPLE_GATE + "\n" + _SIMPLE_NORM + "\n" + _SIMPLE_FN


def _parse(src: str):
    tokens = Lexer(src).tokenize()
    return Parser(tokens).parse()


def _compile(src: str, target: str) -> str:
    tree = _parse(src)
    cls = get_transpiler(target)
    return cls().transpile(tree)


# ── Target Registry Tests ───────────────────────────────────────────────────

class TestTargetRegistry(unittest.TestCase):
    def test_python_registered(self):
        self.assertIn("python", list_targets())

    def test_swift_registered(self):
        self.assertIn("swift", list_targets())

    def test_kotlin_registered(self):
        self.assertIn("kotlin", list_targets())

    def test_unknown_target_raises(self):
        with self.assertRaises(ValueError):
            get_transpiler("brainfuck")

    def test_all_targets_have_transpile(self):
        for name in list_targets():
            cls = get_transpiler(name)
            self.assertTrue(hasattr(cls, "transpile"))

    def test_all_targets_instantiate(self):
        for name in list_targets():
            cls = get_transpiler(name)
            instance = cls()
            self.assertIsNotNone(instance)


# ── Python Target Tests ─────────────────────────────────────────────────────

class TestPythonTarget(unittest.TestCase):
    def test_gate_produces_valid_python(self):
        code = _compile(_SIMPLE_GATE, "python")
        python_ast.parse(code)

    def test_fn_produces_valid_python(self):
        code = _compile(_SIMPLE_FN, "python")
        python_ast.parse(code)

    def test_norm_produces_valid_python(self):
        code = _compile(_SIMPLE_NORM, "python")
        python_ast.parse(code)

    def test_belief_produces_valid_python(self):
        code = _compile(_SIMPLE_BELIEF, "python")
        python_ast.parse(code)

    def test_trust_produces_valid_python(self):
        code = _compile(_SIMPLE_TRUST, "python")
        python_ast.parse(code)

    def test_full_program_valid_python(self):
        code = _compile(_FULL_PROGRAM, "python")
        python_ast.parse(code)

    def test_output_contains_validation_gate(self):
        code = _compile(_SIMPLE_GATE, "python")
        self.assertIn("ValidationGate", code)

    def test_output_contains_norm(self):
        code = _compile(_SIMPLE_NORM, "python")
        self.assertIn("Norm(", code)


# ── Swift Target Tests ───────────────────────────────────────────────────────

class TestSwiftTarget(unittest.TestCase):
    def test_gate_produces_swift(self):
        code = _compile(_SIMPLE_GATE, "swift")
        self.assertIn("func verify(", code)
        self.assertIn("precondition(", code)
        self.assertIn("NRSIData<String>", code)

    def test_fn_produces_swift(self):
        code = _compile(_SIMPLE_FN, "swift")
        self.assertIn("func analyze(", code)
        self.assertIn("return", code)

    def test_norm_produces_swift(self):
        code = _compile(_SIMPLE_NORM, "swift")
        self.assertIn("NRSINorm(", code)
        self.assertIn(".prohibition", code)

    def test_belief_produces_swift(self):
        code = _compile(_SIMPLE_BELIEF, "swift")
        self.assertIn("BeliefBase(", code)
        self.assertIn("Axiom(", code)

    def test_trust_produces_swift(self):
        code = _compile(_SIMPLE_TRUST, "swift")
        self.assertIn("NRSIData<", code)
        self.assertIn(".validated", code)

    def test_preamble_included(self):
        code = _compile(_SIMPLE_GATE, "swift")
        self.assertIn("enum TrustLevel", code)
        self.assertIn("struct NRSIData<T>", code)

    def test_camel_case_conversion(self):
        code = _compile(_SIMPLE_FN, "swift")
        self.assertIn("rawData", code)

    def test_full_program_swift(self):
        code = _compile(_FULL_PROGRAM, "swift")
        self.assertIn("func verify(", code)
        self.assertIn("NRSINorm(", code)
        self.assertIn("func analyze(", code)


# ── Kotlin Target Tests ──────────────────────────────────────────────────────

class TestKotlinTarget(unittest.TestCase):
    def test_gate_produces_kotlin(self):
        code = _compile(_SIMPLE_GATE, "kotlin")
        self.assertIn("fun verify(", code)
        self.assertIn("require(", code)
        self.assertIn("NRSIData<String>", code)

    def test_fn_produces_kotlin(self):
        code = _compile(_SIMPLE_FN, "kotlin")
        self.assertIn("fun analyze(", code)
        self.assertIn("return", code)

    def test_norm_produces_kotlin(self):
        code = _compile(_SIMPLE_NORM, "kotlin")
        self.assertIn("NRSINorm(", code)
        self.assertIn("DeonticType.PROHIBITION", code)

    def test_belief_produces_kotlin(self):
        code = _compile(_SIMPLE_BELIEF, "kotlin")
        self.assertIn("BeliefBase(", code)
        self.assertIn("Axiom(", code)
        self.assertIn("listOf(", code)

    def test_trust_produces_kotlin(self):
        code = _compile(_SIMPLE_TRUST, "kotlin")
        self.assertIn("NRSIData<", code)
        self.assertIn("TrustLevel.VALIDATED", code)

    def test_preamble_included(self):
        code = _compile(_SIMPLE_GATE, "kotlin")
        self.assertIn("enum class TrustLevel", code)
        self.assertIn("data class NRSIData<T>", code)

    def test_camel_case_conversion(self):
        code = _compile(_SIMPLE_FN, "kotlin")
        self.assertIn("rawData", code)

    def test_full_program_kotlin(self):
        code = _compile(_FULL_PROGRAM, "kotlin")
        self.assertIn("fun verify(", code)
        self.assertIn("NRSINorm(", code)
        self.assertIn("fun analyze(", code)

    def test_val_vs_var(self):
        src = """fn test() {
            let x: int = 5
            let mut y: int = 10
        }"""
        code = _compile(src, "kotlin")
        self.assertIn("val x", code)
        self.assertIn("var y", code)


# ── Cross-Target Consistency ─────────────────────────────────────────────────

class TestCrossTarget(unittest.TestCase):
    def test_same_ast_all_targets(self):
        """All targets should accept the same parsed AST."""
        tree = _parse(_FULL_PROGRAM)
        for target in list_targets():
            cls = get_transpiler(target)
            code = cls().transpile(tree)
            self.assertIsInstance(code, str)
            self.assertGreater(len(code), 50)

    def test_gate_in_all_targets(self):
        """Gate declaration should produce gate-like output in all targets."""
        tree = _parse(_SIMPLE_GATE)
        for target in list_targets():
            code = get_transpiler(target)().transpile(tree)
            self.assertIn("verify", code.lower())

    def test_norm_in_all_targets(self):
        tree = _parse(_SIMPLE_NORM)
        for target in list_targets():
            code = get_transpiler(target)().transpile(tree)
            self.assertIn("medical", code.lower())


# ── Stdlib Compilation Tests ─────────────────────────────────────────────────

class TestStdlibCompilation(unittest.TestCase):
    """Verify that all stdlib .nrsi files compile to every target."""

    def _stdlib_files(self):
        if not os.path.isdir(_STDLIB_DIR):
            self.skipTest("stdlib not found")
        return sorted(
            os.path.join(_STDLIB_DIR, f)
            for f in os.listdir(_STDLIB_DIR)
            if f.endswith(".nrsi")
        )

    def test_stdlib_compiles_to_python(self):
        for path in self._stdlib_files():
            source = open(path).read()
            tree = _parse(source)
            code = get_transpiler("python")().transpile(tree)
            self.assertGreater(len(code), 10, f"{path} produced empty Python")

    def test_stdlib_compiles_to_swift(self):
        for path in self._stdlib_files():
            source = open(path).read()
            tree = _parse(source)
            code = get_transpiler("swift")().transpile(tree)
            self.assertIn("TrustLevel", code, f"{path} missing Swift preamble")

    def test_stdlib_compiles_to_kotlin(self):
        for path in self._stdlib_files():
            source = open(path).read()
            tree = _parse(source)
            code = get_transpiler("kotlin")().transpile(tree)
            self.assertIn("TrustLevel", code, f"{path} missing Kotlin preamble")


# ── CLI Tests ────────────────────────────────────────────────────────────────

class TestCLI(unittest.TestCase):
    def test_build_parser(self):
        from nrsi.lang.cli import build_parser
        parser = build_parser()
        self.assertIsNotNone(parser)

    def test_compile_to_stdout(self):
        from nrsi.lang.cli import _compile_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".nrsi", delete=False) as f:
            f.write(_SIMPLE_GATE)
            f.flush()
            try:
                code = _compile_file(f.name, "python")
                self.assertIn("ValidationGate", code)
            finally:
                os.unlink(f.name)

    def test_compile_swift_via_cli(self):
        from nrsi.lang.cli import _compile_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".nrsi", delete=False) as f:
            f.write(_SIMPLE_GATE)
            f.flush()
            try:
                code = _compile_file(f.name, "swift")
                self.assertIn("func verify(", code)
            finally:
                os.unlink(f.name)

    def test_compile_kotlin_via_cli(self):
        from nrsi.lang.cli import _compile_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".nrsi", delete=False) as f:
            f.write(_SIMPLE_GATE)
            f.flush()
            try:
                code = _compile_file(f.name, "kotlin")
                self.assertIn("fun verify(", code)
            finally:
                os.unlink(f.name)

    def test_check_valid_file(self):
        from nrsi.lang.cli import _check_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".nrsi", delete=False) as f:
            f.write(_SIMPLE_GATE)
            f.flush()
            try:
                errors = _check_file(f.name)
                self.assertEqual(errors, 0)
            finally:
                os.unlink(f.name)

    def test_check_bad_syntax(self):
        from nrsi.lang.cli import _check_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".nrsi", delete=False) as f:
            f.write("fn {{{{{ broken")
            f.flush()
            try:
                errors = _check_file(f.name)
                self.assertGreater(errors, 0)
            finally:
                os.unlink(f.name)

    def test_collect_files_from_dir(self):
        from nrsi.lang.cli import _collect_nrsi_files
        if os.path.isdir(_STDLIB_DIR):
            files = _collect_nrsi_files([_STDLIB_DIR])
            self.assertGreater(len(files), 0)
            self.assertTrue(all(f.endswith(".nrsi") for f in files))

    def test_targets_command(self):
        from nrsi.lang.cli import cmd_targets
        import argparse
        ns = argparse.Namespace()
        cmd_targets(ns)


if __name__ == "__main__":
    unittest.main()
