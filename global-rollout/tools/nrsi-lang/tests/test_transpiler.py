"""Smoke tests for the NRSI transpiler."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nrsi.lang.lexer import Lexer
from nrsi.lang.parser import Parser
from nrsi.lang.transpiler import Transpiler


def _transpile(src):
    tokens = Lexer(src, "<test>").tokenize()
    ast = Parser(tokens, src).parse()
    t = Transpiler()
    return t.transpile(ast)


def test_with_transpiles():
    code = _transpile('module t\nfn main() { with open("f") as h { let x = 1 } }')
    assert "with" in code
    assert "as h" in code


def test_assert_transpiles():
    code = _transpile('module t\nfn main() { assert x > 0, "msg" }')
    assert "assert" in code


def test_ternary_transpiles():
    code = _transpile('module t\nfn main() { let r = a if b else c }')
    assert "if" in code
    assert "else" in code


def test_comprehension_transpiles():
    code = _transpile('module t\nfn main() { let x = [i for i in items] }')
    assert "for" in code
    assert "in" in code


def test_floor_div_transpiles():
    code = _transpile('module t\nfn main() { let x = a ~~ b }')
    assert "//" in code


def test_power_transpiles():
    code = _transpile('module t\nfn main() { let x = a ** b }')
    assert "**" in code
