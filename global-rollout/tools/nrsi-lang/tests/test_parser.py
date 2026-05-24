"""Smoke tests for the NRSI parser."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nrsi.lang.lexer import Lexer
from nrsi.lang.parser import Parser


def _parse(src):
    tokens = Lexer(src, "<test>").tokenize()
    return Parser(tokens, src).parse()


def test_with_stmt():
    tree = _parse('module t\nfn main() { with open("f") as h { let x = 1 } }')
    assert tree is not None


def test_assert_stmt():
    tree = _parse('module t\nfn main() { assert x > 0, "positive" }')
    assert tree is not None


def test_ternary():
    tree = _parse('module t\nfn main() { let x = a if b > 0 else c }')
    assert tree is not None


def test_list_comprehension():
    tree = _parse('module t\nfn main() { let x = [i for i in items if i > 0] }')
    assert tree is not None


def test_lambda():
    tree = _parse('module t\nfn main() { let f = lambda x: x + 1 }')
    assert tree is not None


def test_yield():
    tree = _parse('module t\nfn gen() { yield 42 }')
    assert tree is not None


def test_multiple_catch():
    tree = _parse('module t\nfn main() { try { let x = 1 } catch(e) { print(e) } catch(e2) { log(e2) } }')
    assert tree is not None


def test_raise_from():
    tree = _parse('module t\nfn main() { raise ValueError("x") from original }')
    assert tree is not None
