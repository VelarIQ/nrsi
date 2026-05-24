"""Smoke tests for the NRSI lexer."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nrsi.lang.lexer import Lexer, TokenType


def test_basic_tokens():
    tokens = Lexer("let x = 42", "<test>").tokenize()
    types = [t.type for t in tokens if t.type not in (TokenType.NEWLINE, TokenType.EOF)]
    assert TokenType.KW_LET in types
    assert TokenType.IDENTIFIER in types
    assert TokenType.ASSIGN in types
    assert TokenType.INTEGER in types


def test_hex_literal():
    tokens = Lexer("let x = 0xFF", "<test>").tokenize()
    nums = [t for t in tokens if t.type == TokenType.INTEGER]
    assert len(nums) == 1
    assert nums[0].value == "0xFF"


def test_binary_literal():
    tokens = Lexer("let y = 0b1010", "<test>").tokenize()
    nums = [t for t in tokens if t.type == TokenType.INTEGER]
    assert len(nums) == 1
    assert nums[0].value == "0b1010"


def test_triple_quoted_string():
    tokens = Lexer('let s = \"\"\"hello\nworld\"\"\"', "<test>").tokenize()
    strings = [t for t in tokens if t.type == TokenType.STRING]
    assert len(strings) == 1
    assert "hello" in strings[0].value
    assert "world" in strings[0].value


def test_power_operator():
    tokens = Lexer("x ** 2", "<test>").tokenize()
    ops = [t for t in tokens if t.type == TokenType.POWER]
    assert len(ops) == 1


def test_floor_div():
    tokens = Lexer("x ~~ 2", "<test>").tokenize()
    ops = [t for t in tokens if t.type == TokenType.FLOOR_DIV]
    assert len(ops) == 1


def test_keywords():
    for kw in ["with", "async", "await", "yield", "lambda", "nonlocal"]:
        tokens = Lexer(kw, "<test>").tokenize()
        assert tokens[0].type.name.startswith("KW_"), f"{kw} not recognized as keyword"


def test_fstring_marker():
    tokens = Lexer('f"hello {name}"', "<test>").tokenize()
    strings = [t for t in tokens if t.type == TokenType.STRING]
    assert len(strings) == 1
    assert strings[0].value.startswith("\x00f\x00")
