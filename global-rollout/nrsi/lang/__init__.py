"""NRSI Language Toolchain — Lexer, Parser, Type Checker, Transpiler, Runtime.

The nrsi.lang package provides the complete toolchain for the NRSI
programming language:

  Lexer        — .nrsi source → token stream
  Parser       — tokens → Abstract Syntax Tree
  TypeChecker  — AST → diagnostics (trust flow, gate requirements, norm compliance)
  Transpiler   — AST → Python source (using nrsi.core types)
  Runtime      — .nrsi files → Python modules (full pipeline)

Usage::

    from nrsi.lang import compile_nrsi, load_nrsi

    # Compile to Python:
    python_code = compile_nrsi(nrsi_source)

    # Load directly:
    module = load_nrsi("path/to/file.nrsi")
"""

from nrsi.lang.runtime import load_nrsi, compile_nrsi, install, NRSICompileError
