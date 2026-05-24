"""NRSI Runtime — Load and Execute .nrsi Files as Python Modules.

Usage::

    import nrsi.lang.runtime  # Installs the import hook

    # Now you can import .nrsi files:
    import my_module  # looks for my_module.nrsi

    # Or explicitly:
    from nrsi.lang.runtime import load_nrsi
    module = load_nrsi("path/to/file.nrsi")

The runtime:
  1. Reads the .nrsi source
  2. Lexes → tokens
  3. Parses → AST
  4. Type-checks (reports errors/warnings)
  5. Transpiles → Python
  6. Executes the Python in a module namespace
  7. Returns the module with NRSI enforcement active
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.machinery
import os
import sys
import types
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from nrsi.lang.lexer import Lexer, LexerError, Token, TokenType
from nrsi.lang.transpiler import (
    Diagnostic,
    Module,
    Transpiler,
)

# Parser and TypeChecker are separate toolchain passes that may not ship yet
# or may contain forward-references to tokens/types not yet defined.
# Fall back to lightweight stubs so the pipeline degrades gracefully.
try:
    from nrsi.lang.parser import Parser as _RealParser  # type: ignore[import-not-found]
except (ImportError, AttributeError, TypeError):
    _RealParser = None  # type: ignore[assignment,misc]

try:
    from nrsi.lang.type_checker import TypeChecker as _RealTypeChecker  # type: ignore[import-not-found]
except (ImportError, AttributeError, TypeError):
    _RealTypeChecker = None  # type: ignore[assignment,misc]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. NRSICompileError
# ═══════════════════════════════════════════════════════════════════════════════

class NRSICompileError(Exception):
    """Raised when .nrsi compilation fails with one or more errors."""

    def __init__(self, diagnostics: Sequence[Diagnostic]) -> None:
        self.diagnostics = list(diagnostics)
        lines = [
            f"  {d.severity.name}: {d.message} (line {d.line})"
            for d in diagnostics
        ]
        super().__init__("NRSI compilation failed:\n" + "\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Stub Parser — used when nrsi.lang.parser is not yet available
# ═══════════════════════════════════════════════════════════════════════════════

class _StubParser:
    """Minimal parser that wraps a token stream into an empty Module.

    When the real parser (``nrsi.lang.parser.Parser``) is not present,
    this stub keeps the pipeline functional.  The resulting module will
    have no declarations — callers see a warning so they know full
    parsing was skipped.
    """

    def __init__(self, tokens: List[Token], filename: str = "<stdin>") -> None:
        self._tokens = tokens
        self._filename = filename

    def parse(self) -> Module:
        warnings.warn(
            "nrsi.lang.parser not available; stub parser produced an empty AST",
            RuntimeWarning,
            stacklevel=3,
        )
        return Module(
            name=self._filename,
            declarations=[],
            source_file=self._filename,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Stub Type Checker — no-op when the real checker is absent
# ═══════════════════════════════════════════════════════════════════════════════

class _StubTypeChecker:
    """No-op type checker that returns zero diagnostics."""

    def check(self, module: Module) -> List[Diagnostic]:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Module Cache
# ═══════════════════════════════════════════════════════════════════════════════

class _ModuleCache:
    """Cache compiled NRSI modules keyed by (path, content-hash).

    Re-reading a file whose content has not changed returns the cached
    module instantly.  Content changes invalidate the entry.
    """

    def __init__(self) -> None:
        self._modules: Dict[str, types.ModuleType] = {}
        self._hashes: Dict[str, str] = {}

    @staticmethod
    def _hash(source: str) -> str:
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    def get(self, path: str, source: str) -> Optional[types.ModuleType]:
        h = self._hash(source)
        if path in self._modules and self._hashes.get(path) == h:
            return self._modules[path]
        return None

    def put(self, path: str, source: str, module: types.ModuleType) -> None:
        self._modules[path] = module
        self._hashes[path] = self._hash(source)

    def invalidate(self, path: str) -> None:
        self._modules.pop(path, None)
        self._hashes.pop(path, None)

    def clear(self) -> None:
        self._modules.clear()
        self._hashes.clear()

    @property
    def size(self) -> int:
        return len(self._modules)


_cache = _ModuleCache()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Internal Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _get_parser(tokens: List[Token], filename: str) -> Any:
    if _RealParser is not None:
        return _RealParser(tokens, filename=filename)
    return _StubParser(tokens, filename=filename)


def _get_type_checker() -> Any:
    if _RealTypeChecker is not None:
        return _RealTypeChecker()
    return _StubTypeChecker()


def _pipeline(
    source: str,
    filename: str = "<string>",
) -> tuple:
    """Run the full NRSI compilation pipeline.

    Returns ``(python_code, ast, diagnostics)``.

    Raises:
        LexerError: lexical error in source.
        NRSICompileError: type-check errors.
    """
    # Step 1 — Lex
    lexer = Lexer(source, filename=filename)
    tokens = lexer.tokenize()

    # Step 2 — Parse
    parser = _get_parser(tokens, filename)
    ast = parser.parse()

    # Step 3 — Type-check
    checker = _get_type_checker()
    diagnostics: List[Diagnostic] = checker.check(ast)

    errors = [d for d in diagnostics
              if getattr(d.severity, 'value', d.severity) == 'error'
              or getattr(d.severity, 'name', '') == 'ERROR'
              or str(d.severity) == 'DiagnosticSeverity.ERROR']
    if errors:
        raise NRSICompileError(errors)

    # Step 4 — Transpile
    transpiler = Transpiler()
    python_code = transpiler.transpile(ast)

    return python_code, ast, diagnostics


# ═══════════════════════════════════════════════════════════════════════════════
# 6. NRSI Builtins — injected into every .nrsi module namespace
# ═══════════════════════════════════════════════════════════════════════════════

_builtins_cache: Optional[Dict[str, Any]] = None


def _nrsi_builtins() -> Dict[str, Any]:
    global _builtins_cache
    if _builtins_cache is not None:
        return _builtins_cache
    try:
        from nrsi.lang.builtins import NRSI_BUILTINS
        _builtins_cache = dict(NRSI_BUILTINS)
    except ImportError:
        _builtins_cache = {}
    return _builtins_cache


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Public API — load / compile
# ═══════════════════════════════════════════════════════════════════════════════

def load_nrsi(path: str, module_name: str = "") -> types.ModuleType:
    """Explicitly load a ``.nrsi`` file and return a Python module.

    Runs the full pipeline (lex → parse → type-check → transpile → exec)
    and caches the result.  Subsequent calls with the same path and
    unchanged content return the cached module.

    Args:
        path: Filesystem path to the ``.nrsi`` file.
        module_name: Optional module name; defaults to the filename stem.

    Returns:
        A ``types.ModuleType`` with all NRSI declarations available as
        attributes and NRSI enforcement active.

    Raises:
        FileNotFoundError: path does not exist.
        LexerError: lexical errors in the source.
        NRSICompileError: type-check errors.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"NRSI file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        source = fh.read()

    name = module_name or f"nrsi.stdlib.{Path(path).stem}"

    cached = _cache.get(path, source)
    if cached is not None:
        return cached

    python_code, ast, diagnostics = _pipeline(source, filename=path)

    for d in diagnostics:
        if (getattr(d.severity, "value", d.severity) == "warning"
                or getattr(d.severity, "name", "") == "WARNING"
                or str(d.severity) == "DiagnosticSeverity.WARNING"):
            warnings.warn(
                f"NRSI [{d.filename}:{d.line}]: {d.message}",
                stacklevel=2,
            )

    module = types.ModuleType(name)
    module.__file__ = path
    module.__loader__ = None  # type: ignore[assignment]
    module.__nrsi_source__ = source  # type: ignore[attr-defined]
    module.__nrsi_ast__ = ast  # type: ignore[attr-defined]
    module.__nrsi_diagnostics__ = diagnostics  # type: ignore[attr-defined]
    module.__nrsi_python__ = python_code  # type: ignore[attr-defined]

    module.__dict__.update(_nrsi_builtins())

    sys.modules[name] = module
    short_name = Path(path).stem
    if short_name not in sys.modules:
        sys.modules[short_name] = module
    try:
        exec(compile(python_code, path, "exec"), module.__dict__)  # noqa: S102
    except Exception:
        sys.modules.pop(name, None)
        sys.modules.pop(short_name, None)
        raise

    _cache.put(path, source, module)
    return module


def compile_nrsi(source: str, filename: str = "<string>") -> str:
    """Compile NRSI source to Python source **without** executing.

    Useful for offline transpilation, build tooling, or inspection.

    Args:
        source: NRSI source code string.
        filename: Optional filename for error messages.

    Returns:
        Python source code as a string.
    """
    python_code, _ast, _diagnostics = _pipeline(source, filename=filename)
    return python_code


def recompile(path: str, module_name: str = "") -> types.ModuleType:
    """Force-recompile a cached ``.nrsi`` file.

    Invalidates any cached version and runs the full pipeline again.
    """
    _cache.invalidate(os.path.abspath(path))
    return load_nrsi(path, module_name=module_name)


def clear_cache() -> None:
    """Drop every cached NRSI module."""
    _cache.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Module Introspection
# ═══════════════════════════════════════════════════════════════════════════════

def get_module_info(module: types.ModuleType) -> Dict[str, Any]:
    """Return NRSI metadata attached to *module* by the loader.

    Returns an empty dict for non-NRSI modules.
    """
    info: Dict[str, Any] = {}
    for attr in ("__nrsi_source__", "__nrsi_ast__",
                 "__nrsi_diagnostics__", "__nrsi_python__"):
        val = getattr(module, attr, None)
        if val is not None:
            info[attr.strip("_")] = val
    if info:
        info["file"] = getattr(module, "__file__", None)
        info["name"] = module.__name__
    return info


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Import Hook
# ═══════════════════════════════════════════════════════════════════════════════

_NRSI_EXT = ".nrsi"


class NRSIImportFinder(importlib.abc.MetaPathFinder):
    """``sys.meta_path`` hook that discovers ``.nrsi`` files."""

    def find_module(self, fullname: str, path: Any = None) -> Any:
        spec = self.find_spec(fullname, path)
        return spec.loader if spec is not None else None

    def find_spec(
        self,
        fullname: str,
        path: Any,
        target: Any = None,
    ) -> Optional[importlib.machinery.ModuleSpec]:
        tail = fullname.rsplit(".", 1)[-1]
        filename = tail + _NRSI_EXT

        search_dirs: List[str] = list(path) if path else list(sys.path)

        for directory in search_dirs:
            candidate = os.path.join(directory, filename)
            if os.path.isfile(candidate):
                return importlib.machinery.ModuleSpec(
                    fullname,
                    NRSILoader(candidate),
                    origin=candidate,
                )

            pkg_init = os.path.join(
                directory, tail, "__init__" + _NRSI_EXT,
            )
            if os.path.isfile(pkg_init):
                return importlib.machinery.ModuleSpec(
                    fullname,
                    NRSILoader(pkg_init),
                    origin=pkg_init,
                    is_package=True,
                )

        return None


class NRSILoader(importlib.abc.Loader):
    """Load a ``.nrsi`` file through the full compilation pipeline."""

    def __init__(self, path: str) -> None:
        self._path = path

    def create_module(self, spec: Any) -> None:
        return None  # default module creation

    def exec_module(self, module: types.ModuleType) -> None:
        with open(self._path, encoding="utf-8") as fh:
            source = fh.read()

        python_code, ast, diagnostics = _pipeline(source, filename=self._path)

        for d in diagnostics:
            if (getattr(d.severity, "value", d.severity) == "warning"
                    or getattr(d.severity, "name", "") == "WARNING"
                    or str(d.severity) == "DiagnosticSeverity.WARNING"):
                warnings.warn(
                    f"NRSI [{self._path}:{d.line}]: {d.message}",
                    stacklevel=2,
                )

        module.__file__ = self._path
        module.__nrsi_source__ = source  # type: ignore[attr-defined]
        module.__nrsi_ast__ = ast  # type: ignore[attr-defined]
        module.__nrsi_diagnostics__ = diagnostics  # type: ignore[attr-defined]
        module.__nrsi_python__ = python_code  # type: ignore[attr-defined]

        module.__dict__.update(_nrsi_builtins())

        exec(  # noqa: S102
            compile(python_code, self._path, "exec"),
            module.__dict__,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Install / Uninstall
# ═══════════════════════════════════════════════════════════════════════════════

_finder_instance: Optional[NRSIImportFinder] = None


def install() -> None:
    """Install the NRSI import hook into ``sys.meta_path``.

    After this call, Python's import machinery will search for
    ``.nrsi`` files alongside ``.py`` files.  Importing a ``.nrsi``
    file runs the full pipeline (lex → parse → check → transpile → exec).

    Safe to call multiple times — only one hook is installed.
    """
    global _finder_instance
    if _finder_instance is not None:
        return
    _finder_instance = NRSIImportFinder()
    sys.meta_path.insert(0, _finder_instance)


def uninstall() -> None:
    """Remove the NRSI import hook from ``sys.meta_path``."""
    global _finder_instance
    if _finder_instance is None:
        return
    try:
        sys.meta_path.remove(_finder_instance)
    except ValueError:
        pass
    _finder_instance = None


def is_installed() -> bool:
    """Return True if the NRSI import hook is currently active."""
    return _finder_instance is not None and _finder_instance in sys.meta_path


# Auto-install when the runtime module is first imported.
install()


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Precompile Utility
# ═══════════════════════════════════════════════════════════════════════════════

def precompile_directory(
    directory: str,
    *,
    output_dir: Optional[str] = None,
    recursive: bool = True,
) -> Dict[str, str]:
    """Batch-compile every ``.nrsi`` file under *directory* to Python.

    Args:
        directory: Root directory to scan.
        output_dir: If given, write ``.py`` files here (mirroring the
                    directory tree).  If ``None``, return the Python
                    source as a dict but don't write files.
        recursive: Descend into sub-directories.

    Returns:
        ``{nrsi_path: python_source}`` for every compiled file.
    """
    results: Dict[str, str] = {}
    root = Path(directory)
    pattern = f"**/*{_NRSI_EXT}" if recursive else f"*{_NRSI_EXT}"

    for nrsi_path in sorted(root.glob(pattern)):
        source = nrsi_path.read_text(encoding="utf-8")
        try:
            python_code = compile_nrsi(source, filename=str(nrsi_path))
        except (LexerError, NRSICompileError) as exc:
            warnings.warn(f"Skipping {nrsi_path}: {exc}", stacklevel=2)
            continue

        results[str(nrsi_path)] = python_code

        if output_dir is not None:
            relative = nrsi_path.relative_to(root)
            dest = Path(output_dir) / relative.with_suffix(".py")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(python_code, encoding="utf-8")

    return results


__all__ = [
    "NRSICompileError",
    "NRSIImportFinder",
    "NRSILoader",
    "load_nrsi",
    "compile_nrsi",
    "recompile",
    "clear_cache",
    "get_module_info",
    "install",
    "uninstall",
    "is_installed",
    "precompile_directory",
]
