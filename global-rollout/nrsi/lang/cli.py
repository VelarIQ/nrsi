"""NRSI Compiler CLI — ``nrsic``

Command-line interface for the NRSI language toolchain.

Usage::

    nrsic compile file.nrsi                 # Python to stdout
    nrsic compile file.nrsi -o out.py       # Python to file
    nrsic compile dir/ --target python      # batch compile
    nrsic compile file.nrsi --target swift  # Swift output
    nrsic compile file.nrsi --target kotlin # Kotlin output
    nrsic check file.nrsi                   # type-check only
    nrsic check dir/                        # batch check
    nrsic run file.nrsi                     # lex, check, transpile, execute
    nrsic run --module mymod               # run by module name (path + stdlib)
    nrsic repl                              # interactive shell
    nrsic lsp                               # Language Server Protocol (stdio)
    nrsic targets                           # list available targets
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence


def _get_lexer():
    from nrsi.lang.lexer import Lexer
    return Lexer


def _get_parser():
    from nrsi.lang.parser import Parser
    return Parser


def _get_checker():
    from nrsi.lang.type_checker import TypeChecker, DiagnosticSeverity
    return TypeChecker, DiagnosticSeverity


def _get_transpiler(target: str):
    from nrsi.lang.targets import get_transpiler
    return get_transpiler(target)


def _nrsi_package_stdlib_dir() -> str:
    """Directory containing bundled ``nrsi/stdlib`` ``.nrsi`` files."""
    import nrsi as _nrsi

    return str(Path(_nrsi.__file__).resolve().parent / "stdlib")


def _resolve_nrsi_module_path(name: str) -> Optional[str]:
    """Resolve a module name (e.g. ``nrs_core`` or ``pkg.mod``) to a ``.nrsi`` path."""
    tail = name.rsplit(".", 1)[-1]
    filename = tail + ".nrsi"
    search_dirs = [_nrsi_package_stdlib_dir()] + [p for p in sys.path if p]
    for directory in search_dirs:
        candidate = os.path.join(directory, filename)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
        pkg_init = os.path.join(directory, tail, "__init__.nrsi")
        if os.path.isfile(pkg_init):
            return os.path.abspath(pkg_init)
    return None


def _collect_nrsi_files(paths: List[str]) -> List[str]:
    """Expand paths into a flat list of .nrsi files."""
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "**", "*.nrsi"), recursive=True)))
            if not files:
                files.extend(sorted(glob.glob(os.path.join(p, "*.nrsi"))))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"nrsic: error: {p}: no such file or directory", file=sys.stderr)
            sys.exit(1)
    return files


def _compile_file(path: str, target: str) -> str:
    """Lex, parse, type-check, and transpile a single .nrsi file."""
    Lexer = _get_lexer()
    Parser = _get_parser()
    TypeChecker, DiagnosticSeverity = _get_checker()
    transpiler_cls = _get_transpiler(target)

    source = Path(path).read_text(encoding="utf-8")

    tokens = Lexer(source, filename=path).tokenize()
    tree = Parser(tokens, filename=path).parse()

    checker = TypeChecker()
    diagnostics = checker.check(tree)

    errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
    warnings = [d for d in diagnostics if d.severity == DiagnosticSeverity.WARNING]

    for w in warnings:
        print(f"{path}:{w.line}:{w.column}: warning: {w.message} [{w.rule}]", file=sys.stderr)
    for e in errors:
        print(f"{path}:{e.line}:{e.column}: error: {e.message} [{e.rule}]", file=sys.stderr)

    if errors:
        raise SystemExit(1)

    t = transpiler_cls()
    return t.transpile(tree)


def _check_file(path: str) -> int:
    """Lex, parse, and type-check a single .nrsi file. Returns error count."""
    Lexer = _get_lexer()
    Parser = _get_parser()
    TypeChecker, DiagnosticSeverity = _get_checker()

    source = Path(path).read_text(encoding="utf-8")

    try:
        tokens = Lexer(source, filename=path).tokenize()
    except Exception as e:
        print(f"{path}: lex error: {e}", file=sys.stderr)
        return 1

    try:
        tree = Parser(tokens, filename=path).parse()
    except Exception as e:
        print(f"{path}: parse error: {e}", file=sys.stderr)
        return 1

    checker = TypeChecker()
    diagnostics = checker.check(tree)

    error_count = 0
    for d in diagnostics:
        severity = "error" if d.severity == DiagnosticSeverity.ERROR else "warning"
        print(f"{path}:{d.line}:{d.column}: {severity}: {d.message} [{d.rule}]", file=sys.stderr)
        if d.severity == DiagnosticSeverity.ERROR:
            error_count += 1

    return error_count


def cmd_compile(args: argparse.Namespace) -> None:
    """Handle the ``compile`` subcommand."""
    files = _collect_nrsi_files(args.files)
    if not files:
        print("nrsic: error: no .nrsi files found", file=sys.stderr)
        sys.exit(1)

    target = args.target or "python"
    ext_map = {"python": ".py", "swift": ".swift", "kotlin": ".kt"}
    ext = ext_map.get(target, ".out")

    for path in files:
        code = _compile_file(path, target)

        if args.output == "-" or (args.output is None and len(files) == 1):
            if args.output is None and len(files) == 1:
                sys.stdout.write(code)
            else:
                sys.stdout.write(code)
        elif args.output:
            Path(args.output).write_text(code, encoding="utf-8")
            print(f"nrsic: {path} -> {args.output}", file=sys.stderr)
        else:
            out_path = Path(path).with_suffix(ext)
            out_path.write_text(code, encoding="utf-8")
            print(f"nrsic: {path} -> {out_path}", file=sys.stderr)


def cmd_check(args: argparse.Namespace) -> None:
    """Handle the ``check`` subcommand."""
    files = _collect_nrsi_files(args.files)
    if not files:
        print("nrsic: error: no .nrsi files found", file=sys.stderr)
        sys.exit(1)

    total_errors = 0
    for path in files:
        errors = _check_file(path)
        total_errors += errors
        status = "OK" if errors == 0 else f"{errors} error(s)"
        print(f"  {status}  {path}", file=sys.stderr)

    print(f"\nnrsic: checked {len(files)} file(s), {total_errors} error(s)", file=sys.stderr)
    if total_errors > 0:
        sys.exit(1)


def cmd_targets(args: argparse.Namespace) -> None:
    """Handle the ``targets`` subcommand."""
    from nrsi.lang.targets import REGISTRY
    print("Available transpiler targets:\n")
    for name, cls in sorted(REGISTRY.items()):
        doc = (cls.__doc__ or "").strip().split("\n")[0]
        print(f"  {name:12s}  {doc}")


def cmd_run(args: argparse.Namespace) -> None:
    """Handle the ``run`` subcommand."""
    if args.module:
        path = _resolve_nrsi_module_path(args.file)
        if not path:
            print(f"nrsic: error: module {args.file!r}: not found", file=sys.stderr)
            sys.exit(1)
    else:
        path = args.file
        if not os.path.isfile(path):
            print(f"nrsic: error: {path}: no such file", file=sys.stderr)
            sys.exit(1)

    old_argv = sys.argv
    sys.argv = [path] + (args.args or [])

    try:
        code = _compile_file(path, "python")

        try:
            from nrsi.lang.builtins import NRSI_BUILTINS

            ns = dict(NRSI_BUILTINS)
        except ImportError:
            ns = {}

        ns["__name__"] = "__main__"
        ns["__file__"] = os.path.abspath(path)

        exec(compile(code, path, "exec"), ns)
    except SystemExit:
        raise
    except Exception as e:
        print(f"nrsic: runtime error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        sys.argv = old_argv


def cmd_repl(args: argparse.Namespace) -> None:
    """Handle the ``repl`` subcommand — interactive NRSI shell."""
    Lexer = _get_lexer()
    Parser = _get_parser()

    print("NRSI interactive shell (type 'exit' or Ctrl-D to quit)")
    print()

    ns: dict = {"__name__": "__nrsi_repl__"}
    try:
        from nrsi.lang.builtins import NRSI_BUILTINS

        ns.update(NRSI_BUILTINS)
    except ImportError:
        pass

    import readline  # noqa: F401 — enable arrow keys / history

    while True:
        try:
            line = input("nrsi> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        line = line.strip()
        if not line or line == "exit":
            if line == "exit":
                break
            continue

        while line.count("{") > line.count("}"):
            try:
                cont = input("  ... ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            line += "\n" + cont

        source = f"module __repl__\n{line}"

        try:
            TypeChecker, DiagnosticSeverity = _get_checker()
            transpiler_cls = _get_transpiler("python")
            tokens = Lexer(source, filename="<repl>").tokenize()
            tree = Parser(tokens, filename="<repl>").parse()

            checker = TypeChecker()
            diagnostics = checker.check(tree)
            for d in diagnostics:
                if d.severity == DiagnosticSeverity.ERROR:
                    print(f"  error: {d.message} [{d.rule}]", file=sys.stderr)
                elif d.severity == DiagnosticSeverity.WARNING:
                    print(f"  warning: {d.message} [{d.rule}]", file=sys.stderr)

            t = transpiler_cls()
            code = t.transpile(tree)

            exec(compile(code, "<repl>", "exec"), ns)
        except SystemExit:
            raise
        except Exception as e:
            print(f"Error: {e}")


def _normalize_source_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def cmd_lsp(args: argparse.Namespace) -> None:
    """Handle the ``lsp`` subcommand — run the NRSI language server on stdio."""
    from nrsi.lang.lsp import main as lsp_main

    lsp_main()


def cmd_fmt(args: argparse.Namespace) -> None:
    """Handle the ``fmt`` subcommand — parse, pretty-print, write or ``--check``."""
    from nrsi.lang.formatter import Formatter

    files = _collect_nrsi_files(args.files)
    if not files:
        print("nrsic: error: no .nrsi files found", file=sys.stderr)
        sys.exit(1)

    Lexer = _get_lexer()
    Parser = _get_parser()
    formatter = Formatter()
    exit_code = 0

    for path in files:
        raw = Path(path).read_text(encoding="utf-8")
        try:
            tokens = Lexer(raw, filename=path).tokenize()
            tree = Parser(tokens, filename=path).parse()
        except Exception as e:
            print(f"{path}: error: {e}", file=sys.stderr)
            exit_code = 1
            continue

        formatted = formatter.format(tree)
        orig_n = _normalize_source_newlines(raw)
        fmt_n = _normalize_source_newlines(formatted)

        if args.check:
            if orig_n != fmt_n:
                print(f"{path}: would reformat", file=sys.stderr)
                exit_code = 1
        else:
            Path(path).write_text(formatted, encoding="utf-8")
            print(f"nrsic: formatted {path}", file=sys.stderr)

    if exit_code:
        sys.exit(exit_code)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for nrsic."""
    top = argparse.ArgumentParser(
        prog="nrsic",
        description="NRSI Language Compiler — compile, check, and transpile .nrsi files",
    )
    sub = top.add_subparsers(dest="command", required=True)

    # --- compile ---
    p_compile = sub.add_parser("compile", help="Compile .nrsi files to a target language")
    p_compile.add_argument("files", nargs="+", help=".nrsi files or directories")
    p_compile.add_argument("-o", "--output", default=None, help="Output path (- for stdout)")
    p_compile.add_argument(
        "-t", "--target", default="python",
        help="Transpiler target: python, swift, kotlin (default: python)",
    )
    p_compile.set_defaults(func=cmd_compile)

    # --- check ---
    p_check = sub.add_parser("check", help="Type-check .nrsi files without compiling")
    p_check.add_argument("files", nargs="+", help=".nrsi files or directories")
    p_check.set_defaults(func=cmd_check)

    # --- targets ---
    p_targets = sub.add_parser("targets", help="List available transpiler targets")
    p_targets.set_defaults(func=cmd_targets)

    # --- run ---
    p_run = sub.add_parser("run", help="Run a .nrsi file directly")
    p_run.add_argument(
        "--module",
        "-m",
        action="store_true",
        help="Treat the first argument as a module name (search sys.path and nrsi/stdlib)",
    )
    p_run.add_argument("file", help=".nrsi file to run (or module name with -m/--module)")
    p_run.add_argument("args", nargs="*", help="Arguments to pass to the program")
    p_run.set_defaults(func=cmd_run)

    # --- repl ---
    p_repl = sub.add_parser("repl", help="Start an interactive NRSI session")
    p_repl.set_defaults(func=cmd_repl)

    # --- lsp ---
    p_lsp = sub.add_parser(
        "lsp",
        help="Run the NRSI Language Server (LSP over stdio; for editors)",
    )
    p_lsp.set_defaults(func=cmd_lsp)

    # --- fmt ---
    p_fmt = sub.add_parser("fmt", help="Format .nrsi sources (parse → pretty-print)")
    p_fmt.add_argument("files", nargs="+", help=".nrsi files or directories")
    p_fmt.add_argument(
        "--check",
        action="store_true",
        help="Exit with status 1 if any file would change (do not write)",
    )
    p_fmt.set_defaults(func=cmd_fmt)

    return top


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for ``nrsic`` CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
