"""NRSI Language Server (LSP) — MVP diagnostics, hover, go-to-definition, completion.

Run stdio server::

    python -m nrsi.lang.lsp

Or via CLI::

    nrsic lsp
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from lsprotocol import types as lsp
from pygls.lsp.server import LanguageServer
from pygls.uris import to_fs_path

from nrsi.lang.lexer import KEYWORDS, Lexer, LexerError
from nrsi.lang.parser import Identifier, Module, ParseError, Parser
from nrsi.lang.type_checker import (
    Diagnostic,
    DiagnosticSeverity,
    Symbol,
    TypeChecker,
)

logger = logging.getLogger("nrsi.lang.lsp")

NRSI_KEYWORDS_SORTED: Tuple[str, ...] = tuple(sorted(KEYWORDS.keys()))

_SEVERITY_MAP = {
    DiagnosticSeverity.ERROR: lsp.DiagnosticSeverity.Error,
    DiagnosticSeverity.WARNING: lsp.DiagnosticSeverity.Warning,
    DiagnosticSeverity.INFO: lsp.DiagnosticSeverity.Information,
    DiagnosticSeverity.HINT: lsp.DiagnosticSeverity.Hint,
}


@dataclass
class CheckSnapshot:
    """Result of lex / parse / type-check for one document revision."""

    lsp_diagnostics: List[lsp.Diagnostic] = field(default_factory=list)
    checker: Optional[TypeChecker] = None


def _nrsi_to_lsp_line_col(line_1: int, col_1: int) -> Tuple[int, int]:
    return max(0, line_1 - 1), max(0, col_1 - 1)


def _diagnostic_to_lsp(d: Diagnostic) -> lsp.Diagnostic:
    line_idx, col_idx = _nrsi_to_lsp_line_col(d.line, d.column)
    start = lsp.Position(line=line_idx, character=col_idx)
    end = lsp.Position(line=line_idx, character=col_idx + 1)
    return lsp.Diagnostic(
        range=lsp.Range(start=start, end=end),
        message=d.message,
        severity=_SEVERITY_MAP.get(d.severity, lsp.DiagnosticSeverity.Error),
        source="nrsi",
        code=d.rule,
    )


def _lexer_error_to_lsp(err: LexerError) -> lsp.Diagnostic:
    line_idx, col_idx = _nrsi_to_lsp_line_col(err.line, err.column)
    start = lsp.Position(line=line_idx, character=col_idx)
    end = lsp.Position(line=line_idx, character=col_idx + 1)
    return lsp.Diagnostic(
        range=lsp.Range(start=start, end=end),
        message=str(err).split(": ", 1)[-1] if ": " in str(err) else str(err),
        severity=lsp.DiagnosticSeverity.Error,
        source="nrsi",
        code="lexer",
    )


def _parse_error_to_lsp(err: ParseError) -> lsp.Diagnostic:
    tok = err.token
    if tok is not None:
        line_idx, col_idx = _nrsi_to_lsp_line_col(tok.line, tok.column)
    else:
        line_idx, col_idx = 0, 0
    start = lsp.Position(line=line_idx, character=col_idx)
    end = lsp.Position(line=line_idx, character=col_idx + 1)
    return lsp.Diagnostic(
        range=lsp.Range(start=start, end=end),
        message=str(err),
        severity=lsp.DiagnosticSeverity.Error,
        source="nrsi",
        code="parser",
    )


def analyze_document(uri: str, source: str, filename: Optional[str] = None) -> CheckSnapshot:
    """Lex, parse, and type-check *source*; return LSP diagnostics and checker state."""
    path = filename or to_fs_path(uri)
    snap = CheckSnapshot()

    try:
        tokens = Lexer(source, filename=path).tokenize()
    except LexerError as e:
        snap.lsp_diagnostics.append(_lexer_error_to_lsp(e))
        return snap

    try:
        tree = Parser(tokens, filename=path).parse()
    except ParseError as e:
        snap.lsp_diagnostics.append(_parse_error_to_lsp(e))
        return snap

    if not isinstance(tree, Module):
        snap.lsp_diagnostics.append(
            lsp.Diagnostic(
                range=lsp.Range(
                    start=lsp.Position(line=0, character=0),
                    end=lsp.Position(line=0, character=1),
                ),
                message="Internal error: parser did not return a Module",
                severity=lsp.DiagnosticSeverity.Error,
                source="nrsi",
                code="internal",
            )
        )
        return snap

    checker = TypeChecker()
    for d in checker.check(tree):
        snap.lsp_diagnostics.append(_diagnostic_to_lsp(d))
    snap.checker = checker
    return snap


def _snapshots(ls: LanguageServer) -> Dict[str, CheckSnapshot]:
    d = getattr(ls, "_nrsi_snapshots", None)
    if d is None:
        d = {}
        setattr(ls, "_nrsi_snapshots", d)
    return d


def _publish(ls: LanguageServer, uri: str, source: str) -> None:
    snap = analyze_document(uri, source)
    _snapshots(ls)[uri] = snap
    ls.text_document_publish_diagnostics(
        lsp.PublishDiagnosticsParams(uri=uri, diagnostics=snap.lsp_diagnostics)
    )


def _cursor_on_identifier(
    ident: Identifier, lsp_line: int, lsp_char: int
) -> bool:
    if ident.line <= 0:
        return False
    if lsp_line != ident.line - 1:
        return False
    start = ident.column - 1 if ident.column > 0 else 0
    end = start + len(ident.name)
    return start <= lsp_char < end


def _binding_at_position(
    checker: TypeChecker, lsp_line: int, lsp_char: int
) -> Optional[Tuple[Identifier, Optional[Symbol]]]:
    for ident, sym in checker.identifier_bindings:
        if _cursor_on_identifier(ident, lsp_line, lsp_char):
            return (ident, sym)
    return None


def _completion_prefix(lines: List[str], lsp_line: int, lsp_char: int) -> str:
    if lsp_line < 0 or lsp_line >= len(lines):
        return ""
    line = lines[lsp_line]
    if lsp_char < 0:
        return ""
    if lsp_char > len(line):
        lsp_char = len(line)
    i = lsp_char - 1
    while i >= 0 and (line[i].isalnum() or line[i] == "_"):
        i -= 1
    return line[i + 1 : lsp_char]


server = LanguageServer("nrsi-language-server", "0.1.0")


@server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: LanguageServer, params: lsp.DidOpenTextDocumentParams) -> None:
    doc = params.text_document
    _publish(ls, doc.uri, doc.text)


@server.feature(lsp.TEXT_DOCUMENT_DID_SAVE)
def did_save(ls: LanguageServer, params: lsp.DidSaveTextDocumentParams) -> None:
    doc = ls.workspace.get_text_document(params.text_document.uri)
    _publish(ls, params.text_document.uri, doc.source)


@server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: LanguageServer, params: lsp.DidChangeTextDocumentParams) -> None:
    doc = ls.workspace.get_text_document(params.text_document.uri)
    _publish(ls, params.text_document.uri, doc.source)


@server.feature(lsp.TEXT_DOCUMENT_HOVER)
def hover(ls: LanguageServer, params: lsp.HoverParams) -> Optional[lsp.Hover]:
    snap = _snapshots(ls).get(params.text_document.uri)
    if snap is None or snap.checker is None:
        return None
    pos = params.position
    hit = _binding_at_position(snap.checker, pos.line, pos.character)
    if hit is None:
        return None
    ident, sym = hit
    if sym is None:
        return lsp.Hover(contents=lsp.MarkupContent(kind=lsp.MarkupKind.PlainText, value=f"`{ident.name}` (unresolved)"))
    parts = [
        f"**{sym.name}**",
        f"- type: `{sym.nrsi_type.display()}`",
        f"- kind: `{sym.kind}`",
    ]
    if sym.mutable:
        parts.append("- mutable: `true`")
    md = "\n".join(parts)
    return lsp.Hover(contents=lsp.MarkupContent(kind=lsp.MarkupKind.Markdown, value=md))


@server.feature(lsp.TEXT_DOCUMENT_DEFINITION)
def definition(ls: LanguageServer, params: lsp.DefinitionParams) -> Optional[lsp.Location]:
    snap = _snapshots(ls).get(params.text_document.uri)
    if snap is None or snap.checker is None:
        return None
    pos = params.position
    hit = _binding_at_position(snap.checker, pos.line, pos.character)
    if hit is None:
        return None
    _ident, sym = hit
    if sym is None or sym.declared_line <= 0:
        return None
    uri = params.text_document.uri
    line0 = sym.declared_line - 1
    start = lsp.Position(line=line0, character=0)
    end = lsp.Position(line=line0, character=0)
    return lsp.Location(uri=uri, range=lsp.Range(start=start, end=end))


@server.feature(
    lsp.TEXT_DOCUMENT_COMPLETION,
    lsp.CompletionOptions(trigger_characters=[".", "_"]),
)
def completion(ls: LanguageServer, params: lsp.CompletionParams) -> lsp.CompletionList:
    snap = _snapshots(ls).get(params.text_document.uri)
    doc = ls.workspace.get_text_document(params.text_document.uri)
    lines = doc.source.splitlines(keepends=False)
    prefix = _completion_prefix(lines, params.position.line, params.position.character)

    items: List[lsp.CompletionItem] = []

    for kw in NRSI_KEYWORDS_SORTED:
        if not prefix or kw.startswith(prefix):
            items.append(
                lsp.CompletionItem(
                    label=kw,
                    kind=lsp.CompletionItemKind.Keyword,
                    insert_text=kw,
                )
            )

    symbol_names: List[str] = []
    if snap and snap.checker is not None:
        symbol_names = snap.checker.global_completion_symbols()
    for name in symbol_names:
        if not prefix or name.startswith(prefix):
            items.append(
                lsp.CompletionItem(
                    label=name,
                    kind=lsp.CompletionItemKind.Variable,
                    insert_text=name,
                )
            )

    return lsp.CompletionList(is_incomplete=False, items=items)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    server.start_io()


if __name__ == "__main__":
    main()
