"""
NRSI Lexer — tokenizer for the NRSI programming language.

Reads .nrsi source text and produces a flat stream of ``Token`` objects.
Pure-Python, stdlib-only, zero external dependencies.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional


# ---------------------------------------------------------------------------
# Token taxonomy
# ---------------------------------------------------------------------------

class TokenType(Enum):
    """Every distinct token the NRSI grammar can produce."""

    # ── Literals ──────────────────────────────────────────────────────────
    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
    BOOLEAN = auto()
    NONE = auto()
    IDENTIFIER = auto()

    # ── Trust-level keywords ──────────────────────────────────────────────
    KW_TRUST = auto()
    KW_RAW = auto()
    KW_VALIDATED = auto()
    KW_TRUSTED = auto()
    KW_CERTIFIED = auto()

    # ── Gate keywords ─────────────────────────────────────────────────────
    KW_GATE = auto()
    KW_REQUIRE = auto()
    KW_VALIDATE = auto()

    # ── Architecture keywords ─────────────────────────────────────────────
    KW_LOBE = auto()
    KW_PROCESSOR = auto()

    # ── Normative keywords ────────────────────────────────────────────────
    KW_NORM = auto()
    KW_OBLIGATION = auto()
    KW_PERMISSION = auto()
    KW_PROHIBITION = auto()
    KW_EXEMPTION = auto()

    # ── Epistemic base keywords ───────────────────────────────────────────
    KW_BELIEF = auto()
    KW_BASE = auto()
    KW_AXIOM = auto()
    KW_TIER = auto()

    # ── Type-system / control-flow keywords ───────────────────────────────
    KW_TYPE = auto()
    KW_STRUCT = auto()
    KW_ENUM = auto()
    KW_FN = auto()
    KW_RETURN = auto()
    KW_IF = auto()
    KW_ELSE = auto()
    KW_MATCH = auto()
    KW_FOR = auto()
    KW_IN = auto()

    # ── Variable binding keywords ─────────────────────────────────────────
    KW_LET = auto()
    KW_MUT = auto()
    KW_CONST = auto()

    # ── Module keywords ───────────────────────────────────────────────────
    KW_IMPORT = auto()
    KW_FROM = auto()
    KW_AS = auto()
    KW_EXPORT = auto()
    KW_MODULE = auto()
    KW_USE = auto()

    # ── Boolean / null literal keywords ───────────────────────────────────
    KW_TRUE = auto()
    KW_FALSE = auto()
    KW_NONE = auto()

    # ── Logical operator keywords ─────────────────────────────────────────
    KW_AND = auto()
    KW_OR = auto()
    KW_NOT = auto()

    # ── Epistemic-mode keywords ───────────────────────────────────────────
    KW_EPISTEMIC = auto()
    KW_TEMPORAL = auto()
    KW_CAUSAL = auto()
    KW_DEDUCTIVE = auto()
    KW_COMPUTATIONAL = auto()
    KW_OBSERVATIONAL = auto()
    KW_ANALOGICAL = auto()
    KW_SPECULATIVE = auto()
    KW_INDUCTIVE = auto()
    KW_CREATIVE = auto()

    # ── Temporal-tier keywords ────────────────────────────────────────────
    KW_ETERNAL = auto()
    KW_STABLE = auto()
    KW_CURRENT = auto()
    KW_EPHEMERAL = auto()
    KW_HISTORICAL = auto()

    # ── Domain / scope keywords ───────────────────────────────────────────
    KW_DOMAIN = auto()
    KW_SCOPE = auto()
    KW_GLOBAL = auto()
    KW_SESSION = auto()
    KW_QUERY = auto()

    # ── Policy-action keywords ────────────────────────────────────────────
    KW_CONFIDENCE = auto()
    KW_PRIORITY = auto()
    KW_ACTION = auto()
    KW_BLOCK = auto()
    KW_WARN = auto()
    KW_LOG = auto()
    KW_PERMIT = auto()

    # ── Attention keywords ────────────────────────────────────────────────
    KW_ATTENTION = auto()
    KW_FOCUS = auto()
    KW_SALIENCE = auto()
    KW_BUDGET = auto()

    # ── BDI keywords ──────────────────────────────────────────────────────
    KW_BELIEF_STATE = auto()
    KW_DESIRE = auto()
    KW_INTENTION = auto()
    KW_DELIBERATE = auto()

    # ── Affect keywords ───────────────────────────────────────────────────
    KW_AFFECT = auto()
    KW_MOOD = auto()
    KW_AROUSAL = auto()
    KW_VALENCE = auto()

    # ── Speech-act keywords ───────────────────────────────────────────────
    KW_SPEECH_ACT = auto()
    KW_ASSERT = auto()
    KW_ASK = auto()
    KW_COMMAND = auto()
    KW_PROMISE = auto()
    KW_WARN_ACT = auto()

    # ── Explanation keywords ──────────────────────────────────────────────
    KW_EXPLAIN = auto()
    KW_BECAUSE = auto()
    KW_DESPITE = auto()
    KW_IMPLIES = auto()

    # ── Concept-ontology keywords ─────────────────────────────────────────
    KW_CONCEPT = auto()
    KW_SUBCONCEPT_OF = auto()
    KW_INSTANCE_OF = auto()
    KW_PART_OF = auto()

    # ── Multimodal keywords ───────────────────────────────────────────────
    KW_BIND = auto()
    KW_MODALITY = auto()
    KW_FUSE = auto()

    # ── Cognitive-primitive keywords ─────────────────────────────────────
    KW_COMPOSE = auto()
    KW_PERSIST = auto()
    KW_SEMANTIC_DISTANCE = auto()
    KW_DECOMPOSE = auto()
    KW_INTENT_MATCH = auto()

    # ── Extended control-flow keywords ───────────────────────────────────
    KW_WHILE = auto()
    KW_TRY = auto()
    KW_CATCH = auto()
    KW_FINALLY = auto()
    KW_BREAK = auto()
    KW_CONTINUE = auto()
    KW_PASS = auto()
    KW_RAISE = auto()
    KW_DEL = auto()

    # ── Class / method keywords ──────────────────────────────────────────
    KW_CLASS = auto()
    KW_SELF = auto()
    KW_IS = auto()
    KW_SUPER = auto()

    # ── Extended keywords (general-purpose language) ───────────────────
    KW_WITH = auto()
    KW_ASYNC = auto()
    KW_AWAIT = auto()
    KW_YIELD = auto()
    KW_LAMBDA = auto()
    KW_NONLOCAL = auto()
    KW_CASE = auto()

    # ── Extended operators ─────────────────────────────────────────────
    POWER = auto()  # **
    FLOOR_DIV = auto()  # ~~  (NRSI uses ~~ for floor div since // is comment)
    LEFT_SHIFT = auto()  # <<
    RIGHT_SHIFT = auto()  # >>
    CARET = auto()  # ^
    TILDE = auto()  # ~
    POWER_ASSIGN = auto()  # **=
    FLOOR_DIV_ASSIGN = auto()  # ~~=
    AMPERSAND_ASSIGN = auto()  # &=
    PIPE_ASSIGN = auto()  # |=
    CARET_ASSIGN = auto()  # ^=
    LEFT_SHIFT_ASSIGN = auto()  # <<=
    RIGHT_SHIFT_ASSIGN = auto()  # >>=

    # ── Operators / punctuation ───────────────────────────────────────────
    COLON = auto()
    DOUBLE_COLON = auto()
    SEMICOLON = auto()
    LBRACE = auto()
    RBRACE = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LANGLE = auto()
    RANGLE = auto()
    ASSIGN = auto()
    EQUAL = auto()
    NOT_EQUAL = auto()
    GTE = auto()
    LTE = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    LOGICAL_AND = auto()
    LOGICAL_OR = auto()
    BANG = auto()
    ARROW = auto()
    FAT_ARROW = auto()
    DOT = auto()
    RANGE = auto()
    SPREAD = auto()
    COMMA = auto()
    AT = auto()
    HASH = auto()
    PIPE = auto()
    AMPERSAND = auto()
    PLUS_ASSIGN = auto()
    MINUS_ASSIGN = auto()
    STAR_ASSIGN = auto()
    SLASH_ASSIGN = auto()
    PERCENT_ASSIGN = auto()

    # ── Structural / special ──────────────────────────────────────────────
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    EOF = auto()
    DOC_COMMENT = auto()
    COMMENT = auto()


# ---------------------------------------------------------------------------
# Keyword lookup (source text → TokenType)
# ---------------------------------------------------------------------------

KEYWORDS: Dict[str, TokenType] = {
    "trust": TokenType.KW_TRUST,
    "raw": TokenType.KW_RAW,
    "validated": TokenType.KW_VALIDATED,
    "trusted": TokenType.KW_TRUSTED,
    "certified": TokenType.KW_CERTIFIED,
    "gate": TokenType.KW_GATE,
    "require": TokenType.KW_REQUIRE,
    "validate": TokenType.KW_VALIDATE,
    "lobe": TokenType.KW_LOBE,
    "processor": TokenType.KW_PROCESSOR,
    "norm": TokenType.KW_NORM,
    "obligation": TokenType.KW_OBLIGATION,
    "permission": TokenType.KW_PERMISSION,
    "prohibition": TokenType.KW_PROHIBITION,
    "exemption": TokenType.KW_EXEMPTION,
    "belief": TokenType.KW_BELIEF,
    "base": TokenType.KW_BASE,
    "axiom": TokenType.KW_AXIOM,
    "tier": TokenType.KW_TIER,
    "type": TokenType.KW_TYPE,
    "struct": TokenType.KW_STRUCT,
    "enum": TokenType.KW_ENUM,
    "fn": TokenType.KW_FN,
    "return": TokenType.KW_RETURN,
    "if": TokenType.KW_IF,
    "else": TokenType.KW_ELSE,
    "match": TokenType.KW_MATCH,
    "for": TokenType.KW_FOR,
    "in": TokenType.KW_IN,
    "let": TokenType.KW_LET,
    "mut": TokenType.KW_MUT,
    "const": TokenType.KW_CONST,
    "import": TokenType.KW_IMPORT,
    "from": TokenType.KW_FROM,
    "as": TokenType.KW_AS,
    "export": TokenType.KW_EXPORT,
    "module": TokenType.KW_MODULE,
    "use": TokenType.KW_USE,
    "true": TokenType.KW_TRUE,
    "false": TokenType.KW_FALSE,
    "none": TokenType.KW_NONE,
    "and": TokenType.KW_AND,
    "or": TokenType.KW_OR,
    "not": TokenType.KW_NOT,
    "epistemic": TokenType.KW_EPISTEMIC,
    "temporal": TokenType.KW_TEMPORAL,
    "causal": TokenType.KW_CAUSAL,
    "deductive": TokenType.KW_DEDUCTIVE,
    "computational": TokenType.KW_COMPUTATIONAL,
    "observational": TokenType.KW_OBSERVATIONAL,
    "analogical": TokenType.KW_ANALOGICAL,
    "speculative": TokenType.KW_SPECULATIVE,
    "inductive": TokenType.KW_INDUCTIVE,
    "creative": TokenType.KW_CREATIVE,
    "eternal": TokenType.KW_ETERNAL,
    "stable": TokenType.KW_STABLE,
    "current": TokenType.KW_CURRENT,
    "ephemeral": TokenType.KW_EPHEMERAL,
    "historical": TokenType.KW_HISTORICAL,
    "domain": TokenType.KW_DOMAIN,
    "scope": TokenType.KW_SCOPE,
    "global": TokenType.KW_GLOBAL,
    "session": TokenType.KW_SESSION,
    "query": TokenType.KW_QUERY,
    "confidence": TokenType.KW_CONFIDENCE,
    "priority": TokenType.KW_PRIORITY,
    "action": TokenType.KW_ACTION,
    "block": TokenType.KW_BLOCK,
    "warn": TokenType.KW_WARN,
    "log": TokenType.KW_LOG,
    "permit": TokenType.KW_PERMIT,
    "attention": TokenType.KW_ATTENTION,
    "focus": TokenType.KW_FOCUS,
    "salience": TokenType.KW_SALIENCE,
    "budget": TokenType.KW_BUDGET,
    "belief_state": TokenType.KW_BELIEF_STATE,
    "desire": TokenType.KW_DESIRE,
    "intention": TokenType.KW_INTENTION,
    "deliberate": TokenType.KW_DELIBERATE,
    "affect": TokenType.KW_AFFECT,
    "mood": TokenType.KW_MOOD,
    "arousal": TokenType.KW_AROUSAL,
    "valence": TokenType.KW_VALENCE,
    "speech_act": TokenType.KW_SPEECH_ACT,
    "assert": TokenType.KW_ASSERT,
    "ask": TokenType.KW_ASK,
    "command": TokenType.KW_COMMAND,
    "promise": TokenType.KW_PROMISE,
    "warn_act": TokenType.KW_WARN_ACT,
    "explain": TokenType.KW_EXPLAIN,
    "because": TokenType.KW_BECAUSE,
    "despite": TokenType.KW_DESPITE,
    "implies": TokenType.KW_IMPLIES,
    "concept": TokenType.KW_CONCEPT,
    "subconcept_of": TokenType.KW_SUBCONCEPT_OF,
    "instance_of": TokenType.KW_INSTANCE_OF,
    "part_of": TokenType.KW_PART_OF,
    "bind": TokenType.KW_BIND,
    "modality": TokenType.KW_MODALITY,
    "fuse": TokenType.KW_FUSE,
    "compose": TokenType.KW_COMPOSE,
    "persist": TokenType.KW_PERSIST,
    "semantic_distance": TokenType.KW_SEMANTIC_DISTANCE,
    "decompose": TokenType.KW_DECOMPOSE,
    "intent_match": TokenType.KW_INTENT_MATCH,
    "while": TokenType.KW_WHILE,
    "try": TokenType.KW_TRY,
    "catch": TokenType.KW_CATCH,
    "finally": TokenType.KW_FINALLY,
    "break": TokenType.KW_BREAK,
    "continue": TokenType.KW_CONTINUE,
    "pass": TokenType.KW_PASS,
    "raise": TokenType.KW_RAISE,
    "del": TokenType.KW_DEL,
    "class": TokenType.KW_CLASS,
    "self": TokenType.KW_SELF,
    "super": TokenType.KW_SUPER,
    "is": TokenType.KW_IS,
    "with": TokenType.KW_WITH,
    "async": TokenType.KW_ASYNC,
    "await": TokenType.KW_AWAIT,
    "yield": TokenType.KW_YIELD,
    "lambda": TokenType.KW_LAMBDA,
    "nonlocal": TokenType.KW_NONLOCAL,
    "case": TokenType.KW_CASE,
}

_KEYWORD_NAMES: FrozenSet[str] = frozenset(KEYWORDS)


# ---------------------------------------------------------------------------
# Escape-sequence table for string literals
# ---------------------------------------------------------------------------

_STRING_ESCAPES: Dict[str, str] = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "\\": "\\",
    "'": "'",
    '"': '"',
    "0": "\0",
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "v": "\v",
}

_IDENT_START = frozenset(string.ascii_letters + "_")
_IDENT_BODY = frozenset(string.ascii_letters + string.digits + "_")
_DIGIT = frozenset(string.digits)
_DIGIT_OR_SEP = frozenset(string.digits + "_")


# ---------------------------------------------------------------------------
# Token value object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Token:
    """Immutable token emitted by the lexer."""

    type: TokenType
    value: str
    line: int
    column: int
    source_file: str = ""

    def __repr__(self) -> str:  # pragma: no cover
        val = self.value if len(self.value) <= 20 else self.value[:17] + "..."
        return (
            f"Token({self.type.name}, {val!r}, "
            f"{self.line}:{self.column})"
        )


# ---------------------------------------------------------------------------
# Lexer error
# ---------------------------------------------------------------------------

class LexerError(Exception):
    """Raised when the lexer encounters invalid source text."""

    def __init__(self, message: str, line: int, column: int,
                 filename: str = "") -> None:
        loc = f"{filename}:" if filename else ""
        super().__init__(f"{loc}{line}:{column}: {message}")
        self.line = line
        self.column = column
        self.filename = filename


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

class Lexer:
    """Tokenize NRSI source code into a stream of tokens.

    The lexer is single-pass and greedy: at each position it consumes the
    longest matching token.  Whitespace (other than newlines) is skipped
    but tracked for column bookkeeping.
    """

    def __init__(self, source: str, filename: str = "<stdin>") -> None:
        self._src = source
        self._filename = filename
        self._pos = 0
        self._line = 1
        self._col = 1
        self._tokens: List[Token] = []

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def tokenize(self) -> List[Token]:
        """Tokenize the entire source string and return the token list."""
        self._tokens = []
        self._pos = 0
        self._line = 1
        self._col = 1

        while True:
            tok = self._next_token()
            if tok is None:
                continue
            self._tokens.append(tok)
            if tok.type is TokenType.EOF:
                break

        return self._tokens

    # ------------------------------------------------------------------
    # character helpers
    # ------------------------------------------------------------------

    def _at_end(self) -> bool:
        return self._pos >= len(self._src)

    def _peek(self, offset: int = 0) -> str:
        idx = self._pos + offset
        if idx >= len(self._src):
            return "\0"
        return self._src[idx]

    def _advance(self) -> str:
        ch = self._src[self._pos]
        self._pos += 1
        if ch == "\n":
            self._line += 1
            self._col = 1
        else:
            self._col += 1
        return ch

    def _match(self, expected: str) -> bool:
        if self._at_end() or self._src[self._pos] != expected:
            return False
        self._advance()
        return True

    def _make(self, ttype: TokenType, value: str,
              line: int, col: int) -> Token:
        return Token(ttype, value, line, col, self._filename)

    def _error(self, msg: str) -> LexerError:
        return LexerError(msg, self._line, self._col, self._filename)

    def _validate_sep_digit_run(
        self, run: str, *, allow_empty: bool = False
    ) -> None:
        """Reject invalid underscore placement in a digit run (Python-like)."""
        if allow_empty and not run:
            return
        if not run:
            raise self._error("invalid numeric literal")
        if "__" in run:
            raise self._error("invalid numeric literal (consecutive '_')")
        if run.startswith("_") or run.endswith("_"):
            raise self._error("invalid numeric literal (misplaced '_')")
        if not any(c in string.digits for c in run):
            raise self._error("invalid numeric literal")

    def _validate_decimal_numeric(self, text: str, *, is_float: bool) -> None:
        """Validate underscores in decimal integer or float literal text."""
        if "_" not in text:
            return
        t = text
        exp_part = ""
        lower = t.lower()
        if "e" in lower:
            e_idx = lower.rfind("e")
            t, exp_part = t[:e_idx], t[e_idx + 1 :]
            if exp_part.startswith(("+", "-")):
                exp_part = exp_part[1:]
            self._validate_sep_digit_run(exp_part)
        if "." in t:
            whole, frac = t.split(".", 1)
            self._validate_sep_digit_run(whole)
            # Fractional part may be empty (e.g. `123.` or `1.e2`).
            self._validate_sep_digit_run(frac, allow_empty=True)
        else:
            self._validate_sep_digit_run(t)

    def _validate_radix_body(self, body: str) -> None:
        """Validate underscores in hex/octal/binary digit body (after prefix)."""
        if "_" not in body:
            return
        if "__" in body:
            raise self._error("invalid numeric literal (consecutive '_')")
        if body.startswith("_") or body.endswith("_"):
            raise self._error("invalid numeric literal (misplaced '_')")

    # ------------------------------------------------------------------
    # main dispatch
    # ------------------------------------------------------------------

    def _next_token(self) -> Optional[Token]:
        """Read and return the next token, or *None* to skip whitespace."""
        if self._at_end():
            return self._make(TokenType.EOF, "", self._line, self._col)

        ch = self._peek()

        # ── skip horizontal whitespace ────────────────────────────────
        if ch in (" ", "\t", "\r"):
            self._advance()
            return None

        # ── newline ───────────────────────────────────────────────────
        if ch == "\n":
            tok = self._make(TokenType.NEWLINE, "\n",
                             self._line, self._col)
            self._advance()
            return tok

        line, col = self._line, self._col

        # ── comments (//, ///, /* */) ─────────────────────────────────
        if ch == "/" and self._peek(1) == "/":
            return self._read_line_comment(line, col)

        if ch == "/" and self._peek(1) == "*":
            return self._read_block_comment(line, col)

        # ── string literals ───────────────────────────────────────────
        if ch in ('"', "'"):
            return self._read_string(line, col)

        # ── numeric literals ──────────────────────────────────────────
        if ch in _DIGIT or (ch == "." and self._peek(1) in _DIGIT):
            return self._read_number(line, col)

        # ── identifiers / keywords ────────────────────────────────────
        if ch in _IDENT_START:
            return self._read_identifier(line, col)

        # ── multi-char operators (longest match first) ────────────────
        tok = self._try_operator(line, col)
        if tok is not None:
            return tok

        raise self._error(f"unexpected character {ch!r}")

    # ------------------------------------------------------------------
    # identifiers & keywords
    # ------------------------------------------------------------------

    def _read_identifier(self, line: int, col: int) -> Token:
        start = self._pos
        while not self._at_end() and self._peek() in _IDENT_BODY:
            self._advance()
        text = self._src[start:self._pos]

        if text in ("r", "b", "f") and not self._at_end() and self._peek() in ('"', "'"):
            return self._read_prefixed_string(text, line, col)

        kw = KEYWORDS.get(text)
        if kw is not None:
            if kw is TokenType.KW_TRUE or kw is TokenType.KW_FALSE:
                return self._make(TokenType.BOOLEAN, text, line, col)
            if kw is TokenType.KW_NONE:
                return self._make(TokenType.NONE, text, line, col)
            return self._make(kw, text, line, col)

        return self._make(TokenType.IDENTIFIER, text, line, col)

    def _read_prefixed_string(self, prefix: str, line: int, col: int) -> Token:
        quote = self._advance()
        # Check for triple-quoted multi-line string or empty ""
        if not self._at_end() and self._peek() == quote:
            if self._peek(1) == quote:
                self._advance()  # second quote
                self._advance()  # third quote
                tok = self._read_triple_string(
                    quote, line, col, raw=(prefix == "r")
                )
                if prefix == "f":
                    return self._make(
                        TokenType.STRING, "\x00f\x00" + tok.value, line, col
                    )
                return tok
            self._advance()
            return self._make(TokenType.STRING, "", line, col)

        buf: List[str] = []
        while not self._at_end():
            ch = self._peek()
            if ch == "\n":
                break
            if ch == quote:
                self._advance()
                result = "".join(buf)
                if prefix == "f":
                    return self._make(
                        TokenType.STRING, "\x00f\x00" + result, line, col
                    )
                return self._make(TokenType.STRING, result, line, col)
            if ch == "\\" and prefix != "r":
                self._advance()
                esc_ch = self._peek() if not self._at_end() else "\0"
                if esc_ch != "\0":
                    self._advance()
                    replacement = _STRING_ESCAPES.get(esc_ch)
                    if replacement is not None:
                        buf.append(replacement)
                    elif esc_ch == "u":
                        buf.append(self._read_unicode_escape())
                    else:
                        buf.append("\\" + esc_ch)
                continue
            elif ch == "\\" and prefix == "r":
                buf.append(self._advance())
                if not self._at_end() and self._peek() != quote:
                    buf.append(self._advance())
                continue
            buf.append(self._advance())
        raise self._error("unterminated string literal")

    # ------------------------------------------------------------------
    # numeric literals
    # ------------------------------------------------------------------

    def _read_number(self, line: int, col: int) -> Token:
        start = self._pos
        is_float = False

        # Handle hex, octal, binary prefixes
        if self._peek() == "0" and not self._at_end():
            nxt = self._peek(1)
            if nxt in ("x", "X"):
                self._advance()  # '0'
                self._advance()  # 'x'
                hex_digits = frozenset("0123456789abcdefABCDEF_")
                start_hex = self._pos
                while not self._at_end() and self._peek() in hex_digits:
                    self._advance()
                if self._pos == start_hex:
                    raise self._error("expected hex digit after 0x")
                text = self._src[start:self._pos]
                self._validate_radix_body(text[2:])
                return self._make(TokenType.INTEGER, text, line, col)
            if nxt in ("o", "O"):
                self._advance()
                self._advance()
                oct_digits = frozenset("01234567_")
                start_oct = self._pos
                while not self._at_end() and self._peek() in oct_digits:
                    self._advance()
                if self._pos == start_oct:
                    raise self._error("expected octal digit after 0o")
                text = self._src[start:self._pos]
                self._validate_radix_body(text[2:])
                return self._make(TokenType.INTEGER, text, line, col)
            if nxt in ("b", "B"):
                self._advance()
                self._advance()
                bin_digits = frozenset("01_")
                start_bin = self._pos
                while not self._at_end() and self._peek() in bin_digits:
                    self._advance()
                if self._pos == start_bin:
                    raise self._error("expected binary digit after 0b")
                text = self._src[start:self._pos]
                self._validate_radix_body(text[2:])
                return self._make(TokenType.INTEGER, text, line, col)

        while not self._at_end() and self._peek() in _DIGIT_OR_SEP:
            self._advance()

        if not self._at_end() and self._peek() == "." and self._peek(1) != ".":
            is_float = True
            self._advance()  # consume '.'
            while not self._at_end() and self._peek() in _DIGIT_OR_SEP:
                self._advance()

        if not self._at_end() and self._peek() in ("e", "E"):
            is_float = True
            self._advance()
            if not self._at_end() and self._peek() in ("+", "-"):
                self._advance()
            if self._at_end() or self._peek() not in _DIGIT_OR_SEP:
                raise self._error("expected digit after exponent")
            while not self._at_end() and self._peek() in _DIGIT_OR_SEP:
                self._advance()

        text = self._src[start:self._pos]
        self._validate_decimal_numeric(text, is_float=is_float)
        ttype = TokenType.FLOAT if is_float else TokenType.INTEGER
        return self._make(ttype, text, line, col)

    # ------------------------------------------------------------------
    # string literals
    # ------------------------------------------------------------------

    def _read_string(self, line: int, col: int) -> Token:
        quote = self._advance()  # consume opening quote
        # Check for triple-quoted multi-line string or empty ""
        if not self._at_end() and self._peek() == quote:
            if self._peek(1) == quote:
                self._advance()  # second quote
                self._advance()  # third quote
                return self._read_triple_string(quote, line, col)
            self._advance()
            return self._make(TokenType.STRING, "", line, col)

        buf: List[str] = []

        while True:
            if self._at_end():
                raise self._error("unterminated string literal")
            ch = self._peek()

            if ch == "\n":
                raise self._error("unterminated string literal (newline)")

            if ch == quote:
                self._advance()
                break

            if ch == "\\":
                self._advance()
                esc_ch = self._peek()
                if esc_ch == "\0" or self._at_end():
                    raise self._error("unterminated escape sequence")
                self._advance()
                replacement = _STRING_ESCAPES.get(esc_ch)
                if replacement is not None:
                    buf.append(replacement)
                elif esc_ch == "u":
                    buf.append(self._read_unicode_escape())
                else:
                    buf.append("\\" + esc_ch)
            else:
                buf.append(self._advance())

        return self._make(TokenType.STRING, "".join(buf), line, col)

    def _read_triple_string(
        self, quote: str, line: int, col: int, *, raw: bool = False
    ) -> Token:
        """Read a triple-quoted (multi-line) string literal."""
        buf: List[str] = []
        while not self._at_end():
            if (
                self._peek() == quote
                and self._peek(1) == quote
                and self._peek(2) == quote
            ):
                self._advance()
                self._advance()
                self._advance()
                return self._make(TokenType.STRING, "".join(buf), line, col)
            ch = self._peek()
            if ch == "\\" and not raw:
                self._advance()
                esc_ch = self._peek() if not self._at_end() else "\0"
                if esc_ch == "\0":
                    raise self._error("unterminated escape in triple-quoted string")
                self._advance()
                replacement = _STRING_ESCAPES.get(esc_ch)
                if replacement is not None:
                    buf.append(replacement)
                elif esc_ch == "u":
                    buf.append(self._read_unicode_escape())
                else:
                    buf.append("\\" + esc_ch)
            elif ch == "\\" and raw:
                buf.append(self._advance())
                if not self._at_end():
                    buf.append(self._advance())
            else:
                buf.append(self._advance())
        raise self._error("unterminated triple-quoted string")

    def _read_unicode_escape(self) -> str:
        if self._at_end() or self._peek() != "{":
            return "\\u"
        self._advance()  # consume '{'
        digits: List[str] = []
        while not self._at_end() and self._peek() != "}":
            ch = self._peek()
            if ch not in frozenset("0123456789abcdefABCDEF"):
                raise self._error(
                    f"invalid hex digit {ch!r} in unicode escape"
                )
            digits.append(self._advance())
        if self._at_end():
            raise self._error("unterminated unicode escape")
        self._advance()  # consume '}'
        if not digits or len(digits) > 6:
            raise self._error("unicode escape must have 1–6 hex digits")
        codepoint = int("".join(digits), 16)
        try:
            return chr(codepoint)
        except (ValueError, OverflowError):
            raise self._error(
                f"invalid unicode codepoint U+{codepoint:04X}"
            )

    # ------------------------------------------------------------------
    # comments
    # ------------------------------------------------------------------

    def _read_line_comment(self, line: int, col: int) -> Token:
        self._advance()  # first '/'
        self._advance()  # second '/'

        is_doc = not self._at_end() and self._peek() == "/"
        if is_doc:
            self._advance()  # third '/'

        start = self._pos
        while not self._at_end() and self._peek() != "\n":
            self._advance()

        text = self._src[start:self._pos]
        ttype = TokenType.DOC_COMMENT if is_doc else TokenType.COMMENT
        return self._make(ttype, text.strip(), line, col)

    def _read_block_comment(self, line: int, col: int) -> Token:
        self._advance()  # '/'
        self._advance()  # '*'
        depth = 1
        start = self._pos

        while not self._at_end() and depth > 0:
            ch = self._peek()
            if ch == "/" and self._peek(1) == "*":
                self._advance()
                self._advance()
                depth += 1
            elif ch == "*" and self._peek(1) == "/":
                self._advance()
                self._advance()
                depth -= 1
            else:
                self._advance()

        if depth > 0:
            raise self._error("unterminated block comment")

        text = self._src[start:self._pos - 2]
        return self._make(TokenType.COMMENT, text.strip(), line, col)

    # ------------------------------------------------------------------
    # operators & punctuation
    # ------------------------------------------------------------------

    def _try_operator(self, line: int, col: int) -> Optional[Token]:
        """Try to match a punctuation or operator token, longest-first."""
        ch = self._peek()

        # --- three-character operators ---
        if ch == "." and self._peek(1) == "." and self._peek(2) == ".":
            self._advance(); self._advance(); self._advance()
            return self._make(TokenType.SPREAD, "...", line, col)

        if ch == "*" and self._peek(1) == "*" and self._peek(2) == "=":
            self._advance(); self._advance(); self._advance()
            return self._make(TokenType.POWER_ASSIGN, "**=", line, col)

        if ch == "~" and self._peek(1) == "~" and self._peek(2) == "=":
            self._advance(); self._advance(); self._advance()
            return self._make(TokenType.FLOOR_DIV_ASSIGN, "~~=", line, col)

        if ch == "<" and self._peek(1) == "<" and self._peek(2) == "=":
            self._advance(); self._advance(); self._advance()
            return self._make(TokenType.LEFT_SHIFT_ASSIGN, "<<=", line, col)

        if ch == ">" and self._peek(1) == ">" and self._peek(2) == "=":
            self._advance(); self._advance(); self._advance()
            return self._make(TokenType.RIGHT_SHIFT_ASSIGN, ">>=", line, col)

        # --- two-character operators ---
        nxt = self._peek(1)
        pair = ch + nxt if nxt != "\0" else ""

        two_char: Optional[TokenType] = _TWO_CHAR_OPS.get(pair)
        if two_char is not None:
            self._advance(); self._advance()
            return self._make(two_char, pair, line, col)

        # --- single-character operators ---
        one_char: Optional[TokenType] = _ONE_CHAR_OPS.get(ch)
        if one_char is not None:
            self._advance()
            return self._make(one_char, ch, line, col)

        return None


# ---------------------------------------------------------------------------
# Operator tables (used by _try_operator)
# ---------------------------------------------------------------------------

_TWO_CHAR_OPS: Dict[str, TokenType] = {
    "::": TokenType.DOUBLE_COLON,
    "==": TokenType.EQUAL,
    "!=": TokenType.NOT_EQUAL,
    ">=": TokenType.GTE,
    "<=": TokenType.LTE,
    "&&": TokenType.LOGICAL_AND,
    "||": TokenType.LOGICAL_OR,
    "->": TokenType.ARROW,
    "=>": TokenType.FAT_ARROW,
    "..": TokenType.RANGE,
    "+=": TokenType.PLUS_ASSIGN,
    "-=": TokenType.MINUS_ASSIGN,
    "*=": TokenType.STAR_ASSIGN,
    "/=": TokenType.SLASH_ASSIGN,
    "%=": TokenType.PERCENT_ASSIGN,
    "**": TokenType.POWER,
    "~~": TokenType.FLOOR_DIV,
    "<<": TokenType.LEFT_SHIFT,
    ">>": TokenType.RIGHT_SHIFT,
    "&=": TokenType.AMPERSAND_ASSIGN,
    "|=": TokenType.PIPE_ASSIGN,
    "^=": TokenType.CARET_ASSIGN,
}

_ONE_CHAR_OPS: Dict[str, TokenType] = {
    ":": TokenType.COLON,
    ";": TokenType.SEMICOLON,
    "{": TokenType.LBRACE,
    "}": TokenType.RBRACE,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    "[": TokenType.LBRACKET,
    "]": TokenType.RBRACKET,
    "<": TokenType.LANGLE,
    ">": TokenType.RANGLE,
    "=": TokenType.ASSIGN,
    "+": TokenType.PLUS,
    "-": TokenType.MINUS,
    "*": TokenType.STAR,
    "/": TokenType.SLASH,
    "%": TokenType.PERCENT,
    "!": TokenType.BANG,
    ".": TokenType.DOT,
    ",": TokenType.COMMA,
    "@": TokenType.AT,
    "#": TokenType.HASH,
    "|": TokenType.PIPE,
    "&": TokenType.AMPERSAND,
    "^": TokenType.CARET,
    "~": TokenType.TILDE,
}


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def tokenize(source: str, filename: str = "<stdin>") -> List[Token]:
    """Shorthand for ``Lexer(source, filename).tokenize()``."""
    return Lexer(source, filename).tokenize()


def tokenize_file(path: str) -> List[Token]:
    """Read *path* and tokenize its contents."""
    with open(path, encoding="utf-8") as fh:
        return Lexer(fh.read(), filename=path).tokenize()
