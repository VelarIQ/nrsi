"""NRSI Runtime Builtins — Conversion helpers and utilities for transpiled code.

These functions are injected into every .nrsi module namespace before execution.
They bridge NRSI-language idioms to their Python equivalents.
"""

from __future__ import annotations

import datetime
import inspect
import os
from typing import Any, Union


def to_int(x: Any, base: int | None = None) -> int:
    if base is not None:
        return int(x, base)
    if isinstance(x, str):
        x_stripped = x.strip()
        if x_stripped.startswith("0x") or x_stripped.startswith("0X"):
            return int(x_stripped, 16)
    return int(x)


def to_float(x: Any) -> float:
    return float(x)


def to_string(x: Any) -> str:
    return str(x)


def hex_to_int(x: str) -> int:
    return int(x, 16)


def bytes_to_int(b: bytes, byteorder: str = "big") -> int:
    return int.from_bytes(b, byteorder=byteorder)


def to_hex(x: Union[bytes, int]) -> str:
    if isinstance(x, bytes):
        return x.hex()
    return format(x, "x")


def iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def module_dir() -> str:
    frame = inspect.currentframe()
    try:
        caller = frame.f_back  # type: ignore[union-attr]
        filepath = caller.f_globals.get("__file__", "")  # type: ignore[union-attr]
        return os.path.dirname(os.path.abspath(filepath))
    finally:
        del frame


def slice_string(s: str, start: int, end: int | None = None) -> str:
    if end is None:
        return s[start:]
    return s[start:end]


def tuple_get(t: tuple, index: int, default: Any = None) -> Any:
    if 0 <= index < len(t):
        return t[index]
    return default


NRSI_BUILTINS: dict[str, Any] = {
    "to_int": to_int,
    "to_float": to_float,
    "to_string": to_string,
    "hex_to_int": hex_to_int,
    "bytes_to_int": bytes_to_int,
    "to_hex": to_hex,
    "iso_now": iso_now,
    "module_dir": module_dir,
    "slice_string": slice_string,
    "tuple_get": tuple_get,
    "nil": None,
}
