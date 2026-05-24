"""
NRSI — NRS Instructions.

The programming language for neuromorphic AI validation.
Every data type carries trust metadata, every operation has
validation semantics, and every output has provenance.

This package re-exports the full NRSI core API so services
can import directly::

    from nrsi import NRSIData, raw, ValidationGate, NRS

The ``nrsi.lang`` sub-package (lexer, parser, transpiler, runtime)
can be imported independently with zero NRS-core dependencies.
"""


def __getattr__(name):
    """Lazy re-export: only load nrsi.core when something is accessed."""
    import nrsi.core as _core  # noqa: F401
    val = getattr(_core, name, None)
    if val is not None:
        return val
    raise AttributeError(f"module 'nrsi' has no attribute {name!r}")
