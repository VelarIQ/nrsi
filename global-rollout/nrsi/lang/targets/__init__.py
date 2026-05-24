"""NRSI Transpiler Target Registry.

Maps target names to transpiler classes so the CLI and runtime
can dispatch ``--target python``, ``--target swift``, etc.
"""

from __future__ import annotations

from typing import Dict, Type

REGISTRY: Dict[str, Type] = {}


def _register_builtin_targets() -> None:
    """Lazily populate the registry with built-in targets."""
    if REGISTRY:
        return

    from nrsi.lang.transpiler import Transpiler as PythonTranspiler
    REGISTRY["python"] = PythonTranspiler

    try:
        from nrsi.lang.targets.swift import SwiftTranspiler
        REGISTRY["swift"] = SwiftTranspiler
    except ImportError:
        pass

    try:
        from nrsi.lang.targets.kotlin import KotlinTranspiler
        REGISTRY["kotlin"] = KotlinTranspiler
    except ImportError:
        pass


def get_transpiler(target: str) -> Type:
    """Return the transpiler class for the given target name."""
    _register_builtin_targets()
    if target not in REGISTRY:
        available = ", ".join(sorted(REGISTRY.keys()))
        raise ValueError(f"Unknown target {target!r}. Available: {available}")
    return REGISTRY[target]


def list_targets() -> list:
    """Return list of registered target names."""
    _register_builtin_targets()
    return sorted(REGISTRY.keys())
