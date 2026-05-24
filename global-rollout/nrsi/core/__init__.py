"""Lazy exports for `nrsi.core`.

This package is large and several optional subsystems are intentionally
heavyweight. Eagerly importing every symbol at package import time makes
simple submodule imports brittle, because `import nrsi.core.validation`
first executes `nrsi.core.__init__`.

To keep service startup stable, resolve exports lazily instead.
"""

from __future__ import annotations

from importlib import import_module

_SEARCH_MODULES = (
    "nrsi.core.types",
    "nrsi.core.validation",
    "nrsi.core.knowledge",
    "nrsi.core.hierarchy",
    "nrsi.core.signals",
    "nrsi.core.parallel",
    "nrsi.core.memory",
    "nrsi.core.plasticity",
    "nrsi.core.errors",
    "nrsi.core.neurons",
    "nrsi.core.creases",
    "nrsi.core.lobes",
    "nrsi.core.router",
    "nrsi.core.hscore",
    "nrsi.core.media",
    "nrsi.core.streaming",
    "nrsi.core.neural_renderer",
    "nrsi.core.neural_cache",
    "nrsi.core.creative_vision",
    "nrsi.core.neural_audio",
    "nrsi.core.gpu_renderer",
    "nrsi.core.epistemic",
    "nrsi.core.attention",
    "nrsi.core.metacognitive",
    "nrsi.core.uncertainty",
    "nrsi.core.belief_revision",
    "nrsi.core.bdi",
    "nrsi.core.affective",
)


def __getattr__(name: str):
    for module_name in _SEARCH_MODULES:
        try:
            module = import_module(module_name)
        except Exception:
            continue
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module 'nrsi.core' has no attribute {name!r}")


def __dir__() -> list[str]:
    names = set(globals())
    for module_name in _SEARCH_MODULES:
        try:
            module = import_module(module_name)
        except Exception:
            continue
        names.update(getattr(module, "__all__", ()))
        names.update(name for name in vars(module) if not name.startswith("_"))
    return sorted(names)
