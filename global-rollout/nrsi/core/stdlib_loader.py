"""NRSI Stdlib Loader — Bridge Between .nrsi Definitions and the Python Pipeline.

Activates the NRSI language runtime, loads the stdlib .nrsi files, and
extracts typed Python objects (gates, norms, lobes, beliefs) that the NRS
pipeline can consume directly.

The transpiler produces Python that imports from ``nrsi.core.*`` — the exact
same classes the pipeline uses.  This loader materialises those definitions
so the .nrsi stdlib becomes the authoritative source of truth.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from nrsi.core.normative import (
    DeonticType,
    Norm,
    NormScope,
    NormSet,
)

logger = logging.getLogger("nrsi.stdlib")

_STDLIB_DIR = Path(__file__).resolve().parent.parent / "stdlib"

_STDLIB_FILES = [
    "types",
    "gates",
    "norms",
    "lobes",
    "beliefs",
    "reasoning",
    "thinking",
    "verification",
    "tools",
    "context",
    "cognitive",
    "cognitive_loop",
    "mode_policy",
    "routing",
    "speech_acts",
    "bdi",
    "affective",
    "metacognition",
    "uncertainty",
    "attention",
    "self_model",
    "revision",
    "tvs",
    "multimodal",
    "explanations",
    "epistemic",
    "nrs_knowledge_pipeline",
    "nrs_vlt_seeder",
    "nrs_validation",
    "web_retrieval",
    "reasoning_web",
    "reasoning_epistemic",
    "client_response_policy",
    "nrs_domain_registry",
    "parallel",
    "logic_engine",
    "causal_engine",
    "analogy_engine",
    "reasoning_planning",
    "working_memory",
    "self_improvement",
    "reasoning_temporal",
    "reasoning_semantic",
    "metacognition_engine",
]


@dataclass
class NRSIStdlib:
    """Typed container for everything loaded from the .nrsi stdlib."""

    gates: Dict[str, Any] = field(default_factory=dict)
    norms: NormSet = field(default_factory=NormSet)
    lobes: Dict[str, Any] = field(default_factory=dict)
    beliefs: Dict[str, Any] = field(default_factory=dict)
    reasoning_gates: Dict[str, Any] = field(default_factory=dict)
    verification_gates: Dict[str, Any] = field(default_factory=dict)
    tool_gates: Dict[str, Any] = field(default_factory=dict)
    context_gates: Dict[str, Any] = field(default_factory=dict)
    mode_policy_gates: Dict[str, Any] = field(default_factory=dict)
    routing_gates: Dict[str, Any] = field(default_factory=dict)
    speech_act_gates: Dict[str, Any] = field(default_factory=dict)
    bdi_gates: Dict[str, Any] = field(default_factory=dict)
    affective_gates: Dict[str, Any] = field(default_factory=dict)
    metacognition_gates: Dict[str, Any] = field(default_factory=dict)
    uncertainty_gates: Dict[str, Any] = field(default_factory=dict)
    attention_gates: Dict[str, Any] = field(default_factory=dict)
    tvs_gates: Dict[str, Any] = field(default_factory=dict)
    multimodal_gates: Dict[str, Any] = field(default_factory=dict)
    explanation_gates: Dict[str, Any] = field(default_factory=dict)
    modules: Dict[str, Any] = field(default_factory=dict)


_singleton: Optional[NRSIStdlib] = None


def _activate_runtime() -> bool:
    """Install the NRSI import hook so .nrsi files are importable."""
    try:
        from nrsi.lang.runtime import install, is_installed
        if not is_installed():
            install()
        stdlib_str = str(_STDLIB_DIR)
        if stdlib_str not in sys.path:
            sys.path.insert(0, stdlib_str)
        nrsip_dir = str(_STDLIB_DIR.parent.parent / "nrsip")
        if os.path.isdir(nrsip_dir) and nrsip_dir not in sys.path:
            sys.path.insert(0, nrsip_dir)
        return True
    except Exception as exc:
        logger.warning("NRSI runtime activation failed: %s", exc)
        return False


def _safe_load(name: str) -> Optional[Any]:
    """Load a single .nrsi stdlib file, returning the module or None."""
    path = _STDLIB_DIR / f"{name}.nrsi"
    if not path.is_file():
        logger.debug("Stdlib file not found: %s", path)
        return None
    try:
        from nrsi.lang.runtime import load_nrsi
        mod = load_nrsi(str(path), module_name=f"nrsi_stdlib_{name}")
        return mod
    except Exception as exc:
        logger.warning("Failed to load %s.nrsi: %s", name, exc)
        return None


def _extract_gates(mod: Any) -> Dict[str, Any]:
    """Pull ValidationGate instances from a loaded .nrsi module.

    The transpiler emits gates as ``_gate_<name> = ValidationGate(...)``
    and a wrapper function ``<name>(data)`` that calls ``_gate_<name>.process(data)``.
    We extract both the gate objects and their wrapper functions.
    """
    from nrsi.core.validation import ValidationGate
    gates: Dict[str, Any] = {}
    for attr_name in dir(mod):
        obj = getattr(mod, attr_name, None)
        if isinstance(obj, ValidationGate):
            clean_name = attr_name.replace("_gate_", "") if attr_name.startswith("_gate_") else attr_name
            gates[clean_name] = obj
        elif callable(obj) and not attr_name.startswith("_"):
            if hasattr(obj, "gate") and isinstance(obj.gate, ValidationGate):
                gates[attr_name] = obj.gate
            elif hasattr(obj, "_nrsi_gate"):
                gates[attr_name] = obj._nrsi_gate
    return gates


def _extract_norms(mod: Any) -> NormSet:
    """Pull Norm instances from the loaded norms.nrsi module into a NormSet.

    The transpiler emits norms as ``_norm_<name> = Norm(...)`` variables.
    """
    ns = NormSet()
    for attr_name in dir(mod):
        if attr_name.startswith("__"):
            continue
        obj = getattr(mod, attr_name, None)
        if isinstance(obj, Norm):
            ns.add_norm(obj)
    return ns


def _extract_lobes(mod: Any) -> Dict[str, Any]:
    """Pull ProcessingLobe instances from the loaded lobes.nrsi module."""
    lobes: Dict[str, Any] = {}
    for attr_name in dir(mod):
        if attr_name.startswith("_"):
            continue
        obj = getattr(mod, attr_name, None)
        if hasattr(obj, "register_processor") or attr_name.endswith("_lobe"):
            lobes[attr_name] = obj
    return lobes


def _extract_beliefs(mod: Any) -> Dict[str, Any]:
    """Pull BeliefBase instances from the loaded beliefs.nrsi module.

    The transpiler emits belief bases as ``_belief_base_<name> = BeliefBase()``.
    """
    from nrsi.core.belief_revision import BeliefBase
    beliefs: Dict[str, Any] = {}
    for attr_name in dir(mod):
        if attr_name.startswith("__"):
            continue
        obj = getattr(mod, attr_name, None)
        if isinstance(obj, BeliefBase):
            clean = attr_name
            if clean.startswith("_belief_base_"):
                clean = clean[len("_belief_base_"):]
            beliefs[clean] = obj
    return beliefs


def load_stdlib(*, force: bool = False) -> NRSIStdlib:
    """Load the NRSI stdlib and return a typed container.

    Results are cached as a module-level singleton.  Pass ``force=True``
    to reload everything from disk.
    """
    global _singleton
    if _singleton is not None and not force:
        return _singleton

    stdlib = NRSIStdlib()

    if not _STDLIB_DIR.is_dir():
        logger.warning("NRSI stdlib directory not found at %s", _STDLIB_DIR)
        _singleton = stdlib
        return stdlib

    if not _activate_runtime():
        logger.warning("NRSI runtime unavailable — stdlib loaded empty")
        _singleton = stdlib
        return stdlib

    for name in _STDLIB_FILES:
        mod = _safe_load(name)
        if mod is None:
            continue
        stdlib.modules[name] = mod

    if "gates" in stdlib.modules:
        stdlib.gates = _extract_gates(stdlib.modules["gates"])
        logger.info("Loaded %d gates from gates.nrsi", len(stdlib.gates))

    if "norms" in stdlib.modules:
        stdlib.norms = _extract_norms(stdlib.modules["norms"])
        logger.info("Loaded %d norms from norms.nrsi",
                     len(stdlib.norms._norms))

    if "lobes" in stdlib.modules:
        stdlib.lobes = _extract_lobes(stdlib.modules["lobes"])
        logger.info("Loaded %d lobes from lobes.nrsi", len(stdlib.lobes))

    if "beliefs" in stdlib.modules:
        stdlib.beliefs = _extract_beliefs(stdlib.modules["beliefs"])
        logger.info("Loaded %d belief bases from beliefs.nrsi",
                     len(stdlib.beliefs))

    if "reasoning" in stdlib.modules:
        stdlib.reasoning_gates = _extract_gates(stdlib.modules["reasoning"])

    if "verification" in stdlib.modules:
        stdlib.verification_gates = _extract_gates(
            stdlib.modules["verification"])
        logger.info("Loaded %d verification gates from verification.nrsi",
                     len(stdlib.verification_gates))

    if "tools" in stdlib.modules:
        stdlib.tool_gates = _extract_gates(stdlib.modules["tools"])

    if "context" in stdlib.modules:
        stdlib.context_gates = _extract_gates(stdlib.modules["context"])

    _new_gate_modules = {
        "mode_policy": "mode_policy_gates",
        "routing": "routing_gates",
        "speech_acts": "speech_act_gates",
        "bdi": "bdi_gates",
        "affective": "affective_gates",
        "metacognition": "metacognition_gates",
        "uncertainty": "uncertainty_gates",
        "attention": "attention_gates",
        "tvs": "tvs_gates",
        "multimodal": "multimodal_gates",
        "explanations": "explanation_gates",
    }
    for mod_name, attr_name in _new_gate_modules.items():
        if mod_name in stdlib.modules:
            setattr(stdlib, attr_name, _extract_gates(stdlib.modules[mod_name]))

    _belief_modules = ["self_model", "revision"]
    for mod_name in _belief_modules:
        if mod_name in stdlib.modules:
            _beliefs = _extract_beliefs(stdlib.modules[mod_name])
            stdlib.beliefs.update(_beliefs)

    _norm_modules = ["mode_policy", "routing", "speech_acts", "bdi",
                     "affective", "metacognition", "uncertainty",
                     "attention", "tvs", "multimodal", "explanations",
                     "epistemic", "nrs_knowledge_pipeline",
                     "nrs_validation", "client_response_policy",
                     "parallel", "nrs_vlt_seeder"]
    for mod_name in _norm_modules:
        if mod_name in stdlib.modules:
            _extra_norms = _extract_norms(stdlib.modules[mod_name])
            for norm in _extra_norms._norms.values():
                stdlib.norms.add_norm(norm)

    _total_gates = sum(
        len(getattr(stdlib, a, {})) for a in dir(stdlib)
        if a.endswith("_gates") and isinstance(getattr(stdlib, a, None), dict)
    )
    _total_norms = len(stdlib.norms._norms)
    _total_beliefs = len(stdlib.beliefs)
    logger.info("Stdlib loaded: %d modules, %d gates, %d norms, %d belief bases",
                len(stdlib.modules), _total_gates, _total_norms, _total_beliefs)

    _singleton = stdlib
    return stdlib


def get_stdlib() -> NRSIStdlib:
    """Return the cached stdlib, loading it if necessary."""
    if _singleton is None:
        return load_stdlib()
    return _singleton


_nrsi_module_cache: Dict[str, Any] = {}


def load_nrsi_module(name: str) -> Optional[Any]:
    """Load an implementation module from .nrsi stdlib by name.

    This replaces direct ``from nrsip.X import Y`` imports.
    The .nrsi file is transpiled to Python and executed, returning
    a module object with all exported symbols as attributes.

    Cached after first load.
    """
    if name in _nrsi_module_cache:
        return _nrsi_module_cache[name]

    _activate_runtime()

    mod = _safe_load(name)
    if mod is not None:
        _nrsi_module_cache[name] = mod
        return mod

    logger.debug("NRSI module %s not found in stdlib", name)
    return None
