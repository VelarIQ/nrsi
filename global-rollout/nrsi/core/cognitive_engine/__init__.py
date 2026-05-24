"""NRS Cognitive Engine — Adaptive reasoning, verification, and tool orchestration.

NRSI contracts live in ``nrsi/stdlib/``:
    thinking.nrsi        — phases, budgets, tier-aware allocation
    reasoning.nrsi       — multi-strategy gates, evidence types
    verification.nrsi    — build-test-scan-verify gates
    tools.nrsi           — schema-validated tool execution gates
    context.nrsi         — token tracking, compaction, VLT storage
    cognitive_loop.nrsi  — top-level orchestrator lobe + termination types

These Python modules are the **runtime backing** for those declarations,
the same pattern as ``stdlib/gates.nrsi`` → ``core/validation.py``.
"""

from nrsi.core.cognitive_engine.thinking import (
    ThinkingConfig,
    ThinkingPhase,
    ThinkingBudgetManager,
    AdaptiveThinkingController,
)

from nrsi.core.cognitive_engine.tool_framework import (
    NRSTool,
    ToolRegistry,
    ToolCapability,
    ToolPermission,
    ToolUseContext,
)

from nrsi.core.cognitive_engine.query_loop import (
    CognitiveLoop,
    QueryState,
    LoopEvent,
    LoopTermination,
    HookManager,
)

from nrsi.core.cognitive_engine.context_manager import (
    ContextManager,
    TokenBudgetTracker,
    AutoCompactor,
    SessionMemoryExtractor,
)

from nrsi.core.cognitive_engine.verification import (
    VerificationOrchestrator,
    VerificationPhase,
    VerificationResult,
    CompletionGate,
    DependencyChecker,
    BuildRunner,
    TestRunner,
    LintScanner,
    SecurityScanner,
)

from nrsi.core.cognitive_engine.reasoning import (
    ReasoningChain,
    ReasoningStep,
    ReasoningStrategy,
    EvidenceAccumulator,
    ConflictResolver,
)

__all__ = [
    "ThinkingConfig", "ThinkingPhase", "ThinkingBudgetManager",
    "AdaptiveThinkingController",
    "NRSTool", "ToolRegistry", "ToolCapability", "ToolPermission", "ToolUseContext",
    "CognitiveLoop", "QueryState", "LoopEvent", "LoopTermination", "HookManager",
    "ContextManager", "TokenBudgetTracker", "AutoCompactor", "SessionMemoryExtractor",
    "VerificationOrchestrator", "VerificationPhase", "VerificationResult",
    "CompletionGate", "DependencyChecker", "BuildRunner", "TestRunner",
    "LintScanner", "SecurityScanner",
    "ReasoningChain", "ReasoningStep", "ReasoningStrategy",
    "EvidenceAccumulator", "ConflictResolver",
]
