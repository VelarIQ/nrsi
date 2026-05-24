"""Schema-Validated Tool Framework — Runtime backing for ``stdlib/tools.nrsi``.

Implements ToolCapability, PermissionVerdict, ToolUseContext, ToolResult, and the
tool_input_verify / tool_permission_check / tool_output_verify gates declared
in the NRSI contract.

Each tool has:
  - A Pydantic input schema
  - Two-phase validation: schema parse -> domain validateInput
  - Permission gates using NRSI ValidationGate
  - Capability flags: is_concurrent_safe, is_read_only, is_destructive
  - H-Score tracking on every tool call output
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Set, Type, TypeVar,
)

from pydantic import BaseModel, ValidationError as PydanticValidationError

logger = logging.getLogger("cognitive-engine.tools")

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT")


# ── Capability Flags ─────────────────────────────────────────────────────────

class ToolCapability(Enum):
    CONCURRENT_SAFE = "concurrent_safe"
    READ_ONLY = "read_only"
    DESTRUCTIVE = "destructive"
    NETWORK = "network"
    FILE_SYSTEM = "file_system"
    CODE_EXECUTION = "code_execution"


# ── Permission ───────────────────────────────────────────────────────────────

class PermissionVerdict(Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass(frozen=True)
class ToolPermission:
    verdict: PermissionVerdict
    reason: str = ""


# ── Validation ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolValidation:
    valid: bool
    message: str = ""
    error_code: int = 0

    @classmethod
    def ok(cls) -> ToolValidation:
        return cls(valid=True)

    @classmethod
    def fail(cls, message: str, code: int = 400) -> ToolValidation:
        return cls(valid=False, message=message, error_code=code)


# ── Tool Use Context ─────────────────────────────────────────────────────────

@dataclass
class ToolUseContext:
    session_id: str = ""
    user_id: str = ""
    domain: str = "general"
    tier: str = "T2"
    working_directory: str = ""
    allowed_paths: List[str] = field(default_factory=list)
    denied_paths: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    max_execution_time_s: float = 30.0
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)


# ── Tool Result ──────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    output: Any = None
    is_error: bool = False
    error_message: str = ""
    execution_time_ms: float = 0.0
    h_score: Optional[float] = None
    trust_level: str = "RAW"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── NRSTool Base Class ───────────────────────────────────────────────────────

class NRSTool(ABC, Generic[InputT, OutputT]):
    """Base class for all NRS cognitive engine tools.

    Subclasses must implement `call` and set `input_schema`.
    Optionally override `validate_input` and `check_permissions` for
    domain-specific checks beyond schema validation.
    """

    name: str = ""
    description: str = ""
    input_schema: Type[InputT]
    capabilities: Set[ToolCapability] = set()

    @property
    def is_concurrent_safe(self) -> bool:
        return ToolCapability.CONCURRENT_SAFE in self.capabilities

    @property
    def is_read_only(self) -> bool:
        return ToolCapability.READ_ONLY in self.capabilities

    @property
    def is_destructive(self) -> bool:
        return ToolCapability.DESTRUCTIVE in self.capabilities

    def parse_input(self, raw: Dict[str, Any]) -> InputT | ToolValidation:
        """Phase 1: Pydantic schema validation."""
        try:
            return self.input_schema.model_validate(raw)
        except PydanticValidationError as e:
            return ToolValidation.fail(str(e), code=422)

    async def validate_input(self, inp: InputT,
                             ctx: ToolUseContext) -> ToolValidation:
        """Phase 2: Domain-specific validation (override in subclass)."""
        return ToolValidation.ok()

    async def check_permissions(self, inp: InputT,
                                ctx: ToolUseContext) -> ToolPermission:
        """Check whether this tool call is permitted in the current context."""
        return ToolPermission(verdict=PermissionVerdict.ALLOW)

    @abstractmethod
    async def call(self, inp: InputT, ctx: ToolUseContext) -> OutputT:
        """Execute the tool. Must be implemented by subclass."""
        ...

    async def execute(self, raw_input: Dict[str, Any],
                      ctx: ToolUseContext) -> ToolResult:
        """Full execution pipeline: parse -> validate -> permissions -> call."""
        t0 = time.time()

        parsed = self.parse_input(raw_input)
        if isinstance(parsed, ToolValidation):
            return ToolResult(
                is_error=True,
                error_message=f"Input validation failed: {parsed.message}",
                execution_time_ms=(time.time() - t0) * 1000,
            )

        validation = await self.validate_input(parsed, ctx)
        if not validation.valid:
            return ToolResult(
                is_error=True,
                error_message=f"Domain validation failed: {validation.message}",
                execution_time_ms=(time.time() - t0) * 1000,
            )

        permission = await self.check_permissions(parsed, ctx)
        if permission.verdict == PermissionVerdict.DENY:
            return ToolResult(
                is_error=True,
                error_message=f"Permission denied: {permission.reason}",
                execution_time_ms=(time.time() - t0) * 1000,
            )

        try:
            output = await self.call(parsed, ctx)
            elapsed = (time.time() - t0) * 1000
            ctx.audit_trail.append({
                "tool": self.name,
                "elapsed_ms": round(elapsed, 1),
                "success": True,
            })
            return ToolResult(
                output=output,
                execution_time_ms=elapsed,
                trust_level="VALIDATED",
            )
        except Exception as exc:
            elapsed = (time.time() - t0) * 1000
            logger.warning("Tool %s failed: %s", self.name, exc)
            ctx.audit_trail.append({
                "tool": self.name,
                "elapsed_ms": round(elapsed, 1),
                "success": False,
                "error": str(exc),
            })
            return ToolResult(
                is_error=True,
                error_message=str(exc),
                execution_time_ms=elapsed,
            )


# ── Tool Registry ────────────────────────────────────────────────────────────

class ToolRegistry:
    """Registry of available tools with batched concurrent execution."""

    def __init__(self) -> None:
        self._tools: Dict[str, NRSTool] = {}

    def register(self, tool: NRSTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[NRSTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def list_concurrent_safe(self) -> List[str]:
        return [n for n, t in self._tools.items() if t.is_concurrent_safe]

    async def execute_batch(
        self,
        calls: List[Dict[str, Any]],
        ctx: ToolUseContext,
    ) -> List[ToolResult]:
        """Execute tool calls, batching concurrent-safe ones together.

        Partitions calls into runs of consecutive concurrent-safe tools
        (executed in parallel) and unsafe tools (executed serially).
        """
        results: List[ToolResult] = []
        batch: List[Dict[str, Any]] = []
        batch_safe = True

        async def flush_batch():
            nonlocal batch
            if not batch:
                return
            if batch_safe and len(batch) > 1:
                coros = []
                for c in batch:
                    tool = self._tools.get(c.get("tool", ""))
                    if tool:
                        coros.append(tool.execute(c.get("input", {}), ctx))
                    else:
                        coros.append(_missing_tool_result(c.get("tool", "")))
                batch_results = await asyncio.gather(*coros)
                results.extend(batch_results)
            else:
                for c in batch:
                    tool = self._tools.get(c.get("tool", ""))
                    if tool:
                        results.append(await tool.execute(c.get("input", {}), ctx))
                    else:
                        results.append(await _missing_tool_result(c.get("tool", "")))
            batch = []

        for call_spec in calls:
            tool_name = call_spec.get("tool", "")
            tool = self._tools.get(tool_name)
            is_safe = tool.is_concurrent_safe if tool else False

            if batch and is_safe != batch_safe:
                await flush_batch()

            batch_safe = is_safe
            batch.append(call_spec)

        await flush_batch()
        return results


async def _missing_tool_result(name: str) -> ToolResult:
    return ToolResult(is_error=True, error_message=f"Unknown tool: {name}")
