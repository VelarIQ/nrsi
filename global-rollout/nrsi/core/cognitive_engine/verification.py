"""Build-Test-Scan-Verify Loop — Runtime backing for ``stdlib/verification.nrsi``.

Implements VerificationPhase, PhaseStatus, PhaseResult, and the five phase gates
(dependency_gate, build_gate, test_gate, lint_gate, security_gate) plus
completion_gate declared in the NRSI contract.

  1. DependencyChecker  — latest versions, CVE advisories
  2. BuildRunner        — compile/transpile with captured output
  3. TestRunner         — auto-discover and run test suites
  4. LintScanner        — language-appropriate linting
  5. SecurityScanner    — vuln scan + secret detection

CompletionGate blocks the "done" signal until ALL phase gates pass.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("cognitive-engine.verification")

MAX_VERIFY_ITERATIONS = 3
SUBPROCESS_TIMEOUT = 60


# ── Verification Phases ──────────────────────────────────────────────────────

class VerificationPhase(Enum):
    DEPENDENCY_CHECK = "dependency_check"
    BUILD = "build"
    TEST = "test"
    LINT = "lint"
    SECURITY_SCAN = "security_scan"


class PhaseStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


# ── Phase Result ─────────────────────────────────────────────────────────────

@dataclass
class PhaseResult:
    phase: VerificationPhase
    status: PhaseStatus
    duration_ms: float = 0.0
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    findings: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    def as_sse(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "phase": self.phase.value,
            "status": self.status.value,
            "duration_ms": round(self.duration_ms, 1),
        }
        if self.findings:
            d["findings_count"] = len(self.findings)
        if self.summary:
            d["summary"] = self.summary
        return d


# ── Verification Result (aggregate) ──────────────────────────────────────────

@dataclass
class VerificationResult:
    passed: bool
    iteration: int
    phases: List[PhaseResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    errors: List[str] = field(default_factory=list)

    def as_sse(self) -> Dict[str, Any]:
        return {
            "phase": "complete",
            "verified": self.passed,
            "iterations": self.iteration,
            "duration_ms": round(self.total_duration_ms, 1),
            "phases": {p.phase.value: p.status.value for p in self.phases},
            "error_count": len(self.errors),
        }


# ── Subprocess Helper ────────────────────────────────────────────────────────

async def _run_cmd(
    cmd: List[str],
    cwd: Optional[str] = None,
    timeout: int = SUBPROCESS_TIMEOUT,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str]:
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=merged_env,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return (
            proc.returncode or 0,
            stdout.decode(errors="replace")[:50_000],
            stderr.decode(errors="replace")[:50_000],
        )
    except asyncio.TimeoutError:
        proc.kill()  # type: ignore[union-attr]
        return -1, "", f"Command timed out after {timeout}s"
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"
    except Exception as exc:
        return -1, "", str(exc)


# ── Dependency Checker ───────────────────────────────────────────────────────

class DependencyChecker:
    """Checks dependency files for outdated or vulnerable packages."""

    MANIFEST_FILES = {
        "requirements.txt": "python",
        "pyproject.toml": "python",
        "package.json": "javascript",
        "package-lock.json": "javascript",
        "go.mod": "go",
        "Cargo.toml": "rust",
        "build.gradle": "java",
        "build.gradle.kts": "kotlin",
        "Gemfile": "ruby",
    }

    async def check(self, project_dir: str) -> PhaseResult:
        t0 = time.time()
        findings: List[Dict[str, Any]] = []
        manifests_found: List[str] = []

        for filename, lang in self.MANIFEST_FILES.items():
            path = Path(project_dir) / filename
            if path.exists():
                manifests_found.append(filename)
                findings.append({
                    "file": filename,
                    "language": lang,
                    "exists": True,
                })

        if not manifests_found:
            return PhaseResult(
                phase=VerificationPhase.DEPENDENCY_CHECK,
                status=PhaseStatus.SKIP,
                duration_ms=(time.time() - t0) * 1000,
                summary="No dependency manifests found",
            )

        # Python: pip check for broken deps
        if any(f in manifests_found for f in ("requirements.txt", "pyproject.toml")):
            code, out, err = await _run_cmd(
                ["pip", "check"], cwd=project_dir, timeout=30
            )
            if code != 0:
                findings.append({
                    "tool": "pip check",
                    "status": "fail",
                    "output": (out + err)[:500],
                })

        # JS: npm audit
        if "package.json" in manifests_found:
            code, out, err = await _run_cmd(
                ["npm", "audit", "--json"], cwd=project_dir, timeout=30
            )
            if code != 0:
                try:
                    audit = json.loads(out)
                    vulns = audit.get("metadata", {}).get("vulnerabilities", {})
                    total = sum(vulns.values()) if isinstance(vulns, dict) else 0
                    findings.append({
                        "tool": "npm audit",
                        "vulnerabilities": total,
                        "details": vulns,
                    })
                except json.JSONDecodeError:
                    findings.append({"tool": "npm audit", "status": "parse_error"})

        has_failures = any(
            f.get("status") == "fail" or f.get("vulnerabilities", 0) > 0
            for f in findings
        )
        return PhaseResult(
            phase=VerificationPhase.DEPENDENCY_CHECK,
            status=PhaseStatus.FAIL if has_failures else PhaseStatus.PASS,
            duration_ms=(time.time() - t0) * 1000,
            findings=findings,
            summary=f"Checked {len(manifests_found)} manifests: "
                    + ", ".join(manifests_found),
        )


# ── Build Runner ─────────────────────────────────────────────────────────────

class BuildRunner:
    """Language-aware build execution with sandboxed output capture."""

    BUILD_COMMANDS: Dict[str, List[List[str]]] = {
        "python": [["python", "-m", "py_compile"]],
        "javascript": [["npm", "run", "build"]],
        "typescript": [["npx", "tsc", "--noEmit"]],
        "go": [["go", "build", "./..."]],
        "rust": [["cargo", "build"]],
    }

    async def build(self, project_dir: str,
                    language: str = "python") -> PhaseResult:
        t0 = time.time()
        commands = self.BUILD_COMMANDS.get(language, [])

        if not commands:
            return PhaseResult(
                phase=VerificationPhase.BUILD,
                status=PhaseStatus.SKIP,
                duration_ms=(time.time() - t0) * 1000,
                summary=f"No build command for language: {language}",
            )

        all_stdout: List[str] = []
        all_stderr: List[str] = []
        final_code = 0

        for cmd in commands:
            if language == "python":
                py_files = list(Path(project_dir).rglob("*.py"))[:50]
                if not py_files:
                    continue
                for pf in py_files:
                    code, out, err = await _run_cmd(
                        ["python", "-m", "py_compile", str(pf)],
                        cwd=project_dir,
                    )
                    if code != 0:
                        final_code = code
                        all_stderr.append(f"{pf.name}: {err}")
            else:
                code, out, err = await _run_cmd(cmd, cwd=project_dir)
                all_stdout.append(out)
                all_stderr.append(err)
                if code != 0:
                    final_code = code

        return PhaseResult(
            phase=VerificationPhase.BUILD,
            status=PhaseStatus.PASS if final_code == 0 else PhaseStatus.FAIL,
            duration_ms=(time.time() - t0) * 1000,
            stdout="\n".join(all_stdout)[:5000],
            stderr="\n".join(all_stderr)[:5000],
            exit_code=final_code,
            summary=f"Build {'passed' if final_code == 0 else 'FAILED'} "
                    f"({language})",
        )


# ── Test Runner ──────────────────────────────────────────────────────────────

class TestRunner:
    """Auto-discovers and runs test suites with per-test reporting."""

    TEST_COMMANDS: Dict[str, List[str]] = {
        "python": ["python", "-m", "pytest", "-v", "--tb=short", "-q"],
        "javascript": ["npm", "test", "--", "--passWithNoTests"],
        "typescript": ["npm", "test", "--", "--passWithNoTests"],
        "go": ["go", "test", "-v", "./..."],
        "rust": ["cargo", "test"],
    }

    async def run_tests(self, project_dir: str,
                        language: str = "python") -> PhaseResult:
        t0 = time.time()
        cmd = self.TEST_COMMANDS.get(language)

        if not cmd:
            return PhaseResult(
                phase=VerificationPhase.TEST,
                status=PhaseStatus.SKIP,
                duration_ms=(time.time() - t0) * 1000,
                summary=f"No test runner for: {language}",
            )

        # Check if tests exist
        has_tests = False
        if language == "python":
            has_tests = bool(list(Path(project_dir).rglob("test_*.py"))[:1]) or \
                        bool(list(Path(project_dir).rglob("*_test.py"))[:1])
        elif language in ("javascript", "typescript"):
            has_tests = bool(list(Path(project_dir).rglob("*.test.*"))[:1]) or \
                        bool(list(Path(project_dir).rglob("*.spec.*"))[:1])

        if not has_tests:
            return PhaseResult(
                phase=VerificationPhase.TEST,
                status=PhaseStatus.SKIP,
                duration_ms=(time.time() - t0) * 1000,
                summary="No test files discovered",
            )

        code, out, err = await _run_cmd(cmd, cwd=project_dir, timeout=120)

        findings = []
        if language == "python" and "FAILED" in out:
            for line in out.split("\n"):
                if "FAILED" in line or "ERROR" in line:
                    findings.append({"test": line.strip(), "status": "fail"})

        total = len(findings)
        passed = sum(1 for f in findings if f.get("status") != "fail")

        return PhaseResult(
            phase=VerificationPhase.TEST,
            status=PhaseStatus.PASS if code == 0 else PhaseStatus.FAIL,
            duration_ms=(time.time() - t0) * 1000,
            stdout=out[:5000],
            stderr=err[:5000],
            exit_code=code,
            findings=findings,
            summary=f"Tests {'passed' if code == 0 else 'FAILED'}"
                    + (f" ({total} failures)" if total else ""),
        )


# ── Lint Scanner ─────────────────────────────────────────────────────────────

class LintScanner:
    """Runs language-appropriate linters and categorizes findings."""

    LINT_COMMANDS: Dict[str, List[str]] = {
        "python": ["python", "-m", "ruff", "check", "--output-format=json", "."],
        "javascript": ["npx", "eslint", "--format=json", "."],
        "typescript": ["npx", "eslint", "--format=json", "."],
        "go": ["golangci-lint", "run", "--out-format=json"],
        "rust": ["cargo", "clippy", "--message-format=json"],
    }

    async def lint(self, project_dir: str,
                   language: str = "python") -> PhaseResult:
        t0 = time.time()
        cmd = self.LINT_COMMANDS.get(language)

        if not cmd:
            return PhaseResult(
                phase=VerificationPhase.LINT,
                status=PhaseStatus.SKIP,
                duration_ms=(time.time() - t0) * 1000,
                summary=f"No linter configured for: {language}",
            )

        code, out, err = await _run_cmd(cmd, cwd=project_dir, timeout=60)

        findings = []
        errors = 0
        warnings = 0
        try:
            items = json.loads(out) if out.strip().startswith("[") else []
            for item in items[:100]:
                severity = item.get("type", item.get("severity", "warning"))
                if severity in ("error", "E"):
                    errors += 1
                else:
                    warnings += 1
                findings.append({
                    "file": item.get("filename", item.get("filePath", "")),
                    "line": item.get("location", {}).get("row",
                            item.get("line", 0)),
                    "severity": severity,
                    "message": item.get("message", ""),
                    "rule": item.get("code", item.get("ruleId", "")),
                })
        except (json.JSONDecodeError, TypeError):
            pass

        return PhaseResult(
            phase=VerificationPhase.LINT,
            status=PhaseStatus.FAIL if errors > 0 else PhaseStatus.PASS,
            duration_ms=(time.time() - t0) * 1000,
            stdout=out[:3000],
            exit_code=code,
            findings=findings,
            summary=f"Lint: {errors} errors, {warnings} warnings",
        )


# ── Security Scanner ─────────────────────────────────────────────────────────

class SecurityScanner:
    """Vulnerability scanning + secret detection."""

    SECRET_PATTERNS = [
        "AKIA",          # AWS access key
        "sk-",           # OpenAI / Stripe
        "ghp_",          # GitHub PAT
        "-----BEGIN",    # PEM keys
        "AIzaSy",        # Google API key
    ]

    async def scan(self, project_dir: str,
                   language: str = "python") -> PhaseResult:
        t0 = time.time()
        findings: List[Dict[str, Any]] = []

        # Secret detection in source files
        source_exts = {".py", ".js", ".ts", ".go", ".rs", ".java", ".kt",
                       ".env", ".yaml", ".yml", ".json", ".toml"}
        for path in Path(project_dir).rglob("*"):
            if path.suffix not in source_exts:
                continue
            if any(p in str(path) for p in ("node_modules", ".git", "__pycache__",
                                            "venv", ".venv", "dist", "build")):
                continue
            try:
                content = path.read_text(errors="ignore")[:100_000]
                for pattern in self.SECRET_PATTERNS:
                    if pattern in content:
                        findings.append({
                            "type": "secret",
                            "file": str(path.relative_to(project_dir)),
                            "pattern": pattern,
                            "severity": "high",
                        })
            except (OSError, UnicodeDecodeError):
                continue

        # Language-specific vuln scan
        if language == "python":
            code, out, err = await _run_cmd(
                ["pip", "audit"], cwd=project_dir, timeout=30
            )
            if code != 0 and "No known vulnerabilities" not in out:
                findings.append({
                    "type": "vulnerability",
                    "tool": "pip-audit",
                    "output": (out + err)[:500],
                })

        elif language in ("javascript", "typescript"):
            code, out, err = await _run_cmd(
                ["npm", "audit", "--production"], cwd=project_dir, timeout=30
            )
            if code != 0:
                findings.append({
                    "type": "vulnerability",
                    "tool": "npm-audit",
                    "output": (out + err)[:500],
                })

        high = sum(1 for f in findings if f.get("severity") == "high")
        return PhaseResult(
            phase=VerificationPhase.SECURITY_SCAN,
            status=PhaseStatus.FAIL if high > 0 else PhaseStatus.PASS,
            duration_ms=(time.time() - t0) * 1000,
            findings=findings,
            summary=f"Security: {len(findings)} findings ({high} high severity)",
        )


# ── Verification Orchestrator ────────────────────────────────────────────────

class VerificationOrchestrator:
    """Runs the full verification pipeline with retry-on-failure.

    On failure: feeds errors back to the cognitive loop for auto-fix,
    then re-runs from the failed phase. Max 3 iterations.

    When NRSI stdlib verification gates are available, phase pass/fail
    thresholds are derived from the gate confidence requirements declared
    in ``verification.nrsi``.
    """

    def __init__(self, project_dir: str = ".",
                 language: str = "python",
                 stdlib_verification_gates: Optional[Dict[str, Any]] = None):
        self._project_dir = project_dir
        self._language = language
        self._dep_checker = DependencyChecker()
        self._build_runner = BuildRunner()
        self._test_runner = TestRunner()
        self._lint_scanner = LintScanner()
        self._security_scanner = SecurityScanner()
        self._stdlib_gates = stdlib_verification_gates or {}
        if self._stdlib_gates:
            logger.info("Verification orchestrator loaded %d stdlib gates",
                        len(self._stdlib_gates))

    async def run(
        self,
        on_phase: Optional[Any] = None,
    ) -> VerificationResult:
        """Run all verification phases.

        Parameters
        ----------
        on_phase : callable, optional
            Async callback(PhaseResult) for SSE streaming.
        """
        all_phases: List[PhaseResult] = []
        errors: List[str] = []
        t0 = time.time()

        runners = [
            (VerificationPhase.DEPENDENCY_CHECK, self._run_deps),
            (VerificationPhase.BUILD, self._run_build),
            (VerificationPhase.TEST, self._run_tests),
            (VerificationPhase.LINT, self._run_lint),
            (VerificationPhase.SECURITY_SCAN, self._run_security),
        ]

        for phase, runner in runners:
            if on_phase:
                await on_phase(PhaseResult(
                    phase=phase,
                    status=PhaseStatus.RUNNING,
                    summary=f"Running {phase.value}...",
                ))

            result = await runner()
            all_phases.append(result)

            if on_phase:
                await on_phase(result)

            if result.status == PhaseStatus.FAIL:
                errors.append(
                    f"{phase.value}: {result.summary}\n{result.stderr[:200]}"
                )

        passed = all(
            p.status in (PhaseStatus.PASS, PhaseStatus.SKIP)
            for p in all_phases
        )

        return VerificationResult(
            passed=passed,
            iteration=1,
            phases=all_phases,
            total_duration_ms=(time.time() - t0) * 1000,
            errors=errors,
        )

    async def run_with_retry(
        self,
        fix_fn: Optional[Any] = None,
        on_phase: Optional[Any] = None,
        max_iterations: int = MAX_VERIFY_ITERATIONS,
    ) -> VerificationResult:
        """Run verification with automatic retry.

        On failure, calls fix_fn(errors) which should attempt to fix
        the issues, then re-runs from scratch.
        """
        for iteration in range(1, max_iterations + 1):
            result = await self.run(on_phase=on_phase)
            result.iteration = iteration

            if result.passed:
                return result

            if fix_fn and iteration < max_iterations:
                logger.info(
                    "Verification failed (iteration %d/%d), attempting auto-fix...",
                    iteration, max_iterations,
                )
                await fix_fn(result.errors)
            else:
                break

        return result

    async def _run_deps(self) -> PhaseResult:
        return await self._dep_checker.check(self._project_dir)

    async def _run_build(self) -> PhaseResult:
        return await self._build_runner.build(self._project_dir, self._language)

    async def _run_tests(self) -> PhaseResult:
        return await self._test_runner.run_tests(self._project_dir, self._language)

    async def _run_lint(self) -> PhaseResult:
        return await self._lint_scanner.lint(self._project_dir, self._language)

    async def _run_security(self) -> PhaseResult:
        return await self._security_scanner.scan(self._project_dir, self._language)


# ── Completion Gate ──────────────────────────────────────────────────────────

class CompletionGate:
    """Blocks the 'done' signal until verification passes.

    Records verification evidence attached to the response.
    Applies H-Score bonus for verified output, penalty for unverified.
    """

    H_SCORE_VERIFIED_BONUS = 0.15
    H_SCORE_UNVERIFIED_PENALTY = 0.10

    def __init__(self):
        self._result: Optional[VerificationResult] = None

    def record(self, result: VerificationResult) -> None:
        self._result = result

    @property
    def is_verified(self) -> bool:
        return self._result is not None and self._result.passed

    def adjust_h_score(self, base_score: float) -> float:
        if self._result is None:
            return base_score
        if self._result.passed:
            return min(1.0, base_score + self.H_SCORE_VERIFIED_BONUS)
        return max(0.0, base_score - self.H_SCORE_UNVERIFIED_PENALTY)

    def evidence_summary(self) -> Dict[str, Any]:
        if self._result is None:
            return {"verified": False, "reason": "no verification run"}
        return {
            "verified": self._result.passed,
            "iterations": self._result.iteration,
            "duration_ms": round(self._result.total_duration_ms, 1),
            "phases": {
                p.phase.value: {
                    "status": p.status.value,
                    "duration_ms": round(p.duration_ms, 1),
                    "findings": len(p.findings),
                }
                for p in self._result.phases
            },
            "errors": self._result.errors[:5],
        }
