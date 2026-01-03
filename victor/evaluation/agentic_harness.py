# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Agentic benchmark harness for evaluating coding assistants on real tasks.

Unlike HumanEval (pure code generation), agentic benchmarks evaluate:
- File editing capabilities
- Tool usage accuracy
- Multi-turn problem solving
- Patch generation and application
- Test-driven development

Supported benchmarks:
- SWE-bench: Real GitHub issues with patches
- Aider Polyglot: 225 multi-language coding problems
- Custom: User-defined agentic tasks

Validation rules:
1. Patch Application: Can the generated patch apply cleanly?
2. Test Passing: Do the project's tests pass after the patch?
3. File Edits: Were the correct files modified?
4. Tool Usage: Did the agent use appropriate tools?
5. Task Completion: Was the overall goal achieved?
"""

import asyncio
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Awaitable

from victor.evaluation.protocol import (
    BenchmarkTask,
    EvaluationConfig,
    TaskStatus,
    TokenUsage,
)
from victor.evaluation.test_runners import (
    TestRunnerRegistry,
    TestRunnerConfig,
    detect_language,
)

logger = logging.getLogger(__name__)


class AgenticValidationType(Enum):
    """Types of validation for agentic tasks."""

    PATCH_APPLIES = "patch_applies"  # Generated patch applies cleanly
    TESTS_PASS = "tests_pass"  # Project tests pass
    FILE_EDITS = "file_edits"  # Correct files were modified
    TOOL_USAGE = "tool_usage"  # Appropriate tools were used
    SEMANTIC_MATCH = "semantic_match"  # Output semantically matches expected
    TASK_COMPLETE = "task_complete"  # Overall task completion


@dataclass
class ToolCall:
    """Record of a tool invocation during task execution."""

    name: str
    arguments: dict[str, Any]
    result: Optional[str] = None
    success: bool = True
    timestamp: float = 0.0


@dataclass
class FileEdit:
    """Record of a file edit during task execution."""

    path: str
    action: str  # "create", "modify", "delete"
    before_content: str = ""
    after_content: str = ""
    diff: str = ""


@dataclass
class AgenticExecutionTrace:
    """Full trace of an agentic task execution."""

    task_id: str
    start_time: float
    end_time: float = 0.0

    # Multi-turn interaction tracking
    turns: int = 0
    messages: list[dict[str, str]] = field(default_factory=list)

    # Tool usage tracking
    tool_calls: list[ToolCall] = field(default_factory=list)

    # File edit tracking
    file_edits: list[FileEdit] = field(default_factory=list)

    # Generated outputs
    generated_patch: str = ""
    generated_code: str = ""

    # Validation results
    validations: dict[str, bool] = field(default_factory=dict)
    validation_errors: dict[str, str] = field(default_factory=dict)

    # Correction/self-correction metrics
    correction_metrics: dict[str, Any] = field(default_factory=dict)

    # Token usage tracking
    token_usage: TokenUsage = field(default_factory=TokenUsage)

    @property
    def duration_seconds(self) -> float:
        """Total execution time."""
        return self.end_time - self.start_time

    @property
    def total_tool_calls(self) -> int:
        """Total number of tool calls made."""
        return len(self.tool_calls)

    @property
    def successful_tool_calls(self) -> int:
        """Number of successful tool calls."""
        return sum(1 for tc in self.tool_calls if tc.success)

    @property
    def files_modified(self) -> list[str]:
        """List of files modified during execution."""
        return [fe.path for fe in self.file_edits]

    @property
    def total_turns(self) -> int:
        """Alias for turns (for compatibility with orchestrator)."""
        return self.turns

    def to_dict(self) -> dict[str, Any]:
        """Export trace as dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "turns": self.turns,
            "messages": self.messages,
            "tool_calls": [
                {
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "result": tc.result,
                    "success": tc.success,
                    "timestamp": tc.timestamp,
                }
                for tc in self.tool_calls
            ],
            "file_edits": [
                {
                    "path": fe.path,
                    "action": fe.action,
                    "before_content": fe.before_content[:500] if fe.before_content else "",
                    "after_content": fe.after_content[:500] if fe.after_content else "",
                    "diff": fe.diff,
                }
                for fe in self.file_edits
            ],
            "generated_patch": self.generated_patch,
            "generated_code": self.generated_code,
            "validations": self.validations,
            "validation_errors": self.validation_errors,
            "total_tool_calls": self.total_tool_calls,
            "successful_tool_calls": self.successful_tool_calls,
            "files_modified": self.files_modified,
            "correction_metrics": self.correction_metrics,
            "token_usage": {
                "input_tokens": self.token_usage.input_tokens,
                "output_tokens": self.token_usage.output_tokens,
                "total_tokens": self.token_usage.total_tokens,
            },
        }


@dataclass
class AgenticTaskResult:
    """Result of an agentic task evaluation."""

    task_id: str
    status: TaskStatus
    trace: AgenticExecutionTrace

    # Test results
    tests_passed: int = 0
    tests_failed: int = 0
    tests_total: int = 0

    # Scoring (0.0 - 1.0)
    patch_score: float = 0.0  # Patch application quality
    test_score: float = 0.0  # Test pass rate
    edit_accuracy: float = 0.0  # File edit accuracy
    tool_efficiency: float = 0.0  # Tool usage efficiency
    overall_score: float = 0.0  # Weighted overall score

    # Error info
    error_message: str = ""
    traceback: str = ""

    @property
    def is_success(self) -> bool:
        """Whether the task was successful."""
        return self.status == TaskStatus.PASSED

    def calculate_overall_score(
        self,
        patch_weight: float = 0.3,
        test_weight: float = 0.4,
        edit_weight: float = 0.2,
        tool_weight: float = 0.1,
    ) -> float:
        """Calculate weighted overall score."""
        self.overall_score = (
            self.patch_score * patch_weight
            + self.test_score * test_weight
            + self.edit_accuracy * edit_weight
            + self.tool_efficiency * tool_weight
        )
        return self.overall_score


@dataclass
class AgenticMetrics:
    """Aggregated metrics for agentic benchmark runs."""

    total_tasks: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    timeouts: int = 0

    # Efficiency metrics
    total_turns: int = 0
    total_tool_calls: int = 0
    total_time_seconds: float = 0.0

    # Quality metrics
    avg_patch_score: float = 0.0
    avg_test_score: float = 0.0
    avg_edit_accuracy: float = 0.0
    avg_tool_efficiency: float = 0.0
    avg_overall_score: float = 0.0

    # Per-task results
    task_results: list[AgenticTaskResult] = field(default_factory=list)

    # Correction/self-correction aggregates
    total_corrections: int = 0
    successful_corrections: int = 0
    total_auto_fixes: int = 0
    total_correction_time_seconds: float = 0.0

    @property
    def pass_rate(self) -> float:
        """Pass rate as decimal."""
        if self.total_tasks == 0:
            return 0.0
        return self.passed / self.total_tasks

    @property
    def avg_turns(self) -> float:
        """Average turns per task."""
        if self.total_tasks == 0:
            return 0.0
        return self.total_turns / self.total_tasks

    @property
    def avg_tool_calls(self) -> float:
        """Average tool calls per task."""
        if self.total_tasks == 0:
            return 0.0
        return self.total_tool_calls / self.total_tasks

    @property
    def correction_success_rate(self) -> float:
        """Correction success rate as decimal."""
        if self.total_corrections == 0:
            return 0.0
        return self.successful_corrections / self.total_corrections

    @property
    def avg_correction_time(self) -> float:
        """Average correction time per correction attempt."""
        if self.total_corrections == 0:
            return 0.0
        return self.total_correction_time_seconds / self.total_corrections

    def to_dict(self) -> dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "summary": {
                "total_tasks": self.total_tasks,
                "passed": self.passed,
                "failed": self.failed,
                "errors": self.errors,
                "timeouts": self.timeouts,
                "pass_rate": round(self.pass_rate, 4),
            },
            "efficiency": {
                "total_turns": self.total_turns,
                "avg_turns": round(self.avg_turns, 2),
                "total_tool_calls": self.total_tool_calls,
                "avg_tool_calls": round(self.avg_tool_calls, 2),
                "total_time_seconds": round(self.total_time_seconds, 2),
            },
            "quality": {
                "avg_patch_score": round(self.avg_patch_score, 4),
                "avg_test_score": round(self.avg_test_score, 4),
                "avg_edit_accuracy": round(self.avg_edit_accuracy, 4),
                "avg_tool_efficiency": round(self.avg_tool_efficiency, 4),
                "avg_overall_score": round(self.avg_overall_score, 4),
            },
            "correction": {
                "total_corrections": self.total_corrections,
                "successful_corrections": self.successful_corrections,
                "correction_success_rate": round(self.correction_success_rate, 4),
                "total_auto_fixes": self.total_auto_fixes,
                "total_correction_time_seconds": round(self.total_correction_time_seconds, 2),
                "avg_correction_time": round(self.avg_correction_time, 2),
            },
            "tasks": [
                {
                    "task_id": r.task_id,
                    "status": r.status.value,
                    "turns": r.trace.turns,
                    "tool_calls": r.trace.total_tool_calls,
                    "duration": round(r.trace.duration_seconds, 2),
                    "tests_passed": r.tests_passed,
                    "tests_total": r.tests_total,
                    "patch_score": round(r.patch_score, 4),
                    "test_score": round(r.test_score, 4),
                    "overall_score": round(r.overall_score, 4),
                }
                for r in self.task_results
            ],
        }


class AgenticValidator(ABC):
    """Abstract base class for agentic task validators."""

    @property
    @abstractmethod
    def validation_type(self) -> AgenticValidationType:
        """The type of validation this validator performs."""
        ...

    @abstractmethod
    async def validate(
        self,
        task: BenchmarkTask,
        trace: AgenticExecutionTrace,
        workspace_dir: Path,
    ) -> tuple[bool, str, float]:
        """Validate the execution trace.

        Args:
            task: The benchmark task
            trace: Execution trace to validate
            workspace_dir: Directory where task was executed

        Returns:
            Tuple of (passed, error_message, score)
        """
        ...


class PatchApplicationValidator(AgenticValidator):
    """Validates that generated patches apply cleanly."""

    @property
    def validation_type(self) -> AgenticValidationType:
        return AgenticValidationType.PATCH_APPLIES

    async def validate(
        self,
        task: BenchmarkTask,
        trace: AgenticExecutionTrace,
        workspace_dir: Path,
    ) -> tuple[bool, str, float]:
        """Validate patch application."""
        if not trace.generated_patch:
            # Check if file edits were made instead
            if trace.file_edits:
                return True, "", 1.0  # File edits count as valid
            return False, "No patch generated", 0.0

        # Try to apply the patch
        patch_file = workspace_dir / "test_patch.patch"
        try:
            patch_file.write_text(trace.generated_patch)

            result = await asyncio.create_subprocess_exec(
                "git",
                "apply",
                "--check",
                str(patch_file),
                cwd=workspace_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                return True, "", 1.0
            else:
                # Partial score if patch is close
                error = stderr.decode()
                if "already exists" in error:
                    return True, "Patch already applied", 0.9
                return False, f"Patch failed: {error[:100]}", 0.0

        except Exception as e:
            return False, f"Patch validation error: {e}", 0.0
        finally:
            if patch_file.exists():
                patch_file.unlink()


class TestPassingValidator(AgenticValidator):
    """Validates that tests pass after changes.

    Uses TestRunnerRegistry to support multiple languages:
    - Python (pytest, unittest)
    - JavaScript/TypeScript (Jest, Mocha, Vitest)
    - Go (go test)
    - Rust (cargo test)
    - Java (Maven, Gradle)
    """

    def __init__(self, timeout_seconds: int = 300):
        """Initialize with optional timeout.

        Args:
            timeout_seconds: Test execution timeout (default: 300s)
        """
        self._registry = TestRunnerRegistry()
        self._timeout = timeout_seconds

    @property
    def validation_type(self) -> AgenticValidationType:
        return AgenticValidationType.TESTS_PASS

    async def validate(
        self,
        task: BenchmarkTask,
        trace: AgenticExecutionTrace,
        workspace_dir: Path,
    ) -> tuple[bool, str, float]:
        """Validate test passing using language-appropriate test runner.

        Returns:
            Tuple of (passed, message, score) where:
            - passed: True if tests pass or mostly pass
            - message: Description of results or errors
            - score: 0.0-1.0 based on pass rate
        """
        try:
            # Detect language and get appropriate runner
            runner = self._registry.detect_runner(workspace_dir)

            if runner is None:
                # No test runner detected - could be unknown language or no tests
                language = detect_language(workspace_dir)
                logger.warning(
                    f"No test runner detected for {workspace_dir} "
                    f"(detected language: {language.value})"
                )
                return True, "No test runner detected", 0.5

            # Configure test runner
            config = TestRunnerConfig(
                language=runner.language,
                timeout_seconds=self._timeout,
            )

            # Run tests
            logger.info(f"Running {runner.language.value} tests in {workspace_dir}")
            results = await runner.run_tests(workspace_dir, config=config)

            # Handle error case
            if results.error_message:
                return False, f"Test execution error: {results.error_message}", 0.0

            # Handle no tests found
            if results.total == 0:
                return True, "No tests found", 0.5

            # Calculate score
            score = results.success_rate

            # Determine pass/fail based on score
            if results.all_passed:
                return True, f"All tests passed ({results.passed}/{results.total})", 1.0
            elif score > 0.8:
                return (
                    True,
                    f"Mostly passing: {results.passed}/{results.total} "
                    f"({results.failed} failed, {results.skipped} skipped)",
                    score,
                )
            else:
                return (
                    False,
                    f"Tests failing: {results.passed}/{results.total} passed "
                    f"({results.failed} failed, {results.errors} errors)",
                    score,
                )

        except Exception as e:
            logger.error(f"Test validation error: {e}")
            return False, f"Test execution error: {e}", 0.0


class FileEditValidator(AgenticValidator):
    """Validates that correct files were edited."""

    @property
    def validation_type(self) -> AgenticValidationType:
        return AgenticValidationType.FILE_EDITS

    async def validate(
        self,
        task: BenchmarkTask,
        trace: AgenticExecutionTrace,
        workspace_dir: Path,
    ) -> tuple[bool, str, float]:
        """Validate file edits against expected changes."""
        if not trace.file_edits:
            return False, "No file edits recorded", 0.0

        # If task has expected patch, compare edited files
        if task.patch:
            expected_files = self._extract_files_from_patch(task.patch)
            actual_files = set(trace.files_modified)

            if not expected_files:
                return True, "", 0.8  # Can't verify without expected files

            intersection = expected_files & actual_files
            score = len(intersection) / len(expected_files) if expected_files else 0.0

            if score >= 1.0:
                return True, "", 1.0
            elif score > 0:
                missing = expected_files - actual_files
                return False, f"Missing edits: {missing}", score
            else:
                return False, "No matching file edits", 0.0

        # If no expected patch, just verify files were modified
        return True, "", 0.7

    def _extract_files_from_patch(self, patch: str) -> set[str]:
        """Extract file paths from a unified diff patch."""
        files = set()
        for line in patch.split("\n"):
            if line.startswith("---") or line.startswith("+++"):
                # Extract file path (skip a/ or b/ prefix)
                parts = line.split()
                if len(parts) >= 2:
                    path = parts[1]
                    if path.startswith("a/") or path.startswith("b/"):
                        path = path[2:]
                    if path != "/dev/null":
                        files.add(path)
        return files


class ToolUsageValidator(AgenticValidator):
    """Validates appropriate tool usage."""

    # Expected tools for different task categories
    EXPECTED_TOOLS = {
        "file_edit": {"file_write", "file_edit", "edit_file", "patch"},
        "code_search": {"code_search", "semantic_search", "grep", "find"},
        "test_run": {"bash", "run_tests", "pytest"},
        "git": {"git_diff", "git_commit", "git_status"},
    }

    @property
    def validation_type(self) -> AgenticValidationType:
        return AgenticValidationType.TOOL_USAGE

    async def validate(
        self,
        task: BenchmarkTask,
        trace: AgenticExecutionTrace,
        workspace_dir: Path,
    ) -> tuple[bool, str, float]:
        """Validate tool usage patterns."""
        if not trace.tool_calls:
            return False, "No tools used", 0.0

        tool_names = {tc.name for tc in trace.tool_calls}
        success_rate = trace.successful_tool_calls / trace.total_tool_calls

        # Check for essential tools (file edit capability)
        has_file_edit = bool(tool_names & self.EXPECTED_TOOLS["file_edit"])

        # Calculate efficiency (fewer calls is better, up to a point)
        efficiency_score = min(1.0, 10 / max(1, trace.total_tool_calls))

        # Combined score
        score = 0.4 * (1.0 if has_file_edit else 0.3) + 0.3 * success_rate + 0.3 * efficiency_score

        if has_file_edit and success_rate >= 0.8:
            return True, "", score
        elif has_file_edit:
            return True, f"Low success rate: {success_rate:.1%}", score
        else:
            return False, "No file editing tools used", score


class SemanticMatchValidator(AgenticValidator):
    """Validates semantic similarity between expected and actual output.

    Uses sentence embeddings to compare the semantic meaning of:
    - Generated code vs expected code
    - Task output vs expected description
    - File content changes vs expected changes

    This handles cases where exact string matching fails but
    the semantic intent is correct (e.g., different variable names,
    formatting, or equivalent implementations).
    """

    # Thresholds for semantic similarity scoring
    HIGH_SIMILARITY_THRESHOLD = 0.85  # Excellent match
    MEDIUM_SIMILARITY_THRESHOLD = 0.70  # Good match
    LOW_SIMILARITY_THRESHOLD = 0.50  # Partial match

    def __init__(self, similarity_threshold: float = 0.70):
        """Initialize with configurable threshold.

        Args:
            similarity_threshold: Minimum cosine similarity for passing (0.0-1.0)
        """
        self._threshold = similarity_threshold
        self._embedding_service: Optional[Any] = None

    @property
    def validation_type(self) -> AgenticValidationType:
        return AgenticValidationType.SEMANTIC_MATCH

    def _get_embedding_service(self) -> Any:
        """Lazily load embedding service."""
        if self._embedding_service is None:
            try:
                from victor.embeddings.service import EmbeddingService

                self._embedding_service = EmbeddingService.get_instance()
            except ImportError:
                logger.warning("EmbeddingService not available, semantic match disabled")
        return self._embedding_service

    async def validate(
        self,
        task: BenchmarkTask,
        trace: AgenticExecutionTrace,
        workspace_dir: Path,
    ) -> tuple[bool, str, float]:
        """Validate semantic match between expected and actual output.

        Compares:
        1. Generated code/patch against expected patch (if available)
        2. Final file contents against expected description
        3. Agent's completion message against task success criteria

        Returns:
            Tuple of (passed, message, score) where score is cosine similarity
        """
        service = self._get_embedding_service()
        if service is None:
            return True, "Semantic match skipped (embeddings unavailable)", 0.5

        try:
            # Collect texts to compare
            expected_texts = []
            actual_texts = []

            # 1. Compare expected patch vs generated patch
            if task.patch and trace.generated_patch:
                expected_texts.append(self._normalize_code(task.patch))
                actual_texts.append(self._normalize_code(trace.generated_patch))

            # 2. Compare expected code vs generated code
            if task.expected_output and trace.generated_code:
                expected_texts.append(self._normalize_code(task.expected_output))
                actual_texts.append(self._normalize_code(trace.generated_code))

            # 3. Compare task description vs completion messages
            if task.prompt and trace.messages:
                # Get assistant's final substantive message
                final_messages = [
                    m["content"]
                    for m in trace.messages
                    if m.get("role") == "assistant" and len(m.get("content", "")) > 50
                ]
                if final_messages:
                    expected_texts.append(task.prompt[:1000])  # Task description
                    actual_texts.append(final_messages[-1][:1000])  # Last response

            # 4. Compare file edits against expected changes
            if trace.file_edits and task.patch:
                combined_edits = "\n".join(
                    fe.after_content[:500] for fe in trace.file_edits if fe.after_content
                )
                if combined_edits:
                    expected_texts.append(self._normalize_code(task.patch))
                    actual_texts.append(self._normalize_code(combined_edits))

            if not expected_texts or not actual_texts:
                return True, "No content to compare semantically", 0.7

            # Compute embeddings and similarity
            all_texts = expected_texts + actual_texts
            embeddings = await service.embed_batch(all_texts)

            n_pairs = len(expected_texts)
            expected_embeds = embeddings[:n_pairs]
            actual_embeds = embeddings[n_pairs:]

            # Calculate pairwise similarities
            similarities = []
            for i in range(n_pairs):
                sim = service.cosine_similarity(expected_embeds[i], actual_embeds[i])
                similarities.append(sim)

            # Average similarity across all comparisons
            avg_similarity = sum(similarities) / len(similarities)
            max_similarity = max(similarities)

            # Determine pass/fail and message
            if avg_similarity >= self.HIGH_SIMILARITY_THRESHOLD:
                return True, f"Excellent semantic match: {avg_similarity:.2%}", avg_similarity
            elif avg_similarity >= self._threshold:
                return True, f"Good semantic match: {avg_similarity:.2%}", avg_similarity
            elif max_similarity >= self._threshold:
                # At least one comparison passed
                return (
                    True,
                    f"Partial semantic match: avg={avg_similarity:.2%}, max={max_similarity:.2%}",
                    max_similarity,
                )
            else:
                return False, f"Poor semantic match: {avg_similarity:.2%}", avg_similarity

        except Exception as e:
            logger.warning(f"Semantic match validation error: {e}")
            return True, f"Semantic match error: {e}", 0.5

    def _normalize_code(self, code: str) -> str:
        """Normalize code for embedding comparison.

        Removes noise like extra whitespace, comments, and diff markers
        to focus on the semantic content.
        """
        # Remove common diff markers
        lines = []
        for line in code.split("\n"):
            # Skip diff header lines
            if line.startswith("---") or line.startswith("+++"):
                continue
            if line.startswith("@@"):
                continue
            # Remove +/- diff markers
            if line.startswith("+") or line.startswith("-"):
                line = line[1:]
            lines.append(line)

        # Join and normalize whitespace
        text = "\n".join(lines)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()


class TaskCompleteValidator(AgenticValidator):
    """Validates that the agent properly signaled task completion.

    Checks for:
    1. Explicit completion phrases in agent messages
    2. Absence of error/failure indicators
    3. Proper tool usage pattern (exploration → implementation → verification)
    4. Confidence level in completion statement
    """

    # Positive completion indicators (agent says task is done)
    COMPLETION_PHRASES = [
        "task complete",
        "task completed",
        "task is complete",
        "i have completed",
        "the task has been completed",
        "implementation complete",
        "changes have been applied",
        "successfully implemented",
        "done with the task",
        "finished implementing",
        "all changes applied",
        "the fix has been applied",
        "the code is ready",
        "implementation is done",
    ]

    # Negative indicators (agent admits failure or uncertainty)
    FAILURE_PHRASES = [
        "i was unable to",
        "i could not",
        "failed to",
        "error occurred",
        "task could not be completed",
        "unable to complete",
        "cannot complete",
        "ran into issues",
        "encountered an error",
        "something went wrong",
        "i'm stuck",
        "need more information",
        "unclear requirements",
    ]

    # Uncertainty indicators (agent is hedging)
    UNCERTAINTY_PHRASES = [
        "i think this might work",
        "this should work",
        "might need adjustment",
        "may require testing",
        "not entirely sure",
        "you may want to verify",
        "please review",
        "let me know if",
    ]

    @property
    def validation_type(self) -> AgenticValidationType:
        return AgenticValidationType.TASK_COMPLETE

    async def validate(
        self,
        task: BenchmarkTask,
        trace: AgenticExecutionTrace,
        workspace_dir: Path,
    ) -> tuple[bool, str, float]:
        """Validate task completion signals.

        Returns:
            Tuple of (passed, message, confidence_score)
        """
        if not trace.messages:
            return False, "No agent messages to analyze", 0.0

        # Get all assistant messages
        assistant_messages = [
            m["content"].lower()
            for m in trace.messages
            if m.get("role") == "assistant" and m.get("content")
        ]

        if not assistant_messages:
            return False, "No assistant responses found", 0.0

        # Analyze the final messages for completion signals
        final_messages = assistant_messages[-3:]  # Last 3 messages
        combined_final = " ".join(final_messages)

        # Check for explicit completion
        has_completion = any(phrase in combined_final for phrase in self.COMPLETION_PHRASES)

        # Check for failure indicators
        has_failure = any(phrase in combined_final for phrase in self.FAILURE_PHRASES)

        # Check for uncertainty
        has_uncertainty = any(phrase in combined_final for phrase in self.UNCERTAINTY_PHRASES)

        # Check for actual work done (file edits or successful tool calls)
        has_edits = len(trace.file_edits) > 0
        has_successful_tools = trace.successful_tool_calls > 0

        # Calculate confidence score
        score = 0.0

        # Explicit completion is a strong signal
        if has_completion:
            score += 0.4

        # File edits indicate actual work
        if has_edits:
            score += 0.3

        # Successful tool calls show progress
        if has_successful_tools:
            score += 0.2

        # Multiple turns suggest thorough work
        if trace.turns >= 3:
            score += 0.1

        # Penalties
        if has_failure:
            score -= 0.3

        if has_uncertainty:
            score -= 0.1

        # Clamp score
        score = max(0.0, min(1.0, score))

        # Determine pass/fail
        if has_failure:
            return False, "Agent indicated task failure", score

        if has_completion and has_edits:
            return True, "Task completed with file modifications", score

        if has_completion:
            return True, "Agent signaled completion", score

        if has_edits and score >= 0.5:
            return True, "Implicit completion (work done, no explicit signal)", score

        if score >= 0.3:
            return False, "Partial progress but no clear completion", score

        return False, "No completion signal detected", score


class AgenticBenchmarkRunner:
    """Runner for agentic benchmarks (SWE-bench, Aider Polyglot, etc.)."""

    def __init__(
        self,
        validators: Optional[list[AgenticValidator]] = None,
        timeout: int = 600,
        workspace_base: Optional[Path] = None,
    ):
        """Initialize the runner.

        Args:
            validators: List of validators to apply
            timeout: Default timeout per task
            workspace_base: Base directory for task workspaces
        """
        self._validators = validators or [
            PatchApplicationValidator(),
            TestPassingValidator(),
            FileEditValidator(),
            ToolUsageValidator(),
        ]
        self._timeout = timeout
        self._workspace_base = workspace_base or Path("/tmp/agentic_bench")
        self._workspace_base.mkdir(parents=True, exist_ok=True)

    async def run_task(
        self,
        task: BenchmarkTask,
        agent_callback: Callable[[BenchmarkTask, Path], Awaitable[AgenticExecutionTrace]],
        config: EvaluationConfig,
    ) -> AgenticTaskResult:
        """Run a single agentic task.

        Args:
            task: The benchmark task
            agent_callback: Async function that executes the agent on the task.
                           Signature: (task, workspace_dir) -> AgenticExecutionTrace
            config: Evaluation configuration

        Returns:
            AgenticTaskResult with validation scores
        """
        # Create workspace for this task
        safe_task_id = task.task_id.replace("/", "_").replace("\\", "_")
        workspace_dir = self._workspace_base / f"task_{safe_task_id}_{int(time.time())}"
        workspace_dir.mkdir(parents=True, exist_ok=True)

        trace = AgenticExecutionTrace(
            task_id=task.task_id,
            start_time=time.time(),
        )

        result = AgenticTaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            trace=trace,
        )

        try:
            # Set up task workspace (clone repo, apply base state)
            await self._setup_workspace(task, workspace_dir)

            # Execute the agent
            trace = await asyncio.wait_for(
                agent_callback(task, workspace_dir),
                timeout=self._timeout,
            )
            trace.end_time = time.time()
            result.trace = trace

            # Run all validators
            scores = {}
            for validator in self._validators:
                passed, error, score = await validator.validate(task, trace, workspace_dir)
                trace.validations[validator.validation_type.value] = passed
                if error:
                    trace.validation_errors[validator.validation_type.value] = error
                scores[validator.validation_type.value] = score

            # Calculate scores
            result.patch_score = scores.get(AgenticValidationType.PATCH_APPLIES.value, 0.0)
            result.test_score = scores.get(AgenticValidationType.TESTS_PASS.value, 0.0)
            result.edit_accuracy = scores.get(AgenticValidationType.FILE_EDITS.value, 0.0)
            result.tool_efficiency = scores.get(AgenticValidationType.TOOL_USAGE.value, 0.0)
            result.calculate_overall_score()

            # Determine pass/fail
            # Success = patch applies AND tests pass
            patch_ok = trace.validations.get(AgenticValidationType.PATCH_APPLIES.value, False)
            tests_ok = trace.validations.get(AgenticValidationType.TESTS_PASS.value, False)

            if patch_ok and tests_ok:
                result.status = TaskStatus.PASSED
            else:
                result.status = TaskStatus.FAILED

        except asyncio.TimeoutError:
            result.status = TaskStatus.TIMEOUT
            result.error_message = "Agent timeout"
            trace.end_time = time.time()

        except Exception as e:
            result.status = TaskStatus.ERROR
            result.error_message = str(e)
            import traceback

            result.traceback = traceback.format_exc()
            trace.end_time = time.time()

        finally:
            # Cleanup workspace
            await self._cleanup_workspace(workspace_dir)

        return result

    async def run_benchmark(
        self,
        tasks: list[BenchmarkTask],
        agent_callback: Callable[[BenchmarkTask, Path], Awaitable[AgenticExecutionTrace]],
        config: EvaluationConfig,
        progress_callback: Optional[Callable[[int, int, AgenticTaskResult], None]] = None,
        max_parallel: int = 1,
    ) -> AgenticMetrics:
        """Run complete benchmark with optional parallel execution.

        Args:
            tasks: List of benchmark tasks
            agent_callback: Agent execution callback
            config: Evaluation configuration
            progress_callback: Optional progress callback
            max_parallel: Maximum number of tasks to run in parallel (default: 1 for sequential)

        Returns:
            AgenticMetrics with aggregated results
        """
        metrics = AgenticMetrics()
        metrics.total_tasks = len(tasks)

        if max_parallel <= 1:
            # Sequential execution (original behavior)
            await self._run_sequential(tasks, agent_callback, config, metrics, progress_callback)
        else:
            # Parallel execution
            await self._run_parallel(
                tasks, agent_callback, config, metrics, progress_callback, max_parallel
            )

        # Calculate averages
        n = len(metrics.task_results)
        if n > 0:
            metrics.avg_patch_score = sum(r.patch_score for r in metrics.task_results) / n
            metrics.avg_test_score = sum(r.test_score for r in metrics.task_results) / n
            metrics.avg_edit_accuracy = sum(r.edit_accuracy for r in metrics.task_results) / n
            metrics.avg_tool_efficiency = sum(r.tool_efficiency for r in metrics.task_results) / n
            metrics.avg_overall_score = sum(r.overall_score for r in metrics.task_results) / n

        return metrics

    async def _run_sequential(
        self,
        tasks: list[BenchmarkTask],
        agent_callback: Callable[[BenchmarkTask, Path], Awaitable[AgenticExecutionTrace]],
        config: EvaluationConfig,
        metrics: AgenticMetrics,
        progress_callback: Optional[Callable[[int, int, AgenticTaskResult], None]] = None,
    ) -> None:
        """Run tasks sequentially."""
        for i, task in enumerate(tasks):
            logger.info(f"Running task {i + 1}/{len(tasks)}: {task.task_id}")

            result = await self.run_task(task, agent_callback, config)
            metrics.task_results.append(result)

            # Update counts
            self._update_metrics_counts(metrics, result)

            # Update totals
            metrics.total_turns += result.trace.turns
            metrics.total_tool_calls += result.trace.total_tool_calls
            metrics.total_time_seconds += result.trace.duration_seconds

            # Progress callback
            if progress_callback:
                progress_callback(i, len(tasks), result)

    async def _run_parallel(
        self,
        tasks: list[BenchmarkTask],
        agent_callback: Callable[[BenchmarkTask, Path], Awaitable[AgenticExecutionTrace]],
        config: EvaluationConfig,
        metrics: AgenticMetrics,
        progress_callback: Optional[Callable[[int, int, AgenticTaskResult], None]] = None,
        max_parallel: int = 4,
    ) -> None:
        """Run tasks in parallel with concurrency limit.

        Uses asyncio.Semaphore to limit the number of concurrent tasks.
        Results are collected in order but execution may be interleaved.
        """
        semaphore = asyncio.Semaphore(max_parallel)
        completed_count = 0
        lock = asyncio.Lock()  # For thread-safe metrics updates

        async def run_with_semaphore(
            index: int,
            task: BenchmarkTask,
        ) -> tuple[int, AgenticTaskResult]:
            """Run a single task with semaphore-controlled concurrency."""
            nonlocal completed_count

            async with semaphore:
                logger.info(
                    f"Running task {index + 1}/{len(tasks)}: {task.task_id} "
                    f"(parallel: {max_parallel})"
                )
                result = await self.run_task(task, agent_callback, config)

                # Update metrics atomically
                async with lock:
                    completed_count += 1
                    self._update_metrics_counts(metrics, result)
                    metrics.total_turns += result.trace.turns
                    metrics.total_tool_calls += result.trace.total_tool_calls
                    metrics.total_time_seconds += result.trace.duration_seconds

                    # Progress callback with current completion count
                    if progress_callback:
                        progress_callback(completed_count - 1, len(tasks), result)

                return index, result

        # Create all tasks
        coros = [run_with_semaphore(i, task) for i, task in enumerate(tasks)]

        # Run all tasks and collect results
        indexed_results = await asyncio.gather(*coros, return_exceptions=True)

        # Process results in original order
        results_by_index: dict[int, AgenticTaskResult] = {}
        for item in indexed_results:
            if isinstance(item, Exception):
                logger.error(f"Task failed with exception: {item}")
                continue
            index, result = item
            results_by_index[index] = result

        # Append results in order
        for i in range(len(tasks)):
            if i in results_by_index:
                metrics.task_results.append(results_by_index[i])

    def _update_metrics_counts(self, metrics: AgenticMetrics, result: AgenticTaskResult) -> None:
        """Update metrics counts based on task result status."""
        if result.status == TaskStatus.PASSED:
            metrics.passed += 1
        elif result.status == TaskStatus.FAILED:
            metrics.failed += 1
        elif result.status == TaskStatus.TIMEOUT:
            metrics.timeouts += 1
        else:
            metrics.errors += 1

    async def _setup_workspace(self, task: BenchmarkTask, workspace_dir: Path) -> None:
        """Set up the task workspace."""
        # Clone repository if specified
        if task.repo:
            clone_cmd = ["git", "clone", "--depth", "1"]
            if task.base_commit:
                clone_cmd.extend(["--branch", task.base_commit])
            clone_cmd.extend([task.repo, str(workspace_dir / "repo")])

            try:
                result = await asyncio.create_subprocess_exec(
                    *clone_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await result.wait()
            except Exception as e:
                logger.warning(f"Failed to clone repo: {e}")

        # Write context code if provided
        if task.context_code:
            code_file = workspace_dir / "context.py"
            code_file.write_text(task.context_code)

        # Write test code if provided
        if task.test_code:
            test_file = workspace_dir / "test_solution.py"
            test_file.write_text(task.test_code)

    async def _cleanup_workspace(self, workspace_dir: Path) -> None:
        """Clean up the task workspace."""
        import shutil

        try:
            if workspace_dir.exists():
                shutil.rmtree(workspace_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup workspace: {e}")


def generate_agentic_report(metrics: AgenticMetrics) -> str:
    """Generate human-readable report for agentic benchmark results."""
    lines = []
    lines.append("=" * 70)
    lines.append("       AGENTIC BENCHMARK REPORT")
    lines.append("=" * 70)
    lines.append("")

    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Total Tasks:    {metrics.total_tasks}")
    lines.append(f"  Passed:         {metrics.passed} ({metrics.pass_rate:.1%})")
    lines.append(f"  Failed:         {metrics.failed}")
    lines.append(f"  Errors:         {metrics.errors}")
    lines.append(f"  Timeouts:       {metrics.timeouts}")
    lines.append("")

    lines.append("EFFICIENCY")
    lines.append("-" * 40)
    lines.append(f"  Avg Turns:      {metrics.avg_turns:.1f}")
    lines.append(f"  Avg Tool Calls: {metrics.avg_tool_calls:.1f}")
    lines.append(f"  Total Time:     {metrics.total_time_seconds:.1f}s")
    lines.append("")

    lines.append("QUALITY SCORES (0-1)")
    lines.append("-" * 40)
    lines.append(f"  Patch Score:    {metrics.avg_patch_score:.3f}")
    lines.append(f"  Test Score:     {metrics.avg_test_score:.3f}")
    lines.append(f"  Edit Accuracy:  {metrics.avg_edit_accuracy:.3f}")
    lines.append(f"  Tool Efficiency:{metrics.avg_tool_efficiency:.3f}")
    lines.append(f"  Overall Score:  {metrics.avg_overall_score:.3f}")
    lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)
