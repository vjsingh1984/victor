# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation protocol types for benchmark harnesses.

Defines data structures for running evaluations against
benchmarks like SWE-bench, HumanEval, MBPP, etc.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


@dataclass
class TokenUsage:
    """Token usage metrics for evaluation tracking.

    A minimal interface for token data needed by evaluations (ISP compliance).
    Supports addition for aggregation across turns.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage instances for aggregation."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def __iadd__(self, other: "TokenUsage") -> "TokenUsage":
        """In-place addition for aggregation."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        return self


class BenchmarkType(Enum):
    """Types of coding benchmarks."""

    SWE_BENCH = "swe_bench"  # Software engineering tasks
    HUMAN_EVAL = "human_eval"  # Code generation
    MBPP = "mbpp"  # Basic programming
    LIVE_CODE_BENCH = "live_code_bench"  # Live coding evaluation
    BIG_CODE_BENCH = "big_code_bench"  # Large-scale code tasks
    AIDER_POLYGLOT = "aider_polyglot"  # Multi-language tasks
    CUSTOM = "custom"  # User-defined benchmark


class TaskStatus(Enum):
    """Status of a benchmark task."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class EvaluationMetric(Enum):
    """Standard evaluation metrics."""

    PASS_AT_K = "pass@k"  # Pass rate at k attempts
    EXACT_MATCH = "exact_match"  # Exact solution match
    PARTIAL_MATCH = "partial_match"  # Partial solution match
    TEST_PASS_RATE = "test_pass_rate"  # Test passing rate
    EDIT_DISTANCE = "edit_distance"  # Code edit distance
    BLEU = "bleu"  # BLEU score for code
    CODE_BLEU = "code_bleu"  # CodeBLEU score
    SYNTAX_VALID = "syntax_valid"  # Syntactically valid
    SEMANTIC_EQUIV = "semantic_equiv"  # Semantically equivalent


@dataclass
class BenchmarkTask:
    """A single task in a benchmark."""

    task_id: str
    benchmark: BenchmarkType
    description: str
    language: str = "python"

    # Input context
    prompt: str = ""
    context_code: str = ""
    test_code: str = ""

    # Repository info (for SWE-bench)
    repo: Optional[str] = None
    base_commit: Optional[str] = None
    issue_text: Optional[str] = None
    hints: list[str] = field(default_factory=list)

    # Expected solution
    solution: Optional[str] = None
    patch: Optional[str] = None

    # Metadata
    difficulty: str = "medium"
    category: str = ""
    tags: list[str] = field(default_factory=list)
    timeout_seconds: int = 300


@dataclass
class CodeQualityMetrics:
    """Code quality metrics for generated code."""

    # Syntax and style
    syntax_valid: bool = True
    lint_errors: int = 0
    lint_warnings: int = 0
    style_score: float = 1.0  # 0.0 to 1.0

    # Complexity
    cyclomatic_complexity: float = 0.0
    cognitive_complexity: float = 0.0
    maintainability_index: float = 100.0  # 0-100 scale

    # Structure
    lines_of_code: int = 0
    functions_count: int = 0
    classes_count: int = 0
    duplicate_lines: int = 0

    # Type safety
    type_coverage: float = 0.0  # For typed languages

    def get_overall_score(self) -> float:
        """Calculate overall code quality score (0-100)."""
        weights = {
            "syntax": 0.3,
            "style": 0.2,
            "complexity": 0.25,
            "maintainability": 0.25,
        }

        syntax_score = 100.0 if self.syntax_valid else 0.0
        style_score = self.style_score * 100
        complexity_score = max(0, 100 - (self.cyclomatic_complexity * 5))
        maint_score = self.maintainability_index

        return (
            weights["syntax"] * syntax_score
            + weights["style"] * style_score
            + weights["complexity"] * complexity_score
            + weights["maintainability"] * maint_score
        )


@dataclass
class TaskResult:
    """Result of running a single task."""

    task_id: str
    status: TaskStatus
    generated_code: str = ""
    generated_patch: str = ""

    # Test results
    tests_passed: int = 0
    tests_failed: int = 0
    tests_total: int = 0

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Agent metrics
    tokens_used: int = 0
    tokens_input: int = 0  # Input tokens (prompt)
    tokens_output: int = 0  # Output tokens (completion)
    tool_calls: int = 0
    turns: int = 0

    # Code quality metrics
    code_quality: Optional[CodeQualityMetrics] = None

    # Partial completion scoring (0.0 to 1.0)
    completion_score: float = 0.0
    partial_tests_weight: float = 0.0  # Weighted partial test success

    # Pass@k tracking
    attempts: int = 1
    successful_attempts: int = 0

    # Error info
    error_message: str = ""
    traceback: str = ""

    # Detailed output
    stdout: str = ""
    stderr: str = ""

    @property
    def test_pass_rate(self) -> float:
        """Calculate test pass rate."""
        if self.tests_total == 0:
            return 0.0
        return self.tests_passed / self.tests_total

    @property
    def is_success(self) -> bool:
        """Whether the task was successful."""
        return self.status == TaskStatus.PASSED

    @property
    def tokens_per_test(self) -> float:
        """Token efficiency: tokens used per test passed."""
        if self.tests_passed == 0:
            return float("inf")
        return self.tokens_used / self.tests_passed

    @property
    def cost_efficiency(self) -> float:
        """Cost efficiency score (higher is better)."""
        if self.tokens_used == 0:
            return 0.0
        # Score based on success per 1K tokens
        return (self.completion_score * 1000) / self.tokens_used

    def calculate_completion_score(self) -> float:
        """Calculate partial completion score based on tests and quality."""
        score = 0.0

        # Test-based score (60% weight)
        if self.tests_total > 0:
            test_score = self.tests_passed / self.tests_total
            score += test_score * 0.6

        # Code quality score (20% weight)
        if self.code_quality:
            quality_score = self.code_quality.get_overall_score() / 100
            score += quality_score * 0.2

        # Basic completion (20% weight) - did it generate valid code?
        if self.generated_code and self.status != TaskStatus.ERROR:
            score += 0.2

        return min(score, 1.0)


@dataclass
class EvaluationConfig:
    """Configuration for an evaluation run."""

    benchmark: BenchmarkType
    model: str
    max_tasks: Optional[int] = None
    timeout_per_task: int = 300
    max_retries: int = 1
    parallel_tasks: int = 1

    # Filtering
    task_ids: Optional[list[str]] = None
    languages: Optional[list[str]] = None
    categories: Optional[list[str]] = None
    difficulties: Optional[list[str]] = None

    # Agent config
    temperature: float = 0.0
    max_tokens: int = 4096
    max_turns: int = 10

    # Environment
    use_docker: bool = True
    docker_image: str = "python:3.11"
    workspace_dir: Optional[Path] = None

    # Self-correction settings (generic iterative refinement)
    enable_self_correction: bool = False
    self_correction_max_iterations: int = 3
    auto_fix_imports: bool = True


@dataclass
class EvaluationResult:
    """Result of a complete evaluation run."""

    config: EvaluationConfig
    task_results: list[TaskResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Self-correction metrics (populated when enable_self_correction=True)
    correction_metrics: Optional[dict[str, Any]] = None

    @property
    def total_tasks(self) -> int:
        """Total number of tasks evaluated."""
        return len(self.task_results)

    @property
    def passed_tasks(self) -> int:
        """Number of passed tasks."""
        return sum(1 for r in self.task_results if r.status == TaskStatus.PASSED)

    @property
    def failed_tasks(self) -> int:
        """Number of failed tasks."""
        return sum(1 for r in self.task_results if r.status == TaskStatus.FAILED)

    @property
    def error_tasks(self) -> int:
        """Number of tasks with errors."""
        return sum(1 for r in self.task_results if r.status == TaskStatus.ERROR)

    @property
    def timeout_tasks(self) -> int:
        """Number of timed out tasks."""
        return sum(1 for r in self.task_results if r.status == TaskStatus.TIMEOUT)

    @property
    def pass_rate(self) -> float:
        """Overall pass rate."""
        if self.total_tasks == 0:
            return 0.0
        return self.passed_tasks / self.total_tasks

    @property
    def duration_seconds(self) -> float:
        """Total evaluation duration."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return sum(r.duration_seconds for r in self.task_results)

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return sum(r.tokens_used for r in self.task_results)

    @property
    def total_tool_calls(self) -> int:
        """Total tool calls made."""
        return sum(r.tool_calls for r in self.task_results)

    def get_by_status(self, status: TaskStatus) -> list[TaskResult]:
        """Get results by status."""
        return [r for r in self.task_results if r.status == status]

    def get_by_language(self, language: str) -> list[TaskResult]:
        """Get results by language (requires task metadata)."""
        # Would need task_id -> task mapping
        return self.task_results

    def get_metrics(self) -> dict[str, Any]:
        """Get summary metrics."""
        return {
            "total_tasks": self.total_tasks,
            "passed": self.passed_tasks,
            "failed": self.failed_tasks,
            "errors": self.error_tasks,
            "timeouts": self.timeout_tasks,
            "pass_rate": self.pass_rate,
            "duration_seconds": self.duration_seconds,
            "total_tokens": self.total_tokens,
            "total_tool_calls": self.total_tool_calls,
            "avg_tokens_per_task": self.total_tokens / max(1, self.total_tasks),
            "avg_duration_per_task": self.duration_seconds / max(1, self.total_tasks),
        }


@dataclass
class BenchmarkMetadata:
    """Metadata about a benchmark dataset."""

    name: str
    type: BenchmarkType
    version: str
    total_tasks: int
    languages: list[str]
    categories: list[str]
    description: str = ""
    source_url: str = ""
    paper_url: str = ""


@dataclass
class LeaderboardEntry:
    """Entry for evaluation leaderboard."""

    model: str
    benchmark: str
    pass_rate: float
    timestamp: datetime
    total_tasks: int
    passed_tasks: int
    avg_tokens: float
    avg_duration: float
    config: dict = field(default_factory=dict)
