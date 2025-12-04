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
    tool_calls: int = 0
    turns: int = 0

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


@dataclass
class EvaluationResult:
    """Result of a complete evaluation run."""

    config: EvaluationConfig
    task_results: list[TaskResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

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
