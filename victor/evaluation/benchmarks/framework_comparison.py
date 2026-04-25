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

"""Framework comparison benchmarks for AI coding assistants.

Provides standardized comparisons against other AI coding frameworks
including Aider, Claude Code, Cursor, and others.

This module defines:
- Standard metrics for framework comparison
- Leaderboard data structures
- Comparison report generation
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from victor.evaluation.protocol import (
    BenchmarkType,
    EvaluationConfig,
    EvaluationResult,
    LeaderboardEntry,
    get_benchmark_metadata,
)

logger = logging.getLogger(__name__)


class Framework(Enum):
    """Known AI coding frameworks for comparison."""

    VICTOR = "victor"
    AIDER = "aider"
    CLAUDE_CODE = "claude_code"
    CURSOR = "cursor"
    GITHUB_COPILOT = "github_copilot"
    CODY = "cody"
    CONTINUE = "continue"
    TABBY = "tabby"
    CODEGPT = "codegpt"
    CUSTOM = "custom"


@dataclass
class FrameworkCapabilities:
    """Capabilities of a framework for comparison."""

    name: str
    framework: Framework

    # Core capabilities
    code_generation: bool = True
    code_editing: bool = True
    code_review: bool = False
    test_generation: bool = False
    documentation: bool = False
    refactoring: bool = False

    # Agent capabilities
    multi_file_editing: bool = False
    tool_use: bool = False
    autonomous_mode: bool = False
    planning: bool = False

    # Model support
    supported_models: list[str] = field(default_factory=list)
    default_model: str = ""

    # Infrastructure
    local_models: bool = False
    air_gapped: bool = False
    mcp_support: bool = False

    # Pricing/licensing
    open_source: bool = False
    free_tier: bool = False


# Known framework capabilities (based on public information)
FRAMEWORK_CAPABILITIES = {
    Framework.VICTOR: FrameworkCapabilities(
        name="Victor",
        framework=Framework.VICTOR,
        code_generation=True,
        code_editing=True,
        code_review=True,
        test_generation=True,
        documentation=True,
        refactoring=True,
        multi_file_editing=True,
        tool_use=True,
        autonomous_mode=True,
        planning=True,
        supported_models=["claude-3", "gpt-4", "gemini", "ollama/*", "lmstudio/*"],
        default_model="claude-3-sonnet",
        local_models=True,
        air_gapped=True,
        mcp_support=True,
        open_source=True,
        free_tier=True,
    ),
    Framework.AIDER: FrameworkCapabilities(
        name="Aider",
        framework=Framework.AIDER,
        code_generation=True,
        code_editing=True,
        code_review=False,
        test_generation=False,
        documentation=False,
        refactoring=True,
        multi_file_editing=True,
        tool_use=False,
        autonomous_mode=True,
        planning=False,
        supported_models=["claude-3", "gpt-4", "gemini", "ollama/*"],
        default_model="claude-3-sonnet",
        local_models=True,
        air_gapped=False,
        mcp_support=False,
        open_source=True,
        free_tier=True,
    ),
    Framework.CLAUDE_CODE: FrameworkCapabilities(
        name="Claude Code",
        framework=Framework.CLAUDE_CODE,
        code_generation=True,
        code_editing=True,
        code_review=True,
        test_generation=True,
        documentation=True,
        refactoring=True,
        multi_file_editing=True,
        tool_use=True,
        autonomous_mode=True,
        planning=True,
        supported_models=["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
        default_model="claude-3-sonnet",
        local_models=False,
        air_gapped=False,
        mcp_support=True,
        open_source=False,
        free_tier=False,
    ),
    Framework.CURSOR: FrameworkCapabilities(
        name="Cursor",
        framework=Framework.CURSOR,
        code_generation=True,
        code_editing=True,
        code_review=False,
        test_generation=False,
        documentation=False,
        refactoring=True,
        multi_file_editing=True,
        tool_use=False,
        autonomous_mode=False,
        planning=False,
        supported_models=["gpt-4", "claude-3"],
        default_model="gpt-4",
        local_models=False,
        air_gapped=False,
        mcp_support=False,
        open_source=False,
        free_tier=True,
    ),
}


@dataclass
class ComparisonMetrics:
    """Metrics for framework comparison."""

    # Performance
    pass_rate: float = 0.0  # Tasks passed / total
    avg_latency_ms: float = 0.0  # Average response time
    tokens_per_task: float = 0.0  # Average tokens used

    # Quality
    code_quality_score: float = 0.0  # 0-100 scale
    test_pass_rate: float = 0.0  # Test success rate
    partial_completion: float = 0.0  # Partial credit score

    # Efficiency
    cost_per_task: float = 0.0  # Estimated cost
    turns_per_task: float = 0.0  # Conversation turns
    tool_calls_per_task: float = 0.0  # Tool usage

    # Robustness
    error_rate: float = 0.0  # Tasks with errors
    timeout_rate: float = 0.0  # Tasks that timed out


@dataclass
class FrameworkResult:
    """Result of evaluating a framework on a benchmark."""

    framework: Framework
    benchmark: BenchmarkType
    model: str
    metrics: ComparisonMetrics
    timestamp: datetime = field(default_factory=datetime.now)
    config: dict = field(default_factory=dict)

    # Raw results for detailed analysis
    task_results: list[dict] = field(default_factory=list)


@dataclass
class ComparisonReport:
    """Comparison report across multiple frameworks."""

    benchmark: BenchmarkType
    results: list[FrameworkResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def get_leaderboard(self) -> list[tuple[Framework, float]]:
        """Get leaderboard sorted by pass rate."""
        return sorted(
            [(r.framework, r.metrics.pass_rate) for r in self.results],
            key=lambda x: x[1],
            reverse=True,
        )

    def get_winner(self) -> Optional[Framework]:
        """Get the framework with highest pass rate."""
        leaderboard = self.get_leaderboard()
        return leaderboard[0][0] if leaderboard else None

    def to_markdown(self) -> str:
        """Generate markdown comparison table."""
        lines = [
            f"# Framework Comparison: {self.benchmark.value}",
            "",
            f"Generated: {self.timestamp.isoformat()}",
            "",
            "## Leaderboard",
            "",
            "| Rank | Framework | Model | Pass Rate | Avg Latency | Tokens/Task | Cost/Task | Source |",
            "|------|-----------|-------|-----------|-------------|-------------|-----------|--------|",
        ]

        for i, result in enumerate(
            sorted(self.results, key=lambda r: r.metrics.pass_rate, reverse=True)
        ):
            lines.append(
                f"| {i+1} | {result.framework.value} | "
                f"{result.model} | "
                f"{result.metrics.pass_rate:.1%} | "
                f"{result.metrics.avg_latency_ms:.0f}ms | "
                f"{result.metrics.tokens_per_task:.0f} | "
                f"${result.metrics.cost_per_task:.4f} | "
                f"{result.config.get('source', '')} |"
            )

        lines.extend(
            [
                "",
                "## Detailed Metrics",
                "",
            ]
        )

        for result in self.results:
            m = result.metrics
            lines.extend(
                [
                    f"### {result.framework.value}",
                    "",
                    f"- **Model**: {result.model}",
                    f"- **Pass Rate**: {m.pass_rate:.1%}",
                    f"- **Code Quality**: {m.code_quality_score:.1f}/100",
                    f"- **Test Pass Rate**: {m.test_pass_rate:.1%}",
                    f"- **Error Rate**: {m.error_rate:.1%}",
                    f"- **Timeout Rate**: {m.timeout_rate:.1%}",
                    "",
                ]
            )

        return "\n".join(lines)

    def to_json(self) -> str:
        """Export report as JSON."""
        data = {
            "benchmark": self.benchmark.value,
            "timestamp": self.timestamp.isoformat(),
            "results": [
                {
                    "framework": r.framework.value,
                    "model": r.model,
                    "timestamp": r.timestamp.isoformat(),
                    "metrics": {
                        "pass_rate": r.metrics.pass_rate,
                        "avg_latency_ms": r.metrics.avg_latency_ms,
                        "tokens_per_task": r.metrics.tokens_per_task,
                        "code_quality_score": r.metrics.code_quality_score,
                        "cost_per_task": r.metrics.cost_per_task,
                        "test_pass_rate": r.metrics.test_pass_rate,
                        "partial_completion": r.metrics.partial_completion,
                        "error_rate": r.metrics.error_rate,
                        "timeout_rate": r.metrics.timeout_rate,
                    },
                    "config": r.config,
                }
                for r in self.results
            ],
        }
        return json.dumps(data, indent=2)


def compute_metrics_from_result(result: EvaluationResult) -> ComparisonMetrics:
    """Compute comparison metrics from an evaluation result."""
    metrics = ComparisonMetrics()

    if result.total_tasks == 0:
        return metrics

    # Performance
    metrics.pass_rate = result.pass_rate
    metrics.avg_latency_ms = (result.duration_seconds * 1000) / result.total_tasks

    # Calculate average tokens
    total_tokens = sum(r.tokens_used for r in result.task_results)
    metrics.tokens_per_task = total_tokens / result.total_tasks

    # Quality - calculate from task results
    quality_scores = [
        r.code_quality.get_overall_score() for r in result.task_results if r.code_quality
    ]
    if quality_scores:
        metrics.code_quality_score = sum(quality_scores) / len(quality_scores)

    # Test pass rate across all tasks
    total_tests = sum(r.tests_total for r in result.task_results)
    passed_tests = sum(r.tests_passed for r in result.task_results)
    if total_tests > 0:
        metrics.test_pass_rate = passed_tests / total_tests

    # Partial completion
    completion_scores = [r.completion_score for r in result.task_results]
    if completion_scores:
        metrics.partial_completion = sum(completion_scores) / len(completion_scores)

    # Efficiency
    total_turns = sum(r.turns for r in result.task_results)
    metrics.turns_per_task = total_turns / result.total_tasks

    total_tool_calls = sum(r.tool_calls for r in result.task_results)
    metrics.tool_calls_per_task = total_tool_calls / result.total_tasks

    # Estimate cost (rough approximation based on tokens)
    # Assuming ~$0.01 per 1K tokens average
    metrics.cost_per_task = (metrics.tokens_per_task / 1000) * 0.01

    # Robustness
    metrics.error_rate = result.error_tasks / result.total_tasks
    metrics.timeout_rate = result.timeout_tasks / result.total_tasks

    return metrics


def _safe_int(value: Any) -> int:
    """Best-effort integer conversion for saved result ingestion."""
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    """Best-effort float conversion for saved result ingestion."""
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _get_saved_task_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract task result records from supported saved result schemas."""
    tasks = data.get("task_results")
    if isinstance(tasks, list):
        return [task for task in tasks if isinstance(task, dict)]
    tasks = data.get("tasks")
    if isinstance(tasks, list):
        return [task for task in tasks if isinstance(task, dict)]
    return []


def _get_saved_dataset_metadata(data: dict[str, Any]) -> dict[str, Any]:
    """Extract dataset metadata from saved result payloads."""
    dataset_metadata = data.get("dataset_metadata")
    if isinstance(dataset_metadata, dict):
        return dict(dataset_metadata)
    config = data.get("config")
    if isinstance(config, dict) and isinstance(config.get("dataset_metadata"), dict):
        return dict(config["dataset_metadata"])
    return {}


def _resolve_saved_benchmark_type(data: dict[str, Any]) -> BenchmarkType:
    """Resolve a benchmark enum from a saved result payload."""
    benchmark_name = data.get("benchmark")
    if benchmark_name is None:
        config = data.get("config")
        if isinstance(config, dict):
            benchmark_name = config.get("benchmark")
    if benchmark_name is None:
        raise ValueError("Saved benchmark result is missing benchmark metadata")
    metadata = get_benchmark_metadata(str(benchmark_name))
    if metadata is None:
        raise ValueError(f"Unknown benchmark in saved result: {benchmark_name}")
    return metadata.type


def compute_metrics_from_saved_result(data: dict[str, Any]) -> ComparisonMetrics:
    """Compute comparison metrics from a saved benchmark JSON artifact."""
    metrics = ComparisonMetrics()
    summary: dict[str, Any] = {}
    for key in ("summary", "metrics"):
        candidate = data.get(key)
        if isinstance(candidate, dict):
            summary.update(candidate)

    task_results = _get_saved_task_results(data)
    total_tasks = _safe_int(summary.get("total_tasks")) or len(task_results)
    if total_tasks <= 0:
        return metrics

    passed_tasks = sum(
        1 for task in task_results if str(task.get("status", "")).lower() == "passed"
    )
    error_tasks = sum(1 for task in task_results if str(task.get("status", "")).lower() == "error")
    timeout_tasks = sum(
        1 for task in task_results if str(task.get("status", "")).lower() == "timeout"
    )

    if summary.get("pass_rate") is not None:
        metrics.pass_rate = _safe_float(summary.get("pass_rate"))
    else:
        metrics.pass_rate = passed_tasks / total_tasks

    duration_seconds = _safe_float(summary.get("duration_seconds"))
    if duration_seconds > 0:
        metrics.avg_latency_ms = (duration_seconds * 1000) / total_tasks
    elif summary.get("avg_duration_per_task") is not None:
        metrics.avg_latency_ms = _safe_float(summary.get("avg_duration_per_task")) * 1000
    else:
        task_duration = sum(
            _safe_float(task.get("duration_seconds") or task.get("duration"))
            for task in task_results
        )
        metrics.avg_latency_ms = (task_duration * 1000) / total_tasks if task_duration else 0.0

    total_tokens = _safe_float(summary.get("total_tokens"))
    if total_tokens > 0:
        metrics.tokens_per_task = total_tokens / total_tasks
    elif summary.get("avg_tokens_per_task") is not None:
        metrics.tokens_per_task = _safe_float(summary.get("avg_tokens_per_task"))
    else:
        metrics.tokens_per_task = (
            sum(_safe_float(task.get("tokens_used")) for task in task_results) / total_tasks
        )

    total_tests = sum(_safe_int(task.get("tests_total")) for task in task_results)
    passed_tests = sum(_safe_int(task.get("tests_passed")) for task in task_results)
    if total_tests > 0:
        metrics.test_pass_rate = passed_tests / total_tests

    completion_scores = [
        _safe_float(task.get("completion_score"))
        for task in task_results
        if task.get("completion_score") is not None
    ]
    if completion_scores:
        metrics.partial_completion = sum(completion_scores) / len(completion_scores)

    quality_scores = []
    for task in task_results:
        code_quality = task.get("code_quality")
        if isinstance(code_quality, dict) and code_quality.get("overall_score") is not None:
            quality_scores.append(_safe_float(code_quality.get("overall_score")))
    if quality_scores:
        metrics.code_quality_score = sum(quality_scores) / len(quality_scores)

    total_turns = _safe_float(summary.get("total_turns"))
    if total_turns > 0:
        metrics.turns_per_task = total_turns / total_tasks
    else:
        metrics.turns_per_task = (
            sum(_safe_float(task.get("turns")) for task in task_results) / total_tasks
        )

    total_tool_calls = _safe_float(summary.get("total_tool_calls"))
    if total_tool_calls > 0:
        metrics.tool_calls_per_task = total_tool_calls / total_tasks
    else:
        metrics.tool_calls_per_task = (
            sum(_safe_float(task.get("tool_calls")) for task in task_results) / total_tasks
        )

    total_cost_micros = _safe_float(summary.get("cost_usd_micros"))
    if total_cost_micros > 0:
        metrics.cost_per_task = (total_cost_micros / 1_000_000) / total_tasks
    elif metrics.tokens_per_task > 0:
        metrics.cost_per_task = (metrics.tokens_per_task / 1000) * 0.01

    summary_errors = summary.get("errors")
    summary_timeouts = summary.get("timeouts")
    metrics.error_rate = (
        _safe_float(summary_errors) / total_tasks
        if summary_errors is not None
        else error_tasks / total_tasks
    )
    metrics.timeout_rate = (
        _safe_float(summary_timeouts) / total_tasks
        if summary_timeouts is not None
        else timeout_tasks / total_tasks
    )

    return metrics


def load_framework_result_from_file(
    path: Path,
    framework: Framework = Framework.VICTOR,
    model_override: Optional[str] = None,
) -> FrameworkResult:
    """Load a saved benchmark artifact into a FrameworkResult."""
    with open(path) as f:
        data = json.load(f)

    benchmark = _resolve_saved_benchmark_type(data)
    config = data.get("config") if isinstance(data.get("config"), dict) else {}
    dataset_metadata = _get_saved_dataset_metadata(data)
    resolved_model = model_override or data.get("model") or config.get("model") or "unknown"
    source_name = (
        dataset_metadata.get("source_name") or config.get("source") or f"Local result ({path.name})"
    )
    timestamp_value = data.get("timestamp") or data.get("end_time") or data.get("start_time")
    try:
        timestamp = (
            datetime.fromisoformat(str(timestamp_value)) if timestamp_value else datetime.now()
        )
    except ValueError:
        timestamp = datetime.now()

    framework_config: dict[str, Any] = {
        "artifact_path": str(path),
        "source": source_name,
        "dataset_metadata": dataset_metadata,
    }
    if config.get("max_tasks") is not None:
        framework_config["max_tasks"] = config.get("max_tasks")

    return FrameworkResult(
        framework=framework,
        benchmark=benchmark,
        model=str(resolved_model),
        metrics=compute_metrics_from_saved_result(data),
        timestamp=timestamp,
        config=framework_config,
        task_results=_get_saved_task_results(data),
    )


# External benchmark results for comparison (from published data)
PUBLISHED_RESULTS: dict[BenchmarkType, dict[Framework, dict[str, Any]]] = {
    BenchmarkType.SWE_BENCH: {
        # SWE-bench Lite results as of late 2024
        Framework.CLAUDE_CODE: {
            "model": "claude-3-opus",
            "pass_rate": 0.49,  # ~49% on SWE-bench Lite
            "source": "Anthropic blog post, 2024",
        },
        Framework.AIDER: {
            "model": "claude-3-opus + gpt-4",
            "pass_rate": 0.268,  # 26.8% on SWE-bench Lite
            "source": "Aider leaderboard, 2024",
        },
    },
    BenchmarkType.HUMAN_EVAL: {
        # HumanEval pass@1 results
        Framework.CLAUDE_CODE: {
            "model": "claude-3-opus",
            "pass_rate": 0.846,
            "source": "Anthropic technical report",
        },
    },
}


def get_published_result(
    benchmark: BenchmarkType,
    framework: Framework,
) -> Optional[dict[str, Any]]:
    """Get published benchmark result for a framework."""
    if benchmark in PUBLISHED_RESULTS:
        if framework in PUBLISHED_RESULTS[benchmark]:
            return PUBLISHED_RESULTS[benchmark][framework]
    return None


def create_comparison_report(
    benchmark: BenchmarkType,
    victor_result: EvaluationResult,
    include_published: bool = True,
) -> ComparisonReport:
    """Create a comparison report with Victor results and published data."""
    report = ComparisonReport(benchmark=benchmark)

    # Add Victor result
    victor_metrics = compute_metrics_from_result(victor_result)
    victor_framework_result = FrameworkResult(
        framework=Framework.VICTOR,
        benchmark=benchmark,
        model=victor_result.config.model,
        metrics=victor_metrics,
        config={"max_tasks": victor_result.config.max_tasks},
    )
    report.results.append(victor_framework_result)

    # Add published results for comparison
    if include_published and benchmark in PUBLISHED_RESULTS:
        for framework, data in PUBLISHED_RESULTS[benchmark].items():
            metrics = ComparisonMetrics(pass_rate=data.get("pass_rate", 0.0))
            result = FrameworkResult(
                framework=framework,
                benchmark=benchmark,
                model=data.get("model", "unknown"),
                metrics=metrics,
                config={"source": data.get("source", "published")},
            )
            report.results.append(result)

    return report


def create_comparison_report_from_saved_result(
    path: Path,
    framework: Framework = Framework.VICTOR,
    include_published: bool = True,
) -> ComparisonReport:
    """Create a comparison report from a saved benchmark JSON artifact."""
    framework_result = load_framework_result_from_file(path, framework=framework)
    report = ComparisonReport(benchmark=framework_result.benchmark)
    report.results.append(framework_result)

    if include_published and framework_result.benchmark in PUBLISHED_RESULTS:
        for published_framework, data in PUBLISHED_RESULTS[framework_result.benchmark].items():
            metrics = ComparisonMetrics(pass_rate=data.get("pass_rate", 0.0))
            result = FrameworkResult(
                framework=published_framework,
                benchmark=framework_result.benchmark,
                model=data.get("model", "unknown"),
                metrics=metrics,
                config={"source": data.get("source", "published")},
            )
            report.results.append(result)

    return report


def create_quick_comparison(
    benchmark: BenchmarkType = BenchmarkType.SWE_BENCH,
    victor_pass_rate: float = 0.0,
    victor_model: str = "claude-3-sonnet",
    include_published: bool = True,
) -> ComparisonReport:
    """Create a comparison report without running actual benchmarks.

    Useful for testing the comparison pipeline or generating reports
    from pre-computed pass rates.

    Args:
        benchmark: Benchmark type for the report
        victor_pass_rate: Victor's pass rate to use
        victor_model: Model name for Victor's entry
        include_published: Include published competitor data

    Returns:
        ComparisonReport with Victor and (optionally) published results
    """
    report = ComparisonReport(benchmark=benchmark)

    victor_metrics = ComparisonMetrics(pass_rate=victor_pass_rate)
    victor_result = FrameworkResult(
        framework=Framework.VICTOR,
        benchmark=benchmark,
        model=victor_model,
        metrics=victor_metrics,
    )
    report.results.append(victor_result)

    if include_published and benchmark in PUBLISHED_RESULTS:
        for framework, data in PUBLISHED_RESULTS[benchmark].items():
            metrics = ComparisonMetrics(pass_rate=data.get("pass_rate", 0.0))
            result = FrameworkResult(
                framework=framework,
                benchmark=benchmark,
                model=data.get("model", "unknown"),
                metrics=metrics,
                config={"source": data.get("source", "published")},
            )
            report.results.append(result)

    return report
