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

"""Tests for framework comparison module."""

import pytest
from victor.evaluation.benchmarks.framework_comparison import (
    Framework,
    FrameworkCapabilities,
    FrameworkResult,
    ComparisonMetrics,
    ComparisonReport,
    FRAMEWORK_CAPABILITIES,
    PUBLISHED_RESULTS,
    compute_metrics_from_result,
    create_comparison_report,
    get_published_result,
)
from victor.evaluation.protocol import BenchmarkType, EvaluationConfig, EvaluationResult, TaskResult, TaskStatus


class TestFramework:
    """Tests for Framework enum."""

    def test_all_frameworks_defined(self):
        """Test that all expected frameworks are defined."""
        expected = {
            "victor",
            "aider",
            "claude_code",
            "cursor",
            "github_copilot",
            "cody",
            "continue",
            "tabby",
            "codegpt",
            "custom",
        }
        actual = {f.value for f in Framework}
        assert expected == actual

    def test_framework_values(self):
        """Test framework enum values."""
        assert Framework.VICTOR.value == "victor"
        assert Framework.AIDER.value == "aider"
        assert Framework.CLAUDE_CODE.value == "claude_code"


class TestFrameworkCapabilities:
    """Tests for FrameworkCapabilities dataclass."""

    def test_required_values(self):
        """Test that required capability values are set."""
        caps = FrameworkCapabilities(name="Test", framework=Framework.CUSTOM)
        assert caps.name == "Test"
        assert caps.framework == Framework.CUSTOM
        assert caps.code_generation is True
        assert caps.code_editing is True
        assert caps.multi_file_editing is False
        assert caps.autonomous_mode is False
        assert caps.open_source is False

    def test_victor_capabilities(self):
        """Test Victor's capabilities are defined correctly."""
        victor_caps = FRAMEWORK_CAPABILITIES[Framework.VICTOR]
        assert victor_caps.code_generation is True
        assert victor_caps.code_editing is True
        assert victor_caps.multi_file_editing is True
        assert victor_caps.tool_use is True
        assert victor_caps.autonomous_mode is True
        assert victor_caps.planning is True
        assert victor_caps.local_models is True
        assert victor_caps.air_gapped is True
        assert victor_caps.mcp_support is True
        assert victor_caps.open_source is True

    def test_all_frameworks_have_capabilities(self):
        """Test that all frameworks have defined capabilities."""
        for framework in [Framework.VICTOR, Framework.AIDER, Framework.CLAUDE_CODE, Framework.CURSOR]:
            assert framework in FRAMEWORK_CAPABILITIES


class TestComparisonMetrics:
    """Tests for ComparisonMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = ComparisonMetrics()
        assert metrics.pass_rate == 0.0
        assert metrics.avg_latency_ms == 0.0
        assert metrics.tokens_per_task == 0.0
        assert metrics.cost_per_task == 0.0
        assert metrics.code_quality_score == 0.0
        assert metrics.test_pass_rate == 0.0
        assert metrics.error_rate == 0.0
        assert metrics.timeout_rate == 0.0


class TestFrameworkResult:
    """Tests for FrameworkResult dataclass."""

    def test_creation(self):
        """Test creating a framework result."""
        result = FrameworkResult(
            framework=Framework.VICTOR,
            benchmark=BenchmarkType.SWE_BENCH,
            model="claude-3-sonnet",
            metrics=ComparisonMetrics(pass_rate=0.45),
        )
        assert result.framework == Framework.VICTOR
        assert result.benchmark == BenchmarkType.SWE_BENCH
        assert result.model == "claude-3-sonnet"
        assert result.metrics.pass_rate == 0.45


class TestPublishedResults:
    """Tests for published benchmark results."""

    def test_swe_bench_results_exist(self):
        """Test that SWE-bench results exist."""
        assert BenchmarkType.SWE_BENCH in PUBLISHED_RESULTS
        results = PUBLISHED_RESULTS[BenchmarkType.SWE_BENCH]
        assert len(results) > 0

    def test_results_have_required_fields(self):
        """Test that results have required fields."""
        for bench_type, results in PUBLISHED_RESULTS.items():
            for framework, data in results.items():
                assert "model" in data
                assert "pass_rate" in data
                assert isinstance(data["pass_rate"], (int, float))
                assert 0 <= data["pass_rate"] <= 1

    def test_get_published_result(self):
        """Test getting a published result."""
        result = get_published_result(BenchmarkType.SWE_BENCH, Framework.CLAUDE_CODE)
        if result:
            assert result["model"] is not None
            assert "pass_rate" in result

    def test_get_published_result_missing(self):
        """Test getting a non-existent result."""
        result = get_published_result(BenchmarkType.SWE_BENCH, Framework.CUSTOM)
        assert result is None


class TestComputeMetrics:
    """Tests for compute_metrics_from_result function."""

    def test_compute_metrics_empty(self):
        """Test computing metrics from empty result."""
        config = EvaluationConfig(
            benchmark=BenchmarkType.HUMAN_EVAL,
            model="test-model",
        )
        result = EvaluationResult(config=config)
        metrics = compute_metrics_from_result(result)
        assert metrics.pass_rate == 0.0

    def test_compute_metrics_with_results(self):
        """Test computing metrics from results."""
        config = EvaluationConfig(
            benchmark=BenchmarkType.HUMAN_EVAL,
            model="test-model",
        )
        task_results = [
            TaskResult(task_id="1", status=TaskStatus.PASSED, tests_passed=5, tests_total=5, duration_seconds=1.0, tokens_used=100),
            TaskResult(task_id="2", status=TaskStatus.PASSED, tests_passed=3, tests_total=3, duration_seconds=2.0, tokens_used=150),
            TaskResult(task_id="3", status=TaskStatus.FAILED, tests_passed=1, tests_total=5, duration_seconds=1.5, tokens_used=120),
        ]
        result = EvaluationResult(config=config)
        result._task_results = task_results
        metrics = compute_metrics_from_result(result)
        assert metrics.pass_rate == result.pass_rate


class TestComparisonReport:
    """Tests for ComparisonReport dataclass."""

    def test_create_report(self):
        """Test creating a comparison report."""
        report = ComparisonReport(
            benchmark=BenchmarkType.SWE_BENCH,
            results=[
                FrameworkResult(
                    framework=Framework.VICTOR,
                    benchmark=BenchmarkType.SWE_BENCH,
                    model="claude-3-sonnet",
                    metrics=ComparisonMetrics(pass_rate=0.45),
                ),
            ],
        )
        assert report.benchmark == BenchmarkType.SWE_BENCH
        assert len(report.results) == 1

    def test_leaderboard(self):
        """Test getting leaderboard from report."""
        report = ComparisonReport(
            benchmark=BenchmarkType.SWE_BENCH,
            results=[
                FrameworkResult(
                    framework=Framework.VICTOR,
                    benchmark=BenchmarkType.SWE_BENCH,
                    model="claude-3-sonnet",
                    metrics=ComparisonMetrics(pass_rate=0.45),
                ),
                FrameworkResult(
                    framework=Framework.AIDER,
                    benchmark=BenchmarkType.SWE_BENCH,
                    model="claude-3-opus",
                    metrics=ComparisonMetrics(pass_rate=0.268),
                ),
            ],
        )
        leaderboard = report.get_leaderboard()
        assert len(leaderboard) == 2
        assert leaderboard[0][0] == Framework.VICTOR  # Higher pass rate first
        assert leaderboard[1][0] == Framework.AIDER

    def test_get_winner(self):
        """Test getting winner from report."""
        report = ComparisonReport(
            benchmark=BenchmarkType.SWE_BENCH,
            results=[
                FrameworkResult(
                    framework=Framework.VICTOR,
                    benchmark=BenchmarkType.SWE_BENCH,
                    model="claude-3-sonnet",
                    metrics=ComparisonMetrics(pass_rate=0.45),
                ),
                FrameworkResult(
                    framework=Framework.AIDER,
                    benchmark=BenchmarkType.SWE_BENCH,
                    model="claude-3-opus",
                    metrics=ComparisonMetrics(pass_rate=0.268),
                ),
            ],
        )
        winner = report.get_winner()
        assert winner == Framework.VICTOR

    def test_to_markdown(self):
        """Test generating markdown from report."""
        report = ComparisonReport(
            benchmark=BenchmarkType.SWE_BENCH,
            results=[
                FrameworkResult(
                    framework=Framework.VICTOR,
                    benchmark=BenchmarkType.SWE_BENCH,
                    model="claude-3-sonnet",
                    metrics=ComparisonMetrics(pass_rate=0.45),
                ),
            ],
        )
        markdown = report.to_markdown()
        assert "Framework Comparison" in markdown
        assert "swe_bench" in markdown
        assert "victor" in markdown

    def test_to_json(self):
        """Test exporting report as JSON."""
        import json
        report = ComparisonReport(
            benchmark=BenchmarkType.SWE_BENCH,
            results=[
                FrameworkResult(
                    framework=Framework.VICTOR,
                    benchmark=BenchmarkType.SWE_BENCH,
                    model="claude-3-sonnet",
                    metrics=ComparisonMetrics(pass_rate=0.45),
                ),
            ],
        )
        json_str = report.to_json()
        data = json.loads(json_str)
        assert data["benchmark"] == "swe_bench"
        assert len(data["results"]) == 1


class TestCreateComparisonReport:
    """Tests for create_comparison_report function."""

    def test_create_report_with_victor_result(self):
        """Test creating report with Victor results."""
        config = EvaluationConfig(
            benchmark=BenchmarkType.SWE_BENCH,
            model="claude-3-sonnet",
        )
        victor_result = EvaluationResult(config=config)
        report = create_comparison_report(BenchmarkType.SWE_BENCH, victor_result)
        assert report.benchmark == BenchmarkType.SWE_BENCH
        # Should have Victor result plus published results
        assert len(report.results) >= 1
        assert any(r.framework == Framework.VICTOR for r in report.results)

    def test_create_report_without_published(self):
        """Test creating report without published results."""
        config = EvaluationConfig(
            benchmark=BenchmarkType.SWE_BENCH,
            model="claude-3-sonnet",
        )
        victor_result = EvaluationResult(config=config)
        report = create_comparison_report(BenchmarkType.SWE_BENCH, victor_result, include_published=False)
        assert len(report.results) == 1
        assert report.results[0].framework == Framework.VICTOR
