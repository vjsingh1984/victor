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

"""Unit tests for comprehensive benchmark suite.

Tests the benchmark framework without running actual benchmarks
to keep unit tests fast.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts to path for testing
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from benchmark_comprehensive import (
    BenchmarkMetric,
    BenchmarkResult,
    BenchmarkReport,
    StartupBenchmark,
    MemoryBenchmark,
    ThroughputBenchmark,
    LatencyBenchmark,
    BenchmarkRunner,
    MarkdownFormatter,
    JsonFormatter,
    HtmlFormatter,
)


# =============================================================================
# BenchmarkMetric Tests
# =============================================================================


class TestBenchmarkMetric:
    """Tests for BenchmarkMetric data class."""

    def test_metric_creation(self):
        """Test creating a benchmark metric."""
        metric = BenchmarkMetric(
            name="startup_time",
            value=150.5,
            unit="ms",
            description="Time to import victor module",
            threshold=200.0,
        )

        assert metric.name == "startup_time"
        assert metric.value == 150.5
        assert metric.unit == "ms"
        assert metric.description == "Time to import victor module"
        assert metric.threshold == 200.0

    def test_metric_to_dict(self):
        """Test converting metric to dictionary."""
        metric = BenchmarkMetric(
            name="startup_time",
            value=150.5,
            unit="ms",
            threshold=200.0,
        )

        data = metric.to_dict()

        assert data["name"] == "startup_time"
        assert data["value"] == 150.5
        assert data["unit"] == "ms"
        assert data["threshold"] == 200.0


# =============================================================================
# BenchmarkResult Tests
# =============================================================================


class TestBenchmarkResult:
    """Tests for BenchmarkResult data class."""

    def test_result_creation(self):
        """Test creating a benchmark result."""
        result = BenchmarkResult(
            name="Cold Start",
            category="startup",
        )

        assert result.name == "Cold Start"
        assert result.category == "startup"
        assert result.passed is True
        assert len(result.metrics) == 0

    def test_add_metric_passing(self):
        """Test adding a passing metric."""
        result = BenchmarkResult(name="Test", category="test")

        result.add_metric(
            name="latency",
            value=1.5,
            unit="ms",
            threshold=2.0,
        )

        assert result.passed is True
        assert len(result.metrics) == 1
        assert result.metrics[0].name == "latency"

    def test_add_metric_failing(self):
        """Test adding a failing metric (regression)."""
        result = BenchmarkResult(name="Test", category="test")

        result.add_metric(
            name="latency",
            value=5.0,
            unit="ms",
            threshold=2.0,
        )

        assert result.passed is False
        assert len(result.metrics) == 1

    def test_add_metric_throughput_higher_is_better(self):
        """Test that higher throughput is better."""
        result = BenchmarkResult(name="Test", category="test")

        # Throughput above threshold should pass
        result.add_metric(
            name="throughput",
            value=1500,
            unit="ops/sec",
            threshold=1000,
        )

        assert result.passed is True

    def test_add_metric_throughput_lower_is_worse(self):
        """Test that lower throughput is worse (regression)."""
        result = BenchmarkResult(name="Test", category="test")

        # Throughput below threshold should fail
        result.add_metric(
            name="throughput",
            value=500,
            unit="ops/sec",
            threshold=1000,
        )

        assert result.passed is False

    def test_get_metric(self):
        """Test getting a metric by name."""
        result = BenchmarkResult(name="Test", category="test")

        result.add_metric("latency", 1.5, "ms")
        result.add_metric("throughput", 1000, "ops/sec")

        latency = result.get_metric("latency")
        assert latency is not None
        assert latency.value == 1.5

        missing = result.get_metric("memory")
        assert missing is None

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = BenchmarkResult(
            name="Cold Start",
            category="startup",
            passed=True,
        )

        result.add_metric("startup_time", 150.5, "ms", threshold=200.0)

        data = result.to_dict()

        assert data["name"] == "Cold Start"
        assert data["category"] == "startup"
        assert data["passed"] is True
        assert len(data["metrics"]) == 1


# =============================================================================
# BenchmarkReport Tests
# =============================================================================


class TestBenchmarkReport:
    """Tests for BenchmarkReport data class."""

    def test_report_creation(self):
        """Test creating a benchmark report."""
        report = BenchmarkReport()

        assert len(report.results) == 0
        assert report.get_passed_count() == 0
        assert report.get_failed_count() == 0

    def test_add_result(self):
        """Test adding results to report."""
        report = BenchmarkReport()

        result1 = BenchmarkResult(name="Test1", category="test")
        result1.passed = True

        result2 = BenchmarkResult(name="Test2", category="test")
        result2.passed = False

        report.add_result(result1)
        report.add_result(result2)

        assert len(report.results) == 2
        assert report.get_passed_count() == 1
        assert report.get_failed_count() == 1

    def test_to_dict(self):
        """Test converting report to dictionary."""
        report = BenchmarkReport()

        result = BenchmarkResult(name="Test", category="test")
        result.add_metric("latency", 1.5, "ms")

        report.add_result(result)
        report.end_time = datetime.now()

        data = report.to_dict()

        assert "start_time" in data
        assert "end_time" in data
        assert "summary" in data
        assert data["summary"]["total"] == 1


# =============================================================================
# BenchmarkRunner Tests
# =============================================================================


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    def test_runner_initialization(self, tmp_path):
        """Test initializing benchmark runner."""
        runner = BenchmarkRunner(project_root=tmp_path)

        assert runner.project_root == tmp_path
        assert runner.results_dir == tmp_path / ".benchmark_results"

    def test_runner_creates_results_dir(self, tmp_path):
        """Test that runner creates results directory."""
        runner = BenchmarkRunner(project_root=tmp_path)
        _ = runner.run(scenario=None, save=False)

        assert runner.results_dir.exists()

    @patch("benchmark_comprehensive.StartupBenchmark")
    def test_run_startup_benchmarks(self, mock_startup_benchmark, tmp_path):
        """Test running startup benchmarks."""
        # Mock startup benchmark
        mock_instance = MagicMock()
        mock_instance.run_cold_start.return_value = BenchmarkResult(
            name="Cold Start", category="startup"
        )
        mock_instance.run_warm_start.return_value = BenchmarkResult(
            name="Warm Start", category="startup"
        )
        mock_instance.run_bootstrap_time.return_value = BenchmarkResult(
            name="Bootstrap Time", category="startup"
        )
        mock_startup_benchmark.return_value = mock_instance

        runner = BenchmarkRunner(project_root=tmp_path)
        report = runner.run(scenario="startup", save=False)

        assert len(report.results) == 3
        assert report.results[0].name == "Cold Start"
        assert report.results[1].name == "Warm Start"
        assert report.results[2].name == "Bootstrap Time"

    @patch("benchmark_comprehensive.MemoryBenchmark")
    def test_run_memory_benchmarks(self, mock_memory_benchmark, tmp_path):
        """Test running memory benchmarks."""
        # Mock memory benchmark
        mock_instance = MagicMock()
        mock_instance.run_baseline_memory.return_value = BenchmarkResult(
            name="Baseline Memory", category="memory"
        )
        mock_instance.run_memory_leak_detection.return_value = BenchmarkResult(
            name="Memory Leak Detection", category="memory"
        )
        mock_memory_benchmark.return_value = mock_instance

        runner = BenchmarkRunner(project_root=tmp_path)
        report = runner.run(scenario="memory", save=False)

        assert len(report.results) == 2
        assert all(r.category == "memory" for r in report.results)


# =============================================================================
# Formatter Tests
# =============================================================================


class TestMarkdownFormatter:
    """Tests for MarkdownFormatter."""

    def test_format_empty_report(self):
        """Test formatting empty report."""
        formatter = MarkdownFormatter()
        report = BenchmarkReport()

        output = formatter.format(report)

        assert "# Victor AI Performance Benchmark Report" in output
        assert "Executive Summary" in output
        assert "**Total Benchmarks:** 0" in output

    def test_format_report_with_results(self):
        """Test formatting report with results."""
        formatter = MarkdownFormatter()
        report = BenchmarkReport()

        result = BenchmarkResult(name="Test", category="test")
        result.add_metric("latency", 1.5, "ms", threshold=2.0)
        report.add_result(result)

        output = formatter.format(report)

        assert "# Victor AI Performance Benchmark Report" in output
        assert "Test" in output
        assert "latency" in output
        assert "1.50" in output


class TestJsonFormatter:
    """Tests for JsonFormatter."""

    def test_format_report(self):
        """Test formatting report as JSON."""
        formatter = JsonFormatter()
        report = BenchmarkReport()

        result = BenchmarkResult(name="Test", category="test")
        result.add_metric("latency", 1.5, "ms")
        report.add_result(result)
        report.end_time = datetime.now()

        output = formatter.format(report)

        # Parse JSON
        data = json.loads(output)

        assert "start_time" in data
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["name"] == "Test"


class TestHtmlFormatter:
    """Tests for HtmlFormatter."""

    def test_format_report(self):
        """Test formatting report as HTML."""
        formatter = HtmlFormatter()
        report = BenchmarkReport()

        result = BenchmarkResult(name="Test", category="test")
        result.passed = True
        result.add_metric("latency", 1.5, "ms", threshold=2.0)
        report.add_result(result)

        output = formatter.format(report)

        assert "<!DOCTYPE html>" in output
        assert "<title>Victor AI Performance Benchmark Report</title>" in output
        assert "Test" in output
        assert "1.50" in output
        assert "pass" in output.lower()


# =============================================================================
# Integration Tests
# =============================================================================


class TestBenchmarkIntegration:
    """Integration tests for benchmark suite."""

    def test_full_benchmark_workflow(self, tmp_path):
        """Test complete benchmark workflow."""
        runner = BenchmarkRunner(project_root=tmp_path)

        # Run benchmarks (with mocking to avoid actual work)
        with patch("benchmark_comprehensive.StartupBenchmark") as mock_startup:
            mock_instance = MagicMock()

            # Create results with metrics for all three startup benchmarks
            cold_result = BenchmarkResult(name="Cold Start", category="startup")
            cold_result.add_metric("startup_time", 150.0, "ms", threshold=200.0)

            warm_result = BenchmarkResult(name="Warm Start", category="startup")
            warm_result.add_metric("startup_time", 100.0, "ms", threshold=200.0)

            bootstrap_result = BenchmarkResult(name="Bootstrap Time", category="startup")
            bootstrap_result.add_metric("bootstrap_time", 50.0, "ms", threshold=100.0)

            mock_instance.run_cold_start.return_value = cold_result
            mock_instance.run_warm_start.return_value = warm_result
            mock_instance.run_bootstrap_time.return_value = bootstrap_result
            mock_startup.return_value = mock_instance

            report = runner.run(scenario="startup", save=False)

            # Verify report
            assert len(report.results) > 0
            assert report.start_time is not None

            # Format report
            formatter = MarkdownFormatter()
            output = formatter.format(report)

            assert "Victor AI Performance Benchmark Report" in output
            assert "Cold Start" in output

    def test_save_and_load_results(self, tmp_path):
        """Test saving and loading benchmark results."""
        runner = BenchmarkRunner(project_root=tmp_path)

        # Create report
        report = BenchmarkReport()
        result = BenchmarkResult(name="Test", category="test")
        result.add_metric("latency", 1.5, "ms")
        report.add_result(result)
        report.end_time = datetime.now()

        # Save
        runner._save_results(report)

        # Verify file exists
        files = list(tmp_path.glob(".benchmark_results/*.json"))
        assert len(files) > 0

        # Load and verify
        with open(files[0]) as f:
            data = json.load(f)

        assert "results" in data
        assert len(data["results"]) == 1


# =============================================================================
# Regression Detection Tests
# =============================================================================


class TestRegressionDetection:
    """Tests for regression detection in benchmark results."""

    def test_latency_regression(self):
        """Test detecting latency regression (higher is worse)."""
        result = BenchmarkResult(name="Test", category="test")

        # Add metric that exceeds threshold
        result.add_metric(
            name="latency",
            value=10.0,
            unit="ms",
            threshold=5.0,
        )

        assert result.passed is False

    def test_latency_improvement(self):
        """Test detecting latency improvement (lower is better)."""
        result = BenchmarkResult(name="Test", category="test")

        # Add metric that is below threshold
        result.add_metric(
            name="latency",
            value=2.0,
            unit="ms",
            threshold=5.0,
        )

        assert result.passed is True

    def test_throughput_regression(self):
        """Test detecting throughput regression (lower is worse)."""
        result = BenchmarkResult(name="Test", category="test")

        # Add metric that is below threshold
        result.add_metric(
            name="throughput",
            value=500,
            unit="ops/sec",
            threshold=1000,
        )

        assert result.passed is False

    def test_throughput_improvement(self):
        """Test detecting throughput improvement (higher is better)."""
        result = BenchmarkResult(name="Test", category="test")

        # Add metric that exceeds threshold
        result.add_metric(
            name="throughput",
            value=1500,
            unit="ops/sec",
            threshold=1000,
        )

        assert result.passed is True

    def test_memory_regression(self):
        """Test detecting memory regression (higher is worse)."""
        result = BenchmarkResult(name="Test", category="test")

        result.add_metric(
            name="memory",
            value=100,
            unit="MB",
            threshold=50,
        )

        assert result.passed is False

    def test_hit_rate_regression(self):
        """Test detecting hit rate regression (lower is worse)."""
        result = BenchmarkResult(name="Test", category="test")

        result.add_metric(
            name="hit_rate",
            value=0.3,
            unit="%",
            threshold=0.5,
        )

        assert result.passed is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
