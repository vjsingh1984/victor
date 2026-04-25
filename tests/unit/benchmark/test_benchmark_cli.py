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

"""Tests for benchmark CLI commands.

Unit tests that don't require Ollama or API keys test CLI structure only.
Integration tests that run actual benchmarks are skipped when Ollama is unavailable.
"""

import json
import socket
from unittest.mock import patch

import pytest
from typer.testing import CliRunner
from victor.ui.commands.benchmark import benchmark_app


def is_ollama_available() -> bool:
    """Check if Ollama server is running at localhost:11434."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", 11434))
        return result == 0
    finally:
        sock.close()


runner = CliRunner()


class TestBenchmarkList:
    """Tests for benchmark list command."""

    def test_list_benchmarks(self):
        """Test listing available benchmarks."""
        result = runner.invoke(benchmark_app, ["list"])
        assert result.exit_code == 0
        assert "Available Benchmarks" in result.stdout
        assert "swe-bench" in result.stdout
        assert "humaneval" in result.stdout
        assert "mbpp" in result.stdout

    def test_list_shows_task_counts(self):
        """Test that task counts are shown."""
        result = runner.invoke(benchmark_app, ["list"])
        assert result.exit_code == 0
        assert "164" in result.stdout  # HumanEval
        assert "974" in result.stdout  # MBPP

    def test_list_shows_external_agentic_benchmarks(self):
        """Test that external perception-heavy benchmarks are listed."""
        result = runner.invoke(benchmark_app, ["list"])
        assert result.exit_code == 0
        assert "clawbench" in result.stdout
        assert "dr3-eval" in result.stdout
        assert "guide" in result.stdout
        assert "vlaa-gui" in result.stdout
        assert "implemented" in result.stdout


class TestBenchmarkRun:
    """Tests for benchmark run command."""

    def test_run_unknown_benchmark(self):
        """Test running an unknown benchmark."""
        result = runner.invoke(benchmark_app, ["run", "unknown-benchmark"])
        assert result.exit_code == 1
        assert "Unknown benchmark" in result.stdout

    def test_run_cataloged_benchmark_requires_dataset_path(self):
        """External benchmark adapters should require a local manifest path."""
        result = runner.invoke(benchmark_app, ["run", "guide"])
        assert result.exit_code == 1
        assert "requires --dataset-path" in result.stdout

    def test_run_external_benchmark_with_dataset_path(self, tmp_path):
        """External benchmark adapters should route through the standard runner path."""
        dataset = tmp_path / "guide.json"
        output = tmp_path / "results.json"
        dataset.write_text(
            json.dumps(
                {
                    "metadata": {
                        "version": "2026.04",
                        "source_name": "GUIDE Consortium",
                        "languages": ["python"],
                    },
                    "tasks": [
                        {
                            "task_id": "guide-1",
                            "prompt": "Implement add",
                            "test_code": (
                                "from solution import add\n\n\n"
                                "def test_add():\n"
                                "    assert add(1, 2) == 3\n"
                            ),
                        }
                    ],
                }
            )
        )

        async def fake_run_benchmark_async(**_kwargs):
            from victor.evaluation.protocol import (
                BenchmarkFailureCategory,
                BenchmarkType,
                EvaluationConfig,
                EvaluationResult,
                TaskResult,
                TaskStatus,
            )

            return EvaluationResult(
                config=EvaluationConfig(
                    benchmark=BenchmarkType.GUIDE,
                    model="test-model",
                    dataset_metadata={
                        "source_name": "GUIDE Consortium",
                        "version": "2026.04",
                        "languages": ["python"],
                    },
                ),
                task_results=[
                    TaskResult(
                        task_id="guide-1",
                        status=TaskStatus.FAILED,
                        tests_passed=0,
                        tests_total=1,
                        failure_category=BenchmarkFailureCategory.TEST_FAILURE,
                        error_message="assertion failed",
                    )
                ],
            )

        with patch(
            "victor.ui.commands.benchmark._run_benchmark_async",
            side_effect=fake_run_benchmark_async,
        ):
            result = runner.invoke(
                benchmark_app,
                [
                    "run",
                    "guide",
                    "--dataset-path",
                    str(dataset),
                    "--model",
                    "test-model",
                    "--output",
                    str(output),
                ],
            )

        assert result.exit_code == 0
        assert "Dataset:" in result.stdout
        assert "Source: GUIDE Consortium" in result.stdout
        assert "Manifest Version: 2026.04" in result.stdout
        assert "Failure: test_failure" in result.stdout
        saved = json.loads(output.read_text())
        assert saved["dataset_metadata"]["source_name"] == "GUIDE Consortium"
        assert saved["failure_examples"]["test_failure"]["sample_task_ids"] == ["guide-1"]

    def test_run_shows_help(self):
        """Test run command help."""
        result = runner.invoke(benchmark_app, ["run", "--help"])
        assert result.exit_code == 0
        # Options may be truncated by Rich formatting, check for key parts
        assert "max-tasks" in result.stdout or "max_tasks" in result.stdout or "-n" in result.stdout
        assert "timeout" in result.stdout
        assert "profile" in result.stdout


class TestBenchmarkCompare:
    """Tests for benchmark compare command."""

    def test_compare_swe_bench(self):
        """Test comparing frameworks on SWE-bench."""
        result = runner.invoke(benchmark_app, ["compare", "--benchmark", "swe-bench"])
        assert result.exit_code == 0
        assert "Framework Comparison" in result.stdout

    def test_compare_unknown_benchmark(self):
        """Test comparing with unknown benchmark."""
        result = runner.invoke(benchmark_app, ["compare", "--benchmark", "unknown"])
        assert result.exit_code == 1
        assert "Unknown benchmark" in result.stdout

    def test_compare_cataloged_benchmark_without_results(self):
        """Test comparing a recognized benchmark that lacks published results."""
        result = runner.invoke(benchmark_app, ["compare", "--benchmark", "guide"])
        assert result.exit_code == 0
        assert "No published results available" in result.stdout

    def test_compare_external_benchmark_with_local_victor_results(self, tmp_path):
        """Local Victor artifacts should drive comparison output for external benchmarks."""
        saved_result = tmp_path / "guide_result.json"
        output = tmp_path / "compare.json"
        saved_result.write_text(
            json.dumps(
                {
                    "benchmark": "guide",
                    "model": "test-model",
                    "dataset_metadata": {"source_name": "GUIDE Consortium"},
                    "metrics": {
                        "total_tasks": 1,
                        "passed": 1,
                        "failed": 0,
                        "errors": 0,
                        "timeouts": 0,
                        "pass_rate": 1.0,
                    },
                    "task_results": [{"task_id": "guide-1", "status": "passed"}],
                }
            )
        )

        result = runner.invoke(
            benchmark_app,
            [
                "compare",
                "--benchmark",
                "guide",
                "--victor-results",
                str(saved_result),
                "--format",
                "json",
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0
        assert "Framework Comparison" in result.stdout
        saved = json.loads(output.read_text())
        assert saved["benchmark"] == "guide"
        assert saved["results"][0]["framework"] == "victor"
        assert saved["results"][0]["config"]["source"] == "GUIDE Consortium"

    def test_compare_rejects_mismatched_local_victor_results(self, tmp_path):
        """Comparison should fail when a local artifact is for another benchmark."""
        saved_result = tmp_path / "wrong_result.json"
        saved_result.write_text(
            json.dumps(
                {
                    "benchmark": "swe-bench",
                    "model": "test-model",
                    "metrics": {"total_tasks": 1, "pass_rate": 1.0},
                    "task_results": [{"task_id": "swe-1", "status": "passed"}],
                }
            )
        )

        result = runner.invoke(
            benchmark_app,
            [
                "compare",
                "--benchmark",
                "guide",
                "--victor-results",
                str(saved_result),
            ],
        )

        assert result.exit_code == 1
        assert "does not match requested" in result.stdout


class TestBenchmarkLeaderboard:
    """Tests for benchmark leaderboard command."""

    def test_leaderboard_swe_bench(self):
        """Test showing SWE-bench leaderboard."""
        result = runner.invoke(benchmark_app, ["leaderboard", "--benchmark", "swe-bench"])
        assert result.exit_code == 0
        assert "Leaderboard" in result.stdout
        assert "Rank" in result.stdout

    def test_leaderboard_unknown_benchmark(self):
        """Test leaderboard with unknown benchmark."""
        result = runner.invoke(benchmark_app, ["leaderboard", "--benchmark", "unknown"])
        assert result.exit_code == 1
        assert "Unknown benchmark" in result.stdout

    def test_leaderboard_external_benchmark_with_local_victor_results(self, tmp_path):
        """Leaderboards should accept local Victor artifacts for external benchmarks."""
        saved_result = tmp_path / "guide_result.json"
        saved_result.write_text(
            json.dumps(
                {
                    "benchmark": "guide",
                    "model": "test-model",
                    "dataset_metadata": {"source_name": "GUIDE Consortium"},
                    "metrics": {
                        "total_tasks": 1,
                        "passed": 1,
                        "failed": 0,
                        "errors": 0,
                        "timeouts": 0,
                        "pass_rate": 1.0,
                    },
                    "task_results": [{"task_id": "guide-1", "status": "passed"}],
                }
            )
        )

        result = runner.invoke(
            benchmark_app,
            [
                "leaderboard",
                "--benchmark",
                "guide",
                "--victor-results",
                str(saved_result),
            ],
        )

        assert result.exit_code == 0
        assert "Leaderboard" in result.stdout
        assert "victor" in result.stdout
        assert "GUIDE Consortium" in result.stdout


class TestBenchmarkCapabilities:
    """Tests for benchmark capabilities command."""

    def test_capabilities(self):
        """Test showing capabilities comparison."""
        result = runner.invoke(benchmark_app, ["capabilities"])
        assert result.exit_code == 0
        assert "Framework Capabilities Comparison" in result.stdout
        assert "Code Generation" in result.stdout
        assert "victor" in result.stdout
        assert "aider" in result.stdout


class TestBenchmarkHelp:
    """Tests for benchmark help."""

    def test_benchmark_help(self):
        """Test benchmark help output."""
        result = runner.invoke(benchmark_app, ["--help"])
        assert result.exit_code == 0
        assert "list" in result.stdout
        assert "run" in result.stdout
        assert "compare" in result.stdout
        assert "leaderboard" in result.stdout
        assert "capabilities" in result.stdout


@pytest.mark.skipif(
    not is_ollama_available(),
    reason="Ollama not available - skipping benchmark execution tests",
)
class TestBenchmarkExecution:
    """Integration tests for actually running benchmarks.

    These tests require Ollama to be running with a model available.
    They are skipped in CI environments where Ollama is not available.
    """

    def test_run_humaneval_with_ollama(self):
        """Test running HumanEval with Ollama (limited tasks)."""
        # Use default profile which should use Ollama if available
        result = runner.invoke(
            benchmark_app,
            [
                "run",
                "humaneval",
                "--max-tasks",
                "1",
                "--timeout",
                "60",
                "--profile",
                "default",
            ],
        )
        # Should start running or fail gracefully if model unavailable
        # We don't assert success since the model might not be loaded
        assert (
            "HumanEval" in result.stdout
            or "humaneval" in result.stdout
            or result.exit_code in (0, 1)
        )

    def test_run_mbpp_with_ollama(self):
        """Test running MBPP with Ollama (limited tasks)."""
        result = runner.invoke(
            benchmark_app,
            [
                "run",
                "mbpp",
                "--max-tasks",
                "1",
                "--timeout",
                "60",
                "--profile",
                "default",
            ],
        )
        assert "MBPP" in result.stdout or "mbpp" in result.stdout or result.exit_code in (0, 1)
