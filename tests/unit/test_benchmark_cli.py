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

"""Tests for benchmark CLI commands."""

import pytest
from typer.testing import CliRunner
from victor.ui.commands.benchmark import benchmark_app


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


class TestBenchmarkRun:
    """Tests for benchmark run command."""

    def test_run_unknown_benchmark(self):
        """Test running an unknown benchmark."""
        result = runner.invoke(benchmark_app, ["run", "unknown-benchmark"])
        assert result.exit_code == 1
        assert "Unknown benchmark" in result.stdout

    def test_run_shows_help(self):
        """Test run command help."""
        result = runner.invoke(benchmark_app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--max-tasks" in result.stdout
        assert "--model" in result.stdout
        assert "--timeout" in result.stdout


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
