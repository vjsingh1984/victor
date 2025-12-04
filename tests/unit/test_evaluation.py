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

"""Unit tests for evaluation harness and metrics."""

import asyncio
import math
import pytest

from victor.evaluation import (
    CodeQualityMetrics,
    TaskResult,
    TaskStatus,
    pass_at_k,
)
from victor.evaluation.code_quality import CodeQualityAnalyzer
from victor.evaluation.pass_at_k import (
    PassAtKEvaluator,
    PassAtKResult,
    combinations,
    estimate_required_samples,
    generate_report,
)
from victor.evaluation.analyzers import AnalyzerRegistry


class TestPassAtK:
    """Tests for Pass@k metric calculation."""

    def test_pass_at_k_basic(self):
        """Test basic pass@k calculation."""
        # 50% success rate
        result = pass_at_k(n=100, c=50, k=1)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_pass_at_k_high_k(self):
        """With high k, pass@k should approach 1 if any correct samples."""
        result = pass_at_k(n=100, c=50, k=100)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_pass_at_k_zero_correct(self):
        """With no correct samples, pass@k should be 0."""
        result = pass_at_k(n=100, c=0, k=10)
        assert result == 0.0

    def test_pass_at_k_all_correct(self):
        """With all correct samples, pass@k should be 1."""
        result = pass_at_k(n=100, c=100, k=1)
        assert result == 1.0

    def test_pass_at_k_edge_cases(self):
        """Test edge cases."""
        assert pass_at_k(n=0, c=0, k=1) == 0.0
        assert pass_at_k(n=10, c=10, k=5) == 1.0
        assert pass_at_k(n=10, c=5, k=20) > 0.9  # k > n, should use k=n

    def test_pass_at_k_realistic_values(self):
        """Test with realistic benchmark values."""
        # HumanEval-like scenario: 50 samples, 20 correct
        result = pass_at_k(n=50, c=20, k=10)
        assert 0.9 < result < 1.0  # High probability with k=10

        # Low success rate scenario
        result = pass_at_k(n=100, c=5, k=1)
        assert result == pytest.approx(0.05, abs=0.01)

    def test_combinations(self):
        """Test combinations calculation."""
        assert combinations(5, 2) == pytest.approx(10, abs=0.01)
        assert combinations(10, 3) == pytest.approx(120, abs=0.01)
        assert combinations(10, 0) == 1
        assert combinations(10, 10) == 1
        assert combinations(5, 6) == 0  # k > n


class TestCodeQualityMetrics:
    """Tests for CodeQualityMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = CodeQualityMetrics()
        assert metrics.syntax_valid is True
        assert metrics.lint_errors == 0
        assert metrics.style_score == 1.0
        assert metrics.maintainability_index == 100.0

    def test_overall_score_perfect(self):
        """Test overall score with perfect metrics."""
        metrics = CodeQualityMetrics(
            syntax_valid=True,
            style_score=1.0,
            cyclomatic_complexity=1.0,
            maintainability_index=100.0,
        )
        score = metrics.get_overall_score()
        assert score == pytest.approx(97.75, abs=1.0)

    def test_overall_score_syntax_error(self):
        """Test overall score with syntax error."""
        metrics = CodeQualityMetrics(syntax_valid=False)
        score = metrics.get_overall_score()
        assert score < 100  # Syntax invalid penalizes score

    def test_overall_score_high_complexity(self):
        """Test overall score with high complexity."""
        metrics = CodeQualityMetrics(cyclomatic_complexity=20.0)
        score = metrics.get_overall_score()
        assert score < 80  # High complexity should lower score


class TestCodeQualityAnalyzer:
    """Tests for CodeQualityAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return CodeQualityAnalyzer(use_ruff=False, use_radon=False)

    @pytest.mark.asyncio
    async def test_valid_python_code(self, analyzer):
        """Test analysis of valid Python code."""
        code = '''
def hello_world():
    """Print hello world."""
    print("Hello, World!")
    return True
'''
        metrics = await analyzer.analyze(code, language="python")
        assert metrics.syntax_valid is True
        assert metrics.functions_count == 1
        assert metrics.lines_of_code > 0

    @pytest.mark.asyncio
    async def test_invalid_syntax(self, analyzer):
        """Test analysis of code with syntax errors."""
        code = "def broken(:"
        metrics = await analyzer.analyze(code, language="python")
        assert metrics.syntax_valid is False

    @pytest.mark.asyncio
    async def test_empty_code(self, analyzer):
        """Test analysis of empty code."""
        metrics = await analyzer.analyze("", language="python")
        assert metrics.syntax_valid is False

    @pytest.mark.asyncio
    async def test_type_coverage(self, analyzer):
        """Test type hint coverage detection."""
        typed_code = '''
def add(a: int, b: int) -> int:
    return a + b
'''
        untyped_code = '''
def add(a, b):
    return a + b
'''
        typed_metrics = await analyzer.analyze(typed_code, language="python")
        untyped_metrics = await analyzer.analyze(untyped_code, language="python")

        assert typed_metrics.type_coverage > untyped_metrics.type_coverage

    @pytest.mark.asyncio
    async def test_complexity_estimation(self, analyzer):
        """Test complexity estimation without radon."""
        simple_code = '''
def simple():
    return 1
'''
        complex_code = '''
def complex(x):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                print(i)
            elif i % 3 == 0:
                continue
            else:
                break
'''
        simple_metrics = await analyzer.analyze(simple_code, language="python")
        complex_metrics = await analyzer.analyze(complex_code, language="python")

        assert complex_metrics.cyclomatic_complexity > simple_metrics.cyclomatic_complexity


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_test_pass_rate(self):
        """Test test pass rate calculation."""
        result = TaskResult(
            task_id="test1",
            status=TaskStatus.PASSED,
            tests_passed=8,
            tests_total=10,
        )
        assert result.test_pass_rate == 0.8

    def test_test_pass_rate_zero_total(self):
        """Test pass rate with zero total tests."""
        result = TaskResult(
            task_id="test1",
            status=TaskStatus.PASSED,
            tests_total=0,
        )
        assert result.test_pass_rate == 0.0

    def test_is_success(self):
        """Test is_success property."""
        passed = TaskResult(task_id="t1", status=TaskStatus.PASSED)
        failed = TaskResult(task_id="t2", status=TaskStatus.FAILED)

        assert passed.is_success is True
        assert failed.is_success is False

    def test_tokens_per_test(self):
        """Test token efficiency calculation."""
        result = TaskResult(
            task_id="test1",
            status=TaskStatus.PASSED,
            tokens_used=1000,
            tests_passed=10,
        )
        assert result.tokens_per_test == 100.0

    def test_tokens_per_test_zero_passed(self):
        """Test tokens per test with zero passed."""
        result = TaskResult(
            task_id="test1",
            status=TaskStatus.FAILED,
            tokens_used=1000,
            tests_passed=0,
        )
        assert result.tokens_per_test == float('inf')

    def test_cost_efficiency(self):
        """Test cost efficiency calculation."""
        result = TaskResult(
            task_id="test1",
            status=TaskStatus.PASSED,
            tokens_used=1000,
            completion_score=0.8,
        )
        assert result.cost_efficiency == pytest.approx(0.8, abs=0.01)

    def test_calculate_completion_score(self):
        """Test completion score calculation."""
        result = TaskResult(
            task_id="test1",
            status=TaskStatus.PASSED,
            tests_passed=8,
            tests_total=10,
            generated_code="def foo(): pass",
        )

        score = result.calculate_completion_score()
        assert 0.5 < score < 1.0  # Should have good score


class TestPassAtKResult:
    """Tests for PassAtKResult."""

    def test_pass_rate(self):
        """Test pass rate calculation."""
        result = PassAtKResult(
            task_id="test1",
            total_samples=100,
            correct_samples=50,
            k_values=[1, 10],
            pass_at_k_scores={1: 0.5, 10: 0.99},
        )
        assert result.pass_rate == 0.5


class TestAnalyzerRegistry:
    """Tests for AnalyzerRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        AnalyzerRegistry.clear()

    def test_register_and_get(self):
        """Test registering and retrieving analyzers."""
        mock_analyzer = {"type": "mock"}
        AnalyzerRegistry.register("mock", mock_analyzer)

        retrieved = AnalyzerRegistry.get("mock")
        assert retrieved == mock_analyzer

    def test_get_nonexistent(self):
        """Test getting non-existent analyzer."""
        result = AnalyzerRegistry.get("nonexistent")
        assert result is None

    def test_register_factory(self):
        """Test lazy instantiation with factory."""
        factory_calls = [0]

        def factory():
            factory_calls[0] += 1
            return {"created": True}

        AnalyzerRegistry.register_factory("lazy", factory)

        # Factory not called yet
        assert factory_calls[0] == 0

        # Get instance triggers factory
        result = AnalyzerRegistry.get("lazy")
        assert factory_calls[0] == 1
        assert result == {"created": True}

        # Second get uses cached instance
        result2 = AnalyzerRegistry.get("lazy")
        assert factory_calls[0] == 1  # No second call
        assert result2 == result

    def test_get_code_quality_analyzer(self):
        """Test getting code quality analyzer singleton."""
        analyzer1 = AnalyzerRegistry.get_code_quality_analyzer()
        analyzer2 = AnalyzerRegistry.get_code_quality_analyzer()

        assert analyzer1 is analyzer2  # Same instance

    def test_clear(self):
        """Test clearing registry."""
        AnalyzerRegistry.register("test", {})
        assert AnalyzerRegistry.get("test") is not None

        AnalyzerRegistry.clear()
        assert AnalyzerRegistry.get("test") is None


class TestEstimateRequiredSamples:
    """Tests for sample estimation."""

    def test_estimate_basic(self):
        """Test basic sample estimation."""
        # To achieve 95% pass@10, we need some samples
        n = estimate_required_samples(target_pass_rate=0.95, k=10)
        assert n >= 10

    def test_estimate_returns_value(self):
        """Test estimation returns a reasonable value."""
        # The estimation searches for n where pass@k >= target
        n = estimate_required_samples(target_pass_rate=0.5, k=1)
        # With pass@1=0.5, we need c/n=0.5, so any n with c=n/2 works
        assert n >= 1


class TestGenerateReport:
    """Tests for report generation."""

    def test_generate_report(self):
        """Test report generation."""
        from victor.evaluation.pass_at_k import AggregatePassAtKResult

        result = AggregatePassAtKResult(
            total_tasks=10,
            k_values=[1, 10, 100],
            mean_pass_at_k={1: 0.3, 10: 0.8, 100: 0.99},
            task_results=[
                PassAtKResult(
                    task_id=f"task_{i}",
                    total_samples=100,
                    correct_samples=50,
                    k_values=[1, 10, 100],
                    pass_at_k_scores={1: 0.5, 10: 0.99, 100: 1.0},
                )
                for i in range(10)
            ],
        )

        report = generate_report(result)

        assert "PASS@K EVALUATION REPORT" in report
        assert "Pass@  1: 30.00%" in report
        assert "Pass@ 10: 80.00%" in report
        assert "Pass@100: 99.00%" in report
        assert "Total Tasks: 10" in report
