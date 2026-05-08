"""Tests for benchmark task type parameterization (task_type_hint)."""

from victor.benchmark.task_bridge import (
    _infer_task_type,
    benchmark_task_to_framework_task,
    framework_result_to_benchmark_result,
)
from victor.evaluation.protocol import BenchmarkTask, BenchmarkType
from victor.framework.task import FrameworkTaskType, TaskResult as FrameworkTaskResult


class TestBenchmarkTaskBridge:
    """Tests for benchmark/framework result and task bridging."""

    def test_bridge_reads_protocol_benchmark_field(self):
        task = BenchmarkTask(
            task_id="test-bridge",
            benchmark=BenchmarkType.CUSTOM,
            description="bridge task",
            prompt="fix the bug",
        )

        result = benchmark_task_to_framework_task(task)

        assert result.context["benchmark_type"] == BenchmarkType.CUSTOM.value

    def test_framework_result_to_benchmark_result_preserves_task_report_metrics(self):
        framework_result = FrameworkTaskResult(
            content="patched",
            success=True,
            metadata={
                "tokens_used": 54,
                "tokens_input": 31,
                "tokens_output": 23,
                "cached_tokens": 9,
                "reasoning_tokens": 4,
                "cost_usd_micros": 2500,
                "turns": 3,
                "task_report": {
                    "task_id": "task-55",
                    "request_count": 3,
                    "metadata": {
                        "continuation_ledger": "Intent: patch parser; Plan: read tests; fix callsite",
                    },
                },
            },
            tool_calls=[{"name": "read"}, {"name": "edit"}],
        )

        result = framework_result_to_benchmark_result(framework_result, "task-55")

        assert result.tokens_input == 31
        assert result.tokens_output == 23
        assert result.cached_tokens == 9
        assert result.reasoning_tokens == 4
        assert result.cost_usd_micros == 2500
        assert result.turns == 3
        assert result.metadata["task_report"]["task_id"] == "task-55"


class TestTaskTypeHint:
    """Tests for task_type_hint field and hint-first inference."""

    def _make_task(self, prompt: str = "do something", hint: str = None) -> BenchmarkTask:
        """Helper to create a BenchmarkTask with optional hint."""
        return BenchmarkTask(
            task_id="test-1",
            benchmark=BenchmarkType.CUSTOM,
            description="test task",
            prompt=prompt,
            task_type_hint=hint,
        )

    def test_hint_overrides_keyword_inference(self):
        """When task_type_hint is set, it takes precedence over keywords."""
        # Prompt contains 'fix' (would infer EDIT), but hint says ANALYZE
        task = self._make_task(prompt="fix the bug", hint="analyze")
        result = _infer_task_type(task)
        assert result == FrameworkTaskType.ANALYZE

    def test_hint_case_insensitive(self):
        """Hint matching is case-insensitive."""
        task = self._make_task(prompt="do something", hint="SEARCH")
        assert _infer_task_type(task) == FrameworkTaskType.SEARCH

        task2 = self._make_task(prompt="do something", hint="Create")
        assert _infer_task_type(task2) == FrameworkTaskType.CREATE

    def test_invalid_hint_falls_back_to_keywords(self):
        """An invalid hint falls through to keyword inference."""
        task = self._make_task(prompt="fix the bug", hint="nonexistent_type")
        result = _infer_task_type(task)
        assert result == FrameworkTaskType.EDIT  # from keyword 'fix'

    def test_none_hint_uses_keyword_inference(self):
        """When task_type_hint is None, keyword inference is used."""
        task = self._make_task(prompt="analyze the code", hint=None)
        result = _infer_task_type(task)
        assert result == FrameworkTaskType.ANALYZE

    def test_no_hint_no_keyword_defaults_to_edit(self):
        """Without hint or keywords, defaults to EDIT."""
        task = self._make_task(prompt="do something unusual", hint=None)
        result = _infer_task_type(task)
        assert result == FrameworkTaskType.EDIT

    def test_execute_hint(self):
        """Hint 'execute' maps to FrameworkTaskType.EXECUTE."""
        task = self._make_task(prompt="describe the architecture", hint="execute")
        result = _infer_task_type(task)
        assert result == FrameworkTaskType.EXECUTE
