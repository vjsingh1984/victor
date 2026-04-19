"""Tests for benchmark task type parameterization (task_type_hint)."""

from victor.evaluation.protocol import BenchmarkTask, BenchmarkType
from victor.benchmark.task_bridge import _infer_task_type
from victor.framework.task import FrameworkTaskType


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
