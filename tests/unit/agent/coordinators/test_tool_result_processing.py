"""Tests for ToolCoordinator.process_tool_results (extracted from orchestrator)."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock, patch

import pytest

from victor.agent.coordinators.tool_coordinator import (
    ToolCoordinator,
    ToolResultContext,
)


@dataclass
class FakeCallResult:
    """Minimal stand-in for PipelineCallResult."""

    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    execution_time_ms: float = 100.0
    skipped: bool = False
    tool_call_id: Optional[str] = None


@dataclass
class FakePipelineResult:
    """Minimal stand-in for PipelineExecutionResult."""

    results: List[FakeCallResult] = field(default_factory=list)


def _make_ctx(**overrides) -> ToolResultContext:
    defaults = {
        "executed_tools": [],
        "observed_files": set(),
        "failed_tool_signatures": set(),
        "shown_tool_errors": set(),
        "record_tool_execution": MagicMock(),
        "conversation_state": MagicMock(),
        "unified_tracker": MagicMock(),
        "usage_logger": MagicMock(spec=["log_event"]),
        "add_message": MagicMock(),
        "format_tool_output": MagicMock(return_value="formatted"),
        "console": MagicMock(),
        "presentation": MagicMock(),
        "task_type": "analysis",
    }
    defaults.update(overrides)
    return ToolResultContext(**defaults)


class TestProcessToolResults:
    def _make_coordinator(self):
        with pytest.warns(DeprecationWarning, match="ToolCoordinator is deprecated"):
            return ToolCoordinator(
                tool_pipeline=MagicMock(),
                tool_registry=MagicMock(),
            )

    def test_empty_results(self):
        coord = self._make_coordinator()
        pipeline_result = FakePipelineResult(results=[])
        ctx = _make_ctx()
        results = coord.process_tool_results(pipeline_result, ctx)
        assert results == []

    def test_skipped_results_ignored(self):
        coord = self._make_coordinator()
        pipeline_result = FakePipelineResult(
            results=[FakeCallResult(tool_name="read", success=True, skipped=True)]
        )
        ctx = _make_ctx()
        results = coord.process_tool_results(pipeline_result, ctx)
        assert results == []

    def test_successful_tool_call(self):
        coord = self._make_coordinator()
        pipeline_result = FakePipelineResult(
            results=[
                FakeCallResult(
                    tool_name="read",
                    success=True,
                    result={"content": "hello"},
                    arguments={"path": "/tmp/test.py"},
                    execution_time_ms=50.0,
                )
            ]
        )
        ctx = _make_ctx()
        results = coord.process_tool_results(pipeline_result, ctx)

        assert len(results) == 1
        assert results[0]["name"] == "read"
        assert results[0]["success"] is True
        assert results[0]["elapsed"] == 0.05  # 50ms in seconds

        # State mutations
        assert "read" in ctx.executed_tools
        assert "/tmp/test.py" in ctx.observed_files

        # Conversation injection (role=tool per OpenAI spec with name and tool_call_id)
        ctx.add_message.assert_called_once_with("tool", "formatted", name="read", tool_call_id=None)

    def test_failed_tool_call(self):
        coord = self._make_coordinator()
        pipeline_result = FakePipelineResult(
            results=[
                FakeCallResult(
                    tool_name="edit",
                    success=False,
                    error="File not found",
                    arguments={"path": "/missing"},
                    execution_time_ms=10.0,
                )
            ]
        )
        ctx = _make_ctx()
        results = coord.process_tool_results(pipeline_result, ctx)

        assert len(results) == 1
        assert results[0]["success"] is False
        assert results[0]["error"] == "File not found"
        assert len(ctx.failed_tool_signatures) == 1

    def test_semantic_failure_detection(self):
        coord = self._make_coordinator()
        pipeline_result = FakePipelineResult(
            results=[
                FakeCallResult(
                    tool_name="git",
                    success=True,
                    result={"success": False, "error": "merge conflict"},
                    arguments={"op": "merge"},
                    execution_time_ms=200.0,
                )
            ]
        )
        ctx = _make_ctx()
        results = coord.process_tool_results(pipeline_result, ctx)

        assert results[0]["success"] is False
        assert results[0]["error"] == "merge conflict"

    def test_follow_up_suggestions_extracted(self):
        coord = self._make_coordinator()
        pipeline_result = FakePipelineResult(
            results=[
                FakeCallResult(
                    tool_name="search",
                    success=True,
                    result={
                        "matches": [],
                        "metadata": {"follow_up_suggestions": ["try broader search"]},
                    },
                    arguments={"query": "test"},
                )
            ]
        )
        ctx = _make_ctx()
        results = coord.process_tool_results(pipeline_result, ctx)

        assert results[0]["follow_up_suggestions"] == ["try broader search"]

    def test_analytics_recorded(self):
        coord = self._make_coordinator()
        pipeline_result = FakePipelineResult(
            results=[FakeCallResult(tool_name="read", success=True, result="ok")]
        )
        ctx = _make_ctx()
        coord.process_tool_results(pipeline_result, ctx)

        ctx.record_tool_execution.assert_called_once()
        ctx.conversation_state.record_tool_execution.assert_called_once()
        ctx.unified_tracker.update_from_tool_call.assert_called_once()

    def test_usage_logger_called(self):
        coord = self._make_coordinator()
        pipeline_result = FakePipelineResult(
            results=[FakeCallResult(tool_name="read", success=True, result="data")]
        )
        ctx = _make_ctx()
        coord.process_tool_results(pipeline_result, ctx)

        ctx.usage_logger.log_event.assert_called_once_with(
            "tool_result",
            {
                "tool_name": "read",
                "success": True,
                "skipped": False,
                "outcome_kind": None,
                "block_source": None,
                "retryable": None,
                "result": "data",
                "error": None,
            },
        )

    def test_continuation_prompts_reset_on_success(self):
        coord = self._make_coordinator()
        pipeline_result = FakePipelineResult(
            results=[FakeCallResult(tool_name="read", success=True, result="ok")]
        )
        ctx = _make_ctx(continuation_prompts=3, asking_input_prompts=2)
        coord.process_tool_results(pipeline_result, ctx)

        assert ctx.continuation_prompts == 0
        assert ctx.asking_input_prompts == 0

    def test_error_dedup_for_not_found(self):
        coord = self._make_coordinator()
        pipeline_result = FakePipelineResult(
            results=[
                FakeCallResult(
                    tool_name="read",
                    success=False,
                    error="File not found: /a",
                    arguments={"path": "/a"},
                ),
                FakeCallResult(
                    tool_name="read",
                    success=False,
                    error="File not found: /b",
                    arguments={"path": "/b"},
                ),
            ]
        )
        ctx = _make_ctx()
        coord.process_tool_results(pipeline_result, ctx)

        # Second "not found" for same tool should be deduped
        assert ctx.console.print.call_count == 1

    def test_safe_default_pruning_marks_preview_but_llm_gets_full_output(self):
        # Pruning is detected for user preview (was_pruned=True) but LLM always receives
        # the full output for accuracy. The [PRUNED FOR LLM:] marker no longer appears
        # in content — that was the old accuracy-compromising behavior.
        coord = self._make_coordinator()
        formatted = "\n".join(
            [
                '<TOOL_OUTPUT tool="read" path="src/main.py">',
                "═══ ACTUAL FILE CONTENT: src/main.py ═══",
                "[File: src/main.py]",
                "[Lines 1-219 of 219]",
                "[Size: 4096 bytes]",
            ]
            + [f"{i}\tline {i}" for i in range(1, 220)]
            + [
                "═══ END OF FILE: src/main.py ═══",
                "</TOOL_OUTPUT>",
            ]
        )
        pipeline_result = FakePipelineResult(
            results=[
                FakeCallResult(
                    tool_name="read",
                    success=True,
                    result="raw result",
                    arguments={"path": "src/main.py"},
                    execution_time_ms=50.0,
                )
            ]
        )
        ctx = _make_ctx(format_tool_output=MagicMock(return_value=formatted))

        with patch("victor.config.tool_settings.get_tool_settings") as mock_settings:
            from victor.config.tool_settings import ToolSettings

            mock_settings.return_value = ToolSettings(
                tool_output_preview_enabled=True,
                tool_output_pruning_enabled=True,
                tool_output_pruning_safe_only=True,
            )
            results = coord.process_tool_results(pipeline_result, ctx)

        assert results[0]["was_pruned"] is True
        assert results[0]["result"] != formatted
        assert "[PRUNED PREVIEW:" in results[0]["result"]
        assert results[0]["full_result"] == formatted
        # LLM always receives the full output — accuracy-first architecture
        assert results[0]["content"] == formatted
        assert "[PRUNED PREVIEW:" not in results[0]["content"]
        added_content = ctx.add_message.call_args.args[1]
        assert added_content == results[0]["content"]
        assert results[0]["pruning_info"].recovery_hint.startswith("Use read(")

    def test_no_regression_for_safety_critical_diff_like_output(self):
        coord = self._make_coordinator()
        diff_like_output = "\n".join(
            [
                '<TOOL_OUTPUT tool="read" path="artifacts/pytest.diff">',
                "diff --git a/foo.py b/foo.py",
                "--- a/foo.py",
                "+++ b/foo.py",
                "@@ -1,3 +1,3 @@",
                "</TOOL_OUTPUT>",
            ]
        )
        pipeline_result = FakePipelineResult(
            results=[
                FakeCallResult(
                    tool_name="read",
                    success=True,
                    result="raw result",
                    arguments={"path": "artifacts/pytest.diff"},
                )
            ]
        )
        ctx = _make_ctx(format_tool_output=MagicMock(return_value=diff_like_output))

        with patch("victor.config.tool_settings.get_tool_settings") as mock_settings:
            from victor.config.tool_settings import ToolSettings

            mock_settings.return_value = ToolSettings(
                tool_output_preview_enabled=True,
                tool_output_pruning_enabled=True,
                tool_output_pruning_safe_only=True,
            )
            results = coord.process_tool_results(pipeline_result, ctx)

        assert results[0]["was_pruned"] is False
        assert results[0]["content"] == diff_like_output
        assert results[0]["result"] == diff_like_output
        assert results[0]["full_result"] == diff_like_output
