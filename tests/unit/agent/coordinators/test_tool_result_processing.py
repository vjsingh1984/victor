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
    }
    defaults.update(overrides)
    return ToolResultContext(**defaults)


class TestProcessToolResults:
    def _make_coordinator(self):
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

        # Conversation injection
        ctx.add_message.assert_called_once_with("user", "formatted")

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
            results=[
                FakeCallResult(tool_name="read", success=True, result="ok")
            ]
        )
        ctx = _make_ctx()
        coord.process_tool_results(pipeline_result, ctx)

        ctx.record_tool_execution.assert_called_once()
        ctx.conversation_state.record_tool_execution.assert_called_once()
        ctx.unified_tracker.update_from_tool_call.assert_called_once()

    def test_usage_logger_called(self):
        coord = self._make_coordinator()
        pipeline_result = FakePipelineResult(
            results=[
                FakeCallResult(tool_name="read", success=True, result="data")
            ]
        )
        ctx = _make_ctx()
        coord.process_tool_results(pipeline_result, ctx)

        ctx.usage_logger.log_event.assert_called_once_with(
            "tool_result",
            {
                "tool_name": "read",
                "success": True,
                "result": "data",
                "error": None,
            },
        )

    def test_continuation_prompts_reset_on_success(self):
        coord = self._make_coordinator()
        pipeline_result = FakePipelineResult(
            results=[
                FakeCallResult(tool_name="read", success=True, result="ok")
            ]
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
