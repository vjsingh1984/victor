"""Focused tests for ToolService.process_tool_results."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from victor.agent.services.tool_service import (
    ToolResultContext,
    ToolService,
    ToolServiceConfig,
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


def _make_service() -> ToolService:
    return ToolService(
        config=ToolServiceConfig(),
        tool_selector=MagicMock(),
        tool_executor=MagicMock(),
        tool_registrar=MagicMock(),
    )


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
        "format_tool_output": MagicMock(),
        "console": MagicMock(),
        "presentation": MagicMock(),
        "task_type": "analysis",
    }
    defaults.update(overrides)
    return ToolResultContext(**defaults)


def test_safe_default_pruning_runs_on_formatted_read_output():
    service = _make_service()
    formatted_output = "\n".join(
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
    ctx = _make_ctx(format_tool_output=MagicMock(return_value=formatted_output))
    pipeline_result = FakePipelineResult(
        results=[
            FakeCallResult(
                tool_name="read",
                success=True,
                result="raw result",
                arguments={"path": "src/main.py"},
                tool_call_id="call_read_1",
            )
        ]
    )

    with patch(
        "victor.config.tool_settings.get_tool_settings",
        return_value=MagicMock(
            tool_output_preview_enabled=True,
            tool_output_pruning_enabled=True,
            tool_output_pruning_safe_only=True,
        ),
    ):
        results = service.process_tool_results(pipeline_result, ctx)

    assert results[0]["was_pruned"] is True
    assert results[0]["result"] != formatted_output
    assert "[PRUNED PREVIEW:" in results[0]["result"]
    assert results[0]["full_result"] == formatted_output
    assert results[0]["content"] == formatted_output
    ctx.add_message.assert_called_once_with(
        "tool",
        results[0]["content"],
        name="read",
        tool_call_id="call_read_1",
    )


def test_no_regression_for_safety_critical_diff_like_output():
    service = _make_service()
    diff_like_output = (
        "=== TOOL OUTPUT: read ===\n"
        "Path: /tmp/patch.txt\n"
        "--- BEGIN CONTENT ---\n"
        "@@ -10,4 +10,4 @@\n"
        "-old_value\n"
        "+new_value\n"
        "Traceback (most recent call last):\n"
        "RuntimeError: boom\n"
        "--- END CONTENT ---"
    )
    ctx = _make_ctx(format_tool_output=MagicMock(return_value=diff_like_output))
    pipeline_result = FakePipelineResult(
        results=[
            FakeCallResult(
                tool_name="read",
                success=True,
                result={"content": diff_like_output},
                arguments={"path": "/tmp/patch.txt"},
                tool_call_id="call_read_2",
            )
        ]
    )

    with patch(
        "victor.config.tool_settings.get_tool_settings",
        return_value=MagicMock(
            tool_output_preview_enabled=True,
            tool_output_pruning_enabled=True,
            tool_output_pruning_safe_only=True,
        ),
    ):
        results = service.process_tool_results(pipeline_result, ctx)

    assert results[0]["was_pruned"] is False
    assert results[0]["content"] == diff_like_output
    assert results[0]["result"] == diff_like_output
    assert results[0]["full_result"] == diff_like_output


def test_mutating_tools_keep_full_display_output_even_when_preview_pruning_enabled():
    service = _make_service()
    formatted_output = "\n".join(
        [
            '<TOOL_OUTPUT tool="write" path="src/main.py">',
            "Wrote src/main.py",
            "Created 220 lines of code",
            "</TOOL_OUTPUT>",
        ]
    )
    ctx = _make_ctx(format_tool_output=MagicMock(return_value=formatted_output))
    pipeline_result = FakePipelineResult(
        results=[
            FakeCallResult(
                tool_name="write",
                success=True,
                result="raw result",
                arguments={"path": "src/main.py", "content": "print('hello')"},
                tool_call_id="call_write_1",
            )
        ]
    )

    with patch(
        "victor.config.tool_settings.get_tool_settings",
        return_value=MagicMock(
            tool_output_preview_enabled=True,
            tool_output_pruning_enabled=True,
            tool_output_pruning_safe_only=True,
        ),
    ):
        results = service.process_tool_results(pipeline_result, ctx)

    assert results[0]["was_pruned"] is False
    assert results[0]["result"] == formatted_output
    assert results[0]["full_result"] == formatted_output
    assert results[0]["content"] == formatted_output
