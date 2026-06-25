from __future__ import annotations

import asyncio
from typing import Any

from victor.framework.events import AgentExecutionEvent, EventType
from victor.ui.rendering.handler import stream_response


class _RecorderRenderer:
    def __init__(self) -> None:
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.tool_results: list[dict[str, Any]] = []
        self.content: list[str] = []
        self.statuses: list[str] = []

    def start(self) -> None:
        pass

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass

    def on_tool_start(self, name: str, arguments: dict[str, Any]) -> None:
        self.tool_calls.append((name, arguments))

    def on_tool_result(
        self,
        name: str,
        success: bool,
        elapsed: float,
        arguments: dict[str, Any],
        error: str | None = None,
        follow_up_suggestions: list[dict[str, Any]] | None = None,
        was_pruned: bool = False,
        original_result: Any = None,
        result: Any = None,
    ) -> None:
        self.tool_results.append(
            {
                "name": name,
                "success": success,
                "elapsed": elapsed,
                "arguments": arguments,
                "error": error,
                "follow_up_suggestions": follow_up_suggestions,
                "was_pruned": was_pruned,
                "original_result": original_result,
                "result": result,
            }
        )

    def on_status(self, message: str) -> None:
        self.statuses.append(message)

    def on_file_preview(self, path: str, content: str) -> None:
        pass

    def on_edit_preview(self, path: str, diff: str) -> None:
        pass

    def on_content(self, text: str) -> None:
        self.content.append(text)

    def on_thinking_content(self, text: str) -> None:
        pass

    def on_thinking_start(self) -> None:
        pass

    def on_thinking_end(self) -> None:
        pass

    def had_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def finalize(self) -> str:
        return "".join(self.content)

    def cleanup(self) -> None:
        pass


class _AgentStreamSource:
    def __init__(self, events: list[AgentExecutionEvent]) -> None:
        self._events = events

    async def stream(self, _message: str):
        for event in self._events:
            yield event


def test_stream_response_normalizes_agent_tool_events() -> None:
    follow_ups = [
        {
            "command": 'graph(mode="neighbors", node="GraphAnalyzer", depth=2)',
            "description": "Inspect nearby graph nodes.",
        }
    ]
    agent = _AgentStreamSource(
        [
            AgentExecutionEvent(
                type=EventType.TOOL_CALL,
                tool_name="graph",
                arguments={"mode": "find", "query": "graph"},
            ),
            AgentExecutionEvent(
                type=EventType.TOOL_RESULT,
                tool_name="graph",
                result="Tool completed successfully.",
                success=True,
                metadata={
                    "tool_result": {
                        "name": "graph",
                        "success": True,
                        "elapsed": 0.25,
                        "arguments": {"mode": "find", "query": "graph"},
                        "follow_up_suggestions": follow_ups,
                        "result": "preview graph output",
                        "original_result": "full graph output",
                        "was_pruned": True,
                    }
                },
            ),
            AgentExecutionEvent(type=EventType.CONTENT, content="analysis complete"),
        ]
    )
    renderer = _RecorderRenderer()

    result = asyncio.run(stream_response(agent, "review graph code", renderer))

    assert renderer.tool_calls == [("graph", {"mode": "find", "query": "graph"})]
    assert renderer.tool_results == [
        {
            "name": "graph",
            "success": True,
            "elapsed": 0.25,
            "arguments": {"mode": "find", "query": "graph"},
            "error": None,
            "follow_up_suggestions": follow_ups,
            "was_pruned": True,
            "original_result": "full graph output",
            "result": "preview graph output",
        }
    ]
    assert result == "analysis complete"


def test_stream_response_surfaces_error_events() -> None:
    """A terminal ERROR event must be shown to the user, not silently dropped."""
    agent = _AgentStreamSource(
        [
            AgentExecutionEvent(
                type=EventType.ERROR,
                error="Model not found: gpt-oss:latest\n💡 Run `ollama pull gpt-oss:latest`",
                recoverable=False,
            ),
        ]
    )
    renderer = _RecorderRenderer()

    result = asyncio.run(stream_response(agent, "hi", renderer))

    # No content, but the error was surfaced via a status line (with a marker).
    assert result == ""
    assert any("Model not found: gpt-oss:latest" in s for s in renderer.statuses)
    assert any(s.startswith("❌") for s in renderer.statuses)
