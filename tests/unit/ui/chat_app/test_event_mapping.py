# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Tests for the Chainlit-free Victor-event -> render-action mapping.

These run without the optional ``chat-ui`` extra installed (no ``chainlit`` import).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from victor.ui.chat_app.event_mapping import RenderKind, map_event


@dataclass
class FakeEvent:
    """Mimics the public surface of a ``VictorClient.stream()`` event."""

    event_type: Any
    content: Optional[str] = None
    tool_name: Optional[str] = None
    result: Any = None
    metadata: Optional[Dict[str, Any]] = None


def test_content_event_maps_to_token() -> None:
    action = map_event(FakeEvent(event_type="content", content="hello"))
    assert action.kind is RenderKind.TOKEN
    assert action.text == "hello"


def test_thinking_event_prefers_content_then_metadata() -> None:
    assert map_event(FakeEvent("thinking", content="step 1")).text == "step 1"
    from_meta = map_event(FakeEvent("thinking", metadata={"reasoning_content": "deep"}))
    assert from_meta.kind is RenderKind.THINKING
    assert from_meta.text == "deep"


def test_tool_call_carries_name_and_arguments() -> None:
    action = map_event(
        FakeEvent("tool_call", tool_name="grep", metadata={"arguments": {"q": "foo"}})
    )
    assert action.kind is RenderKind.TOOL_START
    assert action.tool_name == "grep"
    assert action.metadata["arguments"] == {"q": "foo"}


def test_tool_result_from_dict_result_payload() -> None:
    action = map_event(
        FakeEvent(
            "tool_result",
            tool_name="grep",
            result={"result": "3 matches", "success": True, "arguments": {"q": "foo"}},
        )
    )
    assert action.kind is RenderKind.TOOL_END
    assert action.tool_name == "grep"
    assert action.text == "3 matches"
    assert action.success is True
    assert action.metadata["arguments"] == {"q": "foo"}


def test_tool_result_content_overrides_payload_result() -> None:
    action = map_event(
        FakeEvent("tool_result", tool_name="bash", content="exit 0", result={"success": True})
    )
    assert action.text == "exit 0"


def test_tool_error_marks_unsuccessful() -> None:
    action = map_event(FakeEvent("tool_error", tool_name="bash", content="boom"))
    assert action.kind is RenderKind.TOOL_END
    assert action.success is False
    assert action.text == "boom"


def test_error_event_uses_content_then_metadata() -> None:
    assert map_event(FakeEvent("error", content="bad")).text == "bad"
    from_meta = map_event(FakeEvent("error", metadata={"error": "timeout"}))
    assert from_meta.kind is RenderKind.ERROR
    assert from_meta.text == "timeout"


def test_error_event_falls_back_to_default_message() -> None:
    action = map_event(FakeEvent("error"))
    assert action.kind is RenderKind.ERROR
    assert action.text == "Unknown streaming error"


def test_lifecycle_and_unknown_events_are_ignored() -> None:
    for et in ("stream_start", "stream_end", "totally_new_event"):
        assert map_event(FakeEvent(et)).kind is RenderKind.IGNORE


def test_enum_like_event_type_is_normalized() -> None:
    class _EnumLike:
        value = "content"

        def __str__(self) -> str:  # not used once .value is read
            return "EventType.CONTENT"

    assert map_event(FakeEvent(_EnumLike(), content="x")).kind is RenderKind.TOKEN


def test_dotted_repr_event_type_is_normalized() -> None:
    # event_type given as a bare repr string "EventType.TOOL_CALL"
    action = map_event(FakeEvent("EventType.TOOL_CALL", tool_name="t"))
    assert action.kind is RenderKind.TOOL_START


def test_missing_tool_name_defaults_to_tool() -> None:
    assert map_event(FakeEvent("tool_call")).tool_name == "tool"
    assert map_event(FakeEvent("tool_result", content="x")).tool_name == "tool"
