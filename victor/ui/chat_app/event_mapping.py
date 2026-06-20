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

"""Map Victor stream events to UI-agnostic render actions.

The chat UI (``app.py``) drives Chainlit from these actions, but this module imports
**nothing** from Chainlit so the mapping is unit-testable without the optional ``chat-ui``
extra installed. It operates on the public surface of ``VictorClient.stream()`` events
(``event_type``, ``content``, ``tool_name``, ``result``, ``metadata``) — see
``victor/framework/client.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


class RenderKind(str, Enum):
    """What the UI should do with a single stream event."""

    TOKEN = "token"  # append a content token to the assistant message
    THINKING = "thinking"  # reasoning text -> render as a collapsed step
    TOOL_START = "tool_start"  # a tool call began (carries arguments)
    TOOL_END = "tool_end"  # a tool produced a result -> render a tool step
    ERROR = "error"  # surface an error to the user
    IGNORE = "ignore"  # lifecycle/no-op events (stream_start, stream_end, ...)


@dataclass
class RenderAction:
    """A UI-agnostic instruction derived from one Victor stream event."""

    kind: RenderKind
    text: str = ""
    tool_name: Optional[str] = None
    call_id: Optional[str] = None  # correlates TOOL_START/TOOL_END for parallel calls
    success: bool = True
    elapsed: float = 0.0
    was_pruned: bool = False
    follow_up_suggestions: Optional[list[dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _extract_call_id(event: Any, metadata: Dict[str, Any]) -> Optional[str]:
    """Best-effort tool call id (correlates a tool_call to its tool_result)."""
    for key in ("tool_call_id", "id", "call_id"):
        value = metadata.get(key)
        if value:
            return str(value)
    for attr in ("tool_call_id", "call_id"):
        value = getattr(event, attr, None)
        if value:
            return str(value)
    return None


def _normalize_event_type(event_type: Any) -> str:
    """Reduce a str / ``EventType`` enum / ``"EventType.CONTENT"`` repr to a bare token.

    ``VictorClient.stream()`` yields events whose ``event_type`` is usually a lowercase
    string ("content", "tool_call", ...) but may be an ``EventType`` member; normalize all
    forms to the bare lowercase name ("content", "tool_call", "error").
    """
    value = getattr(event_type, "value", event_type)
    text = str(value).strip().lower()
    if "." in text:  # e.g. "eventtype.content" -> "content"
        text = text.rsplit(".", 1)[-1]
    return text


def _tool_result_payload(event: Any) -> Dict[str, Any]:
    """Best-effort extraction of the flat tool-result dict from an event.

    ``VictorClient._to_stream_event`` flattens the payload onto ``event.result``;
    fall back to a nested ``metadata['tool_result']`` (or bare ``metadata``) so the
    mapping still works if an event arrives un-flattened.
    """
    result = getattr(event, "result", None)
    if isinstance(result, dict):
        return result
    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, dict):
        nested = metadata.get("tool_result")
        if isinstance(nested, dict):
            return nested
        return metadata
    return {}


def map_event(event: Any) -> RenderAction:
    """Translate one ``VictorClient.stream()`` event into a :class:`RenderAction`.

    Unknown / lifecycle events map to :attr:`RenderKind.IGNORE` so callers can render with a
    single ``match`` and never crash on a new event type.
    """
    event_type = _normalize_event_type(getattr(event, "event_type", ""))
    content = getattr(event, "content", None) or ""
    metadata = getattr(event, "metadata", None) or {}

    if event_type == "content":
        return RenderAction(RenderKind.TOKEN, text=content)

    if event_type == "thinking":
        # Reasoning content arrives on `content` or `metadata['reasoning_content']`.
        text = content or str(metadata.get("reasoning_content", ""))
        return RenderAction(RenderKind.THINKING, text=text, metadata=dict(metadata))

    if event_type == "tool_call":
        return RenderAction(
            RenderKind.TOOL_START,
            tool_name=getattr(event, "tool_name", None) or "tool",
            call_id=_extract_call_id(event, metadata),
            metadata={"arguments": metadata.get("arguments", {})},
        )

    if event_type in ("tool_result", "tool_error"):
        payload = _tool_result_payload(event)
        # Prefer the full output for direct display; the bare ``result`` is a CLI
        # "/expand" placeholder, so it is the last resort behind the real output.
        text = payload.get("original_result") or content or payload.get("result") or ""
        return RenderAction(
            RenderKind.TOOL_END,
            text=str(text),
            tool_name=getattr(event, "tool_name", None) or "tool",
            call_id=_extract_call_id(event, payload) or _extract_call_id(event, metadata),
            success=event_type != "tool_error" and bool(payload.get("success", True)),
            elapsed=float(payload.get("elapsed", 0.0) or 0.0),
            was_pruned=bool(payload.get("was_pruned", False)),
            follow_up_suggestions=payload.get("follow_up_suggestions") or None,
            metadata={"arguments": payload.get("arguments", {})},
        )

    if event_type == "error":
        return RenderAction(
            RenderKind.ERROR,
            text=content or str(metadata.get("error", "Unknown streaming error")),
        )

    return RenderAction(RenderKind.IGNORE)


def segment_turn(kinds: Iterable["RenderKind"]) -> List[str]:
    """Return the ordered render-segment types for a turn — the natural-flow contract.

    A turn's events interleave per agent iteration: [text][tool_call/result…][text]…. To
    render that like the terminal (instead of all tool steps piling at the end), the UI emits
    a NEW text message segment whenever text resumes after a tool run, and groups each
    iteration's tool calls into one tool segment. This pure helper encodes that contract so it
    can be unit-tested; ``app.py`` mirrors it online while streaming.

    Returns a list like ``["text", "tools", "text", "tools", "text"]``. THINKING/IGNORE do not
    open a segment (reasoning renders in its own step; lifecycle events are no-ops).
    """
    segments: List[str] = []
    phase: Optional[str] = None
    for kind in kinds:
        if kind in (RenderKind.TOKEN, RenderKind.ERROR):
            if phase != "text":
                segments.append("text")
                phase = "text"
        elif kind in (RenderKind.TOOL_START, RenderKind.TOOL_END):
            if phase != "tools":
                segments.append("tools")
                phase = "tools"
    return segments
