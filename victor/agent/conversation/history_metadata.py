"""Metadata helpers for interactive history visibility."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional

INTERACTIVE_HISTORY_KEY = "interactive_history"
INTERNAL_PROMPT_KIND_KEY = "internal_prompt_kind"

if TYPE_CHECKING:
    from victor.agent.conversation.types import MessageSource


def build_internal_history_metadata(
    kind: str,
    source: Optional["MessageSource"] = None,
    **extra: Any,
) -> dict[str, Any]:
    """Mark a message as internal so it stays out of interactive history.

    Args:
        kind: Internal prompt kind (e.g. "nudge", "continuation")
        source: MessageSource origin — stored as metadata["source"] for compaction scoring
        **extra: Additional metadata fields
    """
    from victor.agent.conversation.types import MESSAGE_SOURCE_METADATA_KEY, MessageSource

    metadata: dict[str, Any] = {
        INTERACTIVE_HISTORY_KEY: False,
        INTERNAL_PROMPT_KIND_KEY: kind,
    }
    if source is not None:
        metadata[MESSAGE_SOURCE_METADATA_KEY] = source.value
    metadata.update(extra)
    return metadata


def is_hidden_from_interactive_history(metadata: Mapping[str, Any] | None) -> bool:
    """Return whether metadata marks a message as hidden from input history."""
    if not metadata:
        return False
    return metadata.get(INTERACTIVE_HISTORY_KEY) is False
