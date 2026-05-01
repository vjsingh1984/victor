"""Metadata helpers for interactive history visibility."""

from __future__ import annotations

from typing import Any, Mapping

INTERACTIVE_HISTORY_KEY = "interactive_history"
INTERNAL_PROMPT_KIND_KEY = "internal_prompt_kind"


def build_internal_history_metadata(kind: str, **extra: Any) -> dict[str, Any]:
    """Mark a message as internal so it stays out of interactive history."""
    metadata = {
        INTERACTIVE_HISTORY_KEY: False,
        INTERNAL_PROMPT_KIND_KEY: kind,
    }
    metadata.update(extra)
    return metadata


def is_hidden_from_interactive_history(metadata: Mapping[str, Any] | None) -> bool:
    """Return whether metadata marks a message as hidden from input history."""
    if not metadata:
        return False
    return metadata.get(INTERACTIVE_HISTORY_KEY) is False
