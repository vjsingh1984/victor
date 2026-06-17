"""Canonical accessors for tool-selection configuration."""

from __future__ import annotations

from typing import Any


def is_semantic_tool_selection_enabled(settings: Any, default: bool = True) -> bool:
    """Return semantic-selection enablement from the canonical nested config.

    Preferred order:
    1. ``settings.tool_selection.use_semantic_tool_selection`` (canonical)
    2. ``settings.tools.use_semantic_tool_selection`` (older nested mirror)
    3. flat ``settings.use_semantic_tool_selection`` (legacy compatibility)
    """

    tool_selection = getattr(settings, "tool_selection", None)
    if tool_selection is not None and hasattr(tool_selection, "use_semantic_tool_selection"):
        return bool(tool_selection.use_semantic_tool_selection)

    tools = getattr(settings, "tools", None)
    if tools is not None and hasattr(tools, "use_semantic_tool_selection"):
        return bool(tools.use_semantic_tool_selection)

    return bool(getattr(settings, "use_semantic_tool_selection", default))


__all__ = ["is_semantic_tool_selection_enabled"]
