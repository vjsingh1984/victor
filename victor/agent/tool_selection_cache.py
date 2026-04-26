from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from victor.providers.base import ToolDefinition


class SemanticToolSelectionCacheAdapter:
    """Serialization helpers for semantic tool-selection cache payloads."""

    def restore_tools(self, cached_result: dict[str, Any]) -> list[ToolDefinition] | None:
        from victor.providers.base import ToolDefinition

        tool_names = cached_result.get("tool_names", [])
        tool_descriptions = cached_result.get("tool_descriptions", {})
        tool_parameters = cached_result.get("tool_parameters", {})

        reconstructed_tools: list[ToolDefinition] = []
        for name in tool_names:
            if name not in tool_descriptions:
                return None
            reconstructed_tools.append(
                ToolDefinition(
                    name=name,
                    description=tool_descriptions[name],
                    parameters=tool_parameters.get(name, {}),
                )
            )

        return reconstructed_tools or None

    def serialize_tools(self, tools: list[ToolDefinition]) -> dict[str, Any]:
        return {
            "tool_names": [tool.name for tool in tools],
            "tool_descriptions": {tool.name: tool.description for tool in tools},
            "tool_parameters": {tool.name: tool.parameters for tool in tools},
        }
