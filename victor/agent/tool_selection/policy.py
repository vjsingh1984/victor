from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable, Sequence

if TYPE_CHECKING:
    from victor.agent.conversation.state_machine import ConversationStage
    from victor.providers.base import ToolDefinition


@dataclass(frozen=True)
class StageToolSelectionContext:
    current_stage: ConversationStage
    stage_tools: set[str]
    core_tools: set[str]
    web_tools: set[str]
    mandatory_tools: set[str]
    vertical_core_tools: set[str]


class ToolSelectionStagePolicy:
    """Pure stage/fallback policy extracted from ToolSelector.

    The selector remains the orchestration entry point, but stage-pruning and
    semantic-fallback assembly now live in a small, testable unit that accepts
    explicit state-passed inputs.
    """

    def __init__(self, fallback_max_tools: int = 8) -> None:
        self._fallback_max_tools = fallback_max_tools

    def prioritize_by_stage(
        self,
        tools: Sequence[ToolDefinition],
        *,
        context: StageToolSelectionContext,
        should_include_tool: Callable[[str], bool],
        get_tool_priority_boost: Callable[[str], float],
    ) -> list[ToolDefinition]:
        keep = set(context.stage_tools)
        keep.update(context.core_tools)
        keep.update(context.web_tools)
        keep.update(context.mandatory_tools)
        keep.update(context.vertical_core_tools)

        for tool in tools:
            if should_include_tool(tool.name):
                keep.add(tool.name)

        boosted_tools: list[tuple[ToolDefinition, float]] = []
        for tool in tools:
            boost = get_tool_priority_boost(tool.name)
            if tool.name in keep or boost > 0:
                boosted_tools.append((tool, boost))

        boosted_tools.sort(key=lambda item: item[1], reverse=True)
        pruned = [tool for tool, _ in boosted_tools if tool.name in keep]
        if pruned:
            return pruned

        fallback_names = set(context.core_tools)
        fallback_names.update(context.mandatory_tools)
        fallback_names.update(context.vertical_core_tools)
        fallback = [tool for tool in tools if tool.name in fallback_names]
        if fallback:
            return fallback

        return list(tools[: self._fallback_max_tools])

    def build_semantic_fallback_tools(
        self,
        *,
        all_tools: Iterable[ToolDefinition],
        core_tools: set[str],
        keyword_tools: Sequence[ToolDefinition],
    ) -> list[ToolDefinition]:
        from victor.providers.base import ToolDefinition

        selected: list[ToolDefinition] = []
        existing_names: set[str] = set()

        for tool in all_tools:
            if tool.name not in core_tools or tool.name in existing_names:
                continue
            selected.append(
                ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                )
            )
            existing_names.add(tool.name)

        for tool in keyword_tools:
            if tool.name in existing_names:
                continue
            selected.append(tool)
            existing_names.add(tool.name)

        if len(selected) > self._fallback_max_tools:
            return selected[: self._fallback_max_tools]

        return selected
