from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:
    from victor.providers.base import ToolDefinition


@dataclass(frozen=True)
class SemanticToolSelectionAssemblyContext:
    max_tools: int
    include_web_tools: bool
    web_tool_names: set[str]


class SemanticToolSelectionAssembler:
    """Combine semantic results with keyword and explicit web-tool expansion."""

    def assemble(
        self,
        semantic_tools: Sequence[ToolDefinition],
        *,
        keyword_tools: Sequence[ToolDefinition],
        all_tools: Iterable[ToolDefinition],
        context: SemanticToolSelectionAssemblyContext,
    ) -> list[ToolDefinition]:
        from victor.providers.base import ToolDefinition

        tools = list(semantic_tools)

        if keyword_tools:
            existing = {tool.name for tool in tools}
            new_keyword_tools = [tool for tool in keyword_tools if tool.name not in existing]
            max_keyword_additions = max(3, context.max_tools - len(tools))
            if len(new_keyword_tools) > max_keyword_additions:
                new_keyword_tools = new_keyword_tools[:max_keyword_additions]
            tools.extend(new_keyword_tools)

        if context.include_web_tools:
            existing = {tool.name for tool in tools}
            for tool in all_tools:
                if tool.name in context.web_tool_names and tool.name not in existing:
                    tools.append(
                        ToolDefinition(
                            name=tool.name,
                            description=tool.description,
                            parameters=tool.parameters,
                        )
                    )
                    existing.add(tool.name)

        deduped: dict[str, ToolDefinition] = {}
        for tool in tools:
            deduped[tool.name] = tool
        return list(deduped.values())
