from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

if TYPE_CHECKING:
    from victor.agent.conversation.state_machine import ConversationStage
    from victor.providers.base import ToolDefinition


@dataclass(frozen=True)
class ToolSelectionPostProcessContext:
    user_message: str
    stage: ConversationStage | None
    fallback_max_tools: int
    max_mcp_tools: int
    schema_promotion_threshold: float
    max_schema_tokens: int


class ToolSelectionPostProcessor:
    """Apply post-selection transforms after semantic/keyword selection."""

    def apply(
        self,
        tools: Sequence[ToolDefinition],
        *,
        context: ToolSelectionPostProcessContext,
        should_use_edge_filter: bool,
        cap_mcp_tools: Callable[[list[ToolDefinition], int], list[ToolDefinition]],
        apply_edge_filter: (
            Callable[
                [list[ToolDefinition], str, ConversationStage | None],
                list[ToolDefinition],
            ]
            | None
        ) = None,
        selection_scores: Mapping[str, float] | None = None,
        promote_schema_stubs: (
            Callable[
                [list[ToolDefinition], Mapping[str, float], float],
                list[ToolDefinition],
            ]
            | None
        ) = None,
        enforce_token_budget: (
            Callable[[list[ToolDefinition], int], list[ToolDefinition]] | None
        ) = None,
    ) -> list[ToolDefinition]:
        result = list(tools)

        if len(result) > 8 and should_use_edge_filter and apply_edge_filter is not None:
            result = apply_edge_filter(result, context.user_message, context.stage)

        result = cap_mcp_tools(result, context.max_mcp_tools)

        if len(result) > context.fallback_max_tools:
            result = result[: context.fallback_max_tools]

        if (
            selection_scores
            and promote_schema_stubs is not None
            and context.schema_promotion_threshold > 0
        ):
            result = promote_schema_stubs(
                result,
                selection_scores,
                context.schema_promotion_threshold,
            )

        if enforce_token_budget is not None and context.max_schema_tokens > 0:
            result = enforce_token_budget(result, context.max_schema_tokens)

        return result
