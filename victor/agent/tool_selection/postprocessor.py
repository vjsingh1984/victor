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
    # Canonical web-tool names (e.g. web_search). The order-blind edge filter / truncation
    # below must not drop a web tool the selector deliberately included — for a research-
    # flavored task that ranked web_search just below the cap, dropping it leaves the model
    # told about a tool it can't call. Empty by default (no protection).
    web_tools: frozenset[str] = frozenset()
    # Cap behavior past ``fallback_max_tools`` (tool-supply P2):
    #   "hard" — legacy: drop the over-cap tail (kept as an escape hatch).
    #   "stub" — keep ALL tools; demote the over-cap tail to STUB schema instead of
    #            dropping, so a relevant tool is never made invisible to the model.
    #   "none" — no cap (caching providers: the full set is nearly free).
    # Default "hard" preserves prior behavior for direct callers; the selector passes
    # "stub" so the live path stops silently dropping ranked tools.
    cap_mode: str = "hard"


class ToolSelectionPostProcessor:
    """Apply post-selection transforms after semantic/keyword selection."""

    @staticmethod
    def _truncate_preserving_web(
        tools: list[ToolDefinition],
        max_tools: int,
        web_tools: frozenset[str],
    ) -> list[ToolDefinition]:
        """Truncate to ``max_tools`` without dropping a selected web tool.

        The base ``tools[:max_tools]`` is order-blind, so a web tool ranked just below the
        cap is sliced off. Keep every candidate web tool, fill the remaining budget with the
        top-ranked non-web tools, and preserve the original ordering. If web tools alone
        exceed ``max_tools`` (rare — there are only a few), they are all kept.
        """
        if not web_tools or not any(t.name in web_tools for t in tools):
            return tools[:max_tools]
        web = [t for t in tools if t.name in web_tools]
        non_web = [t for t in tools if t.name not in web_tools]
        keep_non_web = max(0, max_tools - len(web))
        keep_names = {t.name for t in web} | {t.name for t in non_web[:keep_non_web]}
        return [t for t in tools if t.name in keep_names]

    @staticmethod
    def _apply_cap(
        tools: list[ToolDefinition],
        cap_mode: str,
        max_tools: int,
        web_tools: frozenset[str],
    ) -> list[ToolDefinition]:
        """Apply the over-cap policy (tool-supply P2).

        - ``none``: return every tool unchanged (no cap).
        - ``stub``: keep the top ``max_tools`` (web-preserving) at their current schema
          and demote the rest to STUB — returning **all** tools so a relevant tool is
          never dropped, only made terser. STUB schemas (~25-40 tokens) keep it callable;
          the downstream token-budget step is the backstop.
        - ``hard`` (default): legacy truncation — drop the over-cap tail.
        """
        if cap_mode == "none" or len(tools) <= max_tools:
            return tools
        if cap_mode == "stub":
            keep = ToolSelectionPostProcessor._truncate_preserving_web(tools, max_tools, web_tools)
            keep_names = {t.name for t in keep}
            for t in tools:
                if t.name not in keep_names:
                    try:
                        t.schema_level = "stub"
                    except Exception:  # never let schema-tier marking break selection
                        pass
            return tools
        # "hard" — legacy behavior.
        return ToolSelectionPostProcessor._truncate_preserving_web(tools, max_tools, web_tools)

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
        web_tools = context.web_tools or frozenset()

        if len(result) > 8 and should_use_edge_filter and apply_edge_filter is not None:
            pre_web = [t for t in result if t.name in web_tools]
            result = apply_edge_filter(result, context.user_message, context.stage)
            # Re-attach any deliberately-selected web tool the relevance filter dropped; the
            # budget-aware truncation below makes room for it.
            have = {t.name for t in result}
            result = result + [t for t in pre_web if t.name not in have]

        result = cap_mcp_tools(result, context.max_mcp_tools)

        result = self._apply_cap(result, context.cap_mode, context.fallback_max_tools, web_tools)

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
