"""Tool selection package.

Back-compat re-export shim: preserves the historical flat-module public
surface so that ``from victor.agent.tool_selection import X`` keeps working
after the F-004 package-ification. All symbols are re-exported from the
package submodules (``selector`` plus the satellite modules).
"""

from __future__ import annotations

from victor.agent.tool_selection.assembler import (
    SemanticToolSelectionAssembler,
    SemanticToolSelectionAssemblyContext,
)
from victor.agent.tool_selection.cache import SemanticToolSelectionCacheAdapter
from victor.agent.tool_selection.cache_key import SemanticToolSelectionCacheKeyBuilder
from victor.agent.tool_selection.policy import (
    StageToolSelectionContext,
    ToolSelectionStagePolicy,
)
from victor.agent.tool_selection.postprocessor import (
    ToolSelectionPostProcessContext,
    ToolSelectionPostProcessor,
)
from victor.agent.tool_selection.recorder import ToolSelectionRecorder
from victor.agent.tool_selection.selector import (
    ToolSelectionStats,
    ToolSelector,
    _enforce_token_budget,
    _estimate_tool_tokens,
    calculate_adaptive_threshold,
    detect_categories_from_message,
    get_all_categories,
    get_category_to_tools_map,
    get_critical_tools,
    get_keyword_matching_metrics,
    get_tools_by_category,
    get_tools_for_categories,
    get_tools_from_message,
    get_tools_from_message_scored,
    get_tools_with_keywords,
    get_web_tools,
    is_small_model,
    needs_web_tools,
    promote_high_confidence_stubs,
    select_tools_by_keywords,
    tool_to_definition,
)

__all__ = [
    # selector.py — public surface
    "ToolSelector",
    "ToolSelectionStats",
    "get_critical_tools",
    "get_tools_by_category",
    "get_all_categories",
    "get_category_to_tools_map",
    "detect_categories_from_message",
    "get_tools_for_categories",
    "get_web_tools",
    "get_tools_with_keywords",
    "tool_to_definition",
    "promote_high_confidence_stubs",
    "get_tools_from_message",
    "get_tools_from_message_scored",
    "get_keyword_matching_metrics",
    "is_small_model",
    "needs_web_tools",
    "calculate_adaptive_threshold",
    "select_tools_by_keywords",
    # selector.py — underscore names relied on by tests
    "_enforce_token_budget",
    "_estimate_tool_tokens",
    # satellites
    "ToolSelectionStagePolicy",
    "StageToolSelectionContext",
    "ToolSelectionPostProcessor",
    "ToolSelectionPostProcessContext",
    "SemanticToolSelectionAssembler",
    "SemanticToolSelectionAssemblyContext",
    "SemanticToolSelectionCacheAdapter",
    "SemanticToolSelectionCacheKeyBuilder",
    "ToolSelectionRecorder",
]
