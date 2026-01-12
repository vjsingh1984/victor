# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Keyword-based tool selection using metadata registry.

This module provides fast keyword-based tool selection as part of HIGH-002:
Unified Tool Selection Architecture - Release 2, Phase 3.

Extracted from ToolSelector.select_keywords() to create a clean strategy
implementation of IToolSelector protocol.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from victor.providers.base import ToolDefinition
from victor.tools.base import AccessMode, ExecutionCategory, ToolRegistry
from victor.tools.selection_common import get_tools_from_message
from victor.tools.selection_filters import is_small_model

if TYPE_CHECKING:
    from victor.agent.conversation_state import ConversationStage, ConversationStateMachine
    from victor.agent.protocols import ToolSelectionContext, ToolSelectorFeatures

logger = logging.getLogger(__name__)


class KeywordToolSelector:
    """Fast keyword-based tool selection using metadata registry.

    Uses keywords defined in @tool decorators for tool selection.
    No embeddings required - <1ms selection time.

    Features:
    - Registry-based keyword matching
    - Vertical mode support (enabled_tools filtering)
    - Stage-aware filtering (read-only for analysis)
    - Small model tool capping

    HIGH-002 Release 2, Phase 3: IToolSelector strategy implementation.
    """

    def __init__(
        self,
        tools: ToolRegistry,
        conversation_state: Optional["ConversationStateMachine"] = None,
        model: str = "",
        provider_name: str = "",
        enabled_tools: Optional[Set[str]] = None,
    ):
        """Initialize keyword-based tool selector.

        Args:
            tools: Tool registry with all available tools
            conversation_state: Optional conversation state machine for stage detection
            model: Model name for small model detection
            provider_name: Provider name
            enabled_tools: Optional vertical-specific tool filter
        """
        self.tools = tools
        self.conversation_state = conversation_state
        self.model = model
        self.provider_name = provider_name
        self._enabled_tools = enabled_tools

        # Cache for core tools
        self._core_tools_cache: Optional[Set[str]] = None
        self._core_readonly_cache: Optional[Set[str]] = None

    def set_enabled_tools(self, tools: Set[str]) -> None:
        """Set the enabled tools filter for this selector.

        This allows the orchestrator to update the enabled tools filter
        after initialization (e.g., when vertical context changes).

        Args:
            tools: Set of tool names to enable
        """
        self._enabled_tools = tools
        logger.info(f"Updated enabled tools filter: {sorted(tools)}")

    def set_tiered_config(self, tiered_config: Any) -> None:
        """Set the tiered configuration for stage-aware filtering.

        This allows the orchestrator to propagate tiered tool configuration
        for vertical-specific stage filtering (e.g., analysis vs. execution).

        Args:
            tiered_config: TieredToolConfig with mandatory, vertical_core, etc.
        """
        # KeywordToolSelector doesn't use tiered config for filtering
        # (it uses _enabled_tools instead), but we accept it for API compatibility
        logger.debug(
            f"Tiered config set on KeywordToolSelector (not used): "
            f"mandatory={len(tiered_config.mandatory) if tiered_config else 0}, "
            f"vertical_core={len(tiered_config.vertical_core) if tiered_config else 0}"
        )

    async def select_tools(
        self,
        prompt: str,
        context: "ToolSelectionContext",
    ) -> List[ToolDefinition]:
        """Select tools using keyword-based category matching.

        If enabled_tools filter is set (from vertical), returns all enabled tools.
        Otherwise falls back to core tools + keyword matching.

        Args:
            prompt: User message
            context: Tool selection context (conversation history, stage, etc.)

        Returns:
            List of relevant ToolDefinition objects
        """
        all_tools = list(self.tools.list_tools())

        # Start with planned tools if provided
        selected_tools: List[ToolDefinition] = []
        existing_names: Set[str] = set()

        if context.planned_tools:
            # Convert planned tool names to ToolDefinition objects
            for tool in all_tools:
                if tool.name in context.planned_tools:
                    selected_tools.append(
                        ToolDefinition(
                            name=tool.name,
                            description=tool.description,
                            parameters=tool.parameters,
                        )
                    )
                    existing_names.add(tool.name)

        # If vertical has set enabled tools, use those directly
        # This allows verticals like "research" to specify web_search, web_fetch, etc.
        if self._enabled_tools:
            logger.info(
                f"Using vertical enabled tools ({len(self._enabled_tools)}): "
                f"{sorted(self._enabled_tools)}"
            )
            for tool in all_tools:
                if tool.name in self._enabled_tools and tool.name not in existing_names:
                    selected_tools.append(
                        ToolDefinition(
                            name=tool.name,
                            description=tool.description,
                            parameters=tool.parameters,
                        )
                    )
                    existing_names.add(tool.name)

            # Apply stage filtering and return (pass prompt for write intent detection)
            selected_tools = self._filter_tools_for_stage(
                selected_tools, context.conversation_stage, prompt
            )

            tool_names = [t.name for t in selected_tools]
            logger.info(
                f"Selected {len(selected_tools)} tools from vertical filter: "
                f"{', '.join(tool_names)}"
            )
            return selected_tools

        # Fallback: Build selected tool names using core tools + registry keyword matches
        # Uses keywords from @tool decorators as single source of truth
        selected_tool_names = self._get_stage_core_tools(context.conversation_stage).copy()

        # Use registry-based keyword matching (from @tool decorators)
        registry_matches = get_tools_from_message(prompt)
        selected_tool_names.update(registry_matches)

        # Check if this is a small model
        small_model = is_small_model(self.model, self.provider_name)

        # Filter tools
        for tool in all_tools:
            if tool.name in selected_tool_names and tool.name not in existing_names:
                selected_tools.append(
                    ToolDefinition(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.parameters,
                    )
                )
                existing_names.add(tool.name)

        # For small models, limit to max 10 tools
        if small_model and len(selected_tools) > 10:
            core_tools_set = self._get_stage_core_tools(context.conversation_stage)
            core_tools = [t for t in selected_tools if t.name in core_tools_set]
            other_tools = [t for t in selected_tools if t.name not in core_tools_set]
            selected_tools = core_tools + other_tools[: max(0, 10 - len(core_tools))]

        # Enforce read-only set during exploration/analysis stages (pass prompt for write intent)
        selected_tools = self._filter_tools_for_stage(
            selected_tools, context.conversation_stage, prompt
        )

        tool_names = [t.name for t in selected_tools]
        logger.info(
            f"Selected {len(selected_tools)} tools (small_model={small_model}): "
            f"{', '.join(tool_names)}"
        )

        return selected_tools

    def get_supported_features(self) -> "ToolSelectorFeatures":
        """Return features supported by keyword tool selector.

        Returns:
            ToolSelectorFeatures with only basic keyword matching enabled
        """
        from victor.agent.protocols import ToolSelectorFeatures

        return ToolSelectorFeatures(
            supports_semantic_matching=False,
            supports_context_awareness=False,
            supports_cost_optimization=False,
            supports_usage_learning=False,
            supports_workflow_patterns=False,
            requires_embeddings=False,
        )

    def record_tool_execution(
        self,
        tool_name: str,
        success: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record tool execution (no-op for keyword selector).

        Keyword selector doesn't track usage statistics or learn from executions.

        Args:
            tool_name: Name of the tool that was executed
            success: Whether the execution succeeded
            context: Optional execution context
        """
        pass  # No learning in keyword selector

    async def close(self) -> None:
        """Cleanup resources (no-op for keyword selector).

        Keyword selector has no resources to clean up.
        """
        pass  # No resources to clean up

    # =========================================================================
    # Helper Methods (extracted from ToolSelector)
    # =========================================================================

    def _get_stage_core_tools(self, stage: Optional["ConversationStage"]) -> Set[str]:
        """Choose core set based on stage (safe for exploration/analysis).

        Args:
            stage: Current conversation stage

        Returns:
            Set of core tool names appropriate for the stage
        """
        if stage is None:
            return self._get_core_tools_cached()

        from victor.agent.conversation_state import ConversationStage

        if stage in {
            ConversationStage.INITIAL,
            ConversationStage.PLANNING,
            ConversationStage.READING,
            ConversationStage.ANALYSIS,
        }:
            return self._get_core_readonly_cached()
        return self._get_core_tools_cached()

    def _get_core_tools_cached(self) -> Set[str]:
        """Get critical tools with caching.

        Returns:
            Set of critical tool names
        """
        if self._core_tools_cache is None:
            from victor.tools.selection_common import get_critical_tools

            self._core_tools_cache = get_critical_tools(self.tools)
        return self._core_tools_cache.copy()

    def _get_core_readonly_cached(self) -> Set[str]:
        """Get core read-only tools with caching.

        Returns:
            Set of core read-only tool names
        """
        if self._core_readonly_cache is None:
            from victor.tools.metadata_registry import get_core_readonly_tools

            # Convert to set since get_core_readonly_tools returns a list
            self._core_readonly_cache = set(get_core_readonly_tools())
        return self._core_readonly_cache.copy()

    def _is_readonly_tool(self, tool_name: str) -> bool:
        """Check if a tool is readonly via metadata registry.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if tool is read-only
        """
        try:
            from victor.tools.metadata_registry import get_global_registry

            entry = get_global_registry().get(tool_name)
            if not entry:
                return False
            return (
                entry.access_mode == AccessMode.READONLY
                or entry.execution_category == ExecutionCategory.READ_ONLY
            )
        except ImportError as e:
            logger.debug(f"Metadata registry module not available for readonly check: {e}")
            return False
        except Exception as e:
            logger.debug(
                f"Failed to check if tool {tool_name} is readonly: {e}",
                extra={"tool_name": tool_name},
            )
            return False

    def _has_write_intent(self, prompt: str) -> bool:
        """Check if prompt indicates write intent.

        Args:
            prompt: User message

        Returns:
            True if write intent is detected
        """
        prompt_lower = prompt.lower()
        write_keywords = [
            "create",
            "write",
            "add",
            "generate",
            "make",
            "build",
            "implement",
            "edit",
            "modify",
            "update",
            "save",
            "fix",
            "change",
            "replace",
            "insert",
            "delete",
            "remove",
            "refactor",
        ]
        return any(kw in prompt_lower for kw in write_keywords)

    def _filter_tools_for_stage(
        self, tools: List[ToolDefinition], stage: Optional["ConversationStage"], prompt: str = ""
    ) -> List[ToolDefinition]:
        """Remove write/execute tools during exploration/analysis stages.

        Note: Vertical core tools would be preserved if tiered config was available,
        but KeywordToolSelector doesn't have access to that (it's in ToolSelector).

        Args:
            tools: List of tool definitions
            stage: Current conversation stage
            prompt: User message (used to detect write intent)

        Returns:
            Filtered list of tools
        """
        if stage is None:
            return tools

        # Skip stage filtering if user has write intent
        if prompt and self._has_write_intent(prompt):
            logger.info("Write intent detected in prompt, skipping stage-based filtering")
            return tools

        from victor.agent.conversation_state import ConversationStage

        if stage not in {
            ConversationStage.INITIAL,
            ConversationStage.PLANNING,
            ConversationStage.READING,
            ConversationStage.ANALYSIS,
        }:
            return tools

        # Filter to readonly tools
        filtered = [t for t in tools if self._is_readonly_tool(t.name)]

        if filtered:
            return filtered

        # Fallback to core readonly if filtering removed everything
        readonly_core = self._get_stage_core_tools(stage)
        fallback: List[ToolDefinition] = []
        for tool in tools:
            if tool.name in readonly_core:
                fallback.append(tool)

        if fallback:
            logger.debug(
                f"Stage filtering fallback: {len(fallback)} core readonly tools "
                f"from {len(tools)} original"
            )
            return fallback

        # Last resort: return first few tools
        logger.warning(
            f"Stage filtering: no readonly tools found, returning first {min(5, len(tools))} tools"
        )
        return tools[:5]
