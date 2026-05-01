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

"""Tool selector service implementation.

Handles tool selection logic, filtering, and validation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Set

if TYPE_CHECKING:
    from victor.tools.base import BaseTool

logger = logging.getLogger(__name__)


class ToolSelectorServiceConfig:
    """Configuration for ToolSelectorService.

    Attributes:
        default_enabled_tools: Set of tools enabled by default
        enable_hallucination_filter: Filter hallucinated tools
        max_selected_tools: Maximum tools to select for a query
    """

    def __init__(
        self,
        default_enabled_tools: Set[str] | None = None,
        enable_hallucination_filter: bool = True,
        max_selected_tools: int = 10,
    ):
        self.default_enabled_tools = default_enabled_tools or set()
        self.enable_hallucination_filter = enable_hallucination_filter
        self.max_selected_tools = max_selected_tools


class ToolSelectorService:
    """Service for tool selection and filtering.

    Responsible for:
    - Tool selection based on query and context
    - Tool enable/disable management
    - Hallucinated tool filtering
    - Tool alias resolution

    This service does NOT handle:
    - Tool execution (delegated to ToolExecutorService)
    - Budget tracking (delegated to ToolTrackerService)
    - Execution planning (delegated to ToolPlannerService)
    - Result processing (delegated to ToolResultProcessor)

    Example:
        config = ToolSelectorServiceConfig()
        selector = ToolSelectorService(
            config=config,
            available_tools={"search", "read_file", "write_file"}
        )

        # Select tools for query
        selected = await selector.select_tools(
            "Search for files",
            available_tools={"search", "read_file", "write_file"}
        )

        # Check if tool enabled
        if selector.is_tool_enabled("search"):
            # Use search tool
            ...
    """

    def __init__(
        self,
        config: ToolSelectorServiceConfig,
        available_tools: Set[str],
        tool_registry: Dict[str, "BaseTool"] | None = None,
    ):
        """Initialize ToolSelectorService.

        Args:
            config: Service configuration
            available_tools: Set of available tool names
            tool_registry: Optional registry of tool instances
        """
        self.config = config
        self.available_tools = available_tools
        self.tool_registry = tool_registry or {}

        # Enabled tools management
        self._enabled_tools: Set[str] = set(config.default_enabled_tools)

        # Tool aliases
        self._tool_aliases: Dict[str, str] = {}

        # Health tracking
        self._healthy = True

    async def select_tools(
        self,
        query: str,
        available_tools: Set[str],
        context: Dict[str, Any] | None = None,
    ) -> List[str]:
        """Select tools based on query and context.

        Uses keyword matching and simple heuristics to select relevant tools.
        Can be enhanced with LLM-based selection in the future.

        Args:
            query: User query or task description
            available_tools: Set of available tool names
            context: Optional context for selection

        Returns:
            List of selected tool names (ordered by relevance)
        """
        # Filter to enabled tools
        enabled_tools = [t for t in available_tools if self.is_tool_enabled(t)]

        # Simple keyword-based selection
        query_lower = query.lower()
        scored_tools = []

        for tool_name in enabled_tools:
            score = self._score_tool_relevance(tool_name, query_lower, context)
            if score > 0:
                scored_tools.append((tool_name, score))

        # Sort by score (descending)
        scored_tools.sort(key=lambda x: x[1], reverse=True)

        # Return top N tools
        selected = [tool_name for tool_name, _ in scored_tools[: self.config.max_selected_tools]]

        logger.debug(f"Selected {len(selected)} tools for query: {selected}")

        return selected

    def _score_tool_relevance(
        self, tool_name: str, query_lower: str, context: Dict[str, Any] | None
    ) -> float:
        """Score tool relevance to query.

        Args:
            tool_name: Name of the tool
            query_lower: Lowercase query string
            context: Optional context

        Returns:
            Relevance score (0-1)
        """
        score = 0.0

        # Direct name match
        if tool_name.lower() in query_lower:
            score += 0.8

        # Partial name match
        for word in tool_name.split("_"):
            if word.lower() in query_lower:
                score += 0.3

        # Check tool description if available
        if tool_name in self.tool_registry:
            tool = self.tool_registry[tool_name]
            if hasattr(tool, "description") and tool.description:
                desc_lower = tool.description.lower()
                if any(word in desc_lower for word in query_lower.split()):
                    score += 0.2

        return min(score, 1.0)

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is enabled.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if tool is enabled, False otherwise
        """
        # Resolve alias first
        resolved_name = self.resolve_tool_alias(tool_name)
        return resolved_name in self._enabled_tools

    def get_enabled_tools(self) -> Set[str]:
        """Get set of enabled tools.

        Returns:
            Set of enabled tool names
        """
        return set(self._enabled_tools)

    def set_enabled_tools(self, tools: Set[str]) -> None:
        """Set the enabled tools.

        Args:
            tools: Set of tool names to enable
        """
        # Validate tools are available
        invalid_tools = tools - self.available_tools
        if invalid_tools:
            logger.warning(f"Cannot enable unavailable tools: {invalid_tools}")

        # Set enabled tools (only available ones)
        self._enabled_tools = tools & self.available_tools

        logger.debug(f"Enabled tools: {len(self._enabled_tools)} tools")

    def enable_tool(self, tool_name: str) -> None:
        """Enable a specific tool.

        Args:
            tool_name: Name of the tool to enable
        """
        if tool_name in self.available_tools:
            self._enabled_tools.add(tool_name)
            logger.debug(f"Enabled tool: {tool_name}")
        else:
            logger.warning(f"Cannot enable unavailable tool: {tool_name}")

    def disable_tool(self, tool_name: str) -> None:
        """Disable a specific tool.

        Args:
            tool_name: Name of the tool to disable
        """
        self._enabled_tools.discard(tool_name)
        logger.debug(f"Disabled tool: {tool_name}")

    def filter_hallucinated_tools(
        self, tool_calls: List[Dict[str, Any]], known_tools: Set[str]
    ) -> List[Dict[str, Any]]:
        """Filter out hallucinated tool calls.

        Args:
            tool_calls: List of tool calls to filter
            known_tools: Set of known tool names

        Returns:
            Filtered list of tool calls
        """
        if not self.config.enable_hallucination_filter:
            return tool_calls

        filtered = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")

            # Check if tool is known
            if tool_name in known_tools:
                filtered.append(tool_call)
            else:
                logger.warning(f"Filtered hallucinated tool: {tool_name}")

        if len(filtered) < len(tool_calls):
            logger.info(f"Filtered {len(tool_calls) - len(filtered)} hallucinated tools")

        return filtered

    def resolve_tool_alias(self, tool_name: str) -> str:
        """Resolve tool alias to canonical name.

        Args:
            tool_name: Tool name or alias

        Returns:
            Canonical tool name
        """
        return self._tool_aliases.get(tool_name, tool_name)

    def register_tool_alias(self, alias: str, canonical_name: str) -> None:
        """Register a tool alias.

        Args:
            alias: Alias name
            canonical_name: Canonical tool name
        """
        if canonical_name not in self.available_tools:
            logger.warning(f"Cannot register alias for unavailable tool: {canonical_name}")
            return

        self._tool_aliases[alias] = canonical_name
        logger.debug(f"Registered alias: {alias} -> {canonical_name}")

    def is_healthy(self) -> bool:
        """Check if service is healthy.

        Returns:
            True if healthy, False otherwise
        """
        return self._healthy
