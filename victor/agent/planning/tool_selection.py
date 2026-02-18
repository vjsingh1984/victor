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

"""Context-aware tool selection for TaskPlanner steps.

This module provides intelligent tool selection based on:
1. Step type (research, feature, test, deploy, etc.)
2. Task complexity (simple, moderate, complex)
3. Conversation stage (initial, reading, executing, verifying)
4. Existing task-type configuration

This enables progressive tool disclosure where LLMs only see relevant tools
for the current step, reducing token usage and improving focus.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from victor.agent.conversation_state import ConversationStage
from victor.agent.planning.readable_schema import TaskComplexity
from victor.agent.tool_selection import ToolSelector, get_critical_tools
from victor.agent.task_tool_config_loader import TaskToolConfigLoader

if TYPE_CHECKING:
    from victor.providers.base import ToolDefinition
    from victor.tools.base import ToolRegistry

logger = logging.getLogger(__name__)


# Step type to tool set mapping
# These are the canonical tool sets for each step type
STEP_TOOL_MAPPING: Dict[str, Set[str]] = {
    # Research steps need read-only exploration tools
    "research": {
        "read",  # Read file contents
        "grep",  # Search for patterns
        "code_search",  # Semantic code search
        "overview",  # Get project overview
        "ls",  # List directories
        "git_readonly",  # Read git history
    },
    # Planning needs exploration + analysis tools
    "planning": {
        "read",
        "grep",
        "code_search",
        "overview",
        "ls",
        "analyze",  # Code analysis
    },
    # Feature implementation needs full toolset
    "feature": {
        "read",
        "write",  # Write files
        "edit",  # Edit files
        "grep",
        "test",  # Run tests
        "code_search",
        "git",  # Git operations
        "shell",  # Execute commands
    },
    # Bugfix needs debugging tools
    "bugfix": {
        "read",
        "grep",
        "edit",
        "test",
        "debugger",  # Debugging tools
        "code_search",
        "shell",
    },
    # Refactor needs code analysis + modification
    "refactor": {
        "read",
        "edit",
        "grep",
        "test",
        "analyze",
        "code_search",
    },
    # Testing needs test execution tools
    "test": {
        "test",
        "read",
        "grep",
        "shell_readonly",  # Read-only shell for test commands
    },
    # Review needs read-only + analysis
    "review": {
        "read",
        "grep",
        "analyze",
        "lint",  # Linting tools
        "code_search",
    },
    # Deploy needs deployment + verification tools
    "deploy": {
        "shell",
        "git",
        "docker",  # Docker operations
        "kubectl",  # Kubernetes operations
        "read",
        "test",
    },
    # Analyze needs exploration tools
    "analyze": {
        "read",
        "grep",
        "code_search",
        "overview",
        "analyze",
        "shell_readonly",
    },
    # Doc needs reading + minimal writing
    "doc": {
        "read",
        "grep",
        "write",
        "code_search",
    },
}


# Complexity-based tool limits
# Simple tasks get minimal tools, complex tasks get comprehensive sets
COMPLEXITY_TOOL_LIMITS: Dict[str, int] = {
    "simple": 5,  # Auto mode, minimal tools for quick execution
    "moderate": 10,  # Plan-mode, balanced tools
    "complex": 15,  # Plan-mode, comprehensive tools for complex tasks
}


# Step type to task type mapping
# Maps planning step types to task_tool_config task types
STEP_TO_TASK_TYPE: Dict[str, str] = {
    "research": "search",
    "planning": "design",
    "feature": "create",
    "bugfix": "edit",
    "refactor": "edit",
    "test": "create",
    "review": "analyze",
    "deploy": "create",
    "analyze": "analyze",
    "doc": "create",
}


class StepAwareToolSelector:
    """Selects tools based on TaskPlanner step types.

    This class bridges the gap between:
    - TaskPlanner's step-based planning (research, feature, test, deploy)
    - Existing tool selection infrastructure (ToolSelector, TaskToolConfigLoader)
    - Task-type aware tool configuration (edit, search, create, analyze)

    The selector enables progressive tool disclosure where LLMs only see
    relevant tools for the current step, providing:
    - 50-80% reduction in tool schema tokens
    - Improved LLM focus and reduced hallucination
    - Better alignment between step goals and available capabilities

    Example:
        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
            task_config_loader=task_config_loader,
        )

        # Get tools for a research step
        tools = selector.get_tools_for_step(
            step_type="research",
            complexity=TaskComplexity.MODERATE,
            step_description="Analyze authentication patterns",
            conversation_stage=ConversationStage.READING,
        )
    """

    def __init__(
        self,
        tool_selector: ToolSelector,
        task_config_loader: Optional[TaskToolConfigLoader] = None,
    ):
        """Initialize the step-aware tool selector.

        Args:
            tool_selector: Base tool selector for tool registry access
            task_config_loader: Optional task config loader for stage-based tools
        """
        self.tool_selector = tool_selector
        self.task_config_loader = task_config_loader or TaskToolConfigLoader()

        # Cache for tool sets to avoid recomputation
        self._tool_set_cache: Dict[tuple, List["ToolDefinition"]] = {}

    def get_tools_for_step(
        self,
        step_type: str,
        complexity: TaskComplexity,
        step_description: str,
        conversation_stage: Optional[ConversationStage] = None,
    ) -> List["ToolDefinition"]:
        """Get context-appropriate tools for a planning step.

        This method performs multi-stage tool selection:
        1. Get base tool set for step type from STEP_TOOL_MAPPING
        2. Add critical tools (always available)
        3. Add task-type specific tools from TaskToolConfigLoader
        4. Filter to available tools in registry
        5. Apply complexity-based limits
        6. Prioritize core tools when limiting

        Args:
            step_type: Step type from plan (e.g., "research", "feature", "test")
            complexity: Task complexity level (simple, moderate, complex)
            step_description: What this step does (for semantic matching)
            conversation_stage: Optional conversation stage for stage-based filtering

        Returns:
            List of ToolDefinition objects appropriate for this step
        """
        # Check cache
        cache_key = (
            step_type,
            complexity.value,
            conversation_stage.name if conversation_stage else None,
        )
        if cache_key in self._tool_set_cache:
            logger.debug(f"Using cached tool set for step_type={step_type}")
            return self._tool_set_cache[cache_key]

        from victor.providers.base import ToolDefinition

        # 1. Get base tool set for step type
        base_tools = STEP_TOOL_MAPPING.get(step_type, set())

        # 2. Always include critical tools
        critical_tools = get_critical_tools(self.tool_selector.tools)
        step_tools = base_tools | critical_tools

        # 3. Add task-type specific tools from config
        task_type = STEP_TO_TASK_TYPE.get(step_type, "general")
        stage_name = conversation_stage.name.lower() if conversation_stage else "initial"
        task_stage_tools = self.task_config_loader.get_stage_tools(task_type, stage_name)
        step_tools.update(task_stage_tools)

        # 4. Get complexity limit
        max_tools = COMPLEXITY_TOOL_LIMITS[complexity.value]

        # 5. Filter available tools to step-relevant set
        available_tools = self._filter_by_step_type(
            self.tool_selector.tools,
            step_tools,
            step_description,
        )

        # 6. Apply complexity limit with prioritization
        if len(available_tools) > max_tools:
            available_tools = self._prioritize_core_tools(
                available_tools,
                base_tools,
                max_tools,
            )

        # Cache result
        self._tool_set_cache[cache_key] = available_tools

        logger.info(
            f"Step-aware tool selection for {step_type}/{complexity.value}: "
            f"{len(available_tools)} tools (max={max_tools}): "
            f"{', '.join(t.name for t in available_tools)}"
        )

        return available_tools

    def _filter_by_step_type(
        self,
        tools: "ToolRegistry",
        step_tools: Set[str],
        description: str,
    ) -> List["ToolDefinition"]:
        """Filter registry tools to step-relevant set.

        Args:
            tools: Tool registry to filter from
            step_tools: Set of tool names to include
            description: Step description for keyword matching

        Returns:
            List of ToolDefinition objects matching step requirements
        """
        from victor.providers.base import ToolDefinition

        result = []
        all_tools_map = {t.name: t for t in tools.list_tools(only_enabled=True)}
        description_lower = description.lower()

        for tool_name, tool in all_tools_map.items():
            # Include if in step tool set
            if tool_name in step_tools:
                result.append(
                    ToolDefinition(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.parameters,
                    )
                )
                continue

            # Also include if tool keywords match description
            # (catches tools not explicitly in step_tools but relevant)
            if hasattr(tool, "metadata") and tool.metadata:
                keywords = getattr(tool.metadata, "keywords", []) or []
                if any(kw.lower() in description_lower for kw in keywords):
                    result.append(
                        ToolDefinition(
                            name=tool.name,
                            description=tool.description,
                            parameters=tool.parameters,
                        )
                    )

        return result

    def _prioritize_core_tools(
        self,
        tools: List["ToolDefinition"],
        base_tools: Set[str],
        max_tools: int,
    ) -> List["ToolDefinition"]:
        """Prioritize core tools when limiting tool count.

        Args:
            tools: List of tools to limit
            base_tools: Base tool set for this step type
            max_tools: Maximum number of tools to return

        Returns:
            Prioritized and limited list of tools
        """
        # Get critical tools
        critical_tools = get_critical_tools(self.tool_selector.tools)

        # Categorize tools
        critical = [t for t in tools if t.name in critical_tools]
        base = [t for t in tools if t.name in base_tools and t not in critical]
        others = [t for t in tools if t not in critical and t not in base]

        # Build result with priority ordering
        result = critical + base

        # Fill remaining slots with others
        remaining_slots = max(0, max_tools - len(result))
        result.extend(others[:remaining_slots])

        logger.debug(
            f"Tool prioritization: {len(result)} tools "
            f"(critical={len(critical)}, base={len(base)}, others={remaining_slots})"
        )

        return result

    def invalidate_cache(self) -> None:
        """Invalidate the tool set cache.

        Call this when:
        - Tools are dynamically added/removed
        - Tool configuration changes
        - Need fresh tool selection
        """
        self._tool_set_cache.clear()
        logger.debug("Step-aware tool selector cache invalidated")

    def get_step_tool_summary(
        self,
        step_type: str,
        complexity: TaskComplexity,
    ) -> Dict[str, any]:
        """Get summary of tools for a step type and complexity.

        Useful for debugging and observability.

        Args:
            step_type: Step type to summarize
            complexity: Task complexity level

        Returns:
            Dictionary with tool selection summary
        """
        base_tools = STEP_TOOL_MAPPING.get(step_type, set())
        critical_tools = get_critical_tools(self.tool_selector.tools)
        max_tools = COMPLEXITY_TOOL_LIMITS[complexity.value]

        return {
            "step_type": step_type,
            "complexity": complexity.value,
            "base_tools": sorted(base_tools),
            "critical_tools": sorted(critical_tools),
            "max_tools": max_tools,
            "total_available": len(base_tools | critical_tools),
        }

    def map_step_type_to_task_type(self, step_type: str) -> str:
        """Map planning step type to task type for config lookup.

        Args:
            step_type: Step type from plan

        Returns:
            Task type string for TaskToolConfigLoader
        """
        return STEP_TO_TASK_TYPE.get(step_type, "general")


def get_step_tool_sets() -> Dict[str, Set[str]]:
    """Get the step tool mapping configuration.

    Returns:
        Copy of STEP_TOOL_MAPPING for inspection
    """
    return STEP_TOOL_MAPPING.copy()


def get_complexity_limits() -> Dict[str, int]:
    """Get the complexity-based tool limits.

    Returns:
        Copy of COMPLEXITY_TOOL_LIMITS for inspection
    """
    return COMPLEXITY_TOOL_LIMITS.copy()


# Legacy aliases for backward compatibility
StepToolMapping = STEP_TOOL_MAPPING
ComplexityToolLimits = COMPLEXITY_TOOL_LIMITS
