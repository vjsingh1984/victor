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

"""Tool planning coordinator for intelligent tool selection and filtering.

This module provides a centralized interface for tool planning operations:
- Planning tool sequences to achieve goals
- Inferring goals from user messages
- Filtering tools based on user intent

Extracted from CRITICAL-001 Phase 2C: Extract ToolPlanner
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.tool_registrar import ToolRegistrar
    from victor.agent.action_authorizer import ActionIntent
    from victor.tools.base import ToolDefinition
    from victor.config.settings import Settings

from victor.core.events import ObservabilityBus

logger = logging.getLogger(__name__)


class ToolPlanner:
    """Coordinates tool planning, goal inference, and intent-based filtering.

    This component provides a semantic interface for tool planning operations,
    consolidating methods that were previously in AgentOrchestrator.

    Architecture:
    - ToolPlanner: High-level coordinator for tool planning operations
    - ToolRegistrar: Tool dependency graph and planning logic
    - ActionAuthorizer: Intent detection and tool filtering rules
    - AgentOrchestrator: Uses ToolPlanner for all tool planning

    Responsibilities:
    - Plan tool sequences using dependency graph
    - Infer goals from user messages
    - Filter tools based on detected user intent
    - Provide clean, semantic API for tool planning

    Design Pattern:
    - Coordinator/Facade: Simplifies tool planning interface
    - Delegation: Delegates to ToolRegistrar and ActionAuthorizer

    Extracted from CRITICAL-001 Phase 2C.
    """

    def __init__(
        self,
        tool_registrar: "ToolRegistrar",
        settings: "Settings",
        event_bus: Optional[ObservabilityBus] = None,
    ):
        """Initialize ToolPlanner.

        Args:
            tool_registrar: Tool registry and dependency graph manager
            settings: Application settings
            event_bus: Optional ObservabilityBus instance. If None, uses DI container.
        """
        self.tool_registrar = tool_registrar
        self.settings = settings
        self._event_bus = event_bus or self._get_default_bus()

    def _get_default_bus(self) -> Optional[ObservabilityBus]:
        """Get default ObservabilityBus from DI container.

        Returns:
            ObservabilityBus instance or None if unavailable
        """
        try:
            from victor.core.events import get_observability_bus

            return get_observability_bus()
        except Exception:
            return None

    # =====================================================================
    # Tool Planning
    # =====================================================================

    def plan_tools(
        self, goals: List[str], available_inputs: Optional[List[str]] = None
    ) -> List["ToolDefinition"]:
        """Plan a sequence of tools to satisfy goals using the dependency graph.

        Uses the tool dependency graph to determine which tools need to be
        executed in what order to produce the desired outputs (goals).

        Args:
            goals: List of desired outputs (e.g., ["summary", "documentation"])
            available_inputs: Optional list of inputs already available

        Returns:
            List of ToolDefinition objects representing the planned tool sequence
        """
        planned_tools = self.tool_registrar.plan_tools(goals, available_inputs)

        # Emit TOOL event for planning decision
        if self._event_bus:
            try:
                import asyncio

                asyncio.run(
                    self._event_bus.emit(
                        topic="tool.planned",
                        data={
                            "goals": goals,
                            "planned_count": len(planned_tools),
                            "tool_names": [t.name for t in planned_tools] if planned_tools else [],
                            "category": "tool",  # Preserve for observability
                        },
                        source="ToolPlanner",
                    )
                )
            except Exception as e:
                logger.debug(f"Failed to emit tool planned event: {e}")

        return planned_tools

    def infer_goals_from_message(self, user_message: str) -> List[str]:
        """Infer planning goals from the user request.

        Analyzes the user's message to determine what outputs they want,
        which can then be used with plan_tools() to determine the required
        tool sequence.

        Args:
            user_message: The user's input message

        Returns:
            List of inferred goal outputs (e.g., ["summary", "test_results"])
        """
        return self.tool_registrar.infer_goals_from_message(user_message)

    # =====================================================================
    # Intent-Based Tool Filtering
    # =====================================================================

    def filter_tools_by_intent(
        self, tools: List[Any], current_intent: Optional["ActionIntent"] = None
    ) -> List[Any]:
        """Filter tools based on detected user intent.

        This method enforces intent-based tool restrictions:
        - DISPLAY_ONLY: Blocks write tools (write_file, edit_files, etc.)
        - READ_ONLY: Blocks write tools AND generation tools
        - WRITE_ALLOWED: No restrictions
        - AMBIGUOUS: No restrictions (relies on prompt guard)

        The blocked tools are defined in action_authorizer.INTENT_BLOCKED_TOOLS,
        which is the single source of truth for tool filtering.

        Args:
            tools: List of tool definitions (ToolDefinition objects or dicts)
            current_intent: The detected user intent (if None, no filtering)

        Returns:
            Filtered list of tools, excluding blocked tools for current intent
        """
        if current_intent is None:
            return tools

        from victor.agent.action_authorizer import INTENT_BLOCKED_TOOLS

        blocked_tools = INTENT_BLOCKED_TOOLS.get(current_intent, frozenset())
        if not blocked_tools:
            return tools

        def get_tool_name(tool: Any) -> str:
            """Extract tool name from ToolDefinition object or dict.

            Returns:
                Tool name string or empty string if not found
            """
            if hasattr(tool, "name"):
                return tool.name
            elif isinstance(tool, dict):
                return tool.get("name", "")
            return ""

        original_count = len(tools)
        filtered = [t for t in tools if get_tool_name(t) not in blocked_tools]
        filtered_count = original_count - len(filtered)

        if filtered_count > 0:
            blocked_names = blocked_tools & {get_tool_name(t) for t in tools}
            logger.info(
                f"Intent {current_intent.value}: filtered {filtered_count} "
                f"write/generation tools (blocked: {blocked_names})"
            )

            # Emit TOOL event for intent-based filtering
            if self._event_bus:
                try:
                    import asyncio

                    asyncio.run(
                        self._event_bus.emit(
                            topic="tool.filtered_by_intent",
                            data={
                                "intent": current_intent.value,
                                "original_count": original_count,
                                "filtered_count": filtered_count,
                                "blocked_tools": list(blocked_names),
                                "category": "tool",  # Preserve for observability
                            },
                            source="ToolPlanner",
                        )
                    )
                except Exception as e:
                    logger.debug(f"Failed to emit tool filtered event: {e}")

        return filtered
