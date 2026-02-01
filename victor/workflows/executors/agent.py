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

"""Agent node executor.

Executes agent nodes by spawning sub-agents with role-specific configurations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, cast

if TYPE_CHECKING:
    from victor.agent.subagents.base import SubAgentRole
    from victor.workflows.definition import AgentNode
    from victor.workflows.adapters import WorkflowState

logger = logging.getLogger(__name__)


class AgentNodeExecutor:
    """Executor for agent nodes.

    Responsibility (SRP):
    - Spawn sub-agents with role-specific configurations
    - Substitute context variables in goals
    - Map agent profiles to orchestrators
    - Handle tool budget and timeout limits

    Non-responsibility:
    - Workflow compilation (handled by WorkflowCompiler)
    - Workflow execution coordination (handled by WorkflowExecutor)
    - State management (handled by StateGraph)

    Attributes:
        _context: Execution context with orchestrator, settings, services

    Example:
        executor = AgentNodeExecutor(context=execution_context)
        new_state = await executor.execute(agent_node, current_state)
    """

    def __init__(self, context: Optional[Any] = None):
        """Initialize the executor.

        Args:
            context: ExecutionContext with orchestrator, settings, services
        """
        self._context = context

    async def execute(self, node: "AgentNode", state: "WorkflowState") -> "WorkflowState":
        """Execute an agent node.

        Args:
            node: Agent node definition
            state: Current workflow state

        Returns:
            Updated workflow state

        Raises:
            Exception: If agent execution fails
        """
        from victor.agent.subagents.orchestrator import SubAgentOrchestrator

        logger.info(f"Executing agent node: {node.id} with role: {node.role}")

        # Step 1: Build input context from input_mapping
        input_context = {}
        if node.input_mapping:
            for key, source_key in node.input_mapping.items():
                if source_key in state:
                    input_context[key] = state[source_key]  # type: ignore[literal-required]

        # Step 2: Substitute context variables in goal
        goal = node.goal
        if goal and "{{" in goal:
            # Substitute variables from both input_context and full state
            substitution_context = {**state, **input_context}
            goal = self._substitute_context(goal, substitution_context)

        # Step 3: Get orchestrator based on profile
        orchestrator = self._get_orchestrator(node.profile)

        # Step 4: Map role to SubAgentRole enum
        role = self._map_role_to_enum(node.role)

        # Step 5: Spawn sub-agent via SubAgentOrchestrator
        sub_orchestrator = SubAgentOrchestrator(orchestrator)

        try:
            result = await sub_orchestrator.spawn(
                role=role,
                task=goal,
                tool_budget=node.tool_budget,
                allowed_tools=node.allowed_tools,
                timeout_seconds=int(node.timeout_seconds) if node.timeout_seconds else 300,
                disable_embeddings=getattr(node, "disable_embeddings", False),
            )
        except Exception as e:
            logger.error(f"Agent node {node.id} execution failed: {e}")
            # Store error in state and return
            if node.output_key:
                # Convert to dict and create new state with error
                state_dict = dict(state)
                state_dict[node.output_key] = {"error": str(e), "success": False}
                return cast("WorkflowState", state_dict)
            return state

        # Step 6: Store result in state
        if node.output_key:
            # Convert to dict and create new state with result
            state_dict = dict(state)
            state_dict[node.output_key] = result
            state = cast("WorkflowState", state_dict)

        # Track node result for observability
        # Convert to regular dict for dynamic access to avoid TypedDict issues
        state_dict = dict(state)
        if "_node_results" not in state_dict:
            state_dict["_node_results"] = {}

        # Create a simple dict for node result instead of using GraphNodeResult
        node_results = state_dict["_node_results"]
        if isinstance(node_results, dict):
            node_results[node.id] = {
                "node_id": node.id,
                "status": "completed",
                "result": result,
                "metadata": {
                    "role": node.role,
                    "tool_budget": node.tool_budget,
                    "profile": node.profile,
                },
            }

        logger.info(f"Agent node {node.id} completed successfully")
        return cast("WorkflowState", state_dict)

    def _get_orchestrator(self, profile: str | None) -> Any:
        """Get orchestrator based on profile.

        Args:
            profile: Provider profile name (or None for default)

        Returns:
            Configured orchestrator instance
        """
        # If context has orchestrator pool, use it
        if self._context and hasattr(self._context, "orchestrator_pool"):
            pool = self._context.orchestrator_pool
            if profile:
                return pool.get_orchestrator(profile)
            else:
                return pool.get_default_orchestrator()

        # Otherwise, use orchestrator from context directly
        if self._context and hasattr(self._context, "orchestrator"):
            return self._context.orchestrator

        # Fallback: raise error
        raise ValueError(f"No orchestrator available for profile: {profile}")

    def _map_role_to_enum(self, role: str) -> "SubAgentRole":
        """Map role string to SubAgentRole enum.

        Args:
            role: Role string (e.g., "researcher", "planner")

        Returns:
            SubAgentRole enum value

        Raises:
            ValueError: If role is not recognized
        """
        from victor.agent.subagents.roles import SubAgentRole

        role_map: dict[str, SubAgentRole] = {
            "researcher": SubAgentRole.RESEARCHER,
            "planner": SubAgentRole.PLANNER,
            "executor": SubAgentRole.EXECUTOR,
            "reviewer": SubAgentRole.REVIEWER,
            "tester": SubAgentRole.TESTER,
            "analyzer": SubAgentRole.RESEARCHER,  # Map researcher
            "critic": SubAgentRole.REVIEWER,  # Map reviewer
            "implementer": SubAgentRole.EXECUTOR,  # Map executor
            "default": SubAgentRole.EXECUTOR,
        }

        if role not in role_map:
            raise ValueError(f"Unknown agent role: {role}. Must be one of {list(role_map.keys())}")

        result = role_map[role]
        # The role_map values are already SubAgentRole enum members
        # But mypy can't infer this, so we use a type ignore
        return result

    def _substitute_context(self, template: str, context: dict[str, Any]) -> str:
        """Substitute context variables in template string.

        Args:
            template: Template string with {{variable}} placeholders
            context: Context dictionary with variable values

        Returns:
            Template with variables substituted
        """
        import re

        pattern = r"\{\{(\w+)\}\}"

        def replace_var(match: Any) -> str:
            from typing import cast

            var_name = match.group(1)
            if var_name in context:
                value = context[var_name]
                # Handle dict values by converting to JSON
                if isinstance(value, dict):
                    import json

                    return json.dumps(value)
                    # Handle primitive values
                    return str(value)  # type: ignore[unreachable]
            # Keep original if not found
            return cast(str, match.group(0))

        return re.sub(pattern, replace_var, template)

    def supports_node_type(self, node_type: str) -> bool:
        """Check if this executor supports the given node type.

        Args:
            node_type: Node type identifier

        Returns:
            bool: True if this executor handles the node type
        """
        return node_type == "agent"


__all__ = [
    "AgentNodeExecutor",
]
