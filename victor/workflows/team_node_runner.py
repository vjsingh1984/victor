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

"""Execute team nodes within workflow graphs.

This module provides the TeamNodeRunner class, which handles the execution
of team nodes within workflow graphs. Team nodes spawn ad-hoc multi-agent
teams using various formation strategies (sequential, parallel, pipeline, etc.)
as part of workflow execution.

Key Features:
- Integration with UnifiedTeamCoordinator for team execution
- Recursion depth tracking via RecursionGuard
- Support for all team formation types
- Proper error handling and result mapping
- Metadata collection for observability

Example:
    from victor.workflows.team_node_runner import TeamNodeRunner

    runner = TeamNodeRunner(orchestrator, tool_registry)
    result = await runner.execute(
        node=team_node,
        context=execution_context,
        recursion_ctx=recursion_context
    )
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.tools.registry import ToolRegistry

from victor.core.errors import RecursionDepthError
from victor.workflows.definition import TeamNodeWorkflow
from victor.workflows.executor import ExecutorNodeStatus, NodeResult
from victor.workflows.recursion import RecursionContext, RecursionGuard

logger = logging.getLogger(__name__)


class TeamNodeRunner:
    """Execute team nodes with recursion tracking.

    Team nodes allow workflows to spawn multi-agent teams using various
    formation strategies. This runner integrates with the UnifiedTeamCoordinator
    and properly tracks recursion depth to prevent infinite nesting.

    Attributes:
        _orchestrator: AgentOrchestrator for agent execution
        _tool_registry: ToolRegistry for tool access
        _enable_observability: Whether to emit observability events
        _enable_metrics: Whether to collect metrics
        _create_coordinator: Factory function for creating coordinators
        _metrics_collector: Team metrics collector instance

    Example:
        runner = TeamNodeRunner(
            orchestrator=my_orchestrator,
            tool_registry=my_tool_registry,
            enable_observability=True,
            enable_metrics=True
        )

        result = await runner.execute(
            node=team_node,
            context={"shared_context": {...}},
            recursion_ctx=RecursionContext(max_depth=3)
        )

        print(f"Team completed: {result.output}")
        print(f"Members: {result.metadata['team_members']}")
    """

    def __init__(
        self,
        orchestrator: "AgentOrchestrator",
        tool_registry: Optional["ToolRegistry"] = None,
        enable_observability: bool = False,
        enable_metrics: bool = True,
    ):
        """Initialize team node runner.

        Args:
            orchestrator: AgentOrchestrator for agent execution
            tool_registry: ToolRegistry for tool access (optional)
            enable_observability: Whether to emit observability events
            enable_metrics: Whether to collect team metrics
        """
        from victor.teams import create_coordinator

        self._orchestrator = orchestrator
        self._tool_registry = tool_registry
        self._enable_observability = enable_observability
        self._enable_metrics = enable_metrics
        self._create_coordinator = create_coordinator

        # Get metrics collector if enabled
        self._metrics_collector = None
        if enable_metrics:
            try:
                from victor.workflows.team_metrics import get_team_metrics_collector

                self._metrics_collector = get_team_metrics_collector()
            except ImportError:
                logger.debug("Team metrics collector not available")

    async def execute(
        self,
        node: TeamNodeWorkflow,
        context: Dict[str, Any],
        recursion_ctx: RecursionContext,
    ) -> NodeResult:
        """Execute a team node.

        This method creates a team coordinator, adds members from the node
        configuration, sets the formation strategy, and executes the team task.
        Recursion depth is tracked via RecursionGuard.

        Args:
            node: TeamNodeWorkflow to execute
            context: Execution context containing shared state and inputs
            recursion_ctx: RecursionContext for depth tracking

        Returns:
            NodeResult with team output and metadata

        Raises:
            RecursionDepthError: If max recursion depth exceeded
            YAMLWorkflowError: If team execution fails

        Example:
            node = TeamNodeWorkflow(
                id="review_team",
                goal="Review the code changes",
                team_formation="parallel",
                members=[
                    {
                        "id": "security_reviewer",
                        "role": "reviewer",
                        "goal": "Check for security issues",
                        "tool_budget": 10
                    },
                    {
                        "id": "quality_reviewer",
                        "role": "reviewer",
                        "goal": "Check code quality",
                        "tool_budget": 10
                    }
                ]
            )

            result = await runner.execute(
                node=node,
                context={"shared_context": {"file": "app.py"}},
                recursion_ctx=RecursionContext(max_depth=3)
            )
        """
        from victor.teams import TeamFormation
        from victor.workflows.yaml_loader import YAMLWorkflowError

        logger.info(
            f"Executing team node '{node.id}': "
            f"{len(node.members)} members, formation={node.team_formation}"
        )

        # Record start time
        start_time = time.time()

        # Track recursion depth
        with RecursionGuard(recursion_ctx, "team", node.id):
            try:
                # Record team start in metrics
                if self._metrics_collector:
                    self._metrics_collector.record_team_start(
                        team_id=node.id,
                        formation=node.team_formation,
                        member_count=len(node.members),
                        recursion_depth=recursion_ctx.current_depth,
                    )

                # Create team coordinator with shared recursion context
                coordinator = self._create_coordinator(
                    orchestrator=self._orchestrator,
                    with_observability=self._enable_observability,
                    recursion_context=recursion_ctx,
                )

                # Add members from configuration
                for member_config in node.members:
                    self._add_member_to_coordinator(coordinator, member_config, context)

                # Set formation strategy
                formation = TeamFormation(node.team_formation)
                coordinator.set_formation(formation)

                # Execute team
                result = await coordinator.execute_task(
                    task=node.goal,
                    context=context.get("shared_context", {}),
                )

                # Extract metadata from result
                # The result can be a dict or a TeamResult object
                if isinstance(result, dict):
                    final_output = result.get("final_output", "")
                    metadata = result.get("metadata", {})
                    iterations = metadata.get("iterations", 0)
                    success = result.get("success", True)
                    member_results = result.get("member_results", {})
                else:
                    # TeamResult object
                    final_output = getattr(result, "final_output", "")
                    metadata = getattr(result, "metadata", {})
                    iterations = metadata.get("iterations", 0) if metadata else 0
                    success = getattr(result, "success", True)
                    member_results = getattr(result, "member_results", {})

                # Calculate duration
                duration_seconds = time.time() - start_time

                # Record member metrics
                if self._metrics_collector and member_results:
                    for member_id, member_result in member_results.items():
                        if hasattr(member_result, "success"):
                            self._metrics_collector.record_member_complete(
                                team_id=node.id,
                                member_id=member_id,
                                success=member_result.success,
                                duration_seconds=member_result.duration_seconds,
                                tool_calls_used=member_result.tool_calls_used,
                                tools_used=set(getattr(member_result, "tools_used", set())),
                                error_message=(
                                    member_result.error if not member_result.success else None
                                ),
                                role=getattr(member_result, "role", "assistant"),
                            )

                # Determine status based on success
                status = ExecutorNodeStatus.COMPLETED if success else ExecutorNodeStatus.FAILED

                # Record team completion in metrics
                if self._metrics_collector:
                    self._metrics_collector.record_team_complete(
                        team_id=node.id,
                        success=success,
                        duration_seconds=duration_seconds,
                        consensus_achieved=(
                            result.get("consensus_achieved") if isinstance(result, dict) else None
                        ),
                        consensus_rounds=(
                            result.get("consensus_rounds") if isinstance(result, dict) else None
                        ),
                    )

                # Return node result (metadata stored in output for now)
                output_with_metadata = {
                    "final_output": final_output,
                    "team_members": len(node.members),
                    "team_formation": node.team_formation,
                    "team_iterations": iterations,
                    "duration_seconds": duration_seconds,
                }

                return NodeResult(
                    node_id=node.id,
                    status=status,
                    output=output_with_metadata,
                )

            except RecursionDepthError:
                # Record recursion depth exceeded in metrics
                if self._metrics_collector:
                    duration_seconds = time.time() - start_time
                    self._metrics_collector.record_team_complete(
                        team_id=node.id,
                        success=False,
                        duration_seconds=duration_seconds,
                    )

                # Emit event for recursion depth exceeded
                await self._emit_recursion_depth_event(node.id, recursion_ctx)

                # Re-raise recursion depth errors as-is
                raise

            except Exception as e:
                # Record failure in metrics
                if self._metrics_collector:
                    duration_seconds = time.time() - start_time
                    self._metrics_collector.record_team_complete(
                        team_id=node.id,
                        success=False,
                        duration_seconds=duration_seconds,
                    )

                # Wrap other exceptions in YAMLWorkflowError
                logger.error(f"Team node '{node.id}' execution failed: {e}", exc_info=True)
                raise YAMLWorkflowError(f"Team execution failed: {e}") from e

    def _add_member_to_coordinator(
        self,
        coordinator: Any,
        member_config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        """Add a member to the team coordinator.

        This method extracts member configuration from the YAML definition
        and adds the member to the coordinator with the specified role,
        goal, tool budget, and other attributes.

        Args:
            coordinator: UnifiedTeamCoordinator instance
            member_config: Member configuration from YAML
            context: Execution context

        Example:
            member_config = {
                "id": "researcher",
                "role": "researcher",
                "goal": "Find code patterns",
                "tool_budget": 15,
                "tools": ["read", "search"],
                "backstory": "Expert in code analysis",
                "expertise": ["python", "security"],
                "personality": "thorough and detail-oriented"
            }

            runner._add_member_to_coordinator(coordinator, member_config, context)
        """
        from victor.agent.subagents.base import SubAgentRole

        member_id = member_config.get("id")
        role_str = member_config.get("role", "assistant")
        goal = member_config.get("goal", "")
        tool_budget = member_config.get("tool_budget", 25)
        allowed_tools = member_config.get("tools")

        # Map role string to SubAgentRole
        try:
            role = SubAgentRole(role_str)
        except ValueError:
            # Fallback to EXECUTOR if role not recognized
            logger.warning(
                f"Unknown role '{role_str}' for member '{member_id}', " f"defaulting to 'executor'"
            )
            role = SubAgentRole.EXECUTOR

        # Add member to coordinator
        coordinator.add_member(
            role=role,
            goal=goal,
            tool_budget=tool_budget,
            allowed_tools=allowed_tools,
            backstory=member_config.get("backstory"),
            expertise=member_config.get("expertise"),
            personality=member_config.get("personality"),
        )

        logger.debug(
            f"Added team member '{member_id}': role={role.value}, "
            f"budget={tool_budget}, tools={allowed_tools}"
        )

    async def _emit_recursion_depth_event(
        self,
        team_id: str,
        recursion_ctx: RecursionContext,
    ) -> None:
        """Emit event when recursion depth is exceeded.

        Args:
            team_id: Team identifier
            recursion_ctx: Recursion context with depth info
        """
        if not self._enable_observability:
            return

        try:
            from victor.core.events import ObservabilityBus as EventBus

            bus = EventBus()
            await bus.connect()
            await bus.emit(
                topic="team.recursion.depth_exceeded",
                data={
                    "team_id": team_id,
                    "current_depth": recursion_ctx.current_depth,
                    "max_depth": recursion_ctx.max_depth,
                    "execution_stack": recursion_ctx.execution_stack.copy(),
                },
            )
        except ImportError:
            logger.debug("EventBus not available, skipping recursion depth event")
        except Exception as e:
            logger.warning(f"Failed to emit recursion depth event: {e}")


__all__ = ["TeamNodeRunner"]
