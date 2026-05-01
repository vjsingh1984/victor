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

"""Tool planner service implementation.

Handles execution planning and strategy for tool operations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Set

if TYPE_CHECKING:
    from victor.agent.services.tools.protocols import (
        ToolSelectorServiceProtocol,
        ToolTrackerServiceProtocol,
    )

logger = logging.getLogger(__name__)


class ToolPlannerServiceConfig:
    """Configuration for ToolPlannerService.

    Attributes:
        enable_cost_estimation: Enable execution cost estimation
        enable_order_optimization: Enable execution order optimization
        parallel_eligible_tools: Tools that can run in parallel
    """

    def __init__(
        self,
        enable_cost_estimation: bool = True,
        enable_order_optimization: bool = True,
        parallel_eligible_tools: Set[str] | None = None,
    ):
        self.enable_cost_estimation = enable_cost_estimation
        self.enable_order_optimization = enable_order_optimization
        self.parallel_eligible_tools = parallel_eligible_tools or set()


class ToolPlannerService:
    """Service for tool execution planning.

    Responsible for:
    - Execution planning for tasks
    - Cost estimation
    - Execution order optimization
    - Dependency analysis

    This service does NOT handle:
    - Tool selection (delegated to ToolSelectorService)
    - Tool execution (delegated to ToolExecutorService)
    - Budget tracking (delegated to ToolTrackerService)
    - Result processing (delegated to ToolResultProcessor)

    Example:
        config = ToolPlannerServiceConfig()
        planner = ToolPlannerService(
            config=config,
            selector_service=selector,
            tracker_service=tracker,
        )

        # Plan execution for task
        plan = await planner.plan_execution(
            "Search and analyze files",
            available_tools={"search", "read_file", "analyze"}
        )

        # Estimate cost
        cost = planner.estimate_execution_cost(plan)
    """

    def __init__(
        self,
        config: ToolPlannerServiceConfig,
        selector_service: ToolSelectorServiceProtocol | None = None,
        tracker_service: ToolTrackerServiceProtocol | None = None,
    ):
        """Initialize ToolPlannerService.

        Args:
            config: Service configuration
            selector_service: Optional tool selector service
            tracker_service: Optional tool tracker service
        """
        self.config = config
        self.selector_service = selector_service
        self.tracker_service = tracker_service

        # Health tracking
        self._healthy = True

    async def plan_execution(
        self,
        task: str,
        available_tools: Set[str],
        context: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Plan tool execution for a task.

        Creates an execution plan by selecting tools and optimizing
        their execution order.

        Args:
            task: Task description
            available_tools: Set of available tool names
            context: Optional context for planning

        Returns:
            List of planned tool calls (ordered for execution)
        """
        # Select tools for task
        if self.selector_service:
            selected_tools = await self.selector_service.select_tools(
                task, available_tools, context
            )
        else:
            # Fallback: use all available tools
            selected_tools = list(available_tools)

        if not selected_tools:
            logger.warning(f"No tools selected for task: {task}")
            return []

        # Create tool call plans
        # For now, create simple plans without arguments
        # In a full implementation, this would use an LLM to generate
        # appropriate tool calls with arguments
        plans = []
        for tool_name in selected_tools:
            plans.append(
                {
                    "name": tool_name,
                    "arguments": {},  # Would be populated by LLM
                    "reason": f"Selected for task: {task}",
                }
            )

        # Optimize execution order
        if self.config.enable_order_optimization:
            plans = self.optimize_execution_order(plans)

        logger.debug(f"Created execution plan with {len(plans)} tools")

        return plans

    def estimate_execution_cost(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, float]:
        """Estimate execution cost for tool calls.

        Provides estimates for:
        - Time cost (seconds)
        - Token cost (if applicable)
        - Resource cost (memory, CPU)

        Args:
            tool_calls: List of tool calls to estimate

        Returns:
            Dictionary with cost estimates
        """
        if not self.config.enable_cost_estimation:
            return {
                "time_seconds": 0.0,
                "tokens": 0,
                "resource_score": 0.0,
            }

        total_time = 0.0
        total_tokens = 0
        max_resource_score = 0.0

        for call in tool_calls:
            tool_name = call.get("name", "unknown")

            # Simple cost estimation based on tool type
            # In a full implementation, this would use historical data
            time_cost = self._estimate_tool_time(tool_name)
            token_cost = self._estimate_tool_tokens(tool_name)
            resource_score = self._estimate_tool_resources(tool_name)

            total_time += time_cost
            total_tokens += token_cost
            max_resource_score = max(max_resource_score, resource_score)

        return {
            "time_seconds": total_time,
            "tokens": total_tokens,
            "resource_score": max_resource_score,
        }

    def _estimate_tool_time(self, tool_name: str) -> float:
        """Estimate execution time for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Estimated time in seconds
        """
        # Simple heuristic estimates
        # In production, use historical data
        time_estimates = {
            "search": 2.0,
            "read_file": 0.5,
            "write_file": 0.5,
            "analyze": 5.0,
            "execute": 3.0,
        }

        return time_estimates.get(tool_name, 1.0)

    def _estimate_tool_tokens(self, tool_name: str) -> int:
        """Estimate token usage for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Estimated token count
        """
        # Simple heuristic estimates
        token_estimates = {
            "search": 100,
            "read_file": 50,
            "write_file": 50,
            "analyze": 500,
            "execute": 200,
        }

        return token_estimates.get(tool_name, 100)

    def _estimate_tool_resources(self, tool_name: str) -> float:
        """Estimate resource usage for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Resource score (0-1, higher = more resources)
        """
        # Simple heuristic estimates
        resource_estimates = {
            "search": 0.3,
            "read_file": 0.2,
            "write_file": 0.2,
            "analyze": 0.8,
            "execute": 0.9,
        }

        return resource_estimates.get(tool_name, 0.5)

    def optimize_execution_order(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize execution order for tool calls.

        Optimization strategies:
        1. Group parallel-eligible tools together
        2. Prioritize fast tools
        3. Consider dependencies

        Args:
            tool_calls: List of tool calls to optimize

        Returns:
            Optimized list of tool calls
        """
        if not tool_calls:
            return tool_calls

        # Separate parallel-eligible and sequential tools
        parallel_tools = []
        sequential_tools = []

        for call in tool_calls:
            tool_name = call.get("name", "")
            if tool_name in self.config.parallel_eligible_tools:
                parallel_tools.append(call)
            else:
                sequential_tools.append(call)

        # Sort sequential tools by estimated time (fastest first)
        sequential_tools.sort(key=lambda c: self._estimate_tool_time(c.get("name", "")))

        # Combine: parallel tools first, then sequential
        optimized = parallel_tools + sequential_tools

        if len(optimized) != len(tool_calls):
            logger.warning("Optimization changed tool call count - this should not happen")

        return optimized

    def analyze_dependencies(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Analyze dependencies between tool calls.

        Args:
            tool_calls: List of tool calls to analyze

        Returns:
            Dictionary mapping tool name to list of dependencies
        """
        dependencies = {}

        for call in tool_calls:
            tool_name = call.get("name", "")
            dependencies[tool_name] = []

            # Check if tool call references results from other tools
            arguments = call.get("arguments", {})
            for arg_name, arg_value in arguments.items():
                if isinstance(arg_value, str) and "$" in arg_value:
                    # Simple dependency detection (e.g., "$result.read_file")
                    ref_tool = arg_value.split(".")[-1].replace("}", "")
                    if ref_tool and ref_tool != tool_name:
                        dependencies[tool_name].append(ref_tool)

        return dependencies

    def is_healthy(self) -> bool:
        """Check if service is healthy.

        Returns:
            True if healthy, False otherwise
        """
        return self._healthy
