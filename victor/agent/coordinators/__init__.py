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

"""Tool coordination package for Victor AI coding assistant.

This package contains coordinator classes that extract and consolidate
specific coordination responsibilities from the AgentOrchestrator:

- ToolCoordinator: Tool selection, budgeting, access control, and execution

Design Philosophy:
------------------
Each coordinator follows the Single Responsibility Principle (SRP) and
provides a focused, testable interface for its domain. Coordinators are
designed to work with existing components (ToolPipeline, ToolSelector, etc.)
and provide a clean API for the orchestrator to delegate to.

Usage:
------
    from victor.agent.coordinators import ToolCoordinator, create_tool_coordinator

    coordinator = create_tool_coordinator(
        tool_pipeline=pipeline,
        tool_registry=registry,
        tool_selector=selector,
    )

    # Select tools for current context
    tools = await coordinator.select_tools(context)

    # Check and consume budget
    remaining = coordinator.get_remaining_budget()
    coordinator.consume_budget(1)

    # Execute tool calls
    result = await coordinator.execute_tool_calls(tool_calls)
"""

from victor.agent.coordinators.tool_coordinator import (
    ToolCoordinator,
    ToolCoordinatorConfig,
    TaskContext,
    IToolCoordinator,
    create_tool_coordinator,
)

# ToolExecutionResult is canonical in victor.agent.tool_executor
# Import from there: from victor.agent.tool_executor import ToolExecutionResult

__all__ = [
    "ToolCoordinator",
    "ToolCoordinatorConfig",
    "TaskContext",
    "IToolCoordinator",
    "create_tool_coordinator",
]
