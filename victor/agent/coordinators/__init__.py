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

"""Coordinator package for Victor agentic AI framework.

This package contains coordinator classes that extract and consolidate
specific coordination responsibilities from the AgentOrchestrator:

- ChatCoordinator: Chat and streaming chat operations
- SessionCoordinator: Session lifecycle, checkpoints, memory context
- ToolCoordinator: Tool selection, budgeting, access control, and execution
- MetricsCoordinator: Metrics collection and reporting
- ProviderCoordinator: Provider management and switching

Design Philosophy:
------------------
Each coordinator follows the Single Responsibility Principle (SRP) and
provides a focused, testable interface for its domain. Coordinators are
designed to work with existing components and provide a clean API for
the orchestrator to delegate to.
"""

from victor.agent.coordinators.tool_coordinator import (
    ToolCoordinator,
    ToolCoordinatorConfig,
    TaskContext,
    IToolCoordinator,
    create_tool_coordinator,
)
from victor.agent.coordinators.chat_coordinator import ChatCoordinator
from victor.agent.coordinators.chat_protocols import (
    ChatContextProtocol,
    ChatOrchestratorProtocol,
    ProviderContextProtocol,
    ToolContextProtocol,
)
from victor.agent.coordinators.session_coordinator import (
    SessionCoordinator,
    SessionInfo,
    create_session_coordinator,
)
from victor.agent.coordinators.planning_coordinator import (
    PlanningCoordinator,
    PlanningConfig,
    PlanningMode,
    PlanningResult,
)

__all__ = [
    "ChatCoordinator",
    "ChatContextProtocol",
    "ChatOrchestratorProtocol",
    "ProviderContextProtocol",
    "ToolContextProtocol",
    "SessionCoordinator",
    "SessionInfo",
    "create_session_coordinator",
    "ToolCoordinator",
    "ToolCoordinatorConfig",
    "TaskContext",
    "IToolCoordinator",
    "create_tool_coordinator",
    "PlanningCoordinator",
    "PlanningConfig",
    "PlanningMode",
    "PlanningResult",
]
