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

"""Workflow Agent Protocol.

Defines the minimal interface that workflow executors need from an agent
orchestrator. This protocol enables workflows to depend on abstractions
rather than concrete implementations, following the Dependency Inversion Principle.

Phase 2 of decoupling workflows from AgentOrchestrator implementation details.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class WorkflowAgentProtocol(Protocol):
    """Protocol defining workflow agent execution requirements.

    This protocol captures the minimal methods that workflow executors
    need from an agent orchestrator. It enables:

    1. Dependency Inversion: Workflows depend on abstractions, not concrete classes
    2. Testability: Easy to mock for testing
    3. Flexibility: Alternative implementations can be used
    4. Type Safety: Protocol provides static type checking

    The protocol is based on how workflows actually use orchestrators:
    - Passed to SubAgentOrchestrator for agent node execution
    - Used to spawn sub-agents with roles and tasks

    Example:
        # In workflow executors
        from victor.protocols import WorkflowAgentProtocol

        def execute_agent(
            node: AgentNode,
            orchestrator: WorkflowAgentProtocol,  # Use protocol, not concrete class
        ) -> NodeResult:
            # SubAgentOrchestrator only needs the protocol interface
            sub_orch = SubAgentOrchestrator(orchestrator)
            result = await sub_orch.spawn(role=role, task=goal)

        # Testing with mock
        mock_orchestrator = cast(WorkflowAgentProtocol, Mock())
        executor = WorkflowExecutor(mock_orchestrator)
    """

    # The protocol is intentionally minimal - it only requires what
    # SubAgentOrchestrator actually uses. SubAgentOrchestrator accesses
    # the orchestrator's provider manager, conversation controller, and
    # streaming controller internally, but doesn't require specific methods
    # to be exposed on the protocol itself.
    #
    # This "marker protocol" approach means:
    # 1. We don't need to specify all internal methods
    # 2. AgentOrchestrator automatically satisfies the protocol
    # 3. Type checkers verify compatible types are passed
    # 4. Runtime checks with @runtime_checkable enable isinstance() checks


__all__ = [
    "WorkflowAgentProtocol",
]
