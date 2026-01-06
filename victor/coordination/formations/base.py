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

"""Base formation strategy for multi-agent coordination.

This module defines the abstract base for all formation strategies,
following the Open/Closed Principle (OCP) and Strategy pattern.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from victor.teams.types import AgentMessage, MemberResult


class TeamContext:
    """Simple context for team execution.

    Attributes:
        team_id: Team identifier
        formation: Formation pattern being used
        shared_state: Shared state between agents
        metadata: Additional metadata
    """

    def __init__(
        self,
        team_id: str,
        formation: str,
        shared_state: Optional[Dict[str, Any]] = None,
        **metadata: Any,
    ):
        self.team_id = team_id
        self.formation = formation
        self.shared_state = shared_state or {}
        self.metadata = metadata


class BaseFormationStrategy(ABC):
    """Abstract base for formation strategies.

    This class defines the contract for all formation strategies,
    allowing the coordinator to delegate execution logic while
    remaining independent of specific implementations (OCP).

    Implementations must define how to:
    - Execute agents in the formation pattern
    - Handle results from agents
    - Manage context flow between agents
    """

    @abstractmethod
    async def execute(
        self,
        agents: List[Any],
        context: TeamContext,
        task: AgentMessage,
    ) -> List[MemberResult]:
        """Execute agents using this formation strategy.

        Args:
            agents: List of agents to execute
            context: Team context with shared state
            task: Task message to process

        Returns:
            List of results from each agent
        """
        pass

    @abstractmethod
    def validate_context(self, context: "TeamContext") -> bool:
        """Validate that context has required fields for this formation.

        Args:
            context: Team context to validate

        Returns:
            True if context is valid for this formation
        """
        pass

    def get_required_roles(self) -> Optional[List[str]]:
        """Get required roles for this formation (if any).

        Returns:
            List of required role names, or None if any role is acceptable
        """
        return None

    def supports_early_termination(self) -> bool:
        """Check if this formation supports early termination.

        Returns:
            True if formation can terminate before all agents complete
        """
        return False

    async def prepare_context(
        self,
        context: "TeamContext",
        task: AgentMessage,
    ) -> "TeamContext":
        """Prepare context before execution.

        Args:
            context: Initial team context
            task: Task message

        Returns:
            Prepared context with formation-specific initialization
        """
        return context

    async def process_results(
        self,
        results: List[MemberResult],
        context: "TeamContext",
    ) -> List[MemberResult]:
        """Process results after execution.

        Args:
            results: Raw results from agents
            context: Team context after execution

        Returns:
            Processed results
        """
        return results
