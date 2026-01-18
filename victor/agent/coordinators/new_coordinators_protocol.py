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

"""Protocol definitions for new coordinators (Track 4 extraction).

This module defines protocol interfaces for coordinators being extracted
from the monolithic orchestrator as part of Phase 1 refactoring.

Protocols:
- IConversationCoordinator: Conversation message management
- ISearchCoordinator: Search query routing and tool recommendation
- ITeamCoordinator: Team specification and suggestion management
"""

from typing import Any, Dict, List, Protocol


class IConversationCoordinator(Protocol):
    """Protocol for conversation message management coordinator.

    This coordinator handles conversation history operations including
    retrieving messages, adding new messages, and resetting conversation
    state.
    """

    @property
    def messages(self) -> List[Any]:
        """Get conversation messages.

        Returns:
            List of messages in conversation history
        """
        ...

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
        """
        ...

    def reset_conversation(self) -> None:
        """Clear conversation history and session state.

        Resets all conversation-related state including:
        - Conversation history
        - Tool call counter
        - Failed tool signatures cache
        - Observed files list
        - Executed tools list
        - Conversation state machine
        """
        ...


class ISearchCoordinator(Protocol):
    """Protocol for search query routing coordinator.

    This coordinator analyzes search queries and routes them to the
    optimal search tool (keyword vs semantic vs hybrid).
    """

    def route_search_query(self, query: str) -> Dict[str, Any]:
        """Route a search query to the optimal search tool.

        Analyzes the query to determine whether keyword search or
        semantic search would yield better results.

        Args:
            query: The search query

        Returns:
            Dictionary with routing recommendation:
                - recommended_tool: "code_search" or "semantic_code_search" or "both"
                - confidence: Confidence in the recommendation (0.0-1.0)
                - reason: Human-readable explanation
                - search_type: SearchType enum value
                - matched_patterns: List of patterns that influenced decision
                - transformed_query: Optionally transformed query
        """
        ...

    def get_recommended_search_tool(self, query: str) -> str:
        """Get the recommended search tool name for a query.

        Convenience method that returns just the tool name.

        Args:
            query: The search query

        Returns:
            Tool name: "code_search", "semantic_code_search", or "both"
        """
        ...


class ITeamCoordinator(Protocol):
    """Protocol for team specification and suggestion coordinator.

    This coordinator manages team specifications for multi-agent
    coordination and provides team formation suggestions.
    """

    def get_team_suggestions(self, task_type: str, complexity: str) -> Any:
        """Get team and workflow suggestions for a task.

        Queries the coordination system to get recommendations for
        teams and workflows based on task classification and current mode.

        Args:
            task_type: Classified task type (e.g., "feature", "bugfix", "refactor")
            complexity: Complexity level (e.g., "low", "medium", "high", "extreme")

        Returns:
            CoordinationSuggestion with team and workflow recommendations
        """
        ...

    def set_team_specs(self, specs: Dict[str, Any]) -> None:
        """Store team specifications.

        Provides a clean interface for setting team specs,
        replacing direct private attribute access.

        Args:
            specs: Dictionary mapping team names to TeamSpec instances
        """
        ...

    def get_team_specs(self) -> Dict[str, Any]:
        """Retrieve team specifications.

        Returns the dictionary of team specs configured by vertical integration.

        Returns:
            Dictionary of team specs, or empty dict if not set
        """
        ...
