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

"""Search coordinator for query routing and tool recommendation.

This coordinator handles search query routing as part of
Track 4 orchestrator extraction (Phase 1).

Responsibilities:
- Route search queries to optimal tools (keyword vs semantic)
- Recommend search tools based on query analysis

Thread Safety:
- All public methods are thread-safe
- Delegates to thread-safe SearchRouter
"""

import logging
from typing import Any, Dict

from victor.agent.search_router import SearchRoute, SearchType

logger = logging.getLogger(__name__)


class SearchCoordinator:
    """Coordinator for search query routing and tool recommendation.

    This coordinator analyzes search queries and routes them to the
    optimal search tool based on query characteristics.

    Design Principles:
    - Single Responsibility: Only handles search routing operations
    - Dependency Injection: SearchRouter injected via constructor
    - Thread Safety: All operations are thread-safe

    Attributes:
        search_router: SearchRouter instance for query analysis
    """

    # Tool name mapping for search types
    TOOL_MAP = {
        SearchType.KEYWORD: "code_search",
        SearchType.SEMANTIC: "semantic_code_search",
        SearchType.HYBRID: "both",
    }

    def __init__(self, search_router: Any):
        """Initialize the search coordinator.

        Args:
            search_router: SearchRouter instance for query analysis
        """
        self._search_router = search_router

    def route_search_query(self, query: str) -> Dict[str, Any]:
        """Route a search query to the optimal search tool.

        Analyzes the query to determine whether keyword search (code_search)
        or semantic search (semantic_code_search) would yield better results.

        Args:
            query: The search query

        Returns:
            Dictionary with routing recommendation:
                - recommended_tool: "code_search" or "semantic_code_search" or "both"
                - confidence: Confidence in the recommendation (0.0-1.0)
                - reason: Human-readable explanation
                - search_type: SearchType enum value (string)
                - matched_patterns: List of patterns that influenced decision
                - transformed_query: Optionally transformed query

        Example:
            >>> coordinator = SearchCoordinator(search_router)
            >>> route = coordinator.route_search_query("class BaseTool")
            >>> print(route["recommended_tool"])
            "code_search"

            >>> route = coordinator.route_search_query("how does error handling work")
            >>> print(route["recommended_tool"])
            "semantic_code_search"
        """
        # Route the query using SearchRouter
        route: SearchRoute = self._search_router.route(query)

        # Map SearchType to tool name
        result = {
            "recommended_tool": self.TOOL_MAP.get(route.search_type, "code_search"),
            "confidence": route.confidence,
            "reason": route.reason,
            "search_type": route.search_type.value,
            "matched_patterns": route.matched_patterns,
            "transformed_query": route.transformed_query,
        }

        logger.debug(
            f"Routed search query: tool={result['recommended_tool']}, "
            f"confidence={result['confidence']:.2f}, query={query[:50]}..."
        )

        return result

    def get_recommended_search_tool(self, query: str) -> str:
        """Get the recommended search tool name for a query.

        Convenience method that returns just the tool name, useful for
        quick routing decisions without detailed analysis.

        Args:
            query: The search query

        Returns:
            Tool name: "code_search", "semantic_code_search", or "both"

        Example:
            >>> coordinator = SearchCoordinator(search_router)
            >>> tool = coordinator.get_recommended_search_tool("def foo()")
            >>> print(tool)
            "code_search"
        """
        return self.route_search_query(query)["recommended_tool"]
