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

"""Unit tests for SearchCoordinator.

Tests the search query routing coordinator extracted from
the monolithic orchestrator as part of Track 4 Phase 1 refactoring.
"""

import pytest
from unittest.mock import Mock

from victor.agent.search_router import SearchType, SearchRoute
from victor.agent.coordinators.search_coordinator import SearchCoordinator


class TestSearchCoordinator:
    """Test suite for SearchCoordinator."""

    def test_initialization(self):
        """Test coordinator initialization with search router."""
        # Arrange
        search_router = Mock()

        # Act
        coordinator = SearchCoordinator(search_router=search_router)

        # Assert
        assert coordinator._search_router == search_router

    def test_route_search_query_keyword(self):
        """Test routing a keyword search query."""
        # Arrange
        search_router = Mock()
        search_route = SearchRoute(
            search_type=SearchType.KEYWORD,
            confidence=1.0,
            reason="Exact class name match",
            transformed_query=None,
            matched_patterns=["class_name"],
        )
        search_router.route.return_value = search_route

        coordinator = SearchCoordinator(search_router=search_router)

        # Act
        result = coordinator.route_search_query("class BaseTool")

        # Assert
        search_router.route.assert_called_once_with("class BaseTool")
        assert result["recommended_tool"] == "code_search"
        assert result["confidence"] == 1.0
        assert result["reason"] == "Exact class name match"
        assert result["search_type"] == "keyword"
        assert result["matched_patterns"] == ["class_name"]
        assert result["transformed_query"] is None

    def test_route_search_query_semantic(self):
        """Test routing a semantic search query."""
        # Arrange
        search_router = Mock()
        search_route = SearchRoute(
            search_type=SearchType.SEMANTIC,
            confidence=0.9,
            reason="Conceptual how-to question",
            transformed_query="error handling patterns",
            matched_patterns=["how_question"],
        )
        search_router.route.return_value = search_route

        coordinator = SearchCoordinator(search_router=search_router)

        # Act
        result = coordinator.route_search_query("how does error handling work")

        # Assert
        search_router.route.assert_called_once_with("how does error handling work")
        assert result["recommended_tool"] == "semantic_code_search"
        assert result["confidence"] == 0.9
        assert result["search_type"] == "semantic"
        assert result["transformed_query"] == "error handling patterns"

    def test_route_search_query_hybrid(self):
        """Test routing a hybrid search query."""
        # Arrange
        search_router = Mock()
        search_route = SearchRoute(
            search_type=SearchType.HYBRID,
            confidence=0.7,
            reason="Mixed signals - use both approaches",
            transformed_query=None,
            matched_patterns=["class_name", "how_question"],
        )
        search_router.route.return_value = search_route

        coordinator = SearchCoordinator(search_router=search_router)

        # Act
        result = coordinator.route_search_query("class BaseTool usage examples")

        # Assert
        assert result["recommended_tool"] == "both"
        assert result["confidence"] == 0.7
        assert result["search_type"] == "hybrid"

    def test_route_search_query_with_transformed_query(self):
        """Test routing with query transformation."""
        # Arrange
        search_router = Mock()
        search_route = SearchRoute(
            search_type=SearchType.SEMANTIC,
            confidence=0.85,
            reason="Query normalized and expanded",
            transformed_query="async def error handling patterns",
            matched_patterns=["function_def"],
        )
        search_router.route.return_value = search_route

        coordinator = SearchCoordinator(search_router=search_router)

        # Act
        result = coordinator.route_search_query("async def handleError")

        # Assert
        assert result["transformed_query"] == "async def error handling patterns"
        assert result["recommended_tool"] == "semantic_code_search"

    def test_route_search_query_empty_query(self):
        """Test routing an empty query."""
        # Arrange
        search_router = Mock()
        search_route = SearchRoute(
            search_type=SearchType.KEYWORD,
            confidence=0.5,
            reason="Empty query defaults to keyword",
            transformed_query=None,
            matched_patterns=[],
        )
        search_router.route.return_value = search_route

        coordinator = SearchCoordinator(search_router=search_router)

        # Act
        result = coordinator.route_search_query("")

        # Assert
        search_router.route.assert_called_once_with("")
        assert result["confidence"] == 0.5

    def test_route_search_query_long_query(self):
        """Test routing a very long query."""
        # Arrange
        search_router = Mock()
        long_query = "search for " + "terms " * 100
        search_route = SearchRoute(
            search_type=SearchType.SEMANTIC,
            confidence=0.8,
            reason="Long semantic query",
            transformed_query=None,
            matched_patterns=["semantic"],
        )
        search_router.route.return_value = search_route

        coordinator = SearchCoordinator(search_router=search_router)

        # Act
        result = coordinator.route_search_query(long_query)

        # Assert
        search_router.route.assert_called_once_with(long_query)
        assert result["search_type"] == "semantic"

    def test_get_recommended_search_tool_keyword(self):
        """Test getting recommended tool for keyword search."""
        # Arrange
        search_router = Mock()
        search_route = SearchRoute(
            search_type=SearchType.KEYWORD,
            confidence=1.0,
            reason="Class name",
            transformed_query=None,
            matched_patterns=["class_name"],
        )
        search_router.route.return_value = search_route

        coordinator = SearchCoordinator(search_router=search_router)

        # Act
        tool = coordinator.get_recommended_search_tool("class MyClass")

        # Assert
        assert tool == "code_search"
        search_router.route.assert_called_once()

    def test_get_recommended_search_tool_semantic(self):
        """Test getting recommended tool for semantic search."""
        # Arrange
        search_router = Mock()
        search_route = SearchRoute(
            search_type=SearchType.SEMANTIC,
            confidence=0.9,
            reason="How question",
            transformed_query=None,
            matched_patterns=["how_question"],
        )
        search_router.route.return_value = search_route

        coordinator = SearchCoordinator(search_router=search_router)

        # Act
        tool = coordinator.get_recommended_search_tool("how do I implement caching")

        # Assert
        assert tool == "semantic_code_search"

    def test_get_recommended_search_tool_hybrid(self):
        """Test getting recommended tool for hybrid search."""
        # Arrange
        search_router = Mock()
        search_route = SearchRoute(
            search_type=SearchType.HYBRID,
            confidence=0.7,
            reason="Mixed signals",
            transformed_query=None,
            matched_patterns=[],
        )
        search_router.route.return_value = search_route

        coordinator = SearchCoordinator(search_router=search_router)

        # Act
        tool = coordinator.get_recommended_search_tool("complex query")

        # Assert
        assert tool == "both"

    def test_get_recommended_search_tool_is_convenience_method(self):
        """Test that get_recommended_search_tool is a convenience wrapper."""
        # Arrange
        search_router = Mock()
        search_route = SearchRoute(
            search_type=SearchType.KEYWORD,
            confidence=1.0,
            reason="Test",
            transformed_query=None,
            matched_patterns=[],
        )
        search_router.route.return_value = search_route

        coordinator = SearchCoordinator(search_router=search_router)

        # Act
        tool = coordinator.get_recommended_search_tool("test query")

        # Assert - Should return just the tool name string
        assert isinstance(tool, str)
        assert tool == "code_search"

    def test_tool_mapping_completeness(self):
        """Test that all SearchType values are mapped to tools."""
        # Arrange
        coordinator = SearchCoordinator(search_router=Mock())

        # Assert - Verify all enum values have mappings
        assert SearchType.KEYWORD in coordinator.TOOL_MAP
        assert SearchType.SEMANTIC in coordinator.TOOL_MAP
        assert SearchType.HYBRID in coordinator.TOOL_MAP

    def test_unknown_search_type_defaults_to_code_search(self):
        """Test that unknown search types default to code_search."""
        # Arrange
        search_router = Mock()
        # Mock route with a search type that's not in TOOL_MAP
        # We'll test the default behavior by using a mock that returns None for .value
        search_route = Mock()
        search_route.search_type = Mock()
        search_route.search_type.value = "unknown_type"
        search_route.confidence = 0.5
        search_route.reason = "Unknown"
        search_route.transformed_query = None
        search_route.matched_patterns = []
        search_router.route.return_value = search_route

        coordinator = SearchCoordinator(search_router=search_router)

        # Act
        result = coordinator.route_search_query("unknown query")

        # Assert - Should default to code_search
        assert result["recommended_tool"] == "code_search"

    def test_multiple_route_search_queries(self):
        """Test routing multiple search queries in sequence."""
        # Arrange
        search_router = Mock()
        search_router.route.side_effect = [
            SearchRoute(SearchType.KEYWORD, 1.0, "Keyword", None, ["pattern"]),
            SearchRoute(SearchType.SEMANTIC, 0.9, "Semantic", None, ["pattern"]),
            SearchRoute(SearchType.HYBRID, 0.7, "Hybrid", None, ["pattern"]),
        ]

        coordinator = SearchCoordinator(search_router=search_router)

        # Act
        result1 = coordinator.route_search_query("keyword query")
        result2 = coordinator.route_search_query("semantic query")
        result3 = coordinator.route_search_query("hybrid query")

        # Assert
        assert result1["recommended_tool"] == "code_search"
        assert result2["recommended_tool"] == "semantic_code_search"
        assert result3["recommended_tool"] == "both"
        assert search_router.route.call_count == 3
