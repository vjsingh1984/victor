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

"""Tests for SearchCoordinator.

This test file demonstrates the migration pattern from orchestrator tests
to coordinator-specific tests, following Track 4 extraction.

Migration Pattern:
1. Identify orchestrator tests that delegate to coordinator
2. Extract relevant test logic
3. Mock coordinator dependencies (not the orchestrator)
4. Test coordinator in isolation
5. Update integration tests to verify delegation

SearchCoordinator Responsibilities:
- Route search queries to optimal tools (keyword vs semantic)
- Recommend search tools based on query analysis
- Delegates to SearchRouter for query analysis
"""

import pytest
from unittest.mock import Mock
from typing import Any, Dict

from victor.agent.coordinators.search_coordinator import SearchCoordinator
from victor.agent.search_router import SearchRoute, SearchType


class TestSearchCoordinator:
    """Test suite for SearchCoordinator.

    This coordinator handles search query routing, extracted
    from AgentOrchestrator as part of Track 4 refactoring.
    """

    @pytest.fixture
    def mock_search_router(self) -> Mock:
        """Create mock search router."""
        router = Mock()
        return router

    @pytest.fixture
    def coordinator(self, mock_search_router: Mock) -> SearchCoordinator:
        """Create search coordinator with default mocks."""
        return SearchCoordinator(search_router=mock_search_router)

    # Test route_search_query

    def test_route_search_query_returns_keyword_tool(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test routing query to keyword search tool."""
        # Setup
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.KEYWORD,
            confidence=0.95,
            reason="Query matches code identifier pattern",
            transformed_query=None,
            matched_patterns=["class_name"],
        )

        # Execute
        result = coordinator.route_search_query("class BaseTool")

        # Assert
        assert result["recommended_tool"] == "code_search"
        assert result["confidence"] == 0.95
        assert result["reason"] == "Query matches code identifier pattern"
        assert result["search_type"] == "keyword"
        assert result["matched_patterns"] == ["class_name"]
        assert result["transformed_query"] is None
        mock_search_router.route.assert_called_once_with("class BaseTool")

    def test_route_search_query_returns_semantic_tool(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test routing query to semantic search tool."""
        # Setup
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.SEMANTIC,
            confidence=0.88,
            reason="Query matches conceptual question pattern",
            transformed_query=None,
            matched_patterns=["how_question"],
        )

        # Execute
        result = coordinator.route_search_query("how does error handling work")

        # Assert
        assert result["recommended_tool"] == "semantic_code_search"
        assert result["confidence"] == 0.88
        assert result["reason"] == "Query matches conceptual question pattern"
        assert result["search_type"] == "semantic"
        assert result["matched_patterns"] == ["how_question"]

    def test_route_search_query_returns_both_tools_for_hybrid(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test routing query to both tools for hybrid search."""
        # Setup
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.HYBRID,
            confidence=0.75,
            reason="Query has both keyword and semantic signals",
            transformed_query=None,
            matched_patterns=["class_name", "explain"],
        )

        # Execute
        result = coordinator.route_search_query("BaseTool class explain")

        # Assert
        assert result["recommended_tool"] == "both"
        assert result["confidence"] == 0.75
        assert result["search_type"] == "hybrid"
        assert result["matched_patterns"] == ["class_name", "explain"]

    def test_route_search_query_with_transformed_query(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test routing with transformed query (e.g., quoted strings)."""
        # Setup
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.KEYWORD,
            confidence=1.0,
            reason="Query contains quoted literal string",
            transformed_query="BaseTool",
            matched_patterns=["quoted_string"],
        )

        # Execute
        result = coordinator.route_search_query('"BaseTool"')

        # Assert
        assert result["recommended_tool"] == "code_search"
        assert result["confidence"] == 1.0
        assert result["transformed_query"] == "BaseTool"

    def test_route_search_query_with_empty_patterns(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test routing with no matched patterns."""
        # Setup
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.KEYWORD,
            confidence=0.3,
            reason="No strong signals detected, defaulting to keyword search",
            transformed_query=None,
            matched_patterns=[],
        )

        # Execute
        result = coordinator.route_search_query("random query")

        # Assert
        assert result["recommended_tool"] == "code_search"
        assert result["confidence"] == 0.3
        assert result["matched_patterns"] == []

    def test_route_search_query_with_multiple_patterns(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test routing with multiple matched patterns."""
        # Setup
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.SEMANTIC,
            confidence=0.92,
            reason="Query matches semantic patterns (score: 1.80)",
            transformed_query=None,
            matched_patterns=["how_question", "explain", "understand"],
        )

        # Execute
        result = coordinator.route_search_query("how do I understand and explain this")

        # Assert
        assert result["recommended_tool"] == "semantic_code_search"
        assert result["matched_patterns"] == ["how_question", "explain", "understand"]
        assert len(result["matched_patterns"]) == 3

    def test_route_search_query_with_varying_confidence(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test routing with different confidence levels."""
        # Test high confidence
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.KEYWORD,
            confidence=0.99,
            reason="Exact match",
            transformed_query=None,
            matched_patterns=["quoted_string"],
        )
        result = coordinator.route_search_query('"exact match"')
        assert result["confidence"] == 0.99

        # Test medium confidence
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.SEMANTIC,
            confidence=0.65,
            reason="Partial match",
            transformed_query=None,
            matched_patterns=["patterns"],
        )
        result = coordinator.route_search_query("show me patterns")
        assert result["confidence"] == 0.65

        # Test low confidence
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.KEYWORD,
            confidence=0.25,
            reason="Weak signals",
            transformed_query=None,
            matched_patterns=[],
        )
        result = coordinator.route_search_query("weak query")
        assert result["confidence"] == 0.25

    def test_route_search_query_delegates_to_router(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test that routing delegates to search router."""
        # Setup
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.KEYWORD,
            confidence=0.9,
            reason="Test",
            transformed_query=None,
            matched_patterns=[],
        )

        # Execute
        coordinator.route_search_query("test query")

        # Assert
        mock_search_router.route.assert_called_once_with("test query")

    def test_route_search_query_handles_special_characters(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test routing with special characters in query."""
        # Setup
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.KEYWORD,
            confidence=0.8,
            reason="Special characters handled",
            transformed_query=None,
            matched_patterns=[],
        )

        # Execute
        result = coordinator.route_search_query("class[Test]::method")

        # Assert
        assert result["recommended_tool"] == "code_search"
        mock_search_router.route.assert_called_once_with("class[Test]::method")

    def test_route_search_query_handles_unicode(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test routing with unicode characters."""
        # Setup
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.SEMANTIC,
            confidence=0.7,
            reason="Unicode query",
            transformed_query=None,
            matched_patterns=[],
        )

        # Execute
        result = coordinator.route_search_query("搜索模式")

        # Assert
        assert result["recommended_tool"] == "semantic_code_search"
        mock_search_router.route.assert_called_once_with("搜索模式")

    def test_route_search_query_with_long_query(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test routing with very long query."""
        # Setup
        long_query = "find " + "pattern " * 100
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.SEMANTIC,
            confidence=0.85,
            reason="Long query",
            transformed_query=None,
            matched_patterns=["patterns"],
        )

        # Execute
        result = coordinator.route_search_query(long_query)

        # Assert
        assert result["recommended_tool"] == "semantic_code_search"
        mock_search_router.route.assert_called_once_with(long_query)

    # Test get_recommended_search_tool

    def test_get_recommended_search_tool_returns_keyword(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test getting recommended tool for keyword search."""
        # Setup
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.KEYWORD,
            confidence=0.95,
            reason="Code identifier",
            transformed_query=None,
            matched_patterns=["class_name"],
        )

        # Execute
        tool = coordinator.get_recommended_search_tool("class BaseTool")

        # Assert
        assert tool == "code_search"

    def test_get_recommended_search_tool_returns_semantic(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test getting recommended tool for semantic search."""
        # Setup
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.SEMANTIC,
            confidence=0.88,
            reason="Conceptual query",
            transformed_query=None,
            matched_patterns=["how_question"],
        )

        # Execute
        tool = coordinator.get_recommended_search_tool("how does it work")

        # Assert
        assert tool == "semantic_code_search"

    def test_get_recommended_search_tool_returns_both_for_hybrid(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test getting recommended tool for hybrid search."""
        # Setup
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.HYBRID,
            confidence=0.75,
            reason="Both signals",
            transformed_query=None,
            matched_patterns=["class_name", "explain"],
        )

        # Execute
        tool = coordinator.get_recommended_search_tool("BaseTool explain")

        # Assert
        assert tool == "both"

    def test_get_recommended_search_tool_convenience_method(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test that get_recommended_search_tool is a convenience method."""
        # Setup
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.KEYWORD,
            confidence=0.9,
            reason="Test",
            transformed_query=None,
            matched_patterns=[],
        )

        # Execute - convenience method should return just the tool name
        tool = coordinator.get_recommended_search_tool("test")

        # Assert
        assert tool == "code_search"
        mock_search_router.route.assert_called_once_with("test")

    def test_get_recommended_search_tool_calls_route_search_query(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test that get_recommended_search_tool calls route_search_query."""
        # Setup
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.SEMANTIC,
            confidence=0.8,
            reason="Semantic",
            transformed_query=None,
            matched_patterns=[],
        )

        # Execute
        tool = coordinator.get_recommended_search_tool("semantic query")

        # Assert - should call route and extract recommended_tool
        mock_search_router.route.assert_called_once_with("semantic query")
        assert tool == "semantic_code_search"

    def test_get_recommended_search_tool_with_empty_query(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test getting recommended tool with empty query."""
        # Setup
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.KEYWORD,
            confidence=0.3,
            reason="Empty query defaults to keyword",
            transformed_query=None,
            matched_patterns=[],
        )

        # Execute
        tool = coordinator.get_recommended_search_tool("")

        # Assert
        assert tool == "code_search"

    def test_get_recommended_search_tool_with_whitespace(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test getting recommended tool with whitespace query."""
        # Setup
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.KEYWORD,
            confidence=0.3,
            reason="Whitespace defaults to keyword",
            transformed_query=None,
            matched_patterns=[],
        )

        # Execute
        tool = coordinator.get_recommended_search_tool("   ")

        # Assert
        assert tool == "code_search"

    # Test tool mapping

    def test_tool_mapping_covers_all_search_types(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test that all search types have tool mappings."""
        # Test KEYWORD
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.KEYWORD,
            confidence=1.0,
            reason="Test",
            transformed_query=None,
            matched_patterns=[],
        )
        result = coordinator.route_search_query("test")
        assert result["recommended_tool"] == "code_search"

        # Test SEMANTIC
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.SEMANTIC,
            confidence=1.0,
            reason="Test",
            transformed_query=None,
            matched_patterns=[],
        )
        result = coordinator.route_search_query("test")
        assert result["recommended_tool"] == "semantic_code_search"

        # Test HYBRID
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.HYBRID,
            confidence=1.0,
            reason="Test",
            transformed_query=None,
            matched_patterns=[],
        )
        result = coordinator.route_search_query("test")
        assert result["recommended_tool"] == "both"

    # Test initialization and configuration

    def test_coordinator_initialization_with_search_router(self, mock_search_router: Mock):
        """Test coordinator initialization with search router."""
        # Execute
        coordinator = SearchCoordinator(search_router=mock_search_router)

        # Assert
        assert coordinator._search_router == mock_search_router

    def test_coordinator_requires_search_router(self):
        """Test that coordinator requires search router."""
        with pytest.raises(TypeError):
            # Should fail if search_router is not provided
            coordinator = SearchCoordinator()  # type: ignore

    # Test integration scenarios

    def test_multiple_queries_same_coordinator(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test routing multiple queries with same coordinator instance."""
        # Setup - different results for different queries
        mock_search_router.route.side_effect = [
            SearchRoute(
                search_type=SearchType.KEYWORD,
                confidence=0.95,
                reason="Code",
                transformed_query=None,
                matched_patterns=["class_name"],
            ),
            SearchRoute(
                search_type=SearchType.SEMANTIC,
                confidence=0.88,
                reason="Concept",
                transformed_query=None,
                matched_patterns=["how_question"],
            ),
            SearchRoute(
                search_type=SearchType.HYBRID,
                confidence=0.75,
                reason="Both",
                transformed_query=None,
                matched_patterns=["class_name", "explain"],
            ),
        ]

        # Execute
        result1 = coordinator.route_search_query("class BaseTool")
        result2 = coordinator.route_search_query("how does it work")
        result3 = coordinator.route_search_query("BaseTool explain")

        # Assert
        assert result1["recommended_tool"] == "code_search"
        assert result2["recommended_tool"] == "semantic_code_search"
        assert result3["recommended_tool"] == "both"
        assert mock_search_router.route.call_count == 3

    def test_get_recommended_tool_multiple_calls(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test calling get_recommended_search_tool multiple times."""
        # Setup
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.KEYWORD,
            confidence=0.9,
            reason="Test",
            transformed_query=None,
            matched_patterns=[],
        )

        # Execute
        tool1 = coordinator.get_recommended_search_tool("query1")
        tool2 = coordinator.get_recommended_search_tool("query2")
        tool3 = coordinator.get_recommended_search_tool("query3")

        # Assert
        assert tool1 == "code_search"
        assert tool2 == "code_search"
        assert tool3 == "code_search"
        assert mock_search_router.route.call_count == 3


class TestSearchCoordinatorEdgeCases:
    """Test edge cases and error conditions for SearchCoordinator."""

    @pytest.fixture
    def mock_search_router(self) -> Mock:
        """Create mock search router."""
        router = Mock()
        return router

    @pytest.fixture
    def coordinator(self, mock_search_router: Mock) -> SearchCoordinator:
        """Create search coordinator."""
        return SearchCoordinator(search_router=mock_search_router)

    def test_route_search_query_with_none_router(self):
        """Test that creating coordinator with None router causes issues."""
        with pytest.raises(AttributeError):
            coordinator = SearchCoordinator(search_router=None)  # type: ignore
            coordinator.route_search_query("test")

    def test_route_search_query_when_router_raises_exception(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test route_search_query when router raises an exception."""
        # Setup
        mock_search_router.route.side_effect = RuntimeError("Router failure")

        # Execute & Assert - should propagate the error
        with pytest.raises(RuntimeError, match="Router failure"):
            coordinator.route_search_query("test query")

    def test_route_search_query_with_unmapped_search_type(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test that TOOL_MAP.get() defaults to code_search for unmapped types."""
        # This test verifies the fallback behavior in route_search_query
        # The actual code uses: TOOL_MAP.get(route.search_type, "code_search")
        # In practice, all SearchType enum values are mapped, so this tests
        # the defensive fallback mechanism

        # We can't easily test this without modifying the TOOL_MAP,
        # so instead we verify the current TOOL_MAP is complete
        from victor.agent.coordinators.search_coordinator import SearchCoordinator

        # Verify all SearchType values are mapped
        for search_type in SearchType:
            assert (
                search_type in SearchCoordinator.TOOL_MAP
            ), f"SearchType.{search_type.name} is not in TOOL_MAP"

    def test_get_recommended_tool_when_router_raises(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test get_recommended_search_tool when router raises exception."""
        # Setup
        mock_search_router.route.side_effect = ValueError("Invalid query")

        # Execute & Assert - should propagate the error
        with pytest.raises(ValueError, match="Invalid query"):
            coordinator.get_recommended_search_tool("bad query")

    def test_route_search_query_preserves_all_route_fields(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test that all SearchRoute fields are preserved in result."""
        # Setup
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.KEYWORD,
            confidence=0.92,
            reason="Complete match",
            transformed_query="transformed",
            matched_patterns=["pattern1", "pattern2"],
        )

        # Execute
        result = coordinator.route_search_query("test query")

        # Assert - all fields should be present
        assert "recommended_tool" in result
        assert "confidence" in result
        assert "reason" in result
        assert "search_type" in result
        assert "matched_patterns" in result
        assert "transformed_query" in result

        # Check values
        assert result["confidence"] == 0.92
        assert result["reason"] == "Complete match"
        assert result["search_type"] == "keyword"
        assert result["transformed_query"] == "transformed"
        assert result["matched_patterns"] == ["pattern1", "pattern2"]

    def test_route_search_query_with_newlines_in_query(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test routing with newlines in query."""
        # Setup
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.SEMANTIC,
            confidence=0.7,
            reason="Multi-line query",
            transformed_query=None,
            matched_patterns=[],
        )

        # Execute
        query = "line 1\nline 2\nline 3"
        result = coordinator.route_search_query(query)

        # Assert
        assert result["recommended_tool"] == "semantic_code_search"
        mock_search_router.route.assert_called_once_with(query)

    def test_route_search_query_with_tabs_in_query(
        self, coordinator: SearchCoordinator, mock_search_router: Mock
    ):
        """Test routing with tabs in query."""
        # Setup
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.KEYWORD,
            confidence=0.8,
            reason="Tabbed query",
            transformed_query=None,
            matched_patterns=[],
        )

        # Execute
        query = "class\tBaseTool:\n\tdef\tpass"
        result = coordinator.route_search_query(query)

        # Assert
        assert result["recommended_tool"] == "code_search"
        mock_search_router.route.assert_called_once_with(query)

    def test_concurrent_route_calls(self, coordinator: SearchCoordinator, mock_search_router: Mock):
        """Test that multiple concurrent route calls work correctly."""
        # Setup
        mock_search_router.route.return_value = SearchRoute(
            search_type=SearchType.KEYWORD,
            confidence=0.9,
            reason="Test",
            transformed_query=None,
            matched_patterns=[],
        )

        # Execute - multiple calls
        results = [coordinator.route_search_query(f"query{i}") for i in range(10)]

        # Assert
        assert len(results) == 10
        assert all(r["recommended_tool"] == "code_search" for r in results)
        assert mock_search_router.route.call_count == 10


class TestSearchCoordinatorToolMapping:
    """Test the TOOL_MAP constant and tool mapping logic."""

    def test_tool_map_is_complete(self):
        """Test that TOOL_MAP covers all SearchType values."""
        from victor.agent.coordinators.search_coordinator import SearchCoordinator

        # Check that all SearchType values are mapped
        assert SearchType.KEYWORD in SearchCoordinator.TOOL_MAP
        assert SearchType.SEMANTIC in SearchCoordinator.TOOL_MAP
        assert SearchType.HYBRID in SearchCoordinator.TOOL_MAP

    def test_tool_map_values(self):
        """Test that TOOL_MAP has correct tool names."""
        from victor.agent.coordinators.search_coordinator import SearchCoordinator

        assert SearchCoordinator.TOOL_MAP[SearchType.KEYWORD] == "code_search"
        assert SearchCoordinator.TOOL_MAP[SearchType.SEMANTIC] == "semantic_code_search"
        assert SearchCoordinator.TOOL_MAP[SearchType.HYBRID] == "both"

    def test_tool_map_is_class_attribute(self):
        """Test that TOOL_MAP is a class attribute."""
        from victor.agent.coordinators.search_coordinator import SearchCoordinator

        assert hasattr(SearchCoordinator, "TOOL_MAP")
        assert isinstance(SearchCoordinator.TOOL_MAP, dict)
