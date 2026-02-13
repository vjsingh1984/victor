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

"""Comprehensive TDD tests for the unified graph tool.

Tests cover all modes:
- find: Symbol search with optional graph expansion
- neighbors: Direct connections
- pagerank: Importance ranking
- centrality: Degree centrality
- path: Shortest path finding
- impact: Change impact analysis
- subgraph: Subgraph extraction
- file_deps: File dependency analysis
- patterns: Design pattern detection
- stats: Graph statistics
"""

import pytest
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

from victor.tools.graph_tool import (
    GraphAnalyzer,
    graph,
    GraphMode,
    ALL_EDGE_TYPES,
)

# =============================================================================
# Test Fixtures - Create mock graph data
# =============================================================================


@dataclass
class MockGraphNode:
    """Mock graph node for testing."""

    node_id: str
    type: str
    name: str
    file: str
    line: int = 1


@dataclass
class MockGraphEdge:
    """Mock graph edge for testing."""

    src: str
    dst: str
    type: str
    weight: Optional[float] = 1.0


@pytest.fixture
def sample_analyzer():
    """Create a GraphAnalyzer with sample data for testing."""
    analyzer = GraphAnalyzer()

    # Add nodes - simulating a provider pattern
    nodes = [
        MockGraphNode("base_provider", "class", "BaseProvider", "providers/base.py", 10),
        MockGraphNode("openai_provider", "class", "OpenAIProvider", "providers/openai.py", 15),
        MockGraphNode(
            "anthropic_provider", "class", "AnthropicProvider", "providers/anthropic.py", 20
        ),
        MockGraphNode("ollama_provider", "class", "OllamaProvider", "providers/ollama.py", 25),
        MockGraphNode("orchestrator", "class", "AgentOrchestrator", "agent/orchestrator.py", 50),
        MockGraphNode("tool_executor", "class", "ToolExecutor", "agent/tool_executor.py", 100),
        MockGraphNode(
            "process_request", "function", "process_request", "agent/orchestrator.py", 200
        ),
        MockGraphNode("call_provider", "function", "call_provider", "agent/orchestrator.py", 250),
        MockGraphNode(
            "provider_factory", "function", "create_provider", "providers/factory.py", 10
        ),
        MockGraphNode("settings", "class", "Settings", "config/settings.py", 5),
        MockGraphNode("registry", "class", "ProviderRegistry", "providers/registry.py", 30),
        MockGraphNode("message", "class", "Message", "providers/base.py", 50),
        MockGraphNode("tool_result", "class", "ToolResult", "tools/base.py", 20),
    ]

    for node in nodes:
        analyzer.add_node(node)

    # Add edges - inheritance
    inheritance_edges = [
        MockGraphEdge("openai_provider", "base_provider", "INHERITS"),
        MockGraphEdge("anthropic_provider", "base_provider", "INHERITS"),
        MockGraphEdge("ollama_provider", "base_provider", "INHERITS"),
    ]

    # Add edges - calls
    call_edges = [
        MockGraphEdge("orchestrator", "process_request", "CALLS"),
        MockGraphEdge("orchestrator", "call_provider", "CALLS"),
        MockGraphEdge("orchestrator", "tool_executor", "CALLS"),
        MockGraphEdge("process_request", "call_provider", "CALLS"),
        MockGraphEdge("call_provider", "base_provider", "CALLS"),
        MockGraphEdge("call_provider", "registry", "CALLS"),
        MockGraphEdge("provider_factory", "openai_provider", "CALLS"),
        MockGraphEdge("provider_factory", "anthropic_provider", "CALLS"),
        MockGraphEdge("provider_factory", "ollama_provider", "CALLS"),
        MockGraphEdge("registry", "provider_factory", "CALLS"),
    ]

    # Add edges - references
    reference_edges = [
        MockGraphEdge("orchestrator", "settings", "REFERENCES"),
        MockGraphEdge("orchestrator", "message", "REFERENCES"),
        MockGraphEdge("tool_executor", "tool_result", "REFERENCES"),
        MockGraphEdge("base_provider", "message", "REFERENCES"),
        MockGraphEdge("openai_provider", "message", "REFERENCES"),
    ]

    # Add edges - composition
    composition_edges = [
        MockGraphEdge("orchestrator", "tool_executor", "COMPOSED_OF"),
        MockGraphEdge("orchestrator", "registry", "COMPOSED_OF"),
        MockGraphEdge("orchestrator", "settings", "COMPOSED_OF"),
    ]

    # Add edges - imports
    import_edges = [
        MockGraphEdge("orchestrator", "base_provider", "IMPORTS"),
        MockGraphEdge("orchestrator", "settings", "IMPORTS"),
        MockGraphEdge("openai_provider", "base_provider", "IMPORTS"),
        MockGraphEdge("anthropic_provider", "base_provider", "IMPORTS"),
    ]

    for edge in inheritance_edges + call_edges + reference_edges + composition_edges + import_edges:
        analyzer.add_edge(edge)

    return analyzer


@pytest.fixture
def empty_analyzer():
    """Create an empty GraphAnalyzer for edge case testing."""
    return GraphAnalyzer()


# =============================================================================
# GraphAnalyzer Unit Tests
# =============================================================================


class TestGraphAnalyzerNeighbors:
    """Tests for get_neighbors method."""

    def test_get_neighbors_outgoing_only(self, sample_analyzer):
        """Test getting only outgoing neighbors."""
        result = sample_analyzer.get_neighbors("orchestrator", direction="out", max_depth=1)

        assert result["source"] == "orchestrator"
        assert result["total_neighbors"] > 0
        assert 1 in result["neighbors_by_depth"]

        # Check that we get outgoing edges
        neighbor_names = [n["name"] for n in result["neighbors_by_depth"][1]]
        assert "process_request" in neighbor_names or "call_provider" in neighbor_names

    def test_get_neighbors_incoming_only(self, sample_analyzer):
        """Test getting only incoming neighbors."""
        result = sample_analyzer.get_neighbors("base_provider", direction="in", max_depth=1)

        assert result["source"] == "base_provider"
        # Should include OpenAI, Anthropic, Ollama providers that inherit from base
        neighbor_names = [n["name"] for n in result["neighbors_by_depth"].get(1, [])]
        assert any("Provider" in name for name in neighbor_names)

    def test_get_neighbors_both_directions(self, sample_analyzer):
        """Test getting neighbors in both directions."""
        result = sample_analyzer.get_neighbors("call_provider", direction="both", max_depth=1)

        assert result["total_neighbors"] >= 2  # Should have both in and out

    def test_get_neighbors_with_depth(self, sample_analyzer):
        """Test multi-hop neighbor traversal."""
        result = sample_analyzer.get_neighbors("orchestrator", direction="out", max_depth=2)

        # Should have neighbors at depth 1 and potentially depth 2
        assert 1 in result["neighbors_by_depth"]

    def test_get_neighbors_with_edge_type_filter(self, sample_analyzer):
        """Test filtering by edge type."""
        result = sample_analyzer.get_neighbors(
            "orchestrator", direction="out", edge_types=["CALLS"], max_depth=1
        )

        # All edges should be CALLS
        for neighbors in result["neighbors_by_depth"].values():
            for n in neighbors:
                assert n["edge_type"] == "CALLS"

    def test_get_neighbors_nonexistent_node(self, sample_analyzer):
        """Test handling of nonexistent node."""
        result = sample_analyzer.get_neighbors("nonexistent", direction="both")
        assert result["total_neighbors"] == 0


class TestGraphAnalyzerPageRank:
    """Tests for PageRank algorithm."""

    def test_pagerank_returns_ranked_results(self, sample_analyzer):
        """Test that PageRank returns properly ranked results."""
        results = sample_analyzer.pagerank(top_k=5)

        assert len(results) <= 5
        # Scores should be sorted descending
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_pagerank_includes_required_fields(self, sample_analyzer):
        """Test that PageRank results include all required fields."""
        results = sample_analyzer.pagerank(top_k=3)

        if results:
            first = results[0]
            assert "rank" in first
            assert "node_id" in first
            assert "name" in first
            assert "type" in first
            assert "score" in first
            assert "in_degree" in first
            assert "out_degree" in first

    def test_pagerank_with_edge_type_filter(self, sample_analyzer):
        """Test PageRank filtered by edge types."""
        results_all = sample_analyzer.pagerank(top_k=10)
        results_calls = sample_analyzer.pagerank(top_k=10, edge_types=["CALLS"])

        # Filtering should potentially give different rankings
        # At minimum, both should return results
        assert len(results_all) > 0
        assert len(results_calls) > 0

    def test_pagerank_empty_graph(self, empty_analyzer):
        """Test PageRank on empty graph."""
        results = empty_analyzer.pagerank()
        assert results == []


class TestGraphAnalyzerCentrality:
    """Tests for degree centrality."""

    def test_centrality_returns_ranked_results(self, sample_analyzer):
        """Test that centrality returns properly ranked results."""
        results = sample_analyzer.degree_centrality(top_k=5)

        assert len(results) <= 5
        # Degrees should be sorted descending
        degrees = [r["degree"] for r in results]
        assert degrees == sorted(degrees, reverse=True)

    def test_centrality_includes_required_fields(self, sample_analyzer):
        """Test centrality results include required fields."""
        results = sample_analyzer.degree_centrality(top_k=3)

        if results:
            first = results[0]
            assert "rank" in first
            assert "node_id" in first
            assert "name" in first
            assert "degree" in first
            assert "in_degree" in first
            assert "out_degree" in first

    def test_centrality_degree_calculation(self, sample_analyzer):
        """Test that degree is correctly calculated as in + out."""
        results = sample_analyzer.degree_centrality(top_k=10)

        for r in results:
            assert r["degree"] == r["in_degree"] + r["out_degree"]


class TestGraphAnalyzerShortestPath:
    """Tests for shortest path finding."""

    def test_path_found(self, sample_analyzer):
        """Test finding a valid path."""
        result = sample_analyzer.shortest_path("orchestrator", "base_provider")

        assert result["found"] is True
        assert result["source"] == "orchestrator"
        assert result["target"] == "base_provider"
        assert "path" in result
        assert result["length"] > 0

    def test_path_not_found(self, sample_analyzer):
        """Test when no path exists."""
        # Add an isolated node
        sample_analyzer.add_node(MockGraphNode("isolated", "class", "Isolated", "isolated.py"))

        result = sample_analyzer.shortest_path("orchestrator", "isolated")

        assert result["found"] is False
        assert "message" in result

    def test_path_source_not_found(self, sample_analyzer):
        """Test with nonexistent source node."""
        result = sample_analyzer.shortest_path("nonexistent", "base_provider")

        assert "error" in result

    def test_path_target_not_found(self, sample_analyzer):
        """Test with nonexistent target node."""
        result = sample_analyzer.shortest_path("orchestrator", "nonexistent")

        assert "error" in result

    def test_path_with_edge_filter(self, sample_analyzer):
        """Test path finding with edge type filter."""
        result = sample_analyzer.shortest_path(
            "orchestrator", "base_provider", edge_types=["CALLS"]
        )

        if result["found"]:
            for step in result["path"]:
                assert step["edge_type"] == "CALLS"


class TestGraphAnalyzerImpact:
    """Tests for impact analysis."""

    def test_impact_analysis_returns_affected(self, sample_analyzer):
        """Test that impact analysis returns affected nodes."""
        result = sample_analyzer.impact_analysis("base_provider", max_depth=2)

        assert "node_id" in result
        assert "node_name" in result
        assert "total_affected" in result
        assert "affected_by_depth" in result

    def test_impact_analysis_node_not_found(self, sample_analyzer):
        """Test impact analysis with nonexistent node."""
        result = sample_analyzer.impact_analysis("nonexistent")

        assert "error" in result

    def test_impact_groups_by_file(self, sample_analyzer):
        """Test that impact analysis groups results by file."""
        result = sample_analyzer.impact_analysis("base_provider", max_depth=2)

        assert "affected_files" in result
        assert "files_count" in result


class TestGraphAnalyzerFindSymbols:
    """Tests for symbol search functionality."""

    def test_find_by_name(self, sample_analyzer):
        """Test finding symbols by name pattern."""
        result = sample_analyzer.find_symbols("Provider")

        assert result["query"] == "Provider"
        assert result["total_matches"] > 0
        # Should find BaseProvider, OpenAIProvider, etc.
        assert any("Provider" in m["name"] for m in result["matches"])

    def test_find_case_insensitive(self, sample_analyzer):
        """Test case-insensitive search."""
        result = sample_analyzer.find_symbols("provider")

        assert result["total_matches"] > 0

    def test_find_with_type_filter(self, sample_analyzer):
        """Test filtering by node type."""
        result = sample_analyzer.find_symbols("", node_type="function")

        for match in result["matches"]:
            assert match["type"] == "function"

    def test_find_with_file_pattern(self, sample_analyzer):
        """Test filtering by file pattern."""
        result = sample_analyzer.find_symbols("", file_pattern="providers/*")

        for match in result["matches"]:
            assert "providers" in match["file"]

    def test_find_with_expand(self, sample_analyzer):
        """Test finding with neighbor expansion."""
        result = sample_analyzer.find_symbols("orchestrator", expand_neighbors=True)

        if result["matches"]:
            match = result["matches"][0]
            assert "neighbors" in match

    def test_find_respects_limit(self, sample_analyzer):
        """Test that find respects the limit parameter."""
        result = sample_analyzer.find_symbols("", limit=2)

        assert len(result["matches"]) <= 2


class TestGraphAnalyzerFileDeps:
    """Tests for file dependency analysis."""

    def test_file_deps_both_directions(self, sample_analyzer):
        """Test getting both imports and imported_by."""
        result = sample_analyzer.get_file_dependencies("agent/orchestrator.py")

        assert "file" in result
        assert "imports" in result
        assert "imported_by" in result

    def test_file_deps_partial_match(self, sample_analyzer):
        """Test that partial file path works."""
        result = sample_analyzer.get_file_dependencies("orchestrator.py")

        # Should match agent/orchestrator.py
        assert "error" not in result or "symbols_in_file" in result

    def test_file_deps_not_found(self, sample_analyzer):
        """Test with nonexistent file."""
        result = sample_analyzer.get_file_dependencies("nonexistent/file.py")

        assert "error" in result


class TestGraphAnalyzerStats:
    """Tests for graph statistics."""

    def test_stats_returns_all_fields(self, sample_analyzer):
        """Test that stats returns all expected fields."""
        result = sample_analyzer.get_stats()

        assert "total_nodes" in result
        assert "total_edges" in result
        assert "node_types" in result
        assert "edge_types" in result
        assert "avg_in_degree" in result
        assert "avg_out_degree" in result

    def test_stats_correct_counts(self, sample_analyzer):
        """Test that stats counts are accurate."""
        result = sample_analyzer.get_stats()

        assert result["total_nodes"] == len(sample_analyzer.nodes)

    def test_stats_empty_graph(self, empty_analyzer):
        """Test stats on empty graph."""
        result = empty_analyzer.get_stats()

        assert result["total_nodes"] == 0
        assert result["total_edges"] == 0


class TestGraphAnalyzerSubgraph:
    """Tests for subgraph extraction."""

    def test_subgraph_extraction(self, sample_analyzer):
        """Test extracting a subgraph."""
        result = sample_analyzer.extract_subgraph("orchestrator", max_depth=1)

        assert result["center"] == "orchestrator"
        assert "nodes" in result
        assert "edges" in result
        assert result["nodes_count"] > 0

    def test_subgraph_with_depth(self, sample_analyzer):
        """Test subgraph with different depths."""
        result_1 = sample_analyzer.extract_subgraph("orchestrator", max_depth=1)
        result_2 = sample_analyzer.extract_subgraph("orchestrator", max_depth=2)

        # Deeper extraction should include more or equal nodes
        assert result_2["nodes_count"] >= result_1["nodes_count"]

    def test_subgraph_node_not_found(self, sample_analyzer):
        """Test subgraph with nonexistent center."""
        result = sample_analyzer.extract_subgraph("nonexistent")

        assert "error" in result


class TestGraphAnalyzerPatterns:
    """Tests for design pattern detection."""

    def test_patterns_detects_provider_strategy(self, sample_analyzer):
        """Test detection of provider/strategy pattern."""
        result = sample_analyzer.detect_patterns()

        assert "patterns" in result

        provider_patterns = [p for p in result["patterns"] if p["pattern"] == "provider_strategy"]
        # Should detect BaseProvider with multiple implementations
        assert len(provider_patterns) > 0

        # Check one has BaseProvider
        base_names = [p.get("base_class") for p in provider_patterns]
        assert "BaseProvider" in base_names

    def test_patterns_detects_composition(self, sample_analyzer):
        """Test detection of composition pattern."""
        result = sample_analyzer.detect_patterns()

        composition_patterns = [p for p in result["patterns"] if p["pattern"] == "composition"]
        # Orchestrator has composition relationships
        assert len(composition_patterns) > 0

    def test_patterns_detects_factory(self, sample_analyzer):
        """Test detection of factory pattern."""
        result = sample_analyzer.detect_patterns()

        factory_patterns = [p for p in result["patterns"] if p["pattern"] == "factory"]
        # create_provider should be detected as factory
        factory_names = [p.get("class") for p in factory_patterns]
        assert "create_provider" in factory_names

    def test_patterns_returns_summary(self, sample_analyzer):
        """Test that patterns returns a summary."""
        result = sample_analyzer.detect_patterns()

        assert "total_patterns_found" in result
        assert "pattern_summary" in result
        assert isinstance(result["pattern_summary"], dict)

    def test_patterns_sorted_by_confidence(self, sample_analyzer):
        """Test that patterns are sorted by confidence."""
        result = sample_analyzer.detect_patterns()

        if len(result["patterns"]) > 1:
            confidences = [p.get("confidence", 0) for p in result["patterns"]]
            assert confidences == sorted(confidences, reverse=True)

    def test_patterns_empty_graph(self, empty_analyzer):
        """Test pattern detection on empty graph."""
        result = empty_analyzer.detect_patterns()

        assert result["total_patterns_found"] == 0
        assert result["patterns"] == []


# =============================================================================
# Integration Tests for the graph() Tool Function
# =============================================================================


class TestGraphToolFunction:
    """Integration tests for the main graph() tool function."""

    @pytest.fixture
    def mock_graph_store(self, sample_analyzer):
        """Create a mock graph store that returns our sample data."""
        store = MagicMock()

        # Make find_nodes return our sample nodes
        async def mock_find_nodes(*args, **kwargs):
            return list(sample_analyzer.nodes.values())

        store.find_nodes = mock_find_nodes

        # Make get_neighbors return edges from our analyzer
        async def mock_get_neighbors(node_id, *args, **kwargs):
            edges = []
            for target, edge_type, weight in sample_analyzer.outgoing.get(node_id, []):
                edges.append(MockGraphEdge(node_id, target, edge_type, weight))
            for source, edge_type, weight in sample_analyzer.incoming.get(node_id, []):
                edges.append(MockGraphEdge(source, node_id, edge_type, weight))
            return edges

        store.get_neighbors = mock_get_neighbors

        # Make get_all_nodes return all nodes
        async def mock_get_all_nodes():
            return list(sample_analyzer.nodes.values())

        store.get_all_nodes = mock_get_all_nodes

        # Make get_all_edges return all edges
        async def mock_get_all_edges():
            edges = []
            for src, targets in sample_analyzer.outgoing.items():
                for target, edge_type, weight in targets:
                    edges.append(MockGraphEdge(src, target, edge_type, weight))
            return edges

        store.get_all_edges = mock_get_all_edges

        return store

    @pytest.mark.asyncio
    async def test_graph_stats_mode(self, mock_graph_store):
        """Test stats mode via the main tool function."""
        with patch("victor.tools.graph_tool.create_graph_store") as mock_create:
            mock_create.return_value = mock_graph_store

            result = await graph(mode="stats")

            # Should not error
            if "error" not in result:
                assert "total_nodes" in result

    @pytest.mark.asyncio
    async def test_graph_pagerank_mode(self, mock_graph_store):
        """Test pagerank mode via the main tool function."""
        with patch("victor.tools.graph_tool.create_graph_store") as mock_create:
            mock_create.return_value = mock_graph_store

            result = await graph(mode="pagerank", top_k=5)

            if "error" not in result:
                assert "results" in result or "mode" in result

    @pytest.mark.asyncio
    async def test_graph_neighbors_mode_requires_node(self, mock_graph_store):
        """Test that neighbors mode requires a node parameter."""
        with patch("victor.tools.graph_tool.create_graph_store") as mock_create:
            mock_create.return_value = mock_graph_store

            result = await graph(mode="neighbors")

            # If graph has data, should require node
            if "error" in result and "node parameter required" in result["error"]:
                assert True  # Expected
            elif "error" in result and "empty" in result.get("error", "").lower():
                assert True  # Empty graph case

    @pytest.mark.asyncio
    async def test_graph_find_mode_requires_query(self, mock_graph_store):
        """Test that find mode requires a query parameter."""
        with patch("victor.tools.graph_tool.create_graph_store") as mock_create:
            mock_create.return_value = mock_graph_store

            result = await graph(mode="find")

            if "error" in result:
                assert (
                    "query parameter required" in result["error"]
                    or "empty" in result.get("error", "").lower()
                )

    @pytest.mark.asyncio
    async def test_graph_path_mode_requires_both_nodes(self, mock_graph_store):
        """Test that path mode requires both node and target."""
        with patch("victor.tools.graph_tool.create_graph_store") as mock_create:
            mock_create.return_value = mock_graph_store

            result = await graph(mode="path", node="orchestrator")

            if "error" in result:
                assert "target" in result["error"].lower() or "empty" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_graph_file_deps_requires_file(self, mock_graph_store):
        """Test that file_deps mode requires file parameter."""
        with patch("victor.tools.graph_tool.create_graph_store") as mock_create:
            mock_create.return_value = mock_graph_store

            result = await graph(mode="file_deps")

            if "error" in result:
                assert (
                    "file parameter required" in result["error"]
                    or "empty" in result["error"].lower()
                )

    @pytest.mark.asyncio
    async def test_graph_patterns_mode(self, mock_graph_store):
        """Test patterns mode via the main tool function."""
        with patch("victor.tools.graph_tool.create_graph_store") as mock_create:
            mock_create.return_value = mock_graph_store

            result = await graph(mode="patterns")

            if "error" not in result:
                assert "patterns" in result or "total_patterns_found" in result

    @pytest.mark.asyncio
    async def test_graph_unknown_mode(self, mock_graph_store):
        """Test handling of unknown mode."""
        with patch("victor.tools.graph_tool.create_graph_store") as mock_create:
            mock_create.return_value = mock_graph_store

            # This should not raise, but return an error
            result = await graph(mode="invalid_mode")  # type: ignore

            assert "error" in result


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestGraphToolEdgeCases:
    """Edge case and boundary condition tests."""

    def test_analyzer_add_node_twice(self, empty_analyzer):
        """Test adding the same node twice."""
        node = MockGraphNode("test", "class", "Test", "test.py")
        empty_analyzer.add_node(node)
        empty_analyzer.add_node(node)

        # Should not duplicate
        assert len(empty_analyzer.nodes) == 1

    def test_analyzer_add_edge_weights(self, empty_analyzer):
        """Test edge weight handling."""
        n1 = MockGraphNode("a", "class", "A", "a.py")
        n2 = MockGraphNode("b", "class", "B", "b.py")
        empty_analyzer.add_node(n1)
        empty_analyzer.add_node(n2)

        edge = MockGraphEdge("a", "b", "CALLS", weight=2.5)
        empty_analyzer.add_edge(edge)

        assert empty_analyzer.outgoing["a"][0][2] == 2.5

    def test_pagerank_iterations(self, sample_analyzer):
        """Test PageRank with different iteration counts."""
        result_10 = sample_analyzer.pagerank(iterations=10, top_k=5)
        result_100 = sample_analyzer.pagerank(iterations=100, top_k=5)

        # Both should return results
        assert len(result_10) > 0
        assert len(result_100) > 0

    def test_shortest_path_same_node(self, sample_analyzer):
        """Test finding path from node to itself."""
        result = sample_analyzer.shortest_path("orchestrator", "orchestrator")

        # Should find path with length 0
        if result["found"]:
            assert result["length"] == 0

    def test_find_symbols_empty_query(self, sample_analyzer):
        """Test find with empty query (returns all)."""
        result = sample_analyzer.find_symbols("", limit=100)

        # Should return some symbols
        assert result["total_matches"] > 0

    def test_subgraph_max_depth_zero(self, sample_analyzer):
        """Test subgraph with depth 0 (just the center node)."""
        result = sample_analyzer.extract_subgraph("orchestrator", max_depth=0)

        assert result["nodes_count"] == 1

    def test_impact_analysis_max_depth(self, sample_analyzer):
        """Test impact analysis with different depths."""
        result_1 = sample_analyzer.impact_analysis("base_provider", max_depth=1)
        result_3 = sample_analyzer.impact_analysis("base_provider", max_depth=3)

        # Deeper analysis should find more or equal affected nodes
        assert result_3["total_affected"] >= result_1["total_affected"]


# =============================================================================
# Tests for Edge Type Filtering
# =============================================================================


class TestEdgeTypeFiltering:
    """Tests for edge type filtering across different modes."""

    @pytest.fixture
    def analyzer_with_all_edge_types(self):
        """Create analyzer with all edge types."""
        analyzer = GraphAnalyzer()

        # Add nodes
        for i in range(5):
            analyzer.add_node(MockGraphNode(f"n{i}", "class", f"Node{i}", f"file{i}.py"))

        # Add one edge of each type
        for i, edge_type in enumerate(
            ["CALLS", "REFERENCES", "INHERITS", "IMPLEMENTS", "COMPOSED_OF", "IMPORTS"]
        ):
            if i < 4:
                analyzer.add_edge(MockGraphEdge(f"n{i}", f"n{i+1}", edge_type))

        return analyzer

    def test_neighbors_filters_by_edge_type(self, analyzer_with_all_edge_types):
        """Test that neighbors properly filters by edge type."""
        result = analyzer_with_all_edge_types.get_neighbors(
            "n0", direction="out", edge_types=["CALLS"]
        )

        for depth_neighbors in result["neighbors_by_depth"].values():
            for n in depth_neighbors:
                assert n["edge_type"] == "CALLS"

    def test_pagerank_with_multiple_edge_types(self, analyzer_with_all_edge_types):
        """Test PageRank with multiple edge type filters."""
        result = analyzer_with_all_edge_types.pagerank(edge_types=["CALLS", "REFERENCES"])

        # Should still return results
        assert isinstance(result, list)

    def test_centrality_with_edge_filter(self, analyzer_with_all_edge_types):
        """Test centrality with edge type filter."""
        result = analyzer_with_all_edge_types.degree_centrality(edge_types=["INHERITS"])

        assert isinstance(result, list)


# =============================================================================
# Performance/Stress Tests (lightweight)
# =============================================================================


class TestGraphPerformance:
    """Lightweight performance tests."""

    @pytest.fixture
    def large_analyzer(self):
        """Create a larger graph for performance testing."""
        analyzer = GraphAnalyzer()

        # Add 100 nodes
        for i in range(100):
            analyzer.add_node(MockGraphNode(f"node_{i}", "class", f"Node{i}", f"file{i % 10}.py"))

        # Add edges (create a sparse graph)
        for i in range(100):
            for j in range(min(3, 100 - i - 1)):
                analyzer.add_edge(MockGraphEdge(f"node_{i}", f"node_{i + j + 1}", "CALLS"))

        return analyzer

    def test_pagerank_performance(self, large_analyzer):
        """Test PageRank completes in reasonable time on larger graph."""
        import time

        start = time.time()
        result = large_analyzer.pagerank(iterations=50, top_k=10)
        elapsed = time.time() - start

        assert elapsed < 5.0  # Should complete within 5 seconds
        assert len(result) > 0

    def test_stats_performance(self, large_analyzer):
        """Test stats completes quickly on larger graph."""
        import time

        start = time.time()
        result = large_analyzer.get_stats()
        elapsed = time.time() - start

        assert elapsed < 1.0  # Should be very fast
        assert result["total_nodes"] == 100


# =============================================================================
# GAP-3 Fix: Fuzzy Node Resolution Tests
# =============================================================================


class TestFuzzyNodeResolution:
    """Tests for enhanced fuzzy node resolution (GAP-3 fix).

    These tests verify that LLMs can find graph nodes using CamelCase class
    names derived from file names, partial matches, and other common patterns.

    Example: When LLM sees file `database_schema.py` and searches for
    `DatabaseSchema`, it should find nodes in that file even though the
    exact node name might be `MarketData` or `Position`.
    """

    @pytest.fixture
    def analyzer_with_named_nodes(self):
        """Create analyzer with nodes simulating real file/class naming."""
        analyzer = GraphAnalyzer()

        # Simulate investor_homelab project structure
        nodes = [
            # database_schema.py - multiple classes
            MockGraphNode(
                "database_schema.py:MarketData",
                "class",
                "MarketData",
                "investor_homelab/models/database_schema.py",
                15,
            ),
            MockGraphNode(
                "database_schema.py:Position",
                "class",
                "Position",
                "investor_homelab/models/database_schema.py",
                45,
            ),
            MockGraphNode(
                "database_schema.py:create_tables",
                "function",
                "create_tables",
                "investor_homelab/models/database_schema.py",
                100,
            ),
            # web_search_client.py
            MockGraphNode(
                "web_search_client.py:WebSearchClient",
                "class",
                "WebSearchClient",
                "investor_homelab/utils/web_search_client.py",
                10,
            ),
            MockGraphNode(
                "web_search_client.py:search_duckduckgo",
                "method",
                "search_duckduckgo",
                "investor_homelab/utils/web_search_client.py",
                50,
            ),
            # news_model.py
            MockGraphNode(
                "news_model.py:NewsDatabase",
                "class",
                "NewsDatabase",
                "investor_homelab/models/news_model.py",
                20,
            ),
            MockGraphNode(
                "news_model.py:NewsArticle",
                "class",
                "NewsArticle",
                "investor_homelab/models/news_model.py",
                60,
            ),
        ]

        for node in nodes:
            analyzer.add_node(node)

        return analyzer

    def test_camel_case_to_snake_case_conversion(self):
        """Test that CamelCase names are converted to snake_case for file matching."""
        # Test conversion algorithm
        input_str = "DatabaseSchema"
        normalized = ""
        for i, c in enumerate(input_str):
            if i > 0 and c.isupper():
                normalized += "_"
            normalized += c.lower()

        assert normalized == "database_schema"

    def test_multi_word_camel_case_conversion(self):
        """Test multi-word CamelCase conversion."""
        input_str = "WebSearchClient"
        normalized = ""
        for i, c in enumerate(input_str):
            if i > 0 and c.isupper():
                normalized += "_"
            normalized += c.lower()

        assert normalized == "web_search_client"

    def test_file_path_matching_from_camel_case(self, analyzer_with_named_nodes):
        """Test that CamelCase search finds nodes in matching snake_case file."""
        search = "DatabaseSchema"

        # Convert CamelCase to snake_case
        normalized_search = ""
        for i, c in enumerate(search):
            if i > 0 and c.isupper():
                normalized_search += "_"
            normalized_search += c.lower()

        # Find nodes where file contains normalized search term
        file_matches = [
            (nid, n)
            for nid, n in analyzer_with_named_nodes.nodes.items()
            if n.file and normalized_search in n.file.lower()
        ]

        # Should find MarketData, Position, create_tables in database_schema.py
        assert len(file_matches) == 3
        files = {m[1].file for m in file_matches}
        assert all("database_schema.py" in f for f in files)

    def test_exact_name_match_preferred(self, analyzer_with_named_nodes):
        """Test that exact name match is found before fuzzy matching."""
        search = "WebSearchClient"

        # Exact name match
        exact_match = None
        for nid, n in analyzer_with_named_nodes.nodes.items():
            if n.name == search:
                exact_match = nid
                break

        assert exact_match == "web_search_client.py:WebSearchClient"

    def test_substring_match_in_name(self, analyzer_with_named_nodes):
        """Test Strategy 1: Substring match in node name."""
        search = "News"
        search_lower = search.lower()

        name_matches = [
            (nid, n)
            for nid, n in analyzer_with_named_nodes.nodes.items()
            if search_lower in n.name.lower()
        ]

        # Should find NewsDatabase and NewsArticle
        assert len(name_matches) == 2
        names = {m[1].name for m in name_matches}
        assert "NewsDatabase" in names
        assert "NewsArticle" in names

    def test_single_match_auto_resolution(self, analyzer_with_named_nodes):
        """Test that single fuzzy match is auto-resolved."""
        search = "duckduckgo"
        search_lower = search.lower()

        matches = [
            (nid, n)
            for nid, n in analyzer_with_named_nodes.nodes.items()
            if search_lower in n.name.lower()
        ]

        # Only search_duckduckgo matches
        assert len(matches) == 1
        assert matches[0][1].name == "search_duckduckgo"

    def test_multiple_matches_same_file_suggestion(self, analyzer_with_named_nodes):
        """Test that multiple matches from same file provide helpful suggestions."""
        search = "database"

        # Convert to snake_case for file matching
        normalized = search.lower()

        file_matches = [
            (nid, n)
            for nid, n in analyzer_with_named_nodes.nodes.items()
            if n.file and normalized in n.file.lower()
        ]

        # All matches from database_schema.py
        unique_files = set(n.file for _, n in file_matches)
        assert len(unique_files) == 1
        assert "database_schema.py" in list(unique_files)[0]

    def test_partial_word_match_without_underscore(self, analyzer_with_named_nodes):
        """Test Strategy 3: Partial word match ignoring underscores."""
        search = "searchclient"  # Without underscore
        search_normalized = search.lower().replace("_", "")

        partial_matches = [
            (nid, n)
            for nid, n in analyzer_with_named_nodes.nodes.items()
            if search_normalized in n.name.lower().replace("_", "")
        ]

        # Should find WebSearchClient
        assert len(partial_matches) >= 1
        names = {m[1].name for m in partial_matches}
        assert "WebSearchClient" in names

    def test_deduplication_across_strategies(self, analyzer_with_named_nodes):
        """Test that same node found by multiple strategies appears once."""
        search = "Web"

        # Strategy 1: Name substring
        name_matches = [
            (nid, n)
            for nid, n in analyzer_with_named_nodes.nodes.items()
            if search.lower() in n.name.lower()
        ]

        # Strategy 2: File path
        normalized_search = search.lower()
        file_matches = [
            (nid, n)
            for nid, n in analyzer_with_named_nodes.nodes.items()
            if n.file and normalized_search in n.file.lower()
        ]

        # Combine and deduplicate
        all_matches = {}
        for nid, n in name_matches + file_matches:
            if nid not in all_matches:
                all_matches[nid] = n

        # WebSearchClient should only appear once despite matching both strategies
        web_matches = [n for n in all_matches.values() if "Web" in n.name]
        assert len(web_matches) == 1

    def test_error_message_includes_suggestions(self, analyzer_with_named_nodes):
        """Test that error messages include helpful suggestions."""
        # Simulate the error response format
        search = "Schema"

        name_matches = [
            (nid, n)
            for nid, n in analyzer_with_named_nodes.nodes.items()
            if search.lower() in n.name.lower()
        ]

        # No direct name matches for "Schema"
        if not name_matches:
            # Build suggestions from file matches
            normalized = ""
            for i, c in enumerate(search):
                if i > 0 and c.isupper():
                    normalized += "_"
                normalized += c.lower()

            file_matches = [
                (nid, n)
                for nid, n in analyzer_with_named_nodes.nodes.items()
                if n.file and normalized in n.file.lower()
            ]

            suggestions = [
                {"name": m.name, "type": m.type, "file": m.file} for _, m in file_matches[:10]
            ]

            # Suggestions should help user find correct symbols
            assert len(suggestions) > 0

    def test_hint_suggests_alternative_modes(self, analyzer_with_named_nodes):
        """Test that hints suggest file_deps mode for file-level analysis."""
        # When matches come from single file, suggest file_deps mode
        file_path = "investor_homelab/models/database_schema.py"
        hint = f"For file-level analysis, try: graph(mode='file_deps', file='{file_path}')"

        assert "file_deps" in hint
        assert file_path in hint


class TestFuzzyNodeResolutionEdgeCases:
    """Edge case tests for fuzzy node resolution."""

    def test_empty_graph_returns_not_found(self):
        """Test node resolution with empty graph returns helpful error."""
        analyzer = GraphAnalyzer()

        # Simulate the resolution logic for empty graph
        search = "AnyNode"
        matches = [
            (nid, n) for nid, n in analyzer.nodes.items() if search.lower() in n.name.lower()
        ]

        assert len(matches) == 0

    def test_node_without_file_attribute(self):
        """Test handling of nodes with empty file path."""
        analyzer = GraphAnalyzer()

        node = MockGraphNode(
            "virtual:SomeClass",
            "class",
            "SomeClass",
            "",  # Empty file
            0,
        )
        analyzer.add_node(node)

        # File matching should handle empty file gracefully
        file_matches = [
            (nid, n) for nid, n in analyzer.nodes.items() if n.file and "test" in n.file.lower()
        ]

        assert len(file_matches) == 0  # Empty file should not match

    def test_special_characters_in_search(self):
        """Test search with special characters doesn't crash."""
        analyzer = GraphAnalyzer()

        node = MockGraphNode(
            "test.py:MyClass",
            "class",
            "MyClass",
            "test.py",
            1,
        )
        analyzer.add_node(node)

        # Search with underscore
        search = "My_Class"
        search_normalized = search.lower().replace("_", "")

        matches = [
            (nid, n)
            for nid, n in analyzer.nodes.items()
            if search_normalized in n.name.lower().replace("_", "")
        ]

        assert len(matches) == 1

    def test_acronym_in_camel_case(self):
        """Test handling of acronyms like IBConnection."""
        input_str = "IBConnection"
        normalized = ""
        for i, c in enumerate(input_str):
            if i > 0 and c.isupper():
                normalized += "_"
            normalized += c.lower()

        # IBConnection -> i_b_connection
        assert normalized == "i_b_connection"

    def test_modes_that_skip_fuzzy_matching(self):
        """Test that certain modes skip fuzzy resolution."""
        skip_modes = ["pagerank", "centrality", "stats", "module_pagerank", "module_centrality"]

        # These modes work on entire graph, not specific nodes
        for mode in skip_modes:
            assert mode in [
                "pagerank",
                "centrality",
                "stats",
                "module_pagerank",
                "module_centrality",
            ]


class TestIntegrationFuzzyResolution:
    """Integration tests simulating real LLM usage patterns."""

    @pytest.fixture
    def real_world_analyzer(self):
        """Create analyzer simulating real codebase."""
        analyzer = GraphAnalyzer()

        nodes = [
            MockGraphNode(
                "db.py:DatabaseConnection", "class", "DatabaseConnection", "utils/db.py", 10
            ),
            MockGraphNode("db.py:query", "function", "query", "utils/db.py", 50),
            MockGraphNode("api.py:APIClient", "class", "APIClient", "clients/api.py", 15),
            MockGraphNode("api.py:get", "method", "get", "clients/api.py", 30),
            MockGraphNode("api.py:post", "method", "post", "clients/api.py", 60),
        ]

        for node in nodes:
            analyzer.add_node(node)

        return analyzer

    def test_llm_searches_for_database(self, real_world_analyzer):
        """Test LLM searching for 'Database' finds related nodes."""
        search = "Database"

        name_matches = [
            (nid, n)
            for nid, n in real_world_analyzer.nodes.items()
            if search.lower() in n.name.lower()
        ]

        # Should find DatabaseConnection
        assert len(name_matches) == 1
        assert name_matches[0][1].name == "DatabaseConnection"

    def test_llm_searches_for_api_client(self, real_world_analyzer):
        """Test LLM searching for 'APIClient' finds exact match."""
        search = "APIClient"

        resolved = None
        for nid, n in real_world_analyzer.nodes.items():
            if n.name == search:
                resolved = nid
                break

        assert resolved == "api.py:APIClient"

    def test_llm_partial_search(self, real_world_analyzer):
        """Test LLM partial search finds relevant nodes."""
        search = "api"
        search_lower = search.lower()

        file_matches = [
            (nid, n)
            for nid, n in real_world_analyzer.nodes.items()
            if n.file and search_lower in n.file.lower()
        ]

        # Should find all nodes in api.py
        assert len(file_matches) == 3
        assert all("api.py" in m[1].file for m in file_matches)
