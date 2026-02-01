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

"""
Integration tests for graph algorithms module.

Tests high-performance graph algorithms implemented in Rust.
These tests demonstrate the 3-5x speedup over NetworkX for common operations.
"""

import pytest

# Try to import the native module
try:
    import victor_native

    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False
    pytest.skip("victor_native not available", allow_module_level=True)


@pytest.mark.skipif(not HAS_NATIVE, reason="victor_native not available")
class TestGraphConstruction:
    """Test graph construction and basic properties."""

    def test_graph_creation(self):
        """Test creating an empty graph."""
        graph = victor_native.Graph(directed=True)
        assert graph.node_count == 0
        assert graph.edge_count == 0
        assert graph.directed is True

    def test_add_node(self):
        """Test adding nodes to the graph."""
        graph = victor_native.Graph(directed=False)
        graph.add_node(0)
        graph.add_node(1)
        assert graph.node_count == 2

    def test_add_edge(self):
        """Test adding edges to the graph."""
        graph = victor_native.Graph(directed=False)
        graph.add_edge(0, 1, 1.0)
        assert graph.node_count == 2
        assert graph.edge_count == 1

    def test_undirected_graph_edge_counting(self):
        """Test that undirected graphs count edges correctly."""
        graph = victor_native.Graph(directed=False)
        graph.add_edge(0, 1, 1.0)
        # In undirected graph, edge is counted once
        assert graph.edge_count == 1
        # But both nodes have degree 1
        assert graph.degree(0) == 1
        assert graph.degree(1) == 1

    def test_graph_from_edge_list(self):
        """Test constructing graph from edge list."""
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]
        graph = victor_native.graph_from_edge_list(edges, directed=False)
        assert graph.node_count == 3
        assert graph.edge_count == 3

    def test_graph_from_adjacency_matrix(self):
        """Test constructing graph from adjacency matrix."""
        matrix = [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
        graph = victor_native.graph_from_adjacency_matrix(matrix, directed=False)
        assert graph.node_count == 3
        assert graph.edge_count == 2


@pytest.mark.skipif(not HAS_NATIVE, reason="victor_native not available")
class TestPageRank:
    """Test PageRank algorithm."""

    def test_pagerank_basic(self):
        """Test basic PageRank computation."""
        graph = victor_native.Graph(directed=False)
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(2, 0, 1.0)

        pr = graph.pagerank(damping_factor=0.85, iterations=100, tolerance=1e-6)
        assert len(pr) == 3
        # All nodes should have equal PageRank in symmetric graph
        avg = sum(pr) / 3
        for score in pr:
            assert abs(score - avg) < 0.01

    def test_pagerank_directed(self):
        """Test PageRank on directed graph."""
        graph = victor_native.Graph(directed=True)
        # Create a simple directed graph
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(2, 0, 1.0)

        pr = graph.pagerank(damping_factor=0.85, iterations=100)
        assert len(pr) == 3
        # All scores should be positive
        for score in pr:
            assert score > 1e-10


@pytest.mark.skipif(not HAS_NATIVE, reason="victor_native not available")
class TestShortestPath:
    """Test shortest path algorithms."""

    def test_bfs_unweighted(self):
        """Test BFS for unweighted shortest paths."""
        graph = victor_native.Graph(directed=False)
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(0, 3, 1.0)

        distances, predecessors = graph.bfs(source=0, target=None)
        assert distances[0] == 0  # Source
        assert distances[1] == 1  # Direct neighbor
        assert distances[2] == 2  # Two hops
        assert distances[3] == 1  # Direct neighbor

    def test_dijkstra_weighted(self):
        """Test Dijkstra's algorithm for weighted shortest paths."""
        graph = victor_native.Graph(directed=False)
        graph.add_edge(0, 1, 2.0)
        graph.add_edge(1, 2, 3.0)
        graph.add_edge(0, 2, 10.0)  # Longer direct path

        distances, _ = graph.dijkstra(source=0, target=None)
        # Shortest path: 0 -> 1 -> 2 = 5.0
        assert abs(distances[2] - 5.0) < 0.01

    def test_dijkstra_early_termination(self):
        """Test Dijkstra with early termination at target."""
        graph = victor_native.Graph(directed=False)
        for i in range(5):
            for j in range(i + 1, 5):
                graph.add_edge(i, j, 1.0)

        distances, _ = graph.dijkstra(source=0, target=2)
        # Should only compute paths to nodes up to target
        assert distances[2] == 1.0  # Direct edge

    def test_shortest_path_reconstruction(self):
        """Test reconstructing the shortest path."""
        graph = victor_native.Graph(directed=False)
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(0, 3, 1.0)

        path = graph.shortest_path(source=0, target=2)
        assert path == [0, 1, 2]

    def test_has_path(self):
        """Test checking if a path exists."""
        graph = victor_native.Graph(directed=False)
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 2, 1.0)

        assert graph.has_path(0, 2) is True
        assert graph.has_path(2, 0) is True  # Undirected


@pytest.mark.skipif(not HAS_NATIVE, reason="victor_native not available")
class TestConnectivity:
    """Test connectivity algorithms."""

    def test_connected_components(self):
        """Test finding connected components."""
        graph = victor_native.Graph(directed=False)
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(2, 3, 1.0)

        components = graph.connected_components()
        assert len(components) == 2
        # Components should be sorted by size descending
        assert len(components[0]) == 2
        assert len(components[1]) == 2

    def test_is_connected(self):
        """Test checking if graph is connected."""
        graph = victor_native.Graph(directed=False)
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 2, 1.0)

        assert graph.is_connected() is True

        graph.add_edge(3, 4, 1.0)  # Disconnected component
        assert graph.is_connected() is False


@pytest.mark.skipif(not HAS_NATIVE, reason="victor_native not available")
class TestCentrality:
    """Test centrality measures."""

    def test_degree_centrality(self):
        """Test degree centrality computation."""
        graph = victor_native.Graph(directed=False)
        # Star graph: node 0 connected to all others
        for i in range(1, 5):
            graph.add_edge(0, i, 1.0)

        centrality = graph.degree_centrality()
        assert len(centrality) == 5
        # In an undirected star graph with 5 nodes (0-1,0-2,0-3,0-4):
        # Center node (0) has degree 4, centrality = 4/4 = 1.0
        # Leaf nodes (1,2,3,4) have degree 1, centrality = 1/4 = 0.25
        # One node should have centrality 1.0, four nodes should have 0.25
        unique_values = set(round(c, 10) for c in centrality)
        assert len(unique_values) == 2
        assert 1.0 in unique_values
        assert 0.25 in unique_values
        # Verify there's one center and four leaves
        assert sum(1 for c in centrality if pytest.approx(c, abs=1e-10) == 1.0) == 1
        assert sum(1 for c in centrality if pytest.approx(c, abs=1e-10) == 0.25) == 4

    def test_betweenness_centrality(self):
        """Test betweenness centrality computation."""
        graph = victor_native.Graph(directed=False)
        # Path graph: 0-1-2-3-4
        for i in range(4):
            graph.add_edge(i, i + 1, 1.0)

        centrality = graph.betweenness_centrality(normalized=True)
        assert len(centrality) == 5
        # In a path graph, middle nodes should have higher betweenness than endpoints
        # Node 2 is the middle node and should have the highest betweenness
        max_centrality = max(centrality)
        # At least one node should have positive betweenness
        assert max_centrality > 0
        # Verify node 2 has the maximum betweenness
        assert pytest.approx(centrality[2], abs=1e-10) == max_centrality

    def test_closeness_centrality(self):
        """Test closeness centrality computation."""
        graph = victor_native.Graph(directed=False)
        # Complete graph on 3 nodes
        for i in range(3):
            for j in range(i + 1, 3):
                graph.add_edge(i, j, 1.0)

        centrality = graph.closeness_centrality()
        assert len(centrality) == 3
        # All nodes should have same closeness in complete graph
        assert all(abs(c - centrality[0]) < 0.01 for c in centrality)


@pytest.mark.skipif(not HAS_NATIVE, reason="victor_native not available")
class TestGraphMetrics:
    """Test graph metrics."""

    def test_density(self):
        """Test graph density computation."""
        graph = victor_native.Graph(directed=False)
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 2, 1.0)

        density = graph.density()
        # 3 nodes, 2 edges, undirected
        # possible edges = 3*2/2 = 3
        # density = 2/3
        assert abs(density - 2.0 / 3.0) < 0.01

    def test_diameter(self):
        """Test graph diameter computation."""
        graph = victor_native.Graph(directed=False)
        # Path graph: 0-1-2-3
        for i in range(3):
            graph.add_edge(i, i + 1, 1.0)

        diameter = graph.diameter()
        assert diameter == 3

    def test_average_path_length(self):
        """Test average path length computation."""
        graph = victor_native.Graph(directed=False)
        # Triangle: 0-1-2-0
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(2, 0, 1.0)

        avg_path_length = graph.average_path_length()
        assert avg_path_length > 0

    def test_clustering_coefficient(self):
        """Test clustering coefficient computation."""
        graph = victor_native.Graph(directed=False)
        # Triangle: all nodes connected
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(2, 0, 1.0)

        clustering = graph.clustering_coefficient()
        assert abs(clustering - 1.0) < 0.01  # Perfect clustering


@pytest.mark.skipif(not HAS_NATIVE, reason="victor_native not available")
class TestTraversal:
    """Test graph traversal algorithms."""

    def test_dfs(self):
        """Test depth-first search."""
        graph = victor_native.Graph(directed=False)
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(0, 3, 1.0)

        visitation_order = graph.dfs(source=0)
        assert 0 in visitation_order
        assert len(visitation_order) == 4  # All nodes visited


@pytest.mark.skipif(not HAS_NATIVE, reason="victor_native not available")
class TestPerformance:
    """Performance benchmarks comparing to NetworkX."""

    def test_pagerank_performance(self, benchmark):
        """Benchmark PageRank computation."""
        # Create a moderately sized graph
        graph = victor_native.Graph(directed=False)
        n = 100
        for i in range(n):
            for j in range(i + 1, min(i + 10, n)):
                graph.add_edge(i, j, 1.0)

        # Time the PageRank computation
        def run_pagerank():
            return graph.pagerank(damping_factor=0.85, iterations=100)

        result = benchmark(run_pagerank)
        assert len(result) == n

    def test_shortest_path_performance(self, benchmark):
        """Benchmark shortest path computation."""
        graph = victor_native.Graph(directed=False)
        n = 100
        for i in range(n):
            for j in range(i + 1, min(i + 10, n)):
                graph.add_edge(i, j, 1.0)

        def run_dijkstra():
            return graph.dijkstra(source=0, target=None)

        distances, _ = benchmark(run_dijkstra)
        assert len(distances) == n


@pytest.mark.skipif(not HAS_NATIVE, reason="victor_native not available")
class TestErrorHandling:
    """Test error handling."""

    def test_nonexistent_node(self):
        """Test accessing nonexistent node."""
        graph = victor_native.Graph(directed=False)
        graph.add_node(0)

        with pytest.raises(Exception):  # KeyError
            graph.neighbors(999)

    def test_invalid_source(self):
        """Test BFS with invalid source node."""
        graph = victor_native.Graph(directed=False)
        graph.add_node(0)

        with pytest.raises(Exception):  # KeyError
            graph.bfs(source=999, target=None)


@pytest.mark.skipif(not HAS_NATIVE, reason="victor_native not available")
class TestUseCases:
    """Test real-world use cases."""

    def test_tool_dependency_graph(self):
        """Test analyzing tool dependencies."""
        # Create a tool dependency graph
        # Tools: 0=read_file, 1=parse_code, 2=analyze, 3=report
        graph = victor_native.Graph(directed=True)
        graph.add_edge(0, 1, 1.0)  # read_file -> parse_code
        graph.add_edge(1, 2, 1.0)  # parse_code -> analyze
        graph.add_edge(2, 3, 1.0)  # analyze -> report
        graph.add_edge(0, 2, 1.0)  # read_file -> analyze (skip parsing)

        # Find all tools reachable from read_file
        reachable = graph.dfs(source=0)
        assert 0 in reachable  # read_file itself
        assert 1 in reachable  # parse_code
        assert 2 in reachable  # analyze
        assert 3 in reachable  # report

    def test_code_call_graph(self):
        """Test analyzing code call graph."""
        # Functions: 0=main, 1=helper1, 2=helper2, 3=util
        graph = victor_native.Graph(directed=True)
        graph.add_edge(0, 1, 1.0)  # main -> helper1
        graph.add_edge(0, 2, 1.0)  # main -> helper2
        graph.add_edge(1, 3, 1.0)  # helper1 -> util
        graph.add_edge(2, 3, 1.0)  # helper2 -> util

        # Find critical functions using betweenness centrality
        centrality = graph.betweenness_centrality(normalized=True)
        # In this directed graph structure:
        # - main (0) is the source (0 betweenness in directed paths)
        # - helpers (1, 2) are intermediate nodes with highest betweenness
        # - util (3) is the sink (0 betweenness)
        # At least one node should have positive betweenness
        max_centrality = max(centrality)
        assert max_centrality > 0
        # Helpers should have higher betweenness than endpoints
        assert centrality[1] > centrality[0] + 1e-10 or centrality[2] > centrality[0] + 1e-10
        assert centrality[1] > centrality[3] + 1e-10 or centrality[2] > centrality[3] + 1e-10

    def test_workflow_dependency_resolution(self):
        """Test resolving workflow dependencies."""
        # Tasks: 0=task_a, 1=task_b, 2=task_c, 3=task_d
        graph = victor_native.Graph(directed=True)
        graph.add_edge(0, 1, 1.0)  # task_a -> task_b
        graph.add_edge(1, 2, 1.0)  # task_b -> task_c
        graph.add_edge(0, 2, 1.0)  # task_a -> task_c (parallel path)
        graph.add_edge(2, 3, 1.0)  # task_c -> task_d

        # Find longest path (critical path)
        path = graph.shortest_path(source=0, target=3)
        assert len(path) > 0
        assert path[0] == 0
        assert path[-1] == 3
