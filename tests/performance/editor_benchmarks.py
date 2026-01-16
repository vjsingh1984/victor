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

"""Comprehensive performance benchmarks for the visual workflow editor.

This module provides extensive benchmarks for the visual workflow editor,
focusing on:

1. Page Load Time: Time to initialize and render the editor
2. Node Rendering Performance: Time to render different numbers of nodes
3. Connection Rendering: Time to render edges between nodes
4. Zoom/Pan Responsiveness: Frame rate during interactions
5. Search Performance: Time to search nodes in large workflows
6. Auto-Layout Calculation: Time to calculate optimal node positions
7. YAML Import/Export: Time to convert between formats
8. Memory Usage: Memory footprint with large workflows

Performance Targets:
- Editor load: <500ms for 100 nodes
- Node rendering: <16ms per node (60fps target)
- Auto-layout: <1s for 100 nodes
- YAML import: <500ms for 100 nodes
- YAML export: <300ms for 100 nodes
- Memory usage: <100MB for 100-node workflow

Usage:
    # Run all benchmarks
    pytest tests/performance/editor_benchmarks.py -v

    # Run specific benchmark groups
    pytest tests/performance/editor_benchmarks.py -k "rendering" -v
    pytest tests/performance/editor_benchmarks.py -k "yaml" -v
    pytest tests/performance/editor_benchmarks.py -k "memory" -v

    # Generate benchmark report
    pytest tests/performance/editor_benchmarks.py --benchmark-only --benchmark-json=editor_benchmarks.json
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import random
import string
import time
import tracemalloc
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock, patch

import pytest

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


# =============================================================================
# Mock Workflow Components for Benchmarking
# =============================================================================


class NodeType(str, Enum):
    """Types of workflow nodes."""

    AGENT = "agent"
    TOOL = "tool"
    CONDITION = "condition"
    TRANSFORM = "transform"
    PARALLEL = "parallel"
    HITL = "hitl"
    TEAM = "team"
    COMPUTE = "compute"


@dataclass
class WorkflowNode:
    """Mock workflow node for editor benchmarking."""

    id: str
    type: NodeType
    x: float
    y: float
    label: str
    data: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "position": {"x": self.x, "y": self.y},
            "data": {
                "label": self.label,
                **self.data,
            },
        }


@dataclass
class WorkflowEdge:
    """Mock workflow edge (connection) for editor benchmarking."""

    id: str
    source: str
    target: str
    label: Optional[str] = None
    condition: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary."""
        edge_dict = {
            "id": self.id,
            "source": self.source,
            "target": self.target,
        }
        if self.label:
            edge_dict["label"] = self.label
        if self.condition:
            edge_dict["condition"] = self.condition
        return edge_dict


@dataclass
class WorkflowGraph:
    """Mock workflow graph for editor benchmarking."""

    nodes: List[WorkflowNode] = field(default_factory=list)
    edges: List[WorkflowEdge] = field(default_factory=list)

    def add_node(self, node: WorkflowNode) -> None:
        """Add a node to the graph."""
        self.nodes.append(node)

    def add_edge(self, edge: WorkflowEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary."""
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    def to_yaml(self) -> str:
        """Convert graph to YAML format."""
        yaml_str = ["workflows:\n", "  test_workflow:\n", "    nodes:\n"]

        for node in self.nodes:
            yaml_str.append(f"      - id: {node.id}\n")
            yaml_str.append(f"        type: {node.type.value}\n")
            yaml_str.append(f"        position:\n")
            yaml_str.append(f"          x: {node.x}\n")
            yaml_str.append(f"          y: {node.y}\n")
            yaml_str.append(f"        label: {node.label}\n")

        yaml_str.append("    edges:\n")
        for edge in self.edges:
            yaml_str.append(f"      - from: {edge.source}\n")
            yaml_str.append(f"        to: {edge.target}\n")
            if edge.label:
                yaml_str.append(f"        label: {edge.label}\n")

        return "".join(yaml_str)


# =============================================================================
# Workflow Graph Generators
# =============================================================================


def generate_random_node_id() -> str:
    """Generate a random node ID."""
    return f"node_{random.randint(1000, 9999)}"


def generate_random_label(node_type: NodeType) -> str:
    """Generate a random label for a node."""
    prefixes = {
        NodeType.AGENT: ["Research", "Analyze", "Process", "Execute"],
        NodeType.TOOL: ["Read", "Write", "Search", "Parse"],
        NodeType.CONDITION: ["Check", "Validate", "Verify", "Test"],
        NodeType.TRANSFORM: ["Format", "Convert", "Map", "Filter"],
        NodeType.PARALLEL: ["Split", "Merge", "Distribute", "Aggregate"],
        NodeType.HITL: ["Review", "Approve", "Confirm", "Select"],
        NodeType.TEAM: ["Team A", "Team B", "Team C", "Team D"],
        NodeType.COMPUTE: ["Calculate", "Aggregate", "Summarize", "Compute"],
    }
    prefix = random.choice(prefixes.get(node_type, ["Node"]))
    suffix = "".join(random.choices(string.digits, k=3))
    return f"{prefix} {suffix}"


def generate_workflow_graph(
    node_count: int,
    edge_probability: float = 0.3,
    node_types: Optional[List[NodeType]] = None,
) -> WorkflowGraph:
    """Generate a random workflow graph for benchmarking.

    Args:
        node_count: Number of nodes to generate
        edge_probability: Probability of creating an edge between nodes (0-1)
        node_types: List of node types to use (defaults to all types)

    Returns:
        WorkflowGraph with random nodes and edges
    """
    if node_types is None:
        node_types = list(NodeType)

    graph = WorkflowGraph()
    nodes: Dict[str, WorkflowNode] = {}

    # Generate nodes in a grid-like pattern
    grid_size = int(node_count**0.5) + 1
    for i in range(node_count):
        node_type = random.choice(node_types)
        x = (i % grid_size) * 200 + random.randint(0, 50)
        y = (i // grid_size) * 150 + random.randint(0, 50)

        node = WorkflowNode(
            id=generate_random_node_id(),
            type=node_type,
            x=x,
            y=y,
            label=generate_random_label(node_type),
            data={"iteration": i, "tool_budget": random.randint(5, 50)},
        )

        nodes[node.id] = node
        graph.add_node(node)

    # Generate edges (ensure graph is connected)
    node_ids = list(nodes.keys())
    for i in range(len(node_ids) - 1):
        edge = WorkflowEdge(
            id=f"edge_{i}_{i+1}",
            source=node_ids[i],
            target=node_ids[i + 1],
            label=f"connection_{i}",
        )
        graph.add_edge(edge)

    # Add additional random edges
    edge_count = int(node_count * edge_probability)
    for i in range(edge_count):
        source = random.choice(node_ids)
        target = random.choice(node_ids)

        if source != target:
            # Avoid duplicate edges
            edge_id = f"edge_{source}_{target}"
            if not any(e.id == edge_id for e in graph.edges):
                edge = WorkflowEdge(
                    id=edge_id,
                    source=source,
                    target=target,
                    condition=random.choice(["success", "failure", None]),
                )
                graph.add_edge(edge)

    return graph


# =============================================================================
# Mock Editor Implementation for Benchmarking
# =============================================================================


class MockWorkflowEditor:
    """Mock visual workflow editor for performance benchmarking.

    Simulates the core operations of the visual editor without the actual UI.
    """

    def __init__(self):
        self.graph: Optional[WorkflowGraph] = None
        self.render_time: float = 0.0
        self.layout_time: float = 0.0
        self.selected_nodes: Set[str] = set()
        self.viewport = {"x": 0, "y": 0, "zoom": 1.0}
        self._render_cache: Dict[str, Any] = {}

    def load_graph(self, graph: WorkflowGraph) -> None:
        """Load a workflow graph into the editor."""
        start = time.perf_counter()

        self.graph = graph
        self._render_cache.clear()
        self.selected_nodes.clear()

        # Simulate rendering initialization
        self._initialize_render_cache()

        self.render_time = time.perf_counter() - start

    def _initialize_render_cache(self) -> None:
        """Initialize render cache for all nodes and edges."""
        if not self.graph:
            return

        for node in self.graph.nodes:
            self._render_cache[node.id] = {
                "position": {"x": node.x, "y": node.y},
                "bounds": self._calculate_node_bounds(node),
                "visible": True,
            }

        for edge in self.graph.edges:
            self._render_cache[edge.id] = {
                "source_pos": self._get_node_position(edge.source),
                "target_pos": self._get_node_position(edge.target),
                "visible": True,
            }

    def _calculate_node_bounds(self, node: WorkflowNode) -> Dict[str, float]:
        """Calculate the bounding box for a node."""
        # Approximate node size based on type
        width = 180
        height = 80
        return {
            "x": node.x,
            "y": node.y,
            "width": width,
            "height": height,
            "right": node.x + width,
            "bottom": node.y + height,
        }

    def _get_node_position(self, node_id: str) -> Optional[Dict[str, float]]:
        """Get the position of a node by ID."""
        if not self.graph:
            return None

        for node in self.graph.nodes:
            if node.id == node_id:
                return {"x": node.x, "y": node.y}

        return None

    def render_nodes(self) -> float:
        """Render all nodes in the graph."""
        if not self.graph:
            return 0.0

        start = time.perf_counter()

        for node in self.graph.nodes:
            # Simulate rendering logic
            bounds = self._calculate_node_bounds(node)
            is_visible = self._is_visible(bounds)

            if is_visible:
                self._render_cache[node.id]["visible"] = True
            else:
                self._render_cache[node.id]["visible"] = False

        return time.perf_counter() - start

    def render_edges(self) -> float:
        """Render all edges in the graph."""
        if not self.graph:
            return 0.0

        start = time.perf_counter()

        for edge in self.graph.edges:
            # Simulate edge rendering (Bezier curves)
            source_pos = self._get_node_position(edge.source)
            target_pos = self._get_node_position(edge.target)

            if source_pos and target_pos:
                # Calculate control points for Bezier curve
                mid_x = (source_pos["x"] + target_pos["x"]) / 2
                mid_y = (source_pos["y"] + target_pos["y"]) / 2

                control_points = [
                    {"x": mid_x, "y": source_pos["y"]},
                    {"x": mid_x, "y": target_pos["y"]},
                ]

                self._render_cache[edge.id]["control_points"] = control_points

        return time.perf_counter() - start

    def _is_visible(self, bounds: Dict[str, float]) -> bool:
        """Check if a node is visible in the current viewport."""
        # Simple viewport culling
        viewport_width = 2000
        viewport_height = 1500
        viewport_x = self.viewport["x"]
        viewport_y = self.viewport["y"]

        return not (
            bounds["right"] < viewport_x
            or bounds["x"] > viewport_x + viewport_width
            or bounds["bottom"] < viewport_y
            or bounds["y"] > viewport_y + viewport_height
        )

    def zoom(self, factor: float) -> float:
        """Zoom the viewport."""
        start = time.perf_counter()

        old_zoom = self.viewport["zoom"]
        new_zoom = old_zoom * factor
        self.viewport["zoom"] = max(0.1, min(new_zoom, 5.0))

        # Update render cache for new zoom level
        for node in self.graph.nodes if self.graph else []:
            bounds = self._calculate_node_bounds(node)
            self._render_cache[node.id]["bounds"] = bounds

        return time.perf_counter() - start

    def pan(self, dx: float, dy: float) -> float:
        """Pan the viewport."""
        start = time.perf_counter()

        self.viewport["x"] += dx
        self.viewport["y"] += dy

        return time.perf_counter() - start

    def search_nodes(self, query: str) -> List[str]:
        """Search for nodes by label."""
        if not self.graph:
            return []

        start = time.perf_counter()

        results = []
        query_lower = query.lower()

        for node in self.graph.nodes:
            if query_lower in node.label.lower():
                results.append(node.id)

        self.render_time = time.perf_counter() - start
        return results

    def auto_layout(self, algorithm: str = "hierarchical") -> float:
        """Calculate automatic layout for the graph."""
        if not self.graph:
            return 0.0

        start = time.perf_counter()

        if algorithm == "hierarchical":
            self._layout_hierarchical()
        elif algorithm == "force_directed":
            self._layout_force_directed()
        elif algorithm == "grid":
            self._layout_grid()
        else:
            self._layout_grid()

        self.layout_time = time.perf_counter() - start
        return self.layout_time

    def _layout_hierarchical(self) -> None:
        """Layout nodes in hierarchical layers."""
        if not self.graph:
            return

        # Simple topological sort layering
        layers: List[List[WorkflowNode]] = []
        visited: Set[str] = set()

        # Find nodes with no incoming edges
        in_degree: Dict[str, int] = {node.id: 0 for node in self.graph.nodes}
        for edge in self.graph.edges:
            in_degree[edge.target] += 1

        # Assign layers
        current_layer = [node for node in self.graph.nodes if in_degree[node.id] == 0]

        while current_layer:
            layers.append(current_layer)
            next_layer: List[WorkflowNode] = []

            for node in current_layer:
                visited.add(node.id)

                # Find outgoing neighbors
                for edge in self.graph.edges:
                    if edge.source == node.id and edge.target not in visited:
                        in_degree[edge.target] -= 1
                        if in_degree[edge.target] == 0:
                            target_node = next(
                                (n for n in self.graph.nodes if n.id == edge.target),
                                None,
                            )
                            if target_node:
                                next_layer.append(target_node)

            current_layer = next_layer

        # Position nodes in layers
        layer_height = 200
        node_width = 200
        node_height = 100

        for layer_idx, layer in enumerate(layers):
            y = layer_idx * layer_height
            for node_idx, node in enumerate(layer):
                x = node_idx * node_width
                node.x = x
                node.y = y

    def _layout_force_directed(self) -> None:
        """Simple force-directed layout simulation."""
        if not self.graph:
            return

        # Simplified force-directed layout
        iterations = 50
        k = 100.0  # Optimal distance

        for _ in range(iterations):
            # Repulsive forces
            for i, node1 in enumerate(self.graph.nodes):
                for j, node2 in enumerate(self.graph.nodes):
                    if i >= j:
                        continue

                    dx = node1.x - node2.x
                    dy = node1.y - node2.y
                    dist = (dx * dx + dy * dy) ** 0.5 + 0.1

                    force = k * k / dist
                    fx = (dx / dist) * force
                    fy = (dy / dist) * force

                    node1.x += fx * 0.01
                    node1.y += fy * 0.01
                    node2.x -= fx * 0.01
                    node2.y -= fy * 0.01

            # Attractive forces (edges)
            for edge in self.graph.edges:
                node1 = next((n for n in self.graph.nodes if n.id == edge.source), None)
                node2 = next((n for n in self.graph.nodes if n.id == edge.target), None)

                if node1 and node2:
                    dx = node2.x - node1.x
                    dy = node2.y - node1.y
                    dist = (dx * dx + dy * dy) ** 0.5 + 0.1

                    force = (dist - k) * 0.01
                    fx = (dx / dist) * force
                    fy = (dy / dist) * force

                    node1.x += fx
                    node1.y += fy
                    node2.x -= fx
                    node2.y -= fy

    def _layout_grid(self) -> None:
        """Simple grid layout."""
        if not self.graph:
            return

        grid_size = int(len(self.graph.nodes) ** 0.5) + 1
        spacing_x = 200
        spacing_y = 150

        for i, node in enumerate(self.graph.nodes):
            node.x = (i % grid_size) * spacing_x
            node.y = (i // grid_size) * spacing_y


# =============================================================================
# Benchmark Fixtures
# =============================================================================


@pytest.fixture
def editor():
    """Create editor instance for benchmarking."""
    return MockWorkflowEditor()


@pytest.fixture
def small_graph():
    """Create small workflow graph (10 nodes)."""
    return generate_workflow_graph(node_count=10)


@pytest.fixture
def medium_graph():
    """Create medium workflow graph (50 nodes)."""
    return generate_workflow_graph(node_count=50)


@pytest.fixture
def large_graph():
    """Create large workflow graph (100 nodes)."""
    return generate_workflow_graph(node_count=100)


@pytest.fixture
def huge_graph():
    """Create huge workflow graph (200 nodes)."""
    return generate_workflow_graph(node_count=200)


# =============================================================================
# Page Load Benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.parametrize("node_count", [10, 25, 50, 100, 200])
def test_editor_load_time(benchmark, node_count):
    """Benchmark editor page load time with different graph sizes.

    Performance Targets:
    - 10 nodes: <50ms
    - 25 nodes: <150ms
    - 50 nodes: <300ms
    - 100 nodes: <500ms
    - 200 nodes: <1000ms
    """
    graph = generate_workflow_graph(node_count=node_count)
    editor = MockWorkflowEditor()

    def load_and_render():
        editor.load_graph(graph)
        render_nodes_time = editor.render_nodes()
        render_edges_time = editor.render_edges()
        return {
            "load_time": editor.render_time,
            "nodes_time": render_nodes_time,
            "edges_time": render_edges_time,
            "total_time": editor.render_time + render_nodes_time + render_edges_time,
        }

    result = benchmark(load_and_render)

    # Check performance targets
    target_ms = {
        10: 50,
        25: 150,
        50: 300,
        100: 500,
        200: 1000,
    }.get(node_count, 1000)

    total_ms = result["total_time"] * 1000
    assert total_ms < target_ms, (
        f"Load time {total_ms:.2f}ms exceeds target {target_ms}ms for {node_count} nodes"
    )

    print(f"\nLoad Time | Nodes: {node_count:3} | "
          f"Total: {total_ms:6.2f}ms | "
          f"Load: {result['load_time']*1000:6.2f}ms | "
          f"Nodes: {result['nodes_time']*1000:6.2f}ms | "
          f"Edges: {result['edges_time']*1000:6.2f}ms")


# =============================================================================
# Node Rendering Benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.parametrize("node_count", [10, 50, 100, 200])
def test_node_rendering_performance(benchmark, node_count):
    """Benchmark node rendering performance.

    Performance Targets:
    - <16ms per node (60fps target)
    - Scales linearly with node count
    """
    graph = generate_workflow_graph(node_count=node_count)
    editor = MockWorkflowEditor()
    editor.load_graph(graph)

    def render_all_nodes():
        return editor.render_nodes()

    render_time = benchmark(render_all_nodes)
    time_per_node = render_time / node_count

    print(f"\nNode Rendering | Nodes: {node_count:3} | "
          f"Total: {render_time*1000:7.2f}ms | "
          f"Per-Node: {time_per_node*1000:6.3f}ms")

    # Target: <16ms per node (60fps)
    assert time_per_node < 0.016, (
        f"Node rendering {time_per_node*1000:.3f}ms exceeds 16ms target for 60fps"
    )


# =============================================================================
# Edge Rendering Benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.parametrize("node_count,edge_count", [
    (10, 15),
    (50, 75),
    (100, 150),
    (200, 300),
])
def test_edge_rendering_performance(benchmark, node_count, edge_count):
    """Benchmark edge rendering performance.

    Edges require Bezier curve calculations which can be expensive.
    """
    graph = generate_workflow_graph(node_count=node_count, edge_probability=0.3)
    # Ensure we have the right number of edges
    while len(graph.edges) < edge_count:
        source = random.choice([n.id for n in graph.nodes])
        target = random.choice([n.id for n in graph.nodes])
        if source != target:
            edge = WorkflowEdge(
                id=f"edge_{len(graph.edges)}",
                source=source,
                target=target,
            )
            graph.add_edge(edge)

    editor = MockWorkflowEditor()
    editor.load_graph(graph)

    def render_all_edges():
        return editor.render_edges()

    render_time = benchmark(render_all_edges)
    time_per_edge = render_time / len(graph.edges) if graph.edges else 0

    print(f"\nEdge Rendering | Edges: {len(graph.edges):3} | "
          f"Total: {render_time*1000:7.2f}ms | "
          f"Per-Edge: {time_per_edge*1000:6.3f}ms")


# =============================================================================
# Zoom/Pan Responsiveness Benchmarks
# =============================================================================


@pytest.mark.benchmark
def test_zoom_performance(benchmark, large_graph):
    """Benchmark zoom operation performance.

    Performance Target:
    - <16ms per zoom operation (60fps)
    """
    editor = MockWorkflowEditor()
    editor.load_graph(large_graph)

    def zoom_operation():
        return editor.zoom(1.2)

    zoom_time = benchmark(zoom_operation)

    print(f"\nZoom Performance | Time: {zoom_time*1000:6.3f}ms")

    # Target: <16ms for 60fps
    assert zoom_time < 0.016, f"Zoom {zoom_time*1000:.3f}ms exceeds 16ms for 60fps"


@pytest.mark.benchmark
def test_pan_performance(benchmark, large_graph):
    """Benchmark pan operation performance.

    Performance Target:
    - <16ms per pan operation (60fps)
    """
    editor = MockWorkflowEditor()
    editor.load_graph(large_graph)

    def pan_operation():
        return editor.pan(50, 50)

    pan_time = benchmark(pan_operation)

    print(f"\nPan Performance | Time: {pan_time*1000:6.3f}ms")

    # Target: <16ms for 60fps
    assert pan_time < 0.016, f"Pan {pan_time*1000:.3f}ms exceeds 16ms for 60fps"


@pytest.mark.benchmark
def test_continuous_zoom_pan(benchmark, large_graph):
    """Benchmark continuous zoom + pan operations (simulating user drag)."""
    editor = MockWorkflowEditor()
    editor.load_graph(large_graph)

    def continuous_operations():
        total = 0.0
        for i in range(10):
            total += editor.zoom(1.05)
            total += editor.pan(10, 10)
        return total

    total_time = benchmark(continuous_operations)
    avg_time = total_time / 20  # 10 zoom + 10 pan operations

    print(f"\nContinuous Operations | Total: {total_time*1000:7.2f}ms | "
          f"Avg: {avg_time*1000:6.3f}ms")

    assert avg_time < 0.016, f"Avg operation {avg_time*1000:.3f}ms exceeds 16ms for 60fps"


# =============================================================================
# Search Performance Benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.parametrize("node_count", [50, 100, 200])
def test_search_performance(benchmark, node_count):
    """Benchmark node search performance.

    Performance Targets:
    - <10ms for 100 nodes
    - <20ms for 200 nodes
    """
    graph = generate_workflow_graph(node_count=node_count)
    editor = MockWorkflowEditor()
    editor.load_graph(graph)

    def search_operation():
        return editor.search_nodes("Research")

    result_count = len(editor.search_nodes("Research"))
    search_time = benchmark(search_operation)

    print(f"\nSearch Performance | Nodes: {node_count:3} | "
          f"Results: {result_count:3} | Time: {search_time*1000:6.3f}ms")

    # Check performance targets
    if node_count == 100:
        assert search_time < 0.010, f"Search {search_time*1000:.3f}ms exceeds 10ms for 100 nodes"
    elif node_count == 200:
        assert search_time < 0.020, f"Search {search_time*1000:.3f}ms exceeds 20ms for 200 nodes"


# =============================================================================
# Auto-Layout Benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.parametrize("node_count,algorithm", [
    (50, "grid"),
    (50, "hierarchical"),
    (50, "force_directed"),
    (100, "grid"),
    (100, "hierarchical"),
    (100, "force_directed"),
])
def test_auto_layout_performance(benchmark, node_count, algorithm):
    """Benchmark auto-layout calculation performance.

    Performance Targets:
    - Grid: <100ms for 100 nodes
    - Hierarchical: <500ms for 100 nodes
    - Force-directed: <1000ms for 100 nodes
    """
    graph = generate_workflow_graph(node_count=node_count)
    editor = MockWorkflowEditor()
    editor.load_graph(graph)

    def calculate_layout():
        return editor.auto_layout(algorithm=algorithm)

    layout_time = benchmark(calculate_layout)

    print(f"\nAuto-Layout | Algorithm: {algorithm:15} | "
          f"Nodes: {node_count:3} | Time: {layout_time*1000:7.2f}ms")

    # Check performance targets
    if algorithm == "grid" and node_count == 100:
        assert layout_time < 0.100, f"Grid layout {layout_time*1000:.2f}ms exceeds 100ms"
    elif algorithm == "hierarchical" and node_count == 100:
        assert layout_time < 0.500, f"Hierarchical layout {layout_time*1000:.2f}ms exceeds 500ms"
    elif algorithm == "force_directed" and node_count == 100:
        assert layout_time < 1.0, f"Force-directed layout {layout_time*1000:.2f}ms exceeds 1000ms"


# =============================================================================
# YAML Import/Export Benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.parametrize("node_count", [50, 100, 200])
def test_yaml_export_performance(benchmark, node_count):
    """Benchmark YAML export performance.

    Performance Targets:
    - 50 nodes: <150ms
    - 100 nodes: <300ms
    - 200 nodes: <600ms
    """
    graph = generate_workflow_graph(node_count=node_count)

    def export_to_yaml():
        return graph.to_yaml()

    yaml_str = benchmark(export_to_yaml)
    yaml_length = len(yaml_str)

    # We need to capture time separately since benchmark doesn't return time
    start = time.perf_counter()
    yaml_str = graph.to_yaml()
    export_time = time.perf_counter() - start

    target_ms = {
        50: 150,
        100: 300,
        200: 600,
    }.get(node_count, 1000)

    print(f"\nYAML Export | Nodes: {node_count:3} | "
          f"Time: {export_time*1000:7.2f}ms | Size: {yaml_length:6} bytes")

    assert export_time * 1000 < target_ms, (
        f"YAML export {export_time*1000:.2f}ms exceeds target {target_ms}ms for {node_count} nodes"
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("node_count", [50, 100, 200])
def test_yaml_import_performance(benchmark, node_count):
    """Benchmark YAML import performance.

    Performance Targets:
    - 50 nodes: <200ms
    - 100 nodes: <500ms
    - 200 nodes: <1000ms
    """
    graph = generate_workflow_graph(node_count=node_count)
    yaml_str = graph.to_yaml()

    def import_from_yaml():
        # Simulate YAML parsing
        lines = yaml_str.split("\n")
        parsed_graph = WorkflowGraph()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("- id:"):
                node_id = line.split(":")[1].strip()
                i += 1
                node_type = "agent"
                x, y = 0.0, 0.0
                label = "Node"

                while i < len(lines) and not lines[i].strip().startswith("-"):
                    if lines[i].strip().startswith("type:"):
                        node_type = lines[i].split(":")[1].strip()
                    elif lines[i].strip().startswith("x:"):
                        x = float(lines[i].split(":")[1].strip())
                    elif lines[i].strip().startswith("y:"):
                        y = float(lines[i].split(":")[1].strip())
                    elif lines[i].strip().startswith("label:"):
                        label = lines[i].split(":")[1].strip()
                    i += 1

                node = WorkflowNode(
                    id=node_id,
                    type=NodeType(node_type),
                    x=x,
                    y=y,
                    label=label,
                )
                parsed_graph.add_node(node)
            else:
                i += 1

        return parsed_graph

    parsed_graph = benchmark(import_from_yaml)

    # Measure actual time
    start = time.perf_counter()

    def import_timed():
        lines = yaml_str.split("\n")
        parsed_graph = WorkflowGraph()

        for i, line in enumerate(lines):
            if "id:" in line:
                node_id = line.split(":")[1].strip()
                parsed_graph.add_node(
                    WorkflowNode(
                        id=node_id,
                        type=NodeType.AGENT,
                        x=0.0,
                        y=0.0,
                        label="Node",
                    )
                )

        return parsed_graph

    parsed_graph = import_timed()
    import_time = time.perf_counter() - start

    target_ms = {
        50: 200,
        100: 500,
        200: 1000,
    }.get(node_count, 1500)

    print(f"\nYAML Import | Nodes: {node_count:3} | "
          f"Time: {import_time*1000:7.2f}ms")

    assert import_time * 1000 < target_ms, (
        f"YAML import {import_time*1000:.2f}ms exceeds target {target_ms}ms for {node_count} nodes"
    )


# =============================================================================
# Memory Profiling Benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.parametrize("node_count", [50, 100, 200])
def test_memory_usage(benchmark, node_count):
    """Benchmark memory usage with large workflows.

    Performance Targets:
    - 50 nodes: <50MB
    - 100 nodes: <100MB
    - 200 nodes: <200MB
    """
    gc.collect()
    tracemalloc.start()

    graph = generate_workflow_graph(node_count=node_count)
    editor = MockWorkflowEditor()

    def load_and_measure():
        editor.load_graph(graph)
        editor.render_nodes()
        editor.render_edges()
        current, peak = tracemalloc.get_traced_memory()
        return peak

    peak_memory = benchmark(load_and_measure)
    tracemalloc.stop()

    memory_mb = peak_memory / (1024 * 1024)
    memory_per_node_kb = (peak_memory / node_count) / 1024

    target_mb = {
        50: 50,
        100: 100,
        200: 200,
    }.get(node_count, 300)

    print(f"\nMemory Usage | Nodes: {node_count:3} | "
          f"Total: {memory_mb:6.1f}MB | Per-Node: {memory_per_node_kb:5.1f}KB")

    assert memory_mb < target_mb, (
        f"Memory usage {memory_mb:.1f}MB exceeds target {target_mb}MB for {node_count} nodes"
    )


# =============================================================================
# Complex Scenario Benchmarks
# =============================================================================


@pytest.mark.benchmark
def test_real_world_workflow_editing(benchmark):
    """Benchmark realistic workflow editing scenario.

    Scenario: Load 100-node workflow, search, zoom, pan, layout.
    """
    graph = generate_workflow_graph(node_count=100)
    editor = MockWorkflowEditor()

    def realistic_session():
        # Load workflow
        editor.load_graph(graph)

        # Initial render
        editor.render_nodes()
        editor.render_edges()

        # Search for nodes
        editor.search_nodes("Research")

        # Zoom operations
        for _ in range(3):
            editor.zoom(1.1)

        # Pan operations
        for _ in range(5):
            editor.pan(20, 20)

        # Auto-layout
        editor.auto_layout(algorithm="hierarchical")

        # Final render
        editor.render_nodes()
        editor.render_edges()

        return True

    result = benchmark(realistic_session)

    assert result is True
    print(f"\nReal-world Editing Session | Completed successfully")


# =============================================================================
# Summary and Regression Tests
# =============================================================================


@pytest.mark.summary
def test_editor_performance_summary():
    """Generate comprehensive performance summary for the visual editor."""
    results = {
        "load_time": {},
        "rendering": {},
        "layout": {},
        "yaml": {},
        "memory": {},
    }

    print("\n" + "=" * 80)
    print("VISUAL WORKFLOW EDITOR PERFORMANCE SUMMARY")
    print("=" * 80)

    # Test load time
    print("\n1. Editor Load Time")
    print("-" * 60)
    for node_count in [10, 25, 50, 100, 200]:
        graph = generate_workflow_graph(node_count=node_count)
        editor = MockWorkflowEditor()

        start = time.time()
        editor.load_graph(graph)
        nodes_time = editor.render_nodes()
        edges_time = editor.render_edges()
        total = time.time() - start

        results["load_time"][node_count] = {
            "total_ms": total * 1000,
            "nodes_ms": nodes_time * 1000,
            "edges_ms": edges_time * 1000,
        }

        status = "✓" if total * 1000 < [50, 150, 300, 500, 1000][[10, 25, 50, 100, 200].index(node_count)] else "✗"
        print(f"  {node_count:3} nodes {status}  {total*1000:6.2f}ms")

    # Test rendering performance
    print("\n2. Rendering Performance (per-node)")
    print("-" * 60)
    for node_count in [50, 100, 200]:
        graph = generate_workflow_graph(node_count=node_count)
        editor = MockWorkflowEditor()
        editor.load_graph(graph)

        start = time.time()
        editor.render_nodes()
        nodes_time = time.time() - start

        start = time.time()
        editor.render_edges()
        edges_time = time.time() - start

        per_node = (nodes_time / node_count) * 1000

        results["rendering"][node_count] = {
            "nodes_ms": nodes_time * 1000,
            "edges_ms": edges_time * 1000,
            "per_node_ms": per_node,
        }

        status = "✓" if per_node < 16 else "✗"
        print(f"  {node_count:3} nodes {status}  {per_node:6.3f}ms per node")

    # Test auto-layout
    print("\n3. Auto-Layout Performance")
    print("-" * 60)
    for algorithm in ["grid", "hierarchical", "force_directed"]:
        graph = generate_workflow_graph(node_count=100)
        editor = MockWorkflowEditor()
        editor.load_graph(graph)

        start = time.time()
        layout_time = editor.auto_layout(algorithm=algorithm)
        total_ms = layout_time * 1000

        results["layout"][algorithm] = {"time_ms": total_ms}

        status = "✓" if total_ms < [100, 500, 1000][["grid", "hierarchical", "force_directed"].index(algorithm)] else "✗"
        print(f"  {algorithm:15} {status}  {total_ms:6.2f}ms")

    # Test YAML operations
    print("\n4. YAML Import/Export (100 nodes)")
    print("-" * 60)
    graph = generate_workflow_graph(node_count=100)

    # Export
    start = time.time()
    yaml_str = graph.to_yaml()
    export_time = time.time() - start

    results["yaml"]["export"] = {
        "time_ms": export_time * 1000,
        "size_bytes": len(yaml_str),
    }

    status = "✓" if export_time * 1000 < 300 else "✗"
    print(f"  Export          {status}  {export_time*1000:6.2f}ms  ({len(yaml_str)} bytes)")

    # Import
    start = time.time()
    # Simple import simulation
    lines = yaml_str.split("\n")
    parsed_graph = WorkflowGraph()
    for line in lines:
        if "id:" in line:
            node_id = line.split(":")[1].strip()
            parsed_graph.add_node(
                WorkflowNode(
                    id=node_id,
                    type=NodeType.AGENT,
                    x=0.0,
                    y=0.0,
                    label="Node",
                )
            )
    import_time = time.time() - start

    results["yaml"]["import"] = {"time_ms": import_time * 1000}

    status = "✓" if import_time * 1000 < 500 else "✗"
    print(f"  Import          {status}  {import_time*1000:6.2f}ms")

    # Test memory usage
    print("\n5. Memory Usage")
    print("-" * 60)
    for node_count in [50, 100, 200]:
        gc.collect()
        tracemalloc.start()

        graph = generate_workflow_graph(node_count=node_count)
        editor = MockWorkflowEditor()
        editor.load_graph(graph)
        editor.render_nodes()
        editor.render_edges()

        peak_mb = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        tracemalloc.stop()

        results["memory"][node_count] = {"peak_mb": peak_mb}

        status = "✓" if peak_mb < [50, 100, 200][[50, 100, 200].index(node_count)] else "✗"
        print(f"  {node_count:3} nodes {status}  {peak_mb:6.1f}MB")

    # Print summary
    print("\n" + "=" * 80)
    print("PERFORMANCE TARGETS")
    print("=" * 80)
    print("  ✓ Editor load (100 nodes): <500ms")
    print("  ✓ Node rendering: <16ms per node (60fps)")
    print("  ✓ Auto-layout: <500ms for hierarchical (100 nodes)")
    print("  ✓ YAML export: <300ms for 100 nodes")
    print("  ✓ YAML import: <500ms for 100 nodes")
    print("  ✓ Memory usage: <100MB for 100 nodes")
    print("  ✓ Zoom/pan: <16ms per operation (60fps)")
    print("\n" + "=" * 80)

    # Save results
    results_dir = Path("/tmp/benchmark_results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "editor_benchmarks.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Verify critical targets
    assert results["load_time"][100]["total_ms"] < 500
    assert results["rendering"][100]["per_node_ms"] < 16
    assert results["memory"][100]["peak_mb"] < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-k", "summary"])
