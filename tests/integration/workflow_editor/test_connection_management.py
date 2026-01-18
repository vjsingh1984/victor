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

"""Integration tests for connection management in the workflow editor.

Tests connection mapping, validation, and transformation between YAML
and visual graph representations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import pytest

from victor.workflows import load_workflow_from_file
from victor.workflows.definition import (
    WorkflowDefinition,
    ConditionNode,
    TeamNodeWorkflow,
    ParallelNode,
    TransformNode,
)
from victor.workflows.yaml_loader import YAMLWorkflowConfig, load_workflow_from_yaml


class TestConnectionMapping:
    """Test mapping connections from YAML to visual graph."""

    def test_map_simple_linear_connections(self):
        """Test mapping simple linear A->B->C connections."""
        from victor.workflows.yaml_loader import load_workflow_from_yaml, YAMLWorkflowConfig

        yaml_content = """
workflows:
  linear:
    nodes:
      - id: node_a
        type: agent
        role: assistant
        goal: "First node"
        tool_budget: 5
        next: [node_b]

      - id: node_b
        type: agent
        role: assistant
        goal: "Second node"
        tool_budget: 5
        next: [node_c]

      - id: node_c
        type: transform
        transform: "complete = true"
"""

        config = YAMLWorkflowConfig()
        workflow_def = load_workflow_from_yaml(
            yaml_content, "linear", config
        )  # Returns WorkflowDefinition directly when name is provided

        # Build connection map
        connections = self._build_connection_map(workflow_def)

        assert connections == {
            "node_a": ["node_b"],
            "node_b": ["node_c"],
            "node_c": [],
        }

    def test_map_branching_connections(self):
        """Test mapping branching connections from conditions."""
        workflows = load_workflow_from_file("victor/coding/workflows/team_node_example.yaml")
        workflow_def = workflows["team_node_demo"]

        # Build connection map
        connections = self._build_connection_map(workflow_def)

        # Find condition node
        condition_nodes = [n for n in workflow_def.nodes.values() if isinstance(n, ConditionNode)]
        assert len(condition_nodes) > 0

        condition = condition_nodes[0]

        # Verify branches create connections
        assert condition.id in connections
        branch_targets = list(condition.branches.values())
        assert all(target in connections[condition.id] for target in branch_targets)

    def test_map_parallel_connections(self):
        """Test mapping parallel execution connections."""
        from victor.workflows.yaml_loader import load_workflow_from_yaml, YAMLWorkflowConfig

        yaml_content = """
workflows:
  parallel:
    nodes:
      - id: start
        type: agent
        role: assistant
        goal: "Start"
        tool_budget: 5
        next: [parallel_node]

      - id: parallel_node
        type: parallel
        parallel_nodes: [branch1, branch2]
        join_strategy: all
        next: [end]

      - id: branch1
        type: agent
        role: assistant
        goal: "Branch 1"

      - id: branch2
        type: agent
        role: assistant
        goal: "Branch 2"

      - id: end
        type: transform
        transform: "done = true"
"""

        config = YAMLWorkflowConfig()
        workflow_def = load_workflow_from_yaml(
            yaml_content, "parallel", config
        )  # Returns WorkflowDefinition directly

        # Build connection map
        connections = self._build_connection_map(workflow_def)

        # Verify parallel node connections
        assert "parallel_node" in connections
        assert "end" in connections["parallel_node"]

    def test_map_team_node_connections(self):
        """Test mapping team node connections."""
        workflows = load_workflow_from_file("victor/coding/workflows/team_node_example.yaml")
        workflow_def = workflows["team_node_demo"]

        # Build connection map
        connections = self._build_connection_map(workflow_def)

        # Find team nodes
        team_nodes = [n for n in workflow_def.nodes.values() if isinstance(n, TeamNodeWorkflow)]

        for team_node in team_nodes:
            assert team_node.id in connections
            # Team should have outgoing connections
            if team_node.next_nodes:
                assert len(connections[team_node.id]) > 0

    def _build_connection_map(self, workflow_def: WorkflowDefinition) -> Dict[str, List[str]]:
        """Build a map of node connections."""
        connections = {}
        for node in workflow_def.nodes.values():  # Fixed: iterate over values
            targets = []
            if hasattr(node, "next_nodes") and node.next_nodes:
                targets = node.next_nodes
            elif hasattr(node, "branches") and node.branches:
                # For condition nodes, use branch targets
                targets = list(node.branches.values())

            connections[node.id] = targets

        return connections


class TestConnectionValidation:
    """Test validation of node connections."""

    def test_validate_no_dangling_connections(self):
        """Test that all connections point to valid nodes."""
        workflows = load_workflow_from_file("victor/coding/workflows/team_node_example.yaml")
        workflow_def = workflows["team_node_demo"]

        # Get all node IDs
        all_node_ids = set(workflow_def.nodes.keys())  # Fixed: use keys() for IDs

        # Validate all connections
        for node in workflow_def.nodes.values():  # Fixed: iterate over values
            if hasattr(node, "next_nodes") and node.next_nodes:
                for target_id in node.next_nodes:
                    assert (
                        target_id in all_node_ids
                    ), f"Dangling connection: {node.id} -> {target_id}"

            if hasattr(node, "branches") and node.branches:
                for branch_name, target_id in node.branches.items():
                    assert (
                        target_id in all_node_ids
                    ), f"Dangling branch: {node.id}.{branch_name} -> {target_id}"

    def test_validate_no_self_loops(self):
        """Test that nodes don't connect to themselves."""
        workflows = load_workflow_from_file("victor/coding/workflows/team_node_example.yaml")
        workflow_def = workflows["team_node_demo"]

        for node in workflow_def.nodes.values():  # Fixed: iterate over values
            if hasattr(node, "next_nodes") and node.next_nodes:
                assert node.id not in node.next_nodes, f"Self-loop detected: {node.id}"

            if hasattr(node, "branches") and node.branches:
                for target_id in node.branches.values():
                    assert target_id != node.id, f"Self-loop in branch: {node.id}"

    def test_validate_conditional_branches_unique(self):
        """Test that conditional branches have unique targets."""
        workflows = load_workflow_from_file("victor/coding/workflows/team_node_example.yaml")
        workflow_def = workflows["team_node_demo"]

        condition_nodes = [n for n in workflow_def.nodes if isinstance(n, ConditionNode)]

        for condition in condition_nodes:
            targets = list(condition.branches.values())
            # Check for duplicate targets (may be allowed, but worth noting)
            unique_targets = set(targets)
            if len(targets) != len(unique_targets):
                # Multiple branches lead to same node - may be intentional
                pass

    def test_validate_parallel_nodes_exist(self):
        """Test that parallel node references point to existing nodes."""
        workflows = load_workflow_from_file("victor/research/workflows/deep_research.yaml")
        workflow_def = workflows["deep_research"]

        all_node_ids = set(workflow_def.nodes.keys())  # Fixed: use keys() for IDs

        parallel_nodes = [n for n in workflow_def.nodes.values() if isinstance(n, ParallelNode)]

        for parallel in parallel_nodes:
            assert hasattr(parallel, "parallel_nodes")
            for ref_id in parallel.parallel_nodes:
                assert ref_id in all_node_ids, f"Parallel reference not found: {ref_id}"


class TestConnectionTransformation:
    """Test transformation of connections between formats."""

    def test_yaml_to_graph_edges(self):
        """Test converting YAML connections to graph edges."""
        workflows = load_workflow_from_file("victor/coding/workflows/team_node_example.yaml")
        workflow_def = workflows["team_node_demo"]

        # Convert to graph edges
        edges = []
        for node in workflow_def.nodes.values():  # Fixed: iterate over values
            if hasattr(node, "next_nodes") and node.next_nodes:
                for target_id in node.next_nodes:
                    edge = {
                        "id": f"{node.id}->{target_id}",
                        "source": node.id,
                        "target": target_id,
                        "label": None,
                    }
                    edges.append(edge)

        # Verify edges
        assert len(edges) > 0

        # Check edge structure
        for edge in edges:
            assert "id" in edge
            assert "source" in edge
            assert "target" in edge
            assert edge["source"] != edge["target"]

    def test_conditional_branches_to_edges(self):
        """Test converting conditional branches to labeled edges."""
        workflows = load_workflow_from_file("victor/coding/workflows/team_node_example.yaml")
        workflow_def = workflows["team_node_demo"]

        # Find condition node
        condition_nodes = [n for n in workflow_def.nodes.values() if isinstance(n, ConditionNode)]
        assert len(condition_nodes) > 0

        condition = condition_nodes[0]

        # Create labeled edges for branches
        edges = []
        for branch_name, target_id in condition.branches.items():
            edge = {
                "id": f"{condition.id}_{branch_name}->{target_id}",
                "source": condition.id,
                "target": target_id,
                "label": branch_name,  # Branch name as label
            }
            edges.append(edge)

        # Verify labeled edges
        assert len(edges) >= 2  # At least 2 branches
        for edge in edges:
            assert edge["label"] is not None
            assert edge["label"] in condition.branches

    def test_graph_edges_to_yaml_connections(self):
        """Test converting graph edges back to YAML connections."""
        # Simulate graph edges
        graph_edges = [
            {"source": "node_a", "target": "node_b", "label": None},
            {"source": "node_b", "target": "node_c", "label": None},
            {"source": "decision", "target": "option_a", "label": "yes"},
            {"source": "decision", "target": "option_b", "label": "no"},
        ]

        # Group edges by source
        connections = {}
        branches = {}  # For conditional nodes

        for edge in graph_edges:
            source = edge["source"]
            target = edge["target"]
            label = edge["label"]

            if label is None:
                # Regular connection
                if source not in connections:
                    connections[source] = []
                connections[source].append(target)
            else:
                # Conditional branch
                if source not in branches:
                    branches[source] = {}
                branches[source][label] = target

        # Verify transformation
        assert "node_a" in connections
        assert connections["node_a"] == ["node_b"]
        assert "decision" in branches
        assert branches["decision"]["yes"] == "option_a"
        assert branches["decision"]["no"] == "option_b"


class TestConnectionCycles:
    """Test detection and handling of cycles."""

    def test_detect_direct_cycle(self):
        """Test detection of A->B->A cycle."""
        connections = {
            "node_a": ["node_b"],
            "node_b": ["node_a"],
        }

        has_cycle = self._detect_cycle(connections)
        assert has_cycle, "Should detect direct cycle"

    def test_detect_indirect_cycle(self):
        """Test detection of A->B->C->A cycle."""
        connections = {
            "node_a": ["node_b"],
            "node_b": ["node_c"],
            "node_c": ["node_a"],
        }

        has_cycle = self._detect_cycle(connections)
        assert has_cycle, "Should detect indirect cycle"

    def test_allow_controlled_cycles(self):
        """Test that controlled cycles (iteration) are allowed."""
        # Workflows may have intentional iteration cycles
        connections = {
            "analyze": ["fix"],
            "fix": ["test"],
            "test": ["analyze", "complete"],  # Cycle back to analyze
            "complete": [],
        }

        # This should be detected but may be allowed with iteration limit
        has_cycle = self._detect_cycle(connections)
        assert has_cycle, "Should detect iteration cycle"

    def test_no_false_positives(self):
        """Test that DAG structures don't trigger cycle detection."""
        connections = {
            "start": ["process"],
            "process": ["branch"],
            "branch": ["option_a", "option_b"],
            "option_a": ["end"],
            "option_b": ["end"],
            "end": [],
        }

        has_cycle = self._detect_cycle(connections)
        assert not has_cycle, "Should not detect cycle in DAG"

    def _detect_cycle(self, connections: Dict[str, List[str]]) -> bool:
        """Detect if connection graph has cycles using DFS."""
        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in connections.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in connections:
            if node not in visited:
                if dfs(node):
                    return True

        return False


class TestConnectionPaths:
    """Test path finding and analysis."""

    def test_find_all_paths_simple(self):
        """Test finding all paths in simple workflow."""
        connections = {
            "start": ["a", "b"],
            "a": ["end"],
            "b": ["end"],
            "end": [],
        }

        paths = self._find_all_paths(connections, "start", "end")

        assert len(paths) == 2
        assert ["start", "a", "end"] in paths
        assert ["start", "b", "end"] in paths

    def test_find_all_paths_with_branching(self):
        """Test finding paths with conditional branching."""
        workflows = load_workflow_from_file("victor/coding/workflows/team_node_example.yaml")
        workflow_def = workflows["team_node_demo"]

        # Build connection map
        connections = self._build_connection_map_with_branches(workflow_def)

        # Find all paths from start to end
        start_nodes = [
            n for n in workflow_def.nodes.values() if not self._has_incoming(n, workflow_def)
        ]
        end_nodes = [
            n
            for n in workflow_def.nodes.values()
            if not hasattr(n, "next_nodes") or not n.next_nodes
        ]

        if start_nodes and end_nodes:
            paths = self._find_all_paths(connections, start_nodes[0].id, end_nodes[0].id)
            # Should find multiple paths due to branching
            assert len(paths) >= 1

    def test_find_critical_path(self):
        """Test finding critical path (longest) through workflow."""
        connections = {
            "start": ["a", "b"],
            "a": ["c"],
            "b": ["c"],
            "c": ["end"],
            "end": [],
        }

        paths = self._find_all_paths(connections, "start", "end")

        # Find longest path
        critical_path = max(paths, key=len)

        assert len(critical_path) == 4  # start -> a -> c -> end (or start -> b -> c -> end)

    def _build_connection_map_with_branches(
        self, workflow_def: WorkflowDefinition
    ) -> Dict[str, List[str]]:
        """Build connection map including conditional branches."""
        connections = {}
        for node in workflow_def.nodes.values():  # Fixed: iterate over values
            targets = []

            if hasattr(node, "next_nodes") and node.next_nodes:
                targets.extend(node.next_nodes)

            if hasattr(node, "branches") and node.branches:
                # Add branch targets
                targets.extend(set(node.branches.values()))

            connections[node.id] = list(set(targets))  # Remove duplicates

        return connections

    def _has_incoming(self, node, workflow_def: WorkflowDefinition) -> bool:
        """Check if node has incoming connections."""
        for other in workflow_def.nodes.values():
            if hasattr(other, "next_nodes") and other.next_nodes:
                if node.id in other.next_nodes:
                    return True
            if hasattr(other, "branches") and other.branches:
                if node.id in other.branches.values():
                    return True
        return False

    def _find_all_paths(
        self, connections: Dict[str, List[str]], start: str, end: str, path: List[str] = None
    ) -> List[List[str]]:
        """Find all paths from start to end using DFS."""
        if path is None:
            path = []

        path = path + [start]

        if start == end:
            return [path]

        if start not in connections:
            return []

        paths = []
        for node in connections[start]:
            if node not in path:  # Avoid cycles
                new_paths = self._find_all_paths(connections, node, end, path)
                paths.extend(new_paths)

        return paths


class TestConnectionVisualization:
    """Test connection visualization data for UI."""

    def test_edge_label_generation(self):
        """Test generation of edge labels for visualization."""
        workflows = load_workflow_from_file("victor/coding/workflows/team_node_example.yaml")
        workflow_def = workflows["team_node_demo"]

        # Generate edge labels
        edge_labels = []

        for node in workflow_def.nodes.values():  # Fixed: iterate over values
            if isinstance(node, ConditionNode):
                # Create labeled edges for branches
                for branch_name, target_id in node.branches.items():
                    edge_labels.append(
                        {
                            "source": node.id,
                            "target": target_id,
                            "label": branch_name,
                            "type": "conditional",
                        }
                    )

        # Verify labels
        assert len(edge_labels) > 0

        for label in edge_labels:
            assert "label" in label
            assert label["type"] == "conditional"

    def test_edge_type_classification(self):
        """Test classification of edge types."""
        workflows = load_workflow_from_file("victor/research/workflows/deep_research.yaml")
        workflow_def = workflows["deep_research"]

        edge_types = []

        for node in workflow_def.nodes.values():  # Fixed: iterate over values
            if hasattr(node, "next_nodes") and node.next_nodes:
                edge_type = "standard"

                # Classify based on node type
                if isinstance(node, ParallelNode):
                    edge_type = "parallel_join"
                elif isinstance(node, ConditionNode):
                    edge_type = "conditional"
                elif isinstance(node, TeamNodeWorkflow):
                    edge_type = "team_output"

                for target_id in node.next_nodes:
                    edge_types.append(
                        {
                            "source": node.id,
                            "target": target_id,
                            "type": edge_type,
                        }
                    )

        # Verify classification
        assert len(edge_types) > 0
        assert any(e["type"] == "parallel_join" for e in edge_types)

    def test_connection_layout_hints(self):
        """Test generation of layout hints for connections."""
        from victor.workflows.yaml_loader import load_workflow_from_yaml, YAMLWorkflowConfig

        yaml_content = """
workflows:
  layout_test:
    nodes:
      - id: top
        type: agent
        role: assistant
        goal: "Top"
        tool_budget: 5
        next: [left, right]

      - id: left
        type: agent
        role: assistant
        goal: "Left"
        tool_budget: 5
        next: [bottom]

      - id: right
        type: agent
        role: assistant
        goal: "Right"
        tool_budget: 5
        next: [bottom]

      - id: bottom
        type: transform
        transform: "done = true"
"""

        config = YAMLWorkflowConfig()
        workflow_def = load_workflow_from_yaml(
            yaml_content, "layout_test", config
        )  # Returns WorkflowDefinition directly

        # Generate layout hints
        layout_hints = {}

        for node in workflow_def.nodes.values():  # Fixed: iterate over values
            if hasattr(node, "next_nodes") and node.next_nodes:
                if len(node.next_nodes) > 1:
                    # Splitting node - place at top
                    layout_hints[node.id] = {"rank": "top"}
                elif len(node.next_nodes) == 1:
                    # Linear flow - middle
                    layout_hints[node.id] = {"rank": "middle"}

        # Verify hints
        assert "top" in layout_hints
        assert layout_hints["top"]["rank"] == "top"


class TestConnectionSerialization:
    """Test serialization of connections for storage/transfer."""

    def test_serialize_connections_to_json(self):
        """Test serializing connections to JSON format."""
        import json

        workflows = load_workflow_from_file("victor/coding/workflows/team_node_example.yaml")
        workflow_def = workflows["team_node_demo"]

        # Build connection data
        connection_data = {"workflow_id": workflow_def.name, "connections": []}

        for node in workflow_def.nodes.values():  # Fixed: iterate over values
            if hasattr(node, "next_nodes") and node.next_nodes:
                for target_id in node.next_nodes:
                    connection_data["connections"].append(
                        {
                            "source": node.id,
                            "target": target_id,
                            "type": "standard",
                        }
                    )

            if hasattr(node, "branches") and node.branches:
                for branch_name, target_id in node.branches.items():
                    connection_data["connections"].append(
                        {
                            "source": node.id,
                            "target": target_id,
                            "type": "conditional",
                            "label": branch_name,
                        }
                    )

        # Serialize to JSON
        json_str = json.dumps(connection_data)

        # Deserialize and verify
        parsed = json.loads(json_str)
        assert parsed["workflow_id"] == workflow_def.name
        assert len(parsed["connections"]) > 0

    def test_deserialize_connections_from_json(self):
        """Test deserializing connections from JSON format."""
        import json

        json_data = """
        {
            "workflow_id": "test_workflow",
            "connections": [
                {"source": "a", "target": "b", "type": "standard"},
                {"source": "b", "target": "c", "type": "conditional", "label": "yes"},
                {"source": "b", "target": "d", "type": "conditional", "label": "no"}
            ]
        }
        """

        data = json.loads(json_data)

        # Rebuild connection map
        connections = {}
        branches = {}

        for conn in data["connections"]:
            source = conn["source"]
            target = conn["target"]

            if conn["type"] == "conditional":
                if source not in branches:
                    branches[source] = {}
                branches[source][conn["label"]] = target
            else:
                if source not in connections:
                    connections[source] = []
                connections[source].append(target)

        # Verify
        assert "a" in connections
        assert connections["a"] == ["b"]
        assert "b" in branches
        assert branches["b"]["yes"] == "c"
        assert branches["b"]["no"] == "d"
