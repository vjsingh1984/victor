"""Tests for the visual workflow editor API.

Tests the FastAPI backend for the workflow editor including:
- Workflow validation
- YAML import/export
- Compilation
- Node type definitions
- Team formation configurations
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def client():
    """Create test client for workflow editor API."""
    # Import here to avoid circular imports
    from tools.workflow_editor.backend.api import app

    return TestClient(app)


@pytest.fixture
def sample_workflow_graph():
    """Create a sample workflow graph for testing."""
    return {
        "nodes": [
            {
                "id": "start",
                "type": "agent",
                "name": "Analyzer",
                "config": {"role": "analyst", "goal": "Analyze the input"},
                "position": {"x": 100, "y": 100},
            },
            {
                "id": "process",
                "type": "compute",
                "name": "Processor",
                "config": {"handler": "process_data"},
                "position": {"x": 300, "y": 100},
            },
            {
                "id": "end",
                "type": "agent",
                "name": "Summarizer",
                "config": {"role": "summarizer", "goal": "Summarize results"},
                "position": {"x": 500, "y": 100},
            },
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "process"},
            {"id": "e2", "source": "process", "target": "end"},
        ],
        "metadata": {"name": "test_workflow", "version": "1.0"},
    }


@pytest.fixture
def sample_team_node():
    """Create a sample team node configuration."""
    return {
        "id": "research_team",
        "type": "team",
        "name": "Research Team",
        "config": {
            "formation": "parallel",
            "goal": "Research the topic thoroughly",
            "max_iterations": 5,
            "members": [
                {
                    "id": "researcher",
                    "role": "researcher",
                    "goal": "Find relevant information",
                    "tool_budget": 25,
                },
                {
                    "id": "analyst",
                    "role": "analyst",
                    "goal": "Analyze findings",
                    "tool_budget": 20,
                },
            ],
        },
        "position": {"x": 200, "y": 200},
    }


# =============================================================================
# Root and Health Endpoint Tests
# =============================================================================


class TestRootEndpoints:
    """Tests for root and health endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Victor Workflow Editor API"
        assert "version" in data
        assert data["status"] == "running"

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


# =============================================================================
# Workflow Validation Tests
# =============================================================================


class TestWorkflowValidation:
    """Tests for workflow validation endpoint."""

    def test_validate_valid_workflow(self, client, sample_workflow_graph):
        """Test validation of a valid workflow."""
        response = client.post("/api/workflows/validate", json=sample_workflow_graph)

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert len(data["errors"]) == 0

    def test_validate_empty_workflow(self, client):
        """Test validation of workflow with no nodes."""
        graph = {"nodes": [], "edges": [], "metadata": {}}

        response = client.post("/api/workflows/validate", json=graph)

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert any("at least one node" in err for err in data["errors"])

    def test_validate_duplicate_node_ids(self, client):
        """Test validation catches duplicate node IDs."""
        graph = {
            "nodes": [
                {"id": "node1", "type": "agent", "name": "A", "config": {"role": "a", "goal": "a"}},
                {"id": "node1", "type": "agent", "name": "B", "config": {"role": "b", "goal": "b"}},
            ],
            "edges": [],
            "metadata": {},
        }

        response = client.post("/api/workflows/validate", json=graph)

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert any("Duplicate" in err for err in data["errors"])

    def test_validate_invalid_edge_source(self, client):
        """Test validation catches invalid edge sources."""
        graph = {
            "nodes": [
                {"id": "node1", "type": "agent", "name": "A", "config": {"role": "a", "goal": "a"}},
            ],
            "edges": [
                {"id": "e1", "source": "nonexistent", "target": "node1"},
            ],
            "metadata": {},
        }

        response = client.post("/api/workflows/validate", json=graph)

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert any("nonexistent" in err for err in data["errors"])

    def test_validate_invalid_edge_target(self, client):
        """Test validation catches invalid edge targets."""
        graph = {
            "nodes": [
                {"id": "node1", "type": "agent", "name": "A", "config": {"role": "a", "goal": "a"}},
            ],
            "edges": [
                {"id": "e1", "source": "node1", "target": "nonexistent"},
            ],
            "metadata": {},
        }

        response = client.post("/api/workflows/validate", json=graph)

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert any("nonexistent" in err for err in data["errors"])

    def test_validate_agent_missing_role(self, client):
        """Test validation catches agent nodes without role."""
        graph = {
            "nodes": [
                {"id": "node1", "type": "agent", "name": "A", "config": {}},
            ],
            "edges": [],
            "metadata": {},
        }

        response = client.post("/api/workflows/validate", json=graph)

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert any("role" in err.lower() or "goal" in err.lower() for err in data["errors"])

    def test_validate_team_missing_members(self, client):
        """Test validation catches team nodes without members."""
        graph = {
            "nodes": [
                {
                    "id": "team1",
                    "type": "team",
                    "name": "Team",
                    "config": {"formation": "parallel"},
                },
            ],
            "edges": [],
            "metadata": {},
        }

        response = client.post("/api/workflows/validate", json=graph)

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert any("member" in err.lower() for err in data["errors"])

    def test_validate_team_missing_formation(self, client):
        """Test validation catches team nodes without formation."""
        graph = {
            "nodes": [
                {
                    "id": "team1",
                    "type": "team",
                    "name": "Team",
                    "config": {"members": [{"id": "m1", "role": "worker"}]},
                },
            ],
            "edges": [],
            "metadata": {},
        }

        response = client.post("/api/workflows/validate", json=graph)

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert any("formation" in err.lower() for err in data["errors"])

    def test_validate_condition_missing_branches(self, client):
        """Test validation catches condition nodes without branches."""
        graph = {
            "nodes": [
                {"id": "cond1", "type": "condition", "name": "Check", "config": {}},
            ],
            "edges": [],
            "metadata": {},
        }

        response = client.post("/api/workflows/validate", json=graph)

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert any(
            "condition" in err.lower() or "branches" in err.lower() for err in data["errors"]
        )


# =============================================================================
# Node Types Tests
# =============================================================================


class TestNodeTypes:
    """Tests for node type definitions endpoint."""

    def test_get_node_types(self, client):
        """Test getting available node types."""
        response = client.get("/api/nodes/types")

        assert response.status_code == 200
        data = response.json()

        # Check all expected node types exist
        expected_types = ["agent", "compute", "team", "condition", "parallel", "transform", "hitl"]
        for node_type in expected_types:
            assert node_type in data
            assert "name" in data[node_type]
            assert "description" in data[node_type]
            assert "config_schema" in data[node_type]

    def test_agent_node_schema(self, client):
        """Test agent node schema has required fields."""
        response = client.get("/api/nodes/types")
        data = response.json()

        agent_schema = data["agent"]["config_schema"]
        assert "role" in agent_schema
        assert "goal" in agent_schema
        assert agent_schema["role"]["required"] is True
        assert agent_schema["goal"]["required"] is True

    def test_team_node_schema(self, client):
        """Test team node schema has required fields."""
        response = client.get("/api/nodes/types")
        data = response.json()

        team_schema = data["team"]["config_schema"]
        assert "formation" in team_schema
        assert "members" in team_schema
        assert "parallel" in team_schema["formation"]["enum"]

    def test_hitl_node_schema(self, client):
        """Test HITL node schema has required fields."""
        response = client.get("/api/nodes/types")
        data = response.json()

        hitl_schema = data["hitl"]["config_schema"]
        assert "hitl_type" in hitl_schema
        assert "prompt" in hitl_schema
        assert "approval" in hitl_schema["hitl_type"]["enum"]


# =============================================================================
# Team Formations Tests
# =============================================================================


class TestFormations:
    """Tests for team formation definitions endpoint."""

    def test_get_formations(self, client):
        """Test getting available team formations."""
        response = client.get("/api/formations")

        assert response.status_code == 200
        data = response.json()

        # Check all expected formations exist
        expected_formations = ["parallel", "sequential", "pipeline", "hierarchical", "consensus"]
        for formation in expected_formations:
            assert formation in data
            assert "name" in data[formation]
            assert "description" in data[formation]
            assert "best_for" in data[formation]
            assert "communication_style" in data[formation]

    def test_parallel_formation_details(self, client):
        """Test parallel formation has correct details."""
        response = client.get("/api/formations")
        data = response.json()

        parallel = data["parallel"]
        assert parallel["name"] == "Parallel Formation"
        assert "simultaneously" in parallel["description"].lower()
        assert "communication_style" in parallel

    def test_hierarchical_formation_details(self, client):
        """Test hierarchical formation has correct details."""
        response = client.get("/api/formations")
        data = response.json()

        hierarchical = data["hierarchical"]
        assert hierarchical["name"] == "Hierarchical Formation"
        assert (
            "manager" in hierarchical["description"].lower()
            or "coordinator" in hierarchical["description"].lower()
        )


# =============================================================================
# YAML Export Tests
# =============================================================================


class TestYAMLExport:
    """Tests for YAML export endpoint."""

    def test_export_valid_workflow(self, client, sample_workflow_graph):
        """Test exporting a valid workflow to YAML."""
        with patch("tools.workflow_editor.backend.api.graph_to_definition") as mock_convert:
            # Create a mock node with to_dict() method (victor.workflows.definition.WorkflowNode)
            mock_node = MagicMock()
            mock_node.to_dict.return_value = {
                "id": "test_node",
                "name": "Test Agent",
                "type": "agent",
                "next_nodes": [],
            }

            mock_def = MagicMock()
            mock_def.name = "test_workflow"
            mock_def.nodes = {"node1": mock_node}
            mock_def.start_node = "node1"
            mock_convert.return_value = mock_def

            response = client.post("/api/workflows/export/yaml", json=sample_workflow_graph)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "yaml_content" in data

    def test_export_handles_error(self, client, sample_workflow_graph):
        """Test export handles errors gracefully."""
        with patch("tools.workflow_editor.backend.api.graph_to_definition") as mock_convert:
            mock_convert.side_effect = ValueError("Conversion failed")

            response = client.post("/api/workflows/export/yaml", json=sample_workflow_graph)

            assert response.status_code == 500


# =============================================================================
# Compilation Tests
# =============================================================================


class TestCompilation:
    """Tests for workflow compilation endpoint."""

    def test_compile_valid_yaml(self, client):
        """Test compiling valid YAML content."""
        yaml_content = """
name: test_workflow
nodes:
  - id: start
    type: agent
    role: analyst
    goal: Analyze data
"""
        request = {
            "yaml_content": yaml_content,
            "workflow_name": "test_workflow",
        }

        with patch("tools.workflow_editor.backend.api.get_compiler") as mock_get:
            mock_compiler = MagicMock()
            mock_graph = MagicMock()
            mock_graph.get_graph_schema.return_value = {"nodes": [], "edges": {}}
            mock_compiler.compile_yaml_content.return_value = mock_graph
            mock_get.return_value = mock_compiler

            response = client.post("/api/workflows/compile", json=request)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["workflow_id"] == "test_workflow"
            assert "graph_schema" in data

    def test_compile_invalid_yaml(self, client):
        """Test compiling invalid YAML returns error."""
        request = {
            "yaml_content": "invalid: yaml: content: :",
            "workflow_name": "test",
        }

        with patch("tools.workflow_editor.backend.api.get_compiler") as mock_get:
            mock_compiler = MagicMock()
            mock_compiler.compile_yaml_content.side_effect = Exception("Parse error")
            mock_get.return_value = mock_compiler

            response = client.post("/api/workflows/compile", json=request)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert len(data["errors"]) > 0


# =============================================================================
# Graph Conversion Tests
# =============================================================================


class TestGraphConversion:
    """Tests for graph conversion utilities."""

    def test_graph_to_definition_agent_nodes(self):
        """Test converting graph with agent nodes to definition."""
        from tools.workflow_editor.backend.api import (
            graph_to_definition,
            WorkflowGraph,
            WorkflowNode,
        )

        graph = WorkflowGraph(
            nodes=[
                WorkflowNode(
                    id="agent1",
                    type="agent",
                    name="Test Agent",
                    config={"role": "assistant", "goal": "Help users", "tool_budget": 30},
                    position={"x": 0, "y": 0},
                )
            ],
            edges=[],
            metadata={"name": "test_workflow"},
        )

        definition = graph_to_definition(graph)

        assert definition.name == "test_workflow"
        # nodes is a Dict[str, WorkflowNode]
        assert len(definition.nodes) == 1
        assert "agent1" in definition.nodes
        assert definition.start_node == "agent1"

    def test_definition_to_graph(self):
        """Test converting definition back to graph."""
        from tools.workflow_editor.backend.api import definition_to_graph
        from victor.workflows.definition import WorkflowDefinition, AgentNode

        # WorkflowDefinition expects nodes as Dict[str, WorkflowNode]
        agent_node = AgentNode(id="a1", name="Agent", role="helper", goal="Help")
        definition = WorkflowDefinition(
            name="test",
            nodes={"a1": agent_node},
            start_node="a1",
        )

        graph = definition_to_graph(definition)

        assert len(graph.nodes) == 1
        assert graph.nodes[0].id == "a1"


# =============================================================================
# Integration Tests with Team Nodes
# =============================================================================


class TestTeamNodeIntegration:
    """Integration tests for team node handling."""

    def test_validate_complete_team_workflow(self, client, sample_team_node):
        """Test validating a complete workflow with team nodes."""
        graph = {
            "nodes": [
                {
                    "id": "start",
                    "type": "agent",
                    "name": "Starter",
                    "config": {"role": "starter", "goal": "Start"},
                },
                sample_team_node,
                {
                    "id": "end",
                    "type": "agent",
                    "name": "Finisher",
                    "config": {"role": "finisher", "goal": "Finish"},
                },
            ],
            "edges": [
                {"id": "e1", "source": "start", "target": "research_team"},
                {"id": "e2", "source": "research_team", "target": "end"},
            ],
            "metadata": {"name": "team_workflow"},
        }

        response = client.post("/api/workflows/validate", json=graph)

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling across endpoints."""

    def test_validation_with_malformed_json(self, client):
        """Test validation handles malformed JSON."""
        response = client.post(
            "/api/workflows/validate",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422  # Validation error

    def test_export_with_missing_fields(self, client):
        """Test export handles missing required fields."""
        graph = {"nodes": []}  # Missing edges and metadata

        response = client.post("/api/workflows/export/yaml", json=graph)

        # Should either succeed with defaults or return validation error
        assert response.status_code in [200, 422]

    def test_compile_with_empty_yaml(self, client):
        """Test compile handles empty YAML."""
        request = {"yaml_content": "", "workflow_name": "empty"}

        with patch("tools.workflow_editor.backend.api.get_compiler") as mock_get:
            mock_compiler = MagicMock()
            mock_compiler.compile_yaml_content.side_effect = Exception("Empty YAML")
            mock_get.return_value = mock_compiler

            response = client.post("/api/workflows/compile", json=request)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
