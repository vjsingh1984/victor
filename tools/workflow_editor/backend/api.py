#!/usr/bin/env python3
"""FastAPI backend for workflow editor.

Provides REST API for:
- Workflow CRUD operations
- YAML import/export
- Real-time validation
- Compilation with UnifiedWorkflowCompiler
- Team node configuration
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from victor.workflows.unified_compiler import UnifiedWorkflowCompiler
from victor.workflows.yaml_loader import load_workflow_from_file, load_workflow_from_yaml, YAMLWorkflowConfig
from victor.workflows.definition import WorkflowDefinition
from victor.core.errors import ConfigurationValidationError

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Victor Workflow Editor API",
    description="REST API for visual workflow editor",
    version="0.1.0",
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global compiler instance
_compiler: Optional[UnifiedWorkflowCompiler] = None


def get_compiler() -> UnifiedWorkflowCompiler:
    """Get or create the global compiler instance."""
    global _compiler
    if _compiler is None:
        _compiler = UnifiedWorkflowCompiler(enable_caching=True)
    return _compiler


# =============================================================================
# Request/Response Models
# =============================================================================


class WorkflowNode(BaseModel):
    """Node in workflow graph."""

    id: str
    type: str  # agent, compute, team, condition, parallel, transform, hitl
    name: str
    config: Dict[str, Any] = Field(default_factory=dict)
    position: Dict[str, float] = Field(default_factory=dict)  # {x, y}


class WorkflowEdge(BaseModel):
    """Connection between nodes."""

    id: str
    source: str
    target: str
    label: Optional[str] = None


class WorkflowGraph(BaseModel):
    """Complete workflow graph."""

    nodes: List[WorkflowNode]
    edges: List[WorkflowEdge]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowCreateRequest(BaseModel):
    """Request to create a new workflow."""

    name: str
    description: Optional[str] = None
    graph: WorkflowGraph


class WorkflowUpdateRequest(BaseModel):
    """Request to update a workflow."""

    workflow_id: str
    graph: WorkflowGraph


class ValidationResponse(BaseModel):
    """Validation result."""

    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class CompileRequest(BaseModel):
    """Request to compile a workflow."""

    yaml_content: str
    workflow_name: str


class CompileResponse(BaseModel):
    """Compilation result."""

    success: bool
    workflow_id: str
    graph_schema: Dict[str, Any]
    errors: List[str] = Field(default_factory=list)


class TeamMemberConfig(BaseModel):
    """Team member configuration."""

    id: str
    role: str  # researcher, planner, executor, reviewer, writer, assistant
    goal: str
    tool_budget: int = 25
    tools: Optional[List[str]] = None
    backstory: Optional[str] = None
    expertise: Optional[List[str]] = None
    personality: Optional[str] = None


class TeamNodeConfig(BaseModel):
    """Team node configuration."""

    id: str
    name: str
    goal: str
    formation: str  # parallel, sequential, pipeline, hierarchical, consensus
    max_iterations: int = 5
    timeout_seconds: Optional[int] = None
    members: List[TeamMemberConfig]


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/")
async def root() -> dict[str, Any]:
    """Root endpoint."""
    return {
        "name": "Victor Workflow Editor API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/workflows/validate", response_model=ValidationResponse)
async def validate_workflow_graph(graph: WorkflowGraph) -> ValidationResponse:
    """Validate a workflow graph.

    Args:
        graph: Workflow graph to validate

    Returns:
        Validation response with errors and warnings
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Basic validation
    if not graph.nodes:
        errors.append("Workflow must have at least one node")

    # Check for duplicate node IDs
    node_ids = [node.id for node in graph.nodes]
    duplicates = [nid for nid in set(node_ids) if node_ids.count(nid) > 1]
    if duplicates:
        errors.append(f"Duplicate node IDs: {', '.join(duplicates)}")

    # Validate edges
    node_ids_set = set(node.id for node in graph.nodes)
    for edge in graph.edges:
        if edge.source not in node_ids_set:
            errors.append(f"Edge source '{edge.source}' not found in nodes")
        if edge.target not in node_ids_set:
            errors.append(f"Edge target '{edge.target}' not found in nodes")

    # Validate node configurations
    for node in graph.nodes:
        if node.type == "agent":
            if "role" not in node.config and "goal" not in node.config:
                errors.append(f"Agent node '{node.id}' must have 'role' and 'goal'")

        elif node.type == "team":
            if "members" not in node.config or not node.config["members"]:
                errors.append(f"Team node '{node.id}' must have at least one member")
            if "formation" not in node.config:
                errors.append(f"Team node '{node.id}' must specify 'formation'")

        elif node.type == "condition":
            if "condition" not in node.config and "branches" not in node.config:
                errors.append(
                    f"Condition node '{node.id}' must have 'condition' and 'branches'"
                )

    return ValidationResponse(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


@app.post("/api/workflows/compile", response_model=CompileResponse)
async def compile_workflow(request: CompileRequest) -> CompileResponse:
    """Compile a workflow from YAML content.

    Args:
        request: Compilation request with YAML content

    Returns:
        Compilation result with graph schema
    """
    try:
        compiler = get_compiler()

        # Compile from YAML content
        cached_graph = compiler.compile_yaml_content(
            yaml_content=request.yaml_content,
            workflow_name=request.workflow_name,
        )

        # Get graph schema
        graph_schema = cached_graph.get_graph_schema()

        return CompileResponse(
            success=True,
            workflow_id=request.workflow_name,
            graph_schema=graph_schema,
        )

    except Exception as e:
        logger.error(f"Compilation failed: {e}", exc_info=True)
        return CompileResponse(
            success=False,
            workflow_id=request.workflow_name,
            graph_schema={},
            errors=[str(e)],
        )


@app.post("/api/workflows/export/yaml")
async def export_to_yaml(graph: WorkflowGraph) -> dict[str, str] | dict[str, bool]:
    """Export workflow graph to YAML format.

    Args:
        graph: Workflow graph to export

    Returns:
        YAML content as string
    """
    try:
        # Convert graph to workflow definition
        workflow_def = graph_to_definition(graph)

        # Export to YAML - create YAML content directly
        import yaml
        yaml_content = yaml.dump({
            "name": workflow_def.name,
            "nodes": {node_id: node.__dict__ for node_id, node in workflow_def.nodes.items()},
            "start_node": workflow_def.start_node,
        }, default_flow_style=False)

        return {"yaml_content": yaml_content, "success": True}

    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workflows/import/yaml")
async def import_from_yaml(file: UploadFile = File(...)) -> dict[str, Any]:
    """Import workflow from YAML file.

    Args:
        file: Uploaded YAML file

    Returns:
        Workflow graph
    """
    try:
        content = await file.read()
        yaml_content = content.decode("utf-8")

        # Load workflow from YAML
        config = YAMLWorkflowConfig()
        result = load_workflow_from_yaml(yaml_content, "imported", config)

        # Convert to graph
        if isinstance(result, dict):
            workflow_def = next(iter(result.values()))
        else:
            workflow_def = result

        graph = definition_to_graph(workflow_def)

        return {"graph": graph, "success": True}

    except Exception as e:
        logger.error(f"Import failed: {e}", exc_info=True)
        return {"error": str(e), "success": False}


@app.get("/api/nodes/types")
async def get_node_types() -> dict[str, dict[str, Any]]:
    """Get available node types and their configurations.

    Returns:
        Dictionary of node types with schemas
    """
    return {
        "agent": {
            "name": "Agent Node",
            "description": "LLM-powered agent with role and goal",
            "color": "#E3F2FD",
            "config_schema": {
                "role": {"type": "string", "required": True},
                "goal": {"type": "string", "required": True},
                "tool_budget": {"type": "integer", "default": 25},
                "tools": {"type": "array", "items": "string"},
                "llm_config": {
                    "type": "object",
                    "properties": {
                        "temperature": {"type": "number", "minimum": 0, "maximum": 1}
                    },
                },
            },
        },
        "compute": {
            "name": "Compute Node",
            "description": "Execute tools without LLM inference",
            "color": "#E8F5E9",
            "config_schema": {
                "handler": {"type": "string"},
                "tools": {"type": "array", "items": "string"},
                "inputs": {"type": "object"},
                "output": {"type": "string"},
                "timeout": {"type": "integer", "default": 60},
            },
        },
        "team": {
            "name": "Team Node",
            "description": "Multi-agent team with configurable formation",
            "color": "#F3E5F5",
            "config_schema": {
                "formation": {
                    "type": "string",
                    "enum": ["parallel", "sequential", "pipeline", "hierarchical", "consensus"],
                    "required": True,
                },
                "goal": {"type": "string", "required": True},
                "max_iterations": {"type": "integer", "default": 5},
                "timeout_seconds": {"type": "integer"},
                "members": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "role": {"type": "string"},
                            "goal": {"type": "string"},
                            "tool_budget": {"type": "integer"},
                            "tools": {"type": "array", "items": "string"},
                            "backstory": {"type": "string"},
                            "expertise": {"type": "array", "items": "string"},
                        },
                    },
                },
            },
        },
        "condition": {
            "name": "Condition Node",
            "description": "Branching logic based on state",
            "color": "#FFF3E0",
            "config_schema": {
                "condition": {"type": "string", "required": True},
                "branches": {"type": "object", "required": True},
            },
        },
        "parallel": {
            "name": "Parallel Node",
            "description": "Execute multiple branches concurrently",
            "color": "#F3E5F5",
            "config_schema": {
                "parallel_nodes": {"type": "array", "items": "string"},
                "join_strategy": {"type": "string", "enum": ["all", "any"], "default": "all"},
            },
        },
        "transform": {
            "name": "Transform Node",
            "description": "State transformation",
            "color": "#ECEFF1",
            "config_schema": {
                "transform": {"type": "string", "required": True},
            },
        },
        "hitl": {
            "name": "Human-in-the-Loop Node",
            "description": "Human interaction point",
            "color": "#FFEBEE",
            "config_schema": {
                "hitl_type": {"type": "string", "enum": ["approval", "input", "review"]},
                "prompt": {"type": "string", "required": True},
                "timeout": {"type": "integer", "default": 300},
            },
        },
    }


@app.get("/api/formations")
async def get_formations() -> dict[str, Any]:
    """Get available team formation types.

    Returns:
        Dictionary of formation types with descriptions
    """
    return {
        "parallel": {
            "name": "Parallel Formation",
            "description": "All members work simultaneously on the task",
            "icon": "||",
            "best_for": ["Independent analysis", "Multi-perspective review", "Diverse tasks"],
            "communication_style": "structured",
        },
        "sequential": {
            "name": "Sequential Formation",
            "description": "Members work in sequence, each building on the previous",
            "icon": "→",
            "best_for": ["Step-by-step processing", "Refinement loops", "Drafting cycles"],
            "communication_style": "pass_through",
        },
        "pipeline": {
            "name": "Pipeline Formation",
            "description": "Output passes through stages like an assembly line",
            "icon": "⇒",
            "best_for": ["Multi-stage processing", "Specialized tasks", "Production workflows"],
            "communication_style": "structured",
        },
        "hierarchical": {
            "name": "Hierarchical Formation",
            "description": "Manager coordinates worker agents",
            "icon": "⬗",
            "best_for": ["Complex coordination", "Task delegation", "Manager-worker patterns"],
            "communication_style": "coordinated",
        },
        "consensus": {
            "name": "Consensus Formation",
            "description": "Members vote on decisions",
            "icon": "◊",
            "best_for": ["Decision making", "Review processes", "Quality assurance"],
            "communication_style": "peer_to_peer",
        },
    }


# =============================================================================
# Utility Functions
# =============================================================================


def graph_to_definition(graph: WorkflowGraph) -> WorkflowDefinition:
    """Convert workflow graph to WorkflowDefinition.

    Args:
        graph: Workflow graph from editor

    Returns:
        WorkflowDefinition for compilation
    """
    # This is a simplified conversion - the real implementation would
    # need to handle all node types and their configurations
    from victor.workflows.definition import (
        AgentNode,
        ComputeNode,
        WorkflowDefinition,
        WorkflowNode,
    )

    # WorkflowDefinition expects nodes as Dict[str, WorkflowNode]
    nodes_dict: dict[str, WorkflowNode] = {}
    first_node_id: str | None = None

    for node in graph.nodes:
        if first_node_id is None:
            first_node_id = node.id

        if node.type == "agent":
            # AgentNode is a subclass of WorkflowNode, so this is safe
            agent_node: WorkflowNode = AgentNode(  # type: ignore[assignment]
                id=node.id,
                name=node.name,
                role=node.config.get("role", "assistant"),
                goal=node.config.get("goal", ""),
                tool_budget=node.config.get("tool_budget", 25),
            )
            nodes_dict[node.id] = agent_node
        # Add other node types...

    return WorkflowDefinition(
        name=graph.metadata.get("name", "workflow"),
        nodes=nodes_dict,
        start_node=first_node_id,
    )


def definition_to_graph(workflow_def: WorkflowDefinition) -> WorkflowGraph:
    """Convert WorkflowDefinition to workflow graph.

    Args:
        workflow_def: WorkflowDefinition from compiler

    Returns:
        Workflow graph for editor
    """
    # Simplified conversion
    nodes: list[WorkflowNode] = []
    edges: list[WorkflowEdge] = []

    # workflow_def.nodes is Dict[str, WorkflowNode]
    for node_id, node in workflow_def.nodes.items():
        nodes.append(
            WorkflowNode(
                id=node.id,
                type=node.__class__.__name__,
                name=node.name,
                config={},  # Extract config
                position={"x": 0, "y": 0},
            )
        )

    return WorkflowGraph(nodes=nodes, edges=edges)


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    """Run the API server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    main()
