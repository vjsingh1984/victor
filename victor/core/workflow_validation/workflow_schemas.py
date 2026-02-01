"""Pydantic schemas for workflow definition validation.

This module defines Pydantic schemas for validating workflow definitions.
These schemas provide type-safe validation with clear error messages
for workflow YAML/JSON definitions.

Schema Layers:
1. WorkflowNodeSchema - Individual node validation
2. WorkflowEdgeSchema - Edge/connection validation
3. WorkflowDefinitionSchema - Complete workflow validation

Example:
    from victor.core.validation.workflow_schemas import WorkflowDefinitionSchema

    # Validate from dict
    workflow = WorkflowDefinitionSchema(**data)

    # Validate from YAML
    import yaml
    with open("workflow.yaml") as f:
        data = yaml.safe_load(f)
    workflow = WorkflowDefinitionSchema(**data)
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Any, Literal
from enum import Enum


class NodeKind(str, Enum):
    """Valid node types for workflows."""

    AGENT = "agent"
    COMPUTE = "compute"
    CONDITION = "condition"
    PARALLEL = "parallel"
    TRANSFORM = "transform"
    TEAM = "team"
    HITL = "hitl"


class WorkflowNodeSchema(BaseModel):
    """Schema for individual workflow nodes.

    Validates node structure including required fields,
    type-specific fields, and constraints.

    Attributes:
        id: Unique node identifier (alphanumeric with underscores/hyphens)
        name: Human-readable node name
        type: Node type (agent, compute, condition, parallel, transform, team, hitl)
        description: Optional node description
        next_nodes: List of successor node IDs
        timeout: Maximum execution time in seconds
        max_retries: Maximum retry attempts
        role: Agent role (for agent nodes)
        goal: Agent's task goal (for agent nodes)
        tool_budget: Maximum tool calls (for agent nodes)
        tools: Specific tools to use (for agent/compute nodes)
        handler: Compute handler name (for compute nodes)
        condition: Condition function name (for condition nodes)
        branches: Branch mappings for condition nodes
        parallel_nodes: List of parallel node IDs (for parallel nodes)
        join_strategy: How to join parallel branches
        transform: Transform function name (for transform nodes)
        members: Team member roles (for team nodes)
        team_formation: Team coordination pattern
        metadata: Additional node metadata
    """

    id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique node identifier (alphanumeric with underscores/hyphens)",
    )
    name: str = Field(..., min_length=1, max_length=200, description="Human-readable node name")
    type: NodeKind = Field(..., description="Node type determining execution behavior")
    description: Optional[str] = Field(
        None, max_length=500, description="Optional node description"
    )

    # Common fields
    next_nodes: list[str] = Field(default_factory=list, description="Successor node IDs")
    timeout: Optional[int] = Field(
        None, ge=0, le=3600, description="Node timeout in seconds (max 1 hour)"
    )
    max_retries: Optional[int] = Field(
        None, ge=0, le=10, description="Maximum retry attempts on failure"
    )

    # Agent node fields
    role: Optional[str] = Field(
        None,
        description="Agent role (researcher, planner, executor, reviewer, writer, analyst, coder)",
    )
    goal: Optional[str] = Field(None, description="Agent's task goal or objective")
    tool_budget: Optional[int] = Field(
        None, ge=0, le=500, description="Tool call budget for this node"
    )
    tools: Optional[list[str]] = Field(
        None, description="Specific tools to use (overrides default tool selection)"
    )

    # Compute node fields
    handler: Optional[str] = Field(None, description="Registered compute handler name")

    # Condition node fields
    condition: Optional[str] = Field(None, description="Condition function name for branching")
    branches: Optional[dict[str, str]] = Field(
        None, description="Branch mappings (condition_value -> node_id)"
    )

    # Parallel node fields
    parallel_nodes: Optional[list[str]] = Field(
        None, description="List of parallel node IDs to execute"
    )
    join_strategy: Optional[Literal["all", "any", "merge"]] = Field(
        None, description="How to join parallel branch results"
    )

    # Transform node fields
    transform: Optional[str] = Field(None, description="Transform function name")

    # Team node fields
    members: Optional[list[str]] = Field(None, description="Team member roles")
    team_formation: Optional[
        Literal["sequential", "parallel", "hierarchical", "pipeline", "consensus"]
    ] = Field(None, description="Team coordination pattern")

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional node metadata")

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate node ID format.

        Node IDs must be alphanumeric with underscores and hyphens allowed.
        This ensures compatibility with various systems and prevents issues
        with special characters.

        Args:
            v: The node ID to validate

        Returns:
            The validated node ID

        Raises:
            ValueError: If the ID contains invalid characters
        """
        import re

        if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
            raise ValueError(
                "Node ID must be alphanumeric (underscores and hyphens allowed). " f"Got: '{v}'"
            )
        return v

    @field_validator("branches")
    @classmethod
    def validate_branches(cls, v: Optional[dict[str, str]]) -> Optional[dict[str, str]]:
        """Validate branch target format.

        Branch target node IDs must follow the same format as node IDs.

        Args:
            v: The branches dict to validate

        Returns:
            The validated branches dict

        Raises:
            ValueError: If any branch target contains invalid characters
        """
        if v:
            import re

            for branch_key, branch_val in v.items():
                if not isinstance(branch_val, str):
                    continue  # type: ignore[unreachable]
                if not re.match(r"^[a-zA-Z0-9_\-]+$", branch_val):
                    # Allow __end__ as a special terminal marker
                    if branch_val != "__end__":
                        raise ValueError(
                            f"Branch target '{branch_val}' must be alphanumeric "
                            "(underscores and hyphens allowed, or __end__ for terminal)"
                        )
        return v


class WorkflowEdgeSchema(BaseModel):
    """Schema for workflow edges/connections.

    Represents a directed edge from a source node to a target node.
    Edges define the flow of execution through the workflow.

    Attributes:
        source: Source node ID
        target: Target node ID (or "__end__" for terminal edges)
        label: Optional edge label for visualization
        conditional: Whether this edge represents a conditional branch
        condition: Condition expression (if conditional)
    """

    source: str = Field(..., min_length=1, description="Source node ID")
    target: str = Field(..., description="Target node ID (or __end__ for terminal)")
    label: Optional[str] = Field(None, description="Optional edge label for visualization")
    conditional: bool = Field(default=False, description="Whether this edge is conditional")
    condition: Optional[str] = Field(
        None, description="Condition expression (if edge is conditional)"
    )


class WorkflowDefinitionSchema(BaseModel):
    """Root schema for workflow definitions.

    Validates the complete workflow structure including nodes, edges,
    and workflow-level configuration.

    Attributes:
        name: Workflow name (must be unique within a collection)
        description: Optional workflow description
        entry_point: ID of the starting node
        nodes: List of workflow nodes (must have at least one)
        edges: List of workflow edges
        max_iterations: Maximum iterations for loops
        max_timeout_seconds: Total workflow timeout
        metadata: Additional workflow metadata

    Example:
        workflow = WorkflowDefinitionSchema(
            name="my_workflow",
            description="A simple workflow",
            entry_point="start",
            nodes=[
                WorkflowNodeSchema(
                    id="start",
                    name="Start",
                    type="agent",
                    role="executor",
                    goal="Do something"
                )
            ]
        )
    """

    name: str = Field(
        ..., min_length=1, max_length=100, description="Workflow name (unique identifier)"
    )
    description: Optional[str] = Field(None, max_length=500, description="Workflow description")
    entry_point: str = Field(
        ..., min_length=1, description="Entry point node ID (must exist in nodes)"
    )
    nodes: list[WorkflowNodeSchema] = Field(
        ..., min_length=1, description="Workflow nodes (must have at least one)"
    )
    edges: list[WorkflowEdgeSchema] = Field(
        default_factory=list, description="Workflow edges defining execution flow"
    )

    # Workflow-level configuration
    max_iterations: Optional[int] = Field(
        None, ge=1, le=500, description="Maximum iterations for loop detection"
    )
    max_timeout_seconds: Optional[float] = Field(
        None, ge=0, description="Total workflow timeout in seconds"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional workflow metadata"
    )

    @field_validator("nodes")
    @classmethod
    def validate_node_ids_unique(cls, v: list[WorkflowNodeSchema]) -> list[WorkflowNodeSchema]:
        """Ensure all node IDs are unique.

        Duplicate node IDs would cause ambiguity in execution and
        graph structure validation.

        Args:
            v: List of nodes to validate

        Returns:
            The validated list of nodes

        Raises:
            ValueError: If duplicate node IDs are found
        """
        ids = [node.id for node in v]
        if len(ids) != len(set(ids)):
            duplicates = [node_id for node_id in ids if ids.count(node_id) > 1]
            raise ValueError(
                f"Duplicate node IDs found: {set(duplicates)}. " "Each node must have a unique ID."
            )
        return v

    @field_validator("entry_point")
    @classmethod
    def validate_entry_point_exists(cls, v: str, info: Any) -> str:
        """Ensure entry point references an existing node.

        The entry point must be a valid node ID in the workflow.

        Args:
            v: The entry point node ID
            info: Pydantic validation info containing node list

        Returns:
            The validated entry point

        Raises:
            ValueError: If entry point doesn't reference a valid node
        """
        # During initial validation, nodes might not be available yet
        # This validator runs after nodes validator, so we can check
        if hasattr(info, "data") and "nodes" in info.data:
            node_ids = {node.id for node in info.data["nodes"]}
            if v not in node_ids:
                raise ValueError(
                    f"Entry point '{v}' not found in node definitions. "
                    f"Available nodes: {sorted(node_ids)}"
                )
        return v

    def get_node_by_id(self, node_id: str) -> Optional[WorkflowNodeSchema]:
        """Get a node by its ID.

        Args:
            node_id: The node ID to look up

        Returns:
            The node if found, None otherwise
        """
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_node_ids(self) -> set[str]:
        """Get all node IDs.

        Returns:
            Set of all node IDs
        """
        return {node.id for node in self.nodes}

    def to_dict(self) -> dict[str, Any]:
        """Convert workflow to dictionary representation.

        Returns:
            Dictionary representation of the workflow
        """
        return self.model_dump(mode="json")

    def to_yaml(self) -> str:
        """Convert workflow to YAML string.

        Returns:
            YAML representation of the workflow
        """
        import yaml

        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)


# Convenience functions for common workflows


def create_simple_agent_workflow(
    name: str,
    goal: str,
    role: str = "executor",
    tool_budget: int = 15,
    description: Optional[str] = None,
) -> WorkflowDefinitionSchema:
    """Create a simple single-agent workflow.

    Args:
        name: Workflow name
        goal: Agent's task goal
        role: Agent role (default: executor)
        tool_budget: Tool call budget (default: 15)
        description: Optional workflow description

    Returns:
        A workflow definition with a single agent node
    """
    return WorkflowDefinitionSchema(
        name=name,
        description=description,
        entry_point="agent",
        max_iterations=None,
        max_timeout_seconds=None,
        nodes=[
            WorkflowNodeSchema(
                id="agent",
                name="Agent",
                type=NodeKind.AGENT,
                description=None,
                timeout=None,
                max_retries=None,
                tools=None,
                handler=None,
                condition=None,
                branches=None,
                parallel_nodes=None,
                join_strategy=None,
                transform=None,
                members=None,
                team_formation=None,
                role=role,
                goal=goal,
                tool_budget=tool_budget,
            )
        ],
    )


def validate_workflow_dict(data: dict[str, Any]) -> WorkflowDefinitionSchema:
    """Validate a workflow from dictionary data.

    Convenience function that creates a WorkflowDefinitionSchema
    from dict data, raising ValidationError if invalid.

    Args:
        data: Dictionary containing workflow definition

    Returns:
        Validated workflow definition

    Raises:
        ValidationError: If the workflow definition is invalid
    """
    return WorkflowDefinitionSchema(**data)
