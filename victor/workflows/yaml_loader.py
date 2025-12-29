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

"""YAML loader for declarative workflow definitions.

Enables defining workflows in YAML format for easier configuration and
version control without code changes.

Example YAML format:
    workflows:
      feature_implementation:
        description: "End-to-end feature development with review"
        metadata:
          version: "1.0"
          author: "team"
        nodes:
          - id: research
            type: agent
            role: researcher
            goal: "Analyze codebase for relevant patterns"
            tool_budget: 20
            tools: [read, grep, code_search, overview]
            output: research_findings
            next: [plan]
          - id: plan
            type: agent
            role: planner
            goal: "Create implementation plan"
            tool_budget: 10
            next: [decide]
          - id: decide
            type: condition
            condition: "has_tests"  # Simple key check
            branches:
              true: implement
              false: add_tests
          - id: implement
            type: agent
            role: executor
            goal: "Implement the feature"
            tool_budget: 30
          - id: review
            type: hitl
            hitl_type: approval
            prompt: "Review the implementation?"
            timeout: 300
            fallback: continue
"""

from __future__ import annotations

import logging
import operator
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml

from victor.workflows.definition import (
    AgentNode,
    ConditionNode,
    ParallelNode,
    TransformNode,
    WorkflowDefinition,
    WorkflowNode,
)

logger = logging.getLogger(__name__)


class YAMLWorkflowError(Exception):
    """Error loading or parsing YAML workflow."""

    pass


@dataclass
class YAMLWorkflowConfig:
    """Configuration for YAML workflow loading."""

    # Allow unsafe conditions (arbitrary Python expressions)
    allow_unsafe_conditions: bool = False
    # Base directory for relative imports
    base_dir: Optional[Path] = None
    # Custom condition functions
    condition_registry: Dict[str, Callable[[Dict[str, Any]], str]] = None
    # Custom transform functions
    transform_registry: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = None

    def __post_init__(self):
        if self.condition_registry is None:
            self.condition_registry = {}
        if self.transform_registry is None:
            self.transform_registry = {}


def _create_simple_condition(expr: str) -> Callable[[Dict[str, Any]], str]:
    """Create a condition function from a simple expression.

    Supported expressions:
    - "key" - check if key exists and is truthy
    - "key == value" - check equality
    - "key != value" - check inequality
    - "key > value" - numeric comparison
    - "key >= value" - numeric comparison
    - "key < value" - numeric comparison
    - "key <= value" - numeric comparison
    - "key in [a, b, c]" - check membership

    Returns:
        Function that evaluates the expression and returns "true" or "false"
    """
    expr = expr.strip()

    # Comparison operators
    comparisons = [
        (r"^(\w+)\s*==\s*(.+)$", operator.eq),
        (r"^(\w+)\s*!=\s*(.+)$", operator.ne),
        (r"^(\w+)\s*>=\s*(.+)$", operator.ge),
        (r"^(\w+)\s*<=\s*(.+)$", operator.le),
        (r"^(\w+)\s*>\s*(.+)$", operator.gt),
        (r"^(\w+)\s*<\s*(.+)$", operator.lt),
    ]

    for pattern, op in comparisons:
        match = re.match(pattern, expr)
        if match:
            key = match.group(1)
            value_str = match.group(2).strip()
            # Parse value
            value = _parse_value(value_str)

            def condition(ctx: Dict[str, Any], k=key, v=value, o=op) -> str:
                ctx_value = ctx.get(k)
                try:
                    return "true" if o(ctx_value, v) else "false"
                except TypeError:
                    return "false"

            return condition

    # Check for "in" operator
    in_match = re.match(r"^(\w+)\s+in\s+\[(.+)\]$", expr)
    if in_match:
        key = in_match.group(1)
        values_str = in_match.group(2)
        values = [_parse_value(v.strip()) for v in values_str.split(",")]

        def in_condition(ctx: Dict[str, Any], k=key, vs=values) -> str:
            return "true" if ctx.get(k) in vs else "false"

        return in_condition

    # Simple truthy check
    if re.match(r"^\w+$", expr):

        def truthy_condition(ctx: Dict[str, Any], k=expr) -> str:
            return "true" if ctx.get(k) else "false"

        return truthy_condition

    # Default: always return "default"
    return lambda ctx: "default"


def _parse_value(value_str: str) -> Any:
    """Parse a string value into appropriate Python type."""
    value_str = value_str.strip()

    # Handle quoted strings
    if (value_str.startswith('"') and value_str.endswith('"')) or (
        value_str.startswith("'") and value_str.endswith("'")
    ):
        return value_str[1:-1]

    # Handle booleans
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    if value_str.lower() == "none":
        return None

    # Handle numbers
    try:
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass

    # Return as string
    return value_str


def _create_transform(expr: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Create a transform function from a simple expression.

    Supported expressions:
    - "key = value" - set a key to a literal value
    - "key = ctx.other_key" - copy from another key
    - "key = merge(a, b)" - merge two dicts

    Returns:
        Function that transforms context
    """
    expr = expr.strip()

    # Assignment: key = value
    assign_match = re.match(r"^(\w+)\s*=\s*(.+)$", expr)
    if assign_match:
        key = assign_match.group(1)
        value_str = assign_match.group(2).strip()

        # Reference to context key
        if value_str.startswith("ctx."):
            ref_key = value_str[4:]

            def ref_transform(ctx: Dict[str, Any], k=key, rk=ref_key) -> Dict[str, Any]:
                result = ctx.copy()
                result[k] = ctx.get(rk)
                return result

            return ref_transform

        # Literal value
        value = _parse_value(value_str)

        def literal_transform(ctx: Dict[str, Any], k=key, v=value) -> Dict[str, Any]:
            result = ctx.copy()
            result[k] = v
            return result

        return literal_transform

    # Default: identity transform
    return lambda ctx: ctx


def _parse_agent_node(node_data: Dict[str, Any]) -> AgentNode:
    """Parse an agent node from YAML data."""
    node_id = node_data["id"]
    return AgentNode(
        id=node_id,
        name=node_data.get("name", node_id),
        role=node_data.get("role", "executor"),
        goal=node_data.get("goal", ""),
        tool_budget=node_data.get("tool_budget", 15),
        allowed_tools=node_data.get("tools"),
        input_mapping=node_data.get("input_mapping", {}),
        output_key=node_data.get("output", node_id),
        next_nodes=node_data.get("next", []),
    )


def _parse_condition_node(
    node_data: Dict[str, Any],
    config: YAMLWorkflowConfig,
) -> ConditionNode:
    """Parse a condition node from YAML data."""
    node_id = node_data["id"]
    condition_expr = node_data.get("condition", "default")

    # Check for registered condition
    if condition_expr in config.condition_registry:
        condition_fn = config.condition_registry[condition_expr]
    else:
        condition_fn = _create_simple_condition(condition_expr)

    return ConditionNode(
        id=node_id,
        name=node_data.get("name", node_id),
        condition=condition_fn,
        branches=node_data.get("branches", {}),
        next_nodes=node_data.get("next", []),
    )


def _parse_parallel_node(node_data: Dict[str, Any]) -> ParallelNode:
    """Parse a parallel node from YAML data."""
    node_id = node_data["id"]
    return ParallelNode(
        id=node_id,
        name=node_data.get("name", node_id),
        parallel_nodes=node_data.get("parallel_nodes", []),
        join_strategy=node_data.get("join_strategy", "all"),
        next_nodes=node_data.get("next", []),
    )


def _parse_transform_node(
    node_data: Dict[str, Any],
    config: YAMLWorkflowConfig,
) -> TransformNode:
    """Parse a transform node from YAML data."""
    node_id = node_data["id"]
    transform_expr = node_data.get("transform", "")

    # Check for registered transform
    if transform_expr in config.transform_registry:
        transform_fn = config.transform_registry[transform_expr]
    else:
        transform_fn = _create_transform(transform_expr)

    return TransformNode(
        id=node_id,
        name=node_data.get("name", node_id),
        transform=transform_fn,
        next_nodes=node_data.get("next", []),
    )


def _parse_hitl_node(node_data: Dict[str, Any]) -> WorkflowNode:
    """Parse a HITL node from YAML data."""
    from victor.workflows.hitl import HITLFallback, HITLNode, HITLNodeType

    node_id = node_data["id"]
    hitl_type_str = node_data.get("hitl_type", "approval")
    hitl_type = HITLNodeType(hitl_type_str)

    fallback_str = node_data.get("fallback", "abort")
    fallback = HITLFallback(fallback_str)

    return HITLNode(
        id=node_id,
        name=node_data.get("name", node_id),
        hitl_type=hitl_type,
        prompt=node_data.get("prompt", ""),
        context_keys=node_data.get("context_keys", []),
        choices=node_data.get("choices", []),
        default_value=node_data.get("default_value"),
        timeout=node_data.get("timeout", 300.0),
        fallback=fallback,
        next_nodes=node_data.get("next", []),
    )


def _parse_node(
    node_data: Dict[str, Any],
    config: YAMLWorkflowConfig,
) -> WorkflowNode:
    """Parse a workflow node from YAML data."""
    node_type = node_data.get("type", "agent")

    if node_type == "agent":
        return _parse_agent_node(node_data)
    elif node_type == "condition":
        return _parse_condition_node(node_data, config)
    elif node_type == "parallel":
        return _parse_parallel_node(node_data)
    elif node_type == "transform":
        return _parse_transform_node(node_data, config)
    elif node_type == "hitl":
        return _parse_hitl_node(node_data)
    else:
        raise YAMLWorkflowError(f"Unknown node type: {node_type}")


def load_workflow_from_dict(
    data: Dict[str, Any],
    name: str,
    config: Optional[YAMLWorkflowConfig] = None,
) -> WorkflowDefinition:
    """Load a workflow definition from a dictionary.

    Args:
        data: Dictionary containing workflow definition
        name: Name for the workflow
        config: Optional loader configuration

    Returns:
        WorkflowDefinition instance
    """
    config = config or YAMLWorkflowConfig()

    nodes: Dict[str, WorkflowNode] = {}
    node_list = data.get("nodes", [])

    for node_data in node_list:
        if "id" not in node_data:
            raise YAMLWorkflowError("Node missing required 'id' field")

        node = _parse_node(node_data, config)
        nodes[node.id] = node

    # Auto-chain sequential nodes (like WorkflowBuilder)
    # Connect nodes that don't have explicit "next" to the following node
    for i, node_data in enumerate(node_list[:-1]):
        node_id = node_data["id"]
        node = nodes[node_id]
        # Skip condition nodes (they use branches)
        if isinstance(node, ConditionNode):
            continue
        # Auto-chain if no explicit next_nodes
        if not node.next_nodes:
            next_node_id = node_list[i + 1]["id"]
            node.next_nodes.append(next_node_id)

    # Determine start node
    start_node = data.get("start_node")
    if not start_node and node_list:
        start_node = node_list[0]["id"]

    workflow = WorkflowDefinition(
        name=name,
        description=data.get("description", ""),
        nodes=nodes,
        start_node=start_node,
        metadata=data.get("metadata", {}),
    )

    # Validate
    errors = workflow.validate()
    if errors:
        raise YAMLWorkflowError(f"Workflow validation failed: {'; '.join(errors)}")

    return workflow


def load_workflow_from_yaml(
    yaml_content: str,
    workflow_name: Optional[str] = None,
    config: Optional[YAMLWorkflowConfig] = None,
) -> Union[WorkflowDefinition, Dict[str, WorkflowDefinition]]:
    """Load workflow(s) from YAML content.

    Args:
        yaml_content: YAML string content
        workflow_name: Optional specific workflow to load
        config: Optional loader configuration

    Returns:
        Single WorkflowDefinition if workflow_name specified,
        otherwise Dict of all workflows
    """
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        raise YAMLWorkflowError(f"Invalid YAML: {e}")

    if not isinstance(data, dict):
        raise YAMLWorkflowError("YAML must contain a dictionary")

    # Check for workflows key
    workflows_data = data.get("workflows", data)

    if workflow_name:
        if workflow_name not in workflows_data:
            raise YAMLWorkflowError(f"Workflow '{workflow_name}' not found")
        return load_workflow_from_dict(
            workflows_data[workflow_name], workflow_name, config
        )

    # Load all workflows
    workflows = {}
    for name, wf_data in workflows_data.items():
        if isinstance(wf_data, dict) and "nodes" in wf_data:
            workflows[name] = load_workflow_from_dict(wf_data, name, config)

    return workflows


def load_workflow_from_file(
    file_path: Union[str, Path],
    workflow_name: Optional[str] = None,
    config: Optional[YAMLWorkflowConfig] = None,
) -> Union[WorkflowDefinition, Dict[str, WorkflowDefinition]]:
    """Load workflow(s) from a YAML file.

    Args:
        file_path: Path to YAML file
        workflow_name: Optional specific workflow to load
        config: Optional loader configuration

    Returns:
        Single WorkflowDefinition if workflow_name specified,
        otherwise Dict of all workflows
    """
    path = Path(file_path)
    if not path.exists():
        raise YAMLWorkflowError(f"File not found: {path}")

    # Set base_dir from file location if not set
    if config is None:
        config = YAMLWorkflowConfig(base_dir=path.parent)
    elif config.base_dir is None:
        config.base_dir = path.parent

    yaml_content = path.read_text()
    return load_workflow_from_yaml(yaml_content, workflow_name, config)


def load_workflows_from_directory(
    directory: Union[str, Path],
    pattern: str = "*.yaml",
    config: Optional[YAMLWorkflowConfig] = None,
) -> Dict[str, WorkflowDefinition]:
    """Load all workflows from YAML files in a directory.

    Args:
        directory: Directory to scan
        pattern: Glob pattern for files (default: *.yaml)
        config: Optional loader configuration

    Returns:
        Dict mapping workflow names to definitions
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise YAMLWorkflowError(f"Not a directory: {dir_path}")

    workflows = {}
    for yaml_file in dir_path.glob(pattern):
        try:
            file_workflows = load_workflow_from_file(yaml_file, config=config)
            if isinstance(file_workflows, dict):
                workflows.update(file_workflows)
            else:
                workflows[file_workflows.name] = file_workflows
        except YAMLWorkflowError as e:
            logger.warning(f"Failed to load {yaml_file}: {e}")

    return workflows


class YAMLWorkflowProvider:
    """Workflow provider that loads workflows from YAML files.

    Can be used as a workflow provider for verticals by loading
    workflow definitions from a YAML file or directory.

    Example:
        provider = YAMLWorkflowProvider.from_file("workflows.yaml")
        workflow = provider.get_workflow("code_review")
    """

    def __init__(
        self,
        workflows: Dict[str, WorkflowDefinition],
        auto_workflows: Optional[List[tuple]] = None,
    ):
        """Initialize provider with pre-loaded workflows.

        Args:
            workflows: Dict of workflow name to definition
            auto_workflows: List of (pattern, workflow_name) for auto-selection
        """
        self._workflows = workflows
        self._auto_workflows = auto_workflows or []

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        config: Optional[YAMLWorkflowConfig] = None,
    ) -> "YAMLWorkflowProvider":
        """Create provider from a YAML file.

        Args:
            file_path: Path to YAML file
            config: Optional loader configuration

        Returns:
            YAMLWorkflowProvider instance
        """
        workflows = load_workflow_from_file(file_path, config=config)
        if not isinstance(workflows, dict):
            workflows = {workflows.name: workflows}
        return cls(workflows)

    @classmethod
    def from_directory(
        cls,
        directory: Union[str, Path],
        pattern: str = "*.yaml",
        config: Optional[YAMLWorkflowConfig] = None,
    ) -> "YAMLWorkflowProvider":
        """Create provider from YAML files in a directory.

        Args:
            directory: Directory to scan
            pattern: Glob pattern for files
            config: Optional loader configuration

        Returns:
            YAMLWorkflowProvider instance
        """
        workflows = load_workflows_from_directory(directory, pattern, config)
        return cls(workflows)

    def get_workflows(self) -> Dict[str, type]:
        """Get all available workflow factories."""
        return {name: type(wf) for name, wf in self._workflows.items()}

    def get_workflow(self, name: str) -> Optional[WorkflowDefinition]:
        """Get a specific workflow by name."""
        return self._workflows.get(name)

    def get_auto_workflows(self) -> List[tuple]:
        """Get auto-selection workflow mappings."""
        return self._auto_workflows.copy()

    def list_workflows(self) -> List[str]:
        """List all available workflow names."""
        return list(self._workflows.keys())


__all__ = [
    "YAMLWorkflowError",
    "YAMLWorkflowConfig",
    "YAMLWorkflowProvider",
    "load_workflow_from_dict",
    "load_workflow_from_yaml",
    "load_workflow_from_file",
    "load_workflows_from_directory",
]
