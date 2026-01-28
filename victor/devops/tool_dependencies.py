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

"""DevOps Tool Dependencies - Tool relationships for infrastructure workflows.

This module provides tool dependency configuration for DevOps workflows.
The core configuration is loaded from YAML (tool_dependencies.yaml),
while composed patterns remain in Python for complex logic.

Uses canonical tool names from ToolNames to ensure consistent naming
across RL Q-values, workflow patterns, and vertical configurations.

Vertical name is auto-inferred from module path to eliminate duplication.

Simplified Usage:
    - Use DevOpsToolDependencyProvider for vertical tool dependency management
    - Use DEVOPS_COMPOSED_PATTERNS for pre-defined DevOps workflow patterns
    - Use get_devops_tool_graph() for tool execution planning and suggestions
    - reset_devops_tool_graph() clears the cached instance for testing

Example:
    from victor.devops.tool_dependencies import (
        DevOpsToolDependencyProvider,
        get_devops_tool_graph,
        DEVOPS_COMPOSED_PATTERNS,
    )

    # Get the canonical tool dependency provider
    provider = DevOpsToolDependencyProvider

    # Get tool graph for DevOps workflows
    graph = get_devops_tool_graph()

    # Suggest next tools after reading a Dockerfile
    suggestions = graph.suggest_next_tools(ToolNames.READ, history=[ToolNames.LS])

    # Plan tool sequence for infrastructure deployment
    plan = graph.plan_for_goal({ToolNames.SHELL, ToolNames.GIT})

    # Access composed patterns
    dockerfile_pattern = DEVOPS_COMPOSED_PATTERNS["dockerfile_pipeline"]
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider
from victor.framework.tool_naming import ToolNames
from victor.tools.tool_graph import ToolExecutionGraph

if TYPE_CHECKING:
    from victor.core.tool_dependency_loader import YAMLToolDependencyProvider


# =============================================================================
# DevOpsToolDependencyProvider (canonical provider)
# =============================================================================
# Create canonical provider for DevOps vertical
# Vertical name is auto-inferred from module path (victor.devops.tool_dependencies -> devops)
DevOpsToolDependencyProvider = create_vertical_tool_dependency_provider()


# These represent higher-level operations composed of multiple tool calls
# that commonly appear together in DevOps workflows.


# Uses canonical ToolNames constants for consistency
DEVOPS_COMPOSED_PATTERNS: Dict[str, Dict[str, Any]] = {
    "dockerfile_pipeline": {
        "description": "Create and validate Dockerfile",
        "sequence": [ToolNames.READ, ToolNames.WRITE, ToolNames.DOCKER, ToolNames.SHELL],
        "inputs": {"application_type", "base_image"},
        "outputs": {"dockerfile_path", "image_id"},
        "weight": 0.9,
    },
    "ci_cd_config": {
        "description": "Set up CI/CD pipeline configuration",
        "sequence": [ToolNames.LS, ToolNames.READ, ToolNames.WRITE, ToolNames.SHELL],
        "inputs": {"repository_type", "ci_platform"},
        "outputs": {"pipeline_config_path"},
        "weight": 0.85,
    },
    "kubernetes_manifest": {
        "description": "Create Kubernetes deployment manifests",
        "sequence": [ToolNames.READ, ToolNames.WRITE, ToolNames.SHELL, ToolNames.GIT],
        "inputs": {"app_name", "replicas", "resources"},
        "outputs": {"manifest_paths"},
        "weight": 0.85,
    },
    "terraform_workflow": {
        "description": "Terraform init/plan/apply workflow",
        "sequence": [
            ToolNames.READ,
            ToolNames.SHELL,
            ToolNames.SHELL,
            ToolNames.SHELL,
            ToolNames.GIT,
        ],
        "inputs": {"terraform_dir", "environment"},
        "outputs": {"apply_output", "state_changes"},
        "weight": 0.8,
    },
    "monitoring_stack": {
        "description": "Set up monitoring with Prometheus/Grafana",
        "sequence": [
            ToolNames.READ,
            ToolNames.WRITE,
            ToolNames.EDIT,
            ToolNames.SHELL,
            ToolNames.SHELL,
        ],
        "inputs": {"services", "metrics_port"},
        "outputs": {"prometheus_config", "grafana_dashboard"},
        "weight": 0.8,
    },
    "security_audit": {
        "description": "Run security scans and generate report",
        "sequence": [ToolNames.SHELL, ToolNames.SHELL, ToolNames.READ, ToolNames.WRITE],
        "inputs": {"scan_target", "scan_type"},
        "outputs": {"scan_report", "remediation_suggestions"},
        "weight": 0.75,
    },
    "helm_deploy": {
        "description": "Deploy application using Helm",
        "sequence": [ToolNames.READ, ToolNames.EDIT, ToolNames.SHELL, ToolNames.SHELL],
        "inputs": {"chart_path", "values_override"},
        "outputs": {"release_name", "deployment_status"},
        "weight": 0.85,
    },
    "log_aggregation": {
        "description": "Set up log aggregation pipeline",
        "sequence": [ToolNames.READ, ToolNames.WRITE, ToolNames.SHELL, ToolNames.SHELL],
        "inputs": {"log_sources", "retention_days"},
        "outputs": {"fluentd_config", "elasticsearch_index"},
        "weight": 0.7,
    },
}


# =============================================================================
# ToolExecutionGraph Factory
# =============================================================================

# Cached tool graph instance
_devops_tool_graph: Optional[ToolExecutionGraph] = None


def get_devops_tool_graph() -> ToolExecutionGraph:
    """Get the DevOps tool execution graph.

    Creates a ToolExecutionGraph configured with DevOps-specific
    dependencies, transitions, sequences, and composed patterns.

    Returns:
        ToolExecutionGraph for DevOps workflows

    Example:
        graph = get_devops_tool_graph()

        # Suggest next tools
        suggestions = graph.suggest_next_tools("read_file")

        # Plan tool sequence
        plan = graph.plan_for_goal({"git_commit"})

        # Validate tool execution
        valid, missing = graph.validate_execution("bash", {"read_file"})
    """
    global _devops_tool_graph

    if _devops_tool_graph is not None:
        return _devops_tool_graph

    # Create graph from provider data
    provider = cast("YAMLToolDependencyProvider", DevOpsToolDependencyProvider)

    # Create new graph
    graph = ToolExecutionGraph(name="devops")

    # Add nodes and dependencies from provider
    for dep in provider.get_dependencies():
        graph.add_node(
            name=dep.tool_name,
            depends_on=dep.depends_on,
            enables=dep.enables,
            weight=dep.weight,
        )

    # Add sequences from provider
    for sequence in provider.get_tool_sequences():
        graph.add_sequence(sequence)

    # Add composed patterns as sequences with higher weights
    for pattern_name, pattern_data in DEVOPS_COMPOSED_PATTERNS.items():
        graph.add_sequence(pattern_data["sequence"], weight=pattern_data["weight"])

    _devops_tool_graph = graph
    return graph


def reset_devops_tool_graph() -> None:
    """Reset the cached DevOps tool graph.

    Useful for testing or when tool configurations change.
    """
    global _devops_tool_graph
    _devops_tool_graph = None


def get_devops_composed_pattern(pattern_name: str) -> Optional[Dict[str, Any]]:
    """Get a DevOps composed tool pattern by name.

    Args:
        pattern_name: Name of the pattern (e.g., "dockerfile_pipeline")

    Returns:
        Pattern configuration dict or None if not found

    Example:
        pattern = get_devops_composed_pattern("dockerfile_pipeline")
        if pattern:
            print(f"Sequence: {pattern['sequence']}")
            print(f"Inputs: {pattern['inputs']}")
    """
    return DEVOPS_COMPOSED_PATTERNS.get(pattern_name)


def list_devops_composed_patterns() -> List[str]:
    """List all available DevOps composed tool patterns.

    Returns:
        List of pattern names
    """
    return list(DEVOPS_COMPOSED_PATTERNS.keys())


__all__ = [
    # Provider class (canonical)
    "DevOpsToolDependencyProvider",
    # Composed patterns
    "DEVOPS_COMPOSED_PATTERNS",
    # Graph functions
    "get_devops_tool_graph",
    "reset_devops_tool_graph",
    "get_devops_composed_pattern",
    "list_devops_composed_patterns",
]
