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

Extends the core BaseToolDependencyProvider with DevOps-specific data.
Also provides composed tool patterns using ToolExecutionGraph.

Uses canonical tool names from ToolNames to ensure consistent naming
across RL Q-values, workflow patterns, and vertical configurations.

Example:
    from victor.verticals.devops.tool_dependencies import (
        DevOpsToolDependencyProvider,
        get_devops_tool_graph,
        DEVOPS_COMPOSED_PATTERNS,
    )

    # Get tool graph for DevOps workflows
    graph = get_devops_tool_graph()

    # Suggest next tools after reading a Dockerfile
    suggestions = graph.suggest_next_tools(ToolNames.READ, history=[ToolNames.LS])

    # Plan tool sequence for infrastructure deployment
    plan = graph.plan_for_goal({ToolNames.SHELL, ToolNames.GIT})
"""

from typing import Dict, List, Optional, Set, Tuple

from victor.core.tool_dependency_base import BaseToolDependencyProvider, ToolDependencyConfig
from victor.core.tool_types import ToolDependency
from victor.framework.tool_naming import ToolNames
from victor.tools.tool_graph import ToolExecutionGraph


# Tool dependency graph for DevOps workflows
# Uses canonical ToolNames constants for consistency
DEVOPS_TOOL_TRANSITIONS: Dict[str, List[Tuple[str, float]]] = {
    # Reading leads to more analysis or writing
    ToolNames.READ: [
        (ToolNames.SHELL, 0.3),  # Execute what we read
        (ToolNames.EDIT, 0.3),  # Modify configuration
        (ToolNames.GREP, 0.2),  # Find related configs
        (ToolNames.WRITE, 0.2),  # Write new config
    ],
    # Code search for finding configurations
    ToolNames.GREP: [
        (ToolNames.READ, 0.5),  # Read found files
        (ToolNames.CODE_SEARCH, 0.2),  # Refine search
        (ToolNames.GREP, 0.2),  # Continue searching
        (ToolNames.OVERVIEW, 0.1),  # Get context
    ],
    # Shell for infrastructure commands
    ToolNames.SHELL: [
        (ToolNames.READ, 0.3),  # Check results
        (ToolNames.SHELL, 0.3),  # Chain commands
        (ToolNames.WRITE, 0.2),  # Save output
        (ToolNames.GIT, 0.2),  # Check git state
    ],
    # Writing configurations
    ToolNames.WRITE: [
        (ToolNames.SHELL, 0.4),  # Validate/apply config
        (ToolNames.READ, 0.3),  # Review what was written
        (ToolNames.GIT, 0.2),  # Check changes
        (ToolNames.EDIT, 0.1),  # Refine
    ],
    # Editing configurations
    ToolNames.EDIT: [
        (ToolNames.SHELL, 0.4),  # Test changes
        (ToolNames.READ, 0.3),  # Verify edit
        (ToolNames.GIT, 0.3),  # Review changes / check status
    ],
    # Git operations (unified)
    ToolNames.GIT: [
        (ToolNames.READ, 0.4),  # Check files
        (ToolNames.EDIT, 0.3),  # Fix issues
        (ToolNames.SHELL, 0.3),  # Run commands
    ],
    # Web for documentation
    ToolNames.WEB_SEARCH: [
        (ToolNames.WEB_FETCH, 0.6),  # Fetch found docs
        (ToolNames.WEB_SEARCH, 0.2),  # Refine search
        (ToolNames.READ, 0.1),  # Check local files
        (ToolNames.WRITE, 0.1),  # Document findings
    ],
    ToolNames.WEB_FETCH: [
        (ToolNames.WRITE, 0.3),  # Apply learnings
        (ToolNames.WEB_SEARCH, 0.3),  # Find more
        (ToolNames.WEB_FETCH, 0.2),  # Fetch related
        (ToolNames.EDIT, 0.2),  # Update configs
    ],
    # Docker operations
    ToolNames.DOCKER: [
        (ToolNames.SHELL, 0.4),  # Run docker commands
        (ToolNames.READ, 0.3),  # Check Dockerfile
        (ToolNames.WRITE, 0.2),  # Create compose file
        (ToolNames.GIT, 0.1),  # Check changes
    ],
}

# Tools that work well together in DevOps
# Uses canonical ToolNames constants for consistency
DEVOPS_TOOL_CLUSTERS: Dict[str, Set[str]] = {
    "file_operations": {ToolNames.READ, ToolNames.WRITE, ToolNames.EDIT, ToolNames.LS},
    "git_operations": {ToolNames.GIT},  # Unified git tool
    "code_exploration": {ToolNames.GREP, ToolNames.CODE_SEARCH, ToolNames.OVERVIEW},
    "infrastructure": {ToolNames.SHELL, ToolNames.READ, ToolNames.WRITE, ToolNames.DOCKER},
    "documentation": {ToolNames.WEB_SEARCH, ToolNames.WEB_FETCH},
    "container_ops": {ToolNames.DOCKER, ToolNames.SHELL, ToolNames.READ, ToolNames.WRITE},
}

# Recommended sequences for common DevOps tasks
# Uses canonical ToolNames constants for consistency
DEVOPS_TOOL_SEQUENCES: Dict[str, List[str]] = {
    "dockerfile_create": [ToolNames.READ, ToolNames.WRITE, ToolNames.SHELL],
    "ci_cd_setup": [ToolNames.LS, ToolNames.GREP, ToolNames.READ, ToolNames.WRITE, ToolNames.SHELL],
    "kubernetes_deploy": [ToolNames.READ, ToolNames.WRITE, ToolNames.SHELL, ToolNames.SHELL],
    "terraform_apply": [ToolNames.READ, ToolNames.SHELL, ToolNames.SHELL, ToolNames.GIT],
    "config_update": [ToolNames.READ, ToolNames.EDIT, ToolNames.SHELL, ToolNames.GIT],
    "docker_build": [ToolNames.READ, ToolNames.WRITE, ToolNames.DOCKER, ToolNames.SHELL],
    "monitoring_setup": [ToolNames.READ, ToolNames.WRITE, ToolNames.EDIT, ToolNames.SHELL],
    "security_scan": [ToolNames.SHELL, ToolNames.READ, ToolNames.WRITE, ToolNames.GIT],
}

# Tool dependencies for DevOps
# Uses canonical ToolNames constants for consistency
DEVOPS_TOOL_DEPENDENCIES: List[ToolDependency] = [
    ToolDependency(
        tool_name=ToolNames.SHELL,
        depends_on={ToolNames.READ},
        enables={ToolNames.WRITE, ToolNames.GIT},
        weight=0.6,
    ),
    ToolDependency(
        tool_name=ToolNames.EDIT,
        depends_on={ToolNames.READ},
        enables={ToolNames.SHELL, ToolNames.GIT},
        weight=0.5,
    ),
    ToolDependency(
        tool_name=ToolNames.GIT,  # Unified git tool
        depends_on=set(),
        enables=set(),
        weight=0.4,
    ),
    ToolDependency(
        tool_name=ToolNames.WRITE,
        depends_on={ToolNames.READ},
        enables={ToolNames.SHELL, ToolNames.GIT},
        weight=0.5,
    ),
    ToolDependency(
        tool_name=ToolNames.DOCKER,
        depends_on={ToolNames.READ},
        enables={ToolNames.SHELL, ToolNames.WRITE},
        weight=0.6,
    ),
]

# Required tools for DevOps
# Uses canonical ToolNames constants for consistency
DEVOPS_REQUIRED_TOOLS: Set[str] = {ToolNames.READ, ToolNames.WRITE, ToolNames.EDIT, ToolNames.SHELL}

# Optional tools that enhance DevOps
# Uses canonical ToolNames constants for consistency
DEVOPS_OPTIONAL_TOOLS: Set[str] = {
    ToolNames.GREP,
    ToolNames.CODE_SEARCH,
    ToolNames.GIT,
    ToolNames.WEB_SEARCH,
    ToolNames.WEB_FETCH,
    ToolNames.DOCKER,
}


# =============================================================================
# Composed Tool Patterns
# =============================================================================
# These represent higher-level operations composed of multiple tool calls
# that commonly appear together in DevOps workflows.


# Uses canonical ToolNames constants for consistency
DEVOPS_COMPOSED_PATTERNS: Dict[str, Dict[str, any]] = {
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
        "sequence": [ToolNames.READ, ToolNames.SHELL, ToolNames.SHELL, ToolNames.SHELL, ToolNames.GIT],
        "inputs": {"terraform_dir", "environment"},
        "outputs": {"apply_output", "state_changes"},
        "weight": 0.8,
    },
    "monitoring_stack": {
        "description": "Set up monitoring with Prometheus/Grafana",
        "sequence": [ToolNames.READ, ToolNames.WRITE, ToolNames.EDIT, ToolNames.SHELL, ToolNames.SHELL],
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


class DevOpsToolDependencyProvider(BaseToolDependencyProvider):
    """Tool dependency provider for DevOps vertical.

    Extends BaseToolDependencyProvider with DevOps-specific tool
    relationships for infrastructure and CI/CD workflows.

    Uses canonical ToolNames constants for consistency.
    """

    def __init__(self):
        """Initialize the provider with DevOps-specific config."""
        super().__init__(
            ToolDependencyConfig(
                dependencies=DEVOPS_TOOL_DEPENDENCIES,
                transitions=DEVOPS_TOOL_TRANSITIONS,
                clusters=DEVOPS_TOOL_CLUSTERS,
                sequences=DEVOPS_TOOL_SEQUENCES,
                required_tools=DEVOPS_REQUIRED_TOOLS,
                optional_tools=DEVOPS_OPTIONAL_TOOLS,
                default_sequence=[ToolNames.READ, ToolNames.WRITE, ToolNames.SHELL],
            )
        )


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

    graph = ToolExecutionGraph(name="devops")

    # Add dependencies
    for dep in DEVOPS_TOOL_DEPENDENCIES:
        graph.add_dependency(
            tool_name=dep.tool_name,
            depends_on=dep.depends_on,
            enables=dep.enables,
            weight=dep.weight,
        )

    # Add transitions
    graph.add_transitions(DEVOPS_TOOL_TRANSITIONS)

    # Add sequences
    for name, sequence in DEVOPS_TOOL_SEQUENCES.items():
        graph.add_sequence(sequence, weight=0.7)

    # Add clusters
    for name, tools in DEVOPS_TOOL_CLUSTERS.items():
        graph.add_cluster(name, tools)

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


def get_composed_pattern(pattern_name: str) -> Optional[Dict[str, any]]:
    """Get a composed tool pattern by name.

    Args:
        pattern_name: Name of the pattern (e.g., "dockerfile_pipeline")

    Returns:
        Pattern configuration dict or None if not found

    Example:
        pattern = get_composed_pattern("dockerfile_pipeline")
        if pattern:
            print(f"Sequence: {pattern['sequence']}")
            print(f"Inputs: {pattern['inputs']}")
    """
    return DEVOPS_COMPOSED_PATTERNS.get(pattern_name)


def list_composed_patterns() -> List[str]:
    """List all available composed tool patterns.

    Returns:
        List of pattern names
    """
    return list(DEVOPS_COMPOSED_PATTERNS.keys())


__all__ = [
    # Provider class
    "DevOpsToolDependencyProvider",
    # Data exports
    "DEVOPS_TOOL_DEPENDENCIES",
    "DEVOPS_TOOL_TRANSITIONS",
    "DEVOPS_TOOL_CLUSTERS",
    "DEVOPS_TOOL_SEQUENCES",
    "DEVOPS_REQUIRED_TOOLS",
    "DEVOPS_OPTIONAL_TOOLS",
    "DEVOPS_COMPOSED_PATTERNS",
    # Graph functions
    "get_devops_tool_graph",
    "reset_devops_tool_graph",
    "get_composed_pattern",
    "list_composed_patterns",
]
