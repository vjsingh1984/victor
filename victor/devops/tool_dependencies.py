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
The core configuration is now loaded from YAML (tool_dependencies.yaml),
while composed patterns remain in Python for complex logic.

Uses canonical tool names from ToolNames to ensure consistent naming
across RL Q-values, workflow patterns, and vertical configurations.

Migration Notes:
    - Transitions, clusters, sequences, dependencies are now in YAML
    - DevOpsToolDependencyProvider uses YAMLToolDependencyProvider
    - Composed patterns and graph functions remain in Python for flexibility
    - All exports preserved for backward compatibility

Example:
    from victor.devops.tool_dependencies import (
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

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from victor.core.tool_dependency_loader import (
    YAMLToolDependencyProvider,
    load_tool_dependency_yaml,
)
from victor.framework.tool_naming import ToolNames
from victor.tools.tool_graph import ToolExecutionGraph


# =============================================================================
# YAML Configuration Path
# =============================================================================
_YAML_CONFIG_PATH = Path(__file__).parent / "tool_dependencies.yaml"


# =============================================================================
# Backward Compatibility Exports
# =============================================================================
# These are derived from the YAML config at module load time for backward
# compatibility with code that imports these constants directly.


def _load_yaml_config():
    """Load YAML config and extract data structures for backward compatibility.

    Note: Canonicalization is disabled to preserve tool names as-is.
    This is important because the YAML uses distinct tool names like 'grep'
    (keyword search) and 'code_search' (semantic/AI search) that would
    otherwise be collapsed by the alias mapping.
    """
    config = load_tool_dependency_yaml(_YAML_CONFIG_PATH, canonicalize=False)
    return config


# Load config once at module import
_config = _load_yaml_config()

# Export transitions, clusters, sequences for backward compatibility
# Code using these constants will get the same data, now from YAML
DEVOPS_TOOL_TRANSITIONS: Dict[str, List[Tuple[str, float]]] = _config.transitions
DEVOPS_TOOL_CLUSTERS: Dict[str, Set[str]] = _config.clusters
DEVOPS_TOOL_SEQUENCES: Dict[str, List[str]] = _config.sequences
DEVOPS_TOOL_DEPENDENCIES = _config.dependencies
DEVOPS_REQUIRED_TOOLS: Set[str] = _config.required_tools
DEVOPS_OPTIONAL_TOOLS: Set[str] = _config.optional_tools


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


class DevOpsToolDependencyProvider(YAMLToolDependencyProvider):
    """Tool dependency provider for DevOps vertical.

    .. deprecated::
        Use ``create_vertical_tool_dependency_provider('devops')`` instead.
        This class is maintained for backward compatibility.

    Loads configuration from tool_dependencies.yaml for infrastructure
    and CI/CD workflows. Extends YAMLToolDependencyProvider with
    DevOps-specific features.

    Uses canonical ToolNames constants for consistency.

    Example:
        # Preferred (new code):
        from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider
        provider = create_vertical_tool_dependency_provider("devops")

        # Deprecated (backward compatible):
        provider = DevOpsToolDependencyProvider()

    Migration Notes:
        - Configuration now loaded from YAML instead of hand-coded Python
        - All functionality preserved via YAMLToolDependencyProvider
        - Composed patterns remain in DEVOPS_COMPOSED_PATTERNS constant
        - For new code, use create_vertical_tool_dependency_provider('devops')
    """

    def __init__(self):
        """Initialize the provider from YAML config.

        Note: Canonicalization is disabled to preserve tool names as-is.
        This is important because the YAML uses distinct tool names like 'grep'
        (keyword search) and 'code_search' (semantic/AI search) that would
        otherwise be collapsed by the alias mapping.

        .. deprecated::
            Use ``create_vertical_tool_dependency_provider('devops')`` instead.
        """
        import warnings

        warnings.warn(
            "DevOpsToolDependencyProvider is deprecated. "
            "Use create_vertical_tool_dependency_provider('devops') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            yaml_path=_YAML_CONFIG_PATH,
            canonicalize=False,
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
