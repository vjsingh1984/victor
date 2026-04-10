"""Base tool dependency implementations for vertical plugins.

This module provides the base classes and utilities that external verticals
use to define tool dependency relationships. By placing these in the SDK,
external verticals can stop importing from victor.core.

Classes:
    BaseToolDependencyProvider: Base implementation with common algorithms
    ToolDependencyConfig: Configuration dataclass for dependency data
    EmptyToolDependencyProvider: Null object fallback for missing configs
    ToolDependencyLoadError: Exception for YAML loading failures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from victor_sdk.verticals.protocols.tools import (
    ToolDependency,
    ToolDependencyProviderProtocol,
)


class ToolDependencyLoadError(Exception):
    """Exception raised when loading tool dependency configuration fails.

    Attributes:
        path: Path to the configuration that failed to load.
        message: Detailed error message.
    """

    def __init__(self, path: str, message: str = ""):
        self.path = path
        self.message = message or f"Failed to load tool dependencies from {path}"
        super().__init__(self.message)


@dataclass
class ToolDependencyConfig:
    """Configuration for tool dependency provider.

    Attributes:
        dependencies: List of ToolDependency objects defining depends_on/enables
        transitions: Dict mapping tool -> [(next_tool, probability), ...]
        clusters: Dict mapping cluster_name -> set of tools
        sequences: Dict mapping task_type -> [tool_sequence]
        required_tools: Tools that are essential for this vertical
        optional_tools: Tools that enhance but aren't required
        default_sequence: Fallback sequence when task type is unknown
    """

    dependencies: List[ToolDependency] = field(default_factory=list)
    transitions: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    clusters: Dict[str, Set[str]] = field(default_factory=dict)
    sequences: Dict[str, List[str]] = field(default_factory=dict)
    required_tools: Set[str] = field(default_factory=set)
    optional_tools: Set[str] = field(default_factory=set)
    default_sequence: List[str] = field(default_factory=lambda: ["read", "edit"])


class BaseToolDependencyProvider(ToolDependencyProviderProtocol):
    """Base implementation of tool dependency provider.

    Provides common algorithms for:
    - Computing transition weights
    - Suggesting next tools
    - Finding tool clusters
    - Recommending sequences

    Subclasses only need to provide configuration data.

    Example::

        class DevOpsToolDeps(BaseToolDependencyProvider):
            def __init__(self):
                super().__init__(ToolDependencyConfig(
                    dependencies=[
                        ToolDependency("shell", depends_on={"read"}),
                    ],
                    transitions={"read": [("shell", 0.4)]},
                    required_tools={"read", "write", "shell"},
                ))
    """

    def __init__(
        self,
        config: Optional[ToolDependencyConfig] = None,
        *,
        dependencies: Optional[List[ToolDependency]] = None,
        transitions: Optional[Dict[str, List[Tuple[str, float]]]] = None,
        clusters: Optional[Dict[str, Set[str]]] = None,
        sequences: Optional[Dict[str, List[str]]] = None,
        required_tools: Optional[Set[str]] = None,
        optional_tools: Optional[Set[str]] = None,
        default_sequence: Optional[List[str]] = None,
    ):
        if config:
            self._config = config
        else:
            self._config = ToolDependencyConfig(
                dependencies=dependencies or [],
                transitions=transitions or {},
                clusters=clusters or {},
                sequences=sequences or {},
                required_tools=required_tools or set(),
                optional_tools=optional_tools or set(),
                default_sequence=default_sequence or ["read", "edit"],
            )
        self._dependency_map: Dict[str, ToolDependency] = {
            d.tool_name: d for d in self._config.dependencies
        }

    # === ToolDependencyProviderProtocol ===

    def get_dependencies(self) -> List[ToolDependency]:
        return self._config.dependencies.copy()

    def get_tool_sequences(self) -> List[List[str]]:
        return [list(seq) for seq in self._config.sequences.values()]

    # === Extended methods ===

    def get_tool_transitions(self) -> Dict[str, List[Tuple[str, float]]]:
        return self._config.transitions.copy()

    def get_tool_clusters(self) -> Dict[str, Set[str]]:
        return {k: v.copy() for k, v in self._config.clusters.items()}

    def get_recommended_sequence(self, task_type: str) -> List[str]:
        return list(self._config.sequences.get(task_type, self._config.default_sequence))

    def get_required_tools(self) -> Set[str]:
        return self._config.required_tools.copy()

    def get_optional_tools(self) -> Set[str]:
        return self._config.optional_tools.copy()

    def get_transition_weight(self, from_tool: str, to_tool: str) -> float:
        if to_tool in self._dependency_map:
            dep = self._dependency_map[to_tool]
            if from_tool in dep.depends_on:
                return dep.weight

        if from_tool in self._dependency_map:
            dep = self._dependency_map[from_tool]
            if to_tool in dep.enables:
                return dep.weight * 0.8

        transitions = self._config.transitions.get(from_tool, [])
        for tool, prob in transitions:
            if tool == to_tool:
                return prob

        for seq in self._config.sequences.values():
            for i in range(len(seq) - 1):
                if seq[i] == from_tool and seq[i + 1] == to_tool:
                    return 0.6

        return 0.3

    def suggest_next_tool(
        self,
        current_tool: str,
        used_tools: Optional[List[str]] = None,
    ) -> str:
        used_tools = used_tools or []
        transitions = self._config.transitions.get(current_tool, [])

        if not transitions:
            if current_tool in self._dependency_map:
                dep = self._dependency_map[current_tool]
                if dep.enables:
                    return next(iter(dep.enables))
            return self._config.default_sequence[0] if self._config.default_sequence else "read"

        recent = set(used_tools[-3:]) if len(used_tools) >= 3 else set(used_tools)
        for tool, _prob in sorted(transitions, key=lambda x: x[1], reverse=True):
            if tool not in recent:
                return tool

        return transitions[0][0]

    def find_cluster(self, tool: str) -> Optional[str]:
        for cluster_name, tools in self._config.clusters.items():
            if tool in tools:
                return cluster_name
        return None

    def get_cluster_tools(self, cluster_name: str) -> Set[str]:
        return self._config.clusters.get(cluster_name, set()).copy()

    def is_valid_transition(self, from_tool: str, to_tool: str) -> bool:
        if to_tool in self._dependency_map:
            dep = self._dependency_map[to_tool]
            if from_tool in dep.depends_on:
                return True
        if from_tool in self._dependency_map:
            dep = self._dependency_map[from_tool]
            if to_tool in dep.enables:
                return True
        transitions = self._config.transitions.get(from_tool, [])
        return any(tool == to_tool for tool, _ in transitions)


class EmptyToolDependencyProvider:
    """Null object fallback when tool dependency config is missing.

    Provides minimal valid defaults so the system degrades gracefully.
    """

    def __init__(self, vertical: str = "unknown"):
        self._vertical = vertical

    @property
    def vertical(self) -> str:
        return self._vertical

    def get_dependencies(self) -> List[ToolDependency]:
        return []

    def get_tool_sequences(self) -> List[List[str]]:
        return [["read"]]

    def get_tool_transitions(self) -> Dict[str, List[Tuple[str, float]]]:
        return {}

    def get_tool_clusters(self) -> Dict[str, Set[str]]:
        return {}

    def get_recommended_sequence(self, task_type: str) -> List[str]:
        return ["read"]

    def get_required_tools(self) -> Set[str]:
        return {"read"}

    def get_optional_tools(self) -> Set[str]:
        return set()

    def get_transition_weight(self, from_tool: str, to_tool: str) -> float:
        return 0.3

    def suggest_next_tool(
        self,
        current_tool: str,
        used_tools: Optional[List[str]] = None,
    ) -> str:
        return "read"

    def find_cluster(self, tool: str) -> Optional[str]:
        return None

    def get_cluster_tools(self, cluster_name: str) -> Set[str]:
        return set()

    def is_valid_transition(self, from_tool: str, to_tool: str) -> bool:
        return True


# Factory function placeholder — the full YAML-based factory lives in
# victor.core.tool_dependency_loader since it depends on core runtime
# utilities. This SDK version provides a simple config-based factory.
def create_vertical_tool_dependency_provider(
    vertical_name: str,
    config: Optional[ToolDependencyConfig] = None,
) -> BaseToolDependencyProvider:
    """Create a tool dependency provider for a vertical.

    Args:
        vertical_name: Name of the vertical
        config: Optional configuration (returns EmptyToolDependencyProvider if None)

    Returns:
        A tool dependency provider instance
    """
    if config is None:
        return EmptyToolDependencyProvider(vertical_name)
    return BaseToolDependencyProvider(config)


__all__ = [
    "BaseToolDependencyProvider",
    "EmptyToolDependencyProvider",
    "ToolDependencyConfig",
    "ToolDependencyLoadError",
    "create_vertical_tool_dependency_provider",
]
