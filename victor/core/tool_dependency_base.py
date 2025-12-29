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

"""Base Tool Dependency Provider for vertical-agnostic tool relationship management.

This module provides a base implementation of ToolDependencyProviderProtocol
that can be extended by any vertical with just data configuration.

Design Philosophy:
- Verticals define DATA (transitions, clusters, sequences)
- Core provides the ALGORITHM (transition weights, suggestions)
- Eliminates duplicate code across verticals

Usage:
    from victor.core.tool_dependency_base import BaseToolDependencyProvider

    class CodingToolDependencyProvider(BaseToolDependencyProvider):
        def __init__(self):
            super().__init__(
                dependencies=CODING_DEPENDENCIES,
                transitions=CODING_TRANSITIONS,
                clusters=CODING_CLUSTERS,
                sequences=CODING_SEQUENCES,
                required_tools={"read", "edit", "shell"},
                optional_tools={"grep", "semantic_search"},
            )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from victor.verticals.protocols import ToolDependency, ToolDependencyProviderProtocol


@dataclass
class ToolDependencyConfig:
    """Configuration for tool dependency provider.

    Attributes:
        dependencies: List of ToolDependency objects defining depends_on/enables
        transitions: Dict mapping tool → [(next_tool, probability), ...]
        clusters: Dict mapping cluster_name → set of tools
        sequences: Dict mapping task_type → [tool_sequence]
        required_tools: Tools that are essential for this vertical
        optional_tools: Tools that enhance but aren't required
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

    Example:
        class DevOpsToolDependencyProvider(BaseToolDependencyProvider):
            def __init__(self):
                super().__init__(ToolDependencyConfig(
                    dependencies=[
                        ToolDependency("shell", depends_on={"read"}, enables={"write"}),
                    ],
                    transitions={
                        "read": [("shell", 0.4), ("edit", 0.3)],
                    },
                    clusters={
                        "file_ops": {"read", "write", "edit"},
                    },
                    sequences={
                        "deploy": ["read", "write", "shell"],
                    },
                    required_tools={"read", "write", "shell"},
                ))
    """

    def __init__(
        self,
        config: Optional[ToolDependencyConfig] = None,
        *,
        # Alternative: pass individual args for simpler initialization
        dependencies: Optional[List[ToolDependency]] = None,
        transitions: Optional[Dict[str, List[Tuple[str, float]]]] = None,
        clusters: Optional[Dict[str, Set[str]]] = None,
        sequences: Optional[Dict[str, List[str]]] = None,
        required_tools: Optional[Set[str]] = None,
        optional_tools: Optional[Set[str]] = None,
        default_sequence: Optional[List[str]] = None,
    ):
        """Initialize the provider.

        Args:
            config: Full configuration object (preferred)
            dependencies: Tool dependency definitions
            transitions: Transition probability mapping
            clusters: Tool cluster groupings
            sequences: Task type to sequence mapping
            required_tools: Essential tools for vertical
            optional_tools: Non-essential enhancing tools
            default_sequence: Fallback sequence when task type unknown
        """
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

        # Build reverse lookup for faster dependency checking
        self._dependency_map: Dict[str, ToolDependency] = {
            d.tool_name: d for d in self._config.dependencies
        }

    # =========================================================================
    # ToolDependencyProviderProtocol Implementation
    # =========================================================================

    def get_dependencies(self) -> List[ToolDependency]:
        """Get tool dependencies.

        Returns:
            List of ToolDependency objects
        """
        return self._config.dependencies.copy()

    def get_tool_sequences(self) -> List[List[str]]:
        """Get common tool sequences.

        Returns:
            List of tool name sequences
        """
        return [list(seq) for seq in self._config.sequences.values()]

    # =========================================================================
    # Extended Methods (commonly used across verticals)
    # =========================================================================

    def get_tool_transitions(self) -> Dict[str, List[Tuple[str, float]]]:
        """Get tool transition probabilities.

        Returns:
            Dict mapping tool → [(next_tool, probability), ...]
        """
        return self._config.transitions.copy()

    def get_tool_clusters(self) -> Dict[str, Set[str]]:
        """Get tool clusters (groups of related tools).

        Returns:
            Dict mapping cluster_name → set of tools
        """
        return {k: v.copy() for k, v in self._config.clusters.items()}

    def get_recommended_sequence(self, task_type: str) -> List[str]:
        """Get recommended tool sequence for a task type.

        Args:
            task_type: Type of task (e.g., "dockerfile_create", "code_review")

        Returns:
            List of tool names in recommended order
        """
        return list(self._config.sequences.get(task_type, self._config.default_sequence))

    def get_required_tools(self) -> Set[str]:
        """Get tools essential for this vertical.

        Returns:
            Set of required tool names
        """
        return self._config.required_tools.copy()

    def get_optional_tools(self) -> Set[str]:
        """Get tools that enhance but aren't required.

        Returns:
            Set of optional tool names
        """
        return self._config.optional_tools.copy()

    def get_transition_weight(self, from_tool: str, to_tool: str) -> float:
        """Get the transition weight between two tools.

        Higher weight = more likely useful transition.

        Args:
            from_tool: Previous tool called
            to_tool: Candidate next tool

        Returns:
            Transition weight (0.0 to 1.0)
        """
        # 1. Check if to_tool depends on from_tool
        if to_tool in self._dependency_map:
            dep = self._dependency_map[to_tool]
            if from_tool in dep.depends_on:
                return dep.weight

        # 2. Check if from_tool enables to_tool
        if from_tool in self._dependency_map:
            dep = self._dependency_map[from_tool]
            if to_tool in dep.enables:
                return dep.weight * 0.8  # Slightly lower for enables

        # 3. Check transitions dict for probability
        transitions = self._config.transitions.get(from_tool, [])
        for tool, prob in transitions:
            if tool == to_tool:
                return prob

        # 4. Check sequences for adjacency
        for seq in self._config.sequences.values():
            for i in range(len(seq) - 1):
                if seq[i] == from_tool and seq[i + 1] == to_tool:
                    return 0.6  # Moderate weight for sequence match

        # 5. Default low weight for unknown transitions
        return 0.3

    def suggest_next_tool(
        self,
        current_tool: str,
        used_tools: Optional[List[str]] = None,
    ) -> str:
        """Suggest the next tool based on current tool and history.

        Args:
            current_tool: Currently used tool
            used_tools: History of recently used tools

        Returns:
            Suggested next tool name
        """
        used_tools = used_tools or []
        transitions = self._config.transitions.get(current_tool, [])

        if not transitions:
            # Check if current tool enables other tools
            if current_tool in self._dependency_map:
                dep = self._dependency_map[current_tool]
                if dep.enables:
                    return next(iter(dep.enables))
            return self._config.default_sequence[0] if self._config.default_sequence else "read"

        # Avoid recently used tools (prevent loops)
        recent = set(used_tools[-3:]) if len(used_tools) >= 3 else set(used_tools)

        for tool, _prob in sorted(transitions, key=lambda x: x[1], reverse=True):
            if tool not in recent:
                return tool

        # Fall back to highest probability
        return transitions[0][0]

    def find_cluster(self, tool: str) -> Optional[str]:
        """Find which cluster a tool belongs to.

        Args:
            tool: Tool name

        Returns:
            Cluster name or None if not in any cluster
        """
        for cluster_name, tools in self._config.clusters.items():
            if tool in tools:
                return cluster_name
        return None

    def get_cluster_tools(self, cluster_name: str) -> Set[str]:
        """Get all tools in a cluster.

        Args:
            cluster_name: Name of the cluster

        Returns:
            Set of tool names in the cluster
        """
        return self._config.clusters.get(cluster_name, set()).copy()

    def is_valid_transition(self, from_tool: str, to_tool: str) -> bool:
        """Check if a transition is valid (has defined relationship).

        Args:
            from_tool: Source tool
            to_tool: Destination tool

        Returns:
            True if transition is explicitly defined
        """
        # Check dependency relationship
        if to_tool in self._dependency_map:
            dep = self._dependency_map[to_tool]
            if from_tool in dep.depends_on:
                return True

        if from_tool in self._dependency_map:
            dep = self._dependency_map[from_tool]
            if to_tool in dep.enables:
                return True

        # Check transitions
        transitions = self._config.transitions.get(from_tool, [])
        return any(tool == to_tool for tool, _ in transitions)


__all__ = [
    "BaseToolDependencyProvider",
    "ToolDependencyConfig",
]
