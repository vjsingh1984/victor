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

"""Declarative tool execution graphs.

This module provides data structures for representing tool execution
as declarative graphs that can be cached, serialized, and optimized.

Key Features:
- Declarative graph representation
- Serializable to/from dict
- Cacheable and hashable
- Validation rules per node
- Dependency tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class CacheStrategy(Enum):
    """Cache strategy for tool execution."""

    NONE = "none"
    LRU = "lru"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class ValidationRuleType(Enum):
    """Validation rule types."""

    REQUIRED = "required"
    FORMAT = "format"
    RANGE = "range"
    PATTERN = "pattern"
    CUSTOM = "custom"


@dataclass
class ValidationRule:
    """Validation rule for tool arguments.

    Attributes:
        rule_type: Type of validation rule
        parameter: Parameter name to validate
        constraint: Constraint value (e.g., regex, min/max)
        error_message: Custom error message
    """

    rule_type: ValidationRuleType
    parameter: str
    constraint: Any
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_type": self.rule_type.value,
            "parameter": self.parameter,
            "constraint": self.constraint,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationRule":
        """Create from dictionary."""
        return cls(
            rule_type=ValidationRuleType(data["rule_type"]),
            parameter=data["parameter"],
            constraint=data["constraint"],
            error_message=data.get("error_message", ""),
        )


@dataclass
class ToolExecutionNode:
    """Single tool execution step (declarative).

    Attributes:
        tool_name: Name of the tool to execute
        validation_rules: List of validation rules
        normalization_strategy: Strategy for argument normalization
        cache_policy: Caching policy for this tool
        retry_policy: Retry policy for failures
        timeout_seconds: Maximum execution time
        metadata: Additional metadata
    """

    tool_name: str
    validation_rules: list[ValidationRule] = field(default_factory=list)
    normalization_strategy: str = "auto"
    cache_policy: str = "default"
    retry_policy: str = "default"
    timeout_seconds: float = 30.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Make node hashable for caching."""
        return hash((self.tool_name, self.normalization_strategy, self.cache_policy))

    def __eq__(self, other: object) -> bool:
        """Check equality for caching."""
        if not isinstance(other, ToolExecutionNode):
            return False
        return (
            self.tool_name == other.tool_name
            and self.normalization_strategy == other.normalization_strategy
            and self.cache_policy == other.cache_policy
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "validation_rules": [rule.to_dict() for rule in self.validation_rules],
            "normalization_strategy": self.normalization_strategy,
            "cache_policy": self.cache_policy,
            "retry_policy": self.retry_policy,
            "timeout_seconds": self.timeout_seconds,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolExecutionNode":
        """Create from dictionary."""
        rules = [ValidationRule.from_dict(rule) for rule in data.get("validation_rules", [])]
        return cls(
            tool_name=data["tool_name"],
            validation_rules=rules,
            normalization_strategy=data.get("normalization_strategy", "auto"),
            cache_policy=data.get("cache_policy", "default"),
            retry_policy=data.get("retry_policy", "default"),
            timeout_seconds=data.get("timeout_seconds", 30.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ToolDependency:
    """Dependency between tool execution nodes.

    Attributes:
        from_node: Source tool name
        to_node: Target tool name
        condition: Optional condition expression
    """

    from_node: str
    to_node: str
    condition: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "from": self.from_node,
            "to": self.to_node,
            "condition": self.condition,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolDependency":
        """Create from dictionary."""
        return cls(
            from_node=data["from"],
            to_node=data["to"],
            condition=data.get("condition"),
        )


@dataclass
class ToolExecutionGraph:
    """Declarative tool execution graph (cacheable, serializable).

    Attributes:
        nodes: List of execution nodes
        edges: Dependencies between nodes
        cache_strategy: Overall caching strategy
        metadata: Additional metadata
        version: Graph format version
    """

    nodes: list[ToolExecutionNode]
    edges: list[ToolDependency] = field(default_factory=list)
    cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"

    def __post_init__(self) -> None:
        """Validate graph on creation."""
        if not self.nodes:
            raise ValueError("Graph must have at least one node")

    def get_node(self, tool_name: str) -> Optional[ToolExecutionNode]:
        """Get node by tool name.

        Args:
            tool_name: Name of the tool

        Returns:
            ToolExecutionNode if found, None otherwise
        """
        for node in self.nodes:
            if node.tool_name == tool_name:
                return node
        return None

    def get_dependencies(self, tool_name: str) -> list[ToolDependency]:
        """Get dependencies for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            List of dependencies from this tool
        """
        return [edge for edge in self.edges if edge.from_node == tool_name]

    def get_dependents(self, tool_name: str) -> list[ToolDependency]:
        """Get tools that depend on this tool.

        Args:
            tool_name: Name of the tool

        Returns:
            List of dependencies to this tool
        """
        return [edge for edge in self.edges if edge.to_node == tool_name]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the graph
        """
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "cache_strategy": self.cache_strategy.value,
            "metadata": self.metadata,
            "version": self.version,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolExecutionGraph":
        """Create graph from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            ToolExecutionGraph instance
        """
        nodes = [ToolExecutionNode.from_dict(node_data) for node_data in data.get("nodes", [])]

        edges = [ToolDependency.from_dict(edge_data) for edge_data in data.get("edges", [])]

        return cls(
            nodes=nodes,
            edges=edges,
            cache_strategy=CacheStrategy(data.get("cache_strategy", "adaptive")),
            metadata=data.get("metadata", {}),
            version=data.get("version", "1.0"),
        )

    def __hash__(self) -> int:
        """Make graph hashable for caching."""
        # Hash based on nodes and edges structure
        node_hashes = [hash(n) for n in self.nodes]
        edge_hashes = [hash((e.from_node, e.to_node, e.condition)) for e in self.edges]
        return hash((tuple(node_hashes), tuple(edge_hashes)))

    def __eq__(self, other: object) -> bool:
        """Check equality for caching."""
        if not isinstance(other, ToolExecutionGraph):
            return False
        return (
            len(self.nodes) == len(other.nodes)
            and len(self.edges) == len(other.edges)
            and hash(self) == hash(other)
        )
