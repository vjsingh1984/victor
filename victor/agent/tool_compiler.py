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

"""Compiles tool execution graphs from tool calls.

This module provides compilation of imperative tool call lists
into declarative execution graphs that can be cached and optimized.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List

from victor.agent.tool_graph import (
    CacheStrategy,
    ToolDependency,
    ToolExecutionGraph,
    ToolExecutionNode,
    ValidationRule,
    ValidationRuleType,
)

logger = logging.getLogger(__name__)


class ToolExecutionCompiler:
    """Compiles execution graphs from tool calls.

    The compiler transforms imperative tool call lists into declarative
    execution graphs with validation rules, caching policies, and
    dependency tracking.
    """

    def __init__(self, tool_registry: Any) -> None:
        """Initialize compiler.

        Args:
            tool_registry: ToolRegistry instance
        """
        self._tool_registry = tool_registry
        self._idempotent_tools = self._load_idempotent_tools()

    def _load_idempotent_tools(self) -> frozenset[str]:
        """Load idempotent tool names from config.

        Returns:
            Frozenset of idempotent tool names
        """
        try:
            from victor.config.tool_selection_defaults import IdempotentTools

            return IdempotentTools.IDEMPOTENT_TOOLS
        except Exception:
            logger.warning("Failed to load idempotent tools config")
            return frozenset()

    def compile(self, tool_calls: List[Dict[str, Any]]) -> ToolExecutionGraph:
        """Create declarative execution graph from tool calls.

        Args:
            tool_calls: List of tool call dictionaries

        Returns:
            Compiled ToolExecutionGraph
        """
        nodes = []
        edges = []
        seen_tools = set()

        for tc in tool_calls:
            # Skip invalid structures
            if not isinstance(tc, dict):
                continue  # type: ignore[unreachable]

            tool_name = tc.get("name", "")
            if not tool_name or tool_name in seen_tools:
                continue

            seen_tools.add(tool_name)

            # Create node
            node = ToolExecutionNode(
                tool_name=tool_name,
                validation_rules=self._get_validation_rules(tool_name),
                normalization_strategy="auto",
                cache_policy=self._get_cache_policy(tool_name),
                retry_policy=self._get_retry_policy(tool_name),
                timeout_seconds=self._get_timeout(tool_name),
                metadata=self._get_tool_metadata(tool_name),
            )
            nodes.append(node)

            # Create edges (dependencies)
            dependencies = self._get_tool_dependencies(tool_name, tc)
            for dep_tool in dependencies:
                edge = ToolDependency(from_node=dep_tool, to_node=tool_name)
                edges.append(edge)

        return ToolExecutionGraph(
            nodes=nodes,
            edges=edges,
            cache_strategy=self._determine_cache_strategy(nodes),
            metadata={
                "tool_count": len(nodes),
                "edge_count": len(edges),
                "compiler_version": "1.0",
            },
        )

    def _get_validation_rules(self, tool_name: str) -> List[ValidationRule]:
        """Get validation rules for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            List of validation rules
        """
        rules: List[ValidationRule] = []
        tool = self._tool_registry.get_tool(tool_name)

        if not tool:
            return rules

        # Add required parameter rules
        required = tool.parameters.get("required", [])

        for param_name in required:
            rule = ValidationRule(
                rule_type=ValidationRuleType.REQUIRED,
                parameter=param_name,
                constraint="required",
                error_message=f"Parameter '{param_name}' is required",
            )
            rules.append(rule)

        return rules

    def _get_cache_policy(self, tool_name: str) -> str:
        """Get cache policy for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Cache policy string
        """
        # Check if tool is idempotent
        if tool_name.lower() in [t.lower() for t in self._idempotent_tools]:
            return "idempotent"
        return "default"

    def _get_retry_policy(self, tool_name: str) -> str:
        """Get retry policy for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Retry policy string
        """
        return "default"

    def _get_timeout(self, tool_name: str) -> float:
        """Get timeout for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Timeout in seconds
        """
        return 30.0  # Default timeout

    def _get_tool_metadata(self, tool_name: str) -> Dict[str, Any]:
        """Get metadata for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Metadata dictionary
        """
        tool = self._tool_registry.get_tool(tool_name)
        if not tool:
            return {}

        return {
            "description": tool.description,
            "category": getattr(tool, "category", None),
            "cost_tier": getattr(tool, "cost_tier", None),
        }

    def _get_tool_dependencies(self, tool_name: str, tool_call: Dict[str, Any]) -> List[str]:
        """Get dependencies for a tool.

        Args:
            tool_name: Name of the tool
            tool_call: Tool call dictionary

        Returns:
            List of tool names this tool depends on
        """
        # Could analyze tool_call to infer dependencies
        # For now, return empty list
        return []

    def _determine_cache_strategy(self, nodes: List[ToolExecutionNode]) -> CacheStrategy:
        """Determine best cache strategy for the graph.

        Args:
            nodes: List of execution nodes

        Returns:
            CacheStrategy enum
        """
        # Check if all tools are idempotent
        all_idempotent = all(node.cache_policy == "idempotent" for node in nodes)

        if all_idempotent:
            return CacheStrategy.TTL
        else:
            return CacheStrategy.ADAPTIVE

    def compute_graph_hash(self, tool_calls: List[Dict[str, Any]]) -> str:
        """Compute hash for tool calls (for cache key).

        Args:
            tool_calls: List of tool call dictionaries

        Returns:
            Hex digest hash string
        """
        # Normalize and serialize
        normalized = sorted(json.dumps(tc, sort_keys=True, default=str) for tc in tool_calls)
        combined = "".join(normalized)

        # Use SHA256 for hash
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
