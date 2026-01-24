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

"""Framework-level tool packs for baseline capabilities.

This module provides shared tool packs that verticals can extend instead of
duplicating tool lists. This reduces duplication and ensures consistency
across verticals.

Design Principles:
    - DRY: Define tools once, reference everywhere
    - Composability: Packs can extend other packs
    - Override: Verticals can customize/extend base packs
    - Type Safety: ToolPack is type-safe with validation

Example:
    from victor.framework.tool_packs import (
        BASE_FILE_OPS,
        DEVOPS_PACK,
        resolve_tool_pack,
    )

    # Use base pack
    tools = resolve_tool_pack(BASE_FILE_OPS)

    # Extend with vertical-specific tools
    devops_tools = resolve_tool_pack(DEVOPS_PACK)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


@dataclass
class ToolPack:
    """A collection of tools with metadata and inheritance.

    Attributes:
        name: Unique identifier for this pack
        tools: List of tool names in this pack
        description: Human-readable description
        extends: Parent pack name to inherit from
        excludes: Tools to explicitly exclude from parent
        metadata: Additional metadata (tags, priority, etc.)

    Example:
        pack = ToolPack(
            name="devops",
            extends="base_file_ops",
            tools=["docker", "kubernetes"],
            excludes=["edit"],  # Don't allow edit in DevOps
        )
    """

    name: str
    tools: List[str] = field(default_factory=list)
    description: str = ""
    extends: Optional[str] = None
    excludes: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        """Validate tool pack configuration."""
        if not self.name:
            raise ValueError("ToolPack must have a name")

        # Validate no circular dependencies
        if self.extends and self.extends == self.name:
            raise ValueError(f"ToolPack '{self.name}' cannot extend itself")


class ToolPackRegistry:
    """Registry for tool packs with resolution and validation.

    Example:
        registry = ToolPackRegistry()
        registry.register(BASE_FILE_OPS)
        registry.register(DEVOPS_PACK)

        # Resolve pack with all dependencies
        tools = registry.resolve("devops")
    """

    def __init__(self):
        self._packs: Dict[str, ToolPack] = {}

    def register(self, pack: ToolPack) -> None:
        """Register a tool pack.

        Args:
            pack: ToolPack to register

        Raises:
            ValueError: If pack name already registered
        """
        if pack.name in self._packs:
            raise ValueError(f"ToolPack '{pack.name}' already registered")

        self._packs[pack.name] = pack
        logger.debug(f"Registered tool pack: {pack.name} with {len(pack.tools)} tools")

    def get(self, name: str) -> Optional[ToolPack]:
        """Get tool pack by name.

        Args:
            name: Tool pack name

        Returns:
            ToolPack or None if not found
        """
        return self._packs.get(name)

    def resolve(
        self,
        name: str,
        include_metadata: bool = False,
    ) -> Union[List[str], Dict[str, Any]]:
        """Resolve tool pack with all inherited tools.

        Args:
            name: Tool pack name to resolve
            include_metadata: If True, return dict with tools and metadata

        Returns:
            List of tool names (or dict if include_metadata=True)

        Raises:
            ValueError: If pack not found or circular dependency detected
        """
        if name not in self._packs:
            available = ", ".join(self._packs.keys())
            raise ValueError(
                f"ToolPack '{name}' not found. " f"Available: {available if available else 'none'}"
            )

        tools: List[str] = []  # Use list to preserve order
        seen: Set[str] = set()  # Track seen tools to avoid duplicates
        excluded_global: Set[str] = set()  # Track tools excluded by any pack
        visited: Set[str] = set()
        metadata: Dict[str, object] = {}

        def _resolve_pack(pack_name: str) -> None:
            """Recursively resolve pack and its parents."""
            if pack_name in visited:
                raise ValueError(f"Circular dependency detected: {pack_name}")

            visited.add(pack_name)
            pack = self._packs[pack_name]

            # Resolve parent first (ensures base tools come first)
            if pack.extends:
                _resolve_pack(pack.extends)

            # Apply exclusions from this pack
            # (these exclusions apply to ALL tools seen so far)
            for tool in pack.excludes:
                excluded_global.add(tool)

            # Add tools from this pack, preserving order
            for tool in pack.tools:
                if tool not in seen:
                    tools.append(tool)
                    seen.add(tool)

            # Merge metadata
            if pack.metadata:
                metadata.update(pack.metadata)

        _resolve_pack(name)

        # Filter out excluded tools from final result
        final_tools = [t for t in tools if t not in excluded_global]

        if include_metadata:
            return {
                "tools": final_tools,
                "excluded": list(excluded_global),
                "metadata": metadata,
                "count": len(final_tools),
            }

        return final_tools

    def list_packs(self) -> List[str]:
        """List all registered pack names.

        Returns:
            List of pack names
        """
        return list(self._packs.keys())

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get dependency graph of all packs.

        Returns:
            Dict mapping pack name to list of packs that extend it
        """
        graph: Dict[str, List[str]] = {}

        for pack_name, pack in self._packs.items():
            if pack.extends:
                if pack.extends not in graph:
                    graph[pack.extends] = []
                graph[pack.extends].append(pack_name)

        return graph


# Global registry instance
_tool_pack_registry = ToolPackRegistry()


def get_tool_pack_registry() -> ToolPackRegistry:
    """Get the global tool pack registry.

    Returns:
        ToolPackRegistry singleton
    """
    return _tool_pack_registry


# =============================================================================
# Baseline Tool Packs (Available to All Verticals)
# =============================================================================

BASE_FILE_OPS = ToolPack(
    name="base_file_ops",
    tools=[
        "read",
        "write",
        "edit",
        "search",
        "grep",
    ],
    description="Core file operations available to all verticals",
    metadata={"category": "filesystem", "priority": 1},
)

BASE_WEB = ToolPack(
    name="base_web",
    tools=[
        "web_search",
        "fetch_url",
    ],
    description="Basic web capabilities",
    metadata={"category": "web", "priority": 2},
)

BASE_GIT = ToolPack(
    name="base_git",
    tools=[
        "git_status",
        "git_diff",
        "git_log",
        "git_blame",
    ],
    description="Git operations",
    metadata={"category": "vcs", "priority": 2},
)

BASE_ANALYSIS = ToolPack(
    name="base_analysis",
    tools=[
        "semantic_search",
        "code_search",
        "find_references",
    ],
    description="Code analysis tools",
    metadata={"category": "analysis", "priority": 3},
)

BASE_EXECUTION = ToolPack(
    name="base_execution",
    tools=[
        "bash",
        "shell",
    ],
    description="Command execution tools",
    metadata={"category": "execution", "priority": 1},
)

BASE_TESTING = ToolPack(
    name="base_testing",
    tools=[
        "test",
        "lint",
        "type_check",
    ],
    description="Testing and validation tools",
    metadata={"category": "testing", "priority": 3},
)

# =============================================================================
# Vertical-Specific Packs (Extend Base Packs)
# =============================================================================

CODING_PACK = ToolPack(
    name="coding",
    extends="base_file_ops",
    tools=[
        "semantic_search",
        "code_search",
        "find_references",
        "test",
        "lint",
        "type_check",
    ],
    description="Software development tools (extends base_file_ops)",
    metadata={"vertical": "coding", "priority": 1},
)

DEVOPS_PACK = ToolPack(
    name="devops",
    extends="base_file_ops",
    tools=[
        "docker",
        "docker_compose",
        "kubernetes",
        "terraform",
        "ansible",
        "helm",
    ],
    description="DevOps and infrastructure tools (extends base_file_ops)",
    metadata={"vertical": "devops", "priority": 1},
)

RAG_PACK = ToolPack(
    name="rag",
    extends="base_file_ops",
    tools=[
        "semantic_search",
        "embedding_search",
        "vector_store_query",
    ],
    description="RAG and retrieval tools (extends base_file_ops)",
    metadata={"vertical": "rag", "priority": 1},
)

DATA_ANALYSIS_PACK = ToolPack(
    name="dataanalysis",
    extends="base_file_ops",
    tools=[
        "pandas",
        "matplotlib",
        "plot",
        "statistics",
    ],
    description="Data analysis and visualization (extends base_file_ops)",
    metadata={"vertical": "dataanalysis", "priority": 1},
)

RESEARCH_PACK = ToolPack(
    name="research",
    extends="base_web",
    tools=[
        "web_search",
        "fetch_url",
        "arxiv_search",
        "paper_search",
    ],
    description="Research and academic tools (extends base_web)",
    metadata={"vertical": "research", "priority": 1},
)

# =============================================================================
# Convenience Functions
# =============================================================================


def register_default_packs(registry: Optional[ToolPackRegistry] = None) -> None:
    """Register all default tool packs.

    Args:
        registry: Registry to register packs with (uses global if None)
    """
    if registry is None:
        registry = get_tool_pack_registry()

    # Register base packs first
    base_packs = [
        BASE_FILE_OPS,
        BASE_WEB,
        BASE_GIT,
        BASE_ANALYSIS,
        BASE_EXECUTION,
        BASE_TESTING,
    ]

    for pack in base_packs:
        try:
            registry.register(pack)
        except ValueError:
            pass  # Already registered

    # Register vertical packs
    vertical_packs = [
        CODING_PACK,
        DEVOPS_PACK,
        RAG_PACK,
        DATA_ANALYSIS_PACK,
        RESEARCH_PACK,
    ]

    for pack in vertical_packs:
        try:
            registry.register(pack)
        except ValueError:
            pass  # Already registered

    logger.info(f"Registered {len(registry.list_packs())} tool packs")


def resolve_tool_pack(
    name: str,
    registry: Optional[ToolPackRegistry] = None,
) -> List[str]:
    """Resolve tool pack to list of tool names.

    This is the primary convenience function for verticals to use.

    Args:
        name: Tool pack name to resolve
        registry: Registry to use (uses global if None)

    Returns:
        List of tool names with all inheritance applied

    Example:
        from victor.framework.tool_packs import resolve_tool_pack

        tools = resolve_tool_pack("devops")
        # ['read', 'write', 'edit', 'search', 'grep',
        #  'docker', 'kubernetes', 'terraform', ...]
    """
    if registry is None:
        registry = get_tool_pack_registry()

    return registry.resolve(name)


def create_custom_pack(
    name: str,
    extends: str,
    additional_tools: List[str],
    excludes: Optional[List[str]] = None,
    description: Optional[str] = None,
) -> ToolPack:
    """Create a custom tool pack extending a base pack.

    Convenience function for verticals to create custom packs.

    Args:
        name: Custom pack name
        extends: Base pack name to extend
        additional_tools: Tools to add to base pack
        excludes: Tools to exclude from base pack
        description: Pack description

    Returns:
        New ToolPack instance

    Example:
        custom_pack = create_custom_pack(
            name="my_vertical",
            extends="base_file_ops",
            additional_tools=["my_tool"],
            excludes=["edit"],
        )
    """
    return ToolPack(
        name=name,
        extends=extends,
        tools=additional_tools,
        excludes=excludes or [],
        description=description or f"Custom pack extending {extends}",
    )


# Auto-register default packs on import
register_default_packs()


__all__ = [
    # Classes
    "ToolPack",
    "ToolPackRegistry",
    # Base packs
    "BASE_FILE_OPS",
    "BASE_WEB",
    "BASE_GIT",
    "BASE_ANALYSIS",
    "BASE_EXECUTION",
    "BASE_TESTING",
    # Vertical packs
    "CODING_PACK",
    "DEVOPS_PACK",
    "RAG_PACK",
    "DATA_ANALYSIS_PACK",
    "RESEARCH_PACK",
    # Functions
    "get_tool_pack_registry",
    "register_default_packs",
    "resolve_tool_pack",
    "create_custom_pack",
]
