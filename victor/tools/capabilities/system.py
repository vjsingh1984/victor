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

"""Tool Capability System.

Groups tools by functional capabilities for easier composition and
reusability across verticals.

Design Patterns:
    - Enum Pattern: ToolCapability for type-safe capability identifiers
    - Registry Pattern: CapabilityRegistry for capability management
    - Strategy Pattern: CapabilitySelector for tool selection strategies

Usage:
    from victor.tools.capabilities import ToolCapability, CapabilitySelector

    # Create selector with built-in capabilities
    selector = CapabilitySelector()

    # Select tools for file operations
    tools = selector.select_tools(
        required_capabilities=[ToolCapability.FILE_READ, ToolCapability.FILE_WRITE]
    )

Phase 2, Work Stream 2.2: Tool Capability Groups
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ToolCapability(Enum):
    """Core tool capabilities.

    Represents functional capabilities that tools can provide.
    Used for grouping tools by their capabilities for easier composition.

    Capabilities:
        FILE_READ: Read files from filesystem
        FILE_WRITE: Write files to filesystem
        FILE_MANAGEMENT: Manage file operations (copy, move, delete)
        WEB_SEARCH: Search the web for information
        CODE_ANALYSIS: Analyze code structure and patterns
        CODE_SEARCH: Search codebase semantically
        CODE_REVIEW: Review code quality and suggest improvements
        CODE_INTELLIGENCE: Advanced code understanding (autocomplete, etc.)
        VERSION_CONTROL: Git and version control operations
        DATABASE: Database operations and queries
        DOCKER: Docker container management
        CI_CD: CI/CD pipeline operations
        TESTING: Test execution and management
        DOCUMENTATION: Documentation generation and analysis
        BASH: Bash shell command execution
        BROWSER: Browser automation and control
        CACHE: Caching and memoization
        BATCH: Batch processing of multiple operations
        AUDIT: Security and compliance auditing
        DEPENDENCY: Dependency management and analysis
    """

    # File capabilities
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_MANAGEMENT = "file_management"

    # Code capabilities
    CODE_ANALYSIS = "code_analysis"
    CODE_SEARCH = "code_search"
    CODE_REVIEW = "code_review"
    CODE_INTELLIGENCE = "code_intelligence"

    # Search capabilities
    WEB_SEARCH = "web_search"

    # Infrastructure capabilities
    VERSION_CONTROL = "version_control"
    DATABASE = "database"
    DOCKER = "docker"
    CI_CD = "ci_cd"

    # Development capabilities
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    DEPENDENCY = "dependency"

    # Execution capabilities
    BASH = "bash"
    BROWSER = "browser"

    # Utility capabilities
    CACHE = "cache"
    BATCH = "batch"
    AUDIT = "audit"

    # Additional capabilities for extensibility
    LSP = "lsp"
    WORKFLOW = "workflow"
    INTELLIGENCE = "intelligence"


@dataclass
class CapabilityDefinition:
    """Definition of a tool capability.

    Attributes:
        name: Capability identifier
        description: Human-readable description
        tools: List of tools that provide this capability
        dependencies: Capabilities that must be included with this one
        conflicts: Capabilities that cannot be used with this one
    """

    name: ToolCapability
    description: str
    tools: List[str]
    dependencies: List[ToolCapability] = field(default_factory=list)
    conflicts: List[ToolCapability] = field(default_factory=list)


class CapabilityRegistry:
    """Registry for tool capability definitions.

    Manages capability definitions and provides lookup and validation
    functionality.

    Responsibilities:
    - Register capability definitions
    - Resolve capability dependencies
    - Check for capability conflicts
    - Query tools by capability

    Example:
        registry = CapabilityRegistry()

        # Register capability
        definition = CapabilityDefinition(
            name=ToolCapability.FILE_READ,
            description="Read files",
            tools=["read", "ls"],
            dependencies=[],
            conflicts=[]
        )
        registry.register_capability(definition)

        # Get tools with dependencies
        tools = registry.get_tools_for_capability(
            ToolCapability.FILE_WRITE,
            include_dependencies=True
        )
    """

    def __init__(self) -> None:
        """Initialize the capability registry."""
        self._capabilities: Dict[ToolCapability, CapabilityDefinition] = {}

    def register_capability(self, definition: CapabilityDefinition) -> None:
        """Register a capability definition.

        Args:
            definition: Capability definition to register

        Raises:
            ValueError: If capability already registered
        """
        if definition.name in self._capabilities:
            raise ValueError(
                f"Capability {definition.name.value} already registered"
            )

        self._capabilities[definition.name] = definition
        logger.debug(f"Registered capability: {definition.name.value}")

    def get_tools_for_capability(
        self, capability: ToolCapability, include_dependencies: bool = True
    ) -> Set[str]:
        """Get tools for a capability.

        Args:
            capability: Capability to get tools for
            include_dependencies: Whether to include dependency tools

        Returns:
            Set of tool names

        Example:
            tools = registry.get_tools_for_capability(
                ToolCapability.FILE_WRITE,
                include_dependencies=True
            )
        """
        if capability not in self._capabilities:
            logger.warning(f"Capability {capability.value} not registered")
            return set()

        definition = self._capabilities[capability]
        tools = set(definition.tools)

        if include_dependencies:
            for dep_capability in definition.dependencies:
                dep_tools = self.get_tools_for_capability(
                    dep_capability, include_dependencies=True
                )
                tools.update(dep_tools)

        return tools

    def check_conflicts(
        self, capabilities: List[ToolCapability]
    ) -> List[Tuple[ToolCapability, ToolCapability]]:
        """Check for conflicts between capabilities.

        Args:
            capabilities: List of capabilities to check

        Returns:
            List of conflicting capability pairs

        Example:
            conflicts = registry.check_conflicts([READ_ONLY, FILE_WRITE])
            if conflicts:
                print(f"Found {len(conflicts)} conflicts")
        """
        conflicts = []

        for i, cap1 in enumerate(capabilities):
            for cap2 in capabilities[i + 1 :]:
                # Check if cap1 conflicts with cap2
                if cap1 in self._capabilities:
                    def1 = self._capabilities[cap1]
                    if cap2 in def1.conflicts:
                        conflicts.append((cap1, cap2))

                # Check if cap2 conflicts with cap1
                if cap2 in self._capabilities:
                    def2 = self._capabilities[cap2]
                    if cap1 in def2.conflicts:
                        conflicts.append((cap2, cap1))

        return conflicts

    def resolve_dependencies(
        self, capabilities: List[ToolCapability]
    ) -> List[ToolCapability]:
        """Resolve capability dependencies.

        Returns a list of capabilities including all dependencies,
        with no duplicates.

        Args:
            capabilities: List of capabilities to resolve

        Returns:
            List of capabilities with dependencies included

        Example:
            required = [ToolCapability.FILE_WRITE]
            all_needed = registry.resolve_dependencies(required)
            # Returns [FILE_WRITE, FILE_READ]
        """
        resolved: List[ToolCapability] = []
        seen: Set[ToolCapability] = set()

        def add_capability(cap: ToolCapability) -> None:
            """Recursively add capability and its dependencies."""
            if cap in seen:
                return

            seen.add(cap)

            # First add dependencies
            if cap in self._capabilities:
                for dep in self._capabilities[cap].dependencies:
                    add_capability(dep)

            # Then add the capability itself
            if cap not in resolved:
                resolved.append(cap)

        for capability in capabilities:
            add_capability(capability)

        return resolved


class CapabilitySelector:
    """Selects tools based on capabilities.

    Provides high-level tool selection functionality using capability
    definitions from the registry.

    Responsibilities:
    - Select tools for required capabilities
    - Exclude specific tools from selection
    - Recommend capabilities for task descriptions

    Example:
        selector = CapabilitySelector()

        # Select tools for file operations
        tools = selector.select_tools(
            required_capabilities=[ToolCapability.FILE_READ],
            excluded_tools={"grep"}  # Don't include grep
        )
    """

    def __init__(self, registry: Optional[CapabilityRegistry] = None) -> None:
        """Initialize the capability selector.

        Args:
            registry: Optional capability registry (creates empty if None)
        """
        self._registry = registry or CapabilityRegistry()

    def select_tools(
        self,
        required_capabilities: List[ToolCapability],
        excluded_tools: Optional[Set[str]] = None,
    ) -> List[str]:
        """Select tools based on required capabilities.

        Args:
            required_capabilities: List of required capabilities
            excluded_tools: Tools to exclude from selection

        Returns:
            List of tool names

        Example:
            selector = CapabilitySelector()
            tools = selector.select_tools(
                required_capabilities=[
                    ToolCapability.FILE_READ,
                    ToolCapability.FILE_WRITE
                ],
                excluded_tools={"cat"}  # Exclude cat tool
            )
        """
        if not required_capabilities:
            return []

        # Resolve dependencies
        resolved_capabilities = self._registry.resolve_dependencies(
            required_capabilities
        )

        # Collect tools from all capabilities
        tool_set: Set[str] = set()
        for capability in resolved_capabilities:
            tools = self._registry.get_tools_for_capability(capability)
            tool_set.update(tools)

        # Remove excluded tools
        if excluded_tools:
            tool_set -= excluded_tools

        # Return as list for deterministic ordering
        return sorted(tool_set)

    def recommend_capabilities(
        self, task_description: str
    ) -> List[ToolCapability]:
        """Recommend capabilities for a task description.

        Analyzes task description and recommends relevant capabilities.
        This is a simple keyword-based implementation.

        Args:
            task_description: Description of the task

        Returns:
            List of recommended capabilities

        Example:
            capabilities = selector.recommend_capabilities(
                "Read and edit Python files"
            )
            # Returns [FILE_READ, FILE_WRITE, CODE_ANALYSIS]
        """
        # Simple keyword-based recommendation
        description_lower = task_description.lower()

        recommendations = []

        # File operation keywords
        if any(
            kw in description_lower
            for kw in ["read", "file", "open", "view", "cat", "display"]
        ):
            recommendations.append(ToolCapability.FILE_READ)

        if any(kw in description_lower for kw in ["write", "edit", "save", "create"]):
            recommendations.append(ToolCapability.FILE_WRITE)

        if any(
            kw in description_lower
            for kw in ["copy", "move", "delete", "remove", "manage"]
        ):
            recommendations.append(ToolCapability.FILE_MANAGEMENT)

        # Code operation keywords
        if any(
            kw in description_lower
            for kw in ["analyz", "code", "syntax", "structure", "pattern"]
        ):
            recommendations.append(ToolCapability.CODE_ANALYSIS)

        if any(
            kw in description_lower for kw in ["search", "find", "grep", "lookup"]
        ):
            recommendations.append(ToolCapability.CODE_SEARCH)

        if any(
            kw in description_lower
            for kw in ["review", "quality", "improv", "refactor"]
        ):
            recommendations.append(ToolCapability.CODE_REVIEW)

        # Web keywords
        if any(kw in description_lower for kw in ["web", "internet", "online"]):
            recommendations.append(ToolCapability.WEB_SEARCH)

        # Version control keywords
        if any(
            kw in description_lower for kw in ["git", "commit", "branch", "merge"]
        ):
            recommendations.append(ToolCapability.VERSION_CONTROL)

        # Testing keywords
        if any(kw in description_lower for kw in ["test", "spec", "unit test"]):
            recommendations.append(ToolCapability.TESTING)

        # Documentation keywords
        if any(kw in description_lower for kw in ["doc", "document", "readme"]):
            recommendations.append(ToolCapability.DOCUMENTATION)

        return recommendations


__all__ = [
    "ToolCapability",
    "CapabilityDefinition",
    "CapabilityRegistry",
    "CapabilitySelector",
]
