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
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.protocols.tool import ITool

logger = logging.getLogger(__name__)


class ToolCapability(Enum):
    """Core tool capabilities.

    Represents functional capabilities that tools can provide.
    Used for grouping tools by their capabilities for easier composition.

    Extended to 30+ capabilities for comprehensive tool coverage.

    File Operations:
        FILE_READ: Read files from filesystem
        FILE_WRITE: Write files to filesystem
        FILE_MANAGEMENT: Manage file operations (copy, move, delete, archive)

    Code Analysis:
        CODE_ANALYSIS: Analyze code structure and patterns
        CODE_SEARCH: Search codebase semantically
        CODE_REVIEW: Review code quality and suggest improvements
        CODE_INTELLIGENCE: Advanced code understanding (autocomplete, etc.)
        CODE_REFACTORING: Automated refactoring operations

    Search & Discovery:
        WEB_SEARCH: Search the web for information
        SEMANTIC_SEARCH: Vector/RAG semantic search
        KNOWLEDGE_BASE: Document ingestion and querying

    Version Control:
        VERSION_CONTROL: Git and version control operations
        CHANGE_MANAGEMENT: Merge, conflicts, PR management

    Infrastructure:
        CONTAINERIZATION: Docker, Kubernetes operations
        CLOUD_INFRA: Terraform, CloudFormation, cloud infrastructure
        CI_CD: CI/CD pipeline operations
        DATABASE: Database operations and queries
        MONITORING: Metrics, logs, alerts

    Development:
        TESTING: Test execution and management
        DOCUMENTATION: Documentation generation and analysis
        DEPENDENCY_MGMT: Package management and dependency analysis
        SCAFFOLDING: Project scaffolding and templates

    Execution:
        CODE_EXECUTION: Run code (Python, etc.)
        BROWSER_AUTOMATION: Browser control and automation
        BASH: Bash shell command execution

    Communication:
        MESSAGING: Slack, Teams notifications
        ISSUE_TRACKING: Jira, GitHub Issues
        NOTIFICATION: Alert management

    Workflow & Automation:
        WORKFLOW_ORCHESTRATION: StateGraph[Any], DAG workflows
        AUTOMATION: Batch processing, scheduling

    Intelligence:
        LSP_INTEGRATION: Language Server Protocol
        AI_ASSISTANCE: LLM-powered features

    Security & Compliance:
        SECURITY_SCANNING: Vulnerability scanning
        COMPLIANCE_AUDIT: Policy checks, compliance audits
        AUDIT: Security and compliance auditing

    Integration:
        API_INTEGRATION: External APIs, HTTP, webhooks

    Utilities:
        CACHE: Caching and memoization
        BATCH: Batch processing of multiple operations
        GRAPH_ANALYSIS: Dependency graphs, call graphs
    """

    # === FILE OPERATIONS ===
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_MANAGEMENT = "file_management"

    # === CODE ANALYSIS ===
    CODE_ANALYSIS = "code_analysis"
    CODE_SEARCH = "code_search"
    CODE_REVIEW = "code_review"
    CODE_INTELLIGENCE = "code_intelligence"
    CODE_REFACTORING = "code_refactoring"

    # === SEARCH & DISCOVERY ===
    WEB_SEARCH = "web_search"
    SEMANTIC_SEARCH = "semantic_search"
    KNOWLEDGE_BASE = "knowledge_base"

    # === VERSION CONTROL ===
    VERSION_CONTROL = "version_control"
    CHANGE_MANAGEMENT = "change_management"

    # === INFRASTRUCTURE ===
    CONTAINERIZATION = "containerization"
    CLOUD_INFRA = "cloud_infra"
    CI_CD = "ci_cd"
    DATABASE = "database"
    MONITORING = "monitoring"

    # === DEVELOPMENT ===
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    DEPENDENCY_MGMT = "dependency_mgmt"
    SCAFFOLDING = "scaffolding"

    # === EXECUTION ===
    CODE_EXECUTION = "code_execution"
    BROWSER_AUTOMATION = "browser_automation"
    BASH = "bash"

    # === COMMUNICATION ===
    MESSAGING = "messaging"
    ISSUE_TRACKING = "issue_tracking"
    NOTIFICATION = "notification"

    # === WORKFLOW & AUTOMATION ===
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"
    AUTOMATION = "automation"

    # === INTELLIGENCE ===
    LSP_INTEGRATION = "lsp_integration"
    AI_ASSISTANCE = "ai_assistance"

    # === SECURITY & COMPLIANCE ===
    SECURITY_SCANNING = "security_scanning"
    COMPLIANCE_AUDIT = "compliance_audit"
    AUDIT = "audit"

    # === INTEGRATION ===
    API_INTEGRATION = "api_integration"

    # === UTILITIES ===
    CACHE = "cache"
    BATCH = "batch"
    GRAPH_ANALYSIS = "graph_analysis"


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
    - Auto-discover capabilities from tool metadata

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

        # Auto-discover from tools
        await registry.auto_discover_capabilities(tools)
    """

    # Category to capability mapping for auto-discovery
    CATEGORY_CAPABILITY_MAP: Dict[str, ToolCapability] = {
        "git": ToolCapability.VERSION_CONTROL,
        "filesystem": ToolCapability.FILE_READ,
        "docker": ToolCapability.CONTAINERIZATION,
        "database": ToolCapability.DATABASE,
        "testing": ToolCapability.TESTING,
        "test": ToolCapability.TESTING,
        "pipeline": ToolCapability.CI_CD,
        "cicd": ToolCapability.CI_CD,
        "ci": ToolCapability.CI_CD,
        "security": ToolCapability.SECURITY_SCANNING,
        "audit": ToolCapability.AUDIT,
        "compliance": ToolCapability.COMPLIANCE_AUDIT,
        "cache": ToolCapability.CACHE,
        "batch": ToolCapability.BATCH,
        "web": ToolCapability.WEB_SEARCH,
        "search": ToolCapability.CODE_SEARCH,
        "semantic": ToolCapability.SEMANTIC_SEARCH,
        "code": ToolCapability.CODE_ANALYSIS,
        "refactoring": ToolCapability.CODE_REFACTORING,
        "refactor": ToolCapability.CODE_REFACTORING,
        "review": ToolCapability.CODE_REVIEW,
        "documentation": ToolCapability.DOCUMENTATION,
        "doc": ToolCapability.DOCUMENTATION,
        "scaffolding": ToolCapability.SCAFFOLDING,
        "bash": ToolCapability.BASH,
        "browser": ToolCapability.BROWSER_AUTOMATION,
        "lsp": ToolCapability.LSP_INTEGRATION,
        "monitoring": ToolCapability.MONITORING,
        "metrics": ToolCapability.MONITORING,
        "cloud": ToolCapability.CLOUD_INFRA,
        "infra": ToolCapability.CLOUD_INFRA,
        "iac": ToolCapability.CLOUD_INFRA,
        "terraform": ToolCapability.CLOUD_INFRA,
        "kubernetes": ToolCapability.CONTAINERIZATION,
        "merge": ToolCapability.CHANGE_MANAGEMENT,
        "conflict": ToolCapability.CHANGE_MANAGEMENT,
        "issue": ToolCapability.ISSUE_TRACKING,
        "jira": ToolCapability.ISSUE_TRACKING,
        "slack": ToolCapability.MESSAGING,
        "messaging": ToolCapability.MESSAGING,
        "notification": ToolCapability.NOTIFICATION,
        "workflow": ToolCapability.WORKFLOW_ORCHESTRATION,
        "automation": ToolCapability.AUTOMATION,
        "graph": ToolCapability.GRAPH_ANALYSIS,
        "dependency": ToolCapability.DEPENDENCY_MGMT,
        "api": ToolCapability.API_INTEGRATION,
        "http": ToolCapability.API_INTEGRATION,
        "execution": ToolCapability.CODE_EXECUTION,
    }

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
            raise ValueError(f"Capability {definition.name.value} already registered")

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
                dep_tools = self.get_tools_for_capability(dep_capability, include_dependencies=True)
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

    def resolve_dependencies(self, capabilities: List[ToolCapability]) -> List[ToolCapability]:
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

    async def auto_discover_capabilities(self, tools: List["ITool"]) -> None:
        """Automatically register capabilities based on tool metadata.

        Analyzes tool metadata (category, keywords) and maps tools to appropriate
        capabilities. Tools are added to existing capabilities without overriding
        manually registered capability definitions.

        Discovery Strategy:
        1. Extract tool metadata (explicit or auto-generated)
        2. Map tool category to capability using CATEGORY_CAPABILITY_MAP
        3. Add tool to appropriate capability (creating capability if needed)
        4. Log warnings for tools without clear mapping

        Args:
            tools: List of ITool instances to discover capabilities from

        Example:
            registry = CapabilityRegistry()
            await registry.auto_discover_capabilities(my_tools)

        Note:
            This method is async to support future extensions (e.g., loading
            capability mappings from external sources).
        """
        for tool in tools:
            # Get tool metadata
            metadata = tool.get_metadata()
            category = metadata.category.lower() if metadata.category else ""

            # Try to map category to capability
            capability = None

            if category in self.CATEGORY_CAPABILITY_MAP:
                capability = self.CATEGORY_CAPABILITY_MAP[category]
            else:
                # Try to match from keywords
                for keyword in metadata.keywords:
                    keyword_lower = keyword.lower()
                    if keyword_lower in self.CATEGORY_CAPABILITY_MAP:
                        capability = self.CATEGORY_CAPABILITY_MAP[keyword_lower]
                        break

            # If capability found, register tool
            if capability:
                if capability in self._capabilities:
                    # Add to existing capability
                    definition = self._capabilities[capability]
                    if tool.name not in definition.tools:
                        definition.tools.append(tool.name)
                        logger.debug(f"Added tool '{tool.name}' to capability '{capability.value}'")
                else:
                    # Create new capability definition
                    definition = CapabilityDefinition(
                        name=capability,
                        description=f"Auto-discovered {capability.value} capability",
                        tools=[tool.name],
                        dependencies=[],
                        conflicts=[],
                    )
                    self._capabilities[capability] = definition
                    logger.debug(f"Created capability '{capability.value}' for tool '{tool.name}'")
            else:
                # Log warning for unmapped tool
                logger.warning(
                    f"Could not map tool '{tool.name}' (category: {category}) "
                    f"to any capability. Keywords: {metadata.keywords[:3]}"
                )

    def get_capability_for_tool(self, tool_name: str) -> Optional[ToolCapability]:
        """Find which capability a tool belongs to.

        Searches through all registered capabilities to find which one contains
        the given tool. Returns the first matching capability.

        Args:
            tool_name: Name of the tool to find

        Returns:
            ToolCapability if found, None otherwise

        Example:
            capability = registry.get_capability_for_tool("git")
            assert capability == ToolCapability.VERSION_CONTROL
        """
        for capability, definition in self._capabilities.items():
            if tool_name in definition.tools:
                return capability
        return None

    def get_all_tools(self) -> Set[str]:
        """Get all unique tools across all capabilities.

        Collects all tool names from all registered capabilities and returns
        them as a set (no duplicates).

        Returns:
            Set of unique tool names

        Example:
            all_tools = registry.get_all_tools()
            print(f"Total unique tools: {len(all_tools)}")
        """
        all_tools: Set[str] = set()
        for definition in self._capabilities.values():
            all_tools.update(definition.tools)
        return all_tools


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
        resolved_capabilities = self._registry.resolve_dependencies(required_capabilities)

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

    def recommend_capabilities(self, task_description: str) -> List[ToolCapability]:
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
            kw in description_lower for kw in ["read", "file", "open", "view", "cat", "display"]
        ):
            recommendations.append(ToolCapability.FILE_READ)

        if any(kw in description_lower for kw in ["write", "edit", "save", "create"]):
            recommendations.append(ToolCapability.FILE_WRITE)

        if any(kw in description_lower for kw in ["copy", "move", "delete", "remove", "manage"]):
            recommendations.append(ToolCapability.FILE_MANAGEMENT)

        # Code operation keywords
        if any(
            kw in description_lower
            for kw in ["analyz", "code", "syntax", "structure", "pattern", "python", "function"]
        ):
            recommendations.append(ToolCapability.CODE_ANALYSIS)

        if any(kw in description_lower for kw in ["search", "find", "grep", "lookup"]):
            recommendations.append(ToolCapability.CODE_SEARCH)

        if any(kw in description_lower for kw in ["review", "quality", "improv", "refactor"]):
            recommendations.append(ToolCapability.CODE_REVIEW)
            recommendations.append(ToolCapability.CODE_ANALYSIS)

        # Refactoring keywords
        if any(kw in description_lower for kw in ["refactor", "restructure", "reorganize"]):
            recommendations.append(ToolCapability.CODE_REFACTORING)

        # Web keywords
        if any(kw in description_lower for kw in ["web", "internet", "online"]):
            recommendations.append(ToolCapability.WEB_SEARCH)

        # Version control keywords
        if any(kw in description_lower for kw in ["git", "commit", "branch", "merge"]):
            recommendations.append(ToolCapability.VERSION_CONTROL)

        # Testing keywords
        if any(kw in description_lower for kw in ["test", "spec", "unit test", "pytest"]):
            recommendations.append(ToolCapability.TESTING)

        # Documentation keywords
        if any(kw in description_lower for kw in ["doc", "document", "readme"]):
            recommendations.append(ToolCapability.DOCUMENTATION)

        # Docker/container keywords
        if any(
            kw in description_lower for kw in ["docker", "container", "kubernetes", "k8s", "deploy"]
        ):
            recommendations.append(ToolCapability.CONTAINERIZATION)

        # CI/CD keywords
        if any(kw in description_lower for kw in ["pipeline", "ci/cd", "cicd", "build", "release"]):
            recommendations.append(ToolCapability.CI_CD)

        # Bash/shell keywords
        if any(kw in description_lower for kw in ["bash", "shell", "command", "terminal"]):
            recommendations.append(ToolCapability.BASH)

        return recommendations

    def get_capability_summary(self) -> Dict[str, List[str]]:
        """Return summary of all capabilities and their tools.

        Returns a dictionary mapping capability names to lists of tool names
        that provide that capability. Useful for debugging and visualization.

        Returns:
            Dictionary with capability names as keys and tool lists as values

        Example:
            selector = CapabilitySelector()
            summary = selector.get_capability_summary()
            # {
            #   "file_read": ["read", "cat", "ls"],
            #   "file_write": ["write", "edit"],
            #   ...
            # }
        """
        summary = {}

        for capability, definition in self._registry._capabilities.items():
            summary[capability.value] = definition.tools.copy()

        return summary


__all__ = [
    "ToolCapability",
    "CapabilityDefinition",
    "CapabilityRegistry",
    "CapabilitySelector",
]
