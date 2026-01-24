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

"""Tool metadata and registry for semantic tool selection."""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.tools.enums import (
    AccessMode,
    CostTier,
    DangerLevel,
    ExecutionCategory,
    Priority,
)

if TYPE_CHECKING:
    from victor.tools.base import BaseTool


@dataclass
class ToolMetadata:
    """Semantic metadata for tool selection and discovery.

    This dataclass allows tools to define their own semantic information
    inline, enabling fully dynamic tool registration without needing to
    manually update tool_knowledge.yaml or tool_selection.py.

    Attributes:
        category: Tool category (e.g., 'git', 'security', 'pipeline')
        keywords: Keywords that trigger this tool in user requests
        use_cases: High-level use cases for semantic matching
        examples: Example requests that should match this tool
        priority_hints: Usage hints for tool selection
        priority: Tool priority for selection (default: MEDIUM)
        access_mode: Access mode for approval tracking (default: READONLY)
        danger_level: Danger level for warnings (default: SAFE)
        stages: Conversation stages where this tool is relevant (e.g., ['reading', 'execution'])
        mandatory_keywords: Keywords that FORCE this tool to be included when matched.
                           Unlike regular keywords, these guarantee inclusion.
                           E.g., ["show diff", "compare"] for shell tool.
        task_types: Task types this tool is relevant for (analysis, action, generation,
                   search, edit, default). Used for classification-aware tool selection.
        progress_params: Parameters that indicate progress in loop detection.
                        If these params change between calls, it's progress not a loop.
                        E.g., ["path", "offset", "limit"] for read tool.
        execution_category: Category for parallel execution and dependency analysis.
                           Determines which tools can safely run concurrently.
    """

    category: str = ""
    keywords: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    priority_hints: List[str] = field(default_factory=list)
    # Selection/approval metadata with sensible defaults
    priority: Optional["Priority"] = None  # Default: MEDIUM when None
    access_mode: Optional["AccessMode"] = None  # Default: READONLY when None
    danger_level: Optional["DangerLevel"] = None  # Default: SAFE when None
    # Stage affinity for conversation state machine
    stages: List[str] = field(default_factory=list)  # e.g., ["initial", "reading", "execution"]
    # NEW: Mandatory keyword triggers that force tool inclusion
    mandatory_keywords: List[str] = field(default_factory=list)  # e.g., ["show diff", "compare"]
    # NEW: Task types for classification-aware selection
    task_types: List[str] = field(default_factory=list)  # e.g., ["analysis", "search", "default"]
    # NEW: Progress parameters for loop detection
    progress_params: List[str] = field(default_factory=list)  # e.g., ["path", "offset", "limit"]
    # NEW: Execution category for parallel execution
    execution_category: Optional["ExecutionCategory"] = None  # Default: READ_ONLY when None

    def __post_init__(self) -> None:
        """Apply defaults for None values to support backward compatibility."""
        if self.priority is None:
            self.priority = Priority.MEDIUM
        if self.access_mode is None:
            self.access_mode = AccessMode.READONLY
        if self.danger_level is None:
            self.danger_level = DangerLevel.SAFE
        if self.execution_category is None:
            self.execution_category = ExecutionCategory.READ_ONLY

    @property
    def is_critical(self) -> bool:
        """Check if this tool has CRITICAL priority."""
        return self.priority == Priority.CRITICAL

    @property
    def requires_approval(self) -> bool:
        """Check if this tool requires user approval."""
        return self.access_mode.requires_approval or self.danger_level.requires_confirmation

    @property
    def is_safe(self) -> bool:
        """Check if this tool is safe (readonly, no danger)."""
        return self.access_mode.is_safe and self.danger_level == DangerLevel.SAFE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for YAML/JSON export."""
        return {
            "category": self.category,
            "keywords": self.keywords,
            "use_cases": self.use_cases,
            "examples": self.examples,
            "priority_hints": self.priority_hints,
            "priority": self.priority.name if self.priority else "MEDIUM",
            "access_mode": self.access_mode.value if self.access_mode else "readonly",
            "danger_level": self.danger_level.value if self.danger_level else "safe",
            "stages": self.stages,
            "mandatory_keywords": self.mandatory_keywords,
            "task_types": self.task_types,
            "progress_params": self.progress_params,
            "execution_category": (
                self.execution_category.value if self.execution_category else "read_only"
            ),
        }

    @classmethod
    def generate_from_tool(
        cls,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        cost_tier: Optional["CostTier"] = None,
    ) -> "ToolMetadata":
        """Auto-generate metadata from tool properties.

        This factory method creates sensible default metadata by extracting
        semantic information from the tool's name, description, and parameters.
        Tools that want more precise control should override the metadata property.

        Args:
            name: Tool name (e.g., 'git', 'file_search')
            description: Tool description text
            parameters: Tool parameters JSON schema
            cost_tier: Optional cost tier for priority hints

        Returns:
            ToolMetadata with auto-generated values
        """
        # Extract category from tool name (first part before underscore)
        name_parts = name.lower().replace("-", "_").split("_")
        category = name_parts[0] if name_parts else "general"

        # Known category mappings for common patterns
        category_mappings = {
            "git": "git",
            "file": "filesystem",
            "filesystem": "filesystem",
            "code": "code",
            "web": "web",
            "docker": "docker",
            "database": "database",
            "db": "database",
            "test": "testing",
            "lint": "code_quality",
            "format": "code_quality",
            "security": "security",
            "mcp": "mcp",
            "lsp": "lsp",
            "refactor": "refactoring",
            "search": "search",
            "semantic": "search",
            "pipeline": "pipeline",
            "ci": "pipeline",
            "coverage": "pipeline",
            "audit": "audit",
            "compliance": "audit",
            "merge": "merge",
            "conflict": "merge",
            "iac": "security",
            "terraform": "security",
            "kubernetes": "security",
        }

        # Try to find a matching category
        for part in name_parts:
            if part in category_mappings:
                category = category_mappings[part]
                break

        # Extract keywords from name and description
        keywords = cls._extract_keywords(name, description)

        # Generate use cases from description
        use_cases = cls._generate_use_cases(name, description)

        # Generate examples from name and parameters
        examples = cls._generate_examples(name, parameters)

        # Generate priority hints from cost tier
        priority_hints = []
        if cost_tier:
            tier_hints = {
                CostTier.FREE: ["Preferred for local operations", "No external costs"],
                CostTier.LOW: ["Efficient compute-only operation"],
                CostTier.MEDIUM: ["Makes external API calls"],
                CostTier.HIGH: ["Resource-intensive operation", "Use sparingly"],
            }
            priority_hints = tier_hints.get(cost_tier, [])

        return cls(
            category=category,
            keywords=keywords,
            use_cases=use_cases,
            examples=examples,
            priority_hints=priority_hints,
        )

    @staticmethod
    def _extract_keywords(name: str, description: str) -> List[str]:
        """Extract keywords from tool name and description.

        Args:
            name: Tool name
            description: Tool description

        Returns:
            List of extracted keywords
        """
        keywords = set()

        # Add name variations
        keywords.add(name.lower())
        keywords.add(name.lower().replace("_", " "))

        # Split camelCase and snake_case
        name_words = re.findall(r"[a-z]+", name.lower())
        keywords.update(name_words)

        # Extract significant words from description (skip common words)
        stopwords = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "and",
            "but",
            "if",
            "or",
            "because",
            "until",
            "while",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "tool",
            "tools",
            "use",
            "using",
        }

        # Extract words from first sentence of description
        first_sentence = description.split(".")[0] if description else ""
        desc_words = re.findall(r"\b[a-z]{3,}\b", first_sentence.lower())
        significant_words = [w for w in desc_words if w not in stopwords]
        keywords.update(significant_words[:5])  # Limit to top 5

        return list(keywords)[:10]  # Cap at 10 keywords

    @staticmethod
    def _generate_use_cases(name: str, description: str) -> List[str]:
        """Generate use cases from description.

        Args:
            name: Tool name
            description: Tool description

        Returns:
            List of use cases
        """
        use_cases = []

        # Primary use case from name
        name_readable = name.replace("_", " ").replace("-", " ")
        use_cases.append(f"{name_readable} operations")

        # Extract action verbs from description
        if description:
            # Look for common action patterns
            action_patterns = [
                (r"\b(create|generate|make)\b", "creating"),
                (r"\b(read|view|show|display|get)\b", "viewing"),
                (r"\b(update|modify|edit|change)\b", "modifying"),
                (r"\b(delete|remove|clear)\b", "removing"),
                (r"\b(search|find|locate|query)\b", "searching"),
                (r"\b(analyze|inspect|check|validate)\b", "analyzing"),
                (r"\b(execute|run|perform)\b", "executing"),
                (r"\b(list|enumerate)\b", "listing"),
            ]

            for pattern, action in action_patterns:
                if re.search(pattern, description.lower()):
                    use_cases.append(f"{action} {name_readable}")
                    break

        return use_cases[:3]  # Limit to 3 use cases

    @staticmethod
    def _generate_examples(name: str, parameters: Dict[str, Any]) -> List[str]:
        """Generate example requests from tool name and parameters.

        Args:
            name: Tool name
            parameters: Tool parameters JSON schema

        Returns:
            List of example requests
        """
        examples = []
        name_readable = name.replace("_", " ").replace("-", " ")

        # Basic example
        examples.append(f"use {name_readable}")

        # Parameter-based examples
        props = parameters.get("properties", {})
        if props:
            # Get first required parameter for more specific example
            required = parameters.get("required", [])
            if required:
                first_param = required[0]
                param_desc = props.get(first_param, {}).get("description", "")
                if param_desc:
                    examples.append(f"{name_readable} with {first_param}")

        return examples[:2]  # Limit to 2 examples


class ToolMetadataRegistry:
    """Centralized registry for tool metadata.

    This singleton class provides a unified interface for accessing tool metadata
    across the application. It supports:
    - Automatic metadata collection from registered tools
    - Hash-based smart reindexing (only reindex when tools change)
    - Caching to avoid regeneration
    - Category and keyword indexing for fast lookup
    - Export for debugging and analysis
    - Plugin tool support (incremental registration)

    Usage:
        registry = ToolMetadataRegistry.get_instance()
        registry.refresh_from_tools(tools)  # Populate from ToolRegistry

        # Check if reindex needed
        if registry.needs_reindex(tools):
            registry.refresh_from_tools(tools)

        # Access metadata
        metadata = registry.get_metadata("git")
        tools_in_category = registry.get_tools_by_category("filesystem")
        tools_with_keyword = registry.get_tools_by_keyword("search")

        # Export all metadata for debugging
        all_metadata = registry.export_all()
    """

    _instance: Optional["ToolMetadataRegistry"] = None

    def __init__(self) -> None:
        """Initialize the registry."""
        self._metadata_cache: Dict[str, ToolMetadata] = {}
        self._category_index: Dict[str, List[str]] = {}  # category -> tool names
        self._keyword_index: Dict[str, List[str]] = {}  # keyword -> tool names
        self._tools_hash: Optional[str] = None  # Hash of registered tools for change detection
        self._last_refresh_count: int = 0  # Number of tools at last refresh

    @classmethod
    def get_instance(cls) -> "ToolMetadataRegistry":
        """Get or create the singleton instance.

        Returns:
            The singleton ToolMetadataRegistry instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None

    @staticmethod
    def _calculate_tools_hash(tools: List["BaseTool"]) -> str:
        """Calculate hash of all tool definitions to detect changes.

        Args:
            tools: List of BaseTool instances

        Returns:
            SHA256 hash of tool definitions (name, description, parameters)
        """
        import hashlib

        # Create deterministic string from all tool definitions
        tool_strings = []
        for tool in sorted(tools, key=lambda t: t.name):
            # Include name, description, and parameters in hash
            tool_string = f"{tool.name}:{tool.description}:{tool.parameters}"
            tool_strings.append(tool_string)

        combined = "|".join(tool_strings)
        return hashlib.sha256(combined.encode()).hexdigest()

    def needs_reindex(self, tools: List["BaseTool"]) -> bool:
        """Check if tools have changed and reindexing is needed.

        Uses hash-based change detection to avoid unnecessary reindexing.
        Returns True if:
        - No previous hash exists (first run)
        - Tool count has changed
        - Tool definitions have changed (hash mismatch)

        Args:
            tools: List of BaseTool instances to check

        Returns:
            True if reindexing is needed, False if cache is valid
        """
        # First run - always needs indexing
        if self._tools_hash is None:
            return True

        # Quick check: tool count changed
        if len(tools) != self._last_refresh_count:
            return True

        # Full check: compute hash and compare
        current_hash = self._calculate_tools_hash(tools)
        return current_hash != self._tools_hash

    def refresh_from_tools(self, tools: List["BaseTool"], force: bool = False) -> bool:
        """Refresh metadata cache from a list of tools.

        Uses smart reindexing: only rebuilds if tools have changed (hash mismatch)
        or if force=True. This enables efficient plugin support where new tools
        can be added without full reindexing.

        Args:
            tools: List of BaseTool instances to collect metadata from
            force: Force reindex even if hash matches (default: False)

        Returns:
            True if reindexing was performed, False if cache was valid
        """
        # Smart reindex: skip if tools haven't changed
        if not force and not self.needs_reindex(tools):
            return False

        # Clear existing cache
        self._metadata_cache.clear()
        self._category_index.clear()
        self._keyword_index.clear()

        # Register all tools
        for tool in tools:
            self.register_tool(tool)

        # Update hash and count for future change detection
        self._tools_hash = self._calculate_tools_hash(tools)
        self._last_refresh_count = len(tools)

        return True

    def register_tool(self, tool: "BaseTool") -> None:
        """Register a single tool's metadata.

        Args:
            tool: BaseTool instance to register
        """
        # Get metadata (explicit or auto-generated)
        metadata = tool.get_metadata()
        self._metadata_cache[tool.name] = metadata

        # Index by category
        if metadata.category:
            if metadata.category not in self._category_index:
                self._category_index[metadata.category] = []
            if tool.name not in self._category_index[metadata.category]:
                self._category_index[metadata.category].append(tool.name)

        # Index by keywords
        for keyword in metadata.keywords:
            keyword_lower = keyword.lower()
            if keyword_lower not in self._keyword_index:
                self._keyword_index[keyword_lower] = []
            if tool.name not in self._keyword_index[keyword_lower]:
                self._keyword_index[keyword_lower].append(tool.name)

    def unregister_tool(self, tool_name: str) -> None:
        """Unregister a tool's metadata.

        Args:
            tool_name: Name of tool to unregister
        """
        if tool_name in self._metadata_cache:
            metadata = self._metadata_cache.pop(tool_name)

            # Remove from category index
            if metadata.category and metadata.category in self._category_index:
                if tool_name in self._category_index[metadata.category]:
                    self._category_index[metadata.category].remove(tool_name)

            # Remove from keyword index
            for keyword in metadata.keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in self._keyword_index:
                    if tool_name in self._keyword_index[keyword_lower]:
                        self._keyword_index[keyword_lower].remove(tool_name)

    def get_metadata(self, tool_name: str) -> Optional[ToolMetadata]:
        """Get metadata for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            ToolMetadata if found, None otherwise
        """
        return self._metadata_cache.get(tool_name)

    def get_all_metadata(self) -> Dict[str, ToolMetadata]:
        """Get all registered metadata.

        Returns:
            Dictionary mapping tool names to ToolMetadata
        """
        return self._metadata_cache.copy()

    def get_tools_by_category(self, category: str) -> List[str]:
        """Get tool names in a specific category.

        Args:
            category: Category name

        Returns:
            List of tool names in the category
        """
        return self._category_index.get(category, []).copy()

    def get_tools_by_keyword(self, keyword: str) -> List[str]:
        """Get tool names matching a keyword.

        Args:
            keyword: Keyword to search for

        Returns:
            List of tool names with this keyword
        """
        return self._keyword_index.get(keyword.lower(), []).copy()

    def get_all_categories(self) -> List[str]:
        """Get all registered categories.

        Returns:
            List of unique category names
        """
        return list(self._category_index.keys())

    def get_all_keywords(self) -> List[str]:
        """Get all registered keywords.

        Returns:
            List of unique keywords
        """
        return list(self._keyword_index.keys())

    def search_tools(self, query: str) -> List[str]:
        """Search for tools matching a query string.

        Searches across tool names, categories, and keywords.

        Args:
            query: Search query

        Returns:
            List of matching tool names (deduplicated)
        """
        query_lower = query.lower()
        matches = set()

        # Direct name match
        for tool_name in self._metadata_cache:
            if query_lower in tool_name.lower():
                matches.add(tool_name)

        # Keyword match
        for keyword, tool_names in self._keyword_index.items():
            if query_lower in keyword:
                matches.update(tool_names)

        # Category match
        for category, tool_names in self._category_index.items():
            if query_lower in category.lower():
                matches.update(tool_names)

        return list(matches)

    def export_all(self) -> Dict[str, Dict[str, Any]]:
        """Export all metadata as dictionaries.

        Useful for debugging, analysis, or generating tool_knowledge.yaml.

        Returns:
            Dictionary mapping tool names to metadata dicts
        """
        return {name: metadata.to_dict() for name, metadata in self._metadata_cache.items()}

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about registered tools and metadata.

        Returns:
            Dictionary with statistics
        """
        total_tools = len(self._metadata_cache)
        tools_with_explicit_metadata = sum(
            1
            for m in self._metadata_cache.values()
            if m.category and m.keywords  # Non-empty = likely explicit
        )

        return {
            "total_tools": total_tools,
            "tools_with_explicit_metadata": tools_with_explicit_metadata,
            "tools_with_auto_metadata": total_tools - tools_with_explicit_metadata,
            "total_categories": len(self._category_index),
            "total_keywords": len(self._keyword_index),
            "categories": list(self._category_index.keys()),
        }

    def get_category_tools_map(self) -> Dict[str, List[str]]:
        """Get mapping of categories to tool names.

        This method returns a dictionary in the same format as the legacy
        TOOL_CATEGORIES constant, enabling migration from hardcoded categories
        to dynamic metadata-based categories.

        Returns:
            Dictionary mapping category names to lists of tool names
        """
        return {category: list(tools) for category, tools in self._category_index.items()}

    def get_tools_for_task_type(self, task_type: str) -> List[str]:
        """Get relevant tools for a task type.

        Maps high-level task types to appropriate categories and returns
        the combined list of tools.

        Args:
            task_type: Task type (edit, search, analyze, design, create, general)

        Returns:
            List of relevant tool names for the task type
        """
        # Task type to category mappings
        task_category_mapping = {
            "edit": ["filesystem", "code", "refactoring", "git"],
            "search": ["search", "code", "code_intel"],
            "analyze": ["code", "pipeline", "security", "audit", "code_quality"],
            "design": ["filesystem", "generation", "code"],
            "create": ["filesystem", "generation", "code", "testing"],
            "general": ["filesystem", "code", "search"],
        }

        # Get categories for this task type
        categories = task_category_mapping.get(task_type, ["filesystem", "code"])

        # Collect tools from all relevant categories
        tools = set()
        for category in categories:
            tools.update(self.get_tools_by_category(category))

        return list(tools)
