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

"""Base tool framework for CodingAgent."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Union, runtime_checkable

import jsonschema
from jsonschema import Draft7Validator, ValidationError as JsonSchemaValidationError
from pydantic import BaseModel, Field


class CostTier(Enum):
    """Cost tier for tools.

    Used for cost-aware tool selection to deprioritize expensive tools
    when cheaper alternatives exist.

    Tiers:
        FREE: Local operations with no external costs (filesystem, bash, git)
        LOW: Compute-only operations (code review, refactoring analysis)
        MEDIUM: External API calls (web search, web fetch)
        HIGH: Resource-intensive operations (batch processing 100+ files)
    """

    FREE = "free"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def weight(self) -> float:
        """Return numeric weight for cost comparison."""
        weights = {
            CostTier.FREE: 0.0,
            CostTier.LOW: 1.0,
            CostTier.MEDIUM: 2.0,
            CostTier.HIGH: 3.0,
        }
        return weights[self]


class Priority(Enum):
    """Tool priority levels for selection availability.

    Decoupled from category - determines when tools are included in LLM context.
    Higher priority tools are more likely to be included in limited contexts.

    Priority Levels:
        CRITICAL (1): Always available - essential for any task
                      Examples: read, ls, grep, shell
        HIGH (2): Included for most tasks - important operations
                  Examples: write, edit, git
        MEDIUM (3): Task-specific - included when relevant
                    Examples: docker, db, test
        LOW (4): Specialized - only when specifically needed
                 Examples: batch, scaffold, workflow
        CONTEXTUAL (5): Only included based on task classification
                        Examples: web_search (only for research tasks)
    """

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    CONTEXTUAL = 5

    @property
    def weight(self) -> int:
        """Lower weight = higher priority (more likely to be selected)."""
        return self.value

    def should_include_for_task(self, task_type: str) -> bool:
        """Check if this priority level should be included for a task type.

        Args:
            task_type: Task type (e.g., "analysis", "edit", "search")

        Returns:
            True if tool should be included
        """
        # CRITICAL and HIGH are always included
        if self.value <= 2:
            return True
        # MEDIUM included for most tasks
        if self.value == 3:
            return task_type != "simple"
        # LOW and CONTEXTUAL need specific conditions
        return False


class AccessMode(Enum):
    """Tool access mode for approval tracking and security policies.

    Determines what kind of access a tool requires, enabling:
    - Auto-approval for read-only tools in safe mode
    - Confirmation prompts for write operations
    - Strict approval for dangerous operations

    Access Modes:
        READONLY: Only reads data, no side effects
                  Examples: read, ls, grep, search
        WRITE: Modifies files or state
               Examples: write, edit
        EXECUTE: Runs external commands or code
                 Examples: shell, sandbox
        NETWORK: Makes external network calls
                 Examples: web_search, web_fetch, api calls
        MIXED: Multiple access types in same operation
               Examples: git (read+write+network), docker (execute+network)
    """

    READONLY = "readonly"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    MIXED = "mixed"

    @property
    def requires_approval(self) -> bool:
        """Check if this access mode requires user approval by default."""
        return self in {AccessMode.WRITE, AccessMode.EXECUTE, AccessMode.MIXED}

    @property
    def is_safe(self) -> bool:
        """Check if this access mode is considered safe (no side effects)."""
        return self == AccessMode.READONLY

    def approval_level(self) -> int:
        """Get approval level (higher = stricter approval needed).

        Returns:
            0: Auto-approve (readonly)
            1: Log only (network)
            2: Confirm on first use (write)
            3: Always confirm (execute, mixed)
        """
        levels = {
            AccessMode.READONLY: 0,
            AccessMode.NETWORK: 1,
            AccessMode.WRITE: 2,
            AccessMode.EXECUTE: 3,
            AccessMode.MIXED: 3,
        }
        return levels[self]


class ExecutionCategory(Enum):
    """Execution category for parallel tool execution and dependency analysis.

    Used by parallel_executor to determine which tools can safely run concurrently.
    Replaces the static TOOL_CATEGORIES dictionary with decorator-driven metadata.

    Categories:
        READ_ONLY: Pure read operations, safe to parallelize (read, ls, grep, search)
        WRITE: File modification operations, may conflict (write, edit)
        COMPUTE: CPU-intensive but isolated (code_review, refactor analysis)
        NETWORK: External network calls, rate-limited (web_search, api calls)
        EXECUTE: Shell commands, may have side effects (shell, sandbox)
        MIXED: Multiple categories, needs careful dependency analysis (git, docker)
    """

    READ_ONLY = "read_only"
    WRITE = "write"
    COMPUTE = "compute"
    NETWORK = "network"
    EXECUTE = "execute"
    MIXED = "mixed"

    @property
    def can_parallelize(self) -> bool:
        """Check if this category is safe to run in parallel with itself."""
        return self in {
            ExecutionCategory.READ_ONLY,
            ExecutionCategory.COMPUTE,
            ExecutionCategory.NETWORK,
        }

    @property
    def conflicts_with(self) -> set:
        """Get categories that conflict with this one (shouldn't run together)."""
        conflict_map = {
            ExecutionCategory.READ_ONLY: set(),  # Safe with everything
            ExecutionCategory.WRITE: {ExecutionCategory.WRITE, ExecutionCategory.MIXED},
            ExecutionCategory.COMPUTE: set(),  # Safe with everything
            ExecutionCategory.NETWORK: set(),  # Safe with everything (rate limiting handled separately)
            ExecutionCategory.EXECUTE: {
                ExecutionCategory.EXECUTE,
                ExecutionCategory.WRITE,
                ExecutionCategory.MIXED,
            },
            ExecutionCategory.MIXED: {
                ExecutionCategory.WRITE,
                ExecutionCategory.EXECUTE,
                ExecutionCategory.MIXED,
            },
        }
        return conflict_map.get(self, set())


class DangerLevel(Enum):
    """Danger level for potentially destructive operations.

    Indicates the potential impact of tool misuse or errors.
    Used for warning messages and approval escalation.

    Danger Levels:
        SAFE: No risk of data loss or system damage
              Examples: read, ls, grep
        LOW: Minor risk, easily reversible
             Examples: write (single file), edit
        MEDIUM: Moderate risk, may require effort to reverse
                Examples: git push, db queries
        HIGH: Significant risk, difficult to reverse
              Examples: shell (arbitrary commands), db drop, docker rm
        CRITICAL: Irreversible or system-wide impact
                  Examples: rm -rf, format disk, drop database
    """

    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def weight(self) -> int:
        """Numeric weight for danger comparison."""
        weights = {
            DangerLevel.SAFE: 0,
            DangerLevel.LOW: 1,
            DangerLevel.MEDIUM: 2,
            DangerLevel.HIGH: 3,
            DangerLevel.CRITICAL: 4,
        }
        return weights[self]

    @property
    def requires_confirmation(self) -> bool:
        """Check if this danger level requires explicit user confirmation."""
        return self.weight >= 2  # MEDIUM and above

    @property
    def warning_message(self) -> str:
        """Get appropriate warning message for this danger level."""
        messages = {
            DangerLevel.SAFE: "",
            DangerLevel.LOW: "This operation modifies data.",
            DangerLevel.MEDIUM: "⚠️ This operation may be difficult to reverse.",
            DangerLevel.HIGH: "⚠️ WARNING: This is a potentially destructive operation.",
            DangerLevel.CRITICAL: "🚨 DANGER: This operation may cause irreversible changes!",
        }
        return messages[self]


class HookError(Exception):
    """Raised when a critical hook fails."""

    def __init__(self, hook_name: str, original_error: Exception, tool_name: str = ""):
        self.hook_name = hook_name
        self.original_error = original_error
        self.tool_name = tool_name
        super().__init__(
            f"Critical hook '{hook_name}' failed for tool '{tool_name}': {original_error}"
        )


class Hook:
    """Tool execution hook with metadata.

    Hooks can be marked as critical, meaning their failure will prevent
    tool execution (useful for safety checks, validation, etc.).
    """

    def __init__(
        self,
        callback: Callable,
        name: str = "",
        critical: bool = False,
        description: str = "",
    ):
        """Initialize hook.

        Args:
            callback: The hook function to call
            name: Human-readable name for the hook
            critical: If True, hook failure blocks tool execution
            description: Description of what the hook does
        """
        self.callback = callback
        self.name = name or callback.__name__
        self.critical = critical
        self.description = description

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the hook callback."""
        return self.callback(*args, **kwargs)


@dataclass
class ValidationResult:
    """Result of parameter validation.

    Provides detailed information about validation failures including
    which parameters failed and why.
    """

    valid: bool
    errors: List[str] = field(default_factory=list)
    invalid_params: Dict[str, str] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Allow using ValidationResult in boolean context."""
        return self.valid

    @classmethod
    def success(cls) -> "ValidationResult":
        """Create a successful validation result."""
        return cls(valid=True)

    @classmethod
    def failure(
        cls, errors: List[str], invalid_params: Optional[Dict[str, str]] = None
    ) -> "ValidationResult":
        """Create a failed validation result."""
        return cls(valid=False, errors=errors, invalid_params=invalid_params or {})


class ToolParameter(BaseModel):
    """Tool parameter definition."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (string, number, boolean, etc.)")
    description: str = Field(..., description="Parameter description")
    enum: Optional[list[str]] = Field(default=None, description="Allowed values for enum types")
    required: bool = Field(default=True, description="Whether parameter is required")


class ToolResult(BaseModel):
    """Result from tool execution."""

    success: bool = Field(..., description="Whether execution succeeded")
    output: Any = Field(..., description="Tool output data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class ToolConfig:
    """Configuration container for tools.

    This replaces global state by providing a centralized configuration
    object that can be passed through context. Tools should access config
    from context['tool_config'] instead of using module-level globals.

    Example usage in tools:
        ```python
        async def git(operation: str, context: Dict[str, Any], **kwargs):
            config = context.get('tool_config')
            if config and config.provider:
                # Use provider for AI features
                pass
        ```

    Example setup in orchestrator:
        ```python
        config = ToolConfig(provider=my_provider, model="gpt-4")
        executor.update_context(tool_config=config)
        ```
    """

    def __init__(
        self,
        provider: Optional[Any] = None,
        model: Optional[str] = None,
        max_complexity: int = 10,
        web_fetch_top: Optional[int] = None,
        web_fetch_pool: Optional[int] = None,
        max_content_length: int = 5000,
        batch_concurrency: int = 5,
        batch_max_files: int = 100,
    ):
        """Initialize tool configuration.

        Args:
            provider: LLM provider for AI-powered features (commit messages, summaries)
            model: Model name to use with the provider
            max_complexity: Maximum cyclomatic complexity threshold for code review
            web_fetch_top: Number of top results to fetch for web search
            web_fetch_pool: Pool size for concurrent web fetches
            max_content_length: Maximum content length for web scraping
            batch_concurrency: Concurrent operations for batch processing
            batch_max_files: Maximum files for batch operations
        """
        self.provider = provider
        self.model = model
        self.max_complexity = max_complexity
        self.web_fetch_top = web_fetch_top
        self.web_fetch_pool = web_fetch_pool
        self.max_content_length = max_content_length
        self.batch_concurrency = batch_concurrency
        self.batch_max_files = batch_max_files

    @classmethod
    def from_context(cls, context: Dict[str, Any]) -> Optional["ToolConfig"]:
        """Extract ToolConfig from context dictionary.

        Args:
            context: Context dictionary passed to tools

        Returns:
            ToolConfig if present in context, None otherwise
        """
        return context.get("tool_config") if context else None


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

    def __post_init__(self):
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


@runtime_checkable
class ToolMetadataProvider(Protocol):
    """Protocol for tools that provide semantic metadata.

    All tools should implement this protocol to enable dynamic discovery
    and semantic tool selection. The protocol enforces that every tool
    can provide metadata either explicitly or through auto-generation.

    Example:
        class MyTool(BaseTool):
            # Explicit metadata (recommended for precise control)
            @property
            def metadata(self) -> ToolMetadata:
                return ToolMetadata(
                    category="my_category",
                    keywords=["keyword1", "keyword2"],
                    use_cases=["use case 1", "use case 2"],
                )

        # Or rely on auto-generation (good for simple tools):
        class SimpleTool(BaseTool):
            # metadata property returns None, get_metadata() auto-generates
            pass
    """

    @property
    def name(self) -> str:
        """Tool name."""
        ...

    @property
    def description(self) -> str:
        """Tool description."""
        ...

    @property
    def parameters(self) -> Dict[str, Any]:
        """Tool parameters JSON schema."""
        ...

    @property
    def cost_tier(self) -> "CostTier":
        """Cost tier for the tool."""
        ...

    def get_metadata(self) -> ToolMetadata:
        """Get semantic metadata for tool selection.

        This method MUST return valid ToolMetadata. Implementations should:
        1. Return explicit metadata if defined (via metadata property)
        2. Auto-generate metadata from tool properties if not explicit

        Returns:
            ToolMetadata with semantic information for tool selection
        """
        ...


class BaseTool(ABC):
    """Abstract base class for all tools.

    Tools should implement name, description, parameters, and execute().
    Optionally, tools can override the metadata property to provide
    semantic information for dynamic tool selection.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema for tool parameters."""
        pass

    @property
    def metadata(self) -> Optional[ToolMetadata]:
        """Semantic metadata for tool selection.

        Override this property to provide category, keywords, use_cases,
        and examples for dynamic tool selection. If None, metadata will
        be auto-generated from tool properties.

        Returns:
            ToolMetadata with semantic information, or None for auto-generation
        """
        return None

    def get_metadata(self) -> ToolMetadata:
        """Get semantic metadata for tool selection (ToolMetadataProvider contract).

        This method fulfills the ToolMetadataProvider protocol and ALWAYS
        returns valid ToolMetadata. It follows a two-tier strategy:

        1. If explicit metadata is defined (via metadata property), use it
        2. Otherwise, auto-generate metadata from tool properties

        This ensures ALL tools can participate in semantic tool selection
        without requiring manual configuration in tool_knowledge.yaml.

        Returns:
            ToolMetadata with semantic information for tool selection
        """
        # Tier 1: Use explicit metadata if defined
        explicit_metadata = self.metadata
        if explicit_metadata is not None:
            return explicit_metadata

        # Tier 2: Auto-generate from tool properties
        return ToolMetadata.generate_from_tool(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            cost_tier=self.cost_tier,
        )

    @property
    def cost_tier(self) -> CostTier:
        """Cost tier for the tool.

        Override this property in subclasses to specify the appropriate tier.

        Tiers:
            FREE: Local operations (filesystem, bash, git) - default
            LOW: Compute-only operations (code review, refactoring)
            MEDIUM: External API calls (web search, fetch)
            HIGH: Resource-intensive (batch processing 100+ files)

        Returns:
            CostTier enum value
        """
        return CostTier.FREE  # Default: local operations are free

    @staticmethod
    def convert_parameters_to_schema(parameters: List[ToolParameter]) -> Dict[str, Any]:
        """Convert list of ToolParameter objects to JSON Schema format.

        Args:
            parameters: List of ToolParameter objects

        Returns:
            JSON Schema dictionary
        """
        properties = {}
        required = []

        for param in parameters:
            param_schema: Dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }

            if param.enum:
                param_schema["enum"] = param.enum

            properties[param.name] = param_schema

            if param.required:
                required.append(param.name)

        schema = {
            "type": "object",
            "properties": properties,
        }

        if required:
            schema["required"] = required

        return schema

    @abstractmethod
    async def execute(self, _exec_ctx: Dict[str, Any], **kwargs: Any) -> ToolResult:
        """Execute the tool.

        Args:
            _exec_ctx: Framework execution context (reserved name to avoid collision
                      with tool parameters). Contains shared resources like code_manager.
            **kwargs: Tool parameters

        Returns:
            ToolResult with execution outcome
        """
        pass

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert tool to JSON Schema format.

        Returns:
            JSON Schema representation
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate_parameters(self, **kwargs: Any) -> bool:
        """Validate provided parameters against schema.

        Simple boolean validation - use validate_parameters_detailed() for
        detailed error information.

        Args:
            **kwargs: Parameters to validate

        Returns:
            True if valid, False otherwise
        """
        return self.validate_parameters_detailed(**kwargs).valid

    def validate_parameters_detailed(self, **kwargs: Any) -> ValidationResult:
        """Validate provided parameters against JSON Schema with detailed errors.

        Uses JSON Schema Draft 7 validation for comprehensive type checking,
        required field validation, enum constraints, and nested object validation.

        Args:
            **kwargs: Parameters to validate

        Returns:
            ValidationResult with detailed error information
        """
        schema = self.parameters

        # Handle empty/minimal schemas gracefully
        if not schema or schema == {"type": "object", "properties": {}}:
            return ValidationResult.success()

        try:
            # Ensure schema has proper structure for validation
            if "type" not in schema:
                schema = {"type": "object", **schema}

            # Create validator with format checking disabled for flexibility
            validator = Draft7Validator(schema)

            # Collect all validation errors
            errors: List[str] = []
            invalid_params: Dict[str, str] = {}

            for error in validator.iter_errors(kwargs):
                # Format error message based on error type
                if error.validator == "required":
                    # Extract missing field from error message
                    missing_fields = error.validator_value
                    for field_name in missing_fields:
                        if field_name not in kwargs:
                            msg = f"Required parameter '{field_name}' is missing"
                            errors.append(msg)
                            invalid_params[field_name] = "required"
                elif error.validator == "type":
                    # Type mismatch error
                    path = ".".join(str(p) for p in error.path) or "root"
                    expected = error.validator_value
                    actual = type(error.instance).__name__
                    msg = f"Parameter '{path}' has wrong type: expected {expected}, got {actual}"
                    errors.append(msg)
                    if error.path:
                        invalid_params[str(error.path[0])] = f"type: expected {expected}"
                elif error.validator == "enum":
                    # Enum constraint violation
                    path = ".".join(str(p) for p in error.path) or "root"
                    allowed = error.validator_value
                    msg = f"Parameter '{path}' must be one of: {allowed}"
                    errors.append(msg)
                    if error.path:
                        invalid_params[str(error.path[0])] = "invalid enum value"
                elif error.validator == "additionalProperties":
                    # Extra properties that aren't allowed
                    path = ".".join(str(p) for p in error.path) or "root"
                    msg = f"Unknown parameter in '{path}': {error.message}"
                    errors.append(msg)
                else:
                    # Generic error handling
                    path = ".".join(str(p) for p in error.path) or "root"
                    msg = f"Validation error at '{path}': {error.message}"
                    errors.append(msg)
                    if error.path:
                        invalid_params[str(error.path[0])] = error.message

            if errors:
                return ValidationResult.failure(errors, invalid_params)

            return ValidationResult.success()

        except JsonSchemaValidationError as e:
            # Single validation error (shouldn't happen with iter_errors but handle it)
            return ValidationResult.failure(
                [str(e.message)],
                {str(e.path[0]): e.message} if e.path else {},
            )
        except jsonschema.SchemaError as e:
            # Invalid schema - this is a programming error
            return ValidationResult.failure(
                [f"Invalid tool schema: {e.message}"],
                {},
            )
        except Exception:
            # Unexpected error - fall back to basic validation
            return self._fallback_validate(**kwargs)

    def _fallback_validate(self, **kwargs: Any) -> ValidationResult:
        """Basic validation fallback when JSON Schema validation fails.

        Args:
            **kwargs: Parameters to validate

        Returns:
            ValidationResult with basic validation
        """
        errors: List[str] = []
        invalid_params: Dict[str, str] = {}

        required_params = self.parameters.get("required", [])
        properties = self.parameters.get("properties", {})

        # Check required parameters
        for param in required_params:
            if param not in kwargs:
                errors.append(f"Required parameter '{param}' is missing")
                invalid_params[param] = "required"

        # Check parameter types (basic check)
        for param, value in kwargs.items():
            if param in properties:
                expected_type = properties[param].get("type")
                if expected_type and not self._check_type(value, expected_type):
                    actual_type = type(value).__name__
                    errors.append(
                        f"Parameter '{param}' has wrong type: expected {expected_type}, got {actual_type}"
                    )
                    invalid_params[param] = f"type: expected {expected_type}"

        if errors:
            return ValidationResult.failure(errors, invalid_params)
        return ValidationResult.success()

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type.

        Args:
            value: Value to check
            expected_type: Expected JSON Schema type

        Returns:
            True if types match
        """
        type_mapping: Dict[str, Union[type[Any], tuple[type[Any], ...]]] = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, allow it

        return isinstance(value, expected_python_type)


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self) -> None:
        """Initialize tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._tool_enabled: Dict[str, bool] = {}  # Track enabled/disabled state
        self._before_hooks: List[Union[Hook, Callable[[str, Dict[str, Any]], None]]] = []
        self._after_hooks: List[Union[Hook, Callable[["ToolResult"], None]]] = []

    def _wrap_hook(
        self, hook: Union[Hook, Callable], critical: bool = False, name: str = ""
    ) -> Hook:
        """Wrap a callable into a Hook object if needed."""
        if isinstance(hook, Hook):
            return hook
        return Hook(
            callback=hook, name=name or getattr(hook, "__name__", "hook"), critical=critical
        )

    def register_before_hook(
        self,
        hook: Union[Hook, Callable[[str, Dict[str, Any]], None]],
        critical: bool = False,
        name: str = "",
    ) -> None:
        """Register a hook to be called before a tool is executed.

        Args:
            hook: Hook instance or callable that takes (tool_name, arguments)
            critical: If True, hook failure will block tool execution
            name: Human-readable name for the hook (used for error messages)
        """
        wrapped = self._wrap_hook(hook, critical=critical, name=name)
        self._before_hooks.append(wrapped)

    def register_after_hook(
        self,
        hook: Union[Hook, Callable[["ToolResult"], None]],
        critical: bool = False,
        name: str = "",
    ) -> None:
        """Register a hook to be called after a tool is executed.

        Args:
            hook: Hook instance or callable that takes (tool_result,)
            critical: If True, hook failure will raise an error
            name: Human-readable name for the hook (used for error messages)
        """
        wrapped = self._wrap_hook(hook, critical=critical, name=name)
        self._after_hooks.append(wrapped)

    def register(self, tool: Any, enabled: bool = True) -> None:
        """Register a tool.

        Can register a BaseTool instance or a function decorated with @tool.

        Args:
            tool: Tool instance or decorated function to register
            enabled: Whether the tool is enabled by default (default: True)
        """
        if hasattr(tool, "Tool"):  # It's a decorated function
            tool_instance = tool.Tool
            self._tools[tool_instance.name] = tool_instance
            self._tool_enabled[tool_instance.name] = enabled
        elif isinstance(tool, BaseTool):  # It's a class instance
            self._tools[tool.name] = tool
            self._tool_enabled[tool.name] = enabled
        else:
            raise TypeError(
                "Can only register BaseTool instances or functions decorated with @tool"
            )

    def register_dict(self, tool_dict: Dict[str, Any], enabled: bool = True) -> None:
        """Register a tool from a dictionary definition.

        Used primarily for MCP tool definitions that come as dictionaries.

        Args:
            tool_dict: Dictionary with 'name', 'description', and 'parameters' keys
            enabled: Whether the tool is enabled by default (default: True)
        """
        name = tool_dict.get("name", "")
        description = tool_dict.get("description", "")
        parameters = tool_dict.get("parameters", {"type": "object", "properties": {}})

        # Create a wrapper tool that stores the dictionary definition
        # This is a placeholder - actual execution is handled by mcp_call
        class DictTool(BaseTool):
            @property
            def name(self) -> str:
                return name

            @property
            def description(self) -> str:
                return description

            @property
            def parameters(self) -> Dict[str, Any]:
                return parameters

            async def execute(self, _exec_ctx: Dict[str, Any], **kwargs: Any) -> ToolResult:
                # MCP tools are executed via mcp_call, not directly
                return ToolResult(
                    success=False,
                    output=None,
                    error="MCP tools should be called via mcp_call",
                )

        self._tools[name] = DictTool()
        self._tool_enabled[name] = enabled

    def unregister(self, name: str) -> None:
        """Unregister a tool.

        Args:
            name: Tool name to unregister
        """
        self._tools.pop(name, None)
        self._tool_enabled.pop(name, None)

    def enable_tool(self, name: str) -> bool:
        """Enable a tool by name.

        Args:
            name: Tool name to enable

        Returns:
            True if tool exists and was enabled, False otherwise
        """
        if name in self._tools:
            self._tool_enabled[name] = True
            return True
        return False

    def disable_tool(self, name: str) -> bool:
        """Disable a tool by name.

        Args:
            name: Tool name to disable

        Returns:
            True if tool exists and was disabled, False otherwise
        """
        if name in self._tools:
            self._tool_enabled[name] = False
            return True
        return False

    def is_tool_enabled(self, name: str) -> bool:
        """Check if a tool is enabled.

        Args:
            name: Tool name

        Returns:
            True if tool is enabled, False otherwise
        """
        return self._tool_enabled.get(name, False)

    def set_tool_states(self, tool_states: Dict[str, bool]) -> None:
        """Set enabled/disabled states for multiple tools.

        Args:
            tool_states: Dictionary mapping tool names to enabled state
        """
        for name, enabled in tool_states.items():
            if name in self._tools:
                self._tool_enabled[name] = enabled

    def get_tool_states(self) -> Dict[str, bool]:
        """Get enabled/disabled states for all tools.

        Returns:
            Dictionary mapping tool names to enabled state
        """
        return self._tool_enabled.copy()

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    def list_tools(self, only_enabled: bool = True) -> list[BaseTool]:
        """List all registered tools.

        Args:
            only_enabled: If True, only return enabled tools (default: True)

        Returns:
            List of tool instances
        """
        if only_enabled:
            return [
                tool for name, tool in self._tools.items() if self._tool_enabled.get(name, False)
            ]
        return list(self._tools.values())

    def get_tool_schemas(self, only_enabled: bool = True) -> list[Dict[str, Any]]:
        """Get JSON schemas for all tools.

        Args:
            only_enabled: If True, only return schemas for enabled tools (default: True)

        Returns:
            List of tool JSON schemas
        """
        if only_enabled:
            return [
                tool.to_json_schema()
                for name, tool in self._tools.items()
                if self._tool_enabled.get(name, False)
            ]
        return [tool.to_json_schema() for tool in self._tools.values()]

    def get_tool_cost(self, name: str) -> Optional[CostTier]:
        """Get the cost tier for a tool.

        Args:
            name: Tool name

        Returns:
            CostTier enum value or None if tool not found
        """
        tool = self.get(name)
        if tool:
            return tool.cost_tier
        return None

    def get_tools_by_cost(
        self, max_tier: CostTier = CostTier.HIGH, only_enabled: bool = True
    ) -> List[BaseTool]:
        """Get tools filtered by maximum cost tier.

        Args:
            max_tier: Maximum cost tier to include
            only_enabled: If True, only return enabled tools

        Returns:
            List of tools at or below the specified cost tier
        """
        tools = self.list_tools(only_enabled=only_enabled)
        return [t for t in tools if t.cost_tier.weight <= max_tier.weight]

    def get_cost_summary(self, only_enabled: bool = True) -> Dict[str, List[str]]:
        """Get a summary of tools grouped by cost tier.

        Args:
            only_enabled: If True, only include enabled tools

        Returns:
            Dictionary mapping cost tier names to lists of tool names
        """
        summary: Dict[str, List[str]] = {tier.value: [] for tier in CostTier}
        for tool in self.list_tools(only_enabled=only_enabled):
            summary[tool.cost_tier.value].append(tool.name)
        return summary

    async def execute(self, name: str, _exec_ctx: Dict[str, Any], **kwargs: Any) -> ToolResult:
        """Execute a tool by name.

        Args:
            name: Tool name
            _exec_ctx: Framework execution context (reserved name to avoid collision
                      with tool parameters). Contains shared resources.
            **kwargs: Tool parameters

        Returns:
            ToolResult with execution outcome
        """
        # Trigger before-execution hooks
        for hook in self._before_hooks:
            hook(name, kwargs)

        tool = self.get(name)
        if tool is None:
            result = ToolResult(
                success=False,
                output=None,
                error=f"Tool '{name}' not found",
                metadata=None,
            )
        elif not self.is_tool_enabled(name):
            result = ToolResult(
                success=False,
                output=None,
                error=f"Tool '{name}' is disabled",
                metadata=None,
            )
        else:
            # Use detailed validation for better error messages
            validation = tool.validate_parameters_detailed(**kwargs)
            if not validation.valid:
                error_msg = f"Invalid parameters for tool '{name}': " + "; ".join(validation.errors)
                result = ToolResult(
                    success=False,
                    output=None,
                    error=error_msg,
                    metadata={"invalid_params": validation.invalid_params},
                )
            else:
                try:
                    result = await tool.execute(_exec_ctx, **kwargs)
                except Exception as e:
                    result = ToolResult(
                        success=False,
                        output=None,
                        error=f"Tool execution failed: {str(e)}",
                        metadata={"exception": type(e).__name__},
                    )

        # Trigger after-execution hooks
        for hook in self._after_hooks:
            hook(result)

        return result


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
    def _calculate_tools_hash(tools: List[BaseTool]) -> str:
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

    def needs_reindex(self, tools: List[BaseTool]) -> bool:
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

    def refresh_from_tools(self, tools: List[BaseTool], force: bool = False) -> bool:
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

    def register_tool(self, tool: BaseTool) -> None:
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
