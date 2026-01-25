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

"""Core vertical types for cross-vertical abstraction.

This module defines fundamental types used across multiple verticals,
placed here in core to:
1. Avoid circular imports between core and verticals
2. Enable framework components to work with vertical abstractions
3. Provide a single source of truth for shared type definitions

These types are re-exported from `victor.core.verticals.base` and
`victor.core.verticals.protocols` for backward compatibility.

Type Categories:
    - Stage Types: StageDefinition for workflow stages
    - Task Types: TaskTypeHint for task-specific prompt hints
    - Middleware Types: MiddlewarePriority, MiddlewareResult
    - Tool Types: TieredToolConfig for intelligent tool selection

Note:
    For vertical-specific protocols (MiddlewareProtocol, SafetyExtensionProtocol,
    etc.), see `victor.core.verticals.protocols`. Only data types are defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Set, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from victor.framework.tools import ToolSet


# =============================================================================
# Stage Types
# =============================================================================


@dataclass
class StageDefinition:
    """Definition of a conversation stage for a vertical.

    Stages represent distinct phases in a conversation workflow (e.g., planning,
    execution, verification). Each vertical can define its own stages with
    appropriate tools and transitions.

    This type is used by:
    - VerticalBase.get_stages() to define vertical stages
    - ConversationStateMachine for stage tracking
    - Agent orchestration for stage-based tool selection

    Attributes:
        name: Stage name (e.g., "PLANNING", "EXECUTION")
        description: Human-readable description
        tools: Tools relevant to this stage
        keywords: Keywords that suggest this stage
        next_stages: Valid stages to transition to
        min_confidence: Minimum confidence to enter this stage
    """

    name: str
    description: str
    tools: Set[str] = field(default_factory=set)
    keywords: List[str] = field(default_factory=list)
    next_stages: Set[str] = field(default_factory=set)
    min_confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "tools": list(self.tools),
            "keywords": self.keywords,
            "next_stages": list(self.next_stages),
            "min_confidence": self.min_confidence,
        }


# =============================================================================
# Task Type Hints
# =============================================================================


@dataclass
class TaskTypeHint:
    """Hint for a specific task type.

    Task type hints provide guidance for handling specific types of tasks
    (edit, search, explain, debug, etc.). They include prompt text,
    tool budget recommendations, and priority tools.

    This type is used by:
    - PromptContributorProtocol implementations in each vertical
    - Task classifier to adjust behavior based on detected task type
    - Tool selection to prioritize appropriate tools

    Attributes:
        task_type: Task type identifier (e.g., "edit", "search")
        hint: Prompt hint text to include in system prompt
        tool_budget: Recommended tool budget for this task type
        priority_tools: Tools to prioritize for this task
    """

    task_type: str
    hint: str
    tool_budget: Optional[int] = None
    priority_tools: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_type": self.task_type,
            "hint": self.hint,
            "tool_budget": self.tool_budget,
            "priority_tools": self.priority_tools,
        }


# =============================================================================
# Standard Task Hints (Common patterns across verticals)
# =============================================================================


class StandardTaskHints:
    """Standard task hints shared across verticals.

    Provides common task type hints that apply to multiple verticals,
    reducing duplication while allowing vertical-specific customization.

    Example:
        # Get a standard hint
        general_hint = StandardTaskHints.get("general")

        # Merge with vertical-specific hints
        vertical_hints = StandardTaskHints.merge_with(my_vertical_hints)

        # Get all standard hints for a vertical type
        hints = StandardTaskHints.for_vertical("coding")
    """

    # Base hint templates that can be customized
    GENERAL: TaskTypeHint = TaskTypeHint(
        task_type="general",
        hint="[GENERAL] Moderate exploration. 3-6 tool calls. Answer concisely.",
        tool_budget=8,
        priority_tools=["read", "grep", "ls"],
    )

    SEARCH: TaskTypeHint = TaskTypeHint(
        task_type="search",
        hint="[SEARCH] Use grep/ls for exploration. Summarize after 2-4 calls.",
        tool_budget=6,
        priority_tools=["grep", "ls", "read"],
    )

    CREATE: TaskTypeHint = TaskTypeHint(
        task_type="create",
        hint="[CREATE] Read 1-2 relevant files for context, then create. Follow existing patterns.",
        tool_budget=5,
        priority_tools=["read", "write"],
    )

    EDIT: TaskTypeHint = TaskTypeHint(
        task_type="edit",
        hint="[EDIT] Read target file first, then modify. Focused changes only.",
        tool_budget=5,
        priority_tools=["read", "edit"],
    )

    ANALYZE: TaskTypeHint = TaskTypeHint(
        task_type="analyze",
        hint="[ANALYZE] Examine content carefully. Read related files. Structured findings.",
        tool_budget=12,
        priority_tools=["read", "grep"],
    )

    # Standard hints as a dictionary
    _STANDARD_HINTS: Dict[str, TaskTypeHint] = {}

    @classmethod
    def _init_standard_hints(cls) -> None:
        """Initialize standard hints dictionary (lazy)."""
        if not cls._STANDARD_HINTS:
            cls._STANDARD_HINTS = {
                "general": cls.GENERAL,
                "search": cls.SEARCH,
                "create": cls.CREATE,
                "edit": cls.EDIT,
                "analyze": cls.ANALYZE,
            }

    @classmethod
    def get(cls, task_type: str) -> Optional[TaskTypeHint]:
        """Get a standard hint by task type.

        Args:
            task_type: Task type name

        Returns:
            TaskTypeHint or None if not found
        """
        cls._init_standard_hints()
        return cls._STANDARD_HINTS.get(task_type.lower())

    @classmethod
    def all(cls) -> Dict[str, TaskTypeHint]:
        """Get all standard hints.

        Returns:
            Dict of task type to TaskTypeHint
        """
        cls._init_standard_hints()
        return cls._STANDARD_HINTS.copy()

    @classmethod
    def merge_with(cls, vertical_hints: Dict[str, TaskTypeHint]) -> Dict[str, TaskTypeHint]:
        """Merge standard hints with vertical-specific hints.

        Vertical-specific hints override standard hints with the same key.

        Args:
            vertical_hints: Vertical-specific task type hints

        Returns:
            Merged dict with standard + vertical hints
        """
        cls._init_standard_hints()
        result = cls._STANDARD_HINTS.copy()
        result.update(vertical_hints)
        return result


# =============================================================================
# Standard Grounding Rules
# =============================================================================


class StandardGroundingRules:
    """Standard grounding rules shared across verticals.

    Provides base grounding rule templates that verticals can use
    or extend with domain-specific rules.
    """

    # Base grounding rule - applies to all verticals
    BASE: str = (
        "GROUNDING: Base ALL responses on tool output only. "
        "Never invent file paths or content. "
        "Quote exactly from tool output. If more info needed, call another tool."
    )

    # Extended grounding for local models
    EXTENDED: str = """CRITICAL - TOOL OUTPUT GROUNDING:
When you receive tool output in <TOOL_OUTPUT> tags:
1. The content between ═══ markers is ACTUAL file/command output - NEVER ignore it
2. You MUST base your analysis ONLY on this actual content
3. NEVER fabricate, invent, or imagine content that differs from tool output
4. If you need more information, call another tool - do NOT guess
5. When citing content, quote EXACTLY from the tool output
6. If tool output is empty or truncated, acknowledge this limitation

VIOLATION OF THESE RULES WILL RESULT IN INCORRECT ANALYSIS."""

    # Research-specific addendum
    RESEARCH_ADDENDUM: str = (
        "Always cite URLs for claims. Acknowledge uncertainty when sources conflict."
    )

    # Data-specific addendum
    DATA_ADDENDUM: str = (
        "Verify calculations with actual data. Always show code that produced results."
    )

    # DevOps-specific addendum
    DEVOPS_ADDENDUM: str = (
        "Verify configuration syntax before suggesting. Always check existing resources first."
    )

    @classmethod
    def get_base(cls, extended: bool = False) -> str:
        """Get base grounding rules.

        Args:
            extended: Whether to use extended rules (for local models)

        Returns:
            Grounding rules string
        """
        return cls.EXTENDED if extended else cls.BASE

    @classmethod
    def for_vertical(cls, vertical: str, extended: bool = False) -> str:
        """Get grounding rules for a specific vertical.

        Args:
            vertical: Vertical name
            extended: Whether to use extended rules

        Returns:
            Grounding rules string with vertical addendum
        """
        base = cls.get_base(extended)
        addendums = {
            "research": cls.RESEARCH_ADDENDUM,
            "data_analysis": cls.DATA_ADDENDUM,
            "devops": cls.DEVOPS_ADDENDUM,
        }
        addendum = addendums.get(vertical, "")
        if addendum:
            return f"{base}\n{addendum}"
        return base


# =============================================================================
# Middleware Types
# =============================================================================


class MiddlewarePriority(Enum):
    """Priority levels for middleware execution order.

    Middleware executes in priority order - lower values execute first
    in before_tool_call, higher values execute first in after_tool_call.

    This enables layered processing:
    - CRITICAL (0): Security validation, permission checks
    - HIGH (25): Core functionality, format validation
    - NORMAL (50): Standard processing, transformations
    - LOW (75): Logging, metrics collection
    - DEFERRED (100): Cleanup, finalization tasks

    Example:
        class SecurityMiddleware(MiddlewareProtocol):
            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.CRITICAL
    """

    CRITICAL = 0  # Security, validation
    HIGH = 25  # Core functionality
    NORMAL = 50  # Standard processing
    LOW = 75  # Logging, metrics
    DEFERRED = 100  # Cleanup, finalization


@dataclass
class MiddlewareResult:
    """Result from middleware processing.

    Middleware returns this result to indicate whether processing should
    continue and optionally provide modified arguments or error messages.

    Attributes:
        proceed: Whether to proceed with the operation (False blocks execution)
        modified_arguments: Modified arguments to pass downstream (if any)
        error_message: Error message if proceed is False
        metadata: Additional metadata for downstream processing
    """

    proceed: bool = True
    modified_arguments: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Tiered Tool Configuration
# =============================================================================


@runtime_checkable
class TieredToolConfigProtocol(Protocol):
    """Protocol for tiered tool configuration.

    Defines the interface that all tiered tool configurations must implement.
    This enables isinstance() checks instead of hasattr() for ISP compliance.

    The protocol captures the essential properties and methods of TieredToolConfig
    that are used by the framework (tool access controller, vertical integration, etc.).

    ISP Compliance:
        - This protocol provides a minimal, focused interface
        - Components depend on this protocol, not concrete implementations
        - Enables duck typing with type safety via runtime_checkable

    Example:
        from victor.core.vertical_types import TieredToolConfigProtocol

        # Type-safe check
        if isinstance(config, TieredToolConfigProtocol):
            tools = config.mandatory | config.vertical_core
    """

    @property
    def mandatory(self) -> Set[str]:
        """Tools always included regardless of task type."""
        ...

    @property
    def vertical_core(self) -> Set[str]:
        """Tools always included for this vertical."""
        ...

    @property
    def semantic_pool(self) -> Set[str]:
        """DEPRECATED - Tools selected via semantic matching."""
        ...

    @property
    def stage_tools(self) -> Dict[str, Set[str]]:
        """DEPRECATED - Tools available at specific stages."""
        ...

    def get_base_tools(self) -> Set[str]:
        """Get tools always included (mandatory + vertical core)."""
        ...

    def get_all_tools(self) -> Set[str]:
        """Get all tools in the configuration."""
        ...

    def get_tools_for_stage(self, stage: str) -> Set[str]:
        """Get tools for a specific stage."""
        ...

    def get_semantic_pool_from_registry(self) -> Set[str]:
        """Get semantic pool dynamically from ToolMetadataRegistry."""
        ...

    def get_effective_semantic_pool(self) -> Set[str]:
        """Get effective semantic pool, preferring registry over static."""
        ...

    def get_tools_for_stage_from_registry(self, stage: str) -> Set[str]:
        """Get tools for a stage using @tool decorator metadata."""
        ...


@dataclass
class TieredToolConfig:
    """Tiered tool configuration for intelligent tool selection.

    Implements a three-tier system for context-efficient tool management:
    1. Mandatory (always included): Essential tools for any task
    2. Vertical Core (always included for this vertical): Domain-specific core tools
    3. Semantic/Contextual (selected dynamically): Additional tools based on task

    Each tier can specify read-only vs read-write tools to enable
    intelligent filtering based on task intent (analysis vs modification).

    Migration Note:
        The `semantic_pool` and `stage_tools` fields are being deprecated in favor
        of using @tool decorator metadata:
        - `semantic_pool`: Will be derived from ToolMetadataRegistry.get_all_tool_names()
          minus mandatory/vertical_core. Most tools should be candidates.
        - `stage_tools`: Will be derived from @tool(stages=[...]) decorator metadata.
          Use ToolMetadataRegistry.get_tools_by_stage() instead.

    Example:
        # Preferred (new style)
        TieredToolConfig(
            mandatory={"read", "ls", "grep"},            # Essential for any task
            vertical_core={"web", "fetch", "overview"},  # Research-specific core
            readonly_only_for_analysis=True,             # Hide write tools for analysis
        )

    Attributes:
        mandatory: Tools always included regardless of task type
        vertical_core: Tools always included for this vertical
        semantic_pool: DEPRECATED - Tools selected via semantic matching
        stage_tools: DEPRECATED - Tools available at specific stages
        readonly_only_for_analysis: If True, hide write/execute tools for analysis tasks
    """

    mandatory: Set[str] = field(default_factory=set)
    vertical_core: Set[str] = field(default_factory=set)
    semantic_pool: Set[str] = field(default_factory=set)  # DEPRECATED: derive from registry
    stage_tools: Dict[str, Set[str]] = field(
        default_factory=dict
    )  # DEPRECATED: use @tool(stages=[])
    readonly_only_for_analysis: bool = True

    def get_base_tools(self) -> Set[str]:
        """Get tools always included (mandatory + vertical core)."""
        return self.mandatory | self.vertical_core

    def get_all_tools(self) -> Set[str]:
        """Get all tools in the configuration."""
        all_tools = self.mandatory | self.vertical_core | self.semantic_pool
        for stage_set in self.stage_tools.values():
            all_tools |= stage_set
        return all_tools

    def get_tools_for_stage(self, stage: str) -> Set[str]:
        """Get tools for a specific stage.

        Args:
            stage: Stage name (e.g., "INITIAL", "SEARCHING", "WRITING")

        Returns:
            Set of tool names for the stage (base + stage-specific)
        """
        base = self.get_base_tools()
        stage_specific = self.stage_tools.get(stage, set())
        return base | stage_specific

    def get_semantic_pool_from_registry(self) -> Set[str]:
        """Get semantic pool dynamically from ToolMetadataRegistry.

        This method derives the semantic pool from all registered tools
        minus the mandatory and vertical_core tools. Use this instead of
        the static semantic_pool field for new implementations.

        Returns:
            Set of tool names for semantic selection
        """
        from victor.tools.metadata_registry import ToolMetadataRegistry

        registry = ToolMetadataRegistry.get_instance()  # type: ignore[attr-defined]
        all_tools = registry.get_all_tool_names()
        # Semantic pool = all tools - base tools (mandatory + vertical_core)
        base = self.get_base_tools()
        return cast(set[str], all_tools - base)

    def get_effective_semantic_pool(self) -> Set[str]:
        """Get effective semantic pool, preferring registry over static.

        Returns:
            semantic_pool if explicitly set, otherwise derives from registry
        """
        if self.semantic_pool:
            return self.semantic_pool
        return self.get_semantic_pool_from_registry()

    def get_tools_for_stage_from_registry(self, stage: str) -> Set[str]:
        """Get tools for a stage using @tool decorator metadata.

        Args:
            stage: Stage name (e.g., "INITIAL", "READING", "EXECUTION")

        Returns:
            Base tools plus stage-specific tools from registry
        """
        from victor.tools.metadata_registry import ToolMetadataRegistry

        registry = ToolMetadataRegistry.get_instance()  # type: ignore[attr-defined]
        base = self.get_base_tools()
        registry_stage_tools = registry.get_tools_by_stage(stage)
        return base | registry_stage_tools


# =============================================================================
# Vertical Config Base Types
# =============================================================================


@dataclass
class VerticalConfigBase:
    """Base configuration for a vertical.

    This is a simplified base type that can be extended by VerticalConfig
    in victor.core.verticals.base. It provides the core structure without
    requiring framework dependencies.

    Note:
        For the full VerticalConfig with ToolSet support, use
        `victor.core.verticals.base.VerticalConfig` instead.

    Attributes:
        name: Vertical identifier
        system_prompt: System prompt text
        stages: Stage definitions
        provider_hints: Hints for provider selection
        evaluation_criteria: Criteria for evaluating agent performance
        metadata: Additional vertical-specific metadata
    """

    name: str = ""
    system_prompt: str = ""
    stages: Dict[str, StageDefinition] = field(default_factory=dict)
    provider_hints: Dict[str, Any] = field(default_factory=dict)
    evaluation_criteria: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Tiered Tool Template (Factory for Reducing Duplication)
# =============================================================================


class TieredToolTemplate:
    """Factory for creating TieredToolConfig with common patterns.

    This template reduces duplication by providing:
    1. Standard mandatory tools used by all verticals
    2. Pre-built configurations for common verticals
    3. Easy customization via factory methods

    Example usage:
        # Simple: use pre-built vertical config
        config = TieredToolTemplate.for_vertical("coding")

        # Custom: specify only vertical-specific tools
        config = TieredToolTemplate.create(
            vertical_core={"edit", "write", "shell", "git"},
            readonly_only_for_analysis=False,
        )

        # Override mandatory tools (rare)
        config = TieredToolTemplate.create(
            mandatory={"read", "ls"},  # Custom mandatory set
            vertical_core={"web_search", "web_fetch"},
        )
    """

    # Standard mandatory tools - essential for any task across all verticals
    DEFAULT_MANDATORY: Set[str] = {"read", "ls", "grep"}

    # Pre-configured vertical cores
    VERTICAL_CORES: Dict[str, Set[str]] = {
        "coding": {"edit", "write", "shell", "git", "search", "overview"},
        "research": {"web_search", "web_fetch", "overview"},
        "devops": {"shell", "git", "docker", "overview"},
        "data_analysis": {"shell", "write", "overview"},
        "rag": {"rag_search", "rag_query", "rag_list", "rag_stats", "rag_delete"},
    }

    # Vertical-specific readonly_only_for_analysis settings
    VERTICAL_READONLY_DEFAULTS: Dict[str, bool] = {
        "coding": False,  # Coding often needs write tools
        "research": True,  # Research is primarily reading
        "devops": False,  # DevOps needs execution tools
        "data_analysis": False,  # Data analysis often writes results
        "rag": True,  # RAG is primarily reading/retrieval
    }

    @classmethod
    def create(
        cls,
        vertical_core: Set[str],
        mandatory: Optional[Set[str]] = None,
        readonly_only_for_analysis: bool = True,
        semantic_pool: Optional[Set[str]] = None,
        stage_tools: Optional[Dict[str, Set[str]]] = None,
    ) -> TieredToolConfig:
        """Create a TieredToolConfig with standard mandatory tools.

        Args:
            vertical_core: Domain-specific core tools for the vertical
            mandatory: Override mandatory tools (uses DEFAULT_MANDATORY if None)
            readonly_only_for_analysis: Whether to hide write tools for analysis
            semantic_pool: DEPRECATED - tools for semantic selection
            stage_tools: DEPRECATED - stage-specific tools

        Returns:
            Configured TieredToolConfig
        """
        return TieredToolConfig(
            mandatory=mandatory if mandatory is not None else cls.DEFAULT_MANDATORY.copy(),
            vertical_core=vertical_core,
            semantic_pool=semantic_pool or set(),
            stage_tools=stage_tools or {},
            readonly_only_for_analysis=readonly_only_for_analysis,
        )

    @classmethod
    def for_vertical(cls, vertical: str) -> Optional[TieredToolConfig]:
        """Get pre-configured TieredToolConfig for a known vertical.

        Args:
            vertical: Vertical name (coding, research, devops, data_analysis, rag)

        Returns:
            Configured TieredToolConfig or None if vertical not known
        """
        if vertical not in cls.VERTICAL_CORES:
            return None

        return cls.create(
            vertical_core=cls.VERTICAL_CORES[vertical].copy(),
            readonly_only_for_analysis=cls.VERTICAL_READONLY_DEFAULTS.get(vertical, True),
        )

    @classmethod
    def for_coding(cls) -> TieredToolConfig:
        """Get TieredToolConfig for coding vertical."""
        return cls.for_vertical("coding")  # type: ignore

    @classmethod
    def for_research(cls) -> TieredToolConfig:
        """Get TieredToolConfig for research vertical."""
        return cls.for_vertical("research")  # type: ignore

    @classmethod
    def for_devops(cls) -> TieredToolConfig:
        """Get TieredToolConfig for devops vertical."""
        return cls.for_vertical("devops")  # type: ignore

    @classmethod
    def for_data_analysis(cls) -> TieredToolConfig:
        """Get TieredToolConfig for data analysis vertical."""
        return cls.for_vertical("data_analysis")  # type: ignore

    @classmethod
    def for_rag(cls) -> TieredToolConfig:
        """Get TieredToolConfig for RAG vertical."""
        return cls.for_vertical("rag")  # type: ignore

    @classmethod
    def register_vertical(
        cls,
        name: str,
        vertical_core: Set[str],
        readonly_only_for_analysis: bool = True,
    ) -> None:
        """Register a new vertical's tool configuration.

        Args:
            name: Vertical name
            vertical_core: Core tools for the vertical
            readonly_only_for_analysis: Whether to hide write tools for analysis
        """
        cls.VERTICAL_CORES[name] = vertical_core
        cls.VERTICAL_READONLY_DEFAULTS[name] = readonly_only_for_analysis

    @classmethod
    def list_verticals(cls) -> List[str]:
        """List all registered verticals."""
        return list(cls.VERTICAL_CORES.keys())


__all__ = [
    # Stage Types
    "StageDefinition",
    # Task Types
    "TaskTypeHint",
    "StandardTaskHints",
    "StandardGroundingRules",
    # Middleware Types
    "MiddlewarePriority",
    "MiddlewareResult",
    # Tool Configuration
    "TieredToolConfigProtocol",
    "TieredToolConfig",
    "TieredToolTemplate",
    # Config Base
    "VerticalConfigBase",
]
