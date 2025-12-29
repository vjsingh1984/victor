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

These types are re-exported from `victor.verticals.base` and
`victor.verticals.protocols` for backward compatibility.

Type Categories:
    - Stage Types: StageDefinition for workflow stages
    - Task Types: TaskTypeHint for task-specific prompt hints
    - Middleware Types: MiddlewarePriority, MiddlewareResult
    - Tool Types: TieredToolConfig for intelligent tool selection

Note:
    For vertical-specific protocols (MiddlewareProtocol, SafetyExtensionProtocol,
    etc.), see `victor.verticals.protocols`. Only data types are defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

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

        registry = ToolMetadataRegistry.get_instance()
        all_tools = registry.get_all_tool_names()
        # Semantic pool = all tools - base tools (mandatory + vertical_core)
        base = self.get_base_tools()
        return all_tools - base

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

        registry = ToolMetadataRegistry.get_instance()
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
    in victor.verticals.base. It provides the core structure without
    requiring framework dependencies.

    Note:
        For the full VerticalConfig with ToolSet support, use
        `victor.verticals.base.VerticalConfig` instead.

    Attributes:
        system_prompt: System prompt text
        stages: Stage definitions
        provider_hints: Hints for provider selection
        evaluation_criteria: Criteria for evaluating agent performance
        metadata: Additional vertical-specific metadata
    """

    system_prompt: str = ""
    stages: Dict[str, StageDefinition] = field(default_factory=dict)
    provider_hints: Dict[str, Any] = field(default_factory=dict)
    evaluation_criteria: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    # Stage Types
    "StageDefinition",
    # Task Types
    "TaskTypeHint",
    # Middleware Types
    "MiddlewarePriority",
    "MiddlewareResult",
    # Tool Configuration
    "TieredToolConfig",
    # Config Base
    "VerticalConfigBase",
]
