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

"""Unified Task Type Registry for the Victor Framework.

This module consolidates the ~5 different TaskType enums and scattered
hint/budget definitions into a single, extensible registry.

The registry provides:
- Canonical task type definitions with categories
- Configurable tool budgets and iteration limits
- System prompt hints for each task type
- Priority tool lists per task type
- Vertical-specific overrides
- Alias resolution for backward compatibility

Usage:
    # Get the singleton registry
    registry = TaskTypeRegistry.get_instance()

    # Look up a task type
    definition = registry.get("edit")
    print(definition.hint)  # "[EDIT] Modify existing code..."
    print(definition.tool_budget)  # 25

    # Vertical-specific lookup
    devops_edit = registry.get("edit", vertical="devops")

    # Register custom task type
    registry.register(TaskTypeDefinition(
        name="custom_task",
        category=TaskCategory.MODIFICATION,
        hint="[CUSTOM] Custom task hint",
        tool_budget=15,
    ))

    # Register vertical override
    registry.register_for_vertical("devops", TaskTypeDefinition(
        name="edit",
        category=TaskCategory.MODIFICATION,
        hint="[DEVOPS EDIT] Edit infrastructure files carefully",
        tool_budget=30,
    ))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class TaskCategory(Enum):
    """High-level task categories for grouping related task types.

    Categories help with:
    - Tool selection (analysis tasks favor read-only tools)
    - Safety rules (modification tasks need file edit permissions)
    - UI grouping (show related tasks together)
    """

    MODIFICATION = "modification"
    """Edit, Create, Refactor - tasks that change files."""

    ANALYSIS = "analysis"
    """Analyze, Search, Review - tasks that read/explore code."""

    EXECUTION = "execution"
    """Action, Test, Build - tasks that run commands/processes."""

    CONVERSATION = "conversation"
    """Chat, General, Help - conversational or ambiguous tasks."""


@dataclass
class TaskTypeDefinition:
    """Definition of a task type with all associated metadata.

    This is the single source of truth for task type configuration,
    consolidating hints, budgets, tool lists, and constraints.

    Attributes:
        name: Canonical name of the task type (lowercase, snake_case)
        category: High-level category for grouping
        hint: System prompt hint injected for this task type
        tool_budget: Maximum tool calls allowed for this task type
        priority_tools: Tools to prefer when selecting for this task
        max_iterations: Maximum exploration iterations
        aliases: Alternative names that map to this type
        vertical: Vertical this definition belongs to (None = core)
        needs_tools: Whether this task type needs tools at all
        force_action_after_read: For EDIT tasks, force action after reading target
        stage_tools: Tools allowed at each conversation stage
        force_action_hints: Hints to show when forcing action
        exploration_multiplier: Multiplier for exploration limits (used by modes)
    """

    name: str
    category: TaskCategory
    hint: str = ""
    tool_budget: int = 10
    priority_tools: List[str] = field(default_factory=list)
    max_iterations: int = 30
    aliases: Union[List[str], Set[str]] = field(default_factory=set)
    vertical: Optional[str] = None
    needs_tools: bool = True
    force_action_after_read: bool = False
    stage_tools: Dict[str, List[str]] = field(default_factory=dict)
    force_action_hints: Dict[str, str] = field(default_factory=dict)
    exploration_multiplier: float = 1.0

    def __post_init__(self) -> None:
        """Normalize name to lowercase."""
        self.name = self.name.lower()
        # Ensure aliases are a set
        if isinstance(self.aliases, list):
            self.aliases = set(self.aliases)


class TaskTypeRegistry:
    """Central registry for all task type definitions.

    The registry is a singleton that provides:
    - Registration of core task types
    - Registration of vertical-specific overrides
    - Alias resolution (e.g., "create" -> "generation")
    - Lookup with vertical fallback to core

    Thread Safety:
        The registry is NOT thread-safe. It should be populated at startup
        and then only read during normal operation.

    Extensibility:
        Verticals can register their own task types or override core ones:
        ```python
        registry = TaskTypeRegistry.get_instance()
        registry.register_for_vertical("devops", TaskTypeDefinition(
            name="infrastructure",
            category=TaskCategory.MODIFICATION,
            hint="[INFRA] Create infrastructure configs",
            tool_budget=25,
        ))
        ```
    """

    _instance: Optional["TaskTypeRegistry"] = None

    def __init__(self) -> None:
        """Initialize empty registry. Use get_instance() for singleton access."""
        self._core_types: Dict[str, TaskTypeDefinition] = {}
        self._vertical_overrides: Dict[str, Dict[str, TaskTypeDefinition]] = {}
        self._aliases: Dict[str, str] = {}
        self._registration_hooks: List[Callable[["TaskTypeRegistry"], None]] = []

    @classmethod
    def get_instance(cls) -> "TaskTypeRegistry":
        """Get the singleton registry instance.

        Creates and populates the registry on first access.

        Returns:
            The global TaskTypeRegistry instance with defaults registered.
        """
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_defaults()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    def register(self, definition: TaskTypeDefinition) -> None:
        """Register a core task type definition.

        Args:
            definition: The task type definition to register

        Raises:
            ValueError: If a task type with the same name already exists
        """
        name = definition.name.lower()

        if name in self._core_types:
            logger.warning(f"TaskTypeRegistry: Overwriting core type '{name}'")

        self._core_types[name] = definition

        # Register aliases
        for alias in definition.aliases:
            alias_lower = alias.lower()
            self._aliases[alias_lower] = name
            logger.debug(f"TaskTypeRegistry: Alias '{alias_lower}' -> '{name}'")

        logger.debug(
            f"TaskTypeRegistry: Registered core type '{name}' "
            f"(category={definition.category.value}, budget={definition.tool_budget})"
        )

    def register_for_vertical(self, vertical: str, definition: TaskTypeDefinition) -> None:
        """Register a task type definition for a specific vertical.

        Vertical-specific definitions override core definitions when
        looking up with that vertical specified.

        Args:
            vertical: The vertical name (e.g., "devops", "data_analysis")
            definition: The task type definition to register
        """
        vertical = vertical.lower()
        name = definition.name.lower()
        definition.vertical = vertical

        if vertical not in self._vertical_overrides:
            self._vertical_overrides[vertical] = {}

        self._vertical_overrides[vertical][name] = definition

        logger.debug(
            f"TaskTypeRegistry: Registered '{name}' for vertical '{vertical}' "
            f"(budget={definition.tool_budget})"
        )

    def add_registration_hook(self, hook: Callable[["TaskTypeRegistry"], None]) -> None:
        """Add a hook to be called when the registry is populated.

        Hooks are called during _register_defaults after core types are registered.
        Verticals can use this to register their task types.

        Args:
            hook: Callable that receives the registry instance
        """
        self._registration_hooks.append(hook)

    def get(self, task_type: str, vertical: Optional[str] = None) -> Optional[TaskTypeDefinition]:
        """Look up a task type definition.

        Resolution order:
        1. If vertical specified, check vertical-specific definitions
        2. Resolve aliases to canonical names
        3. Look up in core types

        Args:
            task_type: Name or alias of the task type
            vertical: Optional vertical for specific override lookup

        Returns:
            TaskTypeDefinition if found, None otherwise
        """
        task_type = task_type.lower()

        # Resolve alias to canonical name
        canonical = self._aliases.get(task_type, task_type)

        # Check vertical-specific first
        if vertical:
            vertical = vertical.lower()
            if vertical in self._vertical_overrides:
                if canonical in self._vertical_overrides[vertical]:
                    return self._vertical_overrides[vertical][canonical]

        # Fall back to core types
        return self._core_types.get(canonical)

    def get_hint(self, task_type: str, vertical: Optional[str] = None) -> str:
        """Get the system prompt hint for a task type.

        Args:
            task_type: Name or alias of the task type
            vertical: Optional vertical for specific override

        Returns:
            Hint string, or empty string if task type not found
        """
        definition = self.get(task_type, vertical)
        return definition.hint if definition else ""

    def get_tool_budget(self, task_type: str, vertical: Optional[str] = None) -> int:
        """Get the tool budget for a task type.

        Args:
            task_type: Name or alias of the task type
            vertical: Optional vertical for specific override

        Returns:
            Tool budget, or 10 (default) if task type not found
        """
        definition = self.get(task_type, vertical)
        return definition.tool_budget if definition else 10

    def get_max_iterations(self, task_type: str, vertical: Optional[str] = None) -> int:
        """Get the max iterations for a task type.

        Args:
            task_type: Name or alias of the task type
            vertical: Optional vertical for specific override

        Returns:
            Max iterations, or 30 (default) if task type not found
        """
        definition = self.get(task_type, vertical)
        return definition.max_iterations if definition else 30

    def get_priority_tools(self, task_type: str, vertical: Optional[str] = None) -> List[str]:
        """Get priority tools for a task type.

        Args:
            task_type: Name or alias of the task type
            vertical: Optional vertical for specific override

        Returns:
            List of tool names to prioritize, or empty list
        """
        definition = self.get(task_type, vertical)
        return definition.priority_tools if definition else []

    def get_category(
        self, task_type: str, vertical: Optional[str] = None
    ) -> Optional[TaskCategory]:
        """Get the category for a task type.

        Args:
            task_type: Name or alias of the task type
            vertical: Optional vertical for specific override

        Returns:
            TaskCategory or None if not found
        """
        definition = self.get(task_type, vertical)
        return definition.category if definition else None

    def list_types(self, vertical: Optional[str] = None) -> List[str]:
        """List all registered task type names.

        Args:
            vertical: If specified, includes vertical-specific types

        Returns:
            Sorted list of task type names
        """
        names = set(self._core_types.keys())
        if vertical and vertical.lower() in self._vertical_overrides:
            names.update(self._vertical_overrides[vertical.lower()].keys())
        return sorted(names)

    def list_verticals(self) -> List[str]:
        """List all registered verticals.

        Returns:
            Sorted list of vertical names
        """
        return sorted(self._vertical_overrides.keys())

    def resolve_alias(self, name: str) -> str:
        """Resolve an alias to its canonical name.

        Args:
            name: Task type name or alias

        Returns:
            Canonical name if alias exists, otherwise the original name
        """
        return self._aliases.get(name.lower(), name.lower())

    def _register_defaults(self) -> None:
        """Register all default task types.

        This is called automatically on first access to get_instance().
        Consolidates definitions from:
        - complexity_classifier.py: PROMPT_HINTS, DEFAULT_BUDGETS
        - unified_task_tracker.py: UnifiedTaskConfigLoader.DEFAULT_CONFIG
        - embeddings/task_classifier.py: TaskType enum
        """
        # =================================================================
        # Core Task Types
        # =================================================================

        # EDIT - Modify existing files
        self.register(
            TaskTypeDefinition(
                name="edit",
                category=TaskCategory.MODIFICATION,
                hint="[EDIT] Modify existing code. Read target files first, then make focused changes.",
                tool_budget=25,
                max_iterations=10,
                priority_tools=["read_file", "edit_files", "code_search"],
                aliases={"modify", "change", "update", "fix"},
                force_action_after_read=True,
                stage_tools={
                    "initial": ["list_directory", "code_search"],
                    "reading": ["read_file", "code_search"],
                    "executing": ["edit_files", "read_file"],
                    "verifying": ["read_file", "run_tests"],
                },
                force_action_hints={
                    "after_target_read": "Use edit_files to make the change.",
                    "max_iterations": "Please make the change or explain blockers.",
                },
            )
        )

        # CREATE - Create new files with context
        self.register(
            TaskTypeDefinition(
                name="create",
                category=TaskCategory.MODIFICATION,
                hint="[CREATE] Create new files. Explore context first, then write files.",
                tool_budget=25,
                max_iterations=10,
                priority_tools=["write_file", "read_file", "list_directory"],
                aliases={"new", "add", "write"},
                stage_tools={
                    "initial": ["list_directory", "read_file"],
                    "reading": ["read_file"],
                    "executing": ["write_file", "edit_files"],
                    "verifying": ["read_file", "run_tests"],
                },
                force_action_hints={
                    "max_iterations": "Please create the file.",
                },
            )
        )

        # CREATE_SIMPLE - Generate code directly (no exploration)
        self.register(
            TaskTypeDefinition(
                name="create_simple",
                category=TaskCategory.MODIFICATION,
                hint="[GENERATE] Write code directly. Minimal exploration. Display or save as requested.",
                tool_budget=5,
                max_iterations=3,
                priority_tools=["write_file"],
                aliases={"generate", "generation", "simple_create"},
                stage_tools={
                    "initial": ["write_file"],
                    "executing": ["write_file"],
                    "verifying": ["read_file"],
                },
                force_action_hints={
                    "immediate": "Create the code directly using write_file.",
                },
            )
        )

        # SEARCH - Find/locate code or files
        self.register(
            TaskTypeDefinition(
                name="search",
                category=TaskCategory.ANALYSIS,
                hint="[SEARCH] Find code or files. Use search tools, then read relevant results.",
                tool_budget=25,
                max_iterations=10,
                priority_tools=[
                    "code_search",
                    "semantic_code_search",
                    "read_file",
                    "list_directory",
                ],
                aliases={"find", "locate", "grep", "look"},
                stage_tools={
                    "initial": ["list_directory", "code_search"],
                    "reading": ["read_file", "code_search"],
                    "executing": ["read_file"],
                    "verifying": ["read_file"],
                },
                force_action_hints={
                    "max_iterations": "Please summarize your findings.",
                },
            )
        )

        # ANALYZE - Count, measure, analyze code
        self.register(
            TaskTypeDefinition(
                name="analyze",
                category=TaskCategory.ANALYSIS,
                hint="[ANALYZE] Analyze code metrics. Read files and run analysis commands.",
                tool_budget=40,
                max_iterations=20,
                priority_tools=["read_file", "code_search", "execute_bash"],
                aliases={"analysis", "review", "audit", "examine"},
                stage_tools={
                    "initial": ["list_directory", "code_search"],
                    "reading": ["read_file", "code_search", "execute_bash"],
                    "executing": ["execute_bash"],
                    "verifying": ["read_file"],
                },
                force_action_hints={
                    "max_iterations": "Please summarize your analysis.",
                },
            )
        )

        # ANALYSIS_DEEP - Comprehensive codebase analysis
        self.register(
            TaskTypeDefinition(
                name="analysis_deep",
                category=TaskCategory.ANALYSIS,
                hint="[DEEP ANALYSIS] Thorough exploration required. Examine all relevant modules. Comprehensive output.",
                tool_budget=60,
                max_iterations=50,
                priority_tools=[
                    "read_file",
                    "code_search",
                    "semantic_code_search",
                    "list_directory",
                ],
                aliases={"comprehensive", "thorough", "full_analysis"},
                exploration_multiplier=1.5,
            )
        )

        # DESIGN - Conceptual/planning tasks
        self.register(
            TaskTypeDefinition(
                name="design",
                category=TaskCategory.ANALYSIS,
                hint="[DESIGN] Architecture/design questions. Explore codebase to understand patterns and structure.",
                tool_budget=40,
                max_iterations=20,
                priority_tools=["read_file", "list_directory", "code_search"],
                aliases={"plan", "architect", "conceptual"},
                needs_tools=True,  # Design tasks benefit from codebase exploration
                stage_tools={
                    "initial": ["list_directory", "code_search", "read_file"],
                    "reading": ["read_file", "code_search", "list_directory"],
                    "verifying": ["read_file"],
                },
                force_action_hints={
                    "max_iterations": "Please summarize the architecture and provide your recommendations.",
                },
            )
        )

        # ACTION - Execute, run, deploy
        self.register(
            TaskTypeDefinition(
                name="action",
                category=TaskCategory.EXECUTION,
                hint="[ACTION] Execute task. Multiple tool calls allowed. Continue until complete.",
                tool_budget=50,
                max_iterations=12,
                priority_tools=["execute_bash", "git_commit", "run_tests"],
                aliases={"execute", "run", "do"},
            )
        )

        # RESEARCH - Web research tasks
        self.register(
            TaskTypeDefinition(
                name="research",
                category=TaskCategory.ANALYSIS,
                hint="[RESEARCH] Web research task. Search and fetch information from the web.",
                tool_budget=20,
                max_iterations=10,
                priority_tools=["web_search", "web_fetch"],
                aliases={"web_search", "internet"},
                stage_tools={
                    "initial": ["web_search"],
                    "reading": ["web_fetch", "web_search"],
                    "executing": ["web_fetch"],
                },
                force_action_hints={
                    "max_iterations": "Please summarize your research findings.",
                },
            )
        )

        # GENERAL - Ambiguous or conversational
        self.register(
            TaskTypeDefinition(
                name="general",
                category=TaskCategory.CONVERSATION,
                hint="[GENERAL] General task. Explore as needed and complete the task.",
                tool_budget=35,
                max_iterations=15,
                priority_tools=["read_file", "list_directory", "code_search"],
                aliases={"default", "chat", "help", "ambiguous"},
                stage_tools={
                    "initial": ["list_directory", "code_search", "read_file"],
                    "reading": ["read_file", "code_search"],
                    "executing": ["edit_files", "write_file", "execute_bash"],
                    "verifying": ["read_file", "run_tests"],
                },
                force_action_hints={
                    "max_iterations": "Please complete the task or explain blockers.",
                },
            )
        )

        # =================================================================
        # Simple Query Types (low budget)
        # =================================================================

        self.register(
            TaskTypeDefinition(
                name="simple",
                category=TaskCategory.CONVERSATION,
                hint="[SIMPLE] Simple query. 1-2 tool calls max. Answer immediately after.",
                tool_budget=2,
                max_iterations=3,
                priority_tools=["list_directory", "read_file"],
                aliases={"quick", "brief"},
            )
        )

        # =================================================================
        # Extended Task Types (from embeddings/task_classifier.py)
        # =================================================================

        # Coding vertical granular types
        self.register(
            TaskTypeDefinition(
                name="refactor",
                category=TaskCategory.MODIFICATION,
                hint="[REFACTOR] Restructure existing code. Read carefully, then make incremental changes.",
                tool_budget=30,
                max_iterations=15,
                priority_tools=["read_file", "edit_files", "code_search"],
                aliases={"restructure", "reorganize"},
            )
        )

        self.register(
            TaskTypeDefinition(
                name="debug",
                category=TaskCategory.ANALYSIS,
                hint="[DEBUG] Find and fix bugs. Read code, trace issues, then fix.",
                tool_budget=25,
                max_iterations=12,
                priority_tools=["read_file", "code_search", "execute_bash"],
                aliases={"troubleshoot", "diagnose"},
            )
        )

        self.register(
            TaskTypeDefinition(
                name="test",
                category=TaskCategory.EXECUTION,
                hint="[TEST] Write or run tests. Create test files or execute test commands.",
                tool_budget=25,
                max_iterations=10,
                priority_tools=["write_file", "execute_bash", "read_file"],
                aliases={"unit_test", "testing"},
            )
        )

        # =================================================================
        # Research vertical types
        # =================================================================

        self.register(
            TaskTypeDefinition(
                name="fact_check",
                category=TaskCategory.ANALYSIS,
                hint="[FACT CHECK] Verify claims with sources. Search for authoritative information.",
                tool_budget=20,
                max_iterations=10,
                priority_tools=["web_search", "web_fetch"],
            )
        )

        self.register(
            TaskTypeDefinition(
                name="literature_review",
                category=TaskCategory.ANALYSIS,
                hint="[LITERATURE REVIEW] Systematic review of knowledge. Search broadly and synthesize.",
                tool_budget=35,
                max_iterations=20,
                priority_tools=["web_search", "web_fetch"],
            )
        )

        self.register(
            TaskTypeDefinition(
                name="competitive_analysis",
                category=TaskCategory.ANALYSIS,
                hint="[COMPETITIVE ANALYSIS] Compare products/services. Research and compare features.",
                tool_budget=35,
                max_iterations=20,
                priority_tools=["web_search", "web_fetch"],
            )
        )

        self.register(
            TaskTypeDefinition(
                name="trend_research",
                category=TaskCategory.ANALYSIS,
                hint="[TREND RESEARCH] Identify patterns and emerging developments.",
                tool_budget=35,
                max_iterations=20,
                priority_tools=["web_search", "web_fetch"],
            )
        )

        self.register(
            TaskTypeDefinition(
                name="technical_research",
                category=TaskCategory.ANALYSIS,
                hint="[TECHNICAL RESEARCH] Deep dive into technical topics.",
                tool_budget=35,
                max_iterations=20,
                priority_tools=["web_search", "web_fetch", "read_file"],
            )
        )

        # =================================================================
        # Call registration hooks for verticals
        # =================================================================

        for hook in self._registration_hooks:
            try:
                hook(self)
            except Exception as e:
                logger.warning(f"TaskTypeRegistry: Hook failed: {e}")

        logger.info(f"TaskTypeRegistry: Initialized with {len(self._core_types)} core types")


# =================================================================
# Convenience Functions
# =================================================================


def get_task_type_registry() -> TaskTypeRegistry:
    """Get the global TaskTypeRegistry instance.

    Returns:
        The singleton TaskTypeRegistry instance
    """
    return TaskTypeRegistry.get_instance()


def get_task_hint(task_type: str, vertical: Optional[str] = None) -> str:
    """Get the system prompt hint for a task type.

    Args:
        task_type: Name or alias of the task type
        vertical: Optional vertical for specific override

    Returns:
        Hint string for the task type
    """
    return TaskTypeRegistry.get_instance().get_hint(task_type, vertical)


def get_task_budget(task_type: str, vertical: Optional[str] = None) -> int:
    """Get the tool budget for a task type.

    Args:
        task_type: Name or alias of the task type
        vertical: Optional vertical for specific override

    Returns:
        Tool budget for the task type
    """
    return TaskTypeRegistry.get_instance().get_tool_budget(task_type, vertical)


def register_vertical_task_type(
    vertical: str,
    name: str,
    category: TaskCategory,
    hint: str,
    tool_budget: int = 10,
    **kwargs: Any,
) -> None:
    """Register a task type for a specific vertical.

    Convenience function for verticals to register their task types.

    Args:
        vertical: Vertical name (e.g., "devops", "data_analysis")
        name: Task type name
        category: Task category
        hint: System prompt hint
        tool_budget: Maximum tool calls
        **kwargs: Additional TaskTypeDefinition fields
    """
    definition = TaskTypeDefinition(
        name=name,
        category=category,
        hint=hint,
        tool_budget=tool_budget,
        **kwargs,
    )
    TaskTypeRegistry.get_instance().register_for_vertical(vertical, definition)


# =================================================================
# Vertical Registration Hooks
# =================================================================


def register_devops_task_types(registry: TaskTypeRegistry) -> None:
    """Register DevOps-specific task types.

    This function is called as a registration hook to add DevOps vertical
    task types and overrides. Can be used directly or via add_registration_hook.

    Args:
        registry: The TaskTypeRegistry instance
    """
    # Infrastructure task
    registry.register_for_vertical(
        "devops",
        TaskTypeDefinition(
            name="infrastructure",
            category=TaskCategory.MODIFICATION,
            hint="[INFRA] Create or modify infrastructure configs. Review existing setup first.",
            tool_budget=30,
            max_iterations=15,
            priority_tools=["read_file", "write_file", "execute_bash", "list_directory"],
        ),
    )

    # CI/CD task
    registry.register_for_vertical(
        "devops",
        TaskTypeDefinition(
            name="ci_cd",
            category=TaskCategory.MODIFICATION,
            hint="[CI/CD] Configure CI/CD pipelines. Understand workflow structure first.",
            tool_budget=25,
            max_iterations=12,
            priority_tools=["read_file", "write_file", "code_search"],
            aliases={"pipeline", "workflow", "github_actions"},
        ),
    )

    # Kubernetes task
    registry.register_for_vertical(
        "devops",
        TaskTypeDefinition(
            name="kubernetes",
            category=TaskCategory.MODIFICATION,
            hint="[K8S] Create or modify Kubernetes manifests. Check existing resources.",
            tool_budget=30,
            max_iterations=15,
            priority_tools=["read_file", "write_file", "execute_bash"],
            aliases={"k8s", "helm"},
        ),
    )

    # Terraform task
    registry.register_for_vertical(
        "devops",
        TaskTypeDefinition(
            name="terraform",
            category=TaskCategory.MODIFICATION,
            hint="[TERRAFORM] Create or modify Terraform configs. Plan before applying.",
            tool_budget=35,
            max_iterations=18,
            priority_tools=["read_file", "write_file", "execute_bash", "code_search"],
            aliases={"tf", "iac"},
        ),
    )

    # Docker task
    registry.register_for_vertical(
        "devops",
        TaskTypeDefinition(
            name="dockerfile",
            category=TaskCategory.MODIFICATION,
            hint="[DOCKER] Create or modify Dockerfiles. Follow best practices.",
            tool_budget=20,
            max_iterations=10,
            priority_tools=["read_file", "write_file"],
            aliases={"docker"},
        ),
    )

    # Docker Compose task
    registry.register_for_vertical(
        "devops",
        TaskTypeDefinition(
            name="docker_compose",
            category=TaskCategory.MODIFICATION,
            hint="[COMPOSE] Create or modify docker-compose files. Check service dependencies.",
            tool_budget=25,
            max_iterations=12,
            priority_tools=["read_file", "write_file", "code_search"],
            aliases={"compose"},
        ),
    )

    # Monitoring task
    registry.register_for_vertical(
        "devops",
        TaskTypeDefinition(
            name="monitoring",
            category=TaskCategory.MODIFICATION,
            hint="[MONITORING] Configure monitoring and alerting. Review current setup first.",
            tool_budget=30,
            max_iterations=15,
            priority_tools=["read_file", "write_file", "code_search"],
            aliases={"observability", "alerts"},
        ),
    )

    # Override edit for DevOps (more cautious with infrastructure)
    registry.register_for_vertical(
        "devops",
        TaskTypeDefinition(
            name="edit",
            category=TaskCategory.MODIFICATION,
            hint="[DEVOPS EDIT] Edit infrastructure files carefully. Verify syntax and dependencies.",
            tool_budget=30,
            max_iterations=12,
            priority_tools=["read_file", "edit_files", "execute_bash"],
            force_action_after_read=True,
        ),
    )

    logger.debug("TaskTypeRegistry: Registered DevOps vertical task types")


def register_data_analysis_task_types(registry: TaskTypeRegistry) -> None:
    """Register Data Analysis-specific task types.

    Args:
        registry: The TaskTypeRegistry instance
    """
    # Data profiling task
    registry.register_for_vertical(
        "data_analysis",
        TaskTypeDefinition(
            name="data_profiling",
            category=TaskCategory.ANALYSIS,
            hint="[PROFILE] Profile dataset. Examine structure, types, distributions.",
            tool_budget=30,
            max_iterations=15,
            priority_tools=["read_file", "execute_bash", "notebook_edit"],
        ),
    )

    # Statistical analysis task
    registry.register_for_vertical(
        "data_analysis",
        TaskTypeDefinition(
            name="statistical_analysis",
            category=TaskCategory.ANALYSIS,
            hint="[STATS] Perform statistical analysis. Calculate metrics and significance.",
            tool_budget=40,
            max_iterations=20,
            priority_tools=["read_file", "execute_bash", "notebook_edit"],
            aliases={"statistics"},
        ),
    )

    # Correlation analysis task
    registry.register_for_vertical(
        "data_analysis",
        TaskTypeDefinition(
            name="correlation_analysis",
            category=TaskCategory.ANALYSIS,
            hint="[CORRELATION] Analyze relationships between variables.",
            tool_budget=30,
            max_iterations=15,
            priority_tools=["read_file", "execute_bash", "notebook_edit"],
        ),
    )

    # Regression task
    registry.register_for_vertical(
        "data_analysis",
        TaskTypeDefinition(
            name="regression",
            category=TaskCategory.ANALYSIS,
            hint="[REGRESSION] Build regression models. Fit and evaluate.",
            tool_budget=40,
            max_iterations=20,
            priority_tools=["read_file", "execute_bash", "notebook_edit", "write_file"],
        ),
    )

    # Clustering task
    registry.register_for_vertical(
        "data_analysis",
        TaskTypeDefinition(
            name="clustering",
            category=TaskCategory.ANALYSIS,
            hint="[CLUSTERING] Cluster data points. Find patterns and groupings.",
            tool_budget=40,
            max_iterations=20,
            priority_tools=["read_file", "execute_bash", "notebook_edit"],
        ),
    )

    # Time series task
    registry.register_for_vertical(
        "data_analysis",
        TaskTypeDefinition(
            name="time_series",
            category=TaskCategory.ANALYSIS,
            hint="[TIME SERIES] Analyze temporal patterns. Decompose and forecast.",
            tool_budget=45,
            max_iterations=25,
            priority_tools=["read_file", "execute_bash", "notebook_edit"],
        ),
    )

    # Visualization task
    registry.register_for_vertical(
        "data_analysis",
        TaskTypeDefinition(
            name="visualization",
            category=TaskCategory.MODIFICATION,
            hint="[VIZ] Create data visualizations. Choose appropriate chart types.",
            tool_budget=25,
            max_iterations=12,
            priority_tools=["read_file", "execute_bash", "notebook_edit", "write_file"],
            aliases={"chart", "plot", "graph"},
        ),
    )

    # Override analyze for data analysis (higher budget)
    registry.register_for_vertical(
        "data_analysis",
        TaskTypeDefinition(
            name="analyze",
            category=TaskCategory.ANALYSIS,
            hint="[DATA ANALYZE] Analyze data comprehensively. Profile, explore, and summarize.",
            tool_budget=50,
            max_iterations=25,
            priority_tools=["read_file", "execute_bash", "notebook_edit"],
        ),
    )

    logger.debug("TaskTypeRegistry: Registered Data Analysis vertical task types")


def register_coding_task_types(registry: TaskTypeRegistry) -> None:
    """Register Coding-specific task types.

    Args:
        registry: The TaskTypeRegistry instance
    """
    # Code generation task
    registry.register_for_vertical(
        "coding",
        TaskTypeDefinition(
            name="code_generation",
            category=TaskCategory.MODIFICATION,
            hint="[CODEGEN] Generate code from specification. Follow project patterns.",
            tool_budget=20,
            max_iterations=10,
            priority_tools=["read_file", "write_file", "code_search"],
        ),
    )

    # Override refactor for coding (more thorough)
    registry.register_for_vertical(
        "coding",
        TaskTypeDefinition(
            name="refactor",
            category=TaskCategory.MODIFICATION,
            hint="[REFACTOR] Refactor code systematically. Preserve behavior, improve structure.",
            tool_budget=35,
            max_iterations=20,
            priority_tools=["read_file", "edit_files", "code_search", "run_tests"],
        ),
    )

    # Override debug for coding
    registry.register_for_vertical(
        "coding",
        TaskTypeDefinition(
            name="debug",
            category=TaskCategory.ANALYSIS,
            hint="[DEBUG] Debug code issues. Read stack traces, add logging, trace execution.",
            tool_budget=30,
            max_iterations=15,
            priority_tools=["read_file", "code_search", "execute_bash", "edit_files"],
        ),
    )

    # Override test for coding
    registry.register_for_vertical(
        "coding",
        TaskTypeDefinition(
            name="test",
            category=TaskCategory.EXECUTION,
            hint="[TEST] Write and run tests. Ensure coverage and edge cases.",
            tool_budget=30,
            max_iterations=15,
            priority_tools=["write_file", "execute_bash", "read_file", "code_search"],
        ),
    )

    logger.debug("TaskTypeRegistry: Registered Coding vertical task types")


def register_research_task_types(registry: TaskTypeRegistry) -> None:
    """Register Research-specific task types.

    Args:
        registry: The TaskTypeRegistry instance
    """
    # General research query
    registry.register_for_vertical(
        "research",
        TaskTypeDefinition(
            name="general_query",
            category=TaskCategory.ANALYSIS,
            hint="[QUERY] Answer research question. Search and synthesize information.",
            tool_budget=25,
            max_iterations=12,
            priority_tools=["web_search", "web_fetch"],
        ),
    )

    # Override research for research vertical (higher budget)
    registry.register_for_vertical(
        "research",
        TaskTypeDefinition(
            name="research",
            category=TaskCategory.ANALYSIS,
            hint="[RESEARCH] Comprehensive web research. Search deeply, verify sources.",
            tool_budget=40,
            max_iterations=20,
            priority_tools=["web_search", "web_fetch"],
        ),
    )

    logger.debug("TaskTypeRegistry: Registered Research vertical task types")


def setup_vertical_task_types() -> None:
    """Set up all vertical task types.

    This function should be called during application startup to register
    all vertical-specific task types. It's safe to call multiple times.

    Usage:
        from victor.framework.task_types import setup_vertical_task_types
        setup_vertical_task_types()
    """
    registry = TaskTypeRegistry.get_instance()

    # Register all vertical task types
    register_devops_task_types(registry)
    register_data_analysis_task_types(registry)
    register_coding_task_types(registry)
    register_research_task_types(registry)

    logger.info(
        f"TaskTypeRegistry: Set up vertical task types. " f"Verticals: {registry.list_verticals()}"
    )
