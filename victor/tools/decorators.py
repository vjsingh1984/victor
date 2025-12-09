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


import inspect
import logging
import warnings
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Union, get_origin, get_args

from docstring_parser import parse

from victor.core.errors import ToolExecutionError, ToolValidationError
from victor.tools.base import (
    AccessMode,
    BaseTool,
    CostTier,
    DangerLevel,
    ExecutionCategory,
    Priority,
    ToolMetadata,
    ToolResult,
)

# Module-level flag to control auto-registration (can be disabled for testing)
_AUTO_REGISTER_TOOLS = True


def set_auto_register(enabled: bool) -> None:
    """Enable or disable automatic tool registration with metadata registry.

    By default, tools are automatically registered with the global
    ToolMetadataRegistry when decorated with @tool. This enables
    registry-driven tool discovery without explicit registration calls.

    Args:
        enabled: If True, tools are auto-registered at decoration time.
                 Set to False for unit tests that need isolated tool instances.
    """
    global _AUTO_REGISTER_TOOLS
    _AUTO_REGISTER_TOOLS = enabled


def _auto_register_tool(tool_instance: BaseTool) -> None:
    """Register a tool instance with the global metadata registry.

    This is called automatically when a tool is decorated with @tool,
    unless auto-registration is disabled via set_auto_register(False).

    Args:
        tool_instance: The tool instance to register
    """
    if not _AUTO_REGISTER_TOOLS:
        return

    try:
        from victor.tools.metadata_registry import register_tool_metadata

        register_tool_metadata(tool_instance)
        logger.debug(f"Auto-registered tool '{tool_instance.name}' with metadata registry")
    except ImportError:
        # metadata_registry not available (e.g., during early import stages)
        logger.debug(f"Could not auto-register tool '{tool_instance.name}': registry not available")


logger = logging.getLogger(__name__)

# Global flag to control deprecation warnings for legacy tool names
_WARN_ON_LEGACY_NAMES = False


def set_legacy_name_warnings(enabled: bool) -> None:
    """Enable or disable warnings when legacy tool names are used.

    Args:
        enabled: If True, log warnings when tools are called with legacy names.
                 Useful for identifying code that needs migration.
    """
    global _WARN_ON_LEGACY_NAMES
    _WARN_ON_LEGACY_NAMES = enabled


def resolve_tool_name(name: str, warn_on_legacy: bool = False) -> str:
    """Resolve a tool name to its canonical form.

    This provides centralized name resolution that can be used across the codebase.
    Uses the tool_names registry if available, otherwise returns the name unchanged.

    Args:
        name: Tool name (canonical or legacy)
        warn_on_legacy: If True, emit a warning when a legacy name is used

    Returns:
        Canonical tool name

    Example:
        >>> resolve_tool_name("execute_bash")
        "shell"
        >>> resolve_tool_name("shell")
        "shell"
    """
    try:
        from victor.tools.tool_names import get_canonical_name, TOOL_ALIASES

        canonical = get_canonical_name(name)

        # Check if this was a legacy name
        if (warn_on_legacy or _WARN_ON_LEGACY_NAMES) and name in TOOL_ALIASES:
            logger.warning(
                f"Legacy tool name '{name}' used. Consider using canonical name '{canonical}' instead."
            )

        return canonical
    except ImportError:
        # tool_names module not available, return unchanged
        return name


def _get_json_schema_type(annotation: Any) -> Dict[str, Any]:
    """Convert Python type annotation to JSON Schema type.

    Args:
        annotation: Python type annotation

    Returns:
        JSON Schema type definition
    """
    if annotation == inspect.Parameter.empty:
        return {"type": "string"}

    # Handle Optional[X] -> X with nullable
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Union:
        # Check if it's Optional (Union with None)
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            # It's Optional[X]
            inner_schema = _get_json_schema_type(non_none_args[0])
            return inner_schema  # JSON Schema handles nullable differently
        else:
            # Complex Union - default to string
            return {"type": "string"}

    if origin is list or annotation is list:
        if args:
            items_schema = _get_json_schema_type(args[0])
            return {"type": "array", "items": items_schema}
        return {"type": "array", "items": {"type": "string"}}

    if origin is dict or annotation is dict:
        return {"type": "object"}

    # Basic types
    if annotation in (int,):
        return {"type": "integer"}
    if annotation in (float,):
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is str:
        return {"type": "string"}

    # Default to string for unknown types
    return {"type": "string"}


def tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    cost_tier: CostTier = CostTier.FREE,
    category: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    use_cases: Optional[List[str]] = None,
    examples: Optional[List[str]] = None,
    priority_hints: Optional[List[str]] = None,
    # Selection/approval metadata parameters
    priority: Priority = Priority.MEDIUM,
    access_mode: AccessMode = AccessMode.READONLY,
    danger_level: DangerLevel = DangerLevel.SAFE,
    # Stage affinity for conversation state machine
    stages: Optional[List[str]] = None,
    # NEW: Mandatory keywords that force tool inclusion
    mandatory_keywords: Optional[List[str]] = None,
    # NEW: Task types for classification-aware selection
    task_types: Optional[List[str]] = None,
    # NEW: Progress parameters for loop detection
    progress_params: Optional[List[str]] = None,
    # NEW: Execution category for parallel execution
    execution_category: Optional[str] = None,
    # NEW: Availability check for optional tools requiring configuration
    availability_check: Optional[Callable[[], bool]] = None,
) -> Union[Callable, Callable[[Callable], Callable]]:
    """
    A decorator that converts a Python function into a Victor tool.

    This decorator automatically extracts the tool's name, description,
    and parameters from the function's signature and docstring. It also
    supports optional semantic metadata for dynamic tool selection.

    Tool Naming Strategy:
    - If `name` is provided, use it as the canonical name
    - If function name is in TOOL_ALIASES, auto-resolve to canonical name
    - Otherwise, use function name as-is

    Category Semantics:
    - category="filesystem": File operations (read, write, ls)
    - category="git": Version control operations
    - category="web": Web search/fetch operations
    - category="analysis": Code analysis and review tools
    - category="execution": Shell and command execution
    - etc.

    Priority Semantics (decoupled from category):
    - CRITICAL: Always available (read, ls, grep, shell)
    - HIGH: Most tasks (write, edit, git)
    - MEDIUM: Task-specific (docker, db, test)
    - LOW: Specialized (batch, scaffold)
    - CONTEXTUAL: Based on task classification

    Args:
        func: The function to wrap (optional, for @tool without parentheses)
        name: Explicit tool name (overrides function name and auto-resolution)
        aliases: Additional names that resolve to this tool (backward compat)
        cost_tier: Cost tier for the tool (FREE, LOW, MEDIUM, HIGH)
        category: Tool category for semantic grouping.
        keywords: Keywords that trigger this tool in user requests
        use_cases: High-level use cases for semantic matching
        examples: Example requests that should match this tool
        priority_hints: Usage hints for tool selection
        priority: Tool priority for selection (CRITICAL, HIGH, MEDIUM, LOW, CONTEXTUAL)
        access_mode: Access mode for approval tracking (READONLY, WRITE, EXECUTE, NETWORK, MIXED)
        danger_level: Danger level for warnings (SAFE, LOW, MEDIUM, HIGH, CRITICAL)
        stages: Conversation stages where this tool is most relevant. Valid stages:
                "initial", "planning", "reading", "analysis", "execution",
                "verification", "completion". Tools with stages defined will
                be prioritized when the conversation is in matching stages.
        mandatory_keywords: Keywords that FORCE this tool to be included when matched.
                           Unlike regular keywords, these guarantee tool inclusion.
                           E.g., ["show diff", "compare"] for shell tool.
        task_types: Task types this tool is relevant for. Valid types:
                   "analysis", "action", "generation", "search", "edit", "default".
                   Used for classification-aware tool selection.
        progress_params: Parameters that indicate progress in loop detection.
                        If these params change between calls, it's progress not a loop.
                        E.g., ["path", "offset", "limit"] for read tool.
        execution_category: Category for parallel execution. Valid values:
                           "read_only", "write", "compute", "network", "execute", "mixed".
                           Determines which tools can safely run concurrently.
        availability_check: Optional callable that returns True if the tool is available.
                           Use this for tools that require external configuration
                           (e.g., Slack, Teams, Jira). When provided, the tool's
                           is_available() method will call this function. Unavailable
                           tools are excluded from tool selection and system prompts.
                           Example: availability_check=is_slack_configured

    The docstring should follow the Google Python Style Guide format.

    Example with auto-generated metadata:
        @tool
        def my_tool(param1: str, param2: int = 5):
            '''This is the tool's description.

            Args:
                param1: Description of the first parameter.
                param2: Description of the second parameter.
            '''
            # ... tool logic ...

    Example with core tool (always available):
        @tool(name="shell", category="core")
        def execute_bash(cmd: str):
            '''Run shell command. Platform-agnostic.'''
            # ... tool logic ...

    Example with explicit metadata:
        @tool(
            name="web",
            cost_tier=CostTier.MEDIUM,
            category="web",
            keywords=["search", "google", "find online"],
        )
        def web_search_tool(query: str):
            '''Search the web - makes external API calls.'''
            # ... tool logic ...
    """
    # Capture semantic metadata parameters
    metadata_params = {
        "category": category,
        "keywords": keywords,
        "use_cases": use_cases,
        "examples": examples,
        "priority_hints": priority_hints,
        "stages": stages,
        "mandatory_keywords": mandatory_keywords,
        "task_types": task_types,
        "progress_params": progress_params,
        "execution_category": execution_category,
        "availability_check": availability_check,
    }

    # Capture selection/approval metadata parameters
    selection_params = {
        "priority": priority,
        "access_mode": access_mode,
        "danger_level": danger_level,
    }

    # Capture naming parameters
    _explicit_name = name
    _explicit_aliases = aliases or []

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            # This wrapper is what gets called if the decorated function is called directly
            return fn(*args, **kwargs)

        # Mark as tool for dynamic discovery
        wrapper._is_tool = True  # type: ignore[attr-defined]
        # We will attach a class to the wrapper that is the actual tool
        wrapper.Tool = _create_tool_class(
            fn,
            cost_tier=cost_tier,
            metadata_params=metadata_params,
            selection_params=selection_params,
            explicit_name=_explicit_name,
            explicit_aliases=_explicit_aliases,
        )

        return wrapper

    # Support both @tool and @tool(cost_tier=...) syntax
    if func is not None:
        # Called without parentheses: @tool
        return decorator(func)
    else:
        # Called with parentheses: @tool(cost_tier=...)
        return decorator


def _resolve_tool_name(func_name: str, explicit_name: Optional[str] = None) -> str:
    """Resolve the canonical tool name.

    Resolution order:
    1. Explicit name parameter (highest priority)
    2. Auto-resolve from TOOL_ALIASES registry
    3. Function name as-is (fallback)

    Args:
        func_name: The original function name
        explicit_name: Explicitly provided name (overrides all)

    Returns:
        The resolved canonical tool name
    """
    if explicit_name:
        return explicit_name

    # Try to auto-resolve from registry
    try:
        from victor.tools.tool_names import get_canonical_name

        return get_canonical_name(func_name)
    except ImportError:
        # Registry not available, use function name
        return func_name


def _create_tool_class(
    func: Callable,
    cost_tier: CostTier = CostTier.FREE,
    metadata_params: Optional[Dict[str, Any]] = None,
    selection_params: Optional[Dict[str, Any]] = None,
    explicit_name: Optional[str] = None,
    explicit_aliases: Optional[List[str]] = None,
) -> type:
    """Dynamically creates a class that wraps the given function to act as a BaseTool.

    Args:
        func: The function to wrap
        cost_tier: The cost tier for this tool
        metadata_params: Optional dict with category, keywords, use_cases, examples, priority_hints
        selection_params: Optional dict with priority, access_mode, danger_level
        explicit_name: Optional explicit tool name (overrides function name)
        explicit_aliases: Optional list of alias names for backward compatibility
    """
    metadata_params = metadata_params or {}
    selection_params = selection_params or {}
    explicit_aliases = explicit_aliases or []

    # Resolve the canonical tool name
    resolved_name = _resolve_tool_name(func.__name__, explicit_name)

    docstring = parse(func.__doc__ or "")
    tool_description = docstring.short_description or "No description provided."
    if docstring.long_description:
        tool_description += "\n\n" + docstring.long_description

    # Create the JSON schema for the parameters
    sig = inspect.signature(func)
    param_docs = {p.arg_name: p.description for p in docstring.params}

    properties = {}
    required = []
    for name, param in sig.parameters.items():
        if param.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue

        # Use the enhanced type handler for proper JSON Schema generation
        type_schema = _get_json_schema_type(param.annotation)

        # Merge type schema with description
        properties[name] = {
            **type_schema,
            "description": param_docs.get(name, "No description."),
        }

        # Add default value if present (helps LLMs understand optional params)
        if param.default != inspect.Parameter.empty and param.default is not None:
            properties[name]["default"] = param.default

        if param.default == inspect.Parameter.empty:
            required.append(name)

    tool_params_schema = {
        "type": "object",
        "properties": properties,
        "required": required,
    }

    # Capture closures
    _cost_tier = cost_tier
    _metadata_params = metadata_params
    _selection_params = selection_params
    _resolved_name = resolved_name
    _original_func_name = func.__name__

    # Build alias set: explicit aliases + original function name (if different from resolved)
    _aliases = set(explicit_aliases)
    if _original_func_name != _resolved_name:
        _aliases.add(_original_func_name)  # Auto-add original name for backward compat

    # Extract selection params with defaults
    _priority = _selection_params.get("priority", Priority.MEDIUM)
    _access_mode = _selection_params.get("access_mode", AccessMode.READONLY)
    _danger_level = _selection_params.get("danger_level", DangerLevel.SAFE)

    # Extract stages for stage-based tool selection
    _stages = _metadata_params.get("stages") or []

    # Extract new decorator-driven fields for semantic selection
    _mandatory_keywords = _metadata_params.get("mandatory_keywords") or []
    _task_types = _metadata_params.get("task_types") or []
    _progress_params = _metadata_params.get("progress_params") or []
    _availability_check = _metadata_params.get("availability_check")

    # Parse execution_category from string to enum
    _exec_cat_str = _metadata_params.get("execution_category")
    _execution_category: Optional[ExecutionCategory] = None
    if _exec_cat_str:
        try:
            _execution_category = ExecutionCategory(_exec_cat_str)
        except ValueError:
            logger.warning(
                f"Invalid execution_category '{_exec_cat_str}' for tool '{resolved_name}', "
                f"using default READ_ONLY"
            )
            _execution_category = ExecutionCategory.READ_ONLY

    # Build explicit metadata if any params were provided
    _explicit_metadata: Optional[ToolMetadata] = None
    has_explicit = any(v is not None for k, v in _metadata_params.items())
    if has_explicit:
        _explicit_metadata = ToolMetadata(
            category=_metadata_params.get("category") or "",
            keywords=_metadata_params.get("keywords") or [],
            use_cases=_metadata_params.get("use_cases") or [],
            examples=_metadata_params.get("examples") or [],
            priority_hints=_metadata_params.get("priority_hints") or [],
            stages=_stages,
            mandatory_keywords=_mandatory_keywords,
            task_types=_task_types,
            progress_params=_progress_params,
            execution_category=_execution_category,
        )

    # Dynamically create the tool class
    class FunctionTool(BaseTool):
        def __init__(self, fn: Callable):
            self._fn = fn
            self._name = _resolved_name  # Use resolved canonical name
            self._original_name = _original_func_name  # Keep original for debugging
            self._aliases = set(_aliases)  # Store aliases for backward compatibility
            self._description = tool_description
            self._parameters = tool_params_schema
            self._cost_tier = _cost_tier
            self._explicit_metadata = _explicit_metadata
            self._category = _metadata_params.get("category")  # Category for grouping
            self._stages = _stages  # Stage affinity for conversation state machine
            # New selection/approval metadata
            self._priority = _priority
            self._access_mode = _access_mode
            self._danger_level = _danger_level
            # New decorator-driven semantic selection fields
            self._mandatory_keywords = _mandatory_keywords
            self._task_types = _task_types
            self._progress_params = _progress_params
            self._execution_category = _execution_category or ExecutionCategory.READ_ONLY
            # Availability check for optional tools requiring configuration
            self._availability_check = _availability_check

        @property
        def name(self) -> str:
            return self._name

        @property
        def description(self) -> str:
            return self._description

        @property
        def parameters(self) -> Dict[str, Any]:
            return self._parameters

        @property
        def cost_tier(self) -> CostTier:
            return self._cost_tier

        @property
        def aliases(self) -> Set[str]:
            """Return alias names for backward compatibility."""
            return self._aliases

        @property
        def original_name(self) -> str:
            """Return the original function name (before renaming)."""
            return self._original_name

        @property
        def is_critical(self) -> bool:
            """Return True if this tool has CRITICAL priority.

            Critical tools are always available for selection regardless of task type.
            This replaces the legacy 'is_core' property which checked category='core'.
            """
            return self._priority == Priority.CRITICAL

        @property
        def category(self) -> Optional[str]:
            """Return the tool category for semantic grouping."""
            return self._category

        @property
        def keywords(self) -> List[str]:
            """Return keywords for semantic matching from @tool decorator."""
            if self._explicit_metadata and self._explicit_metadata.keywords:
                return self._explicit_metadata.keywords
            return []

        @property
        def stages(self) -> List[str]:
            """Return stages where this tool is relevant from @tool decorator.

            Stages indicate conversation phases where this tool is most useful:
            - "initial": First interaction, exploring the request
            - "planning": Understanding scope, searching for files
            - "reading": Examining files, gathering context
            - "analysis": Reviewing code, analyzing structure
            - "execution": Making changes, running commands
            - "verification": Testing, validating changes
            - "completion": Summarizing, wrapping up
            """
            return self._stages

        @property
        def mandatory_keywords(self) -> List[str]:
            """Return mandatory keywords that force tool inclusion.

            Unlike regular keywords, mandatory keywords guarantee tool inclusion
            when matched in user requests. Useful for tools that handle
            specific operations like "show diff" or "compare files".
            """
            return self._mandatory_keywords

        @property
        def task_types(self) -> List[str]:
            """Return task types this tool is relevant for.

            Valid types: "analysis", "action", "generation", "search", "edit", "default".
            Used for classification-aware tool selection to match tools with task intent.
            """
            return self._task_types

        @property
        def progress_params(self) -> List[str]:
            """Return progress parameters for loop detection.

            These parameters indicate meaningful progress when changed between
            tool calls. If these params differ, it's exploration not repetition.
            E.g., ["path", "offset", "limit"] for read tool.
            """
            return self._progress_params

        @property
        def execution_category(self) -> ExecutionCategory:
            """Return execution category for parallel execution planning.

            Categories determine which tools can safely run concurrently:
            - READ_ONLY: Pure reads, safe to parallelize
            - WRITE: File modifications, may conflict
            - COMPUTE: CPU-intensive but isolated
            - NETWORK: External calls, rate-limited
            - EXECUTE: Shell commands, may have side effects
            - MIXED: Multiple categories, careful dependency analysis needed
            """
            return self._execution_category

        @property
        def priority(self) -> Priority:
            """Return tool priority for selection availability."""
            return self._priority

        @property
        def access_mode(self) -> AccessMode:
            """Return tool access mode for approval tracking."""
            return self._access_mode

        @property
        def danger_level(self) -> DangerLevel:
            """Return danger level for warning/confirmation logic."""
            return self._danger_level

        @property
        def requires_approval(self) -> bool:
            """Check if this tool requires user approval before execution."""
            return self._access_mode.requires_approval or self._danger_level.requires_confirmation

        @property
        def is_safe(self) -> bool:
            """Check if this tool is safe (readonly, no danger)."""
            return self._access_mode.is_safe and self._danger_level == DangerLevel.SAFE

        @property
        def requires_configuration(self) -> bool:
            """Check if this tool requires external configuration.

            Returns True if an availability_check was provided in the decorator,
            indicating this tool needs configuration (API keys, credentials, etc.)
            before it can be used. Examples: Slack, Teams, Jira.
            """
            return self._availability_check is not None

        def is_available(self) -> bool:
            """Check if this tool is currently available for use.

            For tools without an availability_check, always returns True.
            For tools with an availability_check (e.g., Slack, Teams), calls
            the provided function to check if the tool is properly configured.

            Returns:
                True if the tool is available, False if it requires configuration
                that hasn't been completed.
            """
            if self._availability_check is None:
                return True
            try:
                return self._availability_check()
            except Exception as e:
                logger.warning(f"Availability check for tool '{self._name}' raised exception: {e}")
                return False

        def get_warning_message(self) -> str:
            """Get appropriate warning message for this tool's danger level."""
            return self._danger_level.warning_message

        def should_include_for_task(self, task_type: str) -> bool:
            """Check if this tool should be included for the given task type."""
            return self._priority.should_include_for_task(task_type)

        def all_names(self) -> Set[str]:
            """Return all valid names (canonical + aliases)."""
            return {self._name} | self._aliases

        def matches_name(self, query_name: str) -> bool:
            """Check if this tool matches the given name (canonical or alias)."""
            return query_name == self._name or query_name in self._aliases

        @property
        def metadata(self) -> Optional[ToolMetadata]:
            """Return explicit metadata if provided via decorator."""
            return self._explicit_metadata

        def get_metadata(self) -> ToolMetadata:
            """Get semantic metadata (ToolMetadataProvider contract).

            Returns explicit metadata if provided via @tool decorator,
            otherwise auto-generates from tool properties.
            """
            if self._explicit_metadata is not None:
                return self._explicit_metadata

            # Auto-generate metadata from tool properties
            return ToolMetadata.generate_from_tool(
                name=self._name,
                description=self._description,
                parameters=self._parameters,
                cost_tier=self._cost_tier,
            )

        async def execute(self, _exec_ctx: Dict[str, Any], **kwargs: Any) -> ToolResult:
            try:
                # Check if the target function wants the framework execution context
                # Note: We use _exec_ctx to avoid collision with tool parameters named 'context'
                sig = inspect.signature(self._fn)
                if "_exec_ctx" in sig.parameters:
                    kwargs["_exec_ctx"] = _exec_ctx

                if inspect.iscoroutinefunction(self._fn):
                    result = await self._fn(**kwargs)
                else:
                    result = self._fn(**kwargs)

                # Handle dict-based error returns for backwards compatibility
                # Tools returning {"success": False, "error": "..."} should be converted
                # to proper ToolResult failures. This bridges legacy patterns.
                if isinstance(result, dict):
                    if result.get("success") is False and "error" in result:
                        return ToolResult(
                            success=False,
                            output=result.get("output"),
                            error=result.get("error"),
                            metadata=result.get("metadata"),
                        )
                    # Some tools return {"error": "..."} without success field
                    if "error" in result and "success" not in result and len(result) <= 2:
                        return ToolResult(
                            success=False,
                            output=None,
                            error=result.get("error"),
                        )

                return ToolResult(success=True, output=result)
            except (ToolValidationError, ToolExecutionError) as e:
                # Handle structured tool errors with recovery hints
                return ToolResult(
                    success=False,
                    output=None,
                    error=e.message,
                    metadata={
                        "exception": type(e).__name__,
                        "category": e.category.value,
                        "recovery_hint": e.recovery_hint,
                        "tool_name": e.tool_name,
                    },
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    output=None,
                    error=str(e),
                    metadata={"exception": type(e).__name__},
                )

    # Create the tool instance and auto-register with global metadata registry
    tool_instance = FunctionTool(func)
    _auto_register_tool(tool_instance)
    return tool_instance
