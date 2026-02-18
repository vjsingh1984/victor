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

"""Vertical Extension Loading Capabilities.

This module provides a comprehensive extension loading system for Victor verticals,
enabling modular integration with framework components through lazy loading and
caching infrastructure.

Key Features:
- Fine-grained extension caching with composite keys to prevent collisions
- Lazy loading with auto-generated class names for reduced boilerplate
- Strict error handling with configurable extension requirements
- Support for 10+ extension types (middleware, safety, prompt, workflow, etc.)

Extension Types:
    - Middleware: Tool execution pipeline processing
    - Safety Extension: Dangerous operation pattern detection
    - Prompt Contributor: Task hints and prompt sections
    - Mode Config Provider: Operational mode configurations
    - Tool Dependency Provider: Tool execution patterns
    - Workflow Provider: Vertical-specific workflow definitions
    - Service Provider: DI container integration
    - RL Config Provider: Reinforcement learning configurations
    - Team Spec Provider: Multi-agent team formations
    - Enrichment Strategy: DSPy-like prompt optimization
    - Tiered Tool Config: Context-aware tool selection

Design Principles:
    - SRP Compliance: Extracted from VerticalBase for focused responsibility
    - Lazy Loading: Extensions loaded only when first accessed
    - Caching: Fine-grained cache with composite keys (ClassName:extension_key)
    - Error Tolerance: Graceful degradation with strict mode optional
    - Auto-Discovery: Auto-generated class names reduce boilerplate

Usage:
    from victor.core.verticals.extension_loader import VerticalExtensionLoader

    class MyVertical(VerticalExtensionLoader):
        @classmethod
        def get_safety_extension(cls):
            # Use generic factory (auto-generates "MyVerticalSafetyExtension")
            return cls._get_extension_factory(
                "safety_extension",
                "myvertical.safety",
            )

        @classmethod
        def get_prompt_contributor(cls):
            # Custom class name
            return cls._get_extension_factory(
                "prompt_contributor",
                "myvertical.prompts",
                "MyCustomPromptContributor",
            )

Error Handling:
    # Strict mode (any failure raises ExtensionLoadError)
    class StrictVertical(VerticalExtensionLoader):
        strict_extension_loading = True

    # Required extensions (must succeed even in non-strict mode)
    class CriticalVertical(VerticalExtensionLoader):
        required_extensions = {"safety", "middleware"}

Related Modules:
    - victor.core.verticals.base: VerticalBase (uses this loader)
    - victor.core.verticals.protocols: Extension protocol definitions
    - victor.core.verticals.metadata: Metadata provider
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Set, Type

if TYPE_CHECKING:
    from victor.core.verticals.protocols import VerticalExtensions
    from victor.core.vertical_types import TieredToolConfig

logger = logging.getLogger(__name__)


class VerticalExtensionLoader(ABC):
    """Loader of vertical extensions.

    Handles loading and caching of vertical extensions including middleware,
    safety, prompt contributors, mode configs, and more.

    This is a mix-in class that provides extension loading capabilities
    to verticals while maintaining SRP compliance.

    Extension Loading Configuration:
        strict_extension_loading: When True, any extension loading failure
            raises ExtensionLoadError.
        required_extensions: Extensions that must load successfully even
            when strict_extension_loading=False.
    """

    # Extension loading configuration
    # When True, any extension loading failure raises ExtensionLoadError
    strict_extension_loading: ClassVar[bool] = False

    # Extensions that must load successfully even when strict_extension_loading=False
    # Valid values: "middleware", "safety", "prompt", "mode_config", "tool_deps",
    #               "workflow", "service", "rl_config", "team_spec", "enrichment",
    #               "tiered_tools"
    required_extensions: ClassVar[Set[str]] = set()

    # Extension cache (shared across all verticals)
    _extensions_cache: Dict[str, Any] = {}
    _extensions_cache_lock: ClassVar[threading.RLock] = threading.RLock()

    @classmethod
    def _cache_namespace(cls) -> str:
        """Return namespaced cache prefix for this vertical class."""
        return f"{cls.__name__}:{cls.__module__}:{cls.__qualname__}"

    # =========================================================================
    # Extension Caching Infrastructure
    # =========================================================================

    @classmethod
    def _get_cached_extension(cls, key: str, factory: callable) -> Any:
        """Get extension from cache or create and cache it.

        This helper enables fine-grained caching of individual extension
        instances, avoiding repeated object creation when extensions are
        accessed multiple times.

        The cache uses a composite key of (class_name, extension_key) to
        ensure proper isolation between different vertical subclasses.

        Args:
            key: Unique key for this extension type (e.g., "middleware",
                 "safety_extension", "workflow_provider")
            factory: Zero-argument callable that creates the extension instance.
                     Only called if the extension is not already cached.

        Returns:
            Cached or newly created extension instance.

        Example:
            @classmethod
            def get_middleware(cls) -> List[MiddlewareProtocol]:
                def _create():
                    from myvertical.middleware import MyMiddleware
                    return [MyMiddleware()]
                return cls._get_cached_extension("middleware", _create)
        """
        # Use namespaced composite key to avoid collisions across modules.
        cache_key = f"{cls._cache_namespace()}:{key}"
        with cls._extensions_cache_lock:
            if cache_key not in cls._extensions_cache:
                cls._extensions_cache[cache_key] = factory()
            return cls._extensions_cache[cache_key]

    @classmethod
    def _get_extension_factory(
        cls,
        extension_key: str,
        import_path: str,
        attribute_name: Optional[str] = None,
    ) -> Any:
        """Generic factory for lazy-loading and caching extensions.

        Eliminates boilerplate across all verticals by providing a single
        implementation of the lazy import + create + cache pattern.

        Args:
            extension_key: Cache key (e.g., "safety_extension", "prompt_contributor")
            import_path: Full Python import path (e.g., "victor.coding.safety")
            attribute_name: Class name to import. If None, auto-generates from vertical name
                          (e.g., "CodingSafetyExtension" for CodingAssistant)

        Returns:
            Cached or newly created extension instance

        Example:
            # Before (13 lines)
            def get_safety_extension(cls):
                def _create():
                    from victor.coding.safety import CodingSafetyExtension
                    return CodingSafetyExtension()
                return cls._get_cached_extension("safety_extension", _create)

            # After (3 lines)
            def get_safety_extension(cls):
                return cls._get_extension_factory(
                    "safety_extension",
                    "victor.coding.safety",
                )
        """

        def _create():
            # Determine the class name to import
            if attribute_name is None:
                # Auto-generate class name
                # Convert "CodingAssistant" -> "Coding"
                vertical_name = cls.__name__.replace("Assistant", "")
                # Convert "safety_extension" -> "SafetyExtension"
                extension_type = extension_key.replace("_", " ").title().replace(" ", "")
                class_name = f"{vertical_name}{extension_type}"
            else:
                class_name = attribute_name

            # Lazy import: only loads module when first called
            module = __import__(import_path, fromlist=[class_name])

            # Import and instantiate
            return getattr(module, class_name)()

        # Use existing caching infrastructure
        return cls._get_cached_extension(extension_key, _create)

    # =========================================================================
    # Extension Protocol Methods (Optional)
    # =========================================================================
    # These methods enable verticals to provide framework extensions.
    # Override them to integrate with the framework's middleware, safety,
    # prompt, and configuration systems.

    @classmethod
    def get_middleware(cls) -> List[Any]:
        """Get middleware implementations for this vertical.

        Override to provide vertical-specific middleware for tool
        execution processing.

        Returns:
            List of middleware implementations (MiddlewareProtocol)
        """
        return []

    @classmethod
    def get_safety_extension(cls) -> Optional[Any]:
        """Get safety extension for this vertical.

        Default implementation uses the extension factory pattern with the vertical's
        safety module. Override only if custom behavior needed.

        This eliminates ~20 LOC of duplicated wrapper code across verticals.

        Returns:
            Safety extension (SafetyExtensionProtocol) or None
        """
        try:
            return cls._get_extension_factory(
                "safety_extension",
                f"victor.{cls.name}.safety",
            )
        except (ImportError, AttributeError):
            return None

    @classmethod
    def get_prompt_contributor(cls) -> Optional[Any]:
        """Get prompt contributor for this vertical.

        Default implementation uses the extension factory pattern with the vertical's
        prompts module. Override only if custom behavior needed.

        This eliminates ~25 LOC of duplicated wrapper code across verticals.

        Returns:
            Prompt contributor (PromptContributorProtocol) or None
        """
        try:
            return cls._get_extension_factory(
                "prompt_contributor",
                f"victor.{cls.name}.prompts",
            )
        except (ImportError, AttributeError):
            return None

    @classmethod
    def get_mode_config_provider(cls) -> Optional[Any]:
        """Get mode configuration provider for this vertical.

        Override to provide vertical-specific operational modes.

        Returns:
            Mode config provider (ModeConfigProviderProtocol) or None
        """
        return None

    @classmethod
    def get_mode_config(cls) -> Dict[str, Any]:
        """Get mode configurations for this vertical.

        Returns operational modes like 'fast', 'thorough', 'explore' with
        their configurations (tool_budget, max_iterations, temperature).

        Default implementation provides standard modes. Override in subclasses
        for vertical-specific mode configurations.

        Returns:
            Dictionary mapping mode names to ModeConfig-like dicts.
        """
        return {
            "fast": {
                "name": "fast",
                "tool_budget": 10,
                "max_iterations": 20,
                "temperature": 0.7,
                "description": "Quick responses with limited tool usage",
            },
            "thorough": {
                "name": "thorough",
                "tool_budget": 50,
                "max_iterations": 50,
                "temperature": 0.7,
                "description": "Comprehensive analysis with extensive tool usage",
            },
            "explore": {
                "name": "explore",
                "tool_budget": 30,
                "max_iterations": 30,
                "temperature": 0.9,
                "description": "Exploratory mode with higher creativity",
            },
        }

    @classmethod
    def get_task_type_hints(cls) -> Dict[str, Any]:
        """Get task-type-specific prompt hints.

        Returns hints for common task types (edit, search, explain, etc.)
        with tool priorities and budget recommendations.

        Default implementation provides standard hints. Override in subclasses
        for vertical-specific task type hints.

        Returns:
            Dictionary mapping task types to TaskTypeHint-like dicts.
        """
        return {
            "edit": {
                "task_type": "edit",
                "hint": "[EDIT MODE] Read target files first, then make focused modifications.",
                "tool_budget": 15,
                "priority_tools": ["read", "edit", "grep"],
            },
            "search": {
                "task_type": "search",
                "hint": "[SEARCH MODE] Use semantic search and grep for efficient discovery.",
                "tool_budget": 10,
                "priority_tools": ["grep", "code_search", "ls"],
            },
            "explain": {
                "task_type": "explain",
                "hint": "[EXPLAIN MODE] Read relevant code and provide clear explanations.",
                "tool_budget": 8,
                "priority_tools": ["read", "grep", "overview"],
            },
            "debug": {
                "task_type": "debug",
                "hint": "[DEBUG MODE] Investigate systematically, check logs and error messages.",
                "tool_budget": 20,
                "priority_tools": ["read", "grep", "shell", "run_tests"],
            },
            "implement": {
                "task_type": "implement",
                "hint": "[IMPLEMENT MODE] Plan first, implement incrementally, verify each step.",
                "tool_budget": 30,
                "priority_tools": ["read", "write", "edit", "shell"],
            },
        }

    @classmethod
    def get_tool_dependency_provider(cls) -> Optional[Any]:
        """Get tool dependency provider for this vertical.

        Default implementation uses the framework's create_vertical_tool_dependency_provider()
        factory function with the vertical's name. Override only if custom behavior needed.

        This eliminates ~25 LOC of duplicated wrapper code across verticals.

        Returns:
            Tool dependency provider (ToolDependencyProviderProtocol) or None
        """
        try:
            from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider

            return cls._get_cached_extension(
                "tool_dependency_provider",
                lambda: create_vertical_tool_dependency_provider(cls.name),
            )
        except (ImportError, ValueError):
            # If factory not available or vertical not recognized, return None
            return None

    @classmethod
    def get_tool_graph(cls) -> Optional[Any]:
        """Get tool execution graph for this vertical.

        Override to provide a custom ToolExecutionGraph that defines
        tool dependencies, transitions, and execution sequences.
        The graph is registered with the global ToolGraphRegistry.

        Returns:
            ToolExecutionGraph instance or None (no graph registered)
        """
        return None

    @classmethod
    def get_tiered_tool_config(cls) -> Optional["TieredToolConfig"]:
        """Get tiered tool configuration for this vertical.

        This is a placeholder for compatibility. The actual implementation
        is in VerticalMetadataProvider but duplicated here for standalone use.

        Returns:
            TieredToolConfig or None if vertical doesn't use tiered config.
        """
        return None

    @classmethod
    def get_tiered_tools(cls) -> Optional[Any]:
        """Get tiered tool configuration for intelligent selection.

        DEPRECATED: Use get_tiered_tool_config() instead.

        This method is maintained for backward compatibility. New verticals
        should override get_tiered_tool_config() which has a default
        implementation using TieredToolTemplate.

        Override to provide vertical-specific tiered tool configuration
        for context-efficient tool selection. When implemented, this enables:

        1. Mandatory tools: Always included (e.g., read, ls)
        2. Vertical core: Always included for this vertical (e.g., web, fetch for research)
        3. Semantic pool: Selected based on query similarity and stage

        Example for research vertical:
            return TieredToolConfig(
                mandatory={"read", "ls"},
                vertical_core={"web", "fetch"},
                semantic_pool={"write", "edit", "grep", "search"},
                stage_tools={
                    "WRITING": {"write", "edit"},
                    "SEARCHING": {"web", "fetch", "grep"},
                },
                readonly_only_for_analysis=True,
            )

        Returns:
            TieredToolConfig or None (falls back to get_tools())
        """
        # Delegate to the canonical method
        return cls.get_tiered_tool_config()

    @classmethod
    def get_rl_config_provider(cls) -> Optional[Any]:
        """Get RL configuration provider for this vertical.

        Default implementation tries direct import pattern first, then falls back
        to extension factory. Override only if custom behavior needed.

        This eliminates ~25 LOC of duplicated wrapper code across verticals.

        Returns:
            RL config provider (RLConfigProviderProtocol) or None
        """
        try:
            # Try direct import pattern (e.g., CodingRLConfig)
            vertical_name = cls.__name__.replace("Assistant", "").replace("Vertical", "")
            class_name = f"{vertical_name}RLConfig"
            module = __import__(f"victor.{cls.name}.rl", fromlist=[class_name])
            return getattr(module, class_name)()
        except (ImportError, AttributeError):
            try:
                # Fall back to extension factory pattern
                return cls._get_extension_factory(
                    "rl_config_provider",
                    f"victor.{cls.name}.rl",
                    class_name,
                )
            except (ImportError, AttributeError):
                return None

    @classmethod
    def get_rl_hooks(cls) -> Optional[Any]:
        """Get RL hooks for outcome recording.

        Override to provide vertical-specific RL hooks for
        recording task outcomes and updating learners.

        Returns:
            RLHooks instance or None
        """
        return None

    @classmethod
    def get_team_spec_provider(cls) -> Optional[Any]:
        """Get team specification provider for this vertical.

        Default implementation tries direct import pattern first, then falls back
        to extension factory. Override only if custom behavior needed.

        This eliminates ~30 LOC of duplicated wrapper code across verticals.

        Returns:
            Team spec provider (TeamSpecProviderProtocol) or None
        """
        try:
            # Try direct import pattern (e.g., CodingTeamSpecProvider)
            vertical_name = cls.__name__.replace("Assistant", "").replace("Vertical", "")
            class_name = f"{vertical_name}TeamSpecProvider"
            module = __import__(f"victor.{cls.name}.teams", fromlist=[class_name])
            return getattr(module, class_name)()
        except (ImportError, AttributeError):
            try:
                # Fall back to extension factory pattern
                return cls._get_extension_factory(
                    "team_spec_provider",
                    f"victor.{cls.name}.teams",
                )
            except (ImportError, AttributeError):
                return None

    @classmethod
    def get_service_provider(cls) -> Optional[Any]:
        """Get service provider for this vertical.

        By default, returns a BaseVerticalServiceProvider that registers
        the vertical's prompt contributor, safety extension, mode config,
        and tool dependency providers with the DI container.

        Override to provide custom service registration logic.

        Returns:
            Service provider (ServiceProviderProtocol) or factory-created provider
        """
        try:
            from victor.core.verticals.base_service_provider import VerticalServiceProviderFactory

            return VerticalServiceProviderFactory.create(cls)
        except ImportError:
            return None

    @classmethod
    def get_enrichment_strategy(cls) -> Optional[Any]:
        """Get vertical-specific enrichment strategy.

        Override to provide vertical-specific prompt enrichment strategies
        for DSPy-like auto prompt optimization. Enrichments can include:
        - Knowledge graph symbols and related code snippets (coding)
        - Web search results and source citations (research)
        - Infrastructure context and command patterns (devops)
        - Schema context and query patterns (data analysis)

        Returns:
            EnrichmentStrategyProtocol implementation or None
        """
        return None

    @classmethod
    def get_extensions(
        cls,
        *,
        use_cache: bool = True,
        strict: Optional[bool] = None,
    ) -> "VerticalExtensions":
        """Get all extensions for this vertical with strict error handling.

        Aggregates all extension implementations for framework integration.
        Override for custom extension aggregation.

        LSP Compliance: This method ALWAYS returns a valid VerticalExtensions
        object, never None. Even on exceptions (in non-strict mode), it returns
        a VerticalExtensions with successfully loaded extensions.

        Error Handling Modes:
        - strict=True: Raises ExtensionLoadError on ANY extension failure
        - strict=False: Collects errors, logs warnings, returns partial extensions
        - strict=None: Uses class-level strict_extension_loading setting

        Required Extensions:
        Even when strict=False, extensions listed in required_extensions will
        raise ExtensionLoadError if they fail to load.

        Args:
            use_cache: If True (default), return cached extensions if available.
                       Set to False to force rebuild.
            strict: Override the class-level strict_extension_loading setting.
                    If None (default), uses cls.strict_extension_loading.

        Returns:
            VerticalExtensions containing all vertical extensions (never None)

        Raises:
            ExtensionLoadError: In strict mode or when a required extension fails
        """
        from victor.core.errors import ExtensionLoadError
        from victor.core.verticals.protocols import VerticalExtensions

        cache_key = cls._cache_namespace()

        # Return cached extensions if available and caching enabled
        if use_cache:
            with cls._extensions_cache_lock:
                cached = cls._extensions_cache.get(cache_key)
            if cached is not None:
                return cached

        # Determine strict mode
        is_strict = strict if strict is not None else cls.strict_extension_loading

        # Collect errors for reporting
        errors: List["ExtensionLoadError"] = []

        def _load_extension(
            extension_type: str,
            loader: callable,
            is_list: bool = False,
        ) -> Any:
            """Load an extension with error handling.

            Args:
                extension_type: Type name for error reporting
                loader: Callable that loads the extension
                is_list: If True, the extension should be a list

            Returns:
                The loaded extension, or default value on error
            """
            try:
                result = loader()
                return result
            except Exception as e:
                is_required = extension_type in cls.required_extensions
                error = ExtensionLoadError(
                    message=f"Failed to load '{extension_type}' extension for vertical '{cls.name}': {e}",
                    extension_type=extension_type,
                    vertical_name=cls.name,
                    original_error=e,
                    is_required=is_required,
                )
                errors.append(error)

                # Log the error with appropriate severity
                if is_strict or is_required:
                    logger.error(
                        f"[{error.correlation_id}] {extension_type} extension failed to load "
                        f"for vertical '{cls.name}': {e}",
                        exc_info=True,
                    )
                else:
                    logger.warning(
                        f"[{error.correlation_id}] {extension_type} extension failed to load "
                        f"for vertical '{cls.name}': {e}"
                    )

                # Return default value
                return [] if is_list else None

        # Load each extension with error handling
        middleware = _load_extension("middleware", cls.get_middleware, is_list=True)
        safety = _load_extension("safety", cls.get_safety_extension)
        prompt = _load_extension("prompt", cls.get_prompt_contributor)
        mode_config = _load_extension("mode_config", cls.get_mode_config_provider)
        tool_deps = _load_extension("tool_deps", cls.get_tool_dependency_provider)
        workflow = _load_extension("workflow", cls.get_workflow_provider)
        service = _load_extension("service", cls.get_service_provider)
        rl_config = _load_extension("rl_config", cls.get_rl_config_provider)
        team_spec = _load_extension("team_spec", cls.get_team_spec_provider)
        enrichment = _load_extension("enrichment", cls.get_enrichment_strategy)
        tiered_tools = _load_extension("tiered_tools", cls.get_tiered_tool_config)

        # Check for critical failures (strict mode or required extensions)
        critical_errors = [e for e in errors if is_strict or e.is_required]
        if critical_errors:
            # Raise the first critical error
            raise critical_errors[0]

        # Log summary if there were non-critical errors
        if errors:
            logger.warning(
                f"Vertical '{cls.name}' loaded with {len(errors)} extension error(s). "
                f"Affected extensions: {', '.join(e.extension_type for e in errors)}"
            )

        # Build extensions object
        extensions = VerticalExtensions(
            middleware=middleware if middleware else [],
            safety_extensions=[safety] if safety else [],
            prompt_contributors=[prompt] if prompt else [],
            mode_config_provider=mode_config,
            tool_dependency_provider=tool_deps,
            workflow_provider=workflow,
            service_provider=service,
            rl_config_provider=rl_config,
            team_spec_provider=team_spec,
            enrichment_strategy=enrichment,
            tiered_tool_config=tiered_tools,
        )

        # Cache the extensions
        with cls._extensions_cache_lock:
            cls._extensions_cache[cache_key] = extensions
        return extensions

    @classmethod
    async def get_extensions_async(
        cls,
        *,
        use_cache: bool = True,
        strict: Optional[bool] = None,
    ) -> "VerticalExtensions":
        """Async version of get_extensions that loads extensions in parallel.

        Uses a thread pool executor to load all extensions concurrently,
        providing faster initialization when extensions involve I/O.

        Shares the same cache as the synchronous get_extensions() method.

        Args:
            use_cache: If True (default), return cached extensions if available.
            strict: Override the class-level strict_extension_loading setting.

        Returns:
            VerticalExtensions containing all vertical extensions (never None)

        Raises:
            ExtensionLoadError: In strict mode or when a required extension fails
        """
        from victor.core.errors import ExtensionLoadError
        from victor.core.verticals.protocols import VerticalExtensions

        cache_key = cls._cache_namespace()

        # Return cached extensions if available (shared with sync get_extensions)
        if use_cache:
            with cls._extensions_cache_lock:
                cached = cls._extensions_cache.get(cache_key)
            if cached is not None:
                return cached

        # Determine strict mode
        is_strict = strict if strict is not None else cls.strict_extension_loading

        # Collect errors for reporting
        errors: List[ExtensionLoadError] = []
        errors_lock = threading.Lock()

        def _load_extension(
            extension_type: str,
            loader: callable,
            is_list: bool = False,
        ) -> Any:
            """Load an extension with error handling (runs in thread pool)."""
            try:
                return loader()
            except Exception as e:
                is_required = extension_type in cls.required_extensions
                error = ExtensionLoadError(
                    message=(
                        f"Failed to load '{extension_type}' extension "
                        f"for vertical '{cls.name}': {e}"
                    ),
                    extension_type=extension_type,
                    vertical_name=cls.name,
                    original_error=e,
                    is_required=is_required,
                )
                with errors_lock:
                    errors.append(error)

                if is_strict or is_required:
                    logger.error(
                        f"[{error.correlation_id}] {extension_type} extension failed "
                        f"for vertical '{cls.name}': {e}",
                        exc_info=True,
                    )
                else:
                    logger.warning(
                        f"[{error.correlation_id}] {extension_type} extension failed "
                        f"for vertical '{cls.name}': {e}"
                    )
                return [] if is_list else None

        # Load all extensions in parallel using a thread pool
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            futures = {
                "middleware": loop.run_in_executor(
                    pool, lambda: _load_extension("middleware", cls.get_middleware, True)
                ),
                "safety": loop.run_in_executor(
                    pool, lambda: _load_extension("safety", cls.get_safety_extension)
                ),
                "prompt": loop.run_in_executor(
                    pool, lambda: _load_extension("prompt", cls.get_prompt_contributor)
                ),
                "mode_config": loop.run_in_executor(
                    pool, lambda: _load_extension("mode_config", cls.get_mode_config_provider)
                ),
                "tool_deps": loop.run_in_executor(
                    pool,
                    lambda: _load_extension("tool_deps", cls.get_tool_dependency_provider),
                ),
                "workflow": loop.run_in_executor(
                    pool, lambda: _load_extension("workflow", cls.get_workflow_provider)
                ),
                "service": loop.run_in_executor(
                    pool, lambda: _load_extension("service", cls.get_service_provider)
                ),
                "rl_config": loop.run_in_executor(
                    pool, lambda: _load_extension("rl_config", cls.get_rl_config_provider)
                ),
                "team_spec": loop.run_in_executor(
                    pool, lambda: _load_extension("team_spec", cls.get_team_spec_provider)
                ),
                "enrichment": loop.run_in_executor(
                    pool, lambda: _load_extension("enrichment", cls.get_enrichment_strategy)
                ),
                "tiered_tools": loop.run_in_executor(
                    pool, lambda: _load_extension("tiered_tools", cls.get_tiered_tool_config)
                ),
            }

            # Await all futures
            results = {}
            for key, future in futures.items():
                results[key] = await future

        # Check for critical failures
        critical_errors = [e for e in errors if is_strict or e.is_required]
        if critical_errors:
            raise critical_errors[0]

        if errors:
            logger.warning(
                f"Vertical '{cls.name}' loaded with {len(errors)} extension error(s). "
                f"Affected extensions: {', '.join(e.extension_type for e in errors)}"
            )

        extensions = VerticalExtensions(
            middleware=results["middleware"] if results["middleware"] else [],
            safety_extensions=[results["safety"]] if results["safety"] else [],
            prompt_contributors=[results["prompt"]] if results["prompt"] else [],
            mode_config_provider=results["mode_config"],
            tool_dependency_provider=results["tool_deps"],
            workflow_provider=results["workflow"],
            service_provider=results["service"],
            rl_config_provider=results["rl_config"],
            team_spec_provider=results["team_spec"],
            enrichment_strategy=results["enrichment"],
            tiered_tool_config=results["tiered_tools"],
        )

        # Cache the extensions (shared with sync get_extensions)
        with cls._extensions_cache_lock:
            cls._extensions_cache[cache_key] = extensions
        return extensions

    @classmethod
    def get_workflow_provider(cls) -> Optional[Any]:
        """Get workflow provider for this vertical.

        Override to provide vertical-specific workflows.

        Returns:
            Workflow provider (WorkflowProviderProtocol) or None
        """
        return None

    @classmethod
    def clear_extension_cache(cls, *, clear_all: bool = False) -> None:
        """Clear the extension cache for this vertical.

        Args:
            clear_all: If True, clear cache for all verticals.
                       If False (default), clear only for this class.
        """
        if clear_all:
            with cls._extensions_cache_lock:
                cls._extensions_cache.clear()
        else:
            namespaced_key = cls._cache_namespace()
            namespaced_prefix = f"{namespaced_key}:"
            legacy_key = cls.__name__
            legacy_prefix = f"{legacy_key}:"
            with cls._extensions_cache_lock:
                # Clear namespaced cache entries
                cls._extensions_cache.pop(namespaced_key, None)
                namespaced_keys = [
                    k for k in cls._extensions_cache if k.startswith(namespaced_prefix)
                ]
                for key in namespaced_keys:
                    cls._extensions_cache.pop(key, None)

                # Backward compatibility: clear legacy class-name-only keys.
                cls._extensions_cache.pop(legacy_key, None)
                legacy_keys = [
                    k
                    for k in cls._extensions_cache
                    if k.startswith(legacy_prefix) and k.count(":") == 1
                ]
                for key in legacy_keys:
                    cls._extensions_cache.pop(key, None)
