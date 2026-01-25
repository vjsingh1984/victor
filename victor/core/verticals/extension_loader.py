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

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, List, Optional, Set, Type, cast

if TYPE_CHECKING:
    from victor.core.verticals.protocols import VerticalExtensions
    from victor.core.vertical_types import TieredToolConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Cache Entry with TTL and Version Tracking
# =============================================================================


@dataclass
class ExtensionCacheEntry:
    """Cache entry with TTL and version tracking.

    Attributes:
        value: The cached extension instance
        timestamp: When the entry was created
        ttl: Time-to-live in seconds (None = no expiration)
        version: Version string for cache invalidation
        access_count: Number of times accessed
    """

    value: Any
    timestamp: float = field(default_factory=time.time)
    ttl: Optional[int] = None  # Default TTL from settings
    version: str = ""
    access_count: int = 0

    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        if not self.ttl:
            return False
        return (time.time() - self.timestamp) > self.ttl

    def get_age(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.timestamp


# =============================================================================
# Extension Loader Base Class
# =============================================================================


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

    Auto-Generated Extension Getters (Phase 2):
        This class uses __init_subclass__ to automatically generate
        extension getter methods, eliminating ~800 lines of duplication
        across verticals.

        Supported patterns:
        - get_middleware() -> _get_cached_extension("middleware", factory)
        - get_safety_extension() -> _get_extension_factory("safety_extension", import_path)
        - get_prompt_contributor() -> _get_extension_factory("prompt_contributor", import_path)
        - get_mode_config_provider() -> _get_extension_factory("mode_config_provider", import_path)
        - get_tool_dependency_provider() -> _get_extension_factory("tool_dependency_provider", import_path)
        - get_workflow_provider() -> _get_extension_factory("workflow_provider", import_path)
        - get_service_provider() -> _get_extension_factory("service_provider", import_path)
        - get_rl_config_provider() -> _get_extension_factory("rl_config_provider", import_path)
        - get_rl_hooks() -> _get_extension_factory("rl_hooks", import_path)
        - get_team_spec_provider() -> _get_extension_factory("team_spec_provider", import_path)
        - get_capability_provider() -> _get_extension_factory("capability_provider", import_path)

        The __init_subclass__ hook automatically adds these methods to subclasses
        unless they're already defined. Verticals can override any getter
        for custom behavior (e.g., middleware with custom initialization).

        Benefits:
        - Eliminates ~800 lines of duplication (50 methods × 5 verticals × ~16 lines each)
        - Maintains backward compatibility - all existing getter calls still work
        - Verticals can override any getter for custom logic
        - New extension types automatically supported by adding to _extension_patterns

    Attributes:
        name: Vertical name (provided by VerticalMetadataProvider in subclasses)
    """

    # Vertical name (provided by VerticalMetadataProvider in subclasses)
    name: ClassVar[str] = ""

    # Extension loading configuration
    # When True, any extension loading failure raises ExtensionLoadError
    strict_extension_loading: ClassVar[bool] = False

    # Extensions that must load successfully even when strict_extension_loading=False
    # Valid values: "middleware", "safety", "prompt", "mode_config", "tool_deps",
    #               "workflow", "service", "rl_config", "team_spec", "enrichment",
    #               "tiered_tools"
    required_extensions: ClassVar[Set[str]] = set()

    # Default TTL for extension cache entries (in seconds)
    # None = no expiration, 3600 = 1 hour default
    default_extension_ttl: ClassVar[Optional[int]] = 3600

    # Extension cache (shared across all verticals)
    # Now uses ExtensionCacheEntry for TTL and version tracking
    _extensions_cache: Dict[str, ExtensionCacheEntry] = {}

    # Version tracker for cache invalidation
    _extension_versions: Dict[str, str] = {}

    # Mapping of extension types to their import path patterns
    # Format: extension_key -> (module_path_suffix, class_name_suffix)
    _extension_patterns: ClassVar[Dict[str, tuple[str, str]]] = {
        "safety_extension": ("safety", "SafetyExtension"),
        "prompt_contributor": ("prompts", "PromptContributor"),
        "mode_config_provider": ("mode_config", "ModeConfigProvider"),
        "tool_dependency_provider": ("tool_dependencies", "ToolDependencyProvider"),
        "workflow_provider": ("workflows", "WorkflowProvider"),
        "service_provider": ("service_provider", "ServiceProvider"),
        "rl_config_provider": ("rl", "RLConfig"),
        "rl_hooks": ("rl", "RLHooks"),
        "team_spec_provider": ("teams", "TeamSpecProvider"),
        "capability_provider": ("capabilities", "CapabilityProvider"),
        "enrichment_strategy": ("enrichment", "EnrichmentStrategy"),
    }

    # Extensions that use _get_cached_extension instead of _get_extension_factory
    _cached_extensions: ClassVar[Set[str]] = {
        "middleware",
        "composed_chains",
        "personas",
    }

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-generate extension getter methods for subclasses.

        This hook is called when a subclass of VerticalExtensionLoader is created.
        It automatically adds getter methods for common extension types unless
        the subclass already defines them.
        """
        super().__init_subclass__(**kwargs)

        # Auto-generate getters for each extension pattern
        for extension_type, (module_suffix, class_suffix) in cls._extension_patterns.items():
            method_name = f"get_{extension_type}"

            # Only add if not already defined in the class
            if method_name not in cls.__dict__:

                def _make_getter(ext_type: str, mod_suffix: str, cls_suffix: str) -> classmethod[type[Any]]:
                    """Factory to create getter methods with proper closure."""

                    def _getter(subcls: type) -> Any:
                        """Auto-generated getter method."""
                        # Build import path from vertical name
                        from victor.core.verticals.naming import get_vertical_module_name

                        vertical_name = get_vertical_module_name(subcls.name)

                        # Skip if vertical name is empty (abstract base class)
                        if not vertical_name:
                            return None

                        import_path = f"victor.{vertical_name}.{mod_suffix}"

                        # Auto-generate class name
                        vertical_class_name = subcls.__name__.replace("Assistant", "")
                        class_name = f"{vertical_class_name}{cls_suffix}"

                        # Use _get_extension_factory to load the extension
                        return cast(type[VerticalExtensionLoader], subcls)._get_extension_factory(ext_type, import_path, class_name)

                    return classmethod(_getter)

                # Add the method to the class
                setattr(cls, method_name, _make_getter(extension_type, module_suffix, class_suffix))

        # Auto-generate getters for cached extensions
        for extension_type in cls._cached_extensions:
            method_name = f"get_{extension_type}"

            # Only add if not already defined in the class
            if method_name not in cls.__dict__:

                def _make_cached_getter(ext_type: str) -> classmethod[type[Any]]:
                    """Factory to create cached getter methods with proper closure."""

                    def _getter(subcls: type) -> Any:
                        """Auto-generated cached getter method."""
                        # Skip if vertical name is empty (abstract base class)
                        from victor.core.verticals.naming import get_vertical_module_name

                        vertical_name = get_vertical_module_name(subcls.name)
                        if not vertical_name:
                            # Return default value for abstract base class
                            return [] if ext_type == "middleware" else None

                        # Special handling for middleware
                        if ext_type == "middleware":

                            def _create_middleware() -> list[Any]:
                                # Try to import from vertical.middleware
                                try:
                                    module = __import__(
                                        f"victor.{vertical_name}.middleware", fromlist=[""]
                                    )

                                    # Look for common middleware classes
                                    middleware_classes = []
                                    for attr_name in dir(module):
                                        attr = getattr(module, attr_name)
                                        if isinstance(attr, type) and "Middleware" in attr_name:
                                            # Try to instantiate with default args
                                            try:
                                                middleware_classes.append(attr())
                                            except Exception:
                                                pass

                                    return middleware_classes
                                except Exception:
                                    # Return empty list on any error
                                    return []

                            return cast(type[VerticalExtensionLoader], subcls)._get_cached_extension(ext_type, _create_middleware)

                        # Special handling for composed_chains and personas
                        if ext_type in ("composed_chains", "personas"):

                            def _create_extension() -> dict[str, Any]:
                                try:
                                    if ext_type == "composed_chains":
                                        module = __import__(
                                            f"victor.{vertical_name}.composed_chains", fromlist=[""]
                                        )
                                        return getattr(module, "COMPOSED_CHAINS", {})
                                    else:  # personas
                                        module = __import__(
                                            f"victor.{vertical_name}.teams", fromlist=[""]
                                        )
                                        return getattr(module, "PERSONAS", {})
                                except Exception:
                                    return {}

                            return cast(type[VerticalExtensionLoader], subcls)._get_cached_extension(ext_type, _create_extension)

                        # Default: return None
                        return None

                    return classmethod(_getter)

                # Add the method to the class
                setattr(cls, method_name, _make_cached_getter(extension_type))

    # =========================================================================
    # Extension Caching Infrastructure
    # =========================================================================

    @classmethod
    def _get_cached_extension(cls, key: str, factory: Callable[[], Any], ttl: Optional[int] = None) -> Any:
        """Get extension from cache or create and cache it with TTL support.

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
            ttl: Time-to-live in seconds (None = use default_extension_ttl)

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
        # Use composite key to avoid collisions between different vertical classes
        cache_key = f"{cls.__name__}:{key}"

        # Get version for this key if available
        version = cls._extension_versions.get(cache_key, "")

        # Check cache with TTL validation
        if cache_key in cls._extensions_cache:
            entry = cls._extensions_cache[cache_key]

            # Check if entry is expired
            if entry.is_expired():
                logger.debug(
                    f"Extension cache entry expired for {cache_key} "
                    f"(age={entry.get_age():.1f}s, ttl={entry.ttl})"
                )
                del cls._extensions_cache[cache_key]
            # Check if version has changed
            elif version and entry.version != version:
                logger.debug(
                    f"Extension cache version mismatch for {cache_key} "
                    f"(cached={entry.version}, current={version})"
                )
                del cls._extensions_cache[cache_key]
            # Cache hit - update access count and return
            else:
                entry.access_count += 1
                logger.debug(
                    f"Extension cache hit for {cache_key} "
                    f"(accesses={entry.access_count}, age={entry.get_age():.1f}s)"
                )
                return entry.value

        # Cache miss - create new entry
        logger.debug(f"Extension cache miss for {cache_key} - creating new instance")
        value = factory()

        # Use provided TTL or default
        if ttl is None:
            ttl = cls.default_extension_ttl

        # Create and cache entry
        entry = ExtensionCacheEntry(
            value=value,
            ttl=ttl,
            version=version,
        )
        cls._extensions_cache[cache_key] = entry

        return value

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

        def _create() -> Any:
            # Skip loading if import_path is invalid (empty vertical name)
            # This handles the case where VerticalBase has name="" (abstract base class)
            # which would create invalid paths like "victor..safety" (double dots)
            if not import_path or ".." in import_path:
                # Invalid import path - return None to indicate no extension available
                return None

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
            # Return None if module doesn't exist (graceful degradation)
            try:
                module = __import__(import_path, fromlist=[class_name])
                # Import and instantiate
                return getattr(module, class_name)()
            except (ModuleNotFoundError, AttributeError):
                # Module or class doesn't exist - return None
                return None

        # Use existing caching infrastructure
        return cls._get_cached_extension(extension_key, _create)

    # =========================================================================
    # Extension Protocol Methods (Optional)
    # =========================================================================
    # These methods enable verticals to provide framework extensions.
    # Override them to integrate with the framework's middleware, safety,
    # prompt, and configuration systems.

    # NOTE: Most extension getters are now auto-generated by VerticalExtensionLoaderMeta
    # to eliminate duplication. Only override for custom behavior.
    #
    # Auto-generated getters (via metaclass __getattr__):
    # - get_safety_extension()
    # - get_prompt_contributor()
    # - get_mode_config_provider()
    # - get_tool_dependency_provider()
    # - get_workflow_provider()
    # - get_service_provider()
    # - get_rl_config_provider()
    # - get_rl_hooks()
    # - get_team_spec_provider()
    # - get_capability_provider()
    # - get_enrichment_strategy()
    # - get_middleware() (if not overridden)
    #
    # Default implementations below are only for methods NOT auto-generated:
    # - get_mode_config()
    # - get_task_type_hints()
    # - get_tool_graph()
    # - get_tiered_tool_config()
    # - get_tiered_tools()

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

        Override to provide vertical-specific tool execution patterns.

        Returns:
            Tool dependency provider (ToolDependencyProviderProtocol) or None
        """
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

        Override to provide vertical-specific RL learner configurations,
        task type mappings, and quality thresholds.

        Returns:
            RL config provider (RLConfigProviderProtocol) or None
        """
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

        Override to provide vertical-specific multi-agent team
        configurations for complex task execution.

        Returns:
            Team spec provider (TeamSpecProviderProtocol) or None
        """
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
        use_lazy: Optional[bool] = None,
    ) -> "VerticalExtensions":
        """Get all extensions for this vertical with strict error handling.

        Aggregates all extension implementations for framework integration.
        Override for custom extension aggregation.

        Lazy Loading (NEW):
        When lazy loading is enabled (default), this returns a LazyVerticalExtensions
        wrapper that defers actual extension loading until first access. This
        significantly improves startup time by avoiding unnecessary imports.

        Lazy loading is controlled by:
        1. VICTOR_LAZY_EXTENSIONS environment variable (true/false/auto)
        2. use_lazy parameter (overrides environment setting)
        3. Defaults to ON_DEMAND (lazy) for better startup performance

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
            use_lazy: Override lazy loading setting. If None (default), uses
                     VICTOR_LAZY_EXTENSIONS environment variable.

        Returns:
            VerticalExtensions containing all vertical extensions (never None)
            May be a LazyVerticalExtensions wrapper if lazy loading enabled

        Raises:
            ExtensionLoadError: In strict mode or when a required extension fails
        """
        from victor.core.errors import ExtensionLoadError
        from victor.core.verticals.protocols import VerticalExtensions

        # Determine if we should use lazy loading
        if use_lazy is None:
            from victor.core.verticals.lazy_extensions import get_extension_load_trigger

            trigger = get_extension_load_trigger()
            use_lazy = trigger != "eager"

        # If lazy loading enabled, return a lazy wrapper
        if use_lazy:
            from victor.core.verticals.lazy_extensions import (
                create_lazy_extensions,
                ExtensionLoadTrigger,
            )

            trigger = (
                get_extension_load_trigger()
                if use_lazy is None
                else (ExtensionLoadTrigger.ON_DEMAND if use_lazy else ExtensionLoadTrigger.EAGER)
            )

            # Create lazy wrapper that defers loading
            return create_lazy_extensions(
                vertical_name=cls.name,
                loader=lambda: cls._load_extensions_eager(use_cache=use_cache, strict=strict),
                trigger=trigger,
            )

        # Eager loading path (legacy behavior)
        return cls._load_extensions_eager(use_cache=use_cache, strict=strict)

    @classmethod
    def _load_extensions_eager(
        cls,
        *,
        use_cache: bool = True,
        strict: Optional[bool] = None,
    ) -> "VerticalExtensions":
        """Eagerly load all extensions for this vertical.

        This is the internal implementation that actually loads extensions.
        It's called by get_extensions() when lazy loading is disabled or when
        the lazy wrapper triggers loading.

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

        cache_key = cls.__name__

        # Return cached extensions if available and caching enabled
        if use_cache and cache_key in cls._extensions_cache:
            # Handle both ExtensionCacheEntry (new) and raw VerticalExtensions (old for compatibility)
            cached = cls._extensions_cache[cache_key]
            if isinstance(cached, ExtensionCacheEntry):
                result: Any = cached.value
            else:
                result = cached
            # Type: ignore because we're handling cached values that may be Any
            return cast("VerticalExtensions", result)

        # Determine strict mode
        is_strict = strict if strict is not None else cls.strict_extension_loading

        # Collect errors for reporting
        errors: List["ExtensionLoadError"] = []

        def _load_extension(
            extension_type: str,
            loader: Callable[[], Any],
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
                vertical_name = cls.name
                error = ExtensionLoadError(
                    message=f"Failed to load '{extension_type}' extension for vertical '{vertical_name}': {e}",
                    extension_type=extension_type,
                    vertical_name=vertical_name,
                    original_error=e,
                    is_required=is_required,
                )
                errors.append(error)

                # Log the error with appropriate severity
                if is_strict or is_required:
                    logger.error(
                        f"[{error.correlation_id}] {extension_type} extension failed to load "
                        f"for vertical '{vertical_name}': {e}",
                        exc_info=True,
                    )
                else:
                    logger.warning(
                        f"[{error.correlation_id}] {extension_type} extension failed to load "
                        f"for vertical '{vertical_name}': {e}"
                    )

                # Return default value
                return [] if is_list else None

        # Load each extension with error handling
        middleware = _load_extension("middleware", cls.get_middleware if hasattr(cls, "get_middleware") else (lambda: []), is_list=True)
        safety = _load_extension("safety", cls.get_safety_extension if hasattr(cls, "get_safety_extension") else (lambda: None))
        prompt = _load_extension("prompt", cls.get_prompt_contributor if hasattr(cls, "get_prompt_contributor") else (lambda: None))
        mode_config = _load_extension("mode_config", cls.get_mode_config_provider if hasattr(cls, "get_mode_config_provider") else (lambda: None))
        tool_deps = _load_extension("tool_deps", cls.get_tool_dependency_provider)
        workflow = _load_extension("workflow", cls.get_workflow_provider)
        service = _load_extension("service", cls.get_service_provider)
        rl_config = _load_extension("rl_config", cls.get_rl_config_provider)
        team_spec = _load_extension("team_spec", cls.get_team_spec_provider)
        enrichment = _load_extension("enrichment", cls.get_enrichment_strategy)
        tiered_tools = _load_extension("tiered_tools", cls.get_tiered_tool_config)

        # Load dynamic extensions from registry (OCP-compliant)
        # This enables third-party extensions without modifying core code
        dynamic_extensions: Dict[str, List[Any]] = {}
        try:
            # Check if this class has ExtensionRegistry integration
            if hasattr(cls, "_get_extension_registry"):
                registry = cls._get_extension_registry()
                extension_types = registry.list_extension_types()

                # Retrieve all dynamic extensions by type
                for ext_type in extension_types:
                    extensions = registry.get_extensions_by_type(ext_type)
                    if extensions:
                        dynamic_extensions[ext_type] = extensions

                logger.debug(
                    f"Loaded {len(dynamic_extensions)} dynamic extension type(s) "
                    f"for vertical '{cls.name}'"
                )
        except Exception as e:
            # Dynamic extensions are optional - don't fail if registry not available
            logger.debug(f"Could not load dynamic extensions for vertical '{cls.name}': {e}")

        # Check for critical failures (strict mode or required extensions)
        critical_errors = [e for e in errors if is_strict or e.is_required]
        if critical_errors:
            # Raise the first critical error
            raise critical_errors[0]

        # Log summary if there were non-critical errors
        if errors:
            vertical_name = cls.name
            logger.warning(
                f"Vertical '{vertical_name}' loaded with {len(errors)} extension error(s). "
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
            _dynamic_extensions=dynamic_extensions,
        )

        # Cache the extensions using ExtensionCacheEntry
        cache_entry = ExtensionCacheEntry(
            value=extensions,
            ttl=cls.default_extension_ttl,
            version="",
        )
        cls._extensions_cache[cache_key] = cache_entry
        return extensions

    @classmethod
    def get_workflow_provider(cls) -> Optional[Any]:
        """Get workflow provider for this vertical.

        Override to provide vertical-specific workflows.

        Returns:
            Workflow provider (WorkflowProviderProtocol) or None
        """
        return None

    # =========================================================================
    # Enhanced Cache Invalidation with Version Tracking
    # =========================================================================

    @classmethod
    def invalidate_extension_cache(
        cls,
        extension_key: Optional[str] = None,
        version: Optional[str] = None,
    ) -> int:
        """Invalidate cache entries by key or version.

        Args:
            extension_key: Specific extension key to invalidate (e.g., "middleware",
                          "safety_extension"). If None, checks version instead.
            version: Version to invalidate. All cache entries with this version
                     will be invalidated. If None and extension_key is None,
                     clears all for this vertical.

        Returns:
            Number of cache entries invalidated.

        Examples:
            # Invalidate specific extension
            cls.invalidate_extension_cache(extension_key="middleware")

            # Invalidate all entries with a specific version
            cls.invalidate_extension_cache(version="1.2.3")

            # Invalidate all for this vertical
            cls.invalidate_extension_cache()
        """
        cache_prefix = f"{cls.__name__}:"
        count = 0

        if extension_key:
            # Invalidate specific extension
            cache_key = f"{cache_prefix}{extension_key}"
            if cache_key in cls._extensions_cache:
                del cls._extensions_cache[cache_key]
                count += 1
                logger.debug(f"Invalidated extension cache: {cache_key}")
        elif version:
            # Invalidate all entries with this version
            keys_to_remove = [
                k
                for k, v in cls._extensions_cache.items()
                if k.startswith(cache_prefix) and v.version == version
            ]
            for key in keys_to_remove:
                del cls._extensions_cache[key]
                count += 1
            logger.debug(
                f"Invalidated {count} extension(s) for version {version} "
                f"in vertical {cls.__name__}"
            )
        else:
            # Clear all for this vertical
            keys_to_remove = [k for k in cls._extensions_cache if k.startswith(cache_prefix)]
            for key in keys_to_remove:
                del cls._extensions_cache[key]
                count = len(keys_to_remove)
            logger.debug(f"Cleared {count} extension(s) for vertical {cls.__name__}")

        return count

    @classmethod
    def update_extension_version(cls, extension_key: str, version: str) -> None:
        """Update version for an extension to trigger cache invalidation.

        When an extension's implementation changes, call this method to update
        the version. Subsequent cache accesses will invalidate old cached entries.

        Args:
            extension_key: Extension key (e.g., "middleware", "safety_extension")
            version: New version string

        Example:
            cls.update_extension_version("middleware", "2.0.0")
        """
        cache_key = f"{cls.__name__}:{extension_key}"
        cls._extension_versions[cache_key] = version
        logger.debug(f"Updated extension version: {cache_key} -> {version}")

    @classmethod
    def get_extension_cache_stats(cls, detailed: bool = False) -> Dict[str, Any]:
        """Get cache statistics for this vertical.

        Args:
            detailed: If True, include per-entry breakdown and additional metrics

        Returns:
            Dictionary with cache statistics including:
            - total_entries: Number of cached extensions
            - expired_entries: Number of expired entries
            - total_accesses: Sum of access counts
            - avg_age: Average age of cache entries
            - min_age: Age of newest entry (seconds)
            - max_age: Age of oldest entry (seconds)
            - cache_hit_rate: Ratio of accesses to entries (effectiveness metric)
            - ttl_remaining: Average TTL remaining across entries
            - entries: (detailed=True) Per-entry breakdown
        """
        cache_prefix = f"{cls.__name__}:"
        entries = [(k, v) for k, v in cls._extensions_cache.items() if k.startswith(cache_prefix)]

        if not entries:
            return {
                "vertical": cls.__name__,
                "total_entries": 0,
                "expired_entries": 0,
                "total_accesses": 0,
                "avg_age": 0.0,
                "min_age": 0.0,
                "max_age": 0.0,
                "cache_hit_rate": 0.0,
                "ttl_remaining": 0.0,
            }

        expired_count = sum(1 for _, v in entries if v.is_expired())
        total_accesses = sum(v.access_count for _, v in entries)
        ages = [v.get_age() for _, v in entries]
        avg_age = sum(ages) / len(entries)

        # Calculate cache hit rate (accesses per entry = effectiveness)
        # Higher is better - means entries are being reused
        cache_hit_rate = total_accesses / len(entries) if entries else 0.0

        # Calculate min/max ages
        min_age = min(ages) if ages else 0.0
        max_age = max(ages) if ages else 0.0

        # Calculate average TTL remaining (for entries with TTL)
        ttl_entries = [v for _, v in entries if v.ttl is not None]
        ttl_remaining = 0.0
        if ttl_entries:
            remaining = [max(0, (v.ttl or 0) - v.get_age()) for v in ttl_entries]
            ttl_remaining = sum(remaining) / len(remaining)

        stats = {
            "vertical": cls.__name__,
            "total_entries": len(entries),
            "expired_entries": expired_count,
            "total_accesses": total_accesses,
            "avg_age": avg_age,
            "min_age": min_age,
            "max_age": max_age,
            "cache_hit_rate": cache_hit_rate,
            "ttl_remaining": ttl_remaining,
        }

        if detailed:
            # Add per-entry breakdown
            stats["entries"] = [
                {
                    "key": k.split(":")[-1],  # Remove vertical prefix
                    "age": v.get_age(),
                    "ttl": v.ttl,
                    "access_count": v.access_count,
                    "version": v.version,
                    "is_expired": v.is_expired(),
                }
                for k, v in entries
            ]

        return stats

    @classmethod
    def get_global_cache_stats(cls) -> Dict[str, Any]:
        """Get global cache statistics across all verticals.

        Returns:
            Dictionary with global statistics including:
            - total_entries: Total cached extensions across all verticals
            - total_verticals: Number of verticals with cached extensions
            - total_accesses: Sum of all access counts
            - verticals: Per-vertical statistics
        """
        # Group entries by vertical
        vertical_stats: Dict[str, Dict[str, Any]] = {}
        for key, entry in cls._extensions_cache.items():
            # Extract vertical name from composite key "ClassName:extension_key"
            parts = key.split(":", 1)
            vertical_name = parts[0] if parts else "unknown"

            if vertical_name not in vertical_stats:
                vertical_stats[vertical_name] = {
                    "entries": 0,
                    "accesses": 0,
                    "expired": 0,
                }

            vertical_stats[vertical_name]["entries"] += 1
            vertical_stats[vertical_name]["accesses"] += entry.access_count
            if entry.is_expired():
                vertical_stats[vertical_name]["expired"] += 1

        total_entries = sum(s["entries"] for s in vertical_stats.values())
        total_accesses = sum(s["accesses"] for s in vertical_stats.values())

        return {
            "total_entries": total_entries,
            "total_verticals": len(vertical_stats),
            "total_accesses": total_accesses,
            "verticals": vertical_stats,
        }

    @classmethod
    def clear_extension_cache(cls, *, clear_all: bool = False) -> None:
        """Clear the extension cache for this vertical.

        Args:
            clear_all: If True, clear cache for all verticals.
                       If False (default), clear only for this class.
        """
        if clear_all:
            cls._extensions_cache.clear()
            cls._extension_versions.clear()
        else:
            cache_key = cls.__name__
            # Clear composite extensions cache entry
            cls._extensions_cache.pop(cache_key, None)
            # Also clear individual extension cache entries (format: "ClassName:key")
            prefix = f"{cache_key}:"
            keys_to_remove = [k for k in cls._extensions_cache if k.startswith(prefix)]
            for key in keys_to_remove:
                cls._extensions_cache.pop(key, None)
            # Also clear version entries
            version_keys = [k for k in cls._extension_versions if k.startswith(prefix)]
            for key in version_keys:
                cls._extension_versions.pop(key, None)
