"""Base class for vertical implementations.

This module defines the abstract base class that all verticals should inherit from.
It contains NO runtime logic - only abstract method definitions and default
implementations that raise NotImplementedError.
"""

from __future__ import annotations

import importlib
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, List, Optional

from victor_sdk.core.types import (
    VerticalConfig,
    VerticalDefinition,
    StageDefinition,
    Tier,
    ToolSet,
    normalize_capability_requirements,
    normalize_tool_requirements,
)
from victor_sdk.verticals.extensions import VerticalExtensions
from victor_sdk.core.api_version import CURRENT_API_VERSION
from victor_sdk.core.exceptions import VerticalConfigurationError
from victor_sdk.verticals.manifest import ExtensionManifest, ExtensionType
from victor_sdk.verticals.mixins import (
    ExtensionProviderMixin,
    PromptMetadataMixin,
    RLMixin,
    TeamMixin,
    WorkflowMetadataMixin,
)


# CONSOLIDATION: plugin-vertical unification — see memory plugin_vertical_consolidation.md
# External packages subclass VerticalBase and expose it via a VictorPlugin's
# register(context): context.register_vertical(MyVertical). "Plugin" is the
# canonical runtime integration seam; "Vertical" is the role the plugin
# provides. A future SDK minor release (S4) introduces PluginBase as an alias
# so new packages can standardize on plugin nomenclature.
class VerticalBase(
    RLMixin,
    TeamMixin,
    WorkflowMetadataMixin,
    PromptMetadataMixin,
    ExtensionProviderMixin,
    ABC,
):
    """Abstract base class for domain-specific assistants (verticals).

    This is the ONLY base class external verticals need to inherit from.
    Contains NO runtime logic - only abstract method definitions.

    External verticals can implement this class with ZERO runtime dependencies:
    ```python
    from victor_sdk.verticals.protocols.base import VerticalBase

    class MyVertical(VerticalBase):
        @classmethod
        def get_name(cls) -> str:
            return "my-vertical"

        @classmethod
        def get_description(cls) -> str:
            return "My custom vertical"

        @classmethod
        def get_tools(cls) -> List[str]:
            return ["read", "write", "search"]

        @classmethod
        def get_system_prompt(cls) -> str:
            return "You are a helpful assistant."
    ```

    The victor-ai framework provides a concrete implementation of this class
    that adds all runtime logic while maintaining backward compatibility.
    """

    # Class attributes that subclasses SHOULD override
    name: str
    description: str
    version: str = "1.0.0"
    _config_cache: ClassVar[Dict[str, VerticalConfig]] = {}
    _extension_cache: ClassVar[Dict[str, Any]] = {}
    _extension_cache_lock: ClassVar[threading.RLock] = threading.RLock()

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Return vertical identifier.

        This should be a unique, lowercase identifier with no spaces.
        Examples: "coding", "research", "devops"

        Returns:
            Vertical name/identifier
        """
        ...

    @classmethod
    @abstractmethod
    def get_description(cls) -> str:
        """Return human-readable description.

        This should be a short, one-line description of what this vertical does.

        Returns:
            Vertical description
        """
        ...

    @classmethod
    @abstractmethod
    def get_tools(cls) -> List[str]:
        """Return list of tool names for this vertical.

        The tool names should match the tool names registered in the framework.
        Common tool names include: "read", "write", "search", "shell", "git",
        "web_search", "database", "docker", etc.

        Returns:
            List of tool names
        """
        ...

    @classmethod
    @abstractmethod
    def get_system_prompt(cls) -> str:
        """Return system prompt text for this vertical.

        The system prompt defines the behavior and personality of the agent.
        It should be specific to the vertical's domain.

        Returns:
            System prompt text
        """
        ...

    @classmethod
    def get_config(cls, *, use_cache: bool = True) -> VerticalConfig:
        """Generate vertical configuration.

        This is a template method that assembles configuration from various
        subclass methods. Subclasses can override specific methods to customize
        the configuration without overriding this method.

        Note: In the SDK version, this creates a simple config. The victor-ai
        framework overrides this to provide full configuration with ToolSet objects.

        Returns:
            VerticalConfig object with all necessary configuration
        """
        cache_key = cls._extension_cache_key("config")
        with cls._extension_cache_lock:
            if use_cache and cache_key in cls._config_cache:
                return cls._config_cache[cache_key]

        config = cls.get_definition().to_config()
        with cls._extension_cache_lock:
            cls._config_cache[cache_key] = config
        return config

    @classmethod
    def _cache_namespace(cls) -> str:
        """Return a stable cache namespace for this vertical class."""

        return f"{cls.__module__}:{cls.__qualname__}"

    @classmethod
    def _extension_cache_key(cls, extension_key: str) -> str:
        """Return the composite cache key for an extension object."""

        return f"{cls._cache_namespace()}:{extension_key}"

    @classmethod
    def _get_cached_extension(cls, key: str, factory: Callable[[], Any]) -> Any:
        """Get an extension value from the SDK-local cache.

        This keeps lazy extension helpers available to SDK-pure verticals
        without pulling in the core runtime extension loader.
        """

        cache_key = cls._extension_cache_key(key)
        with cls._extension_cache_lock:
            if cache_key in cls._extension_cache:
                return cls._extension_cache[cache_key]

        value = factory()
        with cls._extension_cache_lock:
            cls._extension_cache[cache_key] = value
        return value

    @classmethod
    def _auto_extension_class_name(cls, extension_key: str) -> str:
        """Auto-generate a conventional class name for lazy imports."""

        prefix = cls.__name__
        for suffix in ("Assistant", "Vertical"):
            if prefix.endswith(suffix):
                prefix = prefix[: -len(suffix)]
                break
        suffix = "".join(part.capitalize() for part in extension_key.split("_"))
        return f"{prefix}{suffix}"

    @classmethod
    def _get_extension_instance(
        cls,
        extension_key: str,
        import_path: str,
        attribute_name: Optional[str] = None,
    ) -> Any:
        """Lazily import and instantiate an extension class (cached).

        Loads a module via importlib, finds a class by naming convention
        (e.g., extension_key="safety_extension" → CodingSafetyExtension),
        instantiates it, and caches the instance.

        Args:
            extension_key: Cache key and naming convention base
                (e.g., "safety_extension", "prompt_contributor")
            import_path: Dotted module path to import
                (e.g., "victor_coding.safety")
            attribute_name: Explicit class name. If None, derived from
                extension_key via _auto_extension_class_name().

        Returns:
            An instance of the loaded extension class.

        Note:
            Despite the former name "_get_extension_factory", this method
            returns an *instance*, not a factory callable.
        """

        def _create() -> Any:
            module = importlib.import_module(import_path)
            target_name = attribute_name or cls._auto_extension_class_name(
                extension_key
            )
            extension_cls = getattr(module, target_name)
            return extension_cls()

        return cls._get_cached_extension(extension_key, _create)

    # Backward-compat alias
    _get_extension_factory = _get_extension_instance

    @classmethod
    def clear_config_cache(cls, *, clear_all: bool = False) -> None:
        """Clear SDK-local extension caches.

        The SDK base does not cache config objects, but extracted verticals and
        tests rely on this hook existing to invalidate lazy extension helpers.
        """

        with cls._extension_cache_lock:
            if clear_all:
                cls._config_cache.clear()
                cls._extension_cache.clear()
                return

            prefix = f"{cls._cache_namespace()}:"
            stale_config_keys = [
                key for key in cls._config_cache if key.startswith(prefix)
            ]
            for key in stale_config_keys:
                cls._config_cache.pop(key, None)
            stale_keys = [key for key in cls._extension_cache if key.startswith(prefix)]
            for key in stale_keys:
                cls._extension_cache.pop(key, None)

    @classmethod
    def get_definition(cls) -> VerticalDefinition:
        """Return the serializable definition-layer contract for this vertical."""
        vertical_name = getattr(cls, "name", cls.__name__)

        try:
            vertical_name = cls.get_name()
            tool_requirements = normalize_tool_requirements(cls.get_tool_requirements())
            return VerticalDefinition(
                name=vertical_name,
                description=cls.get_description(),
                version=cls.get_version(),
                tools=[requirement.tool_name for requirement in tool_requirements],
                tool_requirements=tool_requirements,
                capability_requirements=normalize_capability_requirements(
                    cls.get_capability_requirements()
                ),
                system_prompt=cls.get_system_prompt(),
                prompt_metadata=cls.get_prompt_metadata(),
                stages=cls.get_stages(),
                team_metadata=cls.get_team_metadata(),
                workflow_metadata=cls.get_workflow_metadata(),
                tier=cls.get_tier(),
                metadata=cls.get_metadata(),
                skills=cls.get_skills(),
            )
        except VerticalConfigurationError as exc:
            if exc.vertical_name is not None:
                raise
            raise VerticalConfigurationError(
                exc.message,
                vertical_name=vertical_name,
                details=exc.details,
            ) from exc
        except Exception as exc:
            raise VerticalConfigurationError(
                "Invalid vertical definition generated from protocol hooks.",
                vertical_name=vertical_name,
                details={"error": str(exc)},
            ) from exc

    @classmethod
    def get_stages(cls) -> Dict[str, StageDefinition]:
        """Return stage definitions for multi-stage workflows.

        Default implementation provides basic 3-stage workflow.
        Subclasses can override to provide custom stages.

        Returns:
            Dictionary mapping stage names to StageDefinition objects
        """
        return {
            "planning": StageDefinition(
                name="planning",
                description="Plan the approach before execution",
                required_tools=[],
                optional_tools=["search", "read"],
            ),
            "execution": StageDefinition(
                name="execution",
                description="Execute the planned approach",
                required_tools=["read", "write"],
                optional_tools=["shell", "git"],
            ),
            "verification": StageDefinition(
                name="verification",
                description="Verify the results",
                required_tools=[],
                optional_tools=["test", "shell"],
            ),
        }

    # Methods from RLMixin, TeamMixin, ExtensionProviderMixin are inherited.
    # See victor_sdk.verticals.mixins for the implementations.

    @classmethod
    def register_tools(cls, registry: Any) -> None:
        """Register vertical-specific tools with the framework registry."""
        pass

    @classmethod
    def clear_extension_cache(cls, *, clear_all: bool = False) -> None:
        """Clear cached extension instances. Delegates to clear_config_cache."""
        cls.clear_config_cache(clear_all=clear_all)

    @classmethod
    def get_capability_configs(cls) -> Dict[str, Any]:
        """Return capability configurations for this vertical."""
        return {}

    # get_mode_config() inherited from ExtensionProviderMixin

    # -----------------------------------------------------------------
    # Extension container
    # -----------------------------------------------------------------

    @classmethod
    def get_extensions(
        cls,
        *,
        use_cache: bool = True,
        strict: Optional[bool] = None,
    ) -> VerticalExtensions:
        """Return a lazy extension container for this SDK vertical.

        The SDK keeps this method lightweight and best-effort. It aggregates
        only hooks already exposed by the vertical class and avoids importing
        core runtime extension infrastructure.
        """

        _ = strict

        def _build() -> VerticalExtensions:
            safety_extension = cls.get_safety_extension()
            prompt_contributor = cls.get_prompt_contributor()
            return VerticalExtensions(
                middleware=lambda: list(cls.get_middleware() or []),
                safety_extensions=lambda: (
                    [safety_extension] if safety_extension else []
                ),
                prompt_contributors=lambda: (
                    [prompt_contributor] if prompt_contributor else []
                ),
                mode_config_provider=cls.get_mode_config_provider,
                tool_dependency_provider=cls.get_tool_dependency_provider,
                workflow_provider=cls.get_workflow_provider,
                service_provider=cls.get_service_provider,
                rl_config_provider=cls.get_rl_config_provider,
                team_spec_provider=cls.get_team_spec_provider,
                enrichment_strategy=cls.get_enrichment_strategy,
                tool_selection_strategy=cls.get_tool_selection_strategy,
                tiered_tool_config=cls.get_tiered_tool_config,
            )

        if not use_cache:
            return _build()
        return cls._get_cached_extension("vertical_extensions", _build)

    @classmethod
    def get_tier(cls) -> Tier:
        """Return capability tier for this vertical.

        Returns:
            Capability tier (basic, standard, advanced)
        """
        return Tier.STANDARD

    @classmethod
    def get_version(cls) -> str:
        """Return the version for this vertical definition."""

        return getattr(cls, "version", "1.0.0")

    # Methods from PromptMetadataMixin, TeamMixin, WorkflowMetadataMixin
    # are inherited via mixin composition. See victor_sdk.verticals.mixins.

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Return additional metadata about this vertical.

        Subclasses can override to provide custom metadata.

        Returns:
            Dictionary of metadata key-value pairs
        """
        return {}

    @classmethod
    def get_manifest(cls) -> ExtensionManifest:
        """Return an ExtensionManifest describing this vertical's capabilities.

        The default implementation auto-builds the manifest by inspecting which
        protocol hooks the subclass overrides.  Verticals may override this to
        declare capabilities explicitly.

        Returns:
            ExtensionManifest with provides/requires derived from overridden methods.
        """
        provides: set[ExtensionType] = set()

        # Detect provided extension types from overridden methods
        _METHOD_TO_TYPE = {
            "get_middleware": ExtensionType.MIDDLEWARE,
            "get_safety_extension": ExtensionType.SAFETY,
            "get_tool_requirements": ExtensionType.TOOLS,
            "get_workflow_spec": ExtensionType.WORKFLOWS,
            "get_team_declarations": ExtensionType.TEAMS,
            "get_mode_config": ExtensionType.MODE_CONFIG,
            "get_rl_config": ExtensionType.RL_CONFIG,
            "get_enrichment_strategy": ExtensionType.ENRICHMENT,
            "get_capability_requirements": ExtensionType.CAPABILITIES,
        }

        for method_name, ext_type in _METHOD_TO_TYPE.items():
            method = getattr(cls, method_name, None)
            if method is None:
                continue
            # Check if the method is overridden from VerticalBase
            base_method = getattr(VerticalBase, method_name, None)
            if base_method is not None and method is not base_method:
                provides.add(ext_type)

        # Tools are always provided if get_tools is implemented (it's abstract)
        provides.add(ExtensionType.TOOLS)

        return ExtensionManifest(
            api_version=CURRENT_API_VERSION,
            name=cls.get_name(),
            version=cls.get_version(),
            provides=provides,
        )

    @classmethod
    def get_skills(cls) -> List[Any]:
        """Return skill definitions for this vertical.

        Subclasses override this to declare composable skills. Each skill
        is a SkillDefinition that binds a prompt fragment with a tool subset.

        Returns:
            List of SkillDefinition objects (empty by default)
        """
        return []

    @classmethod
    def _get_toolset(cls) -> ToolSet:
        """Convert tool names to ToolSet (implementation in core).

        This method is implemented in victor-ai to provide the actual
        ToolSet object with metadata.

        Raises:
            NotImplementedError: Always raised in SDK (implemented in core)
        """
        raise NotImplementedError(
            "_get_toolset is implemented in victor-ai framework. "
            "Use get_tools() to get tool names as a list."
        )
