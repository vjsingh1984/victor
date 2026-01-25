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

"""Core vertical base classes and utilities.

This module provides the base infrastructure for Victor verticals:
- VerticalBase: Abstract base class for domain-specific assistants
- VerticalConfig: Configuration dataclass
- VerticalRegistry: Registry for vertical discovery
- StageDefinition: Stage definitions for conversation flow

Phase 2.3 SRP Compliance:
VerticalBase now composes functionality from focused classes:
- VerticalMetadataProvider: Metadata capabilities
- VerticalExtensionLoader: Extension loading and caching
- VerticalWorkflowProvider: Workflow and handler providers

Verticals (coding, rag, devops, research, dataanalysis) are separate
modules at the same level as core in the victor namespace.
"""

from victor.core.vertical_types import StageDefinition
from victor.core.verticals.base import (
    VerticalBase,
    VerticalConfig,
    VerticalRegistry,
)
from victor.core.verticals.metadata import (
    VerticalMetadataProvider,
    VerticalMetadata,
)
from victor.core.verticals.extension_loader import (
    VerticalExtensionLoader,
)
from victor.core.verticals.workflow_provider import (
    VerticalWorkflowProvider,
    VerticalWorkflowMixin,
)
from victor.core.verticals.protocols import (
    VerticalExtensions,
    ModeConfigProviderProtocol,
    PromptContributorProtocol,
    SafetyExtensionProtocol,
    ToolDependencyProviderProtocol,
    WorkflowProviderProtocol,
    RLConfigProviderProtocol,
    TeamSpecProviderProtocol,
    # ISP-Compliant Provider Protocols
    MiddlewareProvider,
    SafetyProvider,
    WorkflowProvider,
    TeamProvider,
    RLProvider,
    EnrichmentProvider,
    ToolProvider,
    HandlerProvider,
    CapabilityProvider,
    ModeConfigProvider,
    PromptContributorProvider,
    ToolDependencyProvider,
    TieredToolConfigProvider,
    ServiceProvider,
)
from victor.core.verticals.base_service_provider import (
    BaseVerticalServiceProvider,
    VerticalServiceProviderFactory,
)
from victor.core.verticals.capability_provider import (
    ICapabilityProvider,
    IConfigurableCapability,
    BaseCapabilityProvider,
    CapabilityProviderRegistry,
    get_capability_registry,
    reset_global_registry,
)
from victor.core.verticals.capability_injector import (
    CapabilityInjector,
    get_capability_injector,
    create_capability_injector,
)
from victor.core.verticals.capability_migration import (
    deprecated_direct_instantiation,
    migrate_capability_property,
    migrate_to_injector,
    get_capability_or_create,
    check_migration_status,
    print_migration_report,
)
from victor.core.verticals.vertical_loader import (
    VerticalLoader,
    get_vertical_loader,
    load_vertical,
    get_active_vertical,
    get_vertical_extensions,
)
from victor.core.verticals.lazy_extensions import (
    ExtensionLoadTrigger,
    LazyVerticalExtensions,
    create_lazy_extensions,
    get_extension_load_trigger,
)

# Layer boundary enforcement: moved from victor.agent.vertical_context
from victor.core.verticals.context import (
    VerticalContext,
    VerticalContextProtocol,
    MutableVerticalContextProtocol,
    create_vertical_context,
)

__all__ = [
    # Base classes
    "VerticalBase",
    "VerticalConfig",
    "VerticalRegistry",
    "StageDefinition",
    # Phase 2.3 SRP Compliance - Focused capability providers
    "VerticalMetadataProvider",
    "VerticalMetadata",
    "VerticalExtensionLoader",
    "VerticalWorkflowProvider",
    "VerticalWorkflowMixin",
    # Protocols
    "VerticalExtensions",
    "ModeConfigProviderProtocol",
    "PromptContributorProtocol",
    "SafetyExtensionProtocol",
    "ToolDependencyProviderProtocol",
    "WorkflowProviderProtocol",
    "RLConfigProviderProtocol",
    "TeamSpecProviderProtocol",
    # ISP-Compliant Provider Protocols
    "MiddlewareProvider",
    "SafetyProvider",
    "WorkflowProvider",
    "TeamProvider",
    "RLProvider",
    "EnrichmentProvider",
    "ToolProvider",
    "HandlerProvider",
    "CapabilityProvider",
    "ModeConfigProvider",
    "PromptContributorProvider",
    "ToolDependencyProvider",
    "TieredToolConfigProvider",
    "ServiceProvider",
    # Service providers
    "BaseVerticalServiceProvider",
    "VerticalServiceProviderFactory",
    # Capability Injection (DI-based)
    "ICapabilityProvider",
    "IConfigurableCapability",
    "BaseCapabilityProvider",
    "CapabilityProviderRegistry",
    "get_capability_registry",
    "reset_global_registry",
    "CapabilityInjector",
    "get_capability_injector",
    "create_capability_injector",
    # Migration Helpers
    "deprecated_direct_instantiation",
    "migrate_capability_property",
    "migrate_to_injector",
    "get_capability_or_create",
    "check_migration_status",
    "print_migration_report",
    # Vertical loader
    "VerticalLoader",
    "get_vertical_loader",
    "load_vertical",
    "get_active_vertical",
    "get_vertical_extensions",
    # Lazy extensions
    "ExtensionLoadTrigger",
    "LazyVerticalExtensions",
    "create_lazy_extensions",
    "get_extension_load_trigger",
    # Helper functions
    "get_vertical",
    "list_verticals",
    # Layer boundary enforcement: moved from victor.agent.vertical_context
    "VerticalContext",
    "VerticalContextProtocol",
    "MutableVerticalContextProtocol",
    "create_vertical_context",
]


def _register_and_discover_verticals() -> None:
    """Register and discover all verticals using the PluginDiscovery system.

    This function replaces the old hardcoded vertical registration with a
    flexible plugin discovery system that supports:
    - Built-in verticals (always available)
    - Entry point discovery (Python standard for plugins)
    - YAML fallback (for air-gapped environments)

    Phase 3 OCP Compliance:
    This implementation follows the Open/Closed Principle by allowing verticals
    to be added without modifying core code (via entry points or YAML).

    To disable plugin discovery and use the old hardcoded registration, set:
    export VICTOR_USE_PLUGIN_DISCOVERY=false

    To enable air-gapped mode (entry points disabled, YAML fallback enabled):
    export VICTOR_AIRGAPPED=true
    """
    import os
    import logging

    logger = logging.getLogger(__name__)

    # Check if plugin discovery is enabled (default: true)
    use_plugin_discovery = os.getenv("VICTOR_USE_PLUGIN_DISCOVERY", "true").lower() == "true"

    # Check if lazy loading is enabled (default: true)
    lazy_loading_enabled = os.getenv("VICTOR_LAZY_LOADING", "true").lower() == "true"

    if use_plugin_discovery:
        # NEW: Use PluginDiscovery system (Phase 3 OCP Compliance)
        try:
            from victor.core.verticals.plugin_discovery import get_plugin_discovery

            discovery = get_plugin_discovery()
            result = discovery.discover_all()

            # Register discovered verticals
            if lazy_loading_enabled:
                # Register lazy imports from discovery result
                for name, lazy_import in result.lazy_imports.items():
                    try:
                        VerticalRegistry.register_lazy_import(name, lazy_import)
                        logger.debug(f"Registered lazy import for vertical '{name}'")
                    except Exception as e:
                        logger.warning(f"Failed to register lazy import for '{name}': {e}")

                # Register any non-lazy verticals (from entry points or YAML)
                for name, vertical_class in result.verticals.items():
                    if vertical_class is not None:  # Not a lazy import
                        try:
                            VerticalRegistry.register(vertical_class)
                            logger.debug(
                                f"Registered vertical '{name}' from {result.sources.get(name, 'unknown')}"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to register vertical '{name}': {e}")
            else:
                # Eager loading - import and register all verticals immediately
                for name, lazy_import in result.lazy_imports.items():
                    try:
                        module_path, class_name = lazy_import.split(":")
                        module = __import__(module_path, fromlist=[class_name])
                        vertical_class = getattr(module, class_name)
                        VerticalRegistry.register(vertical_class)
                        logger.debug(f"Eagerly loaded and registered vertical '{name}'")
                    except Exception as e:
                        logger.warning(f"Failed to eagerly load vertical '{name}': {e}")

                # Register any non-lazy verticals
                for name, vertical_class in result.verticals.items():
                    if vertical_class is not None:
                        try:
                            VerticalRegistry.register(vertical_class)
                            logger.debug(
                                f"Registered vertical '{name}' from {result.sources.get(name, 'unknown')}"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to register vertical '{name}': {e}")

            logger.info(
                f"Plugin discovery completed: {len(result.lazy_imports)} lazy, {len([v for v in result.verticals.values() if v is not None])} eager"
            )

        except Exception as e:
            logger.error(f"Plugin discovery failed, falling back to hardcoded registration: {e}")
            # Fall back to old hardcoded registration
            _register_builtin_verticals_fallback(lazy_loading_enabled)
    else:
        # OLD: Use hardcoded registration (backward compatibility)
        logger.debug("Using hardcoded vertical registration (plugin discovery disabled)")
        _register_builtin_verticals_fallback(lazy_loading_enabled)


def _register_builtin_verticals_fallback(lazy_loading_enabled: bool) -> None:
    """Fallback function for hardcoded vertical registration (backward compatibility).

    This function is only used when plugin discovery is disabled or fails.
    """
    import logging

    logger = logging.getLogger(__name__)

    if lazy_loading_enabled:
        # Register lazy imports - verticals will be loaded on first access
        VerticalRegistry.register_lazy_import("coding", "victor.coding:CodingAssistant")
        VerticalRegistry.register_lazy_import("research", "victor.research:ResearchAssistant")
        VerticalRegistry.register_lazy_import("devops", "victor.devops:DevOpsAssistant")
        VerticalRegistry.register_lazy_import(
            "dataanalysis", "victor.dataanalysis:DataAnalysisAssistant"
        )
        VerticalRegistry.register_lazy_import("rag", "victor.rag:RAGAssistant")
        VerticalRegistry.register_lazy_import("benchmark", "victor.benchmark:BenchmarkVertical")
        logger.debug("Registered 6 built-in verticals with lazy loading")
    else:
        # Eager loading - import all verticals immediately (legacy behavior)
        verticals_to_load = [
            ("coding", "victor.coding", "CodingAssistant"),
            ("research", "victor.research", "ResearchAssistant"),
            ("devops", "victor.devops", "DevOpsAssistant"),
            ("dataanalysis", "victor.dataanalysis", "DataAnalysisAssistant"),
            ("rag", "victor.rag", "RAGAssistant"),
            ("benchmark", "victor.benchmark", "BenchmarkVertical"),
        ]

        for name, module_path, class_name in verticals_to_load:
            try:
                module = __import__(module_path, fromlist=[class_name])
                vertical_class = getattr(module, class_name)
                VerticalRegistry.register(vertical_class)
                logger.debug(f"Eagerly loaded vertical '{name}'")
            except ImportError:
                logger.warning(f"Failed to import vertical '{name}' from {module_path}")
                pass


def _discover_external_verticals() -> None:
    """Discover and register external verticals from entry points.

    DEPRECATED: This function is kept for backward compatibility but is no longer
    used. The new PluginDiscovery system handles external vertical discovery
    internally.

    This function will be removed in a future release.
    """
    import logging

    logger = logging.getLogger(__name__)

    logger.debug("External vertical discovery is now handled by PluginDiscovery system")
    try:
        VerticalRegistry.discover_external_verticals()
    except Exception:
        # Silently ignore discovery failures during import
        pass


# Register and discover all verticals on import (Phase 3 OCP Compliance)
_register_and_discover_verticals()

# Legacy external discovery (now handled by PluginDiscovery, kept for compatibility)
# _discover_external_verticals()  # Commented out - redundant with PluginDiscovery


def get_vertical(name: str | None) -> type[VerticalBase] | None:
    """Look up a vertical by name.

    Args:
        name: Vertical name (case-insensitive), or None.

    Returns:
        Vertical class or None if not found.
    """
    if name is None:
        return None

    # Try exact match first
    result = VerticalRegistry.get(name)
    if result:
        return result

    # Try case-insensitive match
    name_lower = name.lower()
    for registered_name in VerticalRegistry.list_names():
        if registered_name.lower() == name_lower:
            return VerticalRegistry.get(registered_name)

    return None


def list_verticals() -> list[str]:
    """List all available vertical names.

    Returns:
        List of registered vertical names.
    """
    return VerticalRegistry.list_names()
