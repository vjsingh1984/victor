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

from victor.core.verticals.base import (
    StageDefinition,
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
from victor.core.verticals.vertical_loader import (
    VerticalLoader,
    get_vertical_loader,
    load_vertical,
    get_active_vertical,
    get_vertical_extensions,
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
    # Vertical loader
    "VerticalLoader",
    "get_vertical_loader",
    "load_vertical",
    "get_active_vertical",
    "get_vertical_extensions",
    # Helper functions
    "get_vertical",
    "list_verticals",
    # Layer boundary enforcement: moved from victor.agent.vertical_context
    "VerticalContext",
    "VerticalContextProtocol",
    "MutableVerticalContextProtocol",
    "create_vertical_context",
]


def _register_builtin_verticals() -> None:
    """Register all built-in verticals with the registry using lazy loading.

    Lazy loading defers the actual import of vertical classes until they are
    first accessed, significantly improving startup time.

    To disable lazy loading and load all verticals eagerly, set the environment
    variable: VICTOR_LAZY_LOADING=false
    """
    import os
    from victor.core.verticals.naming import normalize_vertical_name

    # Check if lazy loading is enabled (default: true)
    lazy_loading_enabled = os.getenv("VICTOR_LAZY_LOADING", "true").lower() == "true"

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
    else:
        # Eager loading - import all verticals immediately (legacy behavior)
        try:
            from victor.coding import CodingAssistant

            VerticalRegistry.register(CodingAssistant)
        except ImportError:
            pass

        try:
            from victor.research import ResearchAssistant

            VerticalRegistry.register(ResearchAssistant)
        except ImportError:
            pass

        try:
            from victor.devops import DevOpsAssistant

            VerticalRegistry.register(DevOpsAssistant)
        except ImportError:
            pass

        try:
            from victor.dataanalysis import DataAnalysisAssistant

            VerticalRegistry.register(DataAnalysisAssistant)
        except ImportError:
            pass

        try:
            from victor.rag import RAGAssistant

            VerticalRegistry.register(RAGAssistant)
        except ImportError:
            pass

        try:
            from victor.benchmark import BenchmarkVertical

            VerticalRegistry.register(BenchmarkVertical)
        except ImportError:
            pass


def _discover_external_verticals() -> None:
    """Discover and register external verticals from entry points.

    This function scans installed packages for the 'victor.verticals'
    entry point group and registers any valid vertical classes found.

    External packages can register verticals by adding an entry point
    to their pyproject.toml:

        [project.entry-points."victor.verticals"]
        my_vertical = "my_package:MyVerticalAssistant"

    The vertical class must inherit from VerticalBase and implement
    the required abstract methods (get_tools, get_system_prompt).
    """
    try:
        VerticalRegistry.discover_external_verticals()
    except Exception:
        # Silently ignore discovery failures during import
        # Errors are logged by discover_external_verticals()
        pass


# Register built-in verticals on import
_register_builtin_verticals()

# Discover and register external verticals from entry points
_discover_external_verticals()


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
