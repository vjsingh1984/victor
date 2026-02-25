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
]


def _register_builtin_verticals() -> None:
    """Register all built-in verticals with the registry.

    Built-in verticals are those that exist in the victor namespace.
    External verticals (coding, research, devops, rag, dataanalysis) are
    discovered via entry points in _discover_external_verticals().
    """
    # NOTE: coding, research, devops, rag, dataanalysis are now EXTERNAL verticals
    # They are discovered via entry points, not imported directly.
    # This prevents circular dependencies.

    # Only register verticals that truly exist in the victor package
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


# NOTE: We do NOT call _register_builtin_verticals() or _discover_external_verticals()
# at module import time to avoid circular dependencies.
# These are called lazily when first needed via get_vertical() or list_verticals()

_registration_done = False


def _ensure_registration() -> None:
    """Ensure verticals are registered (lazy initialization).

    This is called on first access to avoid circular dependencies during
    framework initialization. External verticals may import from framework,
    so we must complete framework initialization before loading them.
    """
    global _registration_done
    if _registration_done:
        return

    _register_builtin_verticals()
    _discover_external_verticals()
    _registration_done = True


def get_vertical(name: str | None) -> type[VerticalBase] | None:
    """Look up a vertical by name.

    Args:
        name: Vertical name (case-insensitive), or None.

    Returns:
        Vertical class or None if not found.
    """
    # Ensure verticals are registered before lookup (lazy initialization)
    _ensure_registration()

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
    # Ensure verticals are registered before listing (lazy initialization)
    _ensure_registration()
    return VerticalRegistry.list_names()
