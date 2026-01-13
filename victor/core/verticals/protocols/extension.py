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

"""Dynamic Extension Protocols (OCP-Compliant).

This module provides protocol interfaces for a dynamic, Open/Closed-compliant
extension system that enables third-party extensions without modifying core code.

Design Principles:
    - OCP (Open/Closed): Open for extension, closed for modification
    - ISP (Interface Segregation): Focused, minimal protocols
    - DIP (Dependency Inversion): Depend on abstractions, not concretions

Key Features:
    - IExtension: Base protocol for all extension types
    - IExtensionRegistry: Dynamic registration and discovery
    - ExtensionType: Type-safe extension identifiers
    - Supports unlimited extension types without core modifications

Usage:
    # Define a custom extension type
    @dataclass
    class CustomExtension(IExtension):
        extension_type: ClassVar[str] = "custom"
        name: str
        config: Dict[str, Any]

        def validate(self) -> bool:
            return bool(self.config)

    # Register dynamically
    registry = ExtensionRegistry()
    registry.register_extension(CustomExtension(name="my_ext", config={...}))

    # Retrieve by type
    custom_exts = registry.get_extensions_by_type("custom")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    runtime_checkable,
)

if TYPE_CHECKING:
    pass


class StandardExtensionTypes(str, Enum):
    """Standard extension types built into Victor Framework.

    Built-in verticals automatically support these extension types.
    Third-party verticals can define additional types.
    """

    # Tool extensions
    TOOLS = "tools"
    PROVIDERS = "providers"
    WORKFLOWS = "workflows"
    AGENTS = "agents"

    # Framework extensions
    MIDDLEWARE = "middleware"
    SAFETY = "safety_extensions"
    PROMPT = "prompt_contributors"
    MODE_CONFIG = "mode_config_provider"
    TOOL_DEPENDENCY = "tool_dependency_provider"
    WORKFLOW_PROVIDER = "workflow_provider"
    SERVICE_PROVIDER = "service_provider"
    RL_CONFIG = "rl_config_provider"
    TEAM_SPEC = "team_spec_provider"
    ENRICHMENT = "enrichment_strategy"
    TOOL_SELECTION = "tool_selection_strategy"

    # Add more as needed - OCP compliant!


@runtime_checkable
class IExtension(Protocol):
    """Base protocol for all extension types.

    Any object that implements this protocol can be registered as an extension.
    This enables unlimited extension types without modifying core code (OCP).

    Required Methods:
        extension_type: Unique type identifier (e.g., "tools", "middleware")
        name: Unique instance name within the type
        validate(): Validate configuration

    Example:
        @dataclass
        class AnalyticsExtension(IExtension):
            extension_type: ClassVar[str] = "analytics"

            name: str
            api_key: str

            def validate(self) -> bool:
                return bool(self.api_key)
    """

    @property
    def extension_type(self) -> str:
        """Get the unique type identifier for this extension.

        Returns:
            Extension type string (e.g., "tools", "middleware", "analytics")
        """
        ...

    @property
    def name(self) -> str:
        """Get the unique name for this extension instance.

        Returns:
            Extension name (unique within extension_type)
        """
        ...

    def validate(self) -> bool:
        """Validate that this extension is properly configured.

        Returns:
            True if valid, False otherwise
        """
        ...

    def get_metadata(self) -> Dict[str, Any]:
        """Get optional metadata about this extension.

        Returns:
            Dictionary of metadata (version, description, dependencies, etc.)
        """
        ...


@runtime_checkable
class IExtensionRegistry(Protocol):
    """Protocol for dynamic extension registries.

    Provides methods for registering, unregistering, and discovering
    extensions of any type without modifying core code.

    This enables OCP compliance by allowing new extension types
    to be added without changing the registry implementation.
    """

    def register_extension(self, extension: IExtension) -> None:
        """Register an extension.

        Args:
            extension: Extension to register

        Raises:
            ValueError: If extension already registered
            TypeError: If extension doesn't implement IExtension
        """
        ...

    def unregister_extension(self, extension_type: str, name: str) -> bool:
        """Unregister an extension.

        Args:
            extension_type: Type of extension to unregister
            name: Name of extension to unregister

        Returns:
            True if unregistered, False if not found
        """
        ...

    def get_extension(self, extension_type: str, name: str) -> Optional[IExtension]:
        """Get a specific extension.

        Args:
            extension_type: Type of extension
            name: Name of extension

        Returns:
            Extension if found, None otherwise
        """
        ...

    def get_extensions_by_type(self, extension_type: str) -> List[IExtension]:
        """Get all extensions of a specific type.

        Args:
            extension_type: Type of extension to retrieve

        Returns:
            List of extensions (empty list if none found)
        """
        ...

    def list_extension_types(self) -> List[str]:
        """List all registered extension types.

        Returns:
            List of extension type strings
        """
        ...

    def list_extensions(self, extension_type: Optional[str] = None) -> List[str]:
        """List extension names by type.

        Args:
            extension_type: Optional type filter (None = all types)

        Returns:
            List of extension names
        """
        ...

    def has_extension(self, extension_type: str, name: str) -> bool:
        """Check if an extension is registered.

        Args:
            extension_type: Type of extension
            name: Name of extension

        Returns:
            True if registered, False otherwise
        """
        ...

    def count_extensions(self, extension_type: Optional[str] = None) -> int:
        """Count extensions by type.

        Args:
            extension_type: Optional type filter (None = all types)

        Returns:
            Number of extensions
        """
        ...


@dataclass
class ExtensionMetadata:
    """Metadata container for extensions.

    Attributes:
        version: Extension version
        description: Human-readable description
        author: Extension author
        dependencies: List of required extensions/types
        tags: Categorical tags for discovery
        priority: Loading priority (lower = earlier)
    """

    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    dependencies: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    priority: int = 50


# Export all
__all__ = [
    "StandardExtensionTypes",
    "IExtension",
    "IExtensionRegistry",
    "ExtensionMetadata",
]
