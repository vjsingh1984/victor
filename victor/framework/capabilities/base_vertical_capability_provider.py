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

"""Base vertical capability provider for eliminating duplication across verticals.

This module provides a comprehensive base class for vertical capability providers,
eliminating ~2000 lines of duplication across coding, research, devops, and dataanalysis.

Key Features:
- Declarative capability definitions via _get_capability_definitions()
- Automatic configure/get/set function generation
- @capability decorator support
- CAPABILITIES list generation for CapabilityLoader
- Applied capability tracking
- Type-safe capability retrieval

Design Pattern: Template Method + Strategy
- Template Method: _get_capability_definitions() (abstract)
- Strategy: Different capability types (configure, get, set)

Example:
    class CodingCapabilityProvider(BaseVerticalCapabilityProvider):
        def __init__(self):
            super().__init__("coding")

        def _get_capability_definitions(self) -> Dict[str, CapabilityDefinition]:
            return {
                "git_safety": CapabilityDefinition(
                    name="git_safety",
                    type=CapabilityType.SAFETY,
                    description="Git safety rules",
                    version="1.0",
                    configure_fn="configure_git_safety",
                    get_fn=None,
                    default_config={
                        "block_force_push": True,
                        "block_main_push": True,
                    },
                ),
                "code_style": CapabilityDefinition(
                    name="code_style",
                    type=CapabilityType.MODE,
                    description="Code style configuration",
                    version="1.0",
                    configure_fn="configure_code_style",
                    get_fn="get_code_style",
                    default_config={
                        "formatter": "black",
                        "linter": "ruff",
                    },
                ),
            }

        # Implement configure_* and get_* functions
        def configure_git_safety(self, orchestrator, **kwargs):
            # Implementation
            pass

        def get_code_style(self, orchestrator):
            # Implementation
            pass

    # Usage
    provider = CodingCapabilityProvider()
    provider.apply_git_safety(orchestrator, block_force_push=True)
    style = provider.get_code_style(orchestrator)
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from victor.core.capabilities import Capability as CoreCapability, CapabilityType
from victor.framework.capabilities.base import BaseCapabilityProvider, CapabilityMetadata
from victor.framework.protocols import CapabilityType as FrameworkCapabilityType
from victor.framework.capability_loader import CapabilityEntry, capability

if TYPE_CHECKING:
    from victor.core.protocols import OrchestratorProtocol as AgentOrchestrator

logger = logging.getLogger(__name__)


# Map between core and framework CapabilityType
def _map_capability_type(framework_type: FrameworkCapabilityType) -> str:
    """Map framework CapabilityType to core CapabilityType.

    Args:
        framework_type: Framework capability type

    Returns:
        Core capability type string
    """
    mapping = {
        FrameworkCapabilityType.TOOL: CapabilityType.TOOL,
        FrameworkCapabilityType.WORKFLOW: CapabilityType.WORKFLOW,
        FrameworkCapabilityType.MIDDLEWARE: CapabilityType.MIDDLEWARE,
        FrameworkCapabilityType.VALIDATOR: CapabilityType.VALIDATOR,
        FrameworkCapabilityType.OBSERVER: CapabilityType.OBSERVER,
        FrameworkCapabilityType.MODE: CapabilityType.TOOL,  # Map MODE to TOOL
        FrameworkCapabilityType.SAFETY: CapabilityType.MIDDLEWARE,  # Map SAFETY to MIDDLEWARE
    }
    return mapping.get(framework_type, CapabilityType.TOOL)


@dataclass
class CapabilityDefinition:
    """Definition for a capability in a vertical provider.

    Attributes:
        name: Unique capability identifier
        type: Framework capability type (TOOL, WORKFLOW, MODE, SAFETY, etc.)
        description: Human-readable description
        version: Capability version (default: "1.0")
        configure_fn: Name of configure_* function (e.g., "configure_git_safety")
        get_fn: Name of get_* function (e.g., "get_code_style") or None
        set_fn: Name of set_* function (e.g., "set_code_style") or None
        default_config: Default configuration dictionary
        dependencies: List of capability dependencies
        tags: List of tags for discovery
        enabled: Whether capability is enabled by default
    """

    name: str
    type: FrameworkCapabilityType
    description: str
    version: str = "1.0"
    configure_fn: Optional[str] = None
    get_fn: Optional[str] = None
    set_fn: Optional[str] = None
    default_config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    enabled: bool = True

    def to_capability_metadata(self) -> CapabilityMetadata:
        """Convert to CapabilityMetadata.

        Returns:
            CapabilityMetadata instance
        """
        return CapabilityMetadata(
            name=self.name,
            description=self.description,
            version=self.version,
            dependencies=self.dependencies,
            tags=self.tags,
        )

    def to_core_capability(self, vertical_prefix: str) -> CoreCapability:
        """Convert to core Capability.

        Args:
            vertical_prefix: Vertical name prefix (e.g., "coding", "research")

        Returns:
            CoreCapability instance
        """
        return CoreCapability(
            name=f"{vertical_prefix}_{self.name}",
            type=_map_capability_type(self.type),
            description=self.description,
            version=self.version,
            enabled=self.enabled,
            dependencies=self.dependencies,
            handler=self.configure_fn,
            config=self.default_config.copy(),
        )


class BaseVerticalCapabilityProvider(BaseCapabilityProvider[Callable[..., None]]):
    """Base class for vertical capability providers.

    Eliminates ~500 lines of duplication per vertical by providing:
    - Declarative capability definitions
    - Automatic configure/get/set method generation
    - @capability decorator generation
    - CAPABILITIES list generation
    - Applied capability tracking

    Usage:
        class CodingCapabilityProvider(BaseVerticalCapabilityProvider):
            def __init__(self):
                super().__init__("coding")

            def _get_capability_definitions(self) -> Dict[str, CapabilityDefinition]:
                return {
                    "git_safety": CapabilityDefinition(
                        name="git_safety",
                        type=CapabilityType.SAFETY,
                        description="Git safety rules",
                        configure_fn="configure_git_safety",
                        default_config={"block_force_push": True},
                    ),
                }

            # Implement configure_git_safety method
            def configure_git_safety(self, orchestrator, **kwargs):
                # Implementation
                pass

        # Use the provider
        provider = CodingCapabilityProvider()
        provider.apply_git_safety(orchestrator, block_force_push=True)
    """

    def __init__(self, vertical_name: str):
        """Initialize the vertical capability provider.

        Args:
            vertical_name: Name of the vertical (e.g., "coding", "research")
        """
        self._vertical_name = vertical_name
        self._applied: Set[str] = set()
        self._definitions_cache: Optional[Dict[str, CapabilityDefinition]] = None
        self._capabilities_cache: Optional[Dict[str, Callable[..., None]]] = None
        self._metadata_cache: Optional[Dict[str, CapabilityMetadata]] = None

    @abstractmethod
    def _get_capability_definitions(self) -> Dict[str, CapabilityDefinition]:
        """Define capability definitions for this vertical.

        This method MUST be implemented by subclasses to declare their capabilities.
        Each capability should have a corresponding configure_* method implemented
        on the subclass.

        Example:
            return {
                "git_safety": CapabilityDefinition(
                    name="git_safety",
                    type=CapabilityType.SAFETY,
                    description="Git safety rules for preventing dangerous operations",
                    version="1.0",
                    configure_fn="configure_git_safety",
                    default_config={
                        "block_force_push": True,
                        "block_main_push": True,
                    },
                    tags=["safety", "git", "version-control"],
                ),
                "code_style": CapabilityDefinition(
                    name="code_style",
                    type=CapabilityType.MODE,
                    description="Code style and formatting configuration",
                    version="1.0",
                    configure_fn="configure_code_style",
                    get_fn="get_code_style",
                    default_config={
                        "formatter": "black",
                        "linter": "ruff",
                        "max_line_length": 100,
                    },
                    tags=["style", "formatting", "linting"],
                ),
            }

        Returns:
            Dictionary mapping capability names to their definitions
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _get_capability_definitions()"
        )

    def _get_definitions(self) -> Dict[str, CapabilityDefinition]:
        """Get capability definitions with caching.

        Returns:
            Dictionary of capability definitions
        """
        if self._definitions_cache is None:
            self._definitions_cache = self._get_capability_definitions()
        return self._definitions_cache

    def get_capabilities(self) -> Dict[str, Callable[..., None]]:
        """Return all registered capability functions.

        Returns:
            Dictionary mapping capability names to configure_* functions
        """
        if self._capabilities_cache is None:
            definitions = self._get_definitions()
            self._capabilities_cache = {}

            for name, definition in definitions.items():
                if definition.configure_fn:
                    # Get the configure_* method from this instance
                    configure_fn = getattr(self, definition.configure_fn, None)
                    if configure_fn is not None:
                        self._capabilities_cache[name] = configure_fn
                    else:
                        logger.warning(
                            f"Configure function '{definition.configure_fn}' not found for capability '{name}'"
                        )

        return self._capabilities_cache.copy()

    def get_capability_metadata(self) -> Dict[str, CapabilityMetadata]:
        """Return metadata for all registered capabilities.

        Returns:
            Dictionary mapping capability names to their metadata
        """
        if self._metadata_cache is None:
            definitions = self._get_definitions()
            self._metadata_cache = {
                name: definition.to_capability_metadata()
                for name, definition in definitions.items()
            }
        return self._metadata_cache.copy()

    def get_capability(self, name: str) -> Optional[Callable[..., None]]:
        """Get a specific capability function by name.

        Args:
            name: Capability name

        Returns:
            Configure function or None if not found
        """
        return self.get_capabilities().get(name)

    def list_capabilities(
        self, capability_type: Optional[FrameworkCapabilityType] = None
    ) -> List[str]:
        """List capabilities by type.

        Args:
            capability_type: Filter by type (optional)

        Returns:
            List of capability names
        """
        definitions = self._get_definitions()
        if capability_type:
            return [
                name
                for name, definition in definitions.items()
                if definition.type == capability_type and definition.enabled
            ]
        return [name for name, definition in definitions.items() if definition.enabled]

    def has_capability(self, name: str) -> bool:
        """Check if a capability exists.

        Args:
            name: Capability name

        Returns:
            True if capability exists
        """
        return name in self._get_definitions()

    def get_capability_definition(self, name: str) -> Optional[CapabilityDefinition]:
        """Get capability definition by name.

        Args:
            name: Capability name

        Returns:
            CapabilityDefinition or None
        """
        return self._get_definitions().get(name)

    def apply_capability(self, orchestrator: Any, name: str, **kwargs: Any) -> None:
        """Apply a capability by name.

        Args:
            orchestrator: Target orchestrator
            name: Capability name
            **kwargs: Configuration options

        Raises:
            ValueError: If capability not found
        """
        definition = self.get_capability_definition(name)
        if not definition:
            raise ValueError(f"Unknown capability: {name}")

        if not definition.configure_fn:
            raise ValueError(f"Capability '{name}' has no configure function")

        configure_fn = getattr(self, definition.configure_fn)
        if not configure_fn:
            raise ValueError(
                f"Configure function '{definition.configure_fn}' not found for capability '{name}'"
            )

        # Merge default config with kwargs
        config = definition.default_config.copy()
        config.update(kwargs)

        configure_fn(orchestrator, **config)
        self._applied.add(name)

        logger.info(f"Applied capability '{name}' to {self._vertical_name} vertical")

    def get_capability_config(self, orchestrator: Any, name: str) -> Optional[Dict[str, Any]]:
        """Get current configuration for a capability.

        Args:
            orchestrator: Target orchestrator
            name: Capability name

        Returns:
            Configuration dict or None if get_fn not defined
        """
        definition = self.get_capability_definition(name)
        if not definition or not definition.get_fn:
            return None

        get_fn = getattr(self, definition.get_fn, None)
        if not get_fn:
            logger.warning(f"Get function '{definition.get_fn}' not found for capability '{name}'")
            return None

        return get_fn(orchestrator)

    def get_default_config(self, name: str) -> Dict[str, Any]:
        """Get default configuration for a capability.

        Args:
            name: Capability name

        Returns:
            Default configuration dict

        Raises:
            ValueError: If capability not found
        """
        definition = self.get_capability_definition(name)
        if not definition:
            raise ValueError(f"Unknown capability: {name}")
        return definition.default_config.copy()

    def apply_all(self, orchestrator: Any, **kwargs: Any) -> None:
        """Apply all enabled capabilities with defaults.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Shared options passed to all capabilities
        """
        definitions = self._get_definitions()
        for name, definition in definitions.items():
            if definition.enabled and definition.configure_fn:
                try:
                    self.apply_capability(orchestrator, name, **kwargs)
                except Exception as e:
                    logger.error(
                        f"Failed to apply capability '{name}': {e}",
                        exc_info=True,
                    )

    def get_applied(self) -> Set[str]:
        """Get set of applied capability names.

        Returns:
            Set of applied capability names
        """
        return self._applied.copy()

    def reset_applied(self) -> None:
        """Reset applied capability tracking."""
        self._applied.clear()

    def generate_capabilities_list(self) -> List[CapabilityEntry]:
        """Generate CAPABILITIES list for CapabilityLoader discovery.

        Returns:
            List of CapabilityEntry instances
        """
        from victor.framework.protocols import OrchestratorCapability

        entries: List[CapabilityEntry] = []
        definitions = self._get_definitions()

        for name, definition in definitions.items():
            if not definition.configure_fn:
                continue

            # Build capability metadata
            cap_metadata = {
                "name": f"{self._vertical_name}_{name}",
                "capability_type": definition.type,
                "version": definition.version,
                "setter": definition.configure_fn,
                "description": definition.description,
            }

            # Add getter if available
            if definition.get_fn:
                cap_metadata["getter"] = definition.get_fn

            capability = OrchestratorCapability(**cap_metadata)

            # Get handler functions
            handler = getattr(self, definition.configure_fn, None)
            getter_handler = getattr(self, definition.get_fn, None) if definition.get_fn else None

            if not handler:
                logger.warning(
                    f"Handler function '{definition.configure_fn}' not found for capability '{name}'"
                )
                continue

            entry = CapabilityEntry(
                capability=capability,
                handler=handler,
                getter_handler=getter_handler,
            )
            entries.append(entry)

        return entries

    def generate_capability_configs(self) -> Dict[str, Any]:
        """Generate centralized config storage for VerticalContext.

        Returns:
            Dict with capability configurations for centralized storage
        """
        configs: Dict[str, Any] = {}
        definitions = self._get_definitions()

        for name, definition in definitions.items():
            if definition.default_config:
                # Use configure_fn name as config key (e.g., "code_style" -> "code_style")
                config_key = f"{name}_config"
                configs[config_key] = definition.default_config.copy()

        return configs

    def create_capability_loader(self) -> Any:
        """Create a CapabilityLoader pre-configured for this vertical.

        Returns:
            CapabilityLoader with capabilities registered
        """
        from victor.framework.capability_loader import CapabilityLoader

        loader = CapabilityLoader()

        # Register all capabilities
        for entry in self.generate_capabilities_list():
            loader._register_capability_internal(
                capability=entry.capability,
                handler=entry.handler,
                getter_handler=entry.getter_handler,
                source_module=f"victor.{self._vertical_name}.capabilities",
            )

        return loader


__all__ = [
    "BaseVerticalCapabilityProvider",
    "CapabilityDefinition",
    "_map_capability_type",
]
