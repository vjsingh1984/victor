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

"""Common configuration management capability for verticals.

This module provides generic configuration management patterns that are
duplicated across all verticals:

- Configuration with setter (configure_X)
- Configuration retrieval (get_X)
- Configuration validation
- Default values

Replaces the duplicate @capability decorator pattern for configuration
with a consistent, type-safe framework approach.

Design Pattern: Template Method
- Generic configuration storage and retrieval
- Type-safe via TypeVar
- Vertical-specific defaults via subclass

Example:
    from victor.framework.capabilities.configuration import (
        ConfigurationCapabilityProvider,
        configure_capability,
    )

    # Use generic configuration
    configure_capability(orchestrator, "code_style", {"indent": 4})
    config = get_capability(orchestrator, "code_style")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

from victor.framework.capabilities.base import BaseCapabilityProvider, CapabilityMetadata

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ConfigurationCapability:
    """A configuration capability with type-safe value storage.

    Attributes:
        name: Configuration name
        value: Current configuration value
        value_type: Expected type of the value
        default_value: Default value if not set
        validator: Optional validation function
        description: Human-readable description
    """

    name: str
    value: Any
    value_type: type
    default_value: Any = None
    validator: Optional[Callable[[Any], bool]] = None
    description: str = ""

    def validate(self) -> bool:
        """Validate the current configuration value.

        Returns:
            True if valid, False otherwise
        """
        # Type check
        if not isinstance(self.value, self.value_type):
            logger.warning(
                f"Configuration '{self.name}': expected {self.value_type}, "
                f"got {type(self.value)}"
            )
            return False

        # Custom validator
        if self.validator and not self.validator(self.value):
            logger.warning(f"Configuration '{self.name}': validation failed")
            return False

        return True


class ConfigurationCapabilityProvider(BaseCapabilityProvider[ConfigurationCapability]):
    """Generic configuration management provider.

    This provider implements the common "configure_X + get_X" pattern
    that is duplicated across all verticals with a type-safe, consistent
    implementation.

    Example:
        provider = ConfigurationCapabilityProvider("coding")

        # Register configuration
        provider.register_configuration(
            name="code_style",
            value_type=dict,
            default_value={"indent": 4},
            description="Code formatting style"
        )

        # Set configuration
        provider.set_configuration("code_style", {"indent": 2})

        # Get configuration
        config = provider.get_configuration("code_style")
    """

    def __init__(self, vertical_name: str) -> None:
        """Initialize the configuration provider.

        Args:
            vertical_name: Name of the vertical (for namespacing)
        """
        self._vertical_name = vertical_name
        self._configurations: Dict[str, ConfigurationCapability] = {}

    def register_configuration(
        self,
        name: str,
        value_type: type,
        default_value: Any = None,
        validator: Optional[Callable[[Any], bool]] = None,
        description: str = "",
    ) -> None:
        """Register a configuration capability.

        Args:
            name: Configuration name (unique within vertical)
            value_type: Expected type of the value
            default_value: Default value if not set
            validator: Optional validation function
            description: Human-readable description
        """
        config = ConfigurationCapability(
            name=name,
            value=default_value,
            value_type=value_type,
            default_value=default_value,
            validator=validator,
            description=description,
        )
        self._configurations[name] = config

    def set_configuration(self, name: str, value: Any) -> bool:
        """Set a configuration value.

        Args:
            name: Configuration name
            value: New value

        Returns:
            True if set successfully, False otherwise
        """
        if name not in self._configurations:
            logger.warning(f"Unknown configuration: {name}")
            return False

        config = self._configurations[name]
        config.value = value

        if not config.validate():
            # Reset to default on validation failure
            config.value = config.default_value
            return False

        return True

    def get_configuration(self, name: str) -> Optional[Any]:
        """Get a configuration value.

        Args:
            name: Configuration name

        Returns:
            Configuration value or None if not found
        """
        config = self._configurations.get(name)
        return config.value if config else None

    def get_capabilities(self) -> Dict[str, ConfigurationCapability]:
        """Return all registered configurations.

        Returns:
            Dictionary mapping configuration names to capability objects
        """
        return self._configurations

    def get_capability_metadata(self) -> Dict[str, CapabilityMetadata]:
        """Return metadata for all configurations.

        Returns:
            Dictionary mapping configuration names to their metadata
        """
        return {
            name: CapabilityMetadata(
                name=config.name,
                description=config.description,
                version="1.0",
                tags=["configuration", self._vertical_name],
            )
            for name, config in self._configurations.items()
        }


# =============================================================================
# Generic Configuration Functions (convenience API)
# =============================================================================


def configure_capability(
    orchestrator: Any,
    name: str,
    value: Any,
    vertical: str = "default",
) -> bool:
    """Configure a capability using the generic configuration system.

    This is a convenience function that provides the same functionality as
    the vertical-specific configure_X functions, but with a consistent API.

    Args:
        orchestrator: Target orchestrator instance
        name: Configuration name
        value: Configuration value
        vertical: Vertical name for namespacing (default: "default")

    Returns:
        True if configured successfully, False otherwise

    Example:
        # Instead of:
        orchestrator.configure_code_style({"indent": 4})

        # Use:
        configure_capability(orchestrator, "code_style", {"indent": 4}, vertical="coding")
    """
    # Get or create configuration dict
    if not hasattr(orchestrator, "_capability_config"):
        orchestrator._capability_config = {}

    # Namespace by vertical
    namespace = f"{vertical}.{name}"
    orchestrator._capability_config[namespace] = value

    return True


def get_capability(orchestrator: Any, name: str, vertical: str = "default") -> Optional[Any]:
    """Get a capability configuration value.

    Args:
        orchestrator: Target orchestrator instance
        name: Configuration name
        vertical: Vertical name for namespacing (default: "default")

    Returns:
        Configuration value or None if not found

    Example:
        config = get_capability(orchestrator, "code_style", vertical="coding")
    """
    if not hasattr(orchestrator, "_capability_config"):
        return None

    namespace = f"{vertical}.{name}"
    return orchestrator._capability_config.get(namespace)
