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

"""Auto-generated capability handlers (Phase 5.1).

This module provides auto-generated handlers for capability configure/get operations.
Handlers are generated from CapabilityDefinition instances and provide:
- Config validation against JSON Schema
- Merged config retrieval (defaults + stored + overrides)
- Centralized storage via VerticalContext

Design Pattern: Strategy + Template Method
- Strategy: Different capabilities, same interface
- Template Method: Validate → Store → Return pattern
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Protocol, TYPE_CHECKING

from victor.core.capabilities.types import CapabilityDefinition

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class VerticalContextProtocol(Protocol):
    """Protocol for VerticalContext capability storage.

    This protocol defines the minimal interface needed for capability
    configuration storage, enabling loose coupling with the actual
    VerticalContext implementation.
    """

    def set_capability_config(self, name: str, config: Dict[str, Any]) -> None:
        """Store capability configuration.

        Args:
            name: Capability name
            config: Configuration dictionary
        """
        ...

    def get_capability_config(
        self, name: str, default: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Retrieve capability configuration.

        Args:
            name: Capability name
            default: Default value if not found

        Returns:
            Configuration dictionary
        """
        ...


class CapabilityHandler:
    """Auto-generated handler for capability operations.

    Provides a consistent interface for configuring and retrieving
    capability settings through VerticalContext storage.

    Example:
        # Get handler from registry
        handler = registry.get_handler("code_style")

        # Configure capability
        handler.configure(context, {"formatter": "black", "line_length": 100})

        # Get current config (merged with defaults)
        config = handler.get_config(context)
        print(config["formatter"])  # "black"
    """

    def __init__(self, definition: CapabilityDefinition):
        """Initialize capability handler.

        Args:
            definition: Capability definition to handle
        """
        self._definition = definition
        self._name = definition.name

    @property
    def name(self) -> str:
        """Get capability name."""
        return self._name

    @property
    def definition(self) -> CapabilityDefinition:
        """Get capability definition."""
        return self._definition

    def configure(
        self,
        context: VerticalContextProtocol,
        config: Dict[str, Any],
        validate: bool = True,
    ) -> None:
        """Configure capability with provided settings.

        Args:
            context: VerticalContext for storage
            config: Configuration dictionary
            validate: Whether to validate against schema (default: True)

        Raises:
            ValueError: If config fails validation
        """
        # Validate config against schema
        if validate and self._definition.config_schema:
            errors = self._definition.get_validation_errors(config)
            if errors:
                error_msg = "; ".join(errors)
                raise ValueError(
                    f"Invalid configuration for '{self._name}': {error_msg}"
                )

        # Merge with defaults to ensure complete config
        merged_config = self._definition.merge_config({}, config)

        # Store in context
        context.set_capability_config(self._name, merged_config)

        logger.debug(f"Configured capability '{self._name}'")

    def get_config(
        self,
        context: VerticalContextProtocol,
        override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get current configuration merged with defaults.

        Args:
            context: VerticalContext for storage
            override: Optional overrides to apply

        Returns:
            Merged configuration dictionary
        """
        # Get stored config
        stored = context.get_capability_config(self._name, {})

        # Merge: defaults < stored < override
        return self._definition.merge_config(stored, override)

    def reset(self, context: VerticalContextProtocol) -> None:
        """Reset capability to default configuration.

        Args:
            context: VerticalContext for storage
        """
        context.set_capability_config(self._name, self._definition.default_config.copy())
        logger.debug(f"Reset capability '{self._name}' to defaults")

    def is_configured(self, context: VerticalContextProtocol) -> bool:
        """Check if capability has been configured.

        Args:
            context: VerticalContext for storage

        Returns:
            True if capability has stored configuration
        """
        stored = context.get_capability_config(self._name, None)
        return stored is not None

    def validate(self, config: Dict[str, Any]) -> bool:
        """Validate configuration without storing.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if configuration is valid
        """
        return self._definition.validate_config(config)


__all__ = [
    "CapabilityHandler",
    "VerticalContextProtocol",
]
