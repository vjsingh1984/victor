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

"""Configuration manager for agent orchestrator.

This module provides centralized configuration management for the orchestrator,
extracting configuration-related logic from the monolithic orchestrator class.

Design Principles:
- Read-only configuration access for most operations
- Protocol-based design for testability (DIP)
- Single source of truth for tiered tool configuration
- Clear separation between vertical and session configuration
"""

import logging
from typing import Any, Dict, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.vertical_context import VerticalContext
    from victor.agent.tool_access_controller import ToolAccessController

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """Manages agent configuration with read-only access.

    This manager provides a centralized interface for accessing and applying
    configuration across the agent system, following the Dependency Inversion
    Principle by depending on protocols rather than concrete implementations.

    Example:
        config_manager = ConfigurationManager()
        config_manager.set_tiered_tool_config(tiered_config, vertical_context, access_controller)
        enabled = config_manager.get_enabled_tools(tool_access_controller)
    """

    def __init__(self) -> None:
        """Initialize the configuration manager."""
        self._tiered_config: Optional[Any] = None
        self._vertical_context: Optional["VerticalContext"] = None
        self._tool_access_controller: Optional["ToolAccessController"] = None

    def set_tiered_tool_config(
        self,
        config: Any,
        vertical_context: Optional["VerticalContext"] = None,
        tool_access_controller: Optional["ToolAccessController"] = None,
    ) -> None:
        """Set tiered tool configuration.

        Applies tiered tool config from vertical to:
        1. VerticalContext for storage
        2. ToolAccessController.VerticalLayer for access filtering

        Args:
            config: TieredToolConfig from the active vertical
            vertical_context: Optional VerticalContext to apply config to
            tool_access_controller: Optional ToolAccessController to apply config to
        """
        self._tiered_config = config

        # Store in vertical context
        if vertical_context is not None:
            self._vertical_context = vertical_context
            vertical_context.apply_tiered_config(config)
            logger.debug("Tiered config stored in VerticalContext")

        # Apply to tool access controller
        if tool_access_controller is not None:
            self._tool_access_controller = tool_access_controller
            tool_access_controller.set_tiered_config(config)
            logger.debug("Tiered config applied to ToolAccessController")

        logger.debug("Tiered tool config set in ConfigurationManager")

    def get_tiered_config(self) -> Optional[Any]:
        """Get the current tiered tool configuration.

        Returns:
            Current TieredToolConfig or None if not set
        """
        return self._tiered_config

    def get_enabled_tools(
        self,
        tool_access_controller: Optional["ToolAccessController"] = None,
    ) -> Set[str]:
        """Get the set of enabled tools.

        Args:
            tool_access_controller: Optional ToolAccessController to query.
                If not provided, uses the one previously set via set_tiered_tool_config.

        Returns:
            Set of enabled tool names
        """
        controller = tool_access_controller or self._tool_access_controller
        if controller is None:
            logger.debug("No tool access controller available, returning empty set")
            return set()

        try:
            # Get enabled tools from the access controller
            enabled_tools = controller.get_enabled_tools()
            return enabled_tools
        except Exception as e:
            logger.warning(f"Failed to get enabled tools: {e}")
            return set()

    def get_vertical_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value from the vertical context.

        Args:
            key: Configuration key to retrieve
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if self._vertical_context is None:
            return default

        try:
            # Try to get from vertical context
            if hasattr(self._vertical_context, "get_config"):
                return self._vertical_context.get_config(key, default)
        except Exception as e:
            logger.debug(f"Failed to get vertical config for {key}: {e}")

        return default

    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configuration values as a dictionary.

        Returns:
            Dictionary of all configuration values
        """
        configs: Dict[str, Any] = {}

        if self._tiered_config is not None:
            configs["tiered_config"] = self._tiered_config

        if self._vertical_context is not None:
            try:
                if hasattr(self._vertical_context, "vertical_name"):
                    configs["vertical_name"] = self._vertical_context.vertical_name
                if hasattr(self._vertical_context, "get_all_configs"):
                    configs.update(self._vertical_context.get_all_configs())
            except Exception as e:
                logger.debug(f"Failed to get vertical configs: {e}")

        return configs

    def reset(self) -> None:
        """Reset all configuration to initial state.

        This is primarily useful for testing.
        """
        self._tiered_config = None
        self._vertical_context = None
        self._tool_access_controller = None
        logger.debug("ConfigurationManager reset")


class ConfigurationManagerProtocol:
    """Protocol for configuration management.

    This protocol defines the interface for configuration management,
    enabling dependency inversion and testability.

    Implementations must provide:
    - Read-only access to tiered configuration
    - Query methods for enabled tools
    - Vertical configuration access
    """

    def get_tiered_config(self) -> Optional[Any]:
        """Get the current tiered tool configuration."""

    def get_enabled_tools(self) -> Set[str]:
        """Get the set of enabled tools."""

    def get_vertical_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value from the vertical context."""

    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configuration values as a dictionary."""


def create_configuration_manager() -> ConfigurationManager:
    """Factory function to create a ConfigurationManager.

    Returns:
        ConfigurationManager instance
    """
    return ConfigurationManager()


# Global singleton instance for convenience
_default_manager: Optional[ConfigurationManager] = None


def get_configuration_manager() -> ConfigurationManager:
    """Get the global configuration manager singleton.

    Returns:
        ConfigurationManager singleton instance
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = create_configuration_manager()
    return _default_manager


def reset_configuration_manager() -> None:
    """Reset the global configuration manager.

    This is primarily useful for testing.
    """
    global _default_manager
    if _default_manager is not None:
        _default_manager.reset()
    _default_manager = None
