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

"""Data-Driven Mode Configuration System.

This module provides a centralized, YAML-based configuration system for
agent modes across all verticals, replacing scattered mode_config.py files.

Design Patterns:
    - Registry: Singleton ModeConfigRegistry for config access
    - Factory: Generate defaults when YAML not found
    - Data-Driven: YAML files override code defaults
    - Type-Safe: Pydantic models for validation

Use Cases:
    - Mode configuration (build, plan, explore, debug, review)
    - Vertical-specific mode overrides
    - Tool permission management
    - Exploration budget configuration

Example:
    from victor.core.config import ModeConfigRegistry

    registry = ModeConfigRegistry.get_instance()
    config = registry.get_mode("coding", "plan")
    print(config.exploration)  # ExplorationLevel.THOROUGH
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


class ExplorationLevel(str, Enum):
    """Exploration intensity levels.

    Attributes:
        MINIMAL: Direct approach, minimal exploration
        STANDARD: Balanced exploration
        THOROUGH: 2.5x exploration (planning mode)
        DEEP: 3.0x exploration (exploration mode)
    """

    MINIMAL = "minimal"
    STANDARD = "standard"
    THOROUGH = "thorough"
    DEEP = "deep"

    def get_multiplier(self) -> float:
        """Get the exploration multiplier for this level.

        Returns:
            Multiplier value (1.0 to 3.0)
        """
        multipliers = {
            ExplorationLevel.MINIMAL: 1.0,
            ExplorationLevel.STANDARD: 1.5,
            ExplorationLevel.THOROUGH: 2.5,
            ExplorationLevel.DEEP: 3.0,
        }
        return multipliers.get(self, 1.0)


class EditPermission(str, Enum):
    """Edit permission levels.

    Attributes:
        NONE: Read-only, no edits allowed
        SANDBOX: Edits allowed only in sandbox directory
        FULL: All edits allowed
    """

    NONE = "none"
    SANDBOX = "sandbox"
    FULL = "full"


@dataclass
class AgentMode:
    """Canonical mode configuration.

    Attributes:
        name: Mode identifier (e.g., "build", "plan")
        display_name: Human-readable name
        exploration: Exploration intensity level
        edit_permission: Edit permission level
        tool_budget_multiplier: Tool budget multiplier (0.0 to 10.0)
        max_iterations: Maximum iteration limit
        allowed_tools: Set of allowed tool names
        denied_tools: Set of denied tool names
        description: Human-readable description
        metadata: Additional metadata
    """

    name: str
    display_name: str
    exploration: ExplorationLevel
    edit_permission: EditPermission
    tool_budget_multiplier: float = 1.0
    max_iterations: int = 10
    allowed_tools: set[str] = field(default_factory=set)
    denied_tools: set[str] = field(default_factory=set)
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_exploration_multiplier(self) -> float:
        """Get the exploration multiplier for this mode.

        Returns:
            Exploration multiplier value
        """
        return self.exploration.get_multiplier()

    def allows_tool(self, tool_name: str) -> bool:
        """Check if a tool is allowed in this mode.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if tool is allowed
        """
        # Explicit denial takes precedence
        if tool_name in self.denied_tools:
            return False

        # If allowed_tools is specified, check it
        if self.allowed_tools:
            return tool_name in self.allowed_tools

        # Default to allowing
        return True

    def allows_edits(self) -> bool:
        """Check if this mode allows any edits.

        Returns:
            True if edits are allowed
        """
        return self.edit_permission != EditPermission.NONE

    def allows_sandbox_edits_only(self) -> bool:
        """Check if this mode allows only sandbox edits.

        Returns:
            True if only sandbox edits are allowed
        """
        return self.edit_permission == EditPermission.SANDBOX


@dataclass
class VerticalModeConfig:
    """Vertical-specific mode configuration.

    Attributes:
        vertical_name: Name of the vertical
        default_mode: Default mode name
        modes: Dictionary of mode configurations
        mode_overrides: Vertical-specific overrides
    """

    vertical_name: str
    default_mode: str = "build"
    modes: dict[str, AgentMode] = field(default_factory=dict)
    mode_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_mode(self, mode_name: str) -> Optional[AgentMode]:
        """Get a mode configuration by name.

        Args:
            mode_name: Name of the mode

        Returns:
            AgentMode instance or None if not found
        """
        return self.modes.get(mode_name, self.modes.get(self.default_mode))

    def list_modes(self) -> list[str]:
        """List available mode names.

        Returns:
            List of mode names
        """
        return list(self.modes.keys())


class ModeConfigRegistry:
    """Registry for mode configurations.

    This singleton registry loads mode configurations from YAML files
    and provides them to verticals via a simple API.

    Example:
        registry = ModeConfigRegistry.get_instance()
        config = registry.get_mode("coding", "plan")
    """

    _instance: Optional["ModeConfigRegistry"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize the registry."""
        self._configs: dict[str, VerticalModeConfig] = {}
        # Config directory is victor/config/modes/
        self._config_dir = Path(__file__).parent.parent / "config" / "modes"
        logger.debug(f"ModeConfigRegistry: Initialized with config_dir={self._config_dir}")

    @classmethod
    def get_instance(cls) -> "ModeConfigRegistry":
        """Get singleton instance.

        Returns:
            ModeConfigRegistry instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load_config(self, vertical_name: str) -> VerticalModeConfig:
        """Load vertical mode config from YAML.

        Args:
            vertical_name: Name of the vertical

        Returns:
            VerticalModeConfig instance
        """
        # Check cache first
        if vertical_name in self._configs:
            return self._configs[vertical_name]

        # Try loading from YAML
        config_file = self._config_dir / f"{vertical_name}_modes.yaml"
        if config_file.exists():
            logger.debug(f"ModeConfigRegistry: Loading from YAML: {config_file}")
            config = self._load_from_yaml(config_file)
        else:
            logger.debug(
                f"ModeConfigRegistry: YAML not found, generating default for '{vertical_name}'"
            )
            config = self._generate_default_config(vertical_name)

        self._configs[vertical_name] = config
        return config

    def _load_from_yaml(self, path: Path) -> VerticalModeConfig:
        """Load YAML configuration file.

        Args:
            path: Path to YAML file

        Returns:
            VerticalModeConfig instance
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        # Convert YAML data to AgentMode instances
        modes = {}
        for mode_name, mode_data in data.get("modes", {}).items():
            # Convert exploration string to enum
            exploration_str = mode_data.get("exploration", "standard")
            exploration = ExplorationLevel(exploration_str)

            # Convert edit_permission string to enum
            edit_str = mode_data.get("edit_permission", "full")
            edit_permission = EditPermission(edit_str)

            modes[mode_name] = AgentMode(
                name=mode_data.get("name", mode_name),
                display_name=mode_data.get("display_name", mode_name.title()),
                exploration=exploration,
                edit_permission=edit_permission,
                tool_budget_multiplier=mode_data.get("tool_budget_multiplier", 1.0),
                max_iterations=mode_data.get("max_iterations", 10),
                allowed_tools=set(mode_data.get("allowed_tools", [])),
                denied_tools=set(mode_data.get("denied_tools", [])),
                description=mode_data.get("description", ""),
                metadata=mode_data.get("metadata", {}),
            )

        return VerticalModeConfig(
            vertical_name=data.get("vertical_name", ""),
            default_mode=data.get("default_mode", "build"),
            modes=modes,
            mode_overrides=data.get("mode_overrides", {}),
        )

    def _generate_default_config(self, vertical_name: str) -> VerticalModeConfig:
        """Generate sensible defaults for a vertical.

        Args:
            vertical_name: Name of the vertical

        Returns:
            VerticalModeConfig with default modes
        """
        return VerticalModeConfig(
            vertical_name=vertical_name,
            default_mode="build",
            modes={
                "build": AgentMode(
                    name="build",
                    display_name="Build",
                    exploration=ExplorationLevel.STANDARD,
                    edit_permission=EditPermission.FULL,
                    tool_budget_multiplier=1.0,
                    description="Standard mode with full edit permissions",
                ),
                "plan": AgentMode(
                    name="plan",
                    display_name="Plan",
                    exploration=ExplorationLevel.THOROUGH,
                    edit_permission=EditPermission.SANDBOX,
                    tool_budget_multiplier=2.5,
                    description="Planning mode with 2.5x exploration and sandbox edits",
                ),
                "explore": AgentMode(
                    name="explore",
                    display_name="Explore",
                    exploration=ExplorationLevel.DEEP,
                    edit_permission=EditPermission.NONE,
                    tool_budget_multiplier=3.0,
                    description="Exploration mode with 3.0x exploration and no edits",
                ),
            },
        )

    def get_mode(self, vertical_name: str, mode_name: str) -> AgentMode:
        """Get specific mode configuration for a vertical.

        Args:
            vertical_name: Name of the vertical
            mode_name: Name of the mode

        Returns:
            AgentMode instance

        Raises:
            ValueError: If mode is not found

        Example:
            mode = registry.get_mode("coding", "plan")
        """
        config = self.load_config(vertical_name)
        mode = config.get_mode(mode_name)

        if not mode:
            # Fall back to default mode
            mode = config.get_mode(config.default_mode)

        if not mode:
            # Fall back to build mode
            mode = config.get_mode("build")

        if not mode:
            raise ValueError(f"Mode '{mode_name}' not found for vertical '{vertical_name}'")

        return mode

    def list_modes(self, vertical_name: str) -> list[str]:
        """List available modes for a vertical.

        Args:
            vertical_name: Name of the vertical

        Returns:
            List of mode names

        Example:
            modes = registry.list_modes("coding")
            # ["build", "plan", "explore"]
        """
        config = self.load_config(vertical_name)
        return config.list_modes()

    def reload(self, vertical_name: Optional[str] = None) -> None:
        """Reload configuration from YAML.

        Args:
            vertical_name: Specific vertical to reload, or None for all

        Example:
            # Reload specific vertical
            registry.reload("coding")

            # Reload all
            registry.reload()
        """
        if vertical_name:
            if vertical_name in self._configs:
                del self._configs[vertical_name]
            self.load_config(vertical_name)
            logger.info(f"ModeConfigRegistry: Reloaded '{vertical_name}'")
        else:
            # Reload all
            for name in list(self._configs.keys()):
                self.reload(name)
            logger.info("ModeConfigRegistry: Reloaded all configurations")


# Factory function for dependency injection
def create_mode_config_registry() -> ModeConfigRegistry:
    """Create a ModeConfigRegistry instance for DI registration.

    Returns:
        ModeConfigRegistry instance

    Example:
        from victor.core.container import ServiceContainer
        container = ServiceContainer()

        container.register(
            ModeConfigRegistry,
            lambda c: create_mode_config_registry(),
            lifetime=ServiceLifetime.SINGLETON
        )
    """
    return ModeConfigRegistry.get_instance()


__all__ = [
    "ExplorationLevel",
    "EditPermission",
    "AgentMode",
    "VerticalModeConfig",
    "ModeConfigRegistry",
    "create_mode_config_registry",
]
