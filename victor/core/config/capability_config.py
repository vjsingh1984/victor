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

"""Data-Driven Capability Configuration System.

This module provides a centralized, YAML-based configuration system for
vertical capabilities, complementing the existing BaseCapabilityProvider
pattern while adding data-driven flexibility.

Design Patterns:
    - Registry: Singleton CapabilityConfigRegistry for config access
    - Factory: Generate defaults when YAML not found
    - Data-Driven: YAML files override code defaults
    - Compatibility: Works with existing capability providers

Use Cases:
    - Capability metadata discovery
    - Dynamic capability registration
    - Default configuration management
    - Capability dependency resolution

Example:
    from victor.core.config import CapabilityConfigRegistry

    registry = CapabilityConfigRegistry.get_instance()
    capabilities = registry.list_capabilities("coding")
    for cap in capabilities:
        print(f"{cap.name}: {cap.description}")
"""

from __future__ import annotations

import importlib
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from victor.framework.protocols import CapabilityType

logger = logging.getLogger(__name__)


@dataclass
class CapabilityConfig:
    """Configuration for a single capability.

    Loaded from YAML files in victor/config/capabilities/. This provides
    a data-driven way to define capability metadata and configuration.

    Attributes:
        name: Unique capability identifier
        capability_type: Type of capability (safety, mode, tool)
        version: Semantic version string
        description: Human-readable description
        tags: List of tags for categorization
        handler_module: Python module containing handler function
        handler_function: Function name to configure the capability
        getter_function: Optional function name to get current config
        default_config: Default configuration parameters
        dependencies: List of capability names this depends on
    """

    name: str
    capability_type: CapabilityType
    version: str
    description: str
    tags: List[str] = field(default_factory=list)
    handler_module: str = ""
    handler_function: str = ""
    getter_function: Optional[str] = None
    default_config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)

    def import_handler(self) -> Optional[Callable]:
        """Import the handler function from its module.

        Returns:
            Handler function or None if import fails
        """
        if not self.handler_module or not self.handler_function:
            return None

        try:
            module = importlib.import_module(self.handler_module)
            return getattr(module, self.handler_function, None)
        except Exception as e:
            logger.warning(f"Failed to import handler {self.handler_module}.{self.handler_function}: {e}")
            return None

    def import_getter(self) -> Optional[Callable]:
        """Import the getter function from its module.

        Returns:
            Getter function or None if not available
        """
        if not self.handler_module or not self.getter_function:
            return None

        try:
            module = importlib.import_module(self.handler_module)
            return getattr(module, self.getter_function, None)
        except Exception as e:
            logger.warning(f"Failed to import getter {self.handler_module}.{self.getter_function}: {e}")
            return None

    def get_default_config(self) -> Dict[str, Any]:
        """Get the default configuration for this capability.

        Returns:
            Default configuration dictionary
        """
        return self.default_config.copy()


@dataclass
class VerticalCapabilities:
    """Aggregated capabilities for a vertical.

    Attributes:
        vertical_name: Name of the vertical
        capabilities: List of capability configurations
    """

    vertical_name: str
    capabilities: List[CapabilityConfig] = field(default_factory=list)

    def get_capability(self, name: str) -> Optional[CapabilityConfig]:
        """Get a specific capability by name.

        Args:
            name: Capability name

        Returns:
            CapabilityConfig or None if not found
        """
        for cap in self.capabilities:
            if cap.name == name:
                return cap
        return None

    def list_capabilities(self) -> List[str]:
        """List all capability names.

        Returns:
            List of capability names
        """
        return [cap.name for cap in self.capabilities]

    def filter_by_type(self, capability_type: CapabilityType) -> List[CapabilityConfig]:
        """Filter capabilities by type.

        Args:
            capability_type: Type to filter by

        Returns:
            List of matching capabilities
        """
        return [cap for cap in self.capabilities if cap.capability_type == capability_type]

    def filter_by_tag(self, tag: str) -> List[CapabilityConfig]:
        """Filter capabilities by tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of matching capabilities
        """
        return [cap for cap in self.capabilities if tag in cap.tags]


class CapabilityConfigRegistry:
    """Registry for capability configurations across all verticals.

    Singleton pattern with thread-safe lazy loading. Uses the
    Universal Registry for caching configurations.

    Usage:
        registry = CapabilityConfigRegistry.get_instance()
        caps = registry.get_capabilities("coding")
        for cap in caps:
            print(f"{cap.name}: {cap.description}")
    """

    _instance: Optional["CapabilityConfigRegistry"] = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls) -> "CapabilityConfigRegistry":
        """Create or return singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the registry (only once)."""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._config_dir = Path(__file__).parent.parent.parent / "config" / "capabilities"
                    self._cache: Dict[str, VerticalCapabilities] = {}
                    self._initialized = True
                    logger.debug(f"CapabilityConfigRegistry initialized with config dir: {self._config_dir}")

    @classmethod
    def get_instance(cls) -> "CapabilityConfigRegistry":
        """Get the singleton registry instance."""
        return cls()

    def get_capabilities(self, vertical: str) -> VerticalCapabilities:
        """Get capabilities for a vertical.

        Loads from YAML file with fallback to empty capabilities.

        Args:
            vertical: Vertical name (e.g., "coding", "research")

        Returns:
            VerticalCapabilities instance

        Raises:
            ValueError: If vertical is empty or invalid
        """
        if not vertical:
            raise ValueError("Vertical name cannot be empty")

        vertical = vertical.lower()

        # Check cache
        if vertical in self._cache:
            return self._cache[vertical]

        # Load from YAML
        caps = self._load_from_yaml(vertical)
        self._cache[vertical] = caps
        return caps

    def _load_from_yaml(self, vertical: str) -> VerticalCapabilities:
        """Load configuration from YAML file.

        Args:
            vertical: Vertical name

        Returns:
            VerticalCapabilities with loaded configurations
        """
        yaml_file = self._config_dir / f"{vertical}_capabilities.yaml"

        if not yaml_file.exists():
            logger.debug(f"No capability config file found for {vertical}: {yaml_file}")
            return VerticalCapabilities(vertical_name=vertical)

        try:
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f)

            if not data or data.get("vertical_name") != vertical:
                logger.warning(f"Invalid capability config file for {vertical}")
                return VerticalCapabilities(vertical_name=vertical)

            capabilities = []
            for cap_data in data.get("capabilities", []):
                try:
                    cap_type = CapabilityType(cap_data.get("capability_type", "tool"))
                except ValueError:
                    logger.warning(f"Unknown capability type: {cap_data.get('capability_type')}")
                    cap_type = CapabilityType.TOOL

                cap = CapabilityConfig(
                    name=cap_data.get("name", ""),
                    capability_type=cap_type,
                    version=cap_data.get("version", "1.0"),
                    description=cap_data.get("description", ""),
                    tags=cap_data.get("tags", []),
                    handler_module=cap_data.get("handler_module", ""),
                    handler_function=cap_data.get("handler_function", ""),
                    getter_function=cap_data.get("getter_function"),
                    default_config=cap_data.get("default_config", {}),
                    dependencies=cap_data.get("dependencies", []),
                )
                capabilities.append(cap)

            logger.debug(f"Loaded {len(capabilities)} capabilities for {vertical} from {yaml_file}")
            return VerticalCapabilities(vertical_name=vertical, capabilities=capabilities)

        except Exception as e:
            logger.error(f"Failed to load capability config for {vertical}: {e}")
            return VerticalCapabilities(vertical_name=vertical)

    def list_verticals(self) -> List[str]:
        """List all available vertical configurations.

        Returns:
            List of vertical names with available configs
        """
        if not self._config_dir.exists():
            return []

        verticals = []
        for yaml_file in self._config_dir.glob("*_capabilities.yaml"):
            vertical = yaml_file.stem.replace("_capabilities", "")
            verticals.append(vertical)

        return sorted(verticals)

    def invalidate(self, vertical: Optional[str] = None) -> None:
        """Invalidate cached configuration(s).

        Args:
            vertical: Specific vertical to invalidate, or None for all
        """
        if vertical:
            self._cache.pop(vertical.lower(), None)
        else:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dict with cache size, available verticals, etc.
        """
        total_caps = sum(len(vcaps.capabilities) for vcaps in self._cache.values())
        return {
            "cache_size": len(self._cache),
            "cached_verticals": list(self._cache.keys()),
            "available_verticals": self.list_verticals(),
            "total_capabilities": total_caps,
            "config_dir": str(self._config_dir),
        }


__all__ = [
    "CapabilityConfig",
    "VerticalCapabilities",
    "CapabilityConfigRegistry",
]
