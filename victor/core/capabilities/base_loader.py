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

"""Base capability loader for vertical capabilities.

This module provides a centralized capability loading system with YAML-based
configuration and registry support.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class CapabilityType(str):
    """Capability type identifiers.

    Valid types:
        TOOL: Tool capability
        WORKFLOW: Workflow capability
        MIDDLEWARE: Middleware capability
        VALIDATOR: Validator capability
        OBSERVER: Observer capability
    """

    TOOL = "tool"
    WORKFLOW = "workflow"
    MIDDLEWARE = "middleware"
    VALIDATOR = "validator"
    OBSERVER = "observer"


class Capability:
    """Canonical capability definition.

    Attributes:
        name: Unique capability identifier
        type: Capability type (tool, workflow, etc.)
        description: Human-readable description
        version: Capability version
        enabled: Whether capability is active
        dependencies: List of capability dependencies
        handler: Python import path for handler
        config: Additional configuration
        metadata: Additional metadata
    """

    def __init__(
        self,
        name: str,
        type: str,
        description: str = "",
        version: str = "0.5.0",
        enabled: bool = True,
        dependencies: list[str] | None = None,
        handler: str | None = None,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.name = name
        self.type = type
        self.description = description
        self.version = version
        self.enabled = enabled
        self.dependencies = dependencies or []
        self.handler = handler
        self.config = config or {}
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"Capability(name={self.name}, type={self.type}, enabled={self.enabled})"


class CapabilitySet:
    """Set of capabilities for a vertical.

    Attributes:
        vertical_name: Name of the vertical
        capabilities: Dictionary of capabilities
        handlers: Dictionary of handler import paths
    """

    def __init__(
        self,
        vertical_name: str,
        capabilities: dict[str, Capability] | None = None,
        handlers: dict[str, str] | None = None,
    ):
        self.vertical_name = vertical_name
        self.capabilities = capabilities or {}
        self.handlers = handlers or {}

    def get_capability(self, name: str) -> Capability | None:
        """Get capability by name.

        Args:
            name: Capability name

        Returns:
            Capability instance or None
        """
        return self.capabilities.get(name)

    def list_capabilities(self, capability_type: str | None = None) -> list[str]:
        """List capabilities by type.

        Args:
            capability_type: Filter by type (optional)

        Returns:
            List of capability names
        """
        if capability_type:
            return [
                name
                for name, cap in self.capabilities.items()
                if cap.type == capability_type and cap.enabled
            ]
        return [name for name, cap in self.capabilities.items() if cap.enabled]

    def get_handler(self, handler_name: str) -> str | None:
        """Get handler import path.

        Args:
            handler_name: Handler name

        Returns:
            Import path or None
        """
        return self.handlers.get(handler_name)


class CapabilityLoader:
    """Canonical capability loader with registry support.

    This loader provides centralized capability management with YAML-based
    configuration and automatic fallback to defaults.

    Example:
        loader = CapabilityLoader.from_vertical("coding")
        capabilities = loader.list_capabilities(CapabilityType.TOOL)
        review_cap = loader.get_capability("code_review")
    """

    _instances: dict[str, "CapabilityLoader"] = {}

    def __init__(self, config_dir: Path | None = None):
        """Initialize capability loader.

        Args:
            config_dir: Directory containing capability YAML files
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent.parent / "config" / "capabilities"
        self._config_dir = Path(config_dir)
        self._capability_sets: dict[str, CapabilitySet] = {}

    @classmethod
    def from_vertical(cls, vertical_name: str) -> "CapabilityLoader":
        """Create loader for specific vertical.

        Args:
            vertical_name: Name of the vertical

        Returns:
            CapabilityLoader instance

        Example:
            loader = CapabilityLoader.from_vertical("coding")
            capabilities = loader.list_capabilities()
        """
        if vertical_name not in cls._instances:
            loader = cls()
            loader.load_capabilities(vertical_name)
            cls._instances[vertical_name] = loader
        return cls._instances[vertical_name]

    @classmethod
    def get_instance(cls) -> "CapabilityLoader":
        """Get global capability loader instance.

        Returns:
            CapabilityLoader singleton
        """
        if "global" not in cls._instances:
            cls._instances["global"] = cls()
        return cls._instances["global"]

    def load_capabilities(self, vertical_name: str) -> CapabilitySet:
        """Load capabilities from YAML or generate defaults.

        Args:
            vertical_name: Name of the vertical

        Returns:
            CapabilitySet instance
        """
        if vertical_name in self._capability_sets:
            return self._capability_sets[vertical_name]

        # Try YAML first
        config_file = self._config_dir / f"{vertical_name}_capabilities.yaml"
        if config_file.exists():
            capability_set = self._load_from_yaml(config_file, vertical_name)
        else:
            capability_set = self._generate_default_capabilities(vertical_name)

        self._capability_sets[vertical_name] = capability_set
        return capability_set

    def _load_from_yaml(self, path: Path, vertical_name: str) -> CapabilitySet:
        """Load capabilities from YAML file.

        Args:
            path: Path to YAML file
            vertical_name: Name of the vertical

        Returns:
            CapabilitySet instance
        """
        try:
            with open(path) as f:
                data = yaml.safe_load(f)

            capabilities = {}
            for name, cap_data in data.get("capabilities", {}).items():
                capabilities[name] = Capability(name=name, **cap_data)

            handlers = data.get("handlers", {})

            logger.debug(f"Loaded {len(capabilities)} capabilities from {path}")
            return CapabilitySet(
                vertical_name=vertical_name,
                capabilities=capabilities,
                handlers=handlers,
            )
        except Exception as e:
            logger.error(f"Failed to load capabilities from {path}: {e}")
            return self._generate_default_capabilities(vertical_name)

    def _generate_default_capabilities(self, vertical_name: str) -> CapabilitySet:
        """Generate default capability set.

        Args:
            vertical_name: Name of the vertical

        Returns:
            CapabilitySet instance
        """
        logger.debug(f"Generating default capabilities for {vertical_name}")
        return CapabilitySet(vertical_name=vertical_name)

    def get_capability(self, vertical_name: str, capability_name: str) -> Capability | None:
        """Get specific capability.

        Args:
            vertical_name: Name of the vertical
            capability_name: Name of the capability

        Returns:
            Capability instance or None
        """
        capability_set = self.load_capabilities(vertical_name)
        return capability_set.get_capability(capability_name)

    def list_capabilities(
        self, vertical_name: str, capability_type: str | None = None
    ) -> list[str]:
        """List capabilities for a vertical.

        Args:
            vertical_name: Name of the vertical
            capability_type: Filter by type (optional)

        Returns:
            List of capability names
        """
        capability_set = self.load_capabilities(vertical_name)
        return capability_set.list_capabilities(capability_type)

    def get_capability_set(self, vertical_name: str) -> CapabilitySet:
        """Get full capability set for a vertical.

        Args:
            vertical_name: Name of the vertical

        Returns:
            CapabilitySet instance
        """
        return self.load_capabilities(vertical_name)

    def invalidate_cache(self, vertical_name: str | None = None) -> int:
        """Invalidate capability cache.

        Args:
            vertical_name: Specific vertical to invalidate, or None for all

        Returns:
            Number of entries invalidated
        """
        if vertical_name:
            if vertical_name in self._capability_sets:
                del self._capability_sets[vertical_name]
                if vertical_name in self.__class__._instances:
                    del self.__class__._instances[vertical_name]
                return 1
            return 0
        else:
            count = len(self._capability_sets)
            self._capability_sets.clear()
            self.__class__._instances.clear()
            return count


__all__ = [
    "CapabilityType",
    "Capability",
    "CapabilitySet",
    "CapabilityLoader",
]
