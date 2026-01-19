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

"""Base classes for dynamic capability registration.

This module provides the foundation for Open/Closed Principle (OCP) compliance
in Victor's capability system. External packages can register capabilities
via entry points without modifying core code.

Design Pattern: Abstract Factory + Entry Point Discovery
- CapabilityBase: Abstract interface for capability providers
- CapabilitySpec: Dataclass for capability metadata
- Entry points: Dynamic discovery at import time

Usage (External Package):
    # In your package's capability module
    from victor.agent.capabilities.base import CapabilityBase, CapabilitySpec

    class MyCustomCapability(CapabilityBase):
        @classmethod
        def get_spec(cls) -> CapabilitySpec:
            return CapabilitySpec(
                name="my_custom",
                method_name="set_my_custom",
                version="1.0",
                description="My custom capability"
            )

    # In your package's pyproject.toml
    # [project.entry-points."victor.capabilities"]
    # my_custom = "my_package.capabilities:MyCustomCapability"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CapabilitySpec:
    """Capability specification.

    Defines the metadata for a capability, including how to invoke it.
    This is the canonical specification that all capabilities must provide.

    Attributes:
        name: Unique capability identifier (e.g., "enabled_tools")
        method_name: Method name to call (e.g., "set_enabled_tools")
        version: Semantic version (default "1.0")
        description: Human-readable description

    Example:
        spec = CapabilitySpec(
            name="custom_analytics",
            method_name="set_custom_analytics",
            version="1.0",
            description="Custom analytics integration"
        )
    """

    name: str
    method_name: str
    version: str = "1.0"
    description: str = ""

    def __post_init__(self):
        """Validate capability specification."""
        if not self.name:
            raise ValueError("Capability name cannot be empty")
        if not self.method_name:
            raise ValueError("Capability method_name cannot be empty")
        if not self._is_valid_version(self.version):
            raise ValueError(
                f"Invalid version '{self.version}'. Expected format: 'MAJOR.MINOR' "
                "(e.g., '1.0', '2.1')"
            )

    @staticmethod
    def _is_valid_version(version: str) -> bool:
        """Validate version string format.

        Args:
            version: Version string to validate

        Returns:
            True if version is valid MAJOR.MINOR format
        """
        try:
            parts = version.split(".")
            if len(parts) != 2:
                return False
            major, minor = int(parts[0]), int(parts[1])
            return major >= 0 and minor >= 0
        except (ValueError, AttributeError):
            return False


class CapabilityBase(ABC):
    """Base class for capabilities.

    All capabilities must inherit from this class and implement get_spec().
    This enables dynamic discovery via entry points.

    Design Pattern: Template Method
    - Subclasses implement get_spec() to provide metadata
    - Registry discovers and registers capabilities at import time

    Example:
        class EnabledToolsCapability(CapabilityBase):
            @classmethod
            def get_spec(cls) -> CapabilitySpec:
                return CapabilitySpec(
                    name="enabled_tools",
                    method_name="set_enabled_tools",
                    version="1.0",
                    description="Set enabled tools for the agent"
                )
    """

    @classmethod
    @abstractmethod
    def get_spec(cls) -> CapabilitySpec:
        """Get capability specification.

        All subclasses must implement this method to provide their metadata.

        Returns:
            CapabilitySpec with capability metadata

        Raises:
            NotImplementedError: If subclass doesn't implement
        """
        raise NotImplementedError(f"{cls.__name__} must implement get_spec()")


__all__ = ["CapabilityBase", "CapabilitySpec"]
