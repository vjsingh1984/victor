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

"""Type definitions for the capability registry system (Phase 5.1).

This module provides the core types for YAML-first capability definitions:
- CapabilityType enum for capability classification
- CapabilityDefinition dataclass for declarative definitions
- ConfigSchema type alias for JSON Schema definitions

Design Principles:
- YAML-first: All definitions serializable to/from YAML
- Schema validation: Config validated against JSON Schema
- Backward compatible: Works with existing framework capabilities
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CapabilityType(str, Enum):
    """Capability type classification.

    Defines the categories of capabilities that can be registered.
    Maps to framework CapabilityType for backward compatibility.

    Values:
        TOOL: Tool-based capability (code analysis, refactoring, etc.)
        MODE: Configuration mode (code style, test settings, etc.)
        SAFETY: Safety check capability (git safety, PII detection, etc.)
        PROMPT: Prompt enhancement capability
        RL: Reinforcement learning capability
    """

    TOOL = "tool"
    MODE = "mode"
    SAFETY = "safety"
    PROMPT = "prompt"
    RL = "rl"

    @classmethod
    def from_string(cls, value: str) -> "CapabilityType":
        """Create CapabilityType from string value.

        Args:
            value: String representation (case-insensitive)

        Returns:
            CapabilityType enum value

        Raises:
            ValueError: If value is not a valid capability type
        """
        normalized = value.lower().strip()
        for member in cls:
            if member.value == normalized:
                return member
        valid_types = [m.value for m in cls]
        raise ValueError(f"Invalid capability type: '{value}'. Valid types: {valid_types}")


# Type alias for JSON Schema definitions
ConfigSchema = dict[str, Any]


@dataclass
class CapabilityDefinition:
    """Declarative capability definition.

    Provides a YAML-serializable definition for capabilities that can be:
    - Loaded from YAML files
    - Discovered via entry points
    - Validated against JSON Schema

    Attributes:
        name: Unique capability identifier
        capability_type: Type classification (TOOL, MODE, SAFETY, etc.)
        description: Human-readable description
        config_schema: JSON Schema for configuration validation
        default_config: Default configuration values
        dependencies: List of capability dependencies
        tags: Discovery and filtering tags
        version: Capability version (semver)
        enabled: Whether capability is active by default

    Example:
        definition = CapabilityDefinition(
            name="git_safety",
            capability_type=CapabilityType.SAFETY,
            description="Git operation safety checks",
            default_config={
                "block_force_push": True,
                "block_main_branch_delete": True,
            },
            tags=["safety", "git", "version-control"],
        )

        # Serialize to YAML
        yaml_dict = definition.to_yaml_dict()

        # Load from YAML
        definition = CapabilityDefinition.from_yaml_dict(yaml_dict)
    """

    name: str
    capability_type: CapabilityType
    description: str
    config_schema: ConfigSchema = field(default_factory=dict)
    default_config: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    version: str = "0.5.0"
    enabled: bool = True

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate configuration against JSON Schema.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if config is valid, False otherwise

        Note:
            If no schema is defined, always returns True.
        """
        if not self.config_schema:
            return True

        try:
            import jsonschema

            jsonschema.validate(instance=config, schema=self.config_schema)
            return True
        except ImportError:
            # jsonschema not installed, skip validation
            logger.warning("jsonschema not installed, skipping config validation")
            return True
        except jsonschema.ValidationError as e:
            logger.debug(f"Config validation failed for {self.name}: {e.message}")
            return False

    def get_validation_errors(self, config: dict[str, Any]) -> list[str]:
        """Get detailed validation errors for configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        if not self.config_schema:
            return []

        try:
            import jsonschema

            validator = jsonschema.Draft7Validator(self.config_schema)
            return [error.message for error in validator.iter_errors(config)]
        except ImportError:
            return []

    def to_yaml_dict(self) -> dict[str, Any]:
        """Convert to YAML-serializable dictionary.

        Returns:
            Dictionary suitable for YAML serialization
        """
        result: dict[str, Any] = {
            "name": self.name,
            "capability_type": self.capability_type.value,
            "description": self.description,
        }

        # Only include non-empty optional fields
        if self.config_schema:
            result["config_schema"] = self.config_schema
        if self.default_config:
            result["default_config"] = self.default_config
        if self.dependencies:
            result["dependencies"] = self.dependencies
        if self.tags:
            result["tags"] = self.tags
        if self.version != "0.5.0":
            result["version"] = self.version
        if not self.enabled:
            result["enabled"] = self.enabled

        return result

    @classmethod
    def from_yaml_dict(cls, data: dict[str, Any]) -> "CapabilityDefinition":
        """Create CapabilityDefinition from YAML dictionary.

        Args:
            data: Dictionary loaded from YAML

        Returns:
            CapabilityDefinition instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Required fields
        if "name" not in data:
            raise ValueError("Missing required field: 'name'")
        if "capability_type" not in data:
            raise ValueError("Missing required field: 'capability_type'")

        # Parse capability_type
        cap_type = data["capability_type"]
        if isinstance(cap_type, str):
            cap_type = CapabilityType.from_string(cap_type)
        elif isinstance(cap_type, CapabilityType):
            pass
        else:
            raise ValueError(f"Invalid capability_type: {cap_type}")

        return cls(
            name=data["name"],
            capability_type=cap_type,
            description=data.get("description", ""),
            config_schema=data.get("config_schema", {}),
            default_config=data.get("default_config", {}),
            dependencies=data.get("dependencies", []),
            tags=data.get("tags", []),
            version=data.get("version", "1.0"),
            enabled=data.get("enabled", True),
        )

    def merge_config(
        self, stored_config: dict[str, Any], override_config: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Merge stored and override configs with defaults.

        Order of precedence (highest to lowest):
        1. override_config (explicit overrides)
        2. stored_config (previously stored values)
        3. default_config (definition defaults)

        Args:
            stored_config: Previously stored configuration
            override_config: Explicit overrides (optional)

        Returns:
            Merged configuration dictionary
        """
        result = self.default_config.copy()
        result.update(stored_config)
        if override_config:
            result.update(override_config)
        return result


__all__ = [
    "CapabilityType",
    "CapabilityDefinition",
    "ConfigSchema",
]
