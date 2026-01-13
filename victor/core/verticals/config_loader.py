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

"""Vertical configuration loader for YAML-based vertical definitions.

This module implements the VerticalConfigLoader which loads vertical
configurations from YAML files, replacing the 15+ get_* methods with
a single declarative configuration.

Design Patterns:
    - Loader Pattern: Load configuration from YAML files
    - Validation Pattern: Validate required fields and data types
    - Escape Hatch Pattern: Allow dynamic customization via class methods
    - SRP: Focused only on configuration loading and validation

Phase 2, Work Stream 2.1: Declarative Vertical Configuration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from types import ModuleType

logger = logging.getLogger(__name__)


@dataclass
class VerticalYAMLConfig:
    """Configuration loaded from YAML file.

    Attributes:
        name: Vertical identifier
        version: Configuration version
        description: Human-readable description
        tools: List of tool names to enable
        system_prompt_config: System prompt configuration
        stages: Stage definitions for conversation flow
        middleware: Middleware configuration list
        safety_extension: Safety extension configuration
        workflows: Workflow configuration
        provider_hints: Provider selection hints
        metadata: Additional metadata
    """

    name: str
    version: str
    description: str
    tools: List[str]
    system_prompt_config: Dict[str, Any]
    stages: Dict[str, Dict[str, Any]]
    middleware: List[Dict[str, Any]]
    safety_extension: Optional[Dict[str, Any]]
    workflows: Dict[str, Any]
    provider_hints: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "tools": self.tools,
            "system_prompt_config": self.system_prompt_config,
            "stages": self.stages,
            "middleware": self.middleware,
            "safety_extension": self.safety_extension,
            "workflows": self.workflows,
            "provider_hints": self.provider_hints,
            "metadata": self.metadata,
        }


class VerticalConfigLoader:
    """Load vertical configuration from YAML files.

    This loader reads YAML-based vertical configurations and provides
    validation and escape hatch mechanisms for dynamic customization.

    Example:
        loader = VerticalConfigLoader()

        # Load from YAML file
        config = loader.load_vertical_config(
            "coding",
            Path("/path/to/coding/vertical.yaml")
        )

        # With escape hatch
        config = loader.load_vertical_config(
            "custom",
            Path("/path/to/custom/vertical.yaml"),
            escape_hatch_class=CustomVertical
        )
    """

    # Required YAML fields
    REQUIRED_FIELDS = ["name", "version", "description", "tools"]

    def __init__(self) -> None:
        """Initialize the configuration loader."""
        self._cache: Dict[str, VerticalYAMLConfig] = {}

    def load_vertical_config(
        self,
        vertical_name: str,
        config_path: Path,
        escape_hatch_class: Optional[Any] = None,
    ) -> VerticalYAMLConfig:
        """Load vertical configuration from YAML file.

        Args:
            vertical_name: Name of the vertical
            config_path: Path to YAML configuration file
            escape_hatch_class: Optional class with escape_hatch_tools method

        Returns:
            VerticalYAMLConfig instance

        Raises:
            ValueError: If YAML parsing fails or required fields are missing
            FileNotFoundError: If config file doesn't exist
        """
        # Check cache first
        cache_key = f"{vertical_name}:{config_path}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Check file exists
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load YAML content
        try:
            with open(config_path, "r") as f:
                yaml_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML from {config_path}: {e}")

        # Validate required fields
        self._validate_required_fields(yaml_data, config_path)

        # Extract configuration
        config = self._extract_config(yaml_data)

        # Apply escape hatch if provided
        if escape_hatch_class and hasattr(escape_hatch_class, "escape_hatch_tools"):
            config.tools = escape_hatch_class.escape_hatch_tools(config.tools)

        # Cache the result
        self._cache[cache_key] = config

        logger.info(f"Loaded vertical configuration for '{vertical_name}' from {config_path}")

        return config

    def _validate_required_fields(
        self, yaml_data: Dict[str, Any], config_path: Path
    ) -> None:
        """Validate that required fields are present in YAML data.

        Args:
            yaml_data: Parsed YAML data
            config_path: Path to configuration file (for error messages)

        Raises:
            ValueError: If required fields are missing
        """
        missing_fields = []
        for field in self.REQUIRED_FIELDS:
            if field not in yaml_data:
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(
                f"Missing required fields in {config_path}: {', '.join(missing_fields)}"
            )

    def _extract_config(self, yaml_data: Dict[str, Any]) -> VerticalYAMLConfig:
        """Extract configuration from parsed YAML data.

        Args:
            yaml_data: Parsed YAML data

        Returns:
            VerticalYAMLConfig instance
        """
        # Extract system prompt configuration
        system_prompt_config = yaml_data.get("system_prompt", {})
        if isinstance(system_prompt_config, str):
            system_prompt_config = {"source": "inline", "content": system_prompt_config}

        # Extract stages
        stages = yaml_data.get("stages", {})

        # Extract middleware
        middleware = yaml_data.get("middleware", [])

        # Extract safety extension
        safety_extension = yaml_data.get("safety")

        # Extract workflows
        workflows = yaml_data.get("workflows", {})

        # Extract provider hints
        provider_hints = yaml_data.get("provider_hints", {})

        # Extract metadata
        metadata = yaml_data.get("metadata", {})

        return VerticalYAMLConfig(
            name=yaml_data["name"],
            version=yaml_data["version"],
            description=yaml_data["description"],
            tools=yaml_data["tools"],
            system_prompt_config=system_prompt_config,
            stages=stages,
            middleware=middleware,
            safety_extension=safety_extension,
            workflows=workflows,
            provider_hints=provider_hints,
            metadata=metadata,
        )

    def clear_cache(self, vertical_name: Optional[str] = None) -> None:
        """Clear configuration cache.

        Args:
            vertical_name: Optional vertical name to clear.
                         If None, clears all cached configurations.
        """
        if vertical_name:
            keys_to_remove = [k for k in self._cache if k.startswith(f"{vertical_name}:")]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()

        logger.debug(f"Cleared configuration cache for '{vertical_name or 'all'}'")


__all__ = [
    "VerticalConfigLoader",
    "VerticalYAMLConfig",
]
