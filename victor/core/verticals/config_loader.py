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
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

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
        prompt_contributor: Prompt contributor configuration
        mode_config: Mode configuration provider
        tool_dependencies: Tool dependency configuration
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
    prompt_contributor: Optional[Dict[str, Any]]
    mode_config: Optional[Dict[str, Any]]
    tool_dependencies: Optional[Dict[str, Any]]
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
            "prompt_contributor": self.prompt_contributor,
            "mode_config": self.mode_config,
            "tool_dependencies": self.tool_dependencies,
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

    def load_from_yaml(
        self,
        yaml_path: Path,
        vertical_name: Optional[str] = None,
    ) -> Optional[Any]:
        """Load vertical config from YAML file (convenience method).

        This is a convenience wrapper around load_vertical_config that
        automatically infers the vertical name from the path if not provided.

        Args:
            yaml_path: Path to YAML configuration file
            vertical_name: Optional vertical name (inferred from path if None)

        Returns:
            VerticalConfig instance or None if loading fails

        Example:
            loader = VerticalConfigLoader()
            config = loader.load_from_yaml(Path("/path/to/vertical.yaml"))
        """
        if vertical_name is None:
            # Infer vertical name from path
            vertical_name = yaml_path.parent.name

        try:
            yaml_config = self.load_vertical_config(vertical_name, yaml_path)

            # Convert VerticalYAMLConfig to VerticalConfig
            from victor.core.verticals.base import VerticalConfig
            from victor.framework.tools import ToolSet

            # Build tool set
            tools = ToolSet.from_tools(yaml_config.tools)

            # Build stages from YAML
            from victor.core.vertical_types import StageDefinition

            stages = {}
            for stage_name, stage_data in yaml_config.stages.items():
                stages[stage_name] = StageDefinition(
                    name=stage_name,
                    description=stage_data.get("description", ""),
                    keywords=stage_data.get("keywords", []),
                    next_stages=set(stage_data.get("next_stages", [])),
                )

            # Create VerticalConfig
            config = VerticalConfig(
                tools=tools,
                system_prompt=self._extract_prompt_text(yaml_config.system_prompt_config),
                stages=stages,
                provider_hints=yaml_config.provider_hints,
                metadata=yaml_config.metadata,
            )

            return config

        except Exception as e:
            logger.warning(f"Failed to load vertical config from {yaml_path}: {e}")
            return None

    def _extract_prompt_text(
        self,
        prompt_config: Dict[str, Any],
    ) -> str:
        """Extract prompt text from prompt configuration.

        Args:
            prompt_config: Prompt configuration from YAML

        Returns:
            Prompt text string

        Raises:
            ValueError: If prompt configuration is invalid
        """
        source = prompt_config.get("source", "inline")

        if source == "inline":
            # Prompt text is directly in YAML
            text = prompt_config.get("text", "")
            if not text:
                raise ValueError("Inline prompt requires 'text' field")
            return str(text)  # type: ignore[return-value]

        elif source == "file":
            # Load prompt from file
            file_path = prompt_config.get("path", "")
            if not file_path:
                raise ValueError("File prompt requires 'path' field")

            from pathlib import Path

            path = Path(file_path)
            if not path.exists():
                raise ValueError(f"Prompt file not found: {file_path}")

            with open(path, "r") as f:
                return f.read()

        else:
            raise ValueError(f"Unknown prompt source: {source}")

    def _validate_required_fields(self, yaml_data: Dict[str, Any], config_path: Path) -> None:
        """Validate that required fields are present in YAML data.

        Supports both legacy and structured YAML formats.

        Args:
            yaml_data: Parsed YAML data
            config_path: Path to configuration file (for error messages)

        Raises:
            ValueError: If required fields are missing
        """
        # Detect format type
        has_metadata = "metadata" in yaml_data
        has_core = "core" in yaml_data

        if has_metadata and has_core:
            # Validate structured format
            missing_fields = []
            metadata = yaml_data.get("metadata", {})
            core = yaml_data.get("core", {})

            # Check required metadata fields
            if "name" not in metadata and "name" not in yaml_data:
                missing_fields.append("metadata.name")
            if "version" not in metadata and "version" not in yaml_data:
                missing_fields.append("metadata.version")
            if "description" not in metadata and "description" not in yaml_data:
                missing_fields.append("metadata.description")

            # Check required core fields
            if "tools" not in core:
                missing_fields.append("core.tools")

            if missing_fields:
                raise ValueError(
                    f"Missing required fields in {config_path}: {', '.join(missing_fields)}"
                )
        else:
            # Validate legacy format
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

        Supports both legacy and new YAML structure:
        - Legacy: name, version, description, tools, system_prompt, stages at top level
        - New: metadata.* and core.* structure

        Args:
            yaml_data: Parsed YAML data

        Returns:
            VerticalYAMLConfig instance
        """
        # Detect structure type
        has_metadata = "metadata" in yaml_data
        has_core = "core" in yaml_data

        if has_metadata and has_core:
            # New structured format
            return self._extract_structured_config(yaml_data)
        else:
            # Legacy flat format
            return self._extract_legacy_config(yaml_data)

    def _extract_structured_config(self, yaml_data: Dict[str, Any]) -> VerticalYAMLConfig:
        """Extract configuration from new structured YAML format.

        Args:
            yaml_data: Parsed YAML data with metadata/core structure

        Returns:
            VerticalYAMLConfig instance
        """
        metadata = yaml_data.get("metadata", {})
        core = yaml_data.get("core", {})
        extensions = yaml_data.get("extensions", {})
        provider = yaml_data.get("provider", {})

        # Extract metadata
        name = metadata.get("name", yaml_data.get("name", ""))
        version = metadata.get("version", yaml_data.get("version", "0.5.0"))
        description = metadata.get("description", yaml_data.get("description", ""))

        # Extract tools from core.tools.list
        tools_section = core.get("tools", {})
        if isinstance(tools_section, dict):
            tools = tools_section.get("list", [])
        elif isinstance(tools_section, list):
            tools = tools_section
        else:
            tools = []

        # Extract system prompt
        system_prompt_config = core.get("system_prompt", {})
        if isinstance(system_prompt_config, str):
            system_prompt_config = {"source": "inline", "text": system_prompt_config}

        # Extract stages
        stages = core.get("stages", {})

        # Extract middleware from extensions
        middleware = extensions.get("middleware", [])

        # Extract extensions
        safety_extension = extensions.get("safety")
        prompt_contributor = extensions.get("prompt_contributor")
        mode_config = extensions.get("mode_config")
        tool_dependencies = extensions.get("tool_dependencies")

        # Extract provider hints
        provider_hints = provider.get("hints", {})

        # Extract workflows
        workflows = extensions.get("workflows", {})

        # Extract additional metadata
        additional_metadata = metadata.copy()
        # Add top-level metadata if not in metadata section
        for key in ["name", "version", "description"]:
            additional_metadata.pop(key, None)

        return VerticalYAMLConfig(
            name=name,
            version=version,
            description=description,
            tools=tools,
            system_prompt_config=system_prompt_config,
            stages=stages,
            middleware=middleware,
            safety_extension=safety_extension,
            prompt_contributor=prompt_contributor,
            mode_config=mode_config,
            tool_dependencies=tool_dependencies,
            workflows=workflows,
            provider_hints=provider_hints,
            metadata=additional_metadata,
        )

    def _extract_legacy_config(self, yaml_data: Dict[str, Any]) -> VerticalYAMLConfig:
        """Extract configuration from legacy flat YAML format.

        Args:
            yaml_data: Parsed YAML data in legacy format

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

        # Extract prompt contributor
        prompt_contributor = yaml_data.get("prompt_contributor")

        # Extract mode config
        mode_config = yaml_data.get("mode_config")

        # Extract tool dependencies
        tool_dependencies = yaml_data.get("tool_dependencies")

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
            prompt_contributor=prompt_contributor,
            mode_config=mode_config,
            tool_dependencies=tool_dependencies,
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
