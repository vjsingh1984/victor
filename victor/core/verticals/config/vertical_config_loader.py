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

This module provides VerticalConfigLoader which loads vertical configuration
from YAML files, replacing multiple get_* methods with declarative YAML.

Design Patterns:
    - Builder Pattern: Build VerticalConfig from YAML
    - Strategy Pattern: Multiple prompt sources (inline, file, template)
    - Validation: Schema-based validation
    - SRP: Focused only on loading YAML configs

Usage:
    from victor.core.verticals.config import VerticalConfigLoader

    loader = VerticalConfigLoader()
    config = loader.load_from_yaml("path/to/vertical.yaml")

    # Use the config
    tools = config.tools.tools
    prompt = config.system_prompt
    stages = config.stages
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import yaml

from victor.core.vertical_types import StageDefinition
from victor.core.verticals.base import VerticalConfig
from victor.framework.tools import ToolSet

logger = logging.getLogger(__name__)


class VerticalConfigLoader:
    """Loader for YAML-based vertical configuration.

    This loader reads YAML configuration files and converts them
    into VerticalConfig objects, replacing the need for multiple
    get_* methods in vertical classes.

    Responsibilities:
    - Load YAML from file path
    - Validate required fields
    - Parse tools configuration
    - Parse system prompt (inline, file, or template)
    - Parse stage definitions
    - Parse optional configurations (provider, tiered tools, extensions, evaluation)
    - Build VerticalConfig objects

    YAML Structure:
        metadata:
          name: vertical_name
          description: Vertical description

        core:
          tools: list[Any]: [tool1, tool2]
          system_prompt:
            source: inline
            text: "Prompt text"
          stages:
            STAGE_NAME:
              name: STAGE_NAME
              description: Stage description
              tools: [tool1]
              keywords: [keyword1]
              next_stages: [NEXT_STAGE]

        provider:
          hints:
            preferred: [provider1, provider2]

        extensions:
          middleware:
            enabled: true
            list: [...]

        evaluation:
          criteria: ["criterion1", "criterion2"]
    """

    def __init__(self, strict_validation: bool = False):
        """Initialize the vertical config loader.

        Args:
            strict_validation: If True, raise on validation errors (default: False)
        """
        self._strict_validation = strict_validation

    def load_from_yaml(self, yaml_path: str | Path) -> Optional[VerticalConfig]:
        """Load vertical configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            VerticalConfig object or None if loading fails

        Raises:
            FileNotFoundError: If YAML file doesn't exist (in strict mode)
            ValueError: If validation fails (in strict mode)
            YAMLError: If YAML syntax is invalid
        """
        yaml_path = Path(yaml_path)

        # Check if file exists
        if not yaml_path.exists():
            if self._strict_validation:
                raise FileNotFoundError(f"YAML config not found: {yaml_path}")
            logger.warning(f"YAML config not found: {yaml_path}")
            return None

        # Load YAML
        try:
            with open(yaml_path, "r") as f:
                yaml_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            if self._strict_validation:
                raise
            logger.error(f"Failed to parse YAML {yaml_path}: {e}")
            return None

        # Validate and build config
        try:
            return self._build_config(yaml_data, yaml_path.parent)
        except ValueError as e:
            if self._strict_validation:
                raise
            logger.error(f"Validation failed for {yaml_path}: {e}")
            return None

    def _build_config(self, yaml_data: Dict[str, Any], base_path: Path) -> VerticalConfig:
        """Build VerticalConfig from parsed YAML data.

        Args:
            yaml_data: Parsed YAML dictionary
            base_path: Base directory for resolving relative paths

        Returns:
            VerticalConfig object

        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Validate required fields
        self._validate_required_fields(yaml_data)

        # Extract metadata
        metadata = yaml_data.get("metadata", {})
        metadata_base = {
            "name": metadata.get("name"),
            "version": metadata.get("version", "0.5.0"),
            "description": metadata.get("description"),
        }

        # Extract core configuration
        core = yaml_data["core"]

        # Build tools
        tools_config = core.get("tools", {})
        tools_list = self._parse_tools(tools_config)
        tools = ToolSet(set(tools_list))

        # Build system prompt
        prompt_config = core.get("system_prompt", {})
        system_prompt = self._parse_system_prompt(prompt_config, base_path)

        # Build stages (optional)
        stages = self._parse_stages(core.get("stages", {}))

        # Build provider hints (optional)
        provider_hints = self._parse_provider_hints(yaml_data.get("provider", {}))

        # Build evaluation criteria (optional)
        evaluation_criteria = yaml_data.get("evaluation", {}).get("criteria", [])

        # Build extended metadata
        extended_metadata = self._build_extended_metadata(yaml_data)

        # Merge base and extended metadata
        final_metadata = {**metadata_base, **extended_metadata}

        # Create VerticalConfig
        return VerticalConfig(
            name=metadata.get("name", ""),
            tools=tools,
            system_prompt=system_prompt,
            stages=stages,
            provider_hints=provider_hints,
            evaluation_criteria=evaluation_criteria,
            metadata=final_metadata,
        )

    def _validate_required_fields(self, yaml_data: Dict[str, Any]) -> None:
        """Validate that required fields are present.

        Args:
            yaml_data: Parsed YAML dictionary

        Raises:
            ValueError: If required fields are missing
        """
        # Check metadata
        if "metadata" not in yaml_data:
            raise ValueError("Missing required field: metadata")

        metadata = yaml_data["metadata"]
        if "name" not in metadata:
            raise ValueError("Missing required field: metadata.name")
        if "description" not in metadata:
            raise ValueError("Missing required field: metadata.description")

        # Check core configuration
        if "core" not in yaml_data:
            raise ValueError("Missing required field: core")

        core = yaml_data["core"]
        if "tools" not in core:
            raise ValueError("Missing required field: core.tools")

        tools = core["tools"]
        if "list" not in tools and "capabilities" not in tools:
            raise ValueError("Missing required field: core.tools.list or core.tools.capabilities")

        if "system_prompt" not in core:
            raise ValueError("Missing required field: core.system_prompt")

    def _parse_tools(self, tools_config: Dict[str, Any]) -> List[str]:
        """Parse tools configuration.

        Args:
            tools_config: Tools configuration from YAML

        Returns:
            List of tool names
        """
        # Get tools list
        if "list" in tools_config:
            tools_list = list(tools_config["list"]) if isinstance(tools_config["list"], list) else []
        else:
            # Start with empty if using capabilities (future feature)
            tools_list = []

        # Apply exclusions
        exclusions = tools_config.get("exclude", [])
        if exclusions:
            tools_list = [t for t in tools_list if t not in exclusions]

        return tools_list

    def _parse_system_prompt(self, prompt_config: Dict[str, Any], base_path: Path) -> str:
        """Parse system prompt configuration.

        Args:
            prompt_config: Prompt configuration from YAML
            base_path: Base directory for resolving relative paths

        Returns:
            System prompt string
        """
        source = prompt_config.get("source", "inline")

        if source == "inline":
            # Inline prompt text
            text = prompt_config.get("text", "")
            return str(text) if text else ""

        elif source == "file":
            # Load from file
            file_path = prompt_config.get("file_path")
            if not file_path:
                return ""

            # Resolve relative to base_path
            file_path = base_path / cast(str, file_path)

            if file_path.exists():
                return cast(str, file_path.read_text())
            else:
                logger.warning(f"Prompt file not found: {file_path}")
                return ""

        elif source == "template":
            # Template with variable substitution (future feature)
            template = prompt_config.get("template", "")
            # For now, just return template as-is
            # NOTE: Variable substitution (e.g., {{project_name}}, {{version}}) needs Jinja2-like template engine
            # Deferred: Low priority - static prompts work for current use cases
            return str(template) if template else ""

        else:
            logger.warning(f"Unknown prompt source: {source}")
            return ""

    def _parse_stages(self, stages_config: Dict[str, Any]) -> Dict[str, StageDefinition]:
        """Parse stage definitions.

        Args:
            stages_config: Stages configuration from YAML

        Returns:
            Dictionary mapping stage names to StageDefinition objects
        """
        stages = {}

        for stage_name, stage_data in stages_config.items():
            # Parse tools
            tools = set(stage_data.get("tools", []))

            # Parse keywords
            keywords = stage_data.get("keywords", [])

            # Parse next stages
            next_stages = set(stage_data.get("next_stages", []))

            # Create StageDefinition
            stages[stage_name] = StageDefinition(
                name=stage_name,
                description=stage_data.get("description", ""),
                tools=tools,
                keywords=keywords,
                next_stages=next_stages,
            )

        return stages

    def _parse_provider_hints(self, provider_config: Dict[str, Any]) -> Dict[str, Any]:
        """Parse provider hints configuration.

        Args:
            provider_config: Provider configuration from YAML

        Returns:
            Dictionary of provider hints
        """
        hints = {}

        # Extract hints section
        if "hints" in provider_config:
            hints.update(provider_config["hints"])

        # Extract parameters
        if "parameters" in provider_config:
            hints["parameters"] = provider_config["parameters"]

        return hints

    def _build_extended_metadata(self, yaml_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build extended metadata from optional sections.

        Args:
            yaml_data: Full YAML configuration

        Returns:
            Extended metadata dictionary
        """
        metadata = {}

        # Tiered tools configuration
        if "tiered_tools" in yaml_data:
            metadata["tiered_tools"] = yaml_data["tiered_tools"]

        # Extensions configuration
        if "extensions" in yaml_data:
            metadata["extensions"] = yaml_data["extensions"]

        # Evaluation metrics
        if "evaluation" in yaml_data and "metrics" in yaml_data["evaluation"]:
            metadata["evaluation_metrics"] = yaml_data["evaluation"]["metrics"]

        # Advanced configuration
        if "advanced" in yaml_data:
            metadata["advanced"] = yaml_data["advanced"]

        return metadata


__all__ = [
    "VerticalConfigLoader",
]
