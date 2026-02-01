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

"""Workflow template registry for loading and extending workflow templates.

This module provides a centralized registry for workflow templates that can be:
1. Loaded from YAML files
2. Referenced by name
3. Extended with overrides
4. Composed into complete workflows

Example:
    from victor.workflows.template_registry import (
        WorkflowTemplateRegistry,
        get_workflow_template_registry,
    )

    registry = get_workflow_template_registry()

    # Load templates from YAML
    registry.load_templates_from_yaml("path/to/workflows.yaml")

    # Get a template
    template = registry.get_template("code_review_parallel")

    # Extend a template with overrides
    extended = registry.extend_template(
        "code_review_parallel",
        {"name": "Security Review", "tool_budget": 50}
    )
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


class WorkflowTemplateRegistry:
    """Registry for workflow templates with extension support.

    This registry manages workflow templates loaded from YAML files,
    allowing templates to reference each other and be extended with
    overrides.

    Attributes:
        _templates: Dict mapping template names to template definitions
        _stage_templates: Dict mapping stage names to stage definitions
        _template_paths: List of paths that have been loaded
    """

    def __init__(self) -> None:
        """Initialize the workflow template registry."""
        self._templates: dict[str, dict[str, Any]] = {}
        self._stage_templates: dict[str, dict[str, Any]] = {}
        self._template_paths: list[Path] = []

    def load_templates_from_yaml(
        self,
        yaml_path: str | Path,
        namespace: Optional[str] = None,
    ) -> None:
        """Load workflow templates from a YAML file.

        Args:
            yaml_path: Path to YAML file containing workflow definitions
            namespace: Optional namespace prefix for template names

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML is malformed
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Workflow template file not found: {yaml_path}")

        logger.debug(f"Loading workflow templates from {yaml_path}")

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        # Load workflow templates
        if "templates" in data:
            for name, config in data["templates"].items():
                full_name = f"{namespace}.{name}" if namespace else name
                self._templates[full_name] = config
                logger.debug(f"Registered workflow template: {full_name}")

        # Load workflows (alternative key)
        if "workflows" in data:
            for name, config in data["workflows"].items():
                full_name = f"{namespace}.{name}" if namespace else name
                self._templates[full_name] = config
                logger.debug(f"Registered workflow: {full_name}")

        # Load team definitions (formation-based workflows)
        if "name" in data and "members" in data:
            # This is a team definition file
            name = data.get("name", yaml_path.stem)
            full_name = f"{namespace}.{name}" if namespace else name
            self._templates[full_name] = data
            logger.debug(f"Registered team template: {full_name}")

        # Load stage templates
        if "stage_templates" in data:
            for name, config in data["stage_templates"].items():
                self._stage_templates[name] = config
                logger.debug(f"Registered stage template: {name}")

        # Load override presets
        if "override_presets" in data:
            for name, config in data["override_presets"].items():
                # Store as a special type of template
                self._stage_templates[f"preset:{name}"] = config
                logger.debug(f"Registered override preset: {name}")

        self._template_paths.append(yaml_path)
        logger.info(
            f"Loaded {len(self._templates)} workflow templates, "
            f"{len(self._stage_templates)} stage templates from {yaml_path.name}"
        )

    def load_templates_from_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        pattern: str = "*.yaml",
    ) -> None:
        """Load all workflow templates from a directory.

        Args:
            directory: Directory containing YAML template files
            recursive: If True, search subdirectories
            pattern: Glob pattern for YAML files
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        yaml_files = directory.rglob(pattern) if recursive else directory.glob(pattern)

        for yaml_file in yaml_files:
            try:
                # Use relative path as namespace
                rel_path = yaml_file.relative_to(directory)
                namespace = str(rel_path.parent).replace("/", ".")

                self.load_templates_from_yaml(yaml_file, namespace=namespace)
            except Exception as e:
                logger.warning(f"Failed to load {yaml_file}: {e}")

    def get_template(
        self, name: str, default: Optional[dict[str, Any]] = None
    ) -> Optional[dict[str, Any]]:
        """Get a workflow template by name.

        Args:
            name: Template name (with namespace if applicable)
            default: Default value if template not found

        Returns:
            Template definition or default if not found
        """
        return self._templates.get(name, default)

    def get_stage_template(
        self, name: str, default: Optional[dict[str, Any]] = None
    ) -> Optional[dict[str, Any]]:
        """Get a stage template by name.

        Args:
            name: Stage template name
            default: Default value if template not found

        Returns:
            Stage template definition or default if not found
        """
        return self._stage_templates.get(name, default)

    def has_template(self, name: str) -> bool:
        """Check if a template exists.

        Args:
            name: Template name

        Returns:
            True if template exists
        """
        return name in self._templates

    def has_stage_template(self, name: str) -> bool:
        """Check if a stage template exists.

        Args:
            name: Stage template name

        Returns:
            True if stage template exists
        """
        return name in self._stage_templates

    def extend_template(
        self,
        base_name: str,
        overrides: dict[str, Any],
        deep_merge: bool = True,
    ) -> dict[str, Any]:
        """Extend a base template with overrides.

        This creates a new template by deep copying the base and applying
        overrides. Supports nested merging for complex structures.

        Args:
            base_name: Name of base template to extend
            overrides: Override values to apply
            deep_merge: If True, deep merge dicts; if False, replace

        Returns:
            New template definition with overrides applied

        Raises:
            ValueError: If base template not found
        """
        base = self.get_template(base_name)
        if base is None:
            available = ", ".join(self.list_templates())
            raise ValueError(
                f"Base template '{base_name}' not found. "
                f"Available: {available if available else 'none'}"
            )

        # Deep copy base to avoid mutating original
        extended = copy.deepcopy(base)

        # Apply overrides
        if deep_merge:
            self._deep_merge(extended, overrides)
        else:
            extended.update(overrides)

        # Track inheritance
        if "extends" in extended:
            # Already extends something - add to chain
            if isinstance(extended["extends"], list):
                extended["extends"].append(base_name)
            else:
                extended["extends"] = [extended["extends"], base_name]
        else:
            extended["extends"] = base_name

        logger.debug(f"Extended template '{base_name}' with {len(overrides)} overrides")

        return extended

    def _deep_merge(self, base: dict[str, Any], overrides: dict[str, Any]) -> None:
        """Deep merge overrides into base dict (in-place).

        Args:
            base: Base dict to merge into
            overrides: Overrides to apply
        """
        for key, value in overrides.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                self._deep_merge(base[key], value)
            elif key in base and isinstance(base[key], list) and isinstance(value, list):
                # For lists, replace (could also implement list merging)
                base[key] = value
            else:
                # Replace with override value
                base[key] = value

    def list_templates(self) -> list[str]:
        """List all registered template names.

        Returns:
            List of template names
        """
        return sorted(self._templates.keys())

    def list_stage_templates(self) -> list[str]:
        """List all registered stage template names.

        Returns:
            List of stage template names
        """
        return sorted(self._stage_templates.keys())

    def register_template(self, name: str, template: dict[str, Any]) -> None:
        """Manually register a workflow template.

        Args:
            name: Template name
            template: Template definition

        Raises:
            ValueError: If template already registered
        """
        if name in self._templates:
            raise ValueError(f"Template '{name}' already registered")

        self._templates[name] = template
        logger.debug(f"Registered workflow template: {name}")

    def register_stage_template(self, name: str, stage: dict[str, Any]) -> None:
        """Manually register a stage template.

        Args:
            name: Stage template name
            stage: Stage template definition

        Raises:
            ValueError: If stage template already registered
        """
        if name in self._stage_templates:
            raise ValueError(f"Stage template '{name}' already registered")

        self._stage_templates[name] = stage
        logger.debug(f"Registered stage template: {name}")

    def get_template_info(self, name: str) -> dict[str, Any]:
        """Get metadata about a template.

        Args:
            name: Template name

        Returns:
            Dict with template metadata (name, type, description, etc.)

        Raises:
            ValueError: If template not found
        """
        template = self.get_template(name)
        if template is None:
            raise ValueError(f"Template '{name}' not found")

        return {
            "name": name,
            "type": "workflow",
            "display_name": template.get("display_name", template.get("name", name)),
            "description": template.get("description", ""),
            "version": template.get("version", "unknown"),
            "vertical": template.get("vertical", "general"),
            "formation": template.get("formation", "unknown"),
            "complexity": template.get("complexity", "unknown"),
            "extends": template.get("extends"),
        }

    def get_stage_info(self, name: str) -> dict[str, Any]:
        """Get metadata about a stage template.

        Args:
            name: Stage template name

        Returns:
            Dict with stage metadata

        Raises:
            ValueError: If stage template not found
        """
        stage = self.get_stage_template(name)
        if stage is None:
            raise ValueError(f"Stage template '{name}' not found")

        return {
            "name": name,
            "type": "stage",
            "description": stage.get("description", ""),
            "stage_type": stage.get("type", "agent"),
        }

    def clear(self) -> None:
        """Clear all registered templates."""
        self._templates.clear()
        self._stage_templates.clear()
        self._template_paths.clear()
        logger.debug("Cleared all templates from registry")

    @property
    def template_count(self) -> int:
        """Get number of registered workflow templates."""
        return len(self._templates)

    @property
    def stage_template_count(self) -> int:
        """Get number of registered stage templates."""
        return len(self._stage_templates)


# Global registry instance
_workflow_template_registry = WorkflowTemplateRegistry()


def get_workflow_template_registry() -> WorkflowTemplateRegistry:
    """Get the global workflow template registry.

    Returns:
        WorkflowTemplateRegistry singleton
    """
    return _workflow_template_registry


def register_default_templates() -> None:
    """Register default workflow templates from framework templates directory.

    This loads all templates from victor/workflows/templates/ directory.
    """

    # Get the templates directory
    templates_dir = Path(__file__).parent / "templates"

    if not templates_dir.exists():
        logger.warning(f"Workflow templates directory not found: {templates_dir}")
        return

    registry = get_workflow_template_registry()

    # Load common stages first
    common_stages = templates_dir / "common_stages.yaml"
    if common_stages.exists():
        registry.load_templates_from_yaml(common_stages)

    # Load all workflow templates from subdirectories
    for category_dir in templates_dir.iterdir():
        if category_dir.is_dir() and not category_dir.name.startswith("_"):
            try:
                registry.load_templates_from_directory(
                    category_dir,
                    recursive=True,
                    pattern="*.yaml",
                )
            except Exception as e:
                logger.warning(f"Failed to load templates from {category_dir}: {e}")

    logger.info(
        f"Registered {registry.template_count} workflow templates, "
        f"{registry.stage_template_count} stage templates from default location"
    )


# Auto-register default templates on import
try:
    register_default_templates()
except Exception as e:
    logger.debug(f"Deferred default template registration: {e}")


__all__ = [
    "WorkflowTemplateRegistry",
    "get_workflow_template_registry",
    "register_default_templates",
]
