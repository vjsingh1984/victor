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

"""Vertical Template Registry for template management.

This module provides the VerticalTemplateRegistry class that manages
vertical templates for scaffolding and generation.

The registry supports:
- Loading templates from YAML files
- Registering templates programmatically
- Saving templates to YAML
- Listing and querying templates
- Template validation
- Export and import operations

Example:
    # Load registry
    registry = VerticalTemplateRegistry()

    # Register template from file
    template = registry.load_from_yaml("path/to/template.yaml")
    registry.register(template)

    # Query templates
    coding_template = registry.get("coding")
    all_templates = registry.list_all()

    # Save template
    registry.save_to_yaml(template, "output/path.yaml")
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from victor.framework.vertical_template import VerticalTemplate, VerticalMetadata

logger = logging.getLogger(__name__)


class VerticalTemplateRegistry:
    """Registry for managing vertical templates.

    Thread-safe registry that supports:
    - Template registration and retrieval
    - YAML loading and saving
    - Template validation
    - Query and listing operations

    Attributes:
        templates: Dict mapping template names to templates
        lock: Thread lock for thread-safe operations
    """

    # Singleton instance
    _instance: Optional[VerticalTemplateRegistry] = None
    _lock = threading.Lock()

    def __init__(self):
        """Initialize the registry."""
        self._templates: Dict[str, VerticalTemplate] = {}
        self._lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> VerticalTemplateRegistry:
        """Get singleton instance of the registry.

        Returns:
            VerticalTemplateRegistry singleton instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register(self, template: VerticalTemplate, *, overwrite: bool = False) -> None:
        """Register a vertical template.

        Args:
            template: Template to register
            overwrite: If True, overwrite existing template with same name

        Raises:
            ValueError: If template already exists and overwrite=False
            ValueError: If template is invalid
        """
        # Validate template
        errors = template.validate()
        if errors:
            raise ValueError(f"Invalid template: {', '.join(errors)}")

        name = template.metadata.name

        with self._lock:
            if name in self._templates and not overwrite:
                raise ValueError(
                    f"Template '{name}' already exists. Use overwrite=True to replace."
                )

            self._templates[name] = template
            logger.info(f"Registered vertical template: {name}")

    def unregister(self, name: str) -> None:
        """Unregister a vertical template.

        Args:
            name: Template name to unregister
        """
        with self._lock:
            if name in self._templates:
                del self._templates[name]
                logger.info(f"Unregistered vertical template: {name}")

    def get(self, name: str) -> Optional[VerticalTemplate]:
        """Get a template by name.

        Args:
            name: Template name

        Returns:
            VerticalTemplate or None if not found
        """
        with self._lock:
            return self._templates.get(name)

    def list_all(self) -> List[VerticalTemplate]:
        """List all registered templates.

        Returns:
            List of all templates
        """
        with self._lock:
            return list(self._templates.values())

    def list_names(self) -> List[str]:
        """List names of all registered templates.

        Returns:
            List of template names
        """
        with self._lock:
            return list(self._templates.keys())

    def find_by_category(self, category: str) -> List[VerticalTemplate]:
        """Find templates by category.

        Args:
            category: Category to filter by

        Returns:
            List of templates in the category
        """
        with self._lock:
            return [t for t in self._templates.values() if t.metadata.category == category]

    def find_by_tag(self, tag: str) -> List[VerticalTemplate]:
        """Find templates by tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of templates with the tag
        """
        with self._lock:
            return [t for t in self._templates.values() if tag in t.metadata.tags]

    def clear(self) -> None:
        """Clear all templates from the registry."""
        with self._lock:
            self._templates.clear()
            logger.info("Cleared all vertical templates")

    def load_from_yaml(self, path: str | Path) -> Optional[VerticalTemplate]:
        """Load a template from a YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Loaded VerticalTemplate or None if loading failed
        """
        path = Path(path)

        if not path.exists():
            logger.error(f"Template file not found: {path}")
            return None

        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)

            if not data:
                logger.error(f"Empty YAML file: {path}")
                return None

            template = VerticalTemplate.from_dict(data)

            # Validate loaded template
            errors = template.validate()
            if errors:
                logger.warning(
                    f"Template loaded from {path} has validation errors: {', '.join(errors)}"
                )

            logger.info(f"Loaded vertical template from {path}")
            return template

        except Exception as e:
            logger.error(f"Error loading template from {path}: {e}")
            return None

    def load_and_register(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> Optional[VerticalTemplate]:
        """Load template from YAML and register it.

        Convenience method that combines load_from_yaml and register.

        Args:
            path: Path to YAML file
            overwrite: If True, overwrite existing template

        Returns:
            Registered VerticalTemplate or None if loading failed
        """
        template = self.load_from_yaml(path)
        if template:
            try:
                self.register(template, overwrite=overwrite)
                return template
            except ValueError as e:
                logger.error(f"Failed to register template from {path}: {e}")
                return None
        return None

    def load_directory(
        self,
        directory: str | Path,
        *,
        pattern: str = "*.yaml",
        overwrite: bool = False,
    ) -> int:
        """Load all templates from a directory.

        Args:
            directory: Directory containing template YAML files
            pattern: Glob pattern for template files (default: *.yaml)
            overwrite: If True, overwrite existing templates

        Returns:
            Number of templates successfully loaded
        """
        directory = Path(directory)

        if not directory.is_dir():
            logger.error(f"Not a directory: {directory}")
            return 0

        count = 0
        for yaml_file in directory.glob(pattern):
            template = self.load_and_register(yaml_file, overwrite=overwrite)
            if template:
                count += 1

        logger.info(f"Loaded {count} templates from {directory}")
        return count

    def save_to_yaml(
        self,
        template: VerticalTemplate,
        path: str | Path,
        *,
        validate: bool = True,
    ) -> bool:
        """Save a template to a YAML file.

        Args:
            template: Template to save
            path: Output path for YAML file
            validate: If True, validate template before saving

        Returns:
            True if save succeeded
        """
        if validate:
            errors = template.validate()
            if errors:
                logger.error(f"Cannot save invalid template: {', '.join(errors)}")
                return False

        path = Path(path)

        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = template.to_dict()

            with open(path, "w") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Saved vertical template to {path}")
            return True

        except Exception as e:
            logger.error(f"Error saving template to {path}: {e}")
            return False

    def export_to_dict(self, name: str) -> Optional[Dict[str, Any]]:
        """Export a template to a dictionary.

        Args:
            name: Template name

        Returns:
            Dictionary representation or None if not found
        """
        template = self.get(name)
        if template:
            return template.to_dict()
        return None

    def import_from_dict(
        self,
        data: Dict[str, Any],
        *,
        overwrite: bool = False,
    ) -> Optional[VerticalTemplate]:
        """Import a template from a dictionary.

        Args:
            data: Dictionary containing template data
            overwrite: If True, overwrite existing template

        Returns:
            Imported VerticalTemplate or None if import failed
        """
        try:
            template = VerticalTemplate.from_dict(data)
            self.register(template, overwrite=overwrite)
            return template
        except Exception as e:
            logger.error(f"Error importing template from dict: {e}")
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dictionary with registry statistics
        """
        with self._lock:
            templates = list(self._templates.values())

            categories = {}
            for template in templates:
                cat = template.metadata.category
                categories[cat] = categories.get(cat, 0) + 1

            tools_count = sum(len(t.tools) for t in templates)
            stages_count = sum(len(t.stages) for t in templates)
            extensions_count = sum(len(t.extensions.middleware) for t in templates)

            return {
                "total_templates": len(templates),
                "categories": categories,
                "total_tools": tools_count,
                "total_stages": stages_count,
                "total_middleware": extensions_count,
                "template_names": self.list_names(),
            }

    def validate_registry(self) -> List[str]:
        """Validate all templates in the registry.

        Returns:
            List of validation errors (empty if all valid)
        """
        errors = []

        with self._lock:
            for name, template in self._templates.items():
                template_errors = template.validate()
                for error in template_errors:
                    errors.append(f"{name}: {error}")

        return errors

    def is_registry_valid(self) -> bool:
        """Check if all templates in registry are valid.

        Returns:
            True if all templates are valid
        """
        return len(self.validate_registry()) == 0


# Global convenience functions


def get_template_registry() -> VerticalTemplateRegistry:
    """Get the global template registry singleton.

    Returns:
        VerticalTemplateRegistry instance
    """
    return VerticalTemplateRegistry.get_instance()


def register_template(template: VerticalTemplate, *, overwrite: bool = False) -> None:
    """Register a template with the global registry.

    Args:
        template: Template to register
        overwrite: If True, overwrite existing template
    """
    registry = get_template_registry()
    registry.register(template, overwrite=overwrite)


def get_template(name: str) -> Optional[VerticalTemplate]:
    """Get a template from the global registry.

    Args:
        name: Template name

    Returns:
        VerticalTemplate or None if not found
    """
    registry = get_template_registry()
    return registry.get(name)


def list_templates() -> List[VerticalTemplate]:
    """List all templates in the global registry.

    Returns:
        List of all templates
    """
    registry = get_template_registry()
    return registry.list_all()


__all__ = [
    "VerticalTemplateRegistry",
    "get_template_registry",
    "register_template",
    "get_template",
    "list_templates",
]
