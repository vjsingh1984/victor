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

"""Vertical Template Extractor.

This module provides functionality to extract templates from existing
vertical implementations, enabling migration to the template-based system.

Example:
    # Extract template from existing vertical
    extractor = VerticalExtractor()
    template = extractor.extract_from_class(CodingAssistant)

    # Save template to YAML
    registry = VerticalTemplateRegistry()
    registry.save_to_yaml(template, "templates/coding.yaml")
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from victor.core.verticals.base import VerticalBase
from victor.framework.vertical_template import (
    VerticalTemplate,
    VerticalMetadata,
    ExtensionSpecs,
    MiddlewareSpec,
    SafetyPatternSpec,
    PromptHintSpec,
    WorkflowSpec,
    TeamSpec,
    CapabilitySpec,
)

logger = logging.getLogger(__name__)


class VerticalExtractor:
    """Extract templates from existing vertical implementations.

    Analyzes vertical classes and their associated modules to extract
    template information including metadata, tools, stages, and extensions.
    """

    def extract_from_class(self, vertical_class: Type[VerticalBase]) -> Optional[VerticalTemplate]:
        """Extract template from a vertical class.

        Args:
            vertical_class: Vertical class to extract from

        Returns:
            Extracted VerticalTemplate or None if extraction failed
        """
        try:
            logger.info(f"Extracting template from {vertical_class.__name__}")

            # Extract metadata
            metadata = self._extract_metadata(vertical_class)

            # Extract core configuration
            tools = vertical_class.get_tools()
            system_prompt = vertical_class.get_system_prompt()
            stages = vertical_class.get_stages()

            # Extract extensions
            extensions = self._extract_extensions(vertical_class)

            # Extract workflows
            workflows = self._extract_workflows(vertical_class)

            # Extract teams
            teams = self._extract_teams(vertical_class)

            # Extract capabilities
            capabilities = self._extract_capabilities(vertical_class)

            # Extract custom config
            custom_config = self._extract_custom_config(vertical_class)

            template = VerticalTemplate(
                metadata=metadata,
                tools=tools,
                system_prompt=system_prompt,
                stages=stages,
                extensions=extensions,
                workflows=workflows,
                teams=teams,
                capabilities=capabilities,
                custom_config=custom_config,
            )

            logger.info(f"Successfully extracted template from {vertical_class.__name__}")
            return template

        except Exception as e:
            logger.error(f"Error extracting template from {vertical_class.__name__}: {e}")
            return None

    def extract_from_module(self, module_path: str) -> Optional[VerticalTemplate]:
        """Extract template from a vertical module.

        Args:
            module_path: Import path to vertical module (e.g., "victor.coding")

        Returns:
            Extracted VerticalTemplate or None if extraction failed
        """
        try:
            # Import the module
            from importlib import import_module

            module = import_module(module_path)

            # Find VerticalBase subclass
            vertical_class = None
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, VerticalBase) and obj is not VerticalBase:
                    vertical_class = obj
                    break

            if not vertical_class:
                logger.error(f"No VerticalBase subclass found in {module_path}")
                return None

            return self.extract_from_class(vertical_class)

        except Exception as e:
            logger.error(f"Error extracting template from module {module_path}: {e}")
            return None

    def extract_from_directory(self, directory: str | Path) -> Optional[VerticalTemplate]:
        """Extract template from a vertical directory.

        Args:
            directory: Path to vertical directory (e.g., "victor/coding")

        Returns:
            Extracted VerticalTemplate or None if extraction failed
        """
        directory = Path(directory)

        # Find assistant.py
        assistant_file = directory / "assistant.py"
        if not assistant_file.exists():
            logger.error(f"assistant.py not found in {directory}")
            return None

        # Import the assistant module
        module_name = directory.stem
        if directory.parent.stem == "victor":
            module_path = f"victor.{module_name}"
        else:
            module_path = f"{directory.parent.stem}.{module_name}"

        return self.extract_from_module(module_path)

    def _extract_metadata(self, vertical_class: Type[VerticalBase]) -> VerticalMetadata:
        """Extract metadata from vertical class."""
        return VerticalMetadata(
            name=vertical_class.name,
            description=vertical_class.description,
            version=getattr(vertical_class, "version", "0.5.0"),
            category=getattr(vertical_class, "category", "general"),
        )

    def _extract_extensions(self, vertical_class: Type[VerticalBase]) -> ExtensionSpecs:
        """Extract extension specifications from vertical class."""
        # Try to get middleware
        middleware_specs = []
        try:
            middleware = vertical_class.get_extensions_by_type("middleware")
            for mw in middleware:
                middleware_specs.append(
                    MiddlewareSpec(
                        name=mw.__class__.__name__,
                        class_name=mw.__class__.__name__,
                        module=mw.__class__.__module__,
                        enabled=True,
                        config=getattr(mw, "config", {}),
                    )
                )
        except Exception as e:
            logger.debug(f"Could not extract middleware: {e}")

        # Try to get prompt hints
        prompt_hints = []
        try:
            prompts_module = __import__(
                f"victor.{vertical_class.name}.prompts", fromlist=["TASK_TYPE_HINTS"]
            )
            TASK_TYPE_HINTS = prompts_module.TASK_TYPE_HINTS

            for task_type, hint in TASK_TYPE_HINTS.items():
                prompt_hints.append(
                    PromptHintSpec(
                        task_type=task_type,
                        hint=hint.hint,
                        tool_budget=hint.tool_budget,
                        priority_tools=list(hint.priority_tools) if hint.priority_tools else [],
                    )
                )
        except Exception as e:
            logger.debug(f"Could not extract prompt hints: {e}")

        return ExtensionSpecs(
            middleware=middleware_specs,
            prompt_hints=prompt_hints,
        )

    def _extract_workflows(self, vertical_class: Type[VerticalBase]) -> List[WorkflowSpec]:
        """Extract workflow specifications from vertical."""
        workflows = []

        # Try to load workflow YAML files
        vertical_dir = Path(vertical_class.__module__.replace(".", "/"))
        workflows_dir = vertical_dir / "workflows"

        if workflows_dir.exists():
            for yaml_file in workflows_dir.glob("*.yaml"):
                workflows.append(
                    WorkflowSpec(
                        name=yaml_file.stem,
                        description=f"Workflow from {yaml_file.name}",
                        yaml_path=str(yaml_file),
                    )
                )

        return workflows

    def _extract_teams(self, vertical_class: Type[VerticalBase]) -> List[TeamSpec]:
        """Extract team specifications from vertical."""
        teams = []

        # Try to load teams from teams.py
        try:
            teams_module = __import__(f"victor.{vertical_class.name}.teams", fromlist=["PERSONAS"])
            PERSONAS = teams_module.PERSONAS

            for team_name, persona in PERSONAS.items():
                # Convert persona to team spec
                teams.append(
                    TeamSpec(
                        name=team_name,
                        display_name=persona.name,
                        description=persona.role,
                        formation="parallel",
                    )
                )
        except Exception as e:
            logger.debug(f"Could not extract teams: {e}")

        return teams

    def _extract_capabilities(self, vertical_class: Type[VerticalBase]) -> List[CapabilitySpec]:
        """Extract capability specifications from vertical."""
        capabilities = []

        # Try to load capabilities YAML
        config_dir = Path("victor/config/capabilities")
        capabilities_file = config_dir / f"{vertical_class.name}_capabilities.yaml"

        if capabilities_file.exists():
            try:
                import yaml

                with open(capabilities_file) as f:
                    data = yaml.safe_load(f)

                for cap_name, cap_data in data.get("capabilities", {}).items():
                    capabilities.append(
                        CapabilitySpec(
                            name=cap_name,
                            type=cap_data.get("type", "tool"),
                            description=cap_data.get("description", ""),
                            enabled=cap_data.get("enabled", True),
                            handler=cap_data.get("handler"),
                            config=cap_data.get("config", {}),
                        )
                    )
            except Exception as e:
                logger.debug(f"Could not extract capabilities from YAML: {e}")

        return capabilities

    def _extract_custom_config(self, vertical_class: Type[VerticalBase]) -> Dict[str, Any]:
        """Extract custom configuration from vertical."""
        custom_config = {}

        # Try to get capability configs
        try:
            config = vertical_class.get_capability_configs()
            custom_config.update(config)
        except Exception as e:
            logger.debug(f"Could not extract capability configs: {e}")

        return custom_config


# =============================================================================
# Migration Utility
# =============================================================================


def migrate_vertical_to_template(
    vertical_class: Type[VerticalBase],
    output_path: str | Path,
) -> bool:
    """Migrate existing vertical to template-based system.

    Args:
        vertical_class: Vertical class to migrate
        output_path: Output path for template YAML

    Returns:
        True if migration succeeded
    """
    from victor.framework.vertical_template_registry import VerticalTemplateRegistry

    extractor = VerticalExtractor()
    template = extractor.extract_from_class(vertical_class)

    if not template:
        logger.error(f"Failed to extract template from {vertical_class.__name__}")
        return False

    registry: VerticalTemplateRegistry = VerticalTemplateRegistry()
    success = registry.save_to_yaml(template, output_path)

    if success:
        logger.info(f"Migrated {vertical_class.name} to template: {output_path}")
    else:
        logger.error(f"Failed to save template to {output_path}")

    return success


# =============================================================================
# CLI Integration
# =============================================================================


def main_extract(args) -> int:
    """CLI entry point for extraction.

    Args:
        args: Parsed CLI arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    extractor = VerticalExtractor()
    template = None

    if args.class_name:
        # Extract from class name
        from victor.core.verticals import VerticalRegistry

        vertical_class = VerticalRegistry.get(args.class_name)
        if vertical_class:
            template = extractor.extract_from_class(vertical_class)
        else:
            print(f"Vertical not found: {args.class_name}")
            return 1
    elif args.module:
        # Extract from module path
        template = extractor.extract_from_module(args.module)
    elif args.directory:
        # Extract from directory
        template = extractor.extract_from_directory(args.directory)

    if not template:
        print("Failed to extract template")
        return 1

    # Save template
    from victor.framework.vertical_template_registry import VerticalTemplateRegistry

    registry: VerticalTemplateRegistry = VerticalTemplateRegistry()

    # Validate before saving
    if args.validate:
        errors = template.validate()
        if errors:
            print("Template validation errors:")
            for error in errors:
                print(f"  - {error}")
            if args.strict:
                return 1

    output_path = args.output or f"{template.metadata.name}_template.yaml"
    success = registry.save_to_yaml(template, output_path)

    if success:
        print(f"Template saved to: {output_path}")
        return 0
    else:
        print(f"Failed to save template to: {output_path}")
        return 1


def main() -> int:
    """CLI entry point for vertical extraction.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract vertical templates from existing implementations"
    )

    parser.add_argument(
        "--class-name",
        type=str,
        help="Vertical class name (e.g., 'coding')",
    )
    parser.add_argument(
        "--module",
        type=str,
        help="Module path (e.g., 'victor.coding')",
    )
    parser.add_argument(
        "--directory",
        type=str,
        help="Directory path (e.g., 'victor/coding')",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for template YAML",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate template before saving",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on validation errors",
    )

    args = parser.parse_args()

    if not any([args.class_name, args.module, args.directory]):
        parser.print_help()
        return 1

    return main_extract(args)


if __name__ == "__main__":
    import sys

    sys.exit(main() or 0)


__all__ = [
    "VerticalExtractor",
    "migrate_vertical_to_template",
]
