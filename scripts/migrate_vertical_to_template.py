#!/usr/bin/env python3
"""
Vertical Migration Tool - Extract templates from existing verticals

This script extracts VerticalTemplate definitions from existing vertical
implementations, enabling migration from programmatic to template-based
configuration.

Usage:
    # Extract template from existing vertical
    python scripts/migrate_vertical_to_template.py \
        --vertical victor.coding.CodingAssistant \
        --output templates/coding.yaml

    # Extract and validate
    python scripts/migrate_vertical_to_template.py \
        --vertical victor.coding.CodingAssistant \
        --output templates/coding.yaml \
        --validate

    # Dry run (print without saving)
    python scripts/migrate_vertical_to_template.py \
        --vertical victor.coding.CodingAssistant \
        --dry-run
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from victor.framework.vertical_template import (
    VerticalTemplate,
    VerticalMetadata,
    ExtensionSpecs,
    MiddlewareSpec,
    SafetyPatternSpec,
    PromptHintSpec,
    WorkflowSpec,
    TeamSpec,
    TeamRoleSpec,
    CapabilitySpec,
)
from victor.framework.vertical_template_registry import VerticalTemplateRegistry
from victor.core.verticals.base import VerticalBase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class VerticalExtractor:
    """Extract template from existing vertical implementation.

    Attributes:
        vertical_class: Vertical class to extract from
    """

    def __init__(self, vertical_class: type[VerticalBase]):
        """Initialize the extractor.

        Args:
            vertical_class: Vertical class to extract from
        """
        self.vertical_class = vertical_class

    def extract(self) -> VerticalTemplate:
        """Extract template from vertical.

        Returns:
            VerticalTemplate instance
        """
        logger.info(f"Extracting template from {self.vertical_class.__name__}")

        # Get metadata
        metadata = self._extract_metadata()

        # Get core configuration
        tools = self._extract_tools()
        system_prompt = self._extract_system_prompt()
        stages = self._extract_stages()

        # Get extensions
        extensions = self._extract_extensions()

        # Get workflows
        workflows = self._extract_workflows()

        # Get teams
        teams = self._extract_teams()

        # Get capabilities
        capabilities = self._extract_capabilities()

        # Get custom config
        config = self.vertical_class.get_config()
        custom_config = dict(config.metadata)

        # Create template
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

        logger.info(f"Successfully extracted template from {self.vertical_class.__name__}")
        return template

    def _extract_metadata(self) -> VerticalMetadata:
        """Extract metadata from vertical.

        Returns:
            VerticalMetadata instance
        """
        config = self.vertical_class.get_config()

        return VerticalMetadata(
            name=self.vertical_class.name,
            description=self.vertical_class.description,
            version=getattr(self.vertical_class, "version", "0.5.0"),
            author=getattr(self.vertical_class, "author", None),
            license=getattr(self.vertical_class, "license", "Apache-2.0"),
            category=config.metadata.get("category", "general"),
            tags=config.metadata.get("tags", []),
            provider_hints=self.vertical_class.get_provider_hints(),
            evaluation_criteria=self.vertical_class.get_evaluation_criteria(),
        )

    def _extract_tools(self) -> List[str]:
        """Extract tools from vertical.

        Returns:
            List of tool names
        """
        return self.vertical_class.get_tools()

    def _extract_system_prompt(self) -> str:
        """Extract system prompt from vertical.

        Returns:
            System prompt text
        """
        return self.vertical_class.get_system_prompt()

    def _extract_stages(self) -> Dict[str, Any]:
        """Extract stages from vertical.

        Returns:
            Dictionary of stage definitions
        """
        return self.vertical_class.get_stages()

    def _extract_extensions(self) -> ExtensionSpecs:
        """Extract extension specifications.

        Returns:
            ExtensionSpecs instance
        """
        # Extract middleware
        middleware = []
        try:
            middleware_list = self.vertical_class.get_middleware()
            for mw in middleware_list:
                middleware.append(
                    MiddlewareSpec(
                        name=mw.__class__.__name__,
                        class_name=mw.__class__.__name__,
                        module=mw.__class__.__module__,
                        enabled=True,
                        config=getattr(mw, "config", {}),
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to extract middleware: {e}")

        # Extract safety patterns
        safety_patterns = []
        try:
            safety_ext = self.vertical_class.get_safety_extension()
            patterns = safety_ext.get_bash_patterns()
            for pattern in patterns:
                safety_patterns.append(
                    SafetyPatternSpec(
                        name=pattern.name,
                        pattern=pattern.pattern,
                        description=pattern.description,
                        severity=pattern.severity,
                        category=getattr(pattern, "category", "general"),
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to extract safety patterns: {e}")

        # Extract prompt hints
        prompt_hints = []
        try:
            prompt_contrib = self.vertical_class.get_prompt_contributor()
            hints_dict = prompt_contrib.get_task_type_hints()
            for task_type, hint in hints_dict.items():
                prompt_hints.append(
                    PromptHintSpec(
                        task_type=task_type,
                        hint=hint.hint,
                        tool_budget=getattr(hint, "tool_budget", 10),
                        priority_tools=getattr(hint, "priority_tools", []),
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to extract prompt hints: {e}")

        # Extract handlers
        handlers = {}
        try:
            handlers = self.vertical_class.get_handlers()
        except Exception as e:
            logger.warning(f"Failed to extract handlers: {e}")

        # Extract personas
        personas = {}
        try:
            # Try to get from team spec provider
            team_provider = self.vertical_class.get_team_spec_provider()
            if hasattr(team_provider, "get_personas"):
                personas = team_provider.get_personas()
        except Exception as e:
            logger.warning(f"Failed to extract personas: {e}")

        return ExtensionSpecs(
            middleware=middleware,
            safety_patterns=safety_patterns,
            prompt_hints=prompt_hints,
            handlers=handlers,
            personas=personas,
            composed_chains={},  # TODO: Extract composed chains
        )

    def _extract_workflows(self) -> List[WorkflowSpec]:
        """Extract workflow specifications.

        Returns:
            List of WorkflowSpec instances
        """
        workflows = []

        try:
            workflow_provider = self.vertical_class.get_workflow_provider()
            if hasattr(workflow_provider, "list_workflows"):
                workflow_names = workflow_provider.list_workflows()
                for name in workflow_names:
                    try:
                        workflow = workflow_provider.get_workflow(name)
                        workflows.append(
                            WorkflowSpec(
                                name=name,
                                description=getattr(workflow, "description", ""),
                                yaml_path=getattr(workflow, "yaml_path", None),
                                handler_module=getattr(workflow, "handler_module", None),
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Failed to extract workflow {name}: {e}")
        except Exception as e:
            logger.warning(f"Failed to extract workflows: {e}")

        return workflows

    def _extract_teams(self) -> List[TeamSpec]:
        """Extract team specifications.

        Returns:
            List of TeamSpec instances
        """
        teams = []

        try:
            team_provider = self.vertical_class.get_team_spec_provider()
            if hasattr(team_provider, "list_teams"):
                team_names = team_provider.list_teams()
                for name in team_names:
                    try:
                        team = team_provider.get_team(name)
                        roles = []
                        if hasattr(team, "roles"):
                            for role in team.roles:
                                roles.append(
                                    TeamRoleSpec(
                                        name=role.name,
                                        display_name=getattr(role, "display_name", role.name),
                                        description=getattr(role, "description", ""),
                                        persona=getattr(role, "persona", ""),
                                        tool_categories=getattr(role, "tool_categories", []),
                                        capabilities=getattr(role, "capabilities", []),
                                    )
                                )

                        teams.append(
                            TeamSpec(
                                name=name,
                                display_name=getattr(team, "display_name", name),
                                description=getattr(team, "description", ""),
                                formation=getattr(team, "formation", "parallel"),
                                communication_style=getattr(
                                    team, "communication_style", "structured"
                                ),
                                max_iterations=getattr(team, "max_iterations", 5),
                                roles=roles,
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Failed to extract team {name}: {e}")
        except Exception as e:
            logger.warning(f"Failed to extract teams: {e}")

        return teams

    def _extract_capabilities(self) -> List[CapabilitySpec]:
        """Extract capability specifications.

        Returns:
            List of CapabilitySpec instances
        """
        capabilities = []

        try:
            caps = self.vertical_class.get_capabilities()
            if hasattr(caps, "capabilities"):
                for cap in caps.capabilities:
                    capabilities.append(
                        CapabilitySpec(
                            name=cap.name,
                            type=cap.type,
                            description=cap.description,
                            enabled=cap.enabled,
                            handler=getattr(cap, "handler", None),
                            config=getattr(cap, "default_config", {}),
                        )
                    )
        except Exception as e:
            logger.warning(f"Failed to extract capabilities: {e}")

        return capabilities


def import_vertical_class(import_path: str) -> type[VerticalBase]:
    """Import vertical class from import path.

    Args:
        import_path: Import path (e.g., "victor.coding.CodingAssistant")

    Returns:
        Vertical class

    Raises:
        ImportError: If import fails
    """
    module_path, class_name = import_path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_path)
        vertical_class = getattr(module, class_name)

        if not isinstance(vertical_class, type):
            raise ImportError(f"{class_name} is not a class")

        if not issubclass(vertical_class, VerticalBase):
            raise ImportError(f"{class_name} does not inherit from VerticalBase")

        return vertical_class

    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import {import_path}: {e}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Extract templates from existing verticals")

    parser.add_argument(
        "--vertical",
        type=str,
        required=True,
        help="Vertical import path (e.g., victor.coding.CodingAssistant)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output YAML file path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print template without saving",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate extracted template",
    )

    args = parser.parse_args()

    # Import vertical class
    try:
        vertical_class = import_vertical_class(args.vertical)
    except ImportError as e:
        logger.error(f"Failed to import vertical: {e}")
        return 1

    # Extract template
    extractor = VerticalExtractor(vertical_class)
    template = extractor.extract()

    # Validate if requested
    if args.validate:
        errors = template.validate()
        if errors:
            logger.error("Template validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return 1
        else:
            logger.info("Template validation passed!")

    # Dry run or save
    if args.dry_run:
        import yaml

        logger.info("Template (dry run):")
        print(yaml.safe_dump(template.to_dict(), default_flow_style=False, sort_keys=False))
        return 0

    if args.output:
        registry = VerticalTemplateRegistry.get_instance()
        success = registry.save_to_yaml(template, args.output, validate=False)

        if success:
            logger.info(f"Template saved to {args.output}")
            return 0
        else:
            logger.error("Failed to save template")
            return 1
    else:
        logger.error("Must specify --output or --dry-run")
        return 1


if __name__ == "__main__":
    sys.exit(main())
