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

"""Template system for HITL prompts.

This module provides a template registry and rendering system for
common HITL prompt patterns. Templates support variable substitution
and can be customized for specific workflows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from string import Template
from typing import Any, Optional


@dataclass
class TemplateContext:
    """Context for template rendering.

    Attributes:
        variables: Template variables
        metadata: Additional metadata
    """

    variables: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_variable(self, key: str, value: Any) -> "TemplateContext":
        """Add a variable to the context."""
        return TemplateContext(
            variables={**self.variables, key: value},
            metadata=self.metadata,
        )

    def with_variables(self, **kwargs: Any) -> "TemplateContext":
        """Add multiple variables."""
        return TemplateContext(
            variables={**self.variables, **kwargs},
            metadata=self.metadata,
        )

    def merge(self, other: "TemplateContext") -> "TemplateContext":
        """Merge with another context."""
        return TemplateContext(
            variables={**self.variables, **other.variables},
            metadata={**self.metadata, **other.metadata},
        )


@dataclass
class PromptTemplate:
    """A template for HITL prompts.

    Attributes:
        name: Template identifier
        template_string: The template string
        description: Description of the template
        required_variables: List of required variable names
        optional_variables: List of optional variable names
    """

    name: str
    template_string: str
    description: str = ""
    required_variables: list[str] = field(default_factory=list)
    optional_variables: list[str] = field(default_factory=list)

    def render(
        self,
        context: Optional[TemplateContext] = None,
        **kwargs: Any,
    ) -> str:
        """Render the template with given context.

        Args:
            context: TemplateContext with variables
            **kwargs: Additional variables

        Returns:
            Rendered string
        """
        variables = {}
        if context:
            variables.update(context.variables)
        variables.update(kwargs)

        # Check required variables
        missing = [v for v in self.required_variables if v not in variables]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        try:
            return Template(self.template_string).safe_substitute(variables)
        except (KeyError, ValueError) as e:
            raise ValueError(f"Template rendering failed: {e}")

    def validate_context(self, context: TemplateContext) -> tuple[bool, Optional[str]]:
        """Validate that context has all required variables.

        Args:
            context: Context to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        missing = [v for v in self.required_variables if v not in context.variables]
        if missing:
            return False, f"Missing required variables: {missing}"
        return True, None

    def extract_variables(self) -> list[str]:
        """Extract all variable names from the template.

        Returns:
            List of variable names found in the template
        """
        pattern = r"\$(?:(\w+)|\{(\w+)\})"
        matches = re.findall(pattern, self.template_string)
        # flatten tuples and deduplicate
        return list(set(m for group in matches for m in group if m))


class PromptTemplateRegistry:
    """Registry for HITL prompt templates.

    Provides thread-safe access to predefined and custom templates.
    """

    _instance: Optional["PromptTemplateRegistry"] = None
    _initialized: bool = False

    def __new__(cls) -> "PromptTemplateRegistry":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the registry with default templates."""
        if PromptTemplateRegistry._initialized:
            return
        PromptTemplateRegistry._initialized = True

        self._templates: dict[str, PromptTemplate] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default templates."""
        # Approval templates
        self.register(
            PromptTemplate(
                name="approval.default",
                template_string="Please approve the following action:\n\n$action",
                description="Default approval prompt",
                required_variables=["action"],
            )
        )
        self.register(
            PromptTemplate(
                name="approval.deployment",
                template_string=(
                    "Deployment Approval Required\n\n"
                    "Action: $action\n"
                    "Environment: $environment\n"
                    "Version: $version\n\n"
                    "Please confirm you approve this deployment."
                ),
                description="Deployment approval prompt",
                required_variables=["action", "environment", "version"],
            )
        )
        self.register(
            PromptTemplate(
                name="approval.destructive",
                template_string=(
                    "⚠️ DESTRUCTIVE ACTION APPROVAL ⚠️\n\n"
                    "Action: $action\n"
                    "Target: $target\n"
                    "Risk Level: $risk\n\n"
                    "This action CANNOT be undone. Please confirm."
                ),
                description="Destructive action approval",
                required_variables=["action", "target", "risk"],
            )
        )

        # Text input templates
        self.register(
            PromptTemplate(
                name="input.default",
                template_string="$prompt",
                description="Default input prompt",
                required_variables=["prompt"],
            )
        )
        self.register(
            PromptTemplate(
                name="input.deployment_notes",
                template_string=(
                    "Please provide deployment notes for $version:\n\n"
                    "- What changes are included?\n"
                    "- What are the potential risks?\n"
                    "- What monitoring should be done?"
                ),
                description="Deployment notes input",
                required_variables=["version"],
            )
        )

        # Choice templates
        self.register(
            PromptTemplate(
                name="choice.default",
                template_string="$prompt\n\n$options",
                description="Default choice prompt",
                required_variables=["prompt", "options"],
            )
        )
        self.register(
            PromptTemplate(
                name="choice.environment",
                template_string="Select the deployment environment:\n\n$options",
                description="Environment selection",
                required_variables=["options"],
            )
        )

        # Review templates
        self.register(
            PromptTemplate(
                name="review.default",
                template_string=(
                    "Please review the following:\n\n$content\n\n"
                    "Options:\n"
                    "- [A]pprove as-is\n"
                    "- [M]odify and approve\n"
                    "- [R]eject"
                ),
                description="Default review prompt",
                required_variables=["content"],
            )
        )
        self.register(
            PromptTemplate(
                name="review.code",
                template_string=(
                    "Code Review Required\n\n"
                    "File: $file\n"
                    "Changes:\n$diff\n\n"
                    "Please review for:\n"
                    "- Correctness\n"
                    "- Security\n"
                    "- Performance\n"
                    "- Style"
                ),
                description="Code review prompt",
                required_variables=["file", "diff"],
            )
        )

        # Confirmation templates
        self.register(
            PromptTemplate(
                name="confirmation.default",
                template_string="$prompt\n\n[Y]es / [N]o",
                description="Default confirmation prompt",
                required_variables=["prompt"],
            )
        )
        self.register(
            PromptTemplate(
                name="confirmation.deployment",
                template_string=(
                    "Confirm deployment to $environment:\n\n"
                    "Version: $version\n"
                    "Services: $services\n\n"
                    "Proceed? [Y/n]"
                ),
                description="Deployment confirmation",
                required_variables=["environment", "version", "services"],
            )
        )

    def register(self, template: PromptTemplate) -> None:
        """Register a new template.

        Args:
            template: The template to register
        """
        self._templates[template.name] = template

    def get(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name.

        Args:
            name: Template name

        Returns:
            The template or None if not found
        """
        return self._templates.get(name)

    def list_templates(self, category: Optional[str] = None) -> list[str]:
        """List available template names.

        Args:
            category: Optional category filter (e.g., "approval", "input")

        Returns:
            List of template names
        """
        if category:
            return [name for name in self._templates if name.startswith(f"{category}.")]
        return list(self._templates.keys())

    def render(
        self,
        template_name: str,
        context: Optional[TemplateContext] = None,
        **kwargs: Any,
    ) -> str:
        """Render a template by name.

        Args:
            template_name: Name of the template
            context: TemplateContext with variables
            **kwargs: Additional variables

        Returns:
            Rendered string
        """
        template = self.get(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        return template.render(context, **kwargs)

    def create_from_string(self, name: str, template_string: str) -> PromptTemplate:
        """Create a template from a string.

        Args:
            name: Template name
            template_string: Template content

        Returns:
            The created template
        """
        template = PromptTemplate(name=name, template_string=template_string)
        template.optional_variables = template.extract_variables()
        self.register(template)
        return template


# Global registry instance
_registry: Optional[PromptTemplateRegistry] = None


def get_registry() -> PromptTemplateRegistry:
    """Get the global template registry."""
    global _registry
    if _registry is None:
        _registry = PromptTemplateRegistry()
    return _registry


def register_template(template: PromptTemplate) -> None:
    """Register a template in the global registry.

    Args:
        template: The template to register
    """
    get_registry().register(template)


def get_prompt_template(name: str) -> Optional[PromptTemplate]:
    """Get a template from the global registry.

    Args:
        name: Template name

    Returns:
        The template or None if not found
    """
    return get_registry().get(name)


def render_template(
    template_name: str,
    context: Optional[TemplateContext] = None,
    **kwargs: Any,
) -> str:
    """Render a template by name.

    Args:
        template_name: Name of the template
        context: TemplateContext with variables
        **kwargs: Additional variables

    Returns:
        Rendered string
    """
    return get_registry().render(template_name, context, **kwargs)


def list_templates(category: Optional[str] = None) -> list[str]:
    """List available template names.

    Args:
        category: Optional category filter

    Returns:
        List of template names
    """
    return get_registry().list_templates(category)


__all__ = [
    "TemplateContext",
    "PromptTemplate",
    "PromptTemplateRegistry",
    "get_registry",
    "register_template",
    "get_prompt_template",
    "render_template",
    "list_templates",
]
