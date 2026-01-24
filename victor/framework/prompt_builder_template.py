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

"""Prompt Builder Template for verticals.

Phase 2.3: Create PromptBuilderTemplate.

This module provides a Template Method pattern for consistent prompt structure
across verticals. It reduces boilerplate by providing hook methods that
verticals can override to customize their prompts.

The Template Method pattern defines the skeleton of the prompt building
algorithm, deferring some steps to subclasses. This allows verticals to
customize specific parts of the prompt while maintaining consistent structure.

Usage:
    from victor.framework.prompt_builder_template import PromptBuilderTemplate

    class CodingPromptTemplate(PromptBuilderTemplate):
        vertical_name = "coding"

        def get_grounding(self):
            return {
                "template": "Context: Working on {project}.",
                "variables": {"project": "a coding project"},
                "priority": 10,
            }

        def get_rules(self):
            return [
                "Always read files before editing",
                "Follow coding standards",
            ]

        def get_checklist(self):
            return [
                "Review code quality",
                "Run tests",
            ]

        def get_vertical_prompt(self):
            return "You are an expert software developer."

    template = CodingPromptTemplate()
    prompt = template.build()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.framework.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class PromptBuilderTemplate:
    """Template Method pattern for consistent prompt building.

    This class defines the skeleton of the prompt building algorithm with
    hook methods that can be overridden by subclasses to customize specific
    parts of the prompt.

    Hook Methods (override in subclasses):
        - get_grounding(): Returns grounding configuration
        - get_rules(): Returns list of rules
        - get_checklist(): Returns list of checklist items
        - get_vertical_prompt(): Returns vertical-specific prompt content
        - pre_build(builder): Hook called before building prompt
        - post_build(prompt): Hook called after building prompt

    Attributes:
        vertical_name: Name of the vertical (default: "generic")

    Example:
        class MyTemplate(PromptBuilderTemplate):
            vertical_name = "coding"

            def get_rules(self):
                return ["Follow best practices", "Write tests"]

        template = MyTemplate()
        prompt = template.build()
    """

    # Default section priorities (lower = earlier in prompt)
    DEFAULT_GROUNDING_PRIORITY: int = 10
    DEFAULT_RULES_PRIORITY: int = 20
    DEFAULT_CHECKLIST_PRIORITY: int = 30
    DEFAULT_VERTICAL_PRIORITY: int = 40

    # Default vertical name (override in subclasses)
    vertical_name: str = "generic"

    def __init__(self) -> None:
        """Initialize the prompt builder template."""
        self._builder: Optional["PromptBuilder"] = None

    def get_prompt_builder(self) -> "PromptBuilder":
        """Get a configured PromptBuilder instance.

        Creates a PromptBuilder and applies all hook methods to configure it.
        This is the main entry point for getting a builder that can be
        further customized.

        Returns:
            Configured PromptBuilder instance
        """
        from victor.framework.prompt_builder import PromptBuilder

        builder = PromptBuilder()
        self._builder = builder

        # Apply grounding if provided
        grounding = self.get_grounding()
        if grounding is not None:
            priority = grounding.get("priority", self.DEFAULT_GROUNDING_PRIORITY)
            template = grounding.get("template", "")
            variables = grounding.get("variables", {})
            builder.add_grounding(template, priority=priority, **variables)

        # Apply rules if provided
        rules = self.get_rules()
        if rules:
            rules_priority = (
                self.get_rules_priority()
                if hasattr(self, "get_rules_priority")
                else self.DEFAULT_RULES_PRIORITY
            )
            builder.add_rules(rules, priority=rules_priority)

        # Apply checklist if provided
        checklist = self.get_checklist()
        if checklist:
            checklist_priority = (
                self.get_checklist_priority()
                if hasattr(self, "get_checklist_priority")
                else self.DEFAULT_CHECKLIST_PRIORITY
            )
            builder.add_checklist(checklist, priority=checklist_priority)

        # Apply vertical prompt if provided
        vertical_prompt = self.get_vertical_prompt()
        if vertical_prompt:
            builder.add_vertical_section(
                vertical=self.vertical_name,
                content=vertical_prompt,
                priority=self.DEFAULT_VERTICAL_PRIORITY,
            )

        return builder

    def build(self) -> str:
        """Build the complete prompt string.

        This is the Template Method that defines the algorithm:
        1. Get the configured PromptBuilder
        2. Call pre_build() hook for customization
        3. Build the prompt
        4. Call post_build() hook for final modifications

        Returns:
            The complete prompt string
        """
        # Get configured builder
        builder = self.get_prompt_builder()

        # Call pre_build hook
        builder = self.pre_build(builder)

        # Build the prompt
        prompt = builder.build()

        # Call post_build hook
        prompt = self.post_build(prompt)

        return prompt

    # =========================================================================
    # Hook Methods (override in subclasses)
    # =========================================================================

    def get_grounding(self) -> Optional[Dict[str, Any]]:
        """Get grounding configuration for the prompt.

        Override this method to provide grounding context.

        Returns:
            Dictionary with 'template', 'variables', and optional 'priority',
            or None if no grounding is needed.

        Example:
            def get_grounding(self):
                return {
                    "template": "Context: Working on {project}.",
                    "variables": {"project": "my project"},
                    "priority": 10,
                }
        """
        return None

    def get_rules(self) -> List[str]:
        """Get list of rules for the prompt.

        Override this method to provide vertical-specific rules.

        Returns:
            List of rule strings

        Example:
            def get_rules(self):
                return [
                    "Follow coding standards",
                    "Write tests for new code",
                ]
        """
        return []

    def get_rules_priority(self) -> int:
        """Get priority for rules section.

        Override to customize rules section priority.

        Returns:
            Priority value (lower = earlier in prompt)
        """
        return self.DEFAULT_RULES_PRIORITY

    def get_checklist(self) -> List[str]:
        """Get checklist items for the prompt.

        Override this method to provide task-specific checklist.

        Returns:
            List of checklist item strings

        Example:
            def get_checklist(self):
                return [
                    "Review code changes",
                    "Run tests",
                    "Update documentation",
                ]
        """
        return []

    def get_checklist_priority(self) -> int:
        """Get priority for checklist section.

        Override to customize checklist section priority.

        Returns:
            Priority value (lower = earlier in prompt)
        """
        return self.DEFAULT_CHECKLIST_PRIORITY

    def get_vertical_prompt(self) -> str:
        """Get vertical-specific prompt content.

        Override this method to provide the core vertical-specific content.

        Returns:
            Vertical-specific prompt content

        Example:
            def get_vertical_prompt(self):
                return "You are an expert software developer."
        """
        return ""

    def pre_build(self, builder: "PromptBuilder") -> "PromptBuilder":
        """Hook called before building the prompt.

        Override this method to add additional sections or
        customize the builder before building.

        Args:
            builder: The configured PromptBuilder

        Returns:
            The modified PromptBuilder

        Example:
            def pre_build(self, builder):
                builder.add_section("custom", "Custom content", priority=5)
                return builder
        """
        return builder

    def post_build(self, prompt: str) -> str:
        """Hook called after building the prompt.

        Override this method to perform final modifications
        to the built prompt string.

        Args:
            prompt: The built prompt string

        Returns:
            The modified prompt string

        Example:
            def post_build(self, prompt):
                return prompt + "\\n\\n[End of instructions]"
        """
        return prompt


__all__ = ["PromptBuilderTemplate"]
