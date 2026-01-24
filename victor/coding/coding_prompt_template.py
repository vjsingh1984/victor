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

"""Coding-specific prompt template using PromptBuilderTemplate.

This module provides the Template Method pattern for consistent prompt structure
for the coding vertical.

Usage:
    from victor.coding.coding_prompt_template import CodingPromptTemplate

    template = CodingPromptTemplate()
    builder = template.get_prompt_builder()
    prompt = builder.build()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.framework.prompt_builder_template import PromptBuilderTemplate

if TYPE_CHECKING:
    from victor.framework.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class CodingPromptTemplate(PromptBuilderTemplate):
    """Template Method pattern for coding vertical prompts.

    Provides consistent prompt structure with hook methods that can be
    customized for coding-specific requirements.

    Attributes:
        vertical_name: "coding"
    """

    vertical_name: str = "coding"

    def get_grounding(self) -> Optional[Dict[str, Any]]:
        """Get grounding configuration for the prompt.

        Returns:
            Dictionary with 'template', 'variables', and optional 'priority'
        """
        return {
            "template": "Context: You are assisting with software development for {project}.",
            "variables": {"project": "a coding project"},
            "priority": 10,
        }

    def get_rules(self) -> List[str]:
        """Get list of rules for the prompt.

        Returns:
            List of rule strings
        """
        return [
            "Always read existing files before making changes",
            "Understand the broader codebase context",
            "Follow existing code style and conventions",
            "Add appropriate error handling",
            "Write or update tests for new code",
            "Consider edge cases and potential issues",
            "Document non-obvious code",
            "Ensure changes don't break existing functionality",
        ]

    def get_rules_priority(self) -> int:
        """Get priority for rules section.

        Returns:
            Priority value (lower = earlier in prompt)
        """
        return 20

    def get_checklist(self) -> List[str]:
        """Get checklist items for the prompt.

        Returns:
            List of checklist item strings
        """
        return [
            "Code compiles without errors",
            "Tests pass (add tests if needed)",
            "Code follows project style",
            "No obvious bugs or issues",
            "Error handling is appropriate",
            "Changes are minimal and focused",
        ]

    def get_checklist_priority(self) -> int:
        """Get priority for checklist section.

        Returns:
            Priority value (lower = earlier in prompt)
        """
        return 30

    def get_vertical_prompt(self) -> str:
        """Get vertical-specific prompt content.

        Returns:
            Vertical-specific prompt content
        """
        return """You are an expert software developer assistant.

Your role is to help with software development tasks including:
- Writing and modifying code
- Debugging and troubleshooting
- Code review and optimization
- Testing and test generation
- Documentation

Debugging complex issues
Following project conventions and style guides"""

    def pre_build(self, builder: "PromptBuilder") -> "PromptBuilder":
        """Hook called before building the prompt.

        Args:
            builder: The configured PromptBuilder

        Returns:
            The modified PromptBuilder
        """
        # Add custom sections or modify builder before building
        # This is where vertical-specific customizations can go
        return builder


__all__ = ["CodingPromptTemplate"]
