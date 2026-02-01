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

"""Research-specific prompt template using PromptBuilderTemplate.

This module provides the Template Method pattern for consistent prompt structure
for the research vertical.

Usage:
    from victor.research.research_prompt_template import ResearchPromptTemplate

    template = ResearchPromptTemplate()
    builder = template.get_prompt_builder()
    prompt = builder.build()
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

from victor.framework.prompt_builder_template import PromptBuilderTemplate

if TYPE_CHECKING:
    from victor.framework.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class ResearchPromptTemplate(PromptBuilderTemplate):
    """Template Method pattern for research vertical prompts.

    Provides consistent prompt structure with hook methods that can be
    customized for research-specific requirements.

    Attributes:
        vertical_name: "research"
    """

    vertical_name: str = "research"

    def get_grounding(self) -> Optional[dict[str, Any]]:
        """Get grounding configuration for the prompt.

        Returns:
            Dictionary with 'template', 'variables', and optional 'priority'
        """
        return {
            "template": "Context: You are conducting research on {topic}.",
            "variables": {"topic": "a research question"},
            "priority": 10,
        }

    def get_rules(self) -> list[str]:
        """Get list of rules for the prompt.

        Returns:
            List of rule strings
        """
        return [
            "Use multiple sources to verify information",
            "Distinguish between facts and opinions",
            "Cite sources appropriately",
            "Consider potential biases",
            "Update knowledge based on new evidence",
        ]

    def get_rules_priority(self) -> int:
        """Get priority for rules section.

        Returns:
            Priority value (lower = earlier in prompt)
        """
        return 20

    def get_checklist(self) -> list[str]:
        """Get checklist items for the prompt.

        Returns:
            List of checklist item strings
        """
        return [
            "Information is from credible sources",
            "Multiple perspectives considered",
            "Claims are evidence-based",
            "Sources are properly cited",
            "Conclusions are well-supported",
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
        return """You are an expert researcher with strong skills in:
- Information gathering and synthesis
- Critical analysis and evaluation
- Identifying credible sources
- Drawing evidence-based conclusions
- Clear communication of findings"""

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


__all__ = ["ResearchPromptTemplate"]
