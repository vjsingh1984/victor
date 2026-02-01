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

"""RAG-specific prompt template using PromptBuilderTemplate.

This module provides the Template Method pattern for consistent prompt structure
for the RAG vertical.

Usage:
    from victor.rag.rag_prompt_template import RAGPromptTemplate

    template = RAGPromptTemplate()
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


class RAGPromptTemplate(PromptBuilderTemplate):
    """Template Method pattern for RAG vertical prompts.

    Provides consistent prompt structure with hook methods that can be
    customized for RAG-specific requirements.

    Attributes:
        vertical_name: "rag"
    """

    vertical_name: str = "rag"

    def get_grounding(self) -> Optional[dict[str, Any]]:
        """Get grounding configuration for the prompt.

        Returns:
            Dictionary with 'template', 'variables', and optional 'priority'
        """
        return {
            "template": "Context: You are a RAG assistant with access to a knowledge base for {domain}.",
            "variables": {"domain": "a specific domain"},
            "priority": 10,
        }

    def get_rules(self) -> list[str]:
        """Get list of rules for the prompt.

        Returns:
            List of rule strings
        """
        return [
            "Always search the knowledge base before answering",
            "Cite sources with document names and locations",
            "If information is not in the knowledge base, say so clearly",
            "Use retrieved context to inform your answers",
            "Distinguish between general knowledge and retrieved information",
            "Synthesize information from multiple sources when available",
            "Provide source attribution for all factual claims",
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
            "Knowledge base was searched for relevant information",
            "Retrieved context is used in the answer",
            "Sources are properly cited with document references",
            "General knowledge is distinguished from retrieved information",
            "Answer is grounded in retrieved context",
            "Limitations are acknowledged if information is unavailable",
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
        return """You are a Retrieval-Augmented Generation (RAG) assistant with expertise in:
- Document ingestion and indexing
- Semantic search and retrieval
- Context-aware question answering
- Source attribution and citation
- Knowledge base management
- Synthesizing information from multiple documents"""

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


__all__ = ["RAGPromptTemplate"]
