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

"""Runtime prompt contributor adapter for the RAG vertical."""

from __future__ import annotations

from typing import Dict, List

from victor.core.verticals.prompt_adapter import PromptContributorAdapter
from victor.verticals.contrib.rag.prompt_metadata import (
    RAG_SYSTEM_PROMPT_SECTION,
    RAG_TASK_TYPE_HINTS,
)

class RAGPromptContributor(PromptContributorAdapter):
    """Prompt contributor for RAG vertical.

    Wraps serializable RAG prompt metadata for runtime prompt integration.
    """

    def __init__(self) -> None:
        adapter = PromptContributorAdapter.from_dict(
            task_hints=RAG_TASK_TYPE_HINTS,
            system_prompt_section=RAG_SYSTEM_PROMPT_SECTION,
        )
        super().__init__(
            task_hints=adapter.get_task_type_hints(),
            system_prompt_section=adapter.get_system_prompt_section(),
            grounding_rules=adapter.get_grounding_rules(),
            priority=adapter.get_priority(),
        )

    def get_context_hints(self, context: Dict) -> List[str]:
        """Get context-specific hints.

        Args:
            context: Current context dictionary

        Returns:
            List of hints for the current context
        """
        hints = []

        # Check if this looks like a question
        query = context.get("query", "")
        if query and any(
            q in query.lower() for q in ["what", "how", "why", "when", "where", "who", "?"]
        ):
            hints.append("This appears to be a question. Use rag_query to find relevant context.")

        # Check for file-related queries
        if any(word in query.lower() for word in ["add", "ingest", "upload", "index"]):
            hints.append("This appears to be an ingestion request. Use rag_ingest.")

        return hints


__all__ = [
    "RAGPromptContributor",
]
