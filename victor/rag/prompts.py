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

"""RAG Prompt Contributor - Task hints and prompt contributions for RAG."""

from typing import Any

from victor.core.vertical_types import TaskTypeHint, StandardTaskHints
from victor.core.verticals.protocols import PromptContributorProtocol


class RAGPromptContributor(PromptContributorProtocol):
    """Prompt contributor for RAG vertical.

    Provides task-type hints and prompt contributions for RAG operations.
    """

    def get_task_type_hints(self) -> dict[str, TaskTypeHint]:
        """Get task type hints for RAG operations.

        Returns:
            Dictionary of task type to hints
        """
        hints = {
            "document_ingestion": TaskTypeHint(
                task_type="document_ingestion",
                hint="Ingesting documents into the knowledge base",
                tool_budget=5,
                priority_tools=["rag_ingest", "read", "ls"],
            ),
            "knowledge_search": TaskTypeHint(
                task_type="knowledge_search",
                hint="Searching the knowledge base",
                tool_budget=3,
                priority_tools=["rag_search", "rag_query"],
            ),
            "question_answering": TaskTypeHint(
                task_type="question_answering",
                hint="Answering questions from the knowledge base",
                tool_budget=5,
                priority_tools=["rag_query", "rag_search"],
            ),
            "knowledge_management": TaskTypeHint(
                task_type="knowledge_management",
                hint="Managing the knowledge base",
                tool_budget=5,
                priority_tools=["rag_list", "rag_delete", "rag_stats"],
            ),
        }
        return StandardTaskHints.merge_with(hints)

    def get_system_prompt_section(self) -> str:
        """Get system prompt section for RAG.

        Returns:
            System prompt section text
        """
        return """
## RAG Operations

When answering questions:
1. Always use rag_query first to retrieve relevant context
2. If context is insufficient, use rag_search for broader exploration
3. Cite sources using [Source N] notation from the retrieved context
4. If the answer isn't in the knowledge base, clearly state this

When ingesting documents:
1. Verify the file exists using read or ls
2. Use appropriate doc_type (text, markdown, code, pdf)
3. Report successful ingestion with chunk count

Citation Format:
- Use [1], [2] etc. to reference sources from rag_query results
- Include the source name when summarizing multiple sources
"""

    def get_context_hints(self, context: dict[str, Any]) -> list[str]:
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
