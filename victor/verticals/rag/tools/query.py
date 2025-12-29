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

"""RAG Query Tool - Query with automatic context retrieval."""

import logging
from typing import Any, Dict, List, Optional

from victor.tools.base import BaseTool, CostTier, ToolResult

logger = logging.getLogger(__name__)


class RAGQueryTool(BaseTool):
    """Query the RAG knowledge base with automatic context retrieval.

    Retrieves relevant context and formats it for LLM consumption,
    including source citations.

    Example:
        result = await tool.execute(
            question="What is the authentication flow?",
            k=5,
        )
    """

    name = "rag_query"
    description = (
        "Query the RAG knowledge base and retrieve relevant context for answering. "
        "Returns formatted context with source citations."
    )

    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "Question to answer using the knowledge base",
            },
            "k": {
                "type": "integer",
                "description": "Number of context chunks to retrieve (default: 5)",
                "default": 5,
            },
            "max_context_chars": {
                "type": "integer",
                "description": "Maximum characters of context to return",
                "default": 4000,
            },
        },
        "required": ["question"],
    }

    @property
    def cost_tier(self) -> CostTier:
        return CostTier.LOW

    async def execute(
        self,
        question: str,
        k: int = 5,
        max_context_chars: int = 4000,
        **kwargs,
    ) -> ToolResult:
        """Execute RAG query.

        Args:
            question: Question to answer
            k: Number of context chunks
            max_context_chars: Maximum context length

        Returns:
            ToolResult with formatted context and citations
        """
        from victor.verticals.rag.document_store import DocumentStore

        try:
            store = self._get_document_store()
            await store.initialize()

            # Search for relevant context
            results = await store.search(
                query=question,
                k=k,
                use_hybrid=True,
            )

            if not results:
                return ToolResult(
                    success=True,
                    output=(
                        f"No relevant context found for: '{question}'\n\n"
                        "The knowledge base may not contain information about this topic. "
                        "Consider ingesting relevant documents first."
                    ),
                )

            # Build formatted context
            context_parts = []
            sources = []
            total_chars = 0

            for i, result in enumerate(results, 1):
                chunk = result.chunk
                source = result.doc_source or chunk.metadata.get("source", "unknown")

                # Check if we have space for this chunk
                chunk_text = chunk.content
                if total_chars + len(chunk_text) > max_context_chars:
                    # Truncate to fit
                    remaining = max_context_chars - total_chars
                    if remaining > 100:
                        chunk_text = chunk_text[:remaining] + "..."
                    else:
                        break

                context_parts.append(
                    f"[Source {i}: {source}]\n{chunk_text}"
                )
                sources.append(f"{i}. {source} (relevance: {result.score:.2f})")
                total_chars += len(chunk_text)

            # Format output
            output = (
                f"QUESTION: {question}\n\n"
                f"RETRIEVED CONTEXT ({len(context_parts)} sources):\n"
                f"{'=' * 50}\n\n"
                + "\n\n---\n\n".join(context_parts)
                + f"\n\n{'=' * 50}\n"
                f"SOURCES:\n" + "\n".join(sources)
                + "\n\nUse this context to answer the question. "
                "Cite sources by their number (e.g., [1], [2])."
            )

            return ToolResult(
                success=True,
                output=output,
            )

        except Exception as e:
            logger.exception(f"Query failed: {e}")
            return ToolResult(
                success=False,
                output=f"Query failed: {str(e)}",
            )

    def _get_document_store(self):
        """Get document store instance."""
        from victor.verticals.rag.document_store import DocumentStore

        if not hasattr(self, "_store"):
            self._store = DocumentStore()
        return self._store
