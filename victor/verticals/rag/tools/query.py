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

"""RAG Query Tool - Query with automatic context retrieval and LLM synthesis."""

import logging
from typing import Any, Dict, List, Optional

from victor.agent.prompt_enrichment import EnrichmentContext
from victor.tools.base import BaseTool, CostTier, ToolResult
from victor.verticals.rag.enrichment import get_rag_enrichment_strategy

logger = logging.getLogger(__name__)


# Default RAG system prompt for answer synthesis
RAG_SYSTEM_PROMPT = """You are a helpful assistant answering questions based on retrieved documents.

Instructions:
1. Base your answer ONLY on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Cite sources using [Source N] format
4. Be concise but comprehensive
5. If multiple sources agree, synthesize them into a coherent answer
6. If sources conflict, acknowledge the discrepancy

Do NOT make up information not present in the context."""


class RAGQueryTool(BaseTool):
    """Query the RAG knowledge base with automatic context retrieval and LLM synthesis.

    Retrieves relevant context and uses an LLM to synthesize an answer,
    including source citations.

    Example:
        # Get context only
        result = await tool.execute(question="What is the auth flow?", synthesize=False)

        # Get synthesized answer (default)
        result = await tool.execute(question="What is the auth flow?")

        # Use specific provider/model
        result = await tool.execute(
            question="What is the auth flow?",
            provider="ollama",
            model="llama3.2:3b"
        )
    """

    name = "rag_query"
    description = (
        "Query the RAG knowledge base and synthesize an answer using an LLM. "
        "Returns an answer with source citations grounded in retrieved documents."
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
            "synthesize": {
                "type": "boolean",
                "description": "Use LLM to synthesize answer (default: True)",
                "default": True,
            },
            "provider": {
                "type": "string",
                "description": "LLM provider to use (e.g., 'ollama', 'anthropic', 'openai')",
            },
            "model": {
                "type": "string",
                "description": "Model to use for synthesis (provider-specific)",
            },
            "max_context_chars": {
                "type": "integer",
                "description": "Maximum characters of context to use",
                "default": 4000,
            },
        },
        "required": ["question"],
    }

    @property
    def cost_tier(self) -> CostTier:
        # MEDIUM when synthesizing (LLM call), LOW for context-only
        return CostTier.MEDIUM

    async def execute(
        self,
        question: str,
        k: int = 5,
        synthesize: bool = True,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_context_chars: int = 4000,
        **kwargs,
    ) -> ToolResult:
        """Execute RAG query with optional LLM synthesis.

        Args:
            question: Question to answer
            k: Number of context chunks
            synthesize: Whether to use LLM to synthesize answer
            provider: LLM provider (e.g., 'ollama', 'anthropic')
            model: Model name for synthesis
            max_context_chars: Maximum context length

        Returns:
            ToolResult with synthesized answer or formatted context
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

                context_parts.append(f"[Source {i}: {source}]\n{chunk_text}")
                sources.append(f"{i}. {source} (relevance: {result.score:.2f})")
                total_chars += len(chunk_text)

            # Build context string
            context_str = "\n\n---\n\n".join(context_parts)

            # If not synthesizing, return context only
            if not synthesize:
                output = (
                    f"QUESTION: {question}\n\n"
                    f"RETRIEVED CONTEXT ({len(context_parts)} sources):\n"
                    f"{'=' * 50}\n\n" + context_str + f"\n\n{'=' * 50}\n"
                    f"SOURCES:\n"
                    + "\n".join(sources)
                    + "\n\nUse this context to answer the question. "
                    "Cite sources by their number (e.g., [1], [2])."
                )
                return ToolResult(success=True, output=output)

            # Get enrichment strategy and enrich the prompt
            enrichment_strategy = get_rag_enrichment_strategy()
            enrichment_context = EnrichmentContext(
                task_type="rag_query",
                metadata={
                    "doc_sources": [s.split(" ")[0] for s in sources],  # Extract source paths
                },
            )

            # Get enrichments
            enrichments = await enrichment_strategy.get_context_enrichments(
                prompt=question,
                context=enrichment_context,
            )

            # Synthesize answer using LLM with enriched prompt
            answer = await self._synthesize_answer(
                question=question,
                context=context_str,
                sources=sources,
                provider=provider,
                model=model,
                enrichments=enrichments,
            )

            # Format final output
            output = (
                f"QUESTION: {question}\n\n"
                f"ANSWER:\n{answer}\n\n"
                f"{'=' * 50}\n"
                f"SOURCES USED:\n" + "\n".join(sources)
            )

            return ToolResult(success=True, output=output)

        except Exception as e:
            logger.exception(f"Query failed: {e}")
            return ToolResult(
                success=False,
                output=f"Query failed: {str(e)}",
            )

    async def _synthesize_answer(
        self,
        question: str,
        context: str,
        sources: List[str],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        enrichments: Optional[List[Any]] = None,
    ) -> str:
        """Synthesize an answer using an LLM provider with enrichment.

        Args:
            question: The user's question
            context: Retrieved context from documents
            sources: List of source citations
            provider: LLM provider name
            model: Model name
            enrichments: Optional list of enrichments to include

        Returns:
            Synthesized answer string
        """
        from victor.config.settings import get_settings
        from victor.providers.registry import ProviderRegistry

        settings = get_settings()

        # Determine provider
        if not provider:
            provider = settings.default_provider or "ollama"

        # Get provider instance
        try:
            provider_instance = ProviderRegistry.get(provider)
            if provider_instance is None:
                # Try to create provider
                provider_class = ProviderRegistry.get_class(provider)
                if provider_class is None:
                    raise ValueError(f"Provider '{provider}' not found")
                provider_instance = provider_class()
        except Exception as e:
            logger.warning(f"Failed to get provider {provider}: {e}")
            # Fallback: return context with instruction
            return (
                f"[Could not connect to {provider} for synthesis]\n\n"
                f"Based on the retrieved context:\n{context}\n\n"
                f"Please answer: {question}"
            )

        # Build enriched prompt for synthesis
        enrichment_strategy = get_rag_enrichment_strategy()
        user_prompt = enrichment_strategy.enrich_synthesis_prompt(
            question=question,
            context=context,
            sources=sources,
            enrichments=enrichments,
        )

        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Call provider
        try:
            # Use specified model or provider default
            call_model = model or getattr(provider_instance, "default_model", None)

            response = await provider_instance.chat(
                messages=messages,
                model=call_model,
                temperature=0.3,  # Lower temperature for factual answers
            )

            # Extract answer from response
            if hasattr(response, "content"):
                return response.content
            elif hasattr(response, "message"):
                return response.message.get("content", str(response))
            else:
                return str(response)

        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return (
                f"[Synthesis failed: {str(e)}]\n\n"
                f"Retrieved context for your question:\n{context[:1000]}..."
            )

    def _get_document_store(self):
        """Get document store instance."""
        from victor.verticals.rag.document_store import DocumentStore

        if not hasattr(self, "_store"):
            self._store = DocumentStore()
        return self._store
