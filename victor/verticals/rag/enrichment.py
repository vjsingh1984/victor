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

"""RAG Vertical Enrichment Strategy.

Provides prompt enrichment capabilities specific to RAG (Retrieval-Augmented Generation):
- Query expansion with synonyms and related terms
- Document context formatting for LLM synthesis
- Conversation history integration
- Source metadata enrichment

Example:
    from victor.verticals.rag.enrichment import RAGEnrichmentStrategy

    strategy = RAGEnrichmentStrategy()
    enrichments = await strategy.get_context_enrichments(
        prompt="What is the authentication flow?",
        context=EnrichmentContext(
            task_type="query",
            metadata={"doc_sources": ["auth.md", "api.md"]},
        ),
    )
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from victor.agent.prompt_enrichment import (
    ContextEnrichment,
    EnrichmentContext,
    EnrichmentPriority,
    EnrichmentType,
)

logger = logging.getLogger(__name__)


# Common query expansion mappings
QUERY_EXPANSIONS: Dict[str, List[str]] = {
    "auth": ["authentication", "login", "authorization", "credentials"],
    "api": ["endpoint", "rest", "http", "request", "response"],
    "db": ["database", "sql", "query", "table", "schema"],
    "config": ["configuration", "settings", "options", "parameters"],
    "err": ["error", "exception", "failure", "issue", "bug"],
    "perf": ["performance", "speed", "optimization", "latency"],
    "sec": ["security", "vulnerability", "protection", "encryption"],
    "doc": ["documentation", "readme", "guide", "tutorial"],
}


@dataclass
class RAGEnrichmentConfig:
    """Configuration for RAG enrichment.

    Attributes:
        expand_query: Whether to expand query with synonyms
        include_metadata: Whether to include document metadata in enrichment
        max_expansion_terms: Maximum terms to add from expansion
        context_window_chars: Maximum characters of context to include
        prioritize_recent: Whether to prioritize recent documents
    """

    expand_query: bool = True
    include_metadata: bool = True
    max_expansion_terms: int = 3
    context_window_chars: int = 4000
    prioritize_recent: bool = True


class RAGEnrichmentStrategy:
    """Enrichment strategy for RAG vertical.

    Enhances prompts with:
    - Query expansion for better search recall
    - Structured context formatting
    - Source metadata enrichment
    - Conversation history snippets
    """

    def __init__(self, config: Optional[RAGEnrichmentConfig] = None):
        """Initialize RAG enrichment strategy.

        Args:
            config: Optional configuration overrides
        """
        self.config = config or RAGEnrichmentConfig()

    def get_enrichment_priority(self) -> int:
        """Get the priority level for this strategy.

        Returns:
            Priority value (higher = processed first)
        """
        return 80  # High priority for RAG context

    async def get_context_enrichments(
        self,
        prompt: str,
        context: EnrichmentContext,
    ) -> List[ContextEnrichment]:
        """Get context enrichments for a RAG prompt.

        Args:
            prompt: The user's query
            context: Enrichment context with task metadata

        Returns:
            List of enrichments to apply
        """
        enrichments: List[ContextEnrichment] = []

        # 1. Query expansion enrichment
        if self.config.expand_query:
            expansion = self._expand_query(prompt)
            if expansion:
                enrichments.append(
                    ContextEnrichment(
                        type=EnrichmentType.PROJECT_CONTEXT,
                        content=f"[Query hints: {expansion}]",
                        priority=EnrichmentPriority.NORMAL,
                        source="query_expansion",
                        metadata={"expansion_terms": expansion.split(", ")},
                    )
                )

        # 2. Document metadata enrichment
        if self.config.include_metadata:
            doc_sources = context.metadata.get("doc_sources", [])
            if doc_sources:
                metadata_content = self._format_source_metadata(doc_sources)
                enrichments.append(
                    ContextEnrichment(
                        type=EnrichmentType.PROJECT_CONTEXT,
                        content=metadata_content,
                        priority=EnrichmentPriority.HIGH,
                        source="doc_metadata",
                        metadata={"source_count": len(doc_sources)},
                    )
                )

        # 3. Conversation history enrichment
        if context.tool_history:
            history_content = self._format_conversation_history(context.tool_history)
            if history_content:
                enrichments.append(
                    ContextEnrichment(
                        type=EnrichmentType.CONVERSATION,
                        content=history_content,
                        priority=EnrichmentPriority.NORMAL,
                        source="conversation_history",
                    )
                )

        return enrichments

    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms.

        Args:
            query: Original query string

        Returns:
            Comma-separated expansion terms
        """
        query_lower = query.lower()
        expansions: List[str] = []

        for abbrev, terms in QUERY_EXPANSIONS.items():
            # Check if abbreviation appears in query
            if re.search(rf"\b{abbrev}\b", query_lower):
                # Add related terms not already in query
                for term in terms[: self.config.max_expansion_terms]:
                    if term.lower() not in query_lower:
                        expansions.append(term)

        return ", ".join(expansions[:5])  # Max 5 expansion terms

    def _format_source_metadata(self, sources: List[str]) -> str:
        """Format document source metadata for enrichment.

        Args:
            sources: List of document source paths/URLs

        Returns:
            Formatted metadata string
        """
        if not sources:
            return ""

        lines = ["[Source documents:]"]
        for i, source in enumerate(sources[:5], 1):  # Max 5 sources
            # Extract filename from path or URL
            name = source.split("/")[-1] if "/" in source else source
            lines.append(f"  {i}. {name}")

        return "\n".join(lines)

    def _format_conversation_history(self, tool_history: List[Dict[str, Any]]) -> str:
        """Format relevant conversation history.

        Args:
            tool_history: List of recent tool calls

        Returns:
            Formatted history string
        """
        if not tool_history:
            return ""

        # Filter for RAG-related tool calls
        rag_history = [t for t in tool_history if t.get("tool_name", "").startswith("rag_")]

        if not rag_history:
            return ""

        lines = ["[Previous queries:]"]
        for item in rag_history[-3:]:  # Last 3 RAG queries
            query = item.get("args", {}).get("query", "")
            if query:
                lines.append(f"  - {query[:100]}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def enrich_synthesis_prompt(
        self,
        question: str,
        context: str,
        sources: List[str],
        enrichments: Optional[List[ContextEnrichment]] = None,
    ) -> str:
        """Enrich the synthesis prompt with additional context.

        This is the main entry point for enriching RAG synthesis prompts.

        Args:
            question: The user's question
            context: Retrieved document context
            sources: List of source citations
            enrichments: Optional additional enrichments

        Returns:
            Enriched prompt string for LLM synthesis
        """
        parts = []

        # Add enrichment hints if available
        if enrichments:
            hints = [e.content for e in enrichments if e.content]
            if hints:
                parts.append("ENRICHMENT HINTS:\n" + "\n".join(hints))

        # Add structured context
        parts.append(f"CONTEXT FROM {len(sources)} SOURCES:")
        parts.append(context)

        # Add question
        parts.append(f"QUESTION: {question}")

        # Add synthesis instruction
        parts.append(
            "Provide a clear, comprehensive answer based ONLY on the provided context. "
            "Cite sources using [Source N] format. If the context doesn't contain "
            "sufficient information, acknowledge what's missing."
        )

        return "\n\n".join(parts)


# Singleton instance for easy access
_strategy_instance: Optional[RAGEnrichmentStrategy] = None


def get_rag_enrichment_strategy(
    config: Optional[RAGEnrichmentConfig] = None,
) -> RAGEnrichmentStrategy:
    """Get the RAG enrichment strategy singleton.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        RAGEnrichmentStrategy instance
    """
    global _strategy_instance
    if _strategy_instance is None:
        _strategy_instance = RAGEnrichmentStrategy(config)
    return _strategy_instance
