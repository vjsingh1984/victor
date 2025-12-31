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

"""Research vertical enrichment strategy.

Provides prompt enrichment using web search and conversation history
to inject relevant research context such as:
- Pre-fetched web search results for fact grounding
- Source citations and references
- Relevant snippets from previous research conversations

This module delegates to the framework's enrichment utilities
(victor.framework.enrichment) for domain-agnostic functionality.

Example:
    from victor.research.enrichment import ResearchEnrichmentStrategy

    # Create strategy
    strategy = ResearchEnrichmentStrategy()

    # Register with enrichment service
    enrichment_service.register_strategy("research", strategy)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Awaitable

from victor.agent.prompt_enrichment import (
    ContextEnrichment,
    EnrichmentContext,
    EnrichmentPriority,
    EnrichmentType,
)

# Import framework enrichment utilities (DRY principle)
from victor.framework.enrichment import (
    extract_search_terms as _framework_extract_search_terms,
    format_web_results as _framework_format_web_results,
    extract_tool_context as _framework_extract_tool_context,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# DEPRECATED: Use victor.framework.enrichment.extract_search_terms instead
# Kept for backward compatibility
def _extract_search_terms(prompt: str) -> List[str]:
    """Extract potential search terms from a prompt.

    DEPRECATED: Use victor.framework.enrichment.extract_search_terms instead.

    Args:
        prompt: The prompt text to analyze

    Returns:
        List of search term candidates
    """
    return _framework_extract_search_terms(prompt, max_terms=5)


class ResearchEnrichmentStrategy:
    """Enrichment strategy for the research vertical.

    Uses web search to provide relevant factual context and
    citation-ready information for research tasks.

    Attributes:
        web_search_fn: Async function to perform web searches
        max_results: Maximum search results to include
    """

    def __init__(
        self,
        web_search_fn: Optional[Callable[[str], Awaitable[List[Dict[str, Any]]]]] = None,
        max_results: int = 3,
    ):
        """Initialize the research enrichment strategy.

        Args:
            web_search_fn: Optional async function for web searches
            max_results: Max search results to include (default: 3)
        """
        self._web_search_fn = web_search_fn
        self._max_results = max_results

    def set_web_search_fn(
        self,
        fn: Callable[[str], Awaitable[List[Dict[str, Any]]]],
    ) -> None:
        """Set the web search function.

        Args:
            fn: Async function that takes a query and returns search results
        """
        self._web_search_fn = fn

    async def get_enrichments(
        self,
        prompt: str,
        context: EnrichmentContext,
    ) -> List[ContextEnrichment]:
        """Get enrichments for a research prompt.

        Performs web searches based on prompt content to provide
        factual context and source citations.

        Args:
            prompt: The prompt to enrich
            context: Enrichment context with task metadata

        Returns:
            List of context enrichments
        """
        enrichments: List[ContextEnrichment] = []

        # Extract search terms from the prompt
        search_terms = _extract_search_terms(prompt)

        if not search_terms:
            logger.debug("No search terms extracted from research prompt")
            return enrichments

        # If we have a web search function, use it
        if self._web_search_fn:
            try:
                search_enrichment = await self._enrich_from_web_search(
                    search_terms,
                    context.task_type,
                )
                if search_enrichment:
                    enrichments.append(search_enrichment)
            except Exception as e:
                logger.warning("Error during web search enrichment: %s", e)

        # Add conversation context if available
        if context.tool_history:
            history_enrichment = self._enrich_from_tool_history(context.tool_history)
            if history_enrichment:
                enrichments.append(history_enrichment)

        logger.debug(
            "Research enrichment produced %d enrichments for task_type=%s",
            len(enrichments),
            context.task_type,
        )

        return enrichments

    async def _enrich_from_web_search(
        self,
        terms: List[str],
        task_type: Optional[str],
    ) -> Optional[ContextEnrichment]:
        """Enrich from web search results.

        Delegates formatting to framework enrichment utilities.

        Args:
            terms: Search terms to query
            task_type: Type of research task

        Returns:
            Enrichment with search results, or None
        """
        if not self._web_search_fn or not terms:
            return None

        # Build search query
        query = " ".join(terms)

        try:
            results = await self._web_search_fn(query)
        except Exception as e:
            logger.debug("Web search error: %s", e)
            return None

        if not results:
            return None

        # Use framework utility for formatting
        content = _framework_format_web_results(
            results,
            max_results=self._max_results,
            max_snippet_length=200,
            include_urls=True,
        )

        return ContextEnrichment(
            type=EnrichmentType.WEB_SEARCH,
            content=content,
            priority=EnrichmentPriority.HIGH,
            source="web_search",
            metadata={"results_count": len(results), "query": query},
        )

    def _enrich_from_tool_history(
        self,
        tool_history: List[Dict[str, Any]],
    ) -> Optional[ContextEnrichment]:
        """Enrich from previous tool call results.

        Delegates to framework enrichment utilities for extraction.
        Research vertical filters for web search/fetch tools.

        Args:
            tool_history: List of recent tool calls

        Returns:
            Enrichment with relevant prior results, or None
        """
        # Use framework utility for extraction (research-specific tool filter)
        content = _framework_extract_tool_context(
            tool_history,
            tool_names={"web_search", "web_fetch"},  # Research-specific
            max_results=3,
            max_content_length=300,
            header="Prior research in this session:",
        )

        if not content:
            return None

        return ContextEnrichment(
            type=EnrichmentType.CONVERSATION,
            content=content,
            priority=EnrichmentPriority.NORMAL,
            source="tool_history",
            metadata={"source": "framework_enrichment"},
        )

    def get_priority(self) -> int:
        """Get strategy priority.

        Returns:
            Priority value (50 = normal)
        """
        return 50

    def get_token_allocation(self) -> float:
        """Get token budget allocation.

        Returns:
            Fraction of token budget (0.35 = 35%)
        """
        return 0.35


__all__ = [
    "ResearchEnrichmentStrategy",
]
