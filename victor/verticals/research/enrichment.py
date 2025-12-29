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

Example:
    from victor.verticals.research.enrichment import ResearchEnrichmentStrategy

    # Create strategy
    strategy = ResearchEnrichmentStrategy()

    # Register with enrichment service
    enrichment_service.register_strategy("research", strategy)
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Awaitable

from victor.agent.prompt_enrichment import (
    ContextEnrichment,
    EnrichmentContext,
    EnrichmentPriority,
    EnrichmentType,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _extract_search_terms(prompt: str) -> List[str]:
    """Extract potential search terms from a prompt.

    Args:
        prompt: The prompt text to analyze

    Returns:
        List of search term candidates
    """
    # Remove common question patterns
    prompt_clean = re.sub(
        r"^(what|how|why|when|where|who|which|can|could|would|should|is|are|do|does)\s+",
        "",
        prompt.lower(),
    )

    # Extract quoted phrases
    quoted = re.findall(r'"([^"]+)"', prompt)

    # Extract key noun phrases (simplified)
    # Look for capitalized words and technical terms
    terms = re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", prompt)

    # Combine and dedupe
    all_terms = quoted + terms
    return list(dict.fromkeys(all_terms))[:5]  # Max 5 terms


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
            history_enrichment = self._enrich_from_tool_history(
                context.tool_history
            )
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

        # Format search results
        content_parts = ["Relevant web search results:"]

        for i, result in enumerate(results[: self._max_results], 1):
            title = result.get("title", "Untitled")
            snippet = result.get("snippet", "")
            url = result.get("url", "")

            content_parts.append(f"\n{i}. **{title}**")
            if snippet:
                content_parts.append(f"   {snippet[:200]}...")
            if url:
                content_parts.append(f"   Source: {url}")

        content = "\n".join(content_parts)

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

        Extracts relevant context from recent web search or
        web fetch tool results in the conversation.

        Args:
            tool_history: List of recent tool calls

        Returns:
            Enrichment with relevant prior results, or None
        """
        relevant_results = []

        for call in tool_history[-10:]:  # Last 10 calls
            tool_name = call.get("tool", "")

            # Look for web search/fetch results
            if tool_name in ("web_search", "web_fetch"):
                result = call.get("result", {})
                if isinstance(result, dict) and result.get("success"):
                    content = result.get("content", "")
                    if content and len(content) > 50:
                        relevant_results.append({
                            "tool": tool_name,
                            "content": content[:300],
                        })

        if not relevant_results:
            return None

        content_parts = ["Prior research in this session:"]

        for item in relevant_results[: 3]:  # Max 3 prior results
            content_parts.append(f"\n- From {item['tool']}:")
            content_parts.append(f"  {item['content']}...")

        content = "\n".join(content_parts)

        return ContextEnrichment(
            type=EnrichmentType.CONVERSATION,
            content=content,
            priority=EnrichmentPriority.NORMAL,
            source="tool_history",
            metadata={"prior_results": len(relevant_results)},
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
