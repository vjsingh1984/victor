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

"""Coding vertical enrichment strategy.

Provides prompt enrichment using the codebase knowledge graph to inject
relevant code context such as:
- Symbol definitions and signatures
- Related functions and classes
- Import dependencies
- Code snippets from mentioned files

Example:
    from victor.coding.enrichment import CodingEnrichmentStrategy

    # Create strategy with knowledge graph
    strategy = CodingEnrichmentStrategy(graph_store=graph_store)

    # Register with enrichment service
    enrichment_service.register_strategy("coding", strategy)

    # Enrichments are now automatically applied for coding tasks
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from victor.framework.enrichment import (
    ContextEnrichment,
    EnrichmentContext,
    EnrichmentPriority,
    EnrichmentType,
)

if TYPE_CHECKING:
    from victor.storage.graph.protocol import GraphStoreProtocol, GraphNode

logger = logging.getLogger(__name__)


def _extract_symbols_from_prompt(prompt: str) -> List[str]:
    """Extract potential code symbols from a prompt.

    Looks for:
    - CamelCase identifiers (class names)
    - snake_case identifiers (function names)
    - Dot-separated paths (module.function)

    Args:
        prompt: The prompt text to analyze

    Returns:
        List of potential symbol names
    """
    symbols = []

    # CamelCase patterns (class names)
    camel_pattern = r"\b([A-Z][a-zA-Z0-9]*(?:[A-Z][a-zA-Z0-9]*)+)\b"
    symbols.extend(re.findall(camel_pattern, prompt))

    # snake_case patterns (function/variable names)
    snake_pattern = r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b"
    symbols.extend(re.findall(snake_pattern, prompt))

    # Dot-separated paths (module.function)
    path_pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)\b"
    for match in re.findall(path_pattern, prompt):
        # Add both full path and last component
        symbols.append(match)
        symbols.append(match.split(".")[-1])

    # Backtick-quoted code
    backtick_pattern = r"`([a-zA-Z_][a-zA-Z0-9_]*)`"
    symbols.extend(re.findall(backtick_pattern, prompt))

    # Remove common words and duplicates
    common_words = {
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "from",
        "have",
        "are",
        "was",
        "were",
        "been",
        "being",
        "has",
        "had",
        "can",
        "could",
        "will",
        "would",
        "should",
        "may",
        "might",
        "must",
    }
    symbols = [s for s in symbols if s.lower() not in common_words]

    return list(set(symbols))


def _format_graph_node(node: "GraphNode") -> str:
    """Format a graph node for enrichment content.

    Args:
        node: The graph node to format

    Returns:
        Formatted string representation
    """
    parts = []

    # Type and name
    type_str = node.type.title() if node.type else "Symbol"
    parts.append(f"**{type_str}**: `{node.name}`")

    # Location
    if node.file:
        location = node.file
        if node.line:
            location += f":{node.line}"
            if node.end_line and node.end_line != node.line:
                location += f"-{node.end_line}"
        parts.append(f"Location: {location}")

    # Signature
    if node.signature:
        parts.append(f"Signature: `{node.signature}`")

    # Docstring (truncated)
    if node.docstring:
        doc = node.docstring[:200]
        if len(node.docstring) > 200:
            doc += "..."
        parts.append(f"Description: {doc}")

    return "\n".join(parts)


class CodingEnrichmentStrategy:
    """Enrichment strategy for the coding vertical.

    Uses the codebase knowledge graph to provide relevant code context
    including symbol definitions, relationships, and code snippets.

    Attributes:
        graph_store: Graph store for symbol queries
        max_symbols: Maximum symbols to include per enrichment
        max_snippets: Maximum code snippets to include
    """

    def __init__(
        self,
        graph_store: Optional["GraphStoreProtocol"] = None,
        max_symbols: int = 5,
        max_snippets: int = 3,
    ):
        """Initialize the coding enrichment strategy.

        Args:
            graph_store: Optional graph store for symbol queries
            max_symbols: Max symbols to include (default: 5)
            max_snippets: Max code snippets to include (default: 3)
        """
        self._graph_store = graph_store
        self._max_symbols = max_symbols
        self._max_snippets = max_snippets

    def set_graph_store(self, graph_store: "GraphStoreProtocol") -> None:
        """Set or update the graph store.

        Args:
            graph_store: Graph store instance
        """
        self._graph_store = graph_store

    async def get_enrichments(
        self,
        prompt: str,
        context: EnrichmentContext,
    ) -> List[ContextEnrichment]:
        """Get enrichments for a coding prompt.

        Queries the knowledge graph for:
        1. Symbols explicitly mentioned in the prompt
        2. Symbols from mentioned files
        3. Related symbols based on the task type

        Args:
            prompt: The prompt to enrich
            context: Enrichment context with file/symbol mentions

        Returns:
            List of context enrichments
        """
        enrichments: List[ContextEnrichment] = []

        # If no graph store, return empty
        if not self._graph_store:
            logger.debug("No graph store available for coding enrichment")
            return enrichments

        try:
            # 1. Get symbols mentioned in the context
            if context.symbol_mentions:
                symbol_enrichment = await self._enrich_from_symbols(context.symbol_mentions)
                if symbol_enrichment:
                    enrichments.append(symbol_enrichment)

            # 2. Get context from mentioned files
            if context.file_mentions:
                file_enrichment = await self._enrich_from_files(context.file_mentions)
                if file_enrichment:
                    enrichments.append(file_enrichment)

            # 3. Extract and search for symbols in the prompt
            prompt_symbols = _extract_symbols_from_prompt(prompt)
            if prompt_symbols:
                search_enrichment = await self._enrich_from_search(prompt_symbols)
                if search_enrichment:
                    enrichments.append(search_enrichment)

        except Exception as e:
            logger.warning("Error during coding enrichment: %s", e)

        logger.debug(
            "Coding enrichment produced %d enrichments for task_type=%s",
            len(enrichments),
            context.task_type,
        )

        return enrichments

    async def _enrich_from_symbols(
        self,
        symbols: List[str],
    ) -> Optional[ContextEnrichment]:
        """Enrich from explicitly mentioned symbols.

        Args:
            symbols: List of symbol names

        Returns:
            Enrichment with symbol information, or None
        """
        if not self._graph_store or not symbols:
            return None

        found_symbols = []

        for symbol in symbols[: self._max_symbols]:
            try:
                nodes = await self._graph_store.find_nodes(name=symbol)
                if nodes:
                    found_symbols.extend(nodes[:2])  # Max 2 matches per symbol
            except Exception as e:
                logger.debug("Error searching for symbol %s: %s", symbol, e)

        if not found_symbols:
            return None

        # Format the symbols
        content_parts = []
        for node in found_symbols[: self._max_symbols]:
            content_parts.append(_format_graph_node(node))
            content_parts.append("")  # Blank line between

        content = "\n".join(content_parts).strip()

        return ContextEnrichment(
            type=EnrichmentType.KNOWLEDGE_GRAPH,
            content=content,
            priority=EnrichmentPriority.HIGH,
            source="knowledge_graph",
            metadata={"symbols_found": len(found_symbols)},
        )

    async def _enrich_from_files(
        self,
        files: List[str],
    ) -> Optional[ContextEnrichment]:
        """Enrich from mentioned files.

        Args:
            files: List of file paths

        Returns:
            Enrichment with file symbols, or None
        """
        if not self._graph_store or not files:
            return None

        all_nodes = []

        for file_path in files[: self._max_snippets]:
            try:
                nodes = await self._graph_store.get_nodes_by_file(file_path)
                if nodes:
                    # Sort by importance (classes/functions first)
                    nodes.sort(
                        key=lambda n: (
                            0 if n.type in ("class", "function") else 1,
                            n.line or 0,
                        )
                    )
                    all_nodes.extend(nodes[:3])  # Max 3 per file
            except Exception as e:
                logger.debug("Error getting symbols for file %s: %s", file_path, e)

        if not all_nodes:
            return None

        # Format as file summaries
        content_parts = []
        current_file = None

        for node in all_nodes[: self._max_symbols]:
            if node.file != current_file:
                current_file = node.file
                content_parts.append(f"\n### {current_file}")

            content_parts.append(f"- `{node.name}` ({node.type})")
            if node.signature:
                content_parts.append(f"  Signature: `{node.signature}`")

        content = "\n".join(content_parts).strip()

        return ContextEnrichment(
            type=EnrichmentType.CODE_SNIPPET,
            content=content,
            priority=EnrichmentPriority.NORMAL,
            source="file_analysis",
            metadata={"files_analyzed": len(files)},
        )

    async def _enrich_from_search(
        self,
        symbols: List[str],
    ) -> Optional[ContextEnrichment]:
        """Enrich by searching the knowledge graph.

        Args:
            symbols: List of potential symbol names from prompt

        Returns:
            Enrichment with search results, or None
        """
        if not self._graph_store or not symbols:
            return None

        # Create search query from symbols
        query = " ".join(symbols[:5])  # Use first 5 symbols

        try:
            results = await self._graph_store.search_symbols(
                query,
                limit=self._max_symbols,
            )
        except Exception as e:
            logger.debug("Error searching knowledge graph: %s", e)
            return None

        if not results:
            return None

        # Format search results
        content_parts = ["Related symbols found in codebase:"]

        for node in results:
            parts = [f"- `{node.name}`"]
            if node.type:
                parts.append(f"({node.type})")
            if node.file:
                loc = node.file
                if node.line:
                    loc += f":{node.line}"
                parts.append(f"at {loc}")
            content_parts.append(" ".join(parts))

        content = "\n".join(content_parts)

        return ContextEnrichment(
            type=EnrichmentType.KNOWLEDGE_GRAPH,
            content=content,
            priority=EnrichmentPriority.LOW,
            source="knowledge_graph_search",
            metadata={"results_count": len(results)},
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
            Fraction of token budget (0.4 = 40%)
        """
        return 0.4


__all__ = [
    "CodingEnrichmentStrategy",
]
