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

"""Advanced RAG capabilities with hybrid search, re-ranking, and citations.

This module extends the basic RAG implementation with:
- Hybrid search combining semantic and keyword search
- Advanced re-ranking strategies
- Citation generation and verification
- Query enhancement and expansion
- Multi-stage retrieval pipeline
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from victor.rag.document_store import DocumentStore

logger = logging.getLogger(__name__)


class SearchStrategy(str, Enum):
    """Search strategies for RAG."""

    SEMANTIC = "semantic"  # Pure vector similarity
    KEYWORD = "keyword"  # BM25 or similar
    HYBRID = "hybrid"  # Combined semantic + keyword
    ADAPTIVE = "adaptive"  # Automatically choose based on query


class RerankStrategy(str, Enum):
    """Re-ranking strategies."""

    NONE = "none"  # No re-ranking
    SCORE_BASED = "score_based"  # Re-rank by combined score
    DIVERSITY = "diversity"  # Maximize diversity
    RELEVANCE = "relevance"  # Maximize relevance to query
    CUSTOM = "custom"  # Custom scoring function


@dataclass
class SearchResult:
    """Enhanced search result.

    Attributes:
        chunk_id: Unique chunk identifier
        content: Chunk content
        score: Similarity/relevance score
        source: Source document
        metadata: Additional metadata
        citation: Citation string
        rank: Final rank after re-ranking
    """

    chunk_id: str
    content: str
    score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    citation: str = ""
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata,
            "citation": self.citation,
            "rank": self.rank,
        }


@dataclass
class RAGConfig:
    """Configuration for advanced RAG.

    Attributes:
        search_strategy: Which search strategy to use
        rerank_strategy: How to re-rank results
        top_k: Number of results to retrieve
        rerank_top_k: Number of results to keep after re-ranking
        hybrid_alpha: Weight for hybrid search (0=keyword, 1=semantic)
        enable_citations: Whether to generate citations
        enable_query_expansion: Whether to expand queries
        max_query_expansions: Maximum number of query expansions
    """

    search_strategy: SearchStrategy = SearchStrategy.HYBRID
    rerank_strategy: RerankStrategy = RerankStrategy.RELEVANCE
    top_k: int = 20
    rerank_top_k: int = 10
    hybrid_alpha: float = 0.7  # 70% semantic, 30% keyword
    enable_citations: bool = True
    enable_query_expansion: bool = True
    max_query_expansions: int = 3


class AdvancedRAG:
    """Advanced RAG with hybrid search and re-ranking.

    Example:
        from victor.rag import DocumentStore
        from victor.rag.advanced_rag import AdvancedRAG, RAGConfig

        # Initialize
        store = DocumentStore(path=".victor/rag")
        rag = AdvancedRAG(store, config=RAGConfig())

        # Query with hybrid search
        results = await rag.query(
            "What are the best practices for API design?",
            top_k=10
        )

        # With citations
        results = await rag.query_with_citations(
            "Explain authentication mechanisms",
            top_k=5
        )

        # Get answer with sources
        answer = await rag.generate_answer(
            "How does OAuth work?",
            include_sources=True
        )
    """

    def __init__(
        self,
        document_store: Optional["DocumentStore"] = None,
        config: Optional[RAGConfig] = None,
    ):
        """Initialize advanced RAG.

        Args:
            document_store: DocumentStore instance
            config: RAG configuration
        """
        from victor.rag.document_store import DocumentStore

        self.document_store = document_store
        self.config = config or RAGConfig()

        # Lazy load dependencies
        self._embeddings_service = None
        self._query_enhancer = None

    async def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        search_strategy: Optional[SearchStrategy] = None,
    ) -> List[SearchResult]:
        """Query with advanced search and re-ranking.

        Args:
            query: Search query
            top_k: Number of results to return
            search_strategy: Override search strategy

        Returns:
            List of SearchResult objects
        """
        if not self.document_store:
            logger.warning("No document store configured")
            return []

        top_k = top_k or self.config.top_k
        strategy = search_strategy or self.config.search_strategy

        # Stage 1: Initial retrieval
        if strategy == SearchStrategy.SEMANTIC:
            results = await self._semantic_search(query, top_k)
        elif strategy == SearchStrategy.KEYWORD:
            results = await self._keyword_search(query, top_k)
        elif strategy == SearchStrategy.HYBRID:
            results = await self._hybrid_search(query, top_k)
        else:  # ADAPTIVE
            results = await self._adaptive_search(query, top_k)

        # Stage 2: Re-ranking
        if self.config.rerank_strategy != RerankStrategy.NONE:
            results = await self._rerank_results(query, results)

        # Stage 3: Generate citations
        if self.config.enable_citations:
            results = await self._generate_citations(results)

        return results[: (self.config.rerank_top_k or top_k)]

    async def _semantic_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Pure semantic search using vector similarity."""
        if not self.document_store:
            return []

        try:
            # Use document store's vector search
            results = await self.document_store.search(query, k=top_k)

            return [
                SearchResult(
                    chunk_id=result.chunk.id,
                    content=result.chunk.content,
                    score=result.score,
                    source=result.chunk.metadata.get("source", ""),
                    metadata=result.chunk.metadata,
                )
                for result in results
            ]

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    async def _keyword_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Keyword-based search using BM25 or similar."""
        if not self.document_store:
            return []

        try:
            # For now, use document store's search with keyword filter
            # In production, implement proper BM25 or similar
            results = await self.document_store.search(query, k=top_k)

            # Re-score based on keyword matches
            keyword_results = []
            for result in results:
                # Simple keyword matching score
                keyword_score = self._calculate_keyword_score(query, result.chunk.content)
                keyword_results.append(
                    SearchResult(
                        chunk_id=result.chunk.id,
                        content=result.chunk.content,
                        score=keyword_score,
                        source=result.chunk.metadata.get("source", ""),
                        metadata=result.chunk.metadata,
                    )
                )

            return sorted(keyword_results, key=lambda r: r.score, reverse=True)[:top_k]

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    async def _hybrid_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Hybrid search combining semantic and keyword."""
        # Retrieve more results for combination
        semantic_results = await self._semantic_search(query, top_k * 2)
        keyword_results = await self._keyword_search(query, top_k * 2)

        # Combine scores
        combined_scores: Dict[str, SearchResult] = {}

        # Add semantic scores
        for result in semantic_results:
            combined_scores[result.chunk_id] = result
            combined_scores[result.chunk_id].score *= self.config.hybrid_alpha

        # Add keyword scores
        for result in keyword_results:
            if result.chunk_id in combined_scores:
                combined_scores[result.chunk_id].score += result.score * (
                    1 - self.config.hybrid_alpha
                )
            else:
                result.score *= 1 - self.config.hybrid_alpha
                combined_scores[result.chunk_id] = result

        # Sort by combined score
        results = sorted(combined_scores.values(), key=lambda r: r.score, reverse=True)

        return results[:top_k]

    async def _adaptive_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Adaptive search that chooses strategy based on query."""
        # Analyze query to determine best strategy
        query_type = self._classify_query(query)

        if query_type == "factual":
            # Factual queries benefit from keyword search
            return await self._keyword_search(query, top_k)
        elif query_type == "conceptual":
            # Conceptual queries benefit from semantic search
            return await self._semantic_search(query, top_k)
        else:
            # Mixed queries use hybrid
            return await self._hybrid_search(query, top_k)

    def _classify_query(self, query: str) -> str:
        """Classify query type."""
        query_lower = query.lower()

        # Factual query indicators
        factual_keywords = ["what is", "define", "list of", "who is", "when was"]
        if any(kw in query_lower for kw in factual_keywords):
            return "factual"

        # Conceptual query indicators
        conceptual_keywords = ["how does", "why does", "explain", "compare", "analyze"]
        if any(kw in query_lower for kw in conceptual_keywords):
            return "conceptual"

        # Default to mixed
        return "mixed"

    def _calculate_keyword_score(self, query: str, content: str) -> float:
        """Calculate keyword matching score."""
        query_words = set(query.lower().split())
        content_lower = content.lower()

        matches = sum(1 for word in query_words if word in content_lower)
        total = len(query_words)

        return matches / total if total > 0 else 0.0

    async def _rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Re-rank results using configured strategy."""
        if self.config.rerank_strategy == RerankStrategy.SCORE_BASED:
            return await self._rerank_by_score(results)
        elif self.config.rerank_strategy == RerankStrategy.DIVERSITY:
            return await self._rerank_by_diversity(results)
        elif self.config.rerank_strategy == RerankStrategy.RELEVANCE:
            return await self._rerank_by_relevance(query, results)
        else:
            return results

    async def _rerank_by_score(self, results: List[SearchResult]) -> List[SearchResult]:
        """Re-rank by score (already sorted)."""
        return sorted(results, key=lambda r: r.score, reverse=True)

    async def _rerank_by_diversity(self, results: List[SearchResult]) -> List[SearchResult]:
        """Re-rank to maximize diversity."""
        # Maximal Marginal Relevance (MMR) algorithm
        if not results:
            return []

        reranked = [results[0]]  # Start with highest score
        remaining = results[1:]

        while remaining and len(reranked) < self.config.rerank_top_k:
            # Find result that maximizes: relevance - diversity * similarity
            best_idx = 0
            best_score = -float("inf")

            for i, result in enumerate(remaining):
                # Calculate minimum similarity to already selected
                min_sim = min(self._calculate_similarity(result, selected) for selected in reranked)

                # MMR score: lambda * relevance - (1-lambda) * similarity
                mmr_score = 0.5 * result.score - 0.5 * min_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            reranked.append(remaining.pop(best_idx))

        return reranked

    async def _rerank_by_relevance(
        self, query: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Re-rank by relevance to query.

        Combines original score with query relevance using a weighted formula.
        Original scores are preserved for comparison.
        """
        # Create new list to avoid modifying original scores
        reranked = []
        for result in results:
            relevance = self._calculate_relevance(query, result.content)
            # Combine original score with relevance (60% original, 40% relevance)
            # This boosts scores for content more relevant to the specific query
            new_score = 0.5 * result.score + 0.5 * relevance
            reranked.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    score=new_score,
                    source=result.source,
                    metadata=result.metadata.copy(),
                    citation=result.citation,
                    rank=result.rank,
                )
            )

        return sorted(reranked, key=lambda r: r.score, reverse=True)

    def _calculate_similarity(self, r1: SearchResult, r2: SearchResult) -> float:
        """Calculate similarity between two results."""
        # Simple cosine similarity based on content overlap
        words1 = set(r1.content.lower().split())
        words2 = set(r2.content.lower().split())

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score.

        Returns a value between 0 and 1, where 1 means perfect match.
        """
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words:
            return 0.0

        # Exact matches (query words that appear exactly in content)
        exact_matches = len(query_words & content_words)

        # Partial matches (query words that appear as substrings of content words)
        # Exclude exact matches to avoid double counting
        partial_matches = 0
        for qw in query_words:
            if qw not in content_words:  # Skip if already exact matched
                if any(qw in cw for cw in content_words):
                    partial_matches += 1

        # Normalize to 0-1 range
        # Exact matches get full weight (1.0), partial matches get half weight (0.5)
        max_score = len(query_words)
        return (exact_matches + 0.5 * partial_matches) / max_score

    async def _generate_citations(self, results: List[SearchResult]) -> List[SearchResult]:
        """Generate citations for results.

        Creates citations in the format: [N] "Title" source p. page
        """
        for i, result in enumerate(results):
            # Generate citation string
            source = result.metadata.get("source", result.source)
            page = result.metadata.get("page", "")
            title = result.metadata.get("title", "")

            citation_parts = [f"[{i + 1}]"]  # Add closing bracket here for "[N]" format

            if title:
                citation_parts.append(f'"{title}"')

            if source:
                citation_parts.append(source)

            if page:
                citation_parts.append(f"p. {page}")

            result.citation = " ".join(citation_parts)
            result.rank = i + 1

        return results

    async def query_with_citations(
        self, query: str, top_k: int = 10
    ) -> Tuple[str, List[SearchResult]]:
        """Query and return formatted answer with citations.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            Tuple of (answer_text, results_with_citations)
        """
        results = await self.query(query, top_k=top_k)

        # Format answer with citations
        answer_parts = []

        for result in results:
            answer_parts.append(f"{result.content} {result.citation}")

        answer = "\n\n".join(answer_parts)

        return answer, results

    async def generate_answer(self, query: str, include_sources: bool = True) -> Dict[str, Any]:
        """Generate answer with sources.

        Args:
            query: Query to answer
            include_sources: Whether to include source list

        Returns:
            Dict with answer and sources
        """
        results = await self.query(query, top_k=5)

        # Combine top results into answer
        answer = " ".join(r.content for r in results[:3])

        response: Dict[str, Any] = {"answer": answer, "query": query}

        if include_sources:
            sources_list: List[Dict[str, Any]] = [
                {"citation": r.citation, "source": r.source, "score": r.score} for r in results
            ]
            response["sources"] = sources_list

        return response
