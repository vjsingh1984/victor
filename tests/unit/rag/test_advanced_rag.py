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

"""Unit tests for Advanced RAG."""

import pytest
from victor.rag.advanced_rag import (
    AdvancedRAG,
    RAGConfig,
    SearchResult,
    SearchStrategy,
    RerankStrategy,
)


class TestAdvancedRAG:
    """Test suite for AdvancedRAG."""

    @pytest.fixture
    def rag(self):
        """Create AdvancedRAG instance."""
        config = RAGConfig(
            search_strategy=SearchStrategy.HYBRID,
            rerank_strategy=RerankStrategy.RELEVANCE,
            top_k=10,
            enable_citations=True,
        )
        return AdvancedRAG(document_store=None, config=config)

    def test_initialization(self, rag):
        """Test RAG initialization."""
        assert rag.config is not None
        assert rag.config.search_strategy == SearchStrategy.HYBRID
        assert rag.config.rerank_strategy == RerankStrategy.RELEVANCE
        assert rag.config.top_k == 10

    @pytest.mark.asyncio
    async def test_query_no_store(self, rag):
        """Test query without document store."""
        results = await rag.query("test query")

        assert results == []

    def test_classify_query(self, rag):
        """Test query classification."""
        # Factual query
        assert rag._classify_query("What is the capital of France?") == "factual"

        # Conceptual query
        assert rag._classify_query("How does machine learning work?") == "conceptual"

        # Mixed query
        assert rag._classify_query("Tell me about something") == "mixed"

    def test_calculate_keyword_score(self, rag):
        """Test keyword score calculation."""
        query = "machine learning algorithms"
        content = (
            "This document discusses various machine learning algorithms and their applications."
        )

        score = rag._calculate_keyword_score(query, content)

        assert 0 <= score <= 1
        assert score > 0.5  # Should have good matches

    def test_calculate_keyword_score_no_match(self, rag):
        """Test keyword score with no matches."""
        query = "quantum physics"
        content = "This is about biology and chemistry."

        score = rag._calculate_keyword_score(query, content)

        assert score == 0.0

    def test_calculate_relevance(self, rag):
        """Test relevance calculation."""
        query = "machine learning"
        content = "machine learning is a subset of artificial intelligence"

        relevance = rag._calculate_relevance(query, content)

        assert 0 <= relevance <= 1
        assert relevance > 0.5

    @pytest.mark.asyncio
    async def test_rerank_by_score(self, rag):
        """Test re-ranking by score."""
        results = [
            SearchResult(chunk_id="1", content="Content 1", score=0.5, source="doc1"),
            SearchResult(chunk_id="2", content="Content 2", score=0.8, source="doc2"),
            SearchResult(chunk_id="3", content="Content 3", score=0.3, source="doc3"),
        ]

        reranked = await rag._rerank_by_score(results)

        assert reranked[0].chunk_id == "2"  # Highest score
        assert reranked[1].chunk_id == "1"
        assert reranked[2].chunk_id == "3"  # Lowest score

    @pytest.mark.asyncio
    async def test_rerank_by_relevance(self, rag):
        """Test re-ranking by relevance."""
        results = [
            SearchResult(chunk_id="1", content="machine learning", score=0.5, source="doc1"),
            SearchResult(chunk_id="2", content="deep learning", score=0.8, source="doc2"),
            SearchResult(
                chunk_id="3", content="machine learning algorithms", score=0.6, source="doc3"
            ),
        ]

        query = "machine learning algorithms"

        reranked = await rag._rerank_by_relevance(query, results)

        # Result 3 should be highest due to relevance to query
        assert reranked[0].chunk_id == "3"
        assert reranked[0].score > results[2].score  # Score should be updated

    @pytest.mark.asyncio
    async def test_generate_citations(self, rag):
        """Test citation generation."""
        results = [
            SearchResult(
                chunk_id="1",
                content="Content 1",
                score=0.8,
                source="doc1.pdf",
                metadata={"page": 1, "title": "Test Document"},
            ),
            SearchResult(
                chunk_id="2",
                content="Content 2",
                score=0.7,
                source="doc2.pdf",
                metadata={"page": 5},
            ),
        ]

        cited_results = await rag._generate_citations(results)

        assert cited_results[0].citation != ""
        assert cited_results[0].rank == 1
        assert cited_results[1].rank == 2
        assert "[1]" in cited_results[0].citation
        assert "[2]" in cited_results[1].citation

    @pytest.mark.asyncio
    async def test_query_with_citations_no_store(self, rag):
        """Test query with citations without store."""
        answer, results = await rag.query_with_citations("test query")

        assert answer == ""
        assert results == []


class TestSearchResult:
    """Test suite for SearchResult."""

    def test_result_creation(self):
        """Test creating search result."""
        result = SearchResult(
            chunk_id="test_id",
            content="Test content",
            score=0.85,
            source="test.pdf",
            metadata={"page": 1},
        )

        assert result.chunk_id == "test_id"
        assert result.content == "Test content"
        assert result.score == 0.85
        assert result.source == "test.pdf"
        assert result.metadata["page"] == 1

    def test_result_to_dict(self):
        """Test converting result to dict."""
        result = SearchResult(
            chunk_id="test_id",
            content="Test content",
            score=0.85,
            source="test.pdf",
        )

        result_dict = result.to_dict()

        assert result_dict["chunk_id"] == "test_id"
        assert result_dict["content"] == "Test content"
        assert result_dict["score"] == 0.85
        assert result_dict["source"] == "test.pdf"


class TestRAGConfig:
    """Test suite for RAGConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = RAGConfig()

        assert config.search_strategy == SearchStrategy.HYBRID
        assert config.rerank_strategy == RerankStrategy.RELEVANCE
        assert config.top_k == 20
        assert config.rerank_top_k == 10
        assert config.hybrid_alpha == 0.7
        assert config.enable_citations is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = RAGConfig(
            search_strategy=SearchStrategy.SEMANTIC,
            top_k=15,
            rerank_top_k=5,
            hybrid_alpha=0.5,
        )

        assert config.search_strategy == SearchStrategy.SEMANTIC
        assert config.top_k == 15
        assert config.rerank_top_k == 5
        assert config.hybrid_alpha == 0.5
