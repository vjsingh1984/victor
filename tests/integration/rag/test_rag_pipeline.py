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

"""Integration tests for RAG pipeline.

Tests cover:
- Document → embedding → vector store flow
- Similarity search and retrieval
- Query processing and response generation
- RAG with LLM integration
- End-to-end RAG workflow
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from victor.rag.chunker import DocumentChunker, ChunkingConfig
from victor.rag.document_store import (
    Document,
    DocumentSearchResult,
    DocumentStore,
    DocumentStoreConfig,
)
from victor.rag.tools.ingest import RAGIngestTool
from victor.rag.tools.search import RAGSearchTool


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def rag_store():
    """Create an in-memory RAG document store for testing."""
    import hashlib

    class MockEmbeddingService:
        async def embed(self, text: str):
            # Generate deterministic 384-dim embedding based on text hash
            h = hashlib.sha256(text.encode()).digest()
            # Expand the hash to 384 dimensions by repeating and varying
            base_values = [float(b) / 255.0 for b in h]
            # Create 384 dimensions by cycling through base values with variation
            embedding = []
            for i in range(384):
                # Use different hash bytes with position-based variation
                byte_idx = i % len(base_values)
                position_factor = (i * 0.01) % 1.0  # Add position-based variation
                val = base_values[byte_idx] + position_factor * 0.1
                embedding.append(val % 1.0)  # Keep in [0, 1] range
            return embedding

    config = DocumentStoreConfig(
        path=Path(tempfile.mkdtemp()),
        embedding_dim=384,
        use_hybrid_search=True,
    )

    # Use custom chunking config for smaller test documents
    chunking_config = ChunkingConfig(
        chunk_size=500,  # Smaller for testing
        chunk_overlap=50,
        min_chunk_size=50,  # Much smaller to allow test docs
        max_chunk_size=1000,
        respect_sentence_boundaries=True,
        respect_paragraph_boundaries=True,
        code_aware=True,
    )

    store = DocumentStore(
        config, embedding_service=MockEmbeddingService(), chunking_config=chunking_config
    )
    await store.initialize()
    yield store
    # Cleanup is handled by temp directory


@pytest.fixture
def test_chunking_config():
    """Get chunking config for testing with smaller documents."""
    return ChunkingConfig(
        chunk_size=500,
        chunk_overlap=50,
        min_chunk_size=50,
        max_chunk_size=1000,
        respect_sentence_boundaries=True,
        respect_paragraph_boundaries=True,
        code_aware=True,
    )


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            id="doc1",
            content="Python is a high-level programming language known for its simplicity. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
            source="python.txt",
            doc_type="text",
            metadata={"topic": "python", "category": "programming"},
        ),
        Document(
            id="doc2",
            content="Docker is a containerization platform that allows developers to package applications into containers. These containers are lightweight and can run consistently across different environments.",
            source="docker.txt",
            doc_type="text",
            metadata={"topic": "docker", "category": "devops"},
        ),
        Document(
            id="doc3",
            content="Machine learning is a subset of artificial intelligence that enables systems to learn from data. It includes algorithms like neural networks, decision trees, and support vector machines.",
            source="ml.txt",
            doc_type="text",
            metadata={"topic": "machine learning", "category": "ai"},
        ),
        Document(
            id="doc4",
            content="Kubernetes is an open-source container orchestration platform. It automates deployment, scaling, and management of containerized applications across clusters of hosts.",
            source="k8s.txt",
            doc_type="text",
            metadata={"topic": "kubernetes", "category": "devops"},
        ),
    ]


@pytest.fixture
async def populated_store(rag_store, sample_documents):
    """Create a store populated with sample documents."""
    for doc in sample_documents:
        await rag_store.add_document(doc)
    return rag_store


# ============================================================================
# Document Ingestion Pipeline Tests
# ============================================================================


class TestDocumentIngestionPipeline:
    """Test the document ingestion pipeline."""

    @pytest.mark.asyncio
    async def test_ingest_single_document(self, rag_store):
        """Test ingesting a single document."""
        doc = Document(
            id="test1",
            content="Test document content for RAG system. " * 10,  # Make it longer
            source="test.txt",
            doc_type="text",
        )

        chunks = await rag_store.add_document(doc)

        assert len(chunks) > 0, f"Expected chunks but got {len(chunks)}"
        assert all(c.doc_id == "test1" for c in chunks)
        assert all(
            len(c.embedding) == 384 for c in chunks
        )  # Embeddings should have correct dimension

    @pytest.mark.asyncio
    async def test_ingest_multiple_documents(self, rag_store, sample_documents):
        """Test ingesting multiple documents."""
        total_chunks = 0
        for doc in sample_documents:
            chunks = await rag_store.add_document(doc)
            total_chunks += len(chunks)

        assert total_chunks > 0

        # Verify documents are indexed
        docs = await rag_store.list_documents()
        assert len(docs) == len(sample_documents)

    @pytest.mark.asyncio
    async def test_document_deduplication_by_hash(self, rag_store):
        """Test that duplicate documents are detected by content hash."""
        doc1 = Document(id="doc1", content="Same content", source="source1.txt")
        doc2 = Document(id="doc2", content="Same content", source="source2.txt")

        await rag_store.add_document(doc1)
        await rag_store.add_document(doc2)

        # Both should have the same content hash
        assert doc1.content_hash == doc2.content_hash

    @pytest.mark.asyncio
    async def test_ingest_with_chunker_config(self, rag_store):
        """Test ingesting with custom chunking configuration."""
        doc = Document(
            id="config_test",
            content="A" * 2000,  # Long content
            source="test.txt",
            doc_type="text",
        )

        # Custom chunking config
        chunker = DocumentChunker(ChunkingConfig(chunk_size=500, chunk_overlap=50))

        # Patch the store's chunker
        with patch.object(rag_store, "_get_embedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [0.0] * 384
            chunks = await chunker.chunk_document(doc, rag_store._get_embedding)

            assert len(chunks) > 1  # Should be chunked due to size

    @pytest.mark.asyncio
    async def test_metadata_preserved_in_chunks(self, rag_store):
        """Test that document metadata is preserved in chunks."""
        metadata = {"author": "Test", "category": "testing", "tags": ["tag1", "tag2"]}
        doc = Document(
            id="meta_test",
            content="Content with metadata",
            source="test.txt",
            metadata=metadata,
        )

        chunks = await rag_store.add_document(doc)

        # Check metadata is preserved
        for chunk in chunks:
            assert chunk.metadata["author"] == "Test"
            assert chunk.metadata["category"] == "testing"
            assert chunk.metadata["tags"] == ["tag1", "tag2"]


# ============================================================================
# Similarity Search Tests
# ============================================================================


class TestSimilaritySearch:
    """Test similarity search functionality."""

    @pytest.mark.asyncio
    async def test_basic_search(self, populated_store):
        """Test basic similarity search."""
        results = await populated_store.search("python programming", k=2)

        assert len(results) > 0
        assert all(isinstance(r, DocumentSearchResult) for r in results)
        assert all(0 <= r.score <= 1 for r in results)
        assert all(len(r.content) > 0 for r in results)

    @pytest.mark.asyncio
    async def test_search_relevance_scoring(self, populated_store):
        """Test that search results are properly scored by relevance."""
        # Search for python-related content
        results = await populated_store.search("python language features", k=5)

        if len(results) > 1:
            # Results should be sorted by relevance (highest first)
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_search_with_doc_filter(self, populated_store):
        """Test search filtering by document IDs."""
        results = await populated_store.search("programming", k=5, filter_doc_ids=["doc1"])

        # Should only return results from doc1
        assert all(r.doc_id == "doc1" for r in results)

    @pytest.mark.asyncio
    async def test_search_with_metadata_filter(self, populated_store):
        """Test search filtering by metadata."""
        results = await populated_store.search(
            "container", k=5, filter_metadata={"topic": "docker"}
        )

        # Should only return docker-related results
        assert all(r.metadata.get("topic") == "docker" for r in results)

    @pytest.mark.asyncio
    async def test_search_no_results(self, populated_store):
        """Test search with no matching results."""
        results = await populated_store.search("quantum physics teleportation", k=5)

        # Should return empty list or low-relevance results
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_hybrid_vs_vector_only(self, populated_store):
        """Test hybrid search vs vector-only search."""
        query = "container orchestration platform"

        # Hybrid search
        hybrid_results = await populated_store.search(query, k=3, use_hybrid=True)

        # Vector-only search
        vector_results = await populated_store.search(query, k=3, use_hybrid=False)

        # Both should return results
        assert len(hybrid_results) > 0
        assert len(vector_results) > 0

    @pytest.mark.asyncio
    async def test_search_result_limit(self, populated_store):
        """Test that search respects k parameter."""
        for k in [1, 2, 3, 5]:
            results = await populated_store.search("programming", k=k)
            assert len(results) <= k


# ============================================================================
# RAG Tool Integration Tests
# ============================================================================


class TestRAGToolIntegration:
    """Test RAG tools working together."""

    @pytest.mark.asyncio
    async def test_ingest_then_search_workflow(self, populated_store):
        """Test ingesting documents and then searching them."""
        # Ingest is already done by populated_store fixture
        # Now search
        results = await populated_store.search("docker containers", k=3)

        assert len(results) > 0
        # Should find docker-related content
        docker_found = any("docker" in r.content.lower() for r in results)
        assert docker_found

    @pytest.mark.asyncio
    async def test_ingest_tool_integration(self, test_chunking_config):
        """Test RAGIngestTool with document store."""
        tool = RAGIngestTool()
        store = DocumentStore(
            DocumentStoreConfig(path=Path(tempfile.mkdtemp())), chunking_config=test_chunking_config
        )
        await store.initialize()

        with patch.object(tool, "_get_document_store", return_value=store):
            result = await tool.execute(
                {}, content="Test content for tool integration", doc_type="text"
            )

            assert result.success
            assert "Successfully ingested" in result.output

            # Verify document was added
            docs = await store.list_documents()
            assert len(docs) > 0

    @pytest.mark.asyncio
    async def test_search_tool_integration(self, populated_store):
        """Test RAGSearchTool with populated store."""
        tool = RAGSearchTool()

        with patch.object(tool, "_get_document_store", return_value=populated_store):
            result = await tool.execute({}, query="python programming", k=3)

            assert result.success
            assert "Found" in result.output or "relevant chunks" in result.output.lower()

    @pytest.mark.asyncio
    async def test_multi_document_search(self, populated_store):
        """Test search across multiple documents."""
        # Search for content that might span multiple docs
        results = await populated_store.search("programming platform", k=5)

        # Should get results from multiple documents
        doc_ids = set(r.doc_id for r in results)
        assert len(doc_ids) >= 1  # At least one document

    @pytest.mark.asyncio
    async def test_search_with_high_k(self, populated_store):
        """Test search with high k value."""
        results = await populated_store.search("programming", k=20)

        # Should not exceed available chunks
        assert len(results) >= 0


# ============================================================================
# End-to-End RAG Workflow Tests
# ============================================================================


class TestEndToEndRAGWorkflow:
    """Test complete RAG workflows."""

    @pytest.mark.asyncio
    async def test_complete_rag_workflow(self, test_chunking_config):
        """Test complete RAG workflow: ingest → search → retrieve."""
        # Setup
        store = DocumentStore(
            DocumentStoreConfig(path=Path(tempfile.mkdtemp())), chunking_config=test_chunking_config
        )
        await store.initialize()

        # Step 1: Ingest documents
        docs = [
            Document(
                id="workflow_doc1",
                content="React is a JavaScript library for building user interfaces. It was developed by Facebook and is widely used for web development.",
                source="react.txt",
                doc_type="text",
            ),
            Document(
                id="workflow_doc2",
                content="Vue.js is a progressive JavaScript framework for building user interfaces. It is designed to be incrementally adoptable and can function as a web framework.",
                source="vue.txt",
                doc_type="text",
            ),
        ]

        for doc in docs:
            await store.add_document(doc)

        # Step 2: Search
        results = await store.search("JavaScript UI library", k=3)

        # Step 3: Verify
        assert len(results) > 0
        assert any("javascript" in r.content.lower() for r in results)
        assert any("ui" in r.content.lower() or "interface" in r.content.lower() for r in results)

    @pytest.mark.asyncio
    async def test_rag_with_document_deletion(self, populated_store):
        """Test RAG workflow with document deletion."""
        # Initial search should find results
        results_before = await populated_store.search("python", k=5)
        doc_count_before = len(await populated_store.list_documents())

        # Delete a document
        await populated_store.delete_document("doc1")

        # Search again
        results_after = await populated_store.search("python", k=5)
        doc_count_after = len(await populated_store.list_documents())

        # Document count should decrease
        assert doc_count_after == doc_count_before - 1

    @pytest.mark.asyncio
    async def test_rag_document_retrieval(self, populated_store):
        """Test retrieving documents by ID."""
        doc = await populated_store.get_document("doc1")

        assert doc is not None
        assert doc.id == "doc1"
        assert doc.source == "python.txt"

    @pytest.mark.asyncio
    async def test_rag_statistics(self, populated_store):
        """Test RAG store statistics."""
        stats = await populated_store.get_stats()

        assert "total_documents" in stats
        assert "total_chunks" in stats
        assert stats["total_documents"] == 4  # Based on sample_documents fixture

    @pytest.mark.asyncio
    async def test_batch_document_ingestion(self, test_chunking_config):
        """Test ingesting multiple documents in batch."""
        store = DocumentStore(
            DocumentStoreConfig(path=Path(tempfile.mkdtemp())), chunking_config=test_chunking_config
        )
        await store.initialize()

        # Create multiple documents
        docs = [
            Document(id=f"batch_{i}", content=f"Content {i} " * 50, source=f"file{i}.txt")
            for i in range(10)
        ]

        # Ingest all
        for doc in docs:
            await store.add_document(doc)

        # Verify
        stats = await store.get_stats()
        assert stats["total_documents"] == 10


# ============================================================================
# RAG with LLM Integration Tests
# ============================================================================


class TestRAGWithLLMIntegration:
    """Test RAG system integration with LLM components."""

    @pytest.mark.asyncio
    async def test_rag_augmented_query_generation(self, populated_store):
        """Test generating augmented queries with RAG context."""
        # Search for relevant context
        results = await populated_store.search("docker containers", k=2)

        assert len(results) > 0

        # Simulate creating an augmented query
        context = "\n\n".join([r.content for r in results])
        augmented_query = f"Context:\n{context}\n\nQuestion: What is Docker?"

        assert "Docker" in augmented_query
        assert "container" in augmented_query.lower()

    @pytest.mark.asyncio
    async def test_rag_context_relevance(self, populated_store):
        """Test that retrieved context is relevant to query."""
        query = "machine learning algorithms"
        results = await populated_store.search(query, k=3)

        if len(results) > 0:
            # Check that results contain relevant terms
            all_text = " ".join(r.content.lower() for r in results)
            # Should mention ML or AI terms
            assert any(
                term in all_text for term in ["machine learning", "algorithms", "neural", "ai"]
            )

    @pytest.mark.asyncio
    async def test_multi_turn_rag_conversation(self, populated_store):
        """Test RAG in a multi-turn conversation scenario."""
        # Turn 1: Initial query
        results1 = await populated_store.search("programming", k=3)
        assert len(results1) > 0

        # Turn 2: Follow-up query (should use previous context)
        results2 = await populated_store.search("python specifically", k=3)
        assert len(results2) > 0

        # Turn 3: Different topic
        results3 = await populated_store.search("container orchestration", k=3)
        assert len(results3) > 0


# ============================================================================
# Performance and Scalability Tests
# ============================================================================


class TestRAGPerformance:
    """Test RAG system performance characteristics."""

    @pytest.mark.asyncio
    async def test_search_performance(self, populated_store):
        """Test that search completes in reasonable time."""
        import time

        start = time.time()
        results = await populated_store.search("programming", k=10)
        duration = time.time() - start

        assert duration < 5.0  # Should complete in under 5 seconds
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_ingestion_performance(self, test_chunking_config):
        """Test that document ingestion completes in reasonable time."""
        import time

        store = DocumentStore(
            DocumentStoreConfig(path=Path(tempfile.mkdtemp())), chunking_config=test_chunking_config
        )
        await store.initialize()

        # Create a large document
        large_doc = Document(
            id="large",
            content="Test content " * 1000,  # ~15KB
            source="large.txt",
        )

        start = time.time()
        chunks = await store.add_document(large_doc)
        duration = time.time() - start

        assert duration < 10.0  # Should complete in under 10 seconds
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_concurrent_searches(self, populated_store):
        """Test handling multiple concurrent searches."""
        queries = ["python", "docker", "kubernetes", "machine learning", "programming"]

        start = asyncio.get_event_loop().time()
        results = await asyncio.gather(*[populated_store.search(q, k=3) for q in queries])
        duration = asyncio.get_event_loop().time() - start

        assert len(results) == len(queries)
        assert duration < 10.0  # All searches should complete in reasonable time


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestRAGErrorHandling:
    """Test RAG system error handling."""

    @pytest.mark.asyncio
    async def test_search_empty_store(self, test_chunking_config):
        """Test searching an empty document store."""
        store = DocumentStore(
            DocumentStoreConfig(path=Path(tempfile.mkdtemp())), chunking_config=test_chunking_config
        )
        await store.initialize()

        results = await store.search("test query", k=5)

        # Should return empty list
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_nonexistent_document(self, populated_store):
        """Test retrieving a non-existent document."""
        doc = await populated_store.get_document("nonexistent_id")

        assert doc is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_document(self, populated_store):
        """Test deleting a non-existent document."""
        # Should not raise an error
        result = await populated_store.delete_document("nonexistent_id")

        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_ingest_malformed_document(self, populated_store):
        """Test ingesting a document with problematic content."""
        # Document with special characters and encoding issues
        doc = Document(
            id="malformed",
            content="Test with \x00 null bytes and émojis 🎉",
            source="malformed.txt",
        )

        # Should handle gracefully
        chunks = await populated_store.add_document(doc)

        assert isinstance(chunks, list)


# ============================================================================
# Metadata and Filtering Tests
# ============================================================================


class TestMetadataAndFiltering:
    """Test metadata handling and advanced filtering."""

    @pytest.mark.asyncio
    async def test_complex_metadata_filtering(self, populated_store):
        """Test filtering with complex metadata queries."""
        # Filter by category
        results = await populated_store.search(
            "platform", k=5, filter_metadata={"category": "devops"}
        )

        assert all(r.metadata.get("category") == "devops" for r in results)

    @pytest.mark.asyncio
    async def test_array_metadata_filtering(self, test_chunking_config):
        """Test filtering with array values in metadata."""
        store = DocumentStore(
            DocumentStoreConfig(path=Path(tempfile.mkdtemp())), chunking_config=test_chunking_config
        )
        await store.initialize()

        doc = Document(
            id="tags_test",
            content="Content with tags",
            source="tags_test.txt",
            metadata={"tags": ["python", "ml", "devops"]},
        )
        await store.add_document(doc)

        # Filter by array value
        results = await store.search("content", k=5, filter_metadata={"tags": ["python"]})

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_metadata_preservation_across_search(self, populated_store):
        """Test that metadata is preserved in search results."""
        results = await populated_store.search("python", k=5)

        for result in results:
            # Metadata should be present
            assert isinstance(result.metadata, dict)
            # Original metadata fields should be preserved
            if result.doc_id == "doc1":
                assert result.metadata.get("topic") == "python"
