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

"""Unit tests for RAG document ingestion pipeline.

Tests cover:
- Document parsing (PDF, Markdown, TXT, HTML, JSON)
- Chunking strategies (fixed-size, semantic, code-aware)
- Embedding generation
- Metadata extraction
- Document type detection
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from victor.rag.chunker import (
    ChunkingConfig,
    DocumentChunker,
    detect_document_type,
)
from victor.rag.document_store import (
    Document,
    DocumentChunk,
    DocumentSearchResult,
    DocumentStore,
    DocumentStoreConfig,
)
from victor.rag.tools.ingest import RAGIngestTool


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_text():
    """Sample text content for testing."""
    return """
    Machine learning is a subset of artificial intelligence.
    It focuses on building systems that can learn from data.
    The goal is to enable computers to learn automatically.

    Deep learning is a type of machine learning that uses neural networks.
    These networks are inspired by the human brain's structure.
    They can learn complex patterns from large amounts of data.

    Natural language processing (NLP) is another important field.
    It deals with the interaction between computers and human language.
    Applications include translation, sentiment analysis, and text generation.
    """


@pytest.fixture
def sample_markdown():
    """Sample markdown content for testing."""
    return """
# Introduction to Python

Python is a high-level programming language.

## Features

- Easy to learn and use
- Extensive standard library
- Cross-platform compatibility

## Data Types

Python supports several data types:

### Numbers
Integers, floats, and complex numbers.

### Strings
Text data enclosed in quotes.

### Lists
Ordered collections of items.
"""


@pytest.fixture
def sample_html():
    """Sample HTML content for testing."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Test Document</title>
</head>
<body>
    <h1>Introduction</h1>
    <p>This is a test document for RAG ingestion.</p>
    <p>It contains multiple paragraphs for testing chunking.</p>

    <h2>Section 1</h2>
    <p>This is the first section with some content.</p>
    <p>More text here to ensure sufficient length.</p>

    <h2>Section 2</h2>
    <p>This is the second section.</p>
</body>
</html>
"""


@pytest.fixture
def sample_json():
    """Sample JSON content for testing."""
    data = {
        "users": [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
            {"id": 3, "name": "Charlie", "email": "charlie@example.com"},
        ],
        "metadata": {"version": "1.0", "created": "2025-01-20"},
    }
    return json.dumps(data, indent=2)


@pytest.fixture
def sample_code():
    """Sample Python code for testing."""
    return """
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.data = []

    def load_data(self, source):
        with open(source, 'r') as f:
            self.data = json.load(f)
        return self.data

    def transform(self, func):
        self.data = [func(item) for item in self.data]
        return self.data

    def save(self, destination):
        with open(destination, 'w') as f:
            json.dump(self.data, f, indent=2)


def main():
    processor = DataProcessor({})
    data = processor.load_data('input.json')
    processor.transform(lambda x: x.upper())
    processor.save('output.json')
"""


@pytest.fixture
def mock_embedding_fn():
    """Mock embedding function."""

    async def _embed(text: str) -> List[float]:
        # Return deterministic mock embeddings with 384 dimensions
        import hashlib

        h = hashlib.sha256(text.encode()).digest()
        # Extend to 384 dimensions by repeating if needed
        base_embeddings = [b / 255.0 for b in h]
        # Pad or repeat to get exactly 384 dimensions
        if len(base_embeddings) < 384:
            base_embeddings = base_embeddings * (384 // len(base_embeddings) + 1)
        return base_embeddings[:384]

    return _embed


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Document Type Detection Tests
# ============================================================================


class TestDocumentTypeDetection:
    """Test document type detection from various sources."""

    def test_detect_markdown_from_extension(self):
        """Test detecting markdown from file extension."""
        doc_type = detect_document_type("test.md", "some content")
        assert doc_type == "markdown"

        doc_type = detect_document_type("test.markdown", "some content")
        assert doc_type == "markdown"

    def test_detect_html_from_extension(self):
        """Test detecting HTML from file extension."""
        doc_type = detect_document_type("test.html", "some content")
        assert doc_type == "html"

        doc_type = detect_document_type("test.htm", "some content")
        assert doc_type == "html"

    def test_detect_code_from_extension(self):
        """Test detecting code from file extension."""
        doc_type = detect_document_type("test.py", "some content")
        assert doc_type == "code"

        doc_type = detect_document_type("test.js", "some content")
        assert doc_type == "code"

    def test_detect_json_from_extension(self):
        """Test detecting JSON from file extension."""
        doc_type = detect_document_type("test.json", "some content")
        assert doc_type == "json"

    def test_detect_html_from_content(self):
        """Test detecting HTML from content tags."""
        content = "<html><body><h1>Test</h1></body></html>"
        doc_type = detect_document_type("unknown", content)
        assert doc_type == "html"

    def test_detect_json_from_content(self):
        """Test detecting JSON from content structure."""
        content = '{"key": "value", "array": [1, 2, 3]}'
        doc_type = detect_document_type("unknown", content)
        assert doc_type == "json"

    def test_detect_markdown_from_content(self):
        """Test detecting markdown from content headers."""
        content = "# Header 1\n\nSome content\n\n## Header 2"
        doc_type = detect_document_type("unknown", content)
        assert doc_type == "markdown"

    def test_detect_code_from_content(self):
        """Test detecting code from function definitions."""
        content = "def my_function():\n    pass\n\nclass MyClass:\n    pass"
        doc_type = detect_document_type("unknown", content)
        assert doc_type == "code"

    def test_default_to_text(self):
        """Test defaulting to text type."""
        doc_type = detect_document_type("unknown.xyz", "just plain text")
        assert doc_type == "text"


# ============================================================================
# Chunking Strategy Tests
# ============================================================================


class TestTextChunking:
    """Test plain text chunking."""

    @pytest.mark.asyncio
    async def test_text_chunking_basic(self, sample_text, mock_embedding_fn):
        """Test basic text chunking."""
        chunker = DocumentChunker(ChunkingConfig(chunk_size=300, chunk_overlap=50))
        doc = Document(id="test1", content=sample_text, source="test.txt", doc_type="text")

        chunks = await chunker.chunk_document(doc, mock_embedding_fn)

        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all(c.doc_id == "test1" for c in chunks)
        assert all(len(c.content) > 0 for c in chunks)
        assert all(len(c.embedding) == 384 for c in chunks)

    @pytest.mark.asyncio
    async def test_text_chunking_with_sentence_boundaries(self, sample_text, mock_embedding_fn):
        """Test text chunking respecting sentence boundaries."""
        chunker = DocumentChunker(
            ChunkingConfig(
                chunk_size=300,
                chunk_overlap=50,
                respect_sentence_boundaries=True,
            )
        )
        doc = Document(id="test2", content=sample_text, source="test.txt", doc_type="text")

        chunks = await chunker.chunk_document(doc, mock_embedding_fn)

        # Check that chunks end at sentence boundaries when possible
        assert len(chunks) > 0
        for chunk in chunks:
            # Chunks should end with sentence-ending punctuation or be at end
            content = chunk.content.strip()
            if not content.endswith(("...", "data", "networks")):
                assert (
                    content[-1] in ".!?"
                ), f"Chunk doesn't end at sentence boundary: {content[-50:]}"

    @pytest.mark.asyncio
    async def test_text_chunking_min_chunk_size(self, sample_text, mock_embedding_fn):
        """Test that small chunks are filtered out."""
        chunker = DocumentChunker(ChunkingConfig(chunk_size=500, min_chunk_size=200))
        doc = Document(id="test3", content=sample_text, source="test.txt", doc_type="text")

        chunks = await chunker.chunk_document(doc, mock_embedding_fn)

        # All chunks should meet minimum size requirement
        assert all(len(c.content) >= 200 for c in chunks)

    @pytest.mark.asyncio
    async def test_text_chunking_overlap(self, mock_embedding_fn):
        """Test that chunks have appropriate overlap."""
        content = "word " * 100  # Create long content
        chunker = DocumentChunker(ChunkingConfig(chunk_size=200, chunk_overlap=50))
        doc = Document(id="test4", content=content, source="test.txt", doc_type="text")

        chunks = await chunker.chunk_document(doc, mock_embedding_fn)

        if len(chunks) > 1:
            # Check that consecutive chunks have overlap
            for i in range(len(chunks) - 1):
                current_end = chunks[i].content[-50:]
                next_start = chunks[i + 1].content[:50]
                # Some overlap should exist
                has_overlap = any(word in next_start for word in current_end.split())
                assert has_overlap, "No overlap found between consecutive chunks"

    @pytest.mark.asyncio
    async def test_estimate_chunks(self, sample_text):
        """Test chunk count estimation."""
        chunker = DocumentChunker(ChunkingConfig(chunk_size=300, chunk_overlap=50))

        estimate = chunker.estimate_chunks(sample_text)

        assert isinstance(estimate, int)
        assert estimate > 0
        # Estimate should be close to actual chunk count
        chunks = chunker._chunk_text(sample_text)
        assert abs(estimate - len(chunks)) <= 1  # Allow off by one


class TestMarkdownChunking:
    """Test markdown-specific chunking."""

    @pytest.mark.asyncio
    async def test_markdown_chunking_by_headers(self, sample_markdown, mock_embedding_fn):
        """Test markdown chunking preserving header structure."""
        chunker = DocumentChunker(ChunkingConfig(min_chunk_size=50))
        doc = Document(id="md1", content=sample_markdown, source="test.md", doc_type="markdown")

        chunks = await chunker.chunk_document(doc, mock_embedding_fn)

        assert len(chunks) > 0
        # Check that chunks contain markdown content
        all_content = " ".join(c.content for c in chunks)
        assert "Python" in all_content or "Features" in all_content or "Data" in all_content

    @pytest.mark.asyncio
    async def test_markdown_code_blocks(self, mock_embedding_fn):
        """Test markdown chunking with code blocks."""
        content = """
# Code Examples

```python
def hello():
    print("Hello, World!")
```

```javascript
function hello() {
    console.log("Hello, World!");
}
```
"""
        chunker = DocumentChunker(ChunkingConfig(min_chunk_size=50))
        doc = Document(id="md2", content=content, source="test.md", doc_type="markdown")

        chunks = await chunker.chunk_document(doc, mock_embedding_fn)

        assert len(chunks) > 0
        # Code blocks should be preserved within chunks
        all_content = " ".join(c.content for c in chunks)
        assert "python" in all_content or "javascript" in all_content


class TestCodeChunking:
    """Test code-specific chunking."""

    @pytest.mark.asyncio
    async def test_code_chunking_by_functions(self, sample_code, mock_embedding_fn):
        """Test code chunking preserving function boundaries."""
        chunker = DocumentChunker()
        doc = Document(id="code1", content=sample_code, source="test.py", doc_type="code")

        chunks = await chunker.chunk_document(doc, mock_embedding_fn)

        assert len(chunks) > 0
        # Check that function/class definitions are preserved
        all_content = " ".join(c.content for c in chunks)
        assert "class DataProcessor:" in all_content
        assert "def load_data" in all_content
        assert "def transform" in all_content

    @pytest.mark.asyncio
    async def test_code_chunking_preserves_structure(self, sample_code, mock_embedding_fn):
        """Test that code chunking preserves code structure."""
        chunker = DocumentChunker(ChunkingConfig(chunk_size=200, min_chunk_size=100))
        doc = Document(id="code2", content=sample_code, source="test.py", doc_type="code")

        chunks = await chunker.chunk_document(doc, mock_embedding_fn)

        # Each chunk should be valid code (start and end points)
        for chunk in chunks:
            # Check that chunks don't break in the middle of tokens
            assert not chunk.content.strip().endswith("(")
            assert not chunk.content.strip().endswith(":")


class TestHTMLChunking:
    """Test HTML-specific chunking."""

    @pytest.mark.asyncio
    async def test_html_chunking_semantic_structure(self, sample_html, mock_embedding_fn):
        """Test HTML chunking preserving semantic elements."""
        chunker = DocumentChunker()
        doc = Document(id="html1", content=sample_html, source="test.html", doc_type="html")

        chunks = await chunker.chunk_document(doc, mock_embedding_fn)

        assert len(chunks) > 0
        # Check that script/style tags are removed
        for chunk in chunks:
            assert "<script>" not in chunk.content
            assert "<style>" not in chunk.content
            assert "</script>" not in chunk.content
            assert "</style>" not in chunk.content

    @pytest.mark.asyncio
    async def test_html_chunking_text_extraction(self, sample_html, mock_embedding_fn):
        """Test that HTML chunking extracts readable text."""
        chunker = DocumentChunker()
        doc = Document(id="html2", content=sample_html, source="test.html", doc_type="html")

        chunks = await chunker.chunk_document(doc, mock_embedding_fn)

        all_text = " ".join(c.content for c in chunks)
        # Check that text content is extracted
        assert "Introduction" in all_text
        assert "test document" in all_text.lower()
        assert "Section 1" in all_text or "first section" in all_text.lower()


class TestJSONChunking:
    """Test JSON-specific chunking."""

    @pytest.mark.asyncio
    async def test_json_chunking_by_keys(self, sample_json, mock_embedding_fn):
        """Test JSON chunking by top-level keys."""
        chunker = DocumentChunker()
        doc = Document(id="json1", content=sample_json, source="test.json", doc_type="json")

        chunks = await chunker.chunk_document(doc, mock_embedding_fn)

        assert len(chunks) > 0
        # Check that JSON structure is preserved across chunks
        all_content = " ".join(c.content for c in chunks)
        # At least some JSON content should be present
        assert '"users"' in all_content or '"metadata"' in all_content or "Alice" in all_content

    @pytest.mark.asyncio
    async def test_json_chunking_arrays(self, mock_embedding_fn):
        """Test JSON chunking with large arrays."""
        data = {"items": [{"id": i, "name": f"Item {i}"} for i in range(50)]}
        content = json.dumps(data, indent=2)
        doc = Document(id="json2", content=content, source="test.json", doc_type="json")

        chunker = DocumentChunker(ChunkingConfig(chunk_size=500))
        chunks = await chunker.chunk_document(doc, mock_embedding_fn)

        assert len(chunks) > 1  # Should be chunked due to size


# ============================================================================
# Metadata Extraction Tests
# ============================================================================


class TestMetadataExtraction:
    """Test metadata extraction from documents."""

    def test_document_with_metadata(self):
        """Test document creation with metadata."""
        metadata = {"author": "Test Author", "date": "2025-01-20", "tags": ["test", "rag"]}
        doc = Document(id="meta1", content="Content", source="test.txt", metadata=metadata)

        assert doc.metadata == metadata
        assert doc.metadata["author"] == "Test Author"
        assert doc.metadata["tags"] == ["test", "rag"]

    def test_content_hash(self):
        """Test content hash generation for deduplication."""
        doc1 = Document(id="hash1", content="Same content", source="test1.txt")
        doc2 = Document(id="hash2", content="Same content", source="test2.txt")

        assert doc1.content_hash == doc2.content_hash

    def test_different_content_hashes(self):
        """Test that different content produces different hashes."""
        doc1 = Document(id="hash1", content="Content A", source="test1.txt")
        doc2 = Document(id="hash2", content="Content B", source="test2.txt")

        assert doc1.content_hash != doc2.content_hash


# ============================================================================
# RAG Ingest Tool Tests
# ============================================================================


class TestRAGIngestTool:
    """Test RAG ingest tool functionality."""

    @pytest.fixture
    def ingest_tool(self):
        """Create RAGIngestTool instance."""
        return RAGIngestTool()

    @pytest.mark.asyncio
    async def test_ingest_direct_content(self, ingest_tool, temp_dir, mock_embedding_fn):
        """Test ingesting direct text content."""
        with patch.object(ingest_tool, "_get_document_store") as mock_store_getter:
            mock_store = AsyncMock()
            mock_store.add_document = AsyncMock(return_value=[])
            mock_store_getter.return_value = mock_store

            result = await ingest_tool.execute({}, content="Test content", doc_type="text")

            assert result.success
            assert "Successfully ingested" in result.output
            mock_store.add_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_text_file(self, ingest_tool, temp_dir, mock_embedding_fn):
        """Test ingesting a text file."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Sample text content for testing.")

        with patch.object(ingest_tool, "_get_document_store") as mock_store_getter:
            mock_store = AsyncMock()
            mock_store.add_document = AsyncMock(return_value=[])
            mock_store_getter.return_value = mock_store

            result = await ingest_tool.execute({}, path=str(test_file))

            assert result.success
            assert "Successfully ingested" in result.output

    @pytest.mark.asyncio
    async def test_ingest_markdown_file(self, ingest_tool, temp_dir, mock_embedding_fn):
        """Test ingesting a markdown file."""
        test_file = temp_dir / "test.md"
        test_file.write_text("# Test Header\n\nSome content here.")

        with patch.object(ingest_tool, "_get_document_store") as mock_store_getter:
            mock_store = AsyncMock()
            mock_store.add_document = AsyncMock(return_value=[])
            mock_store_getter.return_value = mock_store

            result = await ingest_tool.execute({}, path=str(test_file))

            assert result.success
            assert "doc_type" in result.output.lower() or "markdown" in result.output.lower()

    @pytest.mark.asyncio
    async def test_ingest_directory(self, ingest_tool, temp_dir, mock_embedding_fn):
        """Test ingesting a directory of files."""
        (temp_dir / "file1.txt").write_text("Content 1")
        (temp_dir / "file2.md").write_text("# Content 2")
        (temp_dir / "file3.txt").write_text("Content 3")

        with patch.object(ingest_tool, "_get_document_store") as mock_store_getter:
            mock_store = AsyncMock()
            mock_store.add_document = AsyncMock(return_value=[])
            mock_store_getter.return_value = mock_store

            result = await ingest_tool.execute({}, path=str(temp_dir), recursive=True)

            assert result.success
            assert "Directory ingestion complete" in result.output

    @pytest.mark.asyncio
    async def test_ingest_with_custom_metadata(self, ingest_tool, temp_dir, mock_embedding_fn):
        """Test ingesting with custom metadata."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Content with metadata")

        metadata = {"author": "Test", "category": "testing"}

        with patch.object(ingest_tool, "_get_document_store") as mock_store_getter:
            mock_store = AsyncMock()
            mock_store.add_document = AsyncMock(return_value=[])
            mock_store_getter.return_value = mock_store

            result = await ingest_tool.execute({}, path=str(test_file), metadata=metadata)

            assert result.success
            # Check that metadata is passed to document creation
            call_args = mock_store.add_document.call_args
            doc = call_args[0][0]
            assert doc.metadata == metadata

    @pytest.mark.asyncio
    async def test_ingest_with_custom_doc_id(self, ingest_tool, temp_dir, mock_embedding_fn):
        """Test ingesting with custom document ID."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Content with custom ID")

        with patch.object(ingest_tool, "_get_document_store") as mock_store_getter:
            mock_store = AsyncMock()
            mock_store.add_document = AsyncMock(return_value=[])
            mock_store_getter.return_value = mock_store

            result = await ingest_tool.execute({}, path=str(test_file), doc_id="custom_id_123")

            assert result.success
            assert "custom_id_123" in result.output

    @pytest.mark.asyncio
    async def test_ingest_nonexistent_file(self, ingest_tool):
        """Test ingesting a non-existent file."""
        result = await ingest_tool.execute({}, path="/nonexistent/file.txt")

        assert not result.success
        assert "not found" in result.output.lower()

    @pytest.mark.asyncio
    async def test_ingest_no_input(self, ingest_tool):
        """Test ingest without any input."""
        result = await ingest_tool.execute({})

        assert not result.success
        assert "Either 'path', 'url', or 'content' must be provided" in result.output

    def test_detect_doc_type_from_extension(self, ingest_tool):
        """Test document type detection from file extension."""
        assert ingest_tool._detect_doc_type(".py") == "code"
        assert ingest_tool._detect_doc_type(".js") == "code"
        assert ingest_tool._detect_doc_type(".md") == "markdown"
        assert ingest_tool._detect_doc_type(".txt") == "text"
        assert ingest_tool._detect_doc_type(".pdf") == "pdf"


# ============================================================================
# Chunk Configuration Tests
# ============================================================================


class TestChunkingConfig:
    """Test chunking configuration."""

    def test_default_config(self):
        """Test default chunking configuration."""
        config = ChunkingConfig()

        assert config.chunk_size == 1344
        assert config.chunk_overlap == 134
        assert config.min_chunk_size == 200
        assert config.max_chunk_size == 2000
        assert config.respect_sentence_boundaries is True
        assert config.respect_paragraph_boundaries is True
        assert config.code_aware is True

    def test_custom_config(self):
        """Test custom chunking configuration."""
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=50,
            min_chunk_size=100,
            max_chunk_size=1000,
            respect_sentence_boundaries=False,
        )

        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.min_chunk_size == 100
        assert config.max_chunk_size == 1000
        assert config.respect_sentence_boundaries is False


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_document(self, mock_embedding_fn):
        """Test chunking an empty document."""
        chunker = DocumentChunker()
        doc = Document(id="empty", content="", source="test.txt", doc_type="text")

        chunks = await chunker.chunk_document(doc, mock_embedding_fn)

        # Should handle empty content gracefully
        assert isinstance(chunks, list)

    @pytest.mark.asyncio
    async def test_very_short_document(self, mock_embedding_fn):
        """Test chunking a very short document."""
        chunker = DocumentChunker(ChunkingConfig(min_chunk_size=50))
        doc = Document(id="short", content="Short text.", source="test.txt", doc_type="text")

        chunks = await chunker.chunk_document(doc, mock_embedding_fn)

        # Short text may be filtered out or produce single chunk
        assert isinstance(chunks, list)

    @pytest.mark.asyncio
    async def test_very_long_document(self, mock_embedding_fn):
        """Test chunking a very long document."""
        long_content = "This is a sentence. " * 1000  # Create long content
        chunker = DocumentChunker(ChunkingConfig(chunk_size=1000))
        doc = Document(id="long", content=long_content, source="test.txt", doc_type="text")

        chunks = await chunker.chunk_document(doc, mock_embedding_fn)

        # Should produce multiple chunks
        assert len(chunks) > 1

    @pytest.mark.asyncio
    async def test_invalid_json(self, mock_embedding_fn):
        """Test chunking invalid JSON."""
        invalid_json = '{"invalid": json, missing quotes}'
        chunker = DocumentChunker()
        doc = Document(id="invalid", content=invalid_json, source="test.json", doc_type="json")

        # Should fall back to text chunking
        chunks = await chunker.chunk_document(doc, mock_embedding_fn)

        assert isinstance(chunks, list)

    @pytest.mark.asyncio
    async def test_chunk_with_special_characters(self, mock_embedding_fn):
        """Test chunking text with special characters."""
        content = "Test with émojis 🎉 and spëcial çharacters.\n\nNewlines.\t\tTabs."
        # Use a smaller min_chunk_size to handle short test content
        chunker = DocumentChunker(ChunkingConfig(min_chunk_size=20))
        doc = Document(id="special", content=content, source="test.txt", doc_type="text")

        chunks = await chunker.chunk_document(doc, mock_embedding_fn)

        assert len(chunks) > 0
        # Special characters should be preserved
        assert any("🎉" in c.content for c in chunks)


# ============================================================================
# Embedding Generation Tests
# ============================================================================


class TestEmbeddingGeneration:
    """Test embedding generation for chunks."""

    @pytest.mark.asyncio
    async def test_embedding_generation(self, mock_embedding_fn):
        """Test that embeddings are generated for all chunks."""
        content = "This is test content. " * 50
        chunker = DocumentChunker(ChunkingConfig(chunk_size=500))
        doc = Document(id="embed1", content=content, source="test.txt", doc_type="text")

        chunks = await chunker.chunk_document(doc, mock_embedding_fn)

        # All chunks should have embeddings
        assert len(chunks) > 0
        assert all(len(c.embedding) == 384 for c in chunks)
        # Embeddings should be lists of floats
        assert all(isinstance(e, float) for c in chunks for e in c.embedding)

    @pytest.mark.asyncio
    async def test_embedding_dimensions(self, mock_embedding_fn):
        """Test that embeddings have consistent dimensions."""
        content = "Test content for dimension check. " * 30
        chunker = DocumentChunker()
        doc = Document(id="embed2", content=content, source="test.txt", doc_type="text")

        chunks = await chunker.chunk_document(doc, mock_embedding_fn)

        # All embeddings should have same dimension
        if len(chunks) > 1:
            first_dim = len(chunks[0].embedding)
            assert all(len(c.embedding) == first_dim for c in chunks)

    @pytest.mark.asyncio
    async def test_embedding_uniqueness(self, mock_embedding_fn):
        """Test that different content produces different embeddings."""
        chunker = DocumentChunker()
        doc1 = Document(id="e1", content="Content A", source="test1.txt", doc_type="text")
        doc2 = Document(id="e2", content="Content B", source="test2.txt", doc_type="text")

        chunks1 = await chunker.chunk_document(doc1, mock_embedding_fn)
        chunks2 = await chunker.chunk_document(doc2, mock_embedding_fn)

        # Different content should produce different embeddings
        if chunks1 and chunks2:
            assert chunks1[0].embedding != chunks2[0].embedding

    @pytest.mark.asyncio
    async def test_embedding_batch_processing(self, mock_embedding_fn):
        """Test processing multiple documents in batch."""
        chunker = DocumentChunker()

        docs = [
            Document(
                id=f"batch{i}",
                content=f"Content {i} " * 100,
                source=f"test{i}.txt",
                doc_type="text",
            )
            for i in range(5)
        ]

        all_chunks = []
        for doc in docs:
            chunks = await chunker.chunk_document(doc, mock_embedding_fn)
            all_chunks.extend(chunks)

        # All documents should be chunked
        assert len(all_chunks) > 0
        # All chunks should have embeddings
        assert all(len(c.embedding) == 384 for c in all_chunks)

    @pytest.mark.asyncio
    async def test_embedding_error_handling(self):
        """Test handling of embedding generation errors."""

        # Mock embedding function that raises an error
        async def failing_embedding_fn(text: str):
            raise RuntimeError("Embedding service unavailable")

        # Use smaller min_chunk_size to ensure content gets chunked
        chunker = DocumentChunker(ChunkingConfig(min_chunk_size=10))
        doc = Document(
            id="error",
            content="Test content for error handling",
            source="test.txt",
            doc_type="text",
        )

        # Should propagate the error
        with pytest.raises(RuntimeError, match="Embedding service unavailable"):
            await chunker.chunk_document(doc, failing_embedding_fn)

    @pytest.mark.asyncio
    async def test_chunk_metadata_with_embeddings(self, mock_embedding_fn):
        """Test that chunk metadata is preserved with embeddings."""
        metadata = {"author": "Test", "category": "test-doc"}
        content = "Test content with metadata. " * 20
        chunker = DocumentChunker()
        doc = Document(
            id="meta", content=content, source="test.txt", doc_type="text", metadata=metadata
        )

        chunks = await chunker.chunk_document(doc, mock_embedding_fn)

        # Metadata should be preserved in chunks
        assert len(chunks) > 0
        assert all(c.metadata.get("author") == "Test" for c in chunks)
        assert all(c.metadata.get("category") == "test-doc" for c in chunks)

    @pytest.mark.asyncio
    async def test_chunk_index_sequential(self, mock_embedding_fn):
        """Test that chunk indices are sequential."""
        content = "Test content for indexing. " * 100
        chunker = DocumentChunker(ChunkingConfig(chunk_size=300))
        doc = Document(id="index", content=content, source="test.txt", doc_type="text")

        chunks = await chunker.chunk_document(doc, mock_embedding_fn)

        # Chunk indices should be sequential
        if len(chunks) > 1:
            assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# ============================================================================
# Document Storage Tests
# ============================================================================


class TestDocumentStorage:
    """Test document storage functionality."""

    @pytest.mark.asyncio
    async def test_document_store_initialization(self, temp_dir):
        """Test document store initialization."""
        from victor.rag.document_store import DocumentStore, DocumentStoreConfig

        config = DocumentStoreConfig(path=temp_dir / "test_db")
        store = DocumentStore(config=config)

        await store.initialize()

        assert store._initialized is True
        assert store.config.path.exists()

    @pytest.mark.asyncio
    async def test_add_document_to_store(self, temp_dir, mock_embedding_fn):
        """Test adding a document to the store."""
        from victor.rag.document_store import DocumentStore, DocumentStoreConfig

        # Use custom chunking config with smaller min size for testing
        store = DocumentStore(
            config=DocumentStoreConfig(path=temp_dir / "test_store"),
            chunking_config=ChunkingConfig(min_chunk_size=20),
        )
        await store.initialize()

        doc = Document(
            id="store1",
            content="Test content for store that is long enough to be chunked",
            source="test.txt",
        )
        chunks = await store.add_document(doc)

        assert len(chunks) > 0
        assert store._stats["total_documents"] == 1
        assert store._stats["total_chunks"] > 0

    @pytest.mark.asyncio
    async def test_get_document_from_store(self, temp_dir, mock_embedding_fn):
        """Test retrieving a document from the store."""
        from victor.rag.document_store import DocumentStore, DocumentStoreConfig

        # Use custom chunking config with smaller min size for testing
        store = DocumentStore(
            config=DocumentStoreConfig(path=temp_dir / "test_store"),
            chunking_config=ChunkingConfig(min_chunk_size=20),
        )
        await store.initialize()

        doc = Document(
            id="get1",
            content="Content for retrieval that is long enough to chunk",
            source="test.txt",
        )
        await store.add_document(doc)

        retrieved = await store.get_document("get1")

        assert retrieved is not None
        assert retrieved.id == "get1"
        assert retrieved.source == "test.txt"

    @pytest.mark.asyncio
    async def test_delete_document_from_store(self, temp_dir, mock_embedding_fn):
        """Test deleting a document from the store."""
        from victor.rag.document_store import DocumentStore, DocumentStoreConfig

        # Use custom chunking config with smaller min size for testing
        store = DocumentStore(
            config=DocumentStoreConfig(path=temp_dir / "test_store"),
            chunking_config=ChunkingConfig(min_chunk_size=20),
        )
        await store.initialize()

        doc = Document(
            id="del1", content="Content to delete that is long enough", source="test.txt"
        )
        await store.add_document(doc)

        # Delete the document
        deleted = await store.delete_document("del1")

        assert deleted > 0

        # Document should no longer exist
        retrieved = await store.get_document("del1")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_list_documents_in_store(self, temp_dir, mock_embedding_fn):
        """Test listing all documents in the store."""
        from victor.rag.document_store import DocumentStore, DocumentStoreConfig

        # Use custom chunking config with smaller min size for testing
        store = DocumentStore(
            config=DocumentStoreConfig(path=temp_dir / "test_store"),
            chunking_config=ChunkingConfig(min_chunk_size=20),
        )
        await store.initialize()

        # Add multiple documents
        for i in range(3):
            doc = Document(
                id=f"list{i}", content=f"Content {i} that is long enough", source=f"test{i}.txt"
            )
            await store.add_document(doc)

        docs = await store.list_documents()

        assert len(docs) == 3
        assert all(isinstance(d, Document) for d in docs)

    @pytest.mark.asyncio
    async def test_document_store_stats(self, temp_dir, mock_embedding_fn):
        """Test getting store statistics."""
        from victor.rag.document_store import DocumentStore, DocumentStoreConfig

        # Use custom chunking config with smaller min size for testing
        store = DocumentStore(
            config=DocumentStoreConfig(path=temp_dir / "test_store"),
            chunking_config=ChunkingConfig(min_chunk_size=20),
        )
        await store.initialize()

        doc = Document(
            id="stats1", content="Content for stats that is long enough", source="test.txt"
        )
        await store.add_document(doc)

        stats = await store.get_stats()

        assert "total_documents" in stats
        assert "total_chunks" in stats
        assert stats["total_documents"] == 1
        assert stats["total_chunks"] > 0

    @pytest.mark.asyncio
    async def test_document_search_in_store(self, temp_dir, mock_embedding_fn):
        """Test searching documents in the store."""
        from victor.rag.document_store import DocumentStore, DocumentStoreConfig

        # Use custom chunking config with smaller min size for testing
        store = DocumentStore(
            config=DocumentStoreConfig(path=temp_dir / "test_store"),
            chunking_config=ChunkingConfig(min_chunk_size=20),
        )
        await store.initialize()

        doc = Document(
            id="search1",
            content="Machine learning is a subset of AI and is very important",
            source="test.txt",
        )
        await store.add_document(doc)

        # Search for content
        results = await store.search("machine learning", k=5)

        assert len(results) > 0
        assert all(isinstance(r, DocumentSearchResult) for r in results)
        assert all(0 <= r.score <= 1 for r in results)
