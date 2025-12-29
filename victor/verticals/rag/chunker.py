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

"""Document Chunker - Intelligent document chunking for RAG.

This module provides semantic chunking strategies for documents:
- Sentence-boundary aware chunking
- Code-aware chunking (preserves functions/classes)
- Markdown-aware chunking (preserves structure)
- Configurable overlap for context continuity

Design:
    - ChunkingConfig: Configuration dataclass
    - DocumentChunker: Main chunking class with strategy pattern

Example:
    chunker = DocumentChunker(ChunkingConfig(chunk_size=512, overlap=50))

    # Chunk a document
    doc = Document(id="1", content="...", source="doc.md", doc_type="markdown")
    chunks = await chunker.chunk_document(doc, embedding_fn)
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, List, Optional, Tuple

from victor.verticals.rag.document_store import Document, DocumentChunk


@dataclass
class ChunkingConfig:
    """Configuration for document chunking.

    Attributes:
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        min_chunk_size: Minimum chunk size (avoid tiny chunks)
        max_chunk_size: Maximum chunk size (hard limit)
        respect_sentence_boundaries: Try to break at sentence ends
        respect_paragraph_boundaries: Try to break at paragraphs
        code_aware: Use code-aware chunking for code files
    """
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    respect_sentence_boundaries: bool = True
    respect_paragraph_boundaries: bool = True
    code_aware: bool = True


# Type alias for embedding function
EmbeddingFn = Callable[[str], Coroutine[Any, Any, List[float]]]


class DocumentChunker:
    """Intelligent document chunker with multiple strategies.

    Supports:
    - Plain text chunking with sentence boundaries
    - Markdown-aware chunking preserving structure
    - Code-aware chunking preserving functions/classes
    - Configurable overlap for context continuity

    Example:
        chunker = DocumentChunker()
        chunks = await chunker.chunk_document(doc, embedding_fn)
    """

    # Sentence ending patterns
    SENTENCE_END = re.compile(r'[.!?]\s+')

    # Code block patterns
    CODE_BLOCK = re.compile(r'```[\s\S]*?```', re.MULTILINE)
    FUNCTION_DEF = re.compile(r'^(def|async def|function|fn|func)\s+\w+', re.MULTILINE)
    CLASS_DEF = re.compile(r'^(class|struct|interface|impl)\s+\w+', re.MULTILINE)

    # Markdown patterns
    MARKDOWN_HEADER = re.compile(r'^#{1,6}\s+.+$', re.MULTILINE)
    MARKDOWN_LIST = re.compile(r'^[\s]*[-*+]\s+', re.MULTILINE)

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig()

    async def chunk_document(
        self,
        doc: Document,
        embedding_fn: EmbeddingFn,
    ) -> List[DocumentChunk]:
        """Chunk a document into indexed chunks.

        Selects the appropriate chunking strategy based on document type
        and generates embeddings for each chunk.

        Args:
            doc: Document to chunk
            embedding_fn: Async function to generate embeddings

        Returns:
            List of document chunks with embeddings
        """
        # Select chunking strategy based on document type
        if doc.doc_type == "code":
            raw_chunks = self._chunk_code(doc.content)
        elif doc.doc_type == "markdown":
            raw_chunks = self._chunk_markdown(doc.content)
        else:
            raw_chunks = self._chunk_text(doc.content)

        # Generate embeddings and create chunks
        chunks = []
        for i, (content, start, end) in enumerate(raw_chunks):
            embedding = await embedding_fn(content)

            chunk = DocumentChunk(
                id=f"{doc.id}_{i}_{uuid.uuid4().hex[:8]}",
                doc_id=doc.id,
                content=content,
                embedding=embedding,
                chunk_index=i,
                start_char=start,
                end_char=end,
                metadata={
                    "doc_type": doc.doc_type,
                    "source": doc.source,
                    **doc.metadata,
                },
            )
            chunks.append(chunk)

        return chunks

    def _chunk_text(self, content: str) -> List[Tuple[str, int, int]]:
        """Chunk plain text with sentence boundaries.

        Args:
            content: Text content

        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        chunks = []
        current_pos = 0

        while current_pos < len(content):
            # Determine chunk end position
            chunk_end = min(current_pos + self.config.chunk_size, len(content))

            # Try to find a sentence boundary
            if self.config.respect_sentence_boundaries and chunk_end < len(content):
                # Look for sentence end in the last portion of the chunk
                search_start = max(current_pos + self.config.min_chunk_size, chunk_end - 100)
                search_text = content[search_start:chunk_end + 50]

                match = None
                for m in self.SENTENCE_END.finditer(search_text):
                    match = m

                if match:
                    chunk_end = search_start + match.end()

            # Extract chunk
            chunk_text = content[current_pos:chunk_end].strip()

            # Skip empty chunks
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append((chunk_text, current_pos, chunk_end))

            # Move to next chunk with overlap
            current_pos = chunk_end - self.config.chunk_overlap
            if current_pos >= len(content) - self.config.min_chunk_size:
                break

        return chunks

    def _chunk_markdown(self, content: str) -> List[Tuple[str, int, int]]:
        """Chunk markdown preserving structure.

        Respects headers and code blocks as natural break points.

        Args:
            content: Markdown content

        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        chunks = []

        # Find all headers as potential break points
        headers = list(self.MARKDOWN_HEADER.finditer(content))

        if not headers:
            # Fall back to text chunking
            return self._chunk_text(content)

        # Chunk by headers
        for i, header in enumerate(headers):
            start = header.start()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(content)

            section = content[start:end].strip()

            # If section is too large, sub-chunk it
            if len(section) > self.config.max_chunk_size:
                sub_chunks = self._chunk_text(section)
                for sub_text, sub_start, sub_end in sub_chunks:
                    chunks.append((sub_text, start + sub_start, start + sub_end))
            elif len(section) >= self.config.min_chunk_size:
                chunks.append((section, start, end))

        return chunks

    def _chunk_code(self, content: str) -> List[Tuple[str, int, int]]:
        """Chunk code preserving function/class boundaries.

        Attempts to keep functions and classes intact.

        Args:
            content: Code content

        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        chunks = []

        # Find function and class definitions
        definitions = []
        for match in self.FUNCTION_DEF.finditer(content):
            definitions.append(("function", match.start()))
        for match in self.CLASS_DEF.finditer(content):
            definitions.append(("class", match.start()))

        # Sort by position
        definitions.sort(key=lambda x: x[1])

        if not definitions:
            # Fall back to text chunking
            return self._chunk_text(content)

        # Add content before first definition
        if definitions[0][1] > self.config.min_chunk_size:
            pre_content = content[:definitions[0][1]].strip()
            if len(pre_content) >= self.config.min_chunk_size:
                chunks.append((pre_content, 0, definitions[0][1]))

        # Chunk by definitions
        for i, (def_type, start) in enumerate(definitions):
            end = definitions[i + 1][1] if i + 1 < len(definitions) else len(content)

            section = content[start:end].strip()

            # If section is too large, sub-chunk it
            if len(section) > self.config.max_chunk_size:
                sub_chunks = self._chunk_text(section)
                for sub_text, sub_start, sub_end in sub_chunks:
                    chunks.append((sub_text, start + sub_start, start + sub_end))
            elif len(section) >= self.config.min_chunk_size:
                chunks.append((section, start, end))

        return chunks

    def estimate_chunks(self, content: str) -> int:
        """Estimate number of chunks for content.

        Useful for progress estimation.

        Args:
            content: Content to estimate

        Returns:
            Estimated chunk count
        """
        if not content:
            return 0

        effective_size = self.config.chunk_size - self.config.chunk_overlap
        return max(1, len(content) // effective_size)
