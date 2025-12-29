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

"""RAG (Retrieval-Augmented Generation) Vertical Package.

This vertical provides a complete RAG implementation showcasing:
- Document ingestion from multiple formats (PDF, Markdown, Text, Code)
- Vector storage with LanceDB (embedded, no server needed)
- Hybrid search (vector + full-text)
- Semantic chunking with overlap
- Interactive TUI for document management and querying
- Reranking for improved relevance

Package Structure:
    assistant.py        - RAGAssistant vertical class
    document_store.py   - LanceDB-based document storage
    chunker.py          - Intelligent document chunking
    tools/              - RAG-specific tools (ingest, search, query)
    ui/                 - Interactive TUI components
    workflows/          - RAG-specific workflows

Usage:
    from victor.verticals.rag import RAGAssistant

    # Get vertical configuration
    config = RAGAssistant.get_config()

    # Create agent with RAG vertical
    agent = await Agent.create(
        tools=config.tools,
        vertical=RAGAssistant,
    )
"""

from victor.verticals.rag.assistant import RAGAssistant
from victor.verticals.rag.document_store import (
    Document,
    DocumentChunk,
    DocumentStore,
    DocumentStoreConfig,
    SearchResult,
)
from victor.verticals.rag.chunker import DocumentChunker, ChunkingConfig
from victor.verticals.rag.prompts import RAGPromptContributor
from victor.verticals.rag.mode_config import RAGModeConfigProvider
from victor.verticals.rag.tools import (
    RAGIngestTool,
    RAGSearchTool,
    RAGQueryTool,
    RAGListTool,
    RAGDeleteTool,
    RAGStatsTool,
)

__all__ = [
    # Main vertical
    "RAGAssistant",
    # Document store
    "Document",
    "DocumentChunk",
    "DocumentStore",
    "DocumentStoreConfig",
    "SearchResult",
    # Chunking
    "DocumentChunker",
    "ChunkingConfig",
    # Extensions
    "RAGPromptContributor",
    "RAGModeConfigProvider",
    # Tools
    "RAGIngestTool",
    "RAGSearchTool",
    "RAGQueryTool",
    "RAGListTool",
    "RAGDeleteTool",
    "RAGStatsTool",
]
