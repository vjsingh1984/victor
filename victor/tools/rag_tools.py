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

"""RAG Tools - Auto-discovery wrappers for RAG vertical tools.

This module re-exports RAG tools from victor.verticals.rag.tools so they
can be auto-discovered by the ToolRegistrar.

Tools:
- rag_ingest: Ingest documents into the RAG knowledge base
- rag_search: Search for relevant chunks
- rag_query: Query with automatic context retrieval
- rag_list: List indexed documents
- rag_delete: Delete documents
- rag_stats: Get store statistics
"""

# Re-export RAG tools for auto-discovery
from victor.verticals.rag.tools.ingest import RAGIngestTool
from victor.verticals.rag.tools.search import RAGSearchTool
from victor.verticals.rag.tools.query import RAGQueryTool
from victor.verticals.rag.tools.management import RAGListTool, RAGDeleteTool, RAGStatsTool

# List of tool classes for auto-discovery
TOOL_CLASSES = [
    RAGIngestTool,
    RAGSearchTool,
    RAGQueryTool,
    RAGListTool,
    RAGDeleteTool,
    RAGStatsTool,
]

__all__ = [
    "RAGIngestTool",
    "RAGSearchTool",
    "RAGQueryTool",
    "RAGListTool",
    "RAGDeleteTool",
    "RAGStatsTool",
    "TOOL_CLASSES",
]
