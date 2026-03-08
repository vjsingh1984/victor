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

This module re-exports RAG tools from victor_rag.tools so they
can be auto-discovered by the ToolRegistrar.

Tools:
- rag_ingest: Ingest documents into the RAG knowledge base
- rag_search: Search for relevant chunks
- rag_query: Query with automatic context retrieval
- rag_list: List indexed documents
- rag_delete: Delete documents
- rag_stats: Get store statistics
"""

from victor.core.verticals.import_resolver import import_module_with_fallback


def _load_rag_attr(module_path: str, attr_name: str):
    """Resolve RAG tool attributes using external-first import fallbacks."""
    module, _resolved = import_module_with_fallback(module_path)
    if module is None or not hasattr(module, attr_name):
        raise ImportError(f"Unable to resolve RAG tool: {module_path}:{attr_name}")
    return getattr(module, attr_name)


# Re-export RAG tools for auto-discovery
try:
    RAGIngestTool = _load_rag_attr("victor.rag.tools.ingest", "RAGIngestTool")
    RAGSearchTool = _load_rag_attr("victor.rag.tools.search", "RAGSearchTool")
    RAGQueryTool = _load_rag_attr("victor.rag.tools.query", "RAGQueryTool")
    RAGListTool = _load_rag_attr("victor.rag.tools.management", "RAGListTool")
    RAGDeleteTool = _load_rag_attr("victor.rag.tools.management", "RAGDeleteTool")
    RAGStatsTool = _load_rag_attr("victor.rag.tools.management", "RAGStatsTool")

    # List of tool classes for auto-discovery (only if imports succeeded)
    TOOL_CLASSES = [
        RAGIngestTool,
        RAGSearchTool,
        RAGQueryTool,
        RAGListTool,
        RAGDeleteTool,
        RAGStatsTool,
    ]
except ImportError:
    # External vertical package may not be installed
    TOOL_CLASSES = []

    # Define empty exports when vertical not available
    RAGIngestTool = None  # type: ignore[misc,assignment]
    RAGSearchTool = None  # type: ignore[misc,assignment]
    RAGQueryTool = None  # type: ignore[misc,assignment]
    RAGListTool = None  # type: ignore[misc,assignment]
    RAGDeleteTool = None  # type: ignore[misc,assignment]
    RAGStatsTool = None  # type: ignore[misc,assignment]
__all__ = [
    "RAGIngestTool",
    "RAGSearchTool",
    "RAGQueryTool",
    "RAGListTool",
    "RAGDeleteTool",
    "RAGStatsTool",
    "TOOL_CLASSES",
]
