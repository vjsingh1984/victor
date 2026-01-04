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

"""RAG vertical workflows.

This package provides workflow definitions for common RAG tasks using
YAML-first architecture with Python escape hatches for complex conditions.

Available workflows (all YAML-defined):
- document_ingest: Document ingestion with parsing, chunking, and embedding
- incremental_update: Update existing index with new/modified documents
- rag_query: Answer questions using retrieved context with citations
- conversation: Multi-turn RAG conversation with context persistence
- agentic_rag: RAG with agentic reasoning and tool use
- maintenance: Index maintenance and optimization

Example:
    provider = RAGWorkflowProvider()

    # Standard execution
    executor = provider.create_executor(orchestrator)
    result = await executor.execute(workflow, context)

    # Streaming execution
    async for chunk in provider.astream("rag_query", orchestrator, context):
        if chunk.event_type == WorkflowEventType.NODE_COMPLETE:
            print(f"Completed: {chunk.node_name}")

Usage:
    from victor.rag.workflows import RAGWorkflowProvider

    provider = RAGWorkflowProvider()

    # List available workflows
    print(provider.get_workflow_names())

    # Get a specific workflow
    workflow = provider.get_workflow("rag_query")

    # Stream workflow execution
    async for chunk in provider.astream("rag_query", orchestrator, {}):
        print(f"[{chunk.progress:.0f}%] {chunk.event_type.value}")
"""

from typing import List, Optional, Tuple

from victor.framework.workflows import BaseYAMLWorkflowProvider


class RAGWorkflowProvider(BaseYAMLWorkflowProvider):
    """Provides RAG-specific workflows.

    Uses YAML-first architecture with Python escape hatches for complex
    conditions and transforms that cannot be expressed in YAML.

    Inherits from BaseYAMLWorkflowProvider which provides:
    - YAML workflow loading and caching
    - Escape hatches registration from victor.rag.escape_hatches
    - Streaming execution via StreamingWorkflowExecutor
    - Standard workflow execution

    Available Workflows (all YAML-defined):
    - document_ingest: Document ingestion with parsing, chunking, and embedding
    - incremental_update: Update existing index with new/modified documents
    - rag_query: Answer questions using retrieved context with citations
    - conversation: Multi-turn RAG conversation with context persistence
    - agentic_rag: RAG with agentic reasoning and tool use
    - maintenance: Index maintenance and optimization

    Example:
        provider = RAGWorkflowProvider()

        # List available workflows
        print(provider.get_workflow_names())

        # Stream RAG query execution
        async for chunk in provider.astream("rag_query", orchestrator, {}):
            print(f"[{chunk.progress:.0f}%] {chunk.event_type.value}")
    """

    def _get_escape_hatches_module(self) -> str:
        """Return the module path for RAG escape hatches.

        Returns:
            Module path string for CONDITIONS and TRANSFORMS dictionaries
        """
        return "victor.rag.escape_hatches"

    def get_auto_workflows(self) -> List[Tuple[str, str]]:
        """Get automatic workflow triggers based on query patterns.

        Returns:
            List of (regex_pattern, workflow_name) tuples for auto-triggering
        """
        return [
            # Ingest triggers
            (r"ingest\s+document", "document_ingest"),
            (r"add\s+(to\s+)?knowledge", "document_ingest"),
            (r"index\s+(new\s+)?document", "document_ingest"),
            (r"import\s+file", "document_ingest"),
            # Incremental update triggers
            (r"update\s+index", "incremental_update"),
            (r"refresh\s+documents", "incremental_update"),
            (r"sync\s+documents", "incremental_update"),
            # Query triggers
            (r"search\s+(for|the)\s+", "rag_query"),
            (r"find\s+(information|answer)", "rag_query"),
            (r"what\s+(does|is|are)", "rag_query"),
            (r"how\s+(do|does|to)", "rag_query"),
            # Conversation triggers
            (r"chat\s+(about|with)", "conversation"),
            (r"discuss\s+", "conversation"),
            # Agentic RAG triggers
            (r"deep\s+search", "agentic_rag"),
            (r"research\s+", "agentic_rag"),
            # Maintenance triggers
            (r"clean(up)?\s+index", "maintenance"),
            (r"optimize\s+(index|search)", "maintenance"),
            (r"maintenance", "maintenance"),
        ]

    def get_workflow_for_task_type(self, task_type: str) -> Optional[str]:
        """Get appropriate workflow for task type.

        Args:
            task_type: Type of task (e.g., "ingest", "query")

        Returns:
            Workflow name string or None if no mapping exists
        """
        mapping = {
            # Ingestion
            "ingest": "document_ingest",
            "ingestion": "document_ingest",
            "index": "document_ingest",
            "update": "incremental_update",
            "sync": "incremental_update",
            # Query
            "query": "rag_query",
            "search": "rag_query",
            "question": "rag_query",
            "qa": "rag_query",
            # Conversation
            "conversation": "conversation",
            "chat": "conversation",
            "dialog": "conversation",
            # Agentic
            "research": "agentic_rag",
            "deep_search": "agentic_rag",
            "agentic": "agentic_rag",
            # Maintenance
            "maintenance": "maintenance",
            "cleanup": "maintenance",
            "optimize": "maintenance",
        }
        return mapping.get(task_type.lower())


# Register RAG domain handlers when this module is loaded
from victor.rag.handlers import register_handlers as _register_handlers

_register_handlers()

__all__ = [
    # YAML-first workflow provider
    "RAGWorkflowProvider",
]
