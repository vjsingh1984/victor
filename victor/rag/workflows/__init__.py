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

This package provides workflow definitions for common RAG tasks:
- Document ingestion pipeline
- Query processing (Search -> Retrieve -> Synthesize)
- Index maintenance and optimization

Uses YAML-first architecture with Python escape hatches for complex conditions
and transforms that cannot be expressed in YAML.

Example:
    provider = RAGWorkflowProvider()

    # Standard execution
    executor = provider.create_executor(orchestrator)
    result = await executor.execute(workflow, context)

    # Streaming execution
    async for chunk in provider.astream("rag_query", orchestrator, context):
        if chunk.event_type == WorkflowEventType.NODE_COMPLETE:
            print(f"Completed: {chunk.node_name}")

Available workflows (all YAML-defined):
- document_ingest: Document ingestion pipeline
- incremental_update: Update existing index with new/modified documents
- rag_query: Answer questions using retrieved context
- conversation: Multi-turn RAG conversation
- agentic_rag: RAG with agentic reasoning and tool use

WorkflowBuilder-based workflows (for backwards compatibility):
- ingest: Document ingestion pipeline (Parse -> Chunk -> Embed -> Store)
- query: Query processing (Enhance -> Search -> Retrieve -> Synthesize)
- maintenance: Index maintenance (Analyze -> Cleanup -> Optimize -> Report)
"""

from typing import Dict, List, Optional, Tuple

from victor.framework.workflows import BaseYAMLWorkflowProvider
from victor.workflows.definition import (
    WorkflowBuilder,
    WorkflowDefinition,
    workflow,
)


@workflow("ingest", "Document ingestion pipeline")
def ingest_workflow() -> WorkflowDefinition:
    """Create document ingestion workflow.

    Pipeline: Parse -> Chunk -> Embed -> Store
    """
    return (
        WorkflowBuilder("ingest")
        .set_metadata("category", "rag")
        .set_metadata("complexity", "medium")
        # Parse and extract content from documents
        .add_agent(
            "parse",
            role="researcher",
            goal="Parse and extract content from input documents",
            tool_budget=15,
            allowed_tools=["read", "ls", "web_fetch"],
            output_key="parsed_content",
        )
        # Process and chunk the content
        .add_agent(
            "process",
            role="executor",
            goal="Process and chunk content for embedding",
            tool_budget=10,
            allowed_tools=["rag_ingest"],
            input_mapping={"content": "parsed_content"},
            output_key="processed_chunks",
        )
        # Validate ingestion
        .add_agent(
            "validate",
            role="reviewer",
            goal="Validate ingested documents and report statistics",
            tool_budget=5,
            allowed_tools=["rag_stats", "rag_list"],
            next_nodes=[],
        )
        .build()
    )


@workflow("query", "RAG query processing workflow")
def query_workflow() -> WorkflowDefinition:
    """Create query processing workflow.

    Pipeline: Enhance Query -> Search -> Retrieve Context -> Synthesize Answer
    """
    return (
        WorkflowBuilder("query")
        .set_metadata("category", "rag")
        .set_metadata("complexity", "medium")
        # Enhance and expand the query
        .add_agent(
            "enhance",
            role="researcher",
            goal="Analyze and enhance the user query for better retrieval",
            tool_budget=5,
            allowed_tools=["rag_search"],
            output_key="enhanced_query",
        )
        # Search and retrieve relevant chunks
        .add_agent(
            "retrieve",
            role="researcher",
            goal="Search knowledge base and retrieve relevant context",
            tool_budget=15,
            allowed_tools=["rag_search", "rag_query"],
            input_mapping={"query": "enhanced_query"},
            output_key="retrieved_context",
        )
        # Synthesize answer from context
        .add_agent(
            "synthesize",
            role="executor",
            goal="Synthesize a comprehensive answer from retrieved context with citations",
            tool_budget=10,
            allowed_tools=["rag_query"],
            input_mapping={"context": "retrieved_context"},
            output_key="answer",
        )
        # Verify and fact-check
        .add_agent(
            "verify",
            role="reviewer",
            goal="Verify answer accuracy and ensure proper source citations",
            tool_budget=10,
            allowed_tools=["rag_search", "rag_query"],
            next_nodes=[],
        )
        .build()
    )


@workflow("maintenance", "Index maintenance and optimization")
def maintenance_workflow() -> WorkflowDefinition:
    """Create index maintenance workflow.

    Pipeline: Analyze -> Cleanup -> Optimize -> Report
    """
    return (
        WorkflowBuilder("maintenance")
        .set_metadata("category", "rag")
        .set_metadata("complexity", "low")
        # Analyze current index state
        .add_agent(
            "analyze",
            role="researcher",
            goal="Analyze current index state and identify issues",
            tool_budget=10,
            allowed_tools=["rag_stats", "rag_list"],
            output_key="analysis",
        )
        # Clean up stale or orphaned entries
        .add_agent(
            "cleanup",
            role="executor",
            goal="Clean up stale documents and orphaned entries",
            tool_budget=15,
            allowed_tools=["rag_delete", "rag_list"],
            input_mapping={"issues": "analysis"},
            output_key="cleanup_result",
        )
        # Optimize index performance
        .add_agent(
            "optimize",
            role="executor",
            goal="Optimize index for better search performance",
            tool_budget=10,
            allowed_tools=["rag_stats"],
            output_key="optimization",
        )
        # Generate maintenance report
        .add_agent(
            "report",
            role="planner",
            goal="Generate maintenance summary report",
            tool_budget=5,
            allowed_tools=["rag_stats"],
            next_nodes=[],
        )
        .build()
    )


class RAGWorkflowProvider(BaseYAMLWorkflowProvider):
    """Provides RAG-specific workflows.

    Uses YAML-first architecture with Python escape hatches for complex
    conditions and transforms that cannot be expressed in YAML.

    Inherits from BaseYAMLWorkflowProvider which provides:
    - YAML workflow loading and caching
    - Escape hatches registration from victor.rag.escape_hatches
    - Streaming execution via StreamingWorkflowExecutor
    - Standard workflow execution

    YAML Workflows (from workflows/*.yaml):
    - document_ingest: Document ingestion with parsing, chunking, and embedding
    - incremental_update: Update existing index with new/modified documents
    - rag_query: Answer questions using retrieved context with citations
    - conversation: Multi-turn RAG conversation with context persistence
    - agentic_rag: RAG with agentic reasoning and tool use

    WorkflowBuilder Workflows (backwards compatibility):
    - ingest: Document ingestion pipeline (Parse -> Chunk -> Embed -> Store)
    - query: Query processing (Enhance -> Search -> Retrieve -> Synthesize)
    - maintenance: Index maintenance (Analyze -> Cleanup -> Optimize -> Report)

    Example:
        provider = RAGWorkflowProvider()

        # List available workflows
        print(provider.get_workflow_names())

        # Stream RAG query execution
        async for chunk in provider.astream("rag_query", orchestrator, {}):
            print(f"[{chunk.progress:.0f}%] {chunk.event_type.value}")
    """

    def __init__(self) -> None:
        """Initialize the RAG workflow provider."""
        super().__init__()
        # Cache for WorkflowBuilder-based workflows (backwards compatibility)
        self._builder_workflows: Optional[Dict[str, WorkflowDefinition]] = None

    def _get_escape_hatches_module(self) -> str:
        """Return the module path for RAG escape hatches.

        Returns:
            Module path string for CONDITIONS and TRANSFORMS dictionaries
        """
        return "victor.rag.escape_hatches"

    def _load_builder_workflows(self) -> Dict[str, WorkflowDefinition]:
        """Load WorkflowBuilder-based workflows for backwards compatibility."""
        if self._builder_workflows is None:
            self._builder_workflows = {
                "ingest": ingest_workflow(),
                "query": query_workflow(),
                "maintenance": maintenance_workflow(),
            }
        return self._builder_workflows

    def get_workflows(self) -> Dict[str, WorkflowDefinition]:
        """Get all workflow definitions for this vertical.

        Combines YAML-loaded workflows with WorkflowBuilder-based workflows.

        Returns:
            Dict mapping workflow names to WorkflowDefinition instances
        """
        # Get YAML workflows from base class
        yaml_workflows = super().get_workflows()

        # Add WorkflowBuilder-based workflows for backwards compatibility
        builder_workflows = self._load_builder_workflows()

        # Combine both (YAML takes precedence if names conflict)
        combined = {**builder_workflows, **yaml_workflows}
        return combined

    def get_workflow(self, name: str) -> Optional[WorkflowDefinition]:
        """Get a specific workflow by name.

        Checks both YAML and WorkflowBuilder-based workflows.

        Args:
            name: The workflow name to retrieve

        Returns:
            WorkflowDefinition if found, None otherwise
        """
        # First try YAML workflows from base class
        workflow = super().get_workflow(name)
        if workflow is not None:
            return workflow

        # Then try WorkflowBuilder-based workflows
        return self._load_builder_workflows().get(name)

    def get_auto_workflows(self) -> List[Tuple[str, str]]:
        """Get automatic workflow triggers based on query patterns.

        Returns:
            List of (regex_pattern, workflow_name) tuples for auto-triggering
        """
        return [
            # Ingest triggers (YAML workflow)
            (r"ingest\s+document", "document_ingest"),
            (r"add\s+(to\s+)?knowledge", "document_ingest"),
            (r"index\s+(new\s+)?document", "document_ingest"),
            (r"import\s+file", "document_ingest"),
            # Incremental update triggers
            (r"update\s+index", "incremental_update"),
            (r"refresh\s+documents", "incremental_update"),
            (r"sync\s+documents", "incremental_update"),
            # Query triggers (YAML workflow)
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
            # Maintenance triggers (WorkflowBuilder workflow)
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
    # WorkflowBuilder-based workflows (backwards compatibility)
    "ingest_workflow",
    "query_workflow",
    "maintenance_workflow",
]
