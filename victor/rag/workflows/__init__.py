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
"""

from typing import Dict, List, Optional, Tuple

from victor.core.verticals.protocols import WorkflowProviderProtocol
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


class RAGWorkflowProvider(WorkflowProviderProtocol):
    """Provides RAG-specific workflows.

    Workflows:
    - ingest: Document ingestion pipeline (Parse -> Chunk -> Embed -> Store)
    - query: Query processing (Enhance -> Search -> Retrieve -> Synthesize)
    - maintenance: Index maintenance (Analyze -> Cleanup -> Optimize -> Report)
    """

    def __init__(self) -> None:
        self._workflows: Optional[Dict[str, WorkflowDefinition]] = None

    def _load_workflows(self) -> Dict[str, WorkflowDefinition]:
        if self._workflows is None:
            self._workflows = {
                "ingest": ingest_workflow(),
                "query": query_workflow(),
                "maintenance": maintenance_workflow(),
            }
        return self._workflows

    def get_workflows(self) -> Dict[str, WorkflowDefinition]:
        """Get workflow definitions for this vertical."""
        return self._load_workflows()

    def get_workflow(self, name: str) -> Optional[WorkflowDefinition]:
        """Get a specific workflow by name."""
        return self._load_workflows().get(name)

    def get_workflow_names(self) -> List[str]:
        """Get list of workflow names."""
        return list(self._load_workflows().keys())

    def get_auto_workflows(self) -> List[Tuple[str, str]]:
        """Get automatically triggered workflows.

        Returns:
            List of (regex_pattern, workflow_name) tuples
        """
        return [
            # Ingest triggers
            (r"ingest\s+document", "ingest"),
            (r"add\s+(to\s+)?knowledge", "ingest"),
            (r"index\s+(new\s+)?document", "ingest"),
            (r"import\s+file", "ingest"),
            # Query triggers
            (r"search\s+(for|the)\s+", "query"),
            (r"find\s+(information|answer)", "query"),
            (r"what\s+(does|is|are)", "query"),
            (r"how\s+(do|does|to)", "query"),
            # Maintenance triggers
            (r"clean(up)?\s+index", "maintenance"),
            (r"optimize\s+(index|search)", "maintenance"),
            (r"maintenance", "maintenance"),
        ]

    def __repr__(self) -> str:
        return f"RAGWorkflowProvider(workflows={len(self._load_workflows())})"


__all__ = [
    # WorkflowBuilder-based workflows
    "RAGWorkflowProvider",
    "ingest_workflow",
    "query_workflow",
    "maintenance_workflow",
]
