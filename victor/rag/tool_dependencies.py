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

"""RAG Tool Dependencies - Tool relationships for RAG workflows.

Extends the core BaseToolDependencyProvider with RAG-specific data.
Also provides composed tool patterns using ToolExecutionGraph.

Uses canonical tool names from ToolNames to ensure consistent naming
across RL Q-values, workflow patterns, and vertical configurations.

Example:
    from victor.rag.tool_dependencies import (
        RAGToolDependencyProvider,
        get_rag_tool_graph,
        RAG_COMPOSED_PATTERNS,
    )

    # Get tool graph for RAG workflows
    graph = get_rag_tool_graph()

    # Suggest next tools after searching
    suggestions = graph.suggest_next_tools("rag_search", history=["rag_query"])

    # Plan tool sequence for document ingestion
    plan = graph.plan_for_goal({"rag_ingest", "rag_stats"})
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from victor.core.tool_dependency_base import BaseToolDependencyProvider, ToolDependencyConfig
from victor.core.tool_types import ToolDependency
from victor.framework.tool_naming import ToolNames
from victor.tools.tool_graph import ToolExecutionGraph


# Tool dependency graph for RAG workflows
# Uses canonical ToolNames constants and RAG-specific tool names
RAG_TOOL_TRANSITIONS: Dict[str, List[Tuple[str, float]]] = {
    # Reading leads to ingestion or search
    ToolNames.READ: [
        ("rag_ingest", 0.5),  # Ingest the file content
        ("rag_search", 0.2),  # Search for similar content
        (ToolNames.LS, 0.2),  # List more files
        (ToolNames.READ, 0.1),  # Read another file
    ],
    # Listing files leads to reading or ingestion
    ToolNames.LS: [
        (ToolNames.READ, 0.5),  # Read a listed file
        ("rag_ingest", 0.3),  # Ingest files
        (ToolNames.LS, 0.2),  # Continue listing
    ],
    # Web fetch leads to ingestion
    ToolNames.WEB_FETCH: [
        ("rag_ingest", 0.6),  # Ingest web content
        (ToolNames.READ, 0.2),  # Read local files
        (ToolNames.WEB_FETCH, 0.2),  # Fetch more pages
    ],
    # RAG search transitions
    "rag_search": [
        ("rag_query", 0.4),  # Query with found context
        ("rag_search", 0.3),  # Refine search
        (ToolNames.READ, 0.2),  # Read source documents
        ("rag_list", 0.1),  # List documents
    ],
    # RAG query transitions
    "rag_query": [
        ("rag_search", 0.4),  # Search for more context
        ("rag_query", 0.3),  # Follow-up query
        (ToolNames.READ, 0.2),  # Read source files
        ("rag_stats", 0.1),  # Check statistics
    ],
    # RAG ingest transitions
    "rag_ingest": [
        ("rag_stats", 0.3),  # Check ingestion stats
        ("rag_list", 0.3),  # List ingested documents
        ("rag_search", 0.2),  # Search the new content
        (ToolNames.READ, 0.2),  # Read more files
    ],
    # RAG list transitions
    "rag_list": [
        ("rag_search", 0.4),  # Search documents
        ("rag_delete", 0.2),  # Delete documents
        ("rag_stats", 0.2),  # Check stats
        (ToolNames.READ, 0.2),  # Read source files
    ],
    # RAG delete transitions
    "rag_delete": [
        ("rag_list", 0.4),  # List remaining documents
        ("rag_stats", 0.3),  # Check updated stats
        ("rag_ingest", 0.3),  # Re-ingest if needed
    ],
    # RAG stats transitions
    "rag_stats": [
        ("rag_list", 0.4),  # List documents
        ("rag_search", 0.3),  # Search content
        ("rag_ingest", 0.2),  # Add more content
        ("rag_delete", 0.1),  # Clean up
    ],
}

# Tools that work well together in RAG
RAG_TOOL_CLUSTERS: Dict[str, Set[str]] = {
    "document_reading": {ToolNames.READ, ToolNames.LS, ToolNames.WEB_FETCH},
    "search_operations": {"rag_search", "rag_query"},
    "index_management": {"rag_ingest", "rag_delete", "rag_list", "rag_stats"},
    "content_retrieval": {"rag_search", "rag_query", ToolNames.READ},
}

# Recommended sequences for common RAG tasks
RAG_TOOL_SEQUENCES: Dict[str, List[str]] = {
    "document_ingest": [ToolNames.READ, "rag_ingest", "rag_stats"],
    "web_ingest": [ToolNames.WEB_FETCH, "rag_ingest", "rag_stats"],
    "batch_ingest": [ToolNames.LS, ToolNames.READ, "rag_ingest", "rag_list"],
    "simple_query": ["rag_query"],
    "thorough_query": ["rag_search", "rag_query", "rag_search"],
    "exploration": ["rag_stats", "rag_list", "rag_search"],
    "cleanup": ["rag_list", "rag_delete", "rag_stats"],
    "maintenance": ["rag_stats", "rag_list", "rag_delete", "rag_stats"],
}

# Tool dependencies for RAG
RAG_TOOL_DEPENDENCIES: List[ToolDependency] = [
    ToolDependency(
        tool_name="rag_ingest",
        depends_on={ToolNames.READ, ToolNames.WEB_FETCH},  # Need content to ingest
        enables={"rag_search", "rag_query", "rag_list"},  # Enables search after ingest
        weight=0.7,
    ),
    ToolDependency(
        tool_name="rag_search",
        depends_on=set(),  # Can search anytime
        enables={"rag_query"},  # Search enables query
        weight=0.6,
    ),
    ToolDependency(
        tool_name="rag_query",
        depends_on=set(),  # Can query directly
        enables={"rag_search"},  # May need more search
        weight=0.6,
    ),
    ToolDependency(
        tool_name="rag_delete",
        depends_on={"rag_list"},  # Should list before delete
        enables={"rag_stats"},  # Check stats after delete
        weight=0.5,
    ),
    ToolDependency(
        tool_name="rag_list",
        depends_on=set(),
        enables={"rag_delete", "rag_search"},
        weight=0.4,
    ),
    ToolDependency(
        tool_name="rag_stats",
        depends_on=set(),
        enables=set(),
        weight=0.3,
    ),
]

# Required tools for RAG
RAG_REQUIRED_TOOLS: Set[str] = {"rag_search", "rag_query", "rag_ingest"}

# Optional tools that enhance RAG
RAG_OPTIONAL_TOOLS: Set[str] = {
    ToolNames.READ,
    ToolNames.LS,
    ToolNames.WEB_FETCH,
    "rag_list",
    "rag_delete",
    "rag_stats",
}


# =============================================================================
# Composed Tool Patterns
# =============================================================================
# These represent higher-level operations composed of multiple tool calls
# that commonly appear together in RAG workflows.

RAG_COMPOSED_PATTERNS: Dict[str, Dict[str, Any]] = {
    "document_ingestion": {
        "description": "Ingest documents from local files",
        "sequence": [ToolNames.LS, ToolNames.READ, "rag_ingest", "rag_stats"],
        "inputs": {"document_path", "document_type"},
        "outputs": {"document_ids", "chunk_count"},
        "weight": 0.9,
    },
    "web_content_ingestion": {
        "description": "Ingest content from web URLs",
        "sequence": [ToolNames.WEB_FETCH, "rag_ingest", "rag_stats"],
        "inputs": {"url"},
        "outputs": {"document_id", "chunk_count"},
        "weight": 0.85,
    },
    "semantic_search": {
        "description": "Search knowledge base with semantic query",
        "sequence": ["rag_search", "rag_query"],
        "inputs": {"query"},
        "outputs": {"relevant_chunks", "answer"},
        "weight": 0.9,
    },
    "comprehensive_query": {
        "description": "Query with multiple search strategies",
        "sequence": ["rag_search", "rag_search", "rag_query"],
        "inputs": {"query", "search_strategies"},
        "outputs": {"answer", "sources"},
        "weight": 0.85,
    },
    "index_cleanup": {
        "description": "Clean up stale documents from index",
        "sequence": ["rag_stats", "rag_list", "rag_delete", "rag_stats"],
        "inputs": {"cleanup_criteria"},
        "outputs": {"deleted_count", "remaining_count"},
        "weight": 0.8,
    },
    "knowledge_base_audit": {
        "description": "Audit knowledge base content and statistics",
        "sequence": ["rag_stats", "rag_list"],
        "inputs": {},
        "outputs": {"document_count", "total_chunks", "document_list"},
        "weight": 0.75,
    },
}


class RAGToolDependencyProvider(BaseToolDependencyProvider):
    """Tool dependency provider for RAG vertical.

    Extends BaseToolDependencyProvider with RAG-specific tool
    relationships for document ingestion, search, and Q&A workflows.

    Uses canonical ToolNames constants for consistency.
    """

    def __init__(self):
        """Initialize the provider with RAG-specific config."""
        super().__init__(
            ToolDependencyConfig(
                dependencies=RAG_TOOL_DEPENDENCIES,
                transitions=RAG_TOOL_TRANSITIONS,
                clusters=RAG_TOOL_CLUSTERS,
                sequences=RAG_TOOL_SEQUENCES,
                required_tools=RAG_REQUIRED_TOOLS,
                optional_tools=RAG_OPTIONAL_TOOLS,
                default_sequence=["rag_search", "rag_query"],
            )
        )


# =============================================================================
# ToolExecutionGraph Factory
# =============================================================================

# Cached tool graph instance
_rag_tool_graph: Optional[ToolExecutionGraph] = None


def get_rag_tool_graph() -> ToolExecutionGraph:
    """Get the RAG tool execution graph.

    Creates a ToolExecutionGraph configured with RAG-specific
    dependencies, transitions, sequences, and composed patterns.

    Returns:
        ToolExecutionGraph for RAG workflows

    Example:
        graph = get_rag_tool_graph()

        # Suggest next tools
        suggestions = graph.suggest_next_tools("rag_search")

        # Plan tool sequence
        plan = graph.plan_for_goal({"rag_query"})

        # Validate tool execution
        valid, missing = graph.validate_execution("rag_delete", {"rag_list"})
    """
    global _rag_tool_graph

    if _rag_tool_graph is not None:
        return _rag_tool_graph

    graph = ToolExecutionGraph(name="rag")

    # Add dependencies
    for dep in RAG_TOOL_DEPENDENCIES:
        graph.add_dependency(
            tool_name=dep.tool_name,
            depends_on=dep.depends_on,
            enables=dep.enables,
            weight=dep.weight,
        )

    # Add transitions
    graph.add_transitions(RAG_TOOL_TRANSITIONS)

    # Add sequences
    for name, sequence in RAG_TOOL_SEQUENCES.items():
        graph.add_sequence(sequence, weight=0.7)

    # Add clusters
    for name, tools in RAG_TOOL_CLUSTERS.items():
        graph.add_cluster(name, tools)

    # Add composed patterns as sequences with higher weights
    for pattern_name, pattern_data in RAG_COMPOSED_PATTERNS.items():
        graph.add_sequence(pattern_data["sequence"], weight=pattern_data["weight"])

    _rag_tool_graph = graph
    return graph


def reset_rag_tool_graph() -> None:
    """Reset the cached RAG tool graph.

    Useful for testing or when tool configurations change.
    """
    global _rag_tool_graph
    _rag_tool_graph = None


def get_composed_pattern(pattern_name: str) -> Optional[Dict[str, Any]]:
    """Get a composed tool pattern by name.

    Args:
        pattern_name: Name of the pattern (e.g., "document_ingestion")

    Returns:
        Pattern configuration dict or None if not found

    Example:
        pattern = get_composed_pattern("document_ingestion")
        if pattern:
            print(f"Sequence: {pattern['sequence']}")
            print(f"Inputs: {pattern['inputs']}")
    """
    return RAG_COMPOSED_PATTERNS.get(pattern_name)


def list_composed_patterns() -> List[str]:
    """List all available composed tool patterns.

    Returns:
        List of pattern names
    """
    return list(RAG_COMPOSED_PATTERNS.keys())


__all__ = [
    # Provider class
    "RAGToolDependencyProvider",
    # Data exports
    "RAG_TOOL_DEPENDENCIES",
    "RAG_TOOL_TRANSITIONS",
    "RAG_TOOL_CLUSTERS",
    "RAG_TOOL_SEQUENCES",
    "RAG_REQUIRED_TOOLS",
    "RAG_OPTIONAL_TOOLS",
    "RAG_COMPOSED_PATTERNS",
    # Graph functions
    "get_rag_tool_graph",
    "reset_rag_tool_graph",
    "get_composed_pattern",
    "list_composed_patterns",
]
