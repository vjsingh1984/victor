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

This module provides tool dependency configuration for RAG workflows.
The core configuration is loaded from YAML (tool_dependencies.yaml),
while composed patterns remain in Python for complex logic.

Uses canonical tool names from ToolNames to ensure consistent naming
across RL Q-values, workflow patterns, and vertical configurations.

Simplified Usage:
    - Use RAGToolDependencyProvider for vertical tool dependency management
    - Use RAG_COMPOSED_PATTERNS for pre-defined RAG workflow patterns
    - Use get_rag_tool_graph() for tool execution planning and suggestions
    - reset_rag_tool_graph() clears the cached instance for testing

Example:
    from victor.rag.tool_dependencies import (
        RAGToolDependencyProvider,
        get_rag_tool_graph,
        RAG_COMPOSED_PATTERNS,
    )

    # Get the canonical tool dependency provider
    provider = RAGToolDependencyProvider

    # Get tool graph for RAG workflows
    graph = get_rag_tool_graph()

    # Suggest next tools after searching
    suggestions = graph.suggest_next_tools("rag_search", history=["rag_query"])

    # Plan tool sequence for document ingestion
    plan = graph.plan_for_goal({"rag_ingest", "rag_stats"})

    # Access composed patterns
    ingestion_pattern = RAG_COMPOSED_PATTERNS["document_ingestion"]
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider
from victor.framework.tool_naming import ToolNames
from victor.tools.tool_graph import ToolExecutionGraph


# =============================================================================
# RAGToolDependencyProvider (canonical provider)
# =============================================================================
# Create canonical provider for RAG vertical
RAGToolDependencyProvider = create_vertical_tool_dependency_provider("rag")


# These represent higher-level operations composed of multiple tool calls
# that commonly appear together in RAG workflows.


# Uses canonical ToolNames constants for consistency
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
        "sequence": ["web_fetch", "rag_ingest", "rag_stats"],
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
        "inputs": set(),
        "outputs": {"document_count", "total_chunks", "document_list"},
        "weight": 0.75,
    },
}


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

    # Load data from YAML via provider (RAGToolDependencyProvider is already an instance)
    # RAGToolDependencyProvider is Union[YAMLToolDependencyProvider, EmptyToolDependencyProvider]
    # Only YAMLToolDependencyProvider has get_config(), so we need to check
    if hasattr(RAGToolDependencyProvider, "get_config"):
        config = RAGToolDependencyProvider.get_config()  # type: ignore[attr-defined]
        # Extract components from config
        transitions = getattr(config, 'transitions', {})
        clusters = getattr(config, 'clusters', {})
        sequences = getattr(config, 'sequences', {})
        dependencies = getattr(config, 'dependencies', [])
    else:
        # Empty provider, use defaults
        transitions: Dict[str, Any] = {}
        clusters: Dict[str, Any] = {}
        sequences: Dict[str, Any] = {}
        dependencies: List[Any] = []

    # Load composed patterns from constants
    composed_patterns = RAG_COMPOSED_PATTERNS

    graph = ToolExecutionGraph(name="rag")

    # Add dependencies
    for dep in dependencies:
        graph.add_dependency(
            dep.tool_name,
            depends_on=dep.depends_on,
            enables=dep.enables,
            weight=dep.weight,
        )

    # Add transitions
    graph.add_transitions(transitions)

    # Add sequences
    for name, sequence in sequences.items():
        graph.add_sequence(sequence, weight=0.7)

    # Add clusters
    for name, tools in clusters.items():
        graph.add_cluster(name, tools)

    # Add composed patterns as sequences with higher weights
    for pattern_name, pattern_data in composed_patterns.items():
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
    "RAGToolDependencyProvider",
    "RAG_COMPOSED_PATTERNS",
    "get_rag_tool_graph",
    "reset_rag_tool_graph",
    "get_composed_pattern",
    "list_composed_patterns",
]