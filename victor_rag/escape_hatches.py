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

"""Escape hatches for RAG YAML workflows.

Complex conditions and transforms that cannot be expressed in YAML.
These are registered with the YAML workflow loader for use in condition nodes.

Example YAML usage:
    - id: check_retrieval
      type: condition
      condition: "retrieval_quality"  # References escape hatch
      branches:
        "sufficient": synthesize
        "needs_more": expand_search
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# =============================================================================
# Condition Functions
# =============================================================================


def retrieval_quality(ctx: Dict[str, Any]) -> str:
    """Assess retrieval quality based on relevance scores.

    Args:
        ctx: Workflow context with keys:
            - retrieved_chunks (list): Retrieved document chunks
            - min_relevance (float): Minimum relevance threshold
            - min_chunks (int): Minimum required chunks

    Returns:
        "sufficient", "marginal", or "insufficient"
    """
    chunks = ctx.get("retrieved_chunks", [])
    min_relevance = ctx.get("min_relevance", 0.7)
    min_chunks = ctx.get("min_chunks", 3)

    if not chunks:
        return "insufficient"

    # Calculate relevance scores
    relevant_chunks = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            score = chunk.get("relevance_score", chunk.get("score", 0))
            if score >= min_relevance:
                relevant_chunks.append(chunk)

    if len(relevant_chunks) >= min_chunks:
        return "sufficient"

    if len(relevant_chunks) > 0:
        return "marginal"

    return "insufficient"


def document_quality(ctx: Dict[str, Any]) -> str:
    """Assess document quality for ingestion.

    Args:
        ctx: Workflow context with keys:
            - parsed_content (dict): Parsed document content
            - min_length (int): Minimum content length
            - language (str): Expected language

    Returns:
        "high", "acceptable", "low", or "skip"
    """
    content = ctx.get("parsed_content", {})
    min_length = ctx.get("min_length", 100)

    text = content.get("text", "")
    metadata = content.get("metadata", {})

    text_length = len(text) if isinstance(text, str) else 0

    if text_length < min_length:
        return "skip"

    # Check for quality indicators
    extraction_confidence = metadata.get("confidence", 0.5)

    if text_length > 500 and extraction_confidence >= 0.8:
        return "high"

    if extraction_confidence >= 0.5:
        return "acceptable"

    return "low"


def chunking_strategy(ctx: Dict[str, Any]) -> str:
    """Determine optimal chunking strategy based on document type.

    Args:
        ctx: Workflow context with keys:
            - document_type (str): Type of document (pdf, code, markdown, etc.)
            - content_length (int): Total content length
            - has_structure (bool): Whether document has clear structure

    Returns:
        "semantic", "fixed", "sentence", or "paragraph"
    """
    doc_type = ctx.get("document_type", "text")
    content_length = ctx.get("content_length", 0)
    has_structure = ctx.get("has_structure", False)

    if doc_type in ("code", "python", "javascript", "typescript"):
        return "semantic"

    if doc_type == "markdown" and has_structure:
        return "paragraph"

    if content_length > 10000:
        return "semantic"

    if content_length < 1000:
        return "sentence"

    return "fixed"


def should_reindex(ctx: Dict[str, Any]) -> str:
    """Determine if document should be reindexed.

    Args:
        ctx: Workflow context with keys:
            - document_hash (str): Current document hash
            - stored_hash (str): Previously stored hash
            - force_reindex (bool): Force reindex flag
            - last_indexed (str): Last indexing timestamp

    Returns:
        "reindex" or "skip"
    """
    current_hash = ctx.get("document_hash", "")
    stored_hash = ctx.get("stored_hash", "")
    force = ctx.get("force_reindex", False)

    if force:
        return "reindex"

    if not stored_hash:
        return "reindex"

    if current_hash != stored_hash:
        return "reindex"

    return "skip"


def query_complexity(ctx: Dict[str, Any]) -> str:
    """Assess query complexity for routing.

    Args:
        ctx: Workflow context with keys:
            - query (str): User query
            - intent (str): Detected intent
            - entities (list): Extracted entities

    Returns:
        "simple", "moderate", "complex", or "multi_step"
    """
    query = ctx.get("query", "")
    intent = ctx.get("intent", "")
    entities = ctx.get("entities", [])

    query_length = len(query.split()) if isinstance(query, str) else 0
    entity_count = len(entities) if isinstance(entities, list) else 0

    if intent in ("comparison", "analysis", "synthesis"):
        return "multi_step"

    if query_length > 20 or entity_count > 3:
        return "complex"

    if query_length > 10 or entity_count > 1:
        return "moderate"

    return "simple"


def answer_confidence(ctx: Dict[str, Any]) -> str:
    """Assess confidence in generated answer.

    Args:
        ctx: Workflow context with keys:
            - answer (str): Generated answer
            - source_count (int): Number of sources used
            - relevance_scores (list): Chunk relevance scores

    Returns:
        "high", "medium", "low", or "uncertain"
    """
    answer = ctx.get("answer", "")
    source_count = ctx.get("source_count", 0)
    relevance_scores = ctx.get("relevance_scores", [])

    if not answer:
        return "uncertain"

    if source_count == 0:
        return "uncertain"

    avg_relevance = 0
    if relevance_scores:
        avg_relevance = sum(relevance_scores) / len(relevance_scores)

    if source_count >= 3 and avg_relevance >= 0.8:
        return "high"

    if source_count >= 2 and avg_relevance >= 0.6:
        return "medium"

    if source_count >= 1 and avg_relevance >= 0.4:
        return "low"

    return "uncertain"


def embedding_batch_size(ctx: Dict[str, Any]) -> str:
    """Determine optimal embedding batch size.

    Args:
        ctx: Workflow context with keys:
            - chunk_count (int): Number of chunks to embed
            - available_memory (int): Available memory in MB
            - model_name (str): Embedding model name

    Returns:
        "small", "medium", "large", or "xlarge"
    """
    chunk_count = ctx.get("chunk_count", 0)
    available_memory = ctx.get("available_memory", 4096)

    if chunk_count <= 10:
        return "small"

    if available_memory < 2048:
        return "small"

    if chunk_count <= 100:
        return "medium"

    if chunk_count <= 1000:
        return "large"

    return "xlarge"


# =============================================================================
# Transform Functions
# =============================================================================


def merge_retrieved_chunks(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Merge and deduplicate retrieved chunks from multiple sources.

    Args:
        ctx: Workflow context with parallel retrieval results

    Returns:
        Merged and ranked chunks
    """
    semantic_results = ctx.get("semantic_results", [])
    keyword_results = ctx.get("keyword_results", [])
    graph_results = ctx.get("graph_results", [])

    all_chunks = []
    seen_ids = set()

    for results, source_type in [
        (semantic_results, "semantic"),
        (keyword_results, "keyword"),
        (graph_results, "graph"),
    ]:
        if isinstance(results, list):
            for chunk in results:
                if isinstance(chunk, dict):
                    chunk_id = chunk.get("id", chunk.get("chunk_id", ""))
                    if chunk_id and chunk_id not in seen_ids:
                        seen_ids.add(chunk_id)
                        chunk["retrieval_source"] = source_type
                        all_chunks.append(chunk)

    # Sort by relevance score
    all_chunks.sort(key=lambda x: x.get("relevance_score", x.get("score", 0)), reverse=True)

    return {
        "chunks": all_chunks,
        "total_count": len(all_chunks),
        "by_source": {
            "semantic": len([c for c in all_chunks if c.get("retrieval_source") == "semantic"]),
            "keyword": len([c for c in all_chunks if c.get("retrieval_source") == "keyword"]),
            "graph": len([c for c in all_chunks if c.get("retrieval_source") == "graph"]),
        },
    }


def format_context_window(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Format retrieved chunks into context window for LLM.

    Args:
        ctx: Workflow context with ranked_chunks

    Returns:
        Formatted context with citations
    """
    chunks = ctx.get("ranked_chunks", [])
    max_tokens = ctx.get("max_context_tokens", 4000)

    context_parts = []
    citations = []
    current_tokens = 0

    for i, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            text = chunk.get("text", chunk.get("content", ""))
            source = chunk.get("source", chunk.get("document_id", f"source_{i}"))

            # Rough token estimation (4 chars per token)
            chunk_tokens = len(text) // 4

            if current_tokens + chunk_tokens > max_tokens:
                break

            context_parts.append(f"[{i+1}] {text}")
            citations.append(
                {
                    "index": i + 1,
                    "source": source,
                    "relevance": chunk.get("relevance_score", 0),
                }
            )
            current_tokens += chunk_tokens

    return {
        "context": "\n\n".join(context_parts),
        "citations": citations,
        "token_count": current_tokens,
        "chunks_used": len(context_parts),
    }


def aggregate_ingestion_stats(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate statistics from parallel ingestion operations.

    Args:
        ctx: Workflow context with parallel ingestion results

    Returns:
        Aggregated statistics
    """
    pdf_stats = ctx.get("pdf_stats", {})
    docx_stats = ctx.get("docx_stats", {})
    markdown_stats = ctx.get("markdown_stats", {})
    code_stats = ctx.get("code_stats", {})

    total_docs = 0
    total_chunks = 0
    total_errors = 0

    for stats in [pdf_stats, docx_stats, markdown_stats, code_stats]:
        if isinstance(stats, dict):
            total_docs += stats.get("documents", 0)
            total_chunks += stats.get("chunks", 0)
            total_errors += stats.get("errors", 0)

    return {
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "total_errors": total_errors,
        "success_rate": (total_docs - total_errors) / total_docs if total_docs > 0 else 0,
        "by_type": {
            "pdf": pdf_stats.get("documents", 0),
            "docx": docx_stats.get("documents", 0),
            "markdown": markdown_stats.get("documents", 0),
            "code": code_stats.get("documents", 0),
        },
    }


# =============================================================================
# Registry Exports
# =============================================================================

# Conditions available in YAML workflows
CONDITIONS = {
    "retrieval_quality": retrieval_quality,
    "document_quality": document_quality,
    "chunking_strategy": chunking_strategy,
    "should_reindex": should_reindex,
    "query_complexity": query_complexity,
    "answer_confidence": answer_confidence,
    "embedding_batch_size": embedding_batch_size,
}

# Transforms available in YAML workflows
TRANSFORMS = {
    "merge_retrieved_chunks": merge_retrieved_chunks,
    "format_context_window": format_context_window,
    "aggregate_ingestion_stats": aggregate_ingestion_stats,
}

__all__ = [
    # Conditions
    "retrieval_quality",
    "document_quality",
    "chunking_strategy",
    "should_reindex",
    "query_complexity",
    "answer_confidence",
    "embedding_batch_size",
    # Transforms
    "merge_retrieved_chunks",
    "format_context_window",
    "aggregate_ingestion_stats",
    # Registries
    "CONDITIONS",
    "TRANSFORMS",
]
