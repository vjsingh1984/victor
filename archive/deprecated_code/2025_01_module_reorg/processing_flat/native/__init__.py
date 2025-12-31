# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""This module has moved to victor.processing.native.

This module is maintained for backward compatibility only.
Please update your imports to use the new location:

    # OLD:
    from victor.native import rolling_hash_blocks, batch_cosine_similarity

    # NEW (preferred):
    from victor.processing.native import rolling_hash_blocks, batch_cosine_similarity
"""

# Re-export everything from the new location for backward compatibility
from victor.processing.native import (
    # Status
    is_native_available,
    get_native_version,
    # Deduplication
    normalize_block,
    rolling_hash_blocks,
    find_duplicate_blocks,
    # Similarity
    cosine_similarity,
    batch_cosine_similarity,
    top_k_similar,
    batch_normalize_vectors,
    batch_cosine_similarity_normalized,
    top_k_similar_normalized,
    # JSON repair
    repair_json,
    extract_json_objects,
    # Hashing
    compute_signature,
    compute_batch_signatures,
    signature_similarity,
    # Streaming filter
    StreamingFilter,
    StreamingChunkResult,
    strip_thinking_tokens,
    contains_thinking_tokens,
    find_thinking_tokens,
    extract_thinking_content,
    # Task classifier
    NativeTaskClassifier,
    NativeClassificationResult,
    NativeTaskType,
    classify_task_native,
    has_action_keywords,
    has_analysis_keywords,
    has_generation_keywords,
    has_negation,
    find_all_keywords,
    # Thinking detector
    ThinkingDetector,
    PatternAnalysis,
    detect_circular_phrases,
    count_circular_patterns,
    find_circular_patterns,
    # Document chunking (NEW)
    chunk_by_sentences,
    chunk_by_chars,
    chunk_by_paragraphs,
    detect_doc_type,
    count_tokens_approx,
    # Secret detection (NEW)
    SecretMatch,
    scan_secrets,
    has_secrets,
    get_secret_types,
    mask_secrets,
    list_secret_patterns,
    scan_secrets_summary,
    # Pattern matching (NEW)
    PatternMatcher,
    PatternMatch,
    contains_any_pattern,
    find_all_patterns,
    count_pattern_matches,
    get_matched_pattern_indices,
    batch_contains_any,
    weighted_pattern_score,
)

__all__ = [
    # Status
    "is_native_available",
    "get_native_version",
    # Deduplication
    "normalize_block",
    "rolling_hash_blocks",
    "find_duplicate_blocks",
    # Similarity
    "cosine_similarity",
    "batch_cosine_similarity",
    "top_k_similar",
    "batch_normalize_vectors",
    "batch_cosine_similarity_normalized",
    "top_k_similar_normalized",
    # JSON repair
    "repair_json",
    "extract_json_objects",
    # Hashing
    "compute_signature",
    "compute_batch_signatures",
    "signature_similarity",
    # Streaming filter
    "StreamingFilter",
    "StreamingChunkResult",
    "strip_thinking_tokens",
    "contains_thinking_tokens",
    "find_thinking_tokens",
    "extract_thinking_content",
    # Task classifier
    "NativeTaskClassifier",
    "NativeClassificationResult",
    "NativeTaskType",
    "classify_task_native",
    "has_action_keywords",
    "has_analysis_keywords",
    "has_generation_keywords",
    "has_negation",
    "find_all_keywords",
    # Thinking detector
    "ThinkingDetector",
    "PatternAnalysis",
    "detect_circular_phrases",
    "count_circular_patterns",
    "find_circular_patterns",
    # Document chunking (NEW)
    "chunk_by_sentences",
    "chunk_by_chars",
    "chunk_by_paragraphs",
    "detect_doc_type",
    "count_tokens_approx",
    # Secret detection (NEW)
    "SecretMatch",
    "scan_secrets",
    "has_secrets",
    "get_secret_types",
    "mask_secrets",
    "list_secret_patterns",
    "scan_secrets_summary",
    # Pattern matching (NEW)
    "PatternMatcher",
    "PatternMatch",
    "contains_any_pattern",
    "find_all_patterns",
    "count_pattern_matches",
    "get_matched_pattern_indices",
    "batch_contains_any",
    "weighted_pattern_score",
]
