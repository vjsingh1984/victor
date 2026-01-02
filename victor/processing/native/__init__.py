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

"""Native extensions for high-performance operations.

This module provides Python bindings to Rust implementations of
performance-critical operations. If the Rust extension is not available,
it falls back to pure Python implementations.

Usage:
    from victor.processing.native import (
        rolling_hash_blocks,
        batch_cosine_similarity,
        repair_json,
        compute_signature,
    )

    # Functions automatically use Rust when available
    blocks = rolling_hash_blocks(content, min_block_length=50)
    similarities = batch_cosine_similarity(query, corpus)

Performance:
    - Deduplication: ~10-50x faster with Rust
    - Cosine similarity: ~2-5x faster with SIMD
    - JSON repair: ~5-10x faster with streaming parser
    - Signature hashing: ~10x faster with xxHash3
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try to import the native extension
_NATIVE_AVAILABLE = False
_native = None

try:
    import victor_native as _native

    _NATIVE_AVAILABLE = True
    logger.info(f"Native extensions loaded (version {_native.__version__})")
except ImportError:
    logger.debug("Native extensions not available, using pure Python fallback")


def is_native_available() -> bool:
    """Check if native Rust extensions are available."""
    return _NATIVE_AVAILABLE


def get_native_version() -> Optional[str]:
    """Get the version of the native extension, if available."""
    if _NATIVE_AVAILABLE:
        return _native.__version__
    return None


# =============================================================================
# DEDUPLICATION FUNCTIONS
# =============================================================================


def normalize_block(block: str) -> str:
    """Normalize a content block for consistent hashing.

    Args:
        block: The content block to normalize

    Returns:
        Normalized string (lowercase, collapsed whitespace, no trailing punctuation)
    """
    if _NATIVE_AVAILABLE:
        return _native.normalize_block(block)

    # Pure Python fallback
    normalized = block.strip()
    if not normalized:
        return ""

    # Collapse whitespace
    normalized = re.sub(r"\s+", " ", normalized)
    # Remove trailing punctuation
    normalized = normalized.rstrip(".,;:")
    return normalized.lower()


def rolling_hash_blocks(content: str, min_block_length: int = 50) -> List[Tuple[str, str, bool]]:
    """Process content and compute hashes for all blocks.

    Args:
        content: The full content to process
        min_block_length: Minimum length for a block to be considered for dedup

    Returns:
        List of tuples: (hash string, original block, is_duplicate boolean)
    """
    if _NATIVE_AVAILABLE:
        return _native.rolling_hash_blocks(content, min_block_length)

    # Pure Python fallback
    blocks = _split_into_blocks(content)
    seen_hashes: set = set()
    results: List[Tuple[str, str, bool]] = []

    for block in blocks:
        if len(block.strip()) < min_block_length:
            results.append(("", block, False))
            continue

        block_hash = _hash_block(block)
        is_duplicate = block_hash in seen_hashes
        if not is_duplicate:
            seen_hashes.add(block_hash)

        results.append((block_hash, block, is_duplicate))

    return results


def find_duplicate_blocks(content: str, min_block_length: int = 50) -> List[Tuple[int, str]]:
    """Find duplicate blocks in content and return their indices.

    Args:
        content: The content to analyze
        min_block_length: Minimum length for dedup consideration

    Returns:
        List of (block_index, hash) for duplicate blocks
    """
    if _NATIVE_AVAILABLE:
        return _native.find_duplicate_blocks(content, min_block_length)

    # Pure Python fallback
    blocks = _split_into_blocks(content)
    seen_hashes: set = set()
    duplicates: List[Tuple[int, str]] = []

    for idx, block in enumerate(blocks):
        if len(block.strip()) < min_block_length:
            continue

        block_hash = _hash_block(block)
        if block_hash in seen_hashes:
            duplicates.append((idx, block_hash))
        else:
            seen_hashes.add(block_hash)

    return duplicates


def _split_into_blocks(content: str) -> List[str]:
    """Split content into logical blocks (pure Python)."""
    if not content:
        return []

    # Preserve code blocks as single units
    code_block_pattern = r"```[\s\S]*?```"
    code_blocks = re.findall(code_block_pattern, content)
    content_with_placeholders = re.sub(code_block_pattern, "<<<CODE_BLOCK>>>", content)

    # Split on paragraph breaks
    raw_blocks = re.split(r"\n{2,}", content_with_placeholders)

    # Replace placeholders
    final_blocks = []
    code_idx = 0
    for block in raw_blocks:
        if "<<<CODE_BLOCK>>>" in block:
            if code_idx < len(code_blocks):
                final_blocks.append(code_blocks[code_idx])
                code_idx += 1
        elif block.strip():
            final_blocks.append(block)

    return final_blocks


def _hash_block(block: str) -> str:
    """Compute hash of a normalized block (pure Python)."""
    normalized = normalize_block(block)
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


# =============================================================================
# SIMILARITY FUNCTIONS
# =============================================================================


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity value between -1 and 1

    Raises:
        ValueError: If vectors have different lengths
    """
    if _NATIVE_AVAILABLE:
        return _native.cosine_similarity(a, b)

    # Pure Python fallback using NumPy
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)

    if len(a_arr) != len(b_arr):
        raise ValueError(f"Vectors must have same length: {len(a_arr)} vs {len(b_arr)}")

    if len(a_arr) == 0:
        return 0.0

    norm_a = np.linalg.norm(a_arr) + 1e-9
    norm_b = np.linalg.norm(b_arr) + 1e-9
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def batch_cosine_similarity(query: List[float], corpus: List[List[float]]) -> List[float]:
    """Compute cosine similarity between a query vector and multiple corpus vectors.

    Args:
        query: Query embedding vector
        corpus: List of corpus embedding vectors

    Returns:
        List of similarity scores, one per corpus vector

    Raises:
        ValueError: If query dimension doesn't match corpus dimensions
    """
    if _NATIVE_AVAILABLE:
        return _native.batch_cosine_similarity(query, corpus)

    # Pure Python fallback using NumPy
    if not corpus:
        return []

    query_arr = np.array(query, dtype=np.float32)
    corpus_arr = np.array(corpus, dtype=np.float32)

    if corpus_arr.shape[1] != len(query_arr):
        raise ValueError(
            f"Dimension mismatch: query has {len(query_arr)} dims, "
            f"corpus has {corpus_arr.shape[1]} dims"
        )

    # Normalize
    query_norm = query_arr / (np.linalg.norm(query_arr) + 1e-9)
    corpus_norms = corpus_arr / (np.linalg.norm(corpus_arr, axis=1, keepdims=True) + 1e-9)

    # Compute similarities
    similarities = np.dot(corpus_norms, query_norm)
    return similarities.tolist()


def top_k_similar(
    query: List[float], corpus: List[List[float]], k: int = 10
) -> List[Tuple[int, float]]:
    """Find top-k most similar vectors from a corpus.

    Args:
        query: Query embedding vector
        corpus: List of corpus embedding vectors
        k: Number of top results to return

    Returns:
        List of (index, similarity) tuples, sorted by similarity descending
    """
    if _NATIVE_AVAILABLE:
        return _native.top_k_similar(query, corpus, k)

    # Pure Python fallback
    similarities = batch_cosine_similarity(query, corpus)
    indexed = [(i, sim) for i, sim in enumerate(similarities)]
    indexed.sort(key=lambda x: x[1], reverse=True)
    return indexed[:k]


def batch_normalize_vectors(vectors: List[List[float]]) -> List[List[float]]:
    """Normalize vectors to unit length for efficient similarity computation.

    Pre-normalizing vectors allows subsequent similarity computations to
    skip redundant norm calculations, providing ~2x speedup for batch operations.

    Args:
        vectors: List of vectors to normalize

    Returns:
        List of normalized vectors (unit length)
    """
    if _NATIVE_AVAILABLE:
        return _native.batch_normalize_vectors(vectors)

    # Pure Python fallback using NumPy
    if not vectors:
        return []

    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    normalized = arr / norms
    return normalized.tolist()


def batch_cosine_similarity_normalized(
    query: List[float], normalized_corpus: List[List[float]]
) -> List[float]:
    """Compute cosine similarities with a pre-normalized corpus.

    This is faster than batch_cosine_similarity when the corpus has already
    been normalized via batch_normalize_vectors, as it avoids redundant
    norm calculations.

    Args:
        query: Query embedding vector (will be normalized internally)
        normalized_corpus: List of pre-normalized corpus vectors (unit length)

    Returns:
        List of similarity scores, one per corpus vector
    """
    if _NATIVE_AVAILABLE:
        return _native.batch_cosine_similarity_normalized(query, normalized_corpus)

    # Pure Python fallback using NumPy
    if not normalized_corpus:
        return []

    query_arr = np.array(query, dtype=np.float32)
    corpus_arr = np.array(normalized_corpus, dtype=np.float32)

    # Normalize query
    query_normalized = query_arr / (np.linalg.norm(query_arr) + 1e-9)

    # For pre-normalized corpus, similarity is just dot product
    similarities = np.dot(corpus_arr, query_normalized)
    return similarities.tolist()


def top_k_similar_normalized(
    query: List[float], normalized_corpus: List[List[float]], k: int = 10
) -> List[Tuple[int, float]]:
    """Find top-k similar vectors from a pre-normalized corpus.

    More efficient version of top_k_similar when corpus is already normalized
    via batch_normalize_vectors.

    Args:
        query: Query embedding vector
        normalized_corpus: List of pre-normalized corpus vectors
        k: Number of top results to return

    Returns:
        List of (index, similarity) tuples, sorted by similarity descending
    """
    if _NATIVE_AVAILABLE:
        return _native.top_k_similar_normalized(query, normalized_corpus, k)

    # Pure Python fallback using heap for efficiency
    import heapq

    similarities = batch_cosine_similarity_normalized(query, normalized_corpus)

    # Use heapq.nlargest for efficient top-k selection
    indexed = [(sim, i) for i, sim in enumerate(similarities)]
    top_k = heapq.nlargest(k, indexed)

    # Convert back to (index, similarity) format
    return [(i, sim) for sim, i in top_k]


# =============================================================================
# JSON REPAIR FUNCTIONS
# =============================================================================


def repair_json(input_str: str) -> str:
    """Repair malformed JSON by converting Python-style syntax to valid JSON.

    Handles:
    - Single quotes → double quotes
    - Python True/False/None → JSON true/false/null

    Args:
        input_str: Potentially malformed JSON string

    Returns:
        Repaired JSON string
    """
    if _NATIVE_AVAILABLE:
        return _native.repair_json(input_str)

    # Pure Python fallback
    result = input_str

    # Fast path: check if already valid
    try:
        json.loads(result)
        return result
    except json.JSONDecodeError:
        pass

    # Replace Python literals
    result = result.replace("True", "true")
    result = result.replace("False", "false")
    result = result.replace("None", "null")

    # Replace single quotes with double quotes (simple approach)
    # This is a simplified version; the Rust implementation handles edge cases better
    in_string = False
    output = []
    i = 0
    while i < len(result):
        c = result[i]
        if c == "\\" and i + 1 < len(result):
            output.append(c)
            output.append(result[i + 1])
            i += 2
            continue
        if c == '"':
            in_string = not in_string
            output.append(c)
        elif c == "'" and not in_string:
            output.append('"')
        else:
            output.append(c)
        i += 1

    return "".join(output)


def extract_json_objects(text: str) -> List[Tuple[int, int, str]]:
    """Extract JSON objects from mixed text content.

    Args:
        text: Text that may contain JSON objects

    Returns:
        List of (start_pos, end_pos, json_string) tuples for each found object
    """
    if _NATIVE_AVAILABLE:
        return _native.extract_json_objects(text)

    # Pure Python fallback
    results: List[Tuple[int, int, str]] = []
    i = 0

    while i < len(text):
        if text[i] in "{[":
            # Find matching bracket
            match = _find_json_end(text, i)
            if match:
                end, json_str = match
                # Validate
                try:
                    json.loads(json_str)
                    results.append((i, end, json_str))
                    i = end
                    continue
                except json.JSONDecodeError:
                    # Try repairing
                    repaired = repair_json(json_str)
                    try:
                        json.loads(repaired)
                        results.append((i, end, repaired))
                        i = end
                        continue
                    except json.JSONDecodeError:
                        pass
        i += 1

    return results


def _find_json_end(text: str, start: int) -> Optional[Tuple[int, str]]:
    """Find the end of a JSON structure starting at given position."""
    open_char = text[start]
    close_char = "}" if open_char == "{" else "]"

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        c = text[i]

        if escape_next:
            escape_next = False
            continue

        if c == "\\":
            escape_next = True
            continue

        if c == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if c == open_char:
            depth += 1
        elif c == close_char:
            depth -= 1
            if depth == 0:
                return (i + 1, text[start : i + 1])

    return None


# =============================================================================
# HASHING FUNCTIONS
# =============================================================================


def compute_signature(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Compute a signature hash for a tool call.

    Args:
        tool_name: Name of the tool being called
        arguments: Tool arguments dictionary

    Returns:
        16-character hex string signature
    """
    if _NATIVE_AVAILABLE:
        return _native.compute_signature(tool_name, arguments)

    # Pure Python fallback
    # Sort keys for deterministic output
    sorted_args = sorted(arguments.items())
    args_str = ",".join(f"{k}={_value_to_str(v)}" for k, v in sorted_args)
    combined = f"{tool_name}:{args_str}"
    return hashlib.md5(combined.encode()).hexdigest()[:16]


def compute_batch_signatures(tool_calls: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
    """Compute signatures for multiple tool calls in batch.

    Args:
        tool_calls: List of (tool_name, arguments_dict) tuples

    Returns:
        List of signature strings, one per tool call
    """
    if _NATIVE_AVAILABLE:
        return _native.compute_batch_signatures(tool_calls)

    # Pure Python fallback
    return [compute_signature(name, args) for name, args in tool_calls]


def signature_similarity(sig1: str, sig2: str) -> float:
    """Compute similarity between two signatures.

    Args:
        sig1: First signature
        sig2: Second signature

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if _NATIVE_AVAILABLE:
        return _native.signature_similarity(sig1, sig2)

    # Pure Python fallback
    if sig1 == sig2:
        return 1.0
    if len(sig1) != len(sig2):
        return 0.0

    matching = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matching / len(sig1)


def _value_to_str(value: Any) -> str:
    """Convert a value to a stable string representation."""
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


# =============================================================================
# STREAMING FILTER (Thinking Token Detection)
# =============================================================================


class StreamingChunkResultFallback:
    """Fallback result class for streaming content filter."""

    def __init__(
        self,
        content: str,
        is_thinking: bool = False,
        state_changed: bool = False,
        entering_thinking: bool = False,
        exiting_thinking: bool = False,
    ):
        self.content = content
        self.is_thinking = is_thinking
        self.state_changed = state_changed
        self.entering_thinking = entering_thinking
        self.exiting_thinking = exiting_thinking


# Thinking token patterns
_THINKING_START_PATTERNS = [
    "<｜begin▁of▁thinking｜>",  # DeepSeek Unicode
    "<|begin_of_thinking|>",  # DeepSeek ASCII
    "<think>",  # Qwen3
]

_THINKING_END_PATTERNS = [
    "<｜end▁of▁thinking｜>",  # DeepSeek Unicode
    "<|end_of_thinking|>",  # DeepSeek ASCII
    "</think>",  # Qwen3
]

_ALL_THINKING_PATTERNS = _THINKING_START_PATTERNS + _THINKING_END_PATTERNS


def strip_thinking_tokens(content: str) -> str:
    """Strip all thinking tokens from content.

    Args:
        content: Text potentially containing thinking tokens

    Returns:
        Content with thinking tokens removed
    """
    if _NATIVE_AVAILABLE:
        return _native.strip_thinking_tokens(content)

    # Pure Python fallback
    result = content
    for pattern in _ALL_THINKING_PATTERNS:
        result = result.replace(pattern, "")
    return result


def contains_thinking_tokens(content: str) -> bool:
    """Check if content contains any thinking tokens.

    Args:
        content: Text to check

    Returns:
        True if thinking tokens are present
    """
    if _NATIVE_AVAILABLE:
        return _native.contains_thinking_tokens(content)

    # Pure Python fallback
    return any(pattern in content for pattern in _ALL_THINKING_PATTERNS)


def find_thinking_tokens(content: str) -> List[Tuple[int, int, int]]:
    """Find all thinking token positions in content.

    Args:
        content: Text to search

    Returns:
        List of (start, end, pattern_index) tuples
    """
    if _NATIVE_AVAILABLE:
        return _native.find_thinking_tokens(content)

    # Pure Python fallback
    results = []
    for idx, pattern in enumerate(_ALL_THINKING_PATTERNS):
        pos = 0
        while True:
            found = content.find(pattern, pos)
            if found == -1:
                break
            results.append((found, found + len(pattern), idx))
            pos = found + 1

    results.sort(key=lambda x: x[0])
    return results


def extract_thinking_content(content: str) -> Tuple[str, str]:
    """Extract thinking content from a complete response.

    Args:
        content: Full response text

    Returns:
        Tuple of (main_content, thinking_content)
    """
    if _NATIVE_AVAILABLE:
        return _native.extract_thinking_content(content)

    # Pure Python fallback
    main_content = []
    thinking_content = []
    in_thinking = False
    pos = 0

    while pos < len(content):
        if not in_thinking:
            # Look for start pattern
            earliest_start = len(content)
            start_pattern_len = 0
            for pattern in _THINKING_START_PATTERNS:
                idx = content.find(pattern, pos)
                if idx != -1 and idx < earliest_start:
                    earliest_start = idx
                    start_pattern_len = len(pattern)

            if earliest_start < len(content):
                main_content.append(content[pos:earliest_start])
                in_thinking = True
                pos = earliest_start + start_pattern_len
            else:
                main_content.append(content[pos:])
                break
        else:
            # Look for end pattern
            earliest_end = len(content)
            end_pattern_len = 0
            for pattern in _THINKING_END_PATTERNS:
                idx = content.find(pattern, pos)
                if idx != -1 and idx < earliest_end:
                    earliest_end = idx
                    end_pattern_len = len(pattern)

            if earliest_end < len(content):
                thinking_content.append(content[pos:earliest_end])
                in_thinking = False
                pos = earliest_end + end_pattern_len
            else:
                thinking_content.append(content[pos:])
                break

    return "".join(main_content), "".join(thinking_content)


# Re-export native classes when available
if _NATIVE_AVAILABLE:
    StreamingFilter = _native.StreamingFilter
    StreamingChunkResult = _native.StreamingChunkResult
else:
    # Use Python fallback from response_sanitizer
    try:
        from victor.agent.response_sanitizer import (
            StreamingContentFilter as StreamingFilter,
            StreamingChunkResult,
        )
    except ImportError:
        StreamingFilter = None
        StreamingChunkResult = StreamingChunkResultFallback


# =============================================================================
# TASK CLASSIFIER
# =============================================================================


def classify_task_native(text: str) -> Any:
    """Classify a task using native classifier.

    Args:
        text: User message to classify

    Returns:
        ClassificationResult with task type and confidence
    """
    if _NATIVE_AVAILABLE:
        return _native.classify_task(text)

    # Fallback to unified classifier
    from victor.agent.unified_classifier import classify_task

    return classify_task(text)


def has_action_keywords(text: str) -> bool:
    """Check if text contains action keywords.

    Args:
        text: Text to check

    Returns:
        True if action keywords are present
    """
    if _NATIVE_AVAILABLE:
        return _native.has_action_keywords(text)

    # Pure Python fallback
    action_keywords = [
        "execute",
        "apply",
        "run",
        "deploy",
        "build",
        "install",
        "start",
        "stop",
        "restart",
        "test",
        "commit",
        "push",
        "pull",
        "merge",
    ]
    text_lower = text.lower()
    return any(kw in text_lower for kw in action_keywords)


def has_analysis_keywords(text: str) -> bool:
    """Check if text contains analysis keywords.

    Args:
        text: Text to check

    Returns:
        True if analysis keywords are present
    """
    if _NATIVE_AVAILABLE:
        return _native.has_analysis_keywords(text)

    # Pure Python fallback
    analysis_keywords = [
        "analyze",
        "explore",
        "review",
        "understand",
        "explain",
        "describe",
        "investigate",
        "examine",
        "study",
        "assess",
        "evaluate",
        "summarize",
    ]
    text_lower = text.lower()
    return any(kw in text_lower for kw in analysis_keywords)


def has_generation_keywords(text: str) -> bool:
    """Check if text contains generation keywords.

    Args:
        text: Text to check

    Returns:
        True if generation keywords are present
    """
    if _NATIVE_AVAILABLE:
        return _native.has_generation_keywords(text)

    # Pure Python fallback
    generation_keywords = [
        "create",
        "generate",
        "write",
        "implement",
        "add",
        "new",
        "scaffold",
        "initialize",
        "setup",
        "bootstrap",
    ]
    text_lower = text.lower()
    return any(kw in text_lower for kw in generation_keywords)


def has_negation(text: str) -> bool:
    """Check if text contains negation patterns.

    Args:
        text: Text to check

    Returns:
        True if negation patterns are present
    """
    if _NATIVE_AVAILABLE:
        return _native.has_negation(text)

    # Pure Python fallback
    negation_patterns = [
        "don't",
        "do not",
        "dont",
        "shouldn't",
        "should not",
        "wouldn't",
        "would not",
        "can't",
        "cannot",
        "not",
        "never",
        "without",
        "avoid",
        "skip",
        "ignore",
    ]
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in negation_patterns)


def find_all_keywords(text: str) -> List[Tuple[int, int, str, str]]:
    """Find all keyword matches in text.

    Args:
        text: Text to search

    Returns:
        List of (start, end, matched_text, category) tuples
    """
    if _NATIVE_AVAILABLE:
        return _native.find_all_keywords(text)

    # Pure Python fallback
    results = []
    text_lower = text.lower()

    keyword_categories = {
        "action": [
            "execute",
            "apply",
            "run",
            "deploy",
            "build",
            "install",
            "start",
            "stop",
            "restart",
            "test",
            "commit",
            "push",
        ],
        "analysis": [
            "analyze",
            "explore",
            "review",
            "understand",
            "explain",
            "describe",
            "investigate",
            "examine",
        ],
        "generation": [
            "create",
            "generate",
            "write",
            "implement",
            "add",
            "scaffold",
        ],
        "search": ["find", "search", "locate", "grep", "look for", "where is"],
        "edit": [
            "modify",
            "refactor",
            "fix",
            "update",
            "change",
            "edit",
            "rename",
        ],
    }

    for category, keywords in keyword_categories.items():
        for keyword in keywords:
            pos = 0
            while True:
                found = text_lower.find(keyword, pos)
                if found == -1:
                    break
                end = found + len(keyword)
                matched = text[found:end]
                results.append((found, end, matched, category))
                pos = found + 1

    results.sort(key=lambda x: x[0])
    return results


# Re-export native classes when available
if _NATIVE_AVAILABLE:
    NativeTaskClassifier = _native.TaskClassifier
    NativeClassificationResult = _native.ClassificationResult
    NativeTaskType = _native.TaskType
else:
    NativeTaskClassifier = None
    NativeClassificationResult = None
    NativeTaskType = None


# =============================================================================
# THINKING DETECTOR
# =============================================================================

# Circular thinking patterns
_CIRCULAR_PATTERNS = [
    "let me read",
    "let me check",
    "let me look at",
    "let me examine",
    "let me see",
    "i need to read",
    "i need to check",
    "i need to look at",
    "first let me",
    "now let me",
    "let me first",
    "let me start by",
    "i'll need to",
    "i will need to",
    "let me actually use",
    "let me use the",
    "i'll actually read",
    "i'll read",
    "now i'll",
    "now i will",
    "i should read",
    "i should examine",
    "i should check",
    "let me continue",
    "let me proceed",
]


def detect_circular_phrases(text: str) -> bool:
    """Detect if text contains circular thinking phrases.

    Args:
        text: Text to check

    Returns:
        True if circular phrases are detected
    """
    if _NATIVE_AVAILABLE:
        return _native.detect_circular_phrases(text)

    # Pure Python fallback
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in _CIRCULAR_PATTERNS)


def count_circular_patterns(text: str) -> int:
    """Count circular patterns in text.

    Args:
        text: Text to analyze

    Returns:
        Number of circular pattern matches
    """
    if _NATIVE_AVAILABLE:
        return _native.count_circular_patterns(text)

    # Pure Python fallback
    text_lower = text.lower()
    count = 0
    for pattern in _CIRCULAR_PATTERNS:
        pos = 0
        while True:
            found = text_lower.find(pattern, pos)
            if found == -1:
                break
            count += 1
            pos = found + 1
    return count


def find_circular_patterns(text: str) -> List[Tuple[int, int, str]]:
    """Find all circular pattern matches.

    Args:
        text: Text to search

    Returns:
        List of (start, end, matched_text) tuples
    """
    if _NATIVE_AVAILABLE:
        return _native.find_circular_patterns(text)

    # Pure Python fallback
    text_lower = text.lower()
    results = []
    for pattern in _CIRCULAR_PATTERNS:
        pos = 0
        while True:
            found = text_lower.find(pattern, pos)
            if found == -1:
                break
            end = found + len(pattern)
            results.append((found, end, text[found:end]))
            pos = found + 1
    results.sort(key=lambda x: x[0])
    return results


# Re-export native classes when available
if _NATIVE_AVAILABLE:
    ThinkingDetector = _native.ThinkingDetector
    PatternAnalysis = _native.PatternAnalysis
else:
    ThinkingDetector = None
    PatternAnalysis = None


# =============================================================================
# DOCUMENT CHUNKING (NEW - HIGH impact)
# =============================================================================


def chunk_by_sentences(text: str, chunk_size: int = 1344, overlap: int = 128) -> List[str]:
    """Chunk text by sentence boundaries with overlap.

    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    if _NATIVE_AVAILABLE:
        return _native.chunk_by_sentences(text, chunk_size, overlap)

    # Pure Python fallback
    import re

    # Simple sentence splitting
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep overlap
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:]
            else:
                current_chunk = ""
        current_chunk += " " + sentence if current_chunk else sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def chunk_by_chars(text: str, chunk_size: int = 1344, overlap: int = 128) -> List[str]:
    """Chunk text by character count with overlap.

    Args:
        text: Text to chunk
        chunk_size: Chunk size in characters
        overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    if _NATIVE_AVAILABLE:
        return _native.chunk_by_chars(text, chunk_size, overlap)

    # Pure Python fallback
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap if overlap < end else 0

    return chunks


def chunk_by_paragraphs(text: str, chunk_size: int = 1344, overlap: int = 128) -> List[str]:
    """Chunk text by paragraph boundaries with overlap.

    Args:
        text: Text to chunk
        chunk_size: Target chunk size
        overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    if _NATIVE_AVAILABLE:
        return _native.chunk_by_paragraphs(text, chunk_size, overlap)

    # Pure Python fallback
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:]
            else:
                current_chunk = ""
        current_chunk += "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def detect_doc_type(source: str) -> str:
    """Detect document type from file extension.

    Args:
        source: File path or name

    Returns:
        Document type string
    """
    if _NATIVE_AVAILABLE:
        return _native.detect_doc_type(source)

    # Pure Python fallback
    source_lower = source.lower()
    extensions = {
        ".py": "code",
        ".js": "code",
        ".ts": "code",
        ".java": "code",
        ".go": "code",
        ".rs": "code",
        ".c": "code",
        ".cpp": "code",
        ".md": "markdown",
        ".markdown": "markdown",
        ".rst": "markdown",
        ".html": "html",
        ".htm": "html",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".xml": "xml",
        ".csv": "csv",
        ".txt": "text",
    }
    for ext, doc_type in extensions.items():
        if source_lower.endswith(ext):
            return doc_type
    return "text"


def count_tokens_approx(text: str) -> int:
    """Count approximate tokens in text.

    Args:
        text: Text to count

    Returns:
        Approximate token count
    """
    if _NATIVE_AVAILABLE:
        return _native.count_tokens_approx(text)

    # Pure Python fallback (words + punctuation)
    import re

    words = len(re.findall(r"\w+", text))
    punctuation = len(re.findall(r"[^\w\s]", text))
    return words + punctuation


# =============================================================================
# SECRET DETECTION (NEW - HIGH impact)
# =============================================================================


class SecretMatchFallback:
    """Fallback class for secret match results."""

    def __init__(
        self,
        secret_type: str,
        matched_text: str,
        severity: str,
        start: int,
        end: int,
        line_number: int,
    ):
        self.secret_type = secret_type
        self.matched_text = matched_text
        self.severity = severity
        self.start = start
        self.end = end
        self.line_number = line_number

    def __repr__(self) -> str:
        return f"SecretMatch(type='{self.secret_type}', severity='{self.severity}', line={self.line_number})"


# Secret patterns for fallback
_SECRET_PATTERNS = [
    ("aws_access_key", r"AKIA[0-9A-Z]{16}", "high"),
    ("github_token", r"gh[pousr]_[A-Za-z0-9_]{36,}", "high"),
    ("google_api_key", r"AIza[0-9A-Za-z_-]{35}", "high"),
    ("stripe_key", r"(?:sk|pk)_(?:live|test)_[a-zA-Z0-9]{24,}", "high"),
    ("jwt_token", r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", "medium"),
    ("private_key", r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", "critical"),
]


def scan_secrets(text: str) -> List[Any]:
    """Scan text for secrets.

    Args:
        text: Text to scan

    Returns:
        List of SecretMatch objects
    """
    if _NATIVE_AVAILABLE:
        return _native.scan_secrets(text)

    # Pure Python fallback
    matches = []
    lines = text.split("\n")
    line_starts = [0]
    for line in lines[:-1]:
        line_starts.append(line_starts[-1] + len(line) + 1)

    for name, pattern, severity in _SECRET_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            # Find line number
            line_num = 1
            for i, start in enumerate(line_starts):
                if start > m.start():
                    break
                line_num = i + 1
            matches.append(
                SecretMatchFallback(name, m.group(), severity, m.start(), m.end(), line_num)
            )

    return matches


def has_secrets(text: str) -> bool:
    """Check if text contains secrets.

    Args:
        text: Text to check

    Returns:
        True if secrets are found
    """
    if _NATIVE_AVAILABLE:
        return _native.has_secrets(text)

    # Pure Python fallback
    for _, pattern, _ in _SECRET_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def get_secret_types(text: str) -> List[str]:
    """Get types of secrets found in text.

    Args:
        text: Text to scan

    Returns:
        List of secret type names
    """
    if _NATIVE_AVAILABLE:
        return _native.get_secret_types(text)

    # Pure Python fallback
    types = []
    for name, pattern, _ in _SECRET_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            types.append(name)
    return types


def mask_secrets(text: str, mask_char: str = "*", visible_chars: int = 4) -> str:
    """Mask secrets in text.

    Args:
        text: Text containing secrets
        mask_char: Character to use for masking
        visible_chars: Number of chars to keep visible at start/end

    Returns:
        Text with secrets masked
    """
    if _NATIVE_AVAILABLE:
        return _native.mask_secrets(text, mask_char, visible_chars)

    # Pure Python fallback
    result = text
    for _, pattern, _ in _SECRET_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            matched = m.group()
            if len(matched) > visible_chars * 2:
                mask_len = len(matched) - visible_chars * 2
                masked = matched[:visible_chars] + mask_char * mask_len + matched[-visible_chars:]
            else:
                masked = mask_char * len(matched)
            result = result.replace(matched, masked)
    return result


def list_secret_patterns() -> List[str]:
    """List available secret pattern names.

    Returns:
        List of pattern names
    """
    if _NATIVE_AVAILABLE:
        return _native.list_secret_patterns()

    # Pure Python fallback
    return [name for name, _, _ in _SECRET_PATTERNS]


def scan_secrets_summary(text: str) -> Dict[str, Any]:
    """Scan secrets and return summary.

    Args:
        text: Text to scan

    Returns:
        Summary dict with counts
    """
    if _NATIVE_AVAILABLE:
        return _native.scan_secrets_summary(text)

    # Pure Python fallback
    matches = scan_secrets(text)
    by_type: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}

    for m in matches:
        by_type[m.secret_type] = by_type.get(m.secret_type, 0) + 1
        by_severity[m.severity] = by_severity.get(m.severity, 0) + 1

    return {
        "total_matches": len(matches),
        "has_secrets": len(matches) > 0,
        "by_type": by_type,
        "by_severity": by_severity,
    }


# Re-export native class when available
if _NATIVE_AVAILABLE:
    SecretMatch = _native.SecretMatch
else:
    SecretMatch = SecretMatchFallback


# =============================================================================
# PATTERN MATCHING (NEW - HIGH impact, Aho-Corasick)
# =============================================================================


class PatternMatchFallback:
    """Fallback class for pattern match results."""

    def __init__(self, pattern_idx: int, matched_text: str, start: int, end: int):
        self.pattern_idx = pattern_idx
        self.matched_text = matched_text
        self.start = start
        self.end = end

    def __repr__(self) -> str:
        return f"PatternMatch(pattern={self.pattern_idx}, text='{self.matched_text}', span=({self.start}, {self.end}))"


class PatternMatcherFallback:
    """Fallback pattern matcher using Python regex."""

    def __init__(self, patterns: List[str], case_insensitive: bool = True):
        self.patterns = patterns
        self.case_insensitive = case_insensitive
        flags = re.IGNORECASE if case_insensitive else 0
        self._compiled = [(i, re.compile(re.escape(p), flags)) for i, p in enumerate(patterns)]

    def find_all(self, text: str) -> List[PatternMatchFallback]:
        matches = []
        for idx, pattern in self._compiled:
            for m in pattern.finditer(text):
                matches.append(PatternMatchFallback(idx, m.group(), m.start(), m.end()))
        matches.sort(key=lambda x: x.start)
        return matches

    def contains_any(self, text: str) -> bool:
        for _, pattern in self._compiled:
            if pattern.search(text):
                return True
        return False

    def count_matches(self, text: str) -> int:
        return len(self.find_all(text))

    def matched_patterns(self, text: str) -> List[int]:
        seen = set()
        for m in self.find_all(text):
            seen.add(m.pattern_idx)
        return sorted(seen)

    def matched_pattern_strings(self, text: str) -> List[str]:
        return [self.patterns[i] for i in self.matched_patterns(text)]

    def count_by_pattern(self, text: str) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for m in self.find_all(text):
            counts[m.pattern_idx] = counts.get(m.pattern_idx, 0) + 1
        return counts

    def get_pattern(self, idx: int) -> Optional[str]:
        return self.patterns[idx] if 0 <= idx < len(self.patterns) else None

    def pattern_count(self) -> int:
        return len(self.patterns)

    def replace_all(self, text: str, replacement: str) -> str:
        result = text
        for _, pattern in self._compiled:
            result = pattern.sub(replacement, result)
        return result


def contains_any_pattern(text: str, patterns: List[str], case_insensitive: bool = True) -> bool:
    """Check if text contains any pattern.

    Args:
        text: Text to search
        patterns: Patterns to match
        case_insensitive: Whether to ignore case

    Returns:
        True if any pattern matches
    """
    if _NATIVE_AVAILABLE:
        return _native.contains_any_pattern(text, patterns, case_insensitive)

    matcher = PatternMatcherFallback(patterns, case_insensitive)
    return matcher.contains_any(text)


def find_all_patterns(text: str, patterns: List[str], case_insensitive: bool = True) -> List[Any]:
    """Find all pattern matches in text.

    Args:
        text: Text to search
        patterns: Patterns to match
        case_insensitive: Whether to ignore case

    Returns:
        List of PatternMatch objects
    """
    if _NATIVE_AVAILABLE:
        return _native.find_all_patterns(text, patterns, case_insensitive)

    matcher = PatternMatcherFallback(patterns, case_insensitive)
    return matcher.find_all(text)


def count_pattern_matches(text: str, patterns: List[str], case_insensitive: bool = True) -> int:
    """Count pattern matches in text.

    Args:
        text: Text to search
        patterns: Patterns to match
        case_insensitive: Whether to ignore case

    Returns:
        Total match count
    """
    if _NATIVE_AVAILABLE:
        return _native.count_pattern_matches(text, patterns, case_insensitive)

    matcher = PatternMatcherFallback(patterns, case_insensitive)
    return matcher.count_matches(text)


def get_matched_pattern_indices(
    text: str, patterns: List[str], case_insensitive: bool = True
) -> List[int]:
    """Get indices of matched patterns.

    Args:
        text: Text to search
        patterns: Patterns to match
        case_insensitive: Whether to ignore case

    Returns:
        List of matched pattern indices
    """
    if _NATIVE_AVAILABLE:
        return _native.get_matched_pattern_indices(text, patterns, case_insensitive)

    matcher = PatternMatcherFallback(patterns, case_insensitive)
    return matcher.matched_patterns(text)


def batch_contains_any(
    texts: List[str], patterns: List[str], case_insensitive: bool = True
) -> List[bool]:
    """Check multiple texts for pattern matches.

    Args:
        texts: List of texts to search
        patterns: Patterns to match
        case_insensitive: Whether to ignore case

    Returns:
        List of booleans, one per text
    """
    if _NATIVE_AVAILABLE:
        return _native.batch_contains_any(texts, patterns, case_insensitive)

    matcher = PatternMatcherFallback(patterns, case_insensitive)
    return [matcher.contains_any(text) for text in texts]


def weighted_pattern_score(
    text: str, patterns: List[str], weights: List[float], case_insensitive: bool = True
) -> float:
    """Calculate weighted score for pattern matches.

    Args:
        text: Text to search
        patterns: Patterns to match
        weights: Weight for each pattern
        case_insensitive: Whether to ignore case

    Returns:
        Sum of weights for matched patterns
    """
    if _NATIVE_AVAILABLE:
        return _native.weighted_pattern_score(text, patterns, weights, case_insensitive)

    if len(patterns) != len(weights):
        raise ValueError(
            f"Pattern count ({len(patterns)}) must match weight count ({len(weights)})"
        )

    matcher = PatternMatcherFallback(patterns, case_insensitive)
    matched = matcher.matched_patterns(text)
    return sum(weights[i] for i in matched)


# Re-export native classes when available
if _NATIVE_AVAILABLE:
    PatternMatcher = _native.PatternMatcher
    PatternMatch = _native.PatternMatch
else:
    PatternMatcher = PatternMatcherFallback
    PatternMatch = PatternMatchFallback


# =============================================================================
# TOOL CALL EXTRACTION (NEW - HIGH impact)
# =============================================================================

# Pre-compiled patterns for Python fallback
_FILE_PATH_PATTERNS = [
    re.compile(
        r"(?:to|file|path|in|create|write|save|modify|update|edit)\s+"
        r"[`'\"]?([a-zA-Z0-9_./-]+\.[a-zA-Z]{1,10})[`'\"]?",
        re.IGNORECASE,
    ),
    re.compile(
        r"^[`'\"]?([a-zA-Z0-9_./-]+\.[a-zA-Z]{1,10})[`'\"]?\s+(?:with|should|will)",
        re.MULTILINE,
    ),
    re.compile(
        r"(?:the|this)\s+file\s+[`'\"]?([a-zA-Z0-9_./-]+\.[a-zA-Z]{1,10})[`'\"]?",
        re.IGNORECASE,
    ),
    re.compile(r"`([a-zA-Z0-9_./-]+\.[a-zA-Z]{1,10})`"),
]

_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:python|py|javascript|js|typescript|ts|bash|sh|json|yaml|yml|toml|"
    r"html|css|markdown|md|sql|go|rust|java|c|cpp|ruby|php)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

_INDENTED_CODE_PATTERN = re.compile(
    r"(?:^|\n)((?:[ ]{4,}|\t).+(?:\n(?:[ ]{4,}|\t).+)*)",
    re.MULTILINE,
)

_SHELL_COMMAND_PATTERNS = [
    re.compile(r"```(?:bash|sh|shell|zsh)?\s*\n(.+?)```", re.DOTALL | re.IGNORECASE),
    re.compile(r"(?:run|execute|command):\s*[`'\"](.+?)[`'\"]", re.IGNORECASE),
    re.compile(r"(?:^|\n)\$\s+(.+?)(?:\n|$)"),
]


def extract_file_path(text: str) -> Optional[str]:
    """Extract a file path from text.

    Uses native Rust implementation when available for ~3x speedup.

    Args:
        text: Text to search for file paths

    Returns:
        Extracted file path or None
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "extract_file_path"):
        return _native.extract_file_path(text)

    # Pure Python fallback
    for pattern in _FILE_PATH_PATTERNS:
        match = pattern.search(text)
        if match:
            path = match.group(1)
            if "/" in path or "." in path:
                return path
    return None


def extract_code_blocks(text: str) -> List[str]:
    """Extract code blocks from text (fenced and indented).

    Uses native Rust implementation when available for ~3x speedup.

    Args:
        text: Text containing code blocks

    Returns:
        List of extracted code block contents
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "extract_code_blocks"):
        return _native.extract_code_blocks(text)

    # Pure Python fallback
    blocks = []

    # Fenced code blocks
    for match in _CODE_BLOCK_PATTERN.finditer(text):
        blocks.append(match.group(1).strip())

    # Indented code blocks (if no fenced blocks found)
    if not blocks:
        for match in _INDENTED_CODE_PATTERN.finditer(text):
            block = match.group(1)
            lines = block.split("\n")
            non_empty = [line for line in lines if line.strip()]
            if non_empty:
                min_indent = min(len(line) - len(line.lstrip()) for line in non_empty)
                dedented = "\n".join(
                    line[min_indent:] if len(line) > min_indent else line for line in lines
                )
                blocks.append(dedented.strip())

    return blocks


def extract_shell_commands(text: str) -> List[str]:
    """Extract shell commands from text.

    Uses native Rust implementation when available for ~3x speedup.

    Args:
        text: Text containing shell commands

    Returns:
        List of extracted shell commands
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "extract_shell_commands"):
        return _native.extract_shell_commands(text)

    # Pure Python fallback
    commands = []
    for pattern in _SHELL_COMMAND_PATTERNS:
        for match in pattern.finditer(text):
            cmd = match.group(1).strip()
            if cmd:
                commands.append(cmd)
    return commands


@dataclass
class ExtractedToolCallResult:
    """Result of tool call extraction."""

    tool_name: str
    arguments: Dict[str, Any]
    confidence: float
    source_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "args": self.arguments,
            "confidence": self.confidence,
        }


def extract_tool_call(
    text: str,
    tool_name: str,
    current_file: Optional[str] = None,
) -> Optional[ExtractedToolCallResult]:
    """Extract a tool call from text for a specific tool.

    Uses native Rust implementation when available for ~3x speedup.

    Args:
        text: Text to extract from
        tool_name: Tool name to extract for
        current_file: Optional current file context

    Returns:
        ExtractedToolCallResult or None if extraction fails
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "extract_tool_call"):
        result = _native.extract_tool_call(text, tool_name, current_file)
        if result:
            return ExtractedToolCallResult(
                tool_name=result["tool"],
                arguments=result["args"],
                confidence=result["confidence"],
                source_text=result.get("source", text[:200]),
            )
        return None

    # Pure Python fallback
    tool_lower = tool_name.lower()

    if tool_lower in ("write", "write_file"):
        file_path = extract_file_path(text) or current_file
        if not file_path:
            return None
        blocks = extract_code_blocks(text)
        if not blocks:
            return None
        content = blocks[0]
        confidence = 0.85
        if file_path.endswith(".py") and ("def " in content or "class " in content):
            confidence = 0.95
        return ExtractedToolCallResult(
            tool_name="write",
            arguments={"path": file_path, "content": content},
            confidence=confidence,
            source_text=text[:200],
        )

    elif tool_lower in ("read", "read_file"):
        file_path = extract_file_path(text)
        if not file_path:
            return None
        return ExtractedToolCallResult(
            tool_name="read",
            arguments={"path": file_path},
            confidence=0.9,
            source_text=text[:100],
        )

    elif tool_lower in ("shell", "bash", "execute", "run"):
        commands = extract_shell_commands(text)
        if not commands:
            return None
        return ExtractedToolCallResult(
            tool_name="shell",
            arguments={"command": commands[0]},
            confidence=0.75,
            source_text=text[:150],
        )

    elif tool_lower in ("ls", "list"):
        # Try backtick-wrapped paths first
        backtick_match = re.search(r"`([a-zA-Z0-9_./-]+)`", text)
        path = backtick_match.group(1) if backtick_match else "."
        return ExtractedToolCallResult(
            tool_name="ls",
            arguments={"path": path},
            confidence=0.8,
            source_text=text[:100],
        )

    return None


def batch_extract_file_paths(texts: List[str]) -> List[Optional[str]]:
    """Extract file paths from multiple texts.

    Uses native Rust implementation when available for ~5x speedup.

    Args:
        texts: List of texts to search

    Returns:
        List of extracted paths (None for texts without paths)
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "batch_extract_file_paths"):
        return _native.batch_extract_file_paths(texts)

    return [extract_file_path(text) for text in texts]


# =============================================================================
# RESPONSE SANITIZATION (NEW - HIGH impact)
# =============================================================================

# Pre-compiled patterns for Python fallback
_LEAKAGE_PATTERNS = [
    re.compile(r"Do not invent any new or additional parameters.*", re.IGNORECASE),
    re.compile(r"The parameter value should be passed as a string.*", re.IGNORECASE),
    re.compile(r"If you want to call multiple functions.*", re.IGNORECASE),
    re.compile(r"Do NOT surround the function call.*", re.IGNORECASE),
    re.compile(r"All parameters are required unless.*", re.IGNORECASE),
    re.compile(r"The agent is not allowed to directly access.*", re.IGNORECASE),
    re.compile(r"Begin by calling list_directory.*", re.IGNORECASE),
]

_GARBAGE_PATTERNS = [
    re.compile(r"FUNCTION_CALL\s*\{"),
    re.compile(r"</function>\s*</function>"),
    re.compile(r"<parameter[^>]*>"),
    re.compile(r'^\s*\{\s*"name":\s*"[^"]+",\s*"arguments":', re.MULTILINE),
    re.compile(r"^\s*<IMPORTANT>", re.MULTILINE),
    re.compile(r"^\s*Do NOT", re.MULTILINE),
    re.compile(r"^\s*NEVER\s+", re.MULTILINE),
    re.compile(r"\[TOOL_REQUEST\]"),
]

_CLEANUP_PATTERNS = [
    (re.compile(r"(</\w+>\s*){3,}"), ""),  # Repeated closing tags
    (re.compile(r"</?function[^>]*>"), ""),  # Function tags
    (re.compile(r"</?parameter[^>]*>"), ""),  # Parameter tags
    (re.compile(r"</?tool[^>]*>"), ""),  # Tool tags
    (re.compile(r"</?IMPORTANT[^>]*>"), ""),  # Important tags
    (re.compile(r'\{"name":\s*"[^"]+",\s*"arguments":\s*\{[^}]*\}\}'), ""),  # JSON tool calls
    (re.compile(r"\n{4,}"), "\n\n\n"),  # Excessive newlines
]


def sanitize_response_fast(text: str) -> str:
    """Sanitize model response by removing malformed patterns.

    Uses native Rust implementation when available for ~3x speedup.

    Args:
        text: Raw response text from the model

    Returns:
        Cleaned text suitable for display
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "sanitize_response"):
        return _native.sanitize_response(text)

    if not text:
        return text

    # Apply cleanup patterns
    for pattern, replacement in _CLEANUP_PATTERNS:
        text = pattern.sub(replacement, text)

    # Strip thinking tokens
    text = strip_thinking_tokens(text)

    # Remove leakage patterns
    for pattern in _LEAKAGE_PATTERNS:
        text = pattern.sub("", text)

    # Remove lines that are just tool call syntax
    lines = text.split("\n")
    cleaned_lines = [
        line
        for line in lines
        if not (line.strip().startswith('{"name":') or line.strip().startswith("</"))
    ]
    text = "\n".join(cleaned_lines)

    return text.strip()


def is_garbage_content_fast(content: str) -> bool:
    """Detect if content is garbage/malformed output.

    Uses native Rust implementation when available for ~3x speedup.

    Args:
        content: Content to check

    Returns:
        True if content appears to be garbage
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "is_garbage_content"):
        return _native.is_garbage_content(content)

    if not content:
        return False

    for pattern in _GARBAGE_PATTERNS:
        if pattern.search(content):
            return True
    return False


def detect_leakage_patterns(text: str) -> List[Tuple[int, int, str]]:
    """Detect training data leakage patterns in text.

    Uses native Rust implementation when available for ~3x speedup.

    Args:
        text: Text to check for leakage

    Returns:
        List of (start, end, pattern_name) tuples for matches
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "detect_leakage_patterns"):
        return _native.detect_leakage_patterns(text)

    # Pure Python fallback
    matches = []
    pattern_names = [
        "no_new_params",
        "string_params",
        "multiple_funcs",
        "no_surround",
        "required_params",
        "no_direct_access",
        "begin_list_dir",
    ]

    for pattern, name in zip(_LEAKAGE_PATTERNS, pattern_names):
        for m in pattern.finditer(text):
            matches.append((m.start(), m.end(), name))

    matches.sort(key=lambda x: x[0])
    return matches


def strip_markup_fast(text: str) -> str:
    """Remove XML/HTML-like tags to salvage plain text.

    Uses native Rust implementation when available for ~3x speedup.

    Args:
        text: Text potentially containing markup

    Returns:
        Plain text with markup removed
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "strip_markup"):
        return _native.strip_markup(text)

    if not text:
        return text
    cleaned = re.sub(r"<[^>]+>", " ", text)
    return " ".join(cleaned.split())


def validate_tool_name(name: str) -> Tuple[bool, Optional[str]]:
    """Validate a tool name is not a hallucination.

    Uses native Rust implementation when available for ~3x speedup.

    Args:
        name: Tool name to validate

    Returns:
        Tuple of (is_valid, rejection_reason)
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "validate_tool_name"):
        return _native.validate_tool_name(name)

    if not name or not isinstance(name, str):
        return False, "empty_or_invalid_type"

    invalid_prefixes = [
        "example_",
        "func_",
        "function_",
        "tool_name",
        "my_",
        "test_tool",
        "sample_",
    ]
    for prefix in invalid_prefixes:
        if name.startswith(prefix):
            return False, f"invalid_prefix:{prefix}"

    if name.endswith("/") or name.endswith(">"):
        return False, "invalid_suffix"
    if name.startswith("<"):
        return False, "starts_with_tag"
    if " " in name or "\t" in name:
        return False, "contains_whitespace"
    if name[0].isdigit():
        return False, "starts_with_number"
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", name):
        return False, "invalid_characters"

    return True, None


# =============================================================================
# TYPE COERCION (v0.4.0 - Argument Normalizer Hot Path)
# =============================================================================


def coerce_string_type(value: str) -> Tuple[str, str, Optional[str]]:
    """Coerce a string to its appropriate type.

    Fast detection and coercion of string values to bool/int/float/null.
    Uses Rust implementation when available for ~3-5x speedup.

    Args:
        value: String value to coerce

    Returns:
        Tuple of (type_name, coerced_str, error_or_none)
        - type_name: "null", "bool", "int", "float", or "string"
        - coerced_str: The value as a string in canonical form
        - error_or_none: Error message if coercion failed, else None
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "coerce_string_type"):
        return _native.coerce_string_type(value)

    # Pure Python fallback
    # Check for null
    if value.lower() in ("none", "null", "nil"):
        return ("null", "null", None)

    # Check for bool
    if value.lower() in ("true", "yes", "on", "1"):
        return ("bool", "true", None)
    if value.lower() in ("false", "no", "off", "0"):
        return ("bool", "false", None)

    # Check for int
    try:
        int_val = int(value)
        return ("int", str(int_val), None)
    except ValueError:
        pass

    # Check for float
    try:
        float_val = float(value)
        return ("float", str(float_val), None)
    except ValueError:
        pass

    # Default to string
    return ("string", value, None)


# =============================================================================
# STDLIB DETECTION (v0.4.0 - Indexer Hot Path)
# =============================================================================

# Python stdlib modules for fallback lookup
_PYTHON_STDLIB_MODULES = frozenset({
    # Core builtins and language
    "abc", "asyncio", "builtins", "collections", "contextlib", "copy",
    "dataclasses", "enum", "functools", "gc", "inspect", "io", "itertools",
    "operator", "sys", "types", "typing", "typing_extensions", "weakref",
    # File/OS operations
    "os", "pathlib", "shutil", "stat", "tempfile", "glob", "fnmatch",
    # Data formats
    "json", "csv", "xml", "html", "pickle", "base64", "codecs", "struct",
    # Text processing
    "re", "string", "textwrap", "unicodedata", "difflib",
    # Date/Time
    "datetime", "time", "calendar", "zoneinfo",
    # Math/Numbers
    "math", "decimal", "fractions", "random", "statistics", "cmath",
    # Networking
    "socket", "ssl", "http", "urllib", "email", "ftplib", "smtplib",
    # Concurrent
    "threading", "multiprocessing", "concurrent", "queue", "subprocess",
    "signal", "selectors",
    # Testing/Debug
    "unittest", "doctest", "pdb", "traceback", "logging", "warnings",
    # Crypto/Hashing
    "hashlib", "hmac", "secrets",
    # Archive/Compression
    "zipfile", "tarfile", "gzip", "bz2", "lzma", "zlib",
    # Other common stdlib
    "argparse", "configparser", "getopt", "pprint", "shelve", "sqlite3",
    "atexit", "sched", "heapq", "bisect", "array", "cProfile", "profile",
    "timeit", "trace", "ast", "dis", "code", "codeop", "compileall",
    "py_compile", "importlib", "pkgutil", "modulefinder", "runpy",
    "venv", "site", "sysconfig", "platform", "ctypes", "mmap",
    "uuid", "ipaddress", "locale", "gettext",
})


def is_stdlib_module(module_name: str) -> bool:
    """Check if a module is part of Python's standard library.

    Uses Rust implementation with perfect hash when available for O(1) lookup.
    Falls back to Python frozenset lookup.

    Args:
        module_name: Full module name (e.g., "os.path", "collections.abc")

    Returns:
        True if the module is a stdlib module
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "is_stdlib_module"):
        return _native.is_stdlib_module(module_name)

    # Pure Python fallback - check top-level module
    top_level = module_name.split(".")[0]
    return top_level in _PYTHON_STDLIB_MODULES


# =============================================================================
# YAML PARSING (v0.4.0 - Workflow Acceleration)
# =============================================================================


def parse_yaml(yaml_content: str) -> Any:
    """Parse YAML string to Python object.

    Uses Rust serde_yaml for ~5-20x speedup on large workflow files.
    Falls back to PyYAML's safe_load.

    Args:
        yaml_content: Raw YAML string to parse

    Returns:
        Parsed Python object (dict, list, or scalar)
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "parse_yaml"):
        return _native.parse_yaml(yaml_content)

    import yaml

    return yaml.safe_load(yaml_content)


def parse_yaml_with_env(yaml_content: str) -> Any:
    """Parse YAML with environment variable interpolation.

    Supports:
    - $env.VAR_NAME - Simple env var reference
    - ${VAR_NAME:-default} - Shell-style with optional default

    Uses Rust implementation when available for ~5-20x speedup.

    Args:
        yaml_content: Raw YAML string to parse

    Returns:
        Parsed Python object with env vars interpolated
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "parse_yaml_with_env"):
        return _native.parse_yaml_with_env(yaml_content)

    # Python fallback with env var interpolation
    import os
    import yaml

    def interpolate_env_vars(value: Any) -> Any:
        if isinstance(value, str):
            # Handle $env.VAR_NAME
            result = re.sub(
                r"\$env\.([A-Za-z_][A-Za-z0-9_]*)",
                lambda m: os.environ.get(m.group(1), f"$env.{m.group(1)}"),
                value,
            )
            # Handle ${VAR:-default}
            result = re.sub(
                r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}",
                lambda m: os.environ.get(m.group(1), m.group(2) or ""),
                result,
            )
            return result
        elif isinstance(value, dict):
            return {k: interpolate_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [interpolate_env_vars(item) for item in value]
        return value

    data = yaml.safe_load(yaml_content)
    return interpolate_env_vars(data)


def parse_yaml_file(file_path: str) -> Any:
    """Parse YAML file directly.

    Args:
        file_path: Path to YAML file

    Returns:
        Parsed Python object
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "parse_yaml_file"):
        return _native.parse_yaml_file(file_path)

    import yaml

    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def parse_yaml_file_with_env(file_path: str) -> Any:
    """Parse YAML file with environment variable interpolation.

    Args:
        file_path: Path to YAML file

    Returns:
        Parsed Python object with env vars interpolated
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "parse_yaml_file_with_env"):
        return _native.parse_yaml_file_with_env(file_path)

    with open(file_path, "r") as f:
        return parse_yaml_with_env(f.read())


def validate_yaml(yaml_content: str) -> bool:
    """Validate YAML syntax without full parsing.

    Args:
        yaml_content: Raw YAML string

    Returns:
        True if valid, False if invalid
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "validate_yaml"):
        return _native.validate_yaml(yaml_content)

    import yaml

    try:
        yaml.safe_load(yaml_content)
        return True
    except yaml.YAMLError:
        return False


def extract_workflow_names(yaml_content: str) -> List[str]:
    """Extract workflow names from YAML content.

    Fast scan to find workflow names without full parsing.

    Args:
        yaml_content: Raw YAML string

    Returns:
        List of workflow names found
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "extract_workflow_names"):
        return _native.extract_workflow_names(yaml_content)

    import yaml

    data = yaml.safe_load(yaml_content)
    if not isinstance(data, dict):
        return []

    names = []
    workflows = data.get("workflows", data)
    for key, val in workflows.items():
        if isinstance(val, dict) and "nodes" in val:
            names.append(key)

    return names


# =============================================================================
# ACCELERATOR PRIORITY SYSTEM (Benchmark-Based)
# =============================================================================
# Based on measured performance, some operations are faster in Rust (text
# processing, chunking) while others are faster in Python (NumPy+BLAS for
# similarity, frozenset for stdlib lookup). This system wires the optimal
# backend by default, with override capability via configuration.
#
# Benchmark Results (Apple M-series, 2025-01):
# | Operation          | Rust   | Python | Winner   | Speedup |
# |--------------------|--------|--------|----------|---------|
# | normalize_block    | 0.21ms | 1.37ms | RUST     | 6.6x    |
# | chunk_with_overlap | 0.03ms | 0.14ms | RUST     | 5.0x    |
# | content_hashing    | 0.71ms | 1.76ms | RUST     | 2.5x    |
# | count_lines        | 0.004ms| 0.009ms| RUST     | 2.4x    |
# | type_coercion      | 0.19ms | 0.40ms | RUST     | 2.3x    |
# | stdlib_detection   | 0.12ms | 0.10ms | PYTHON   | 0.9x    |
# | json_repair        | 0.06ms | 0.04ms | PYTHON   | 0.7x    |
# | batch_similarity   | 0.18ms | 0.03ms | PYTHON   | 0.1x    |

from enum import Enum


class AcceleratorPreference(str, Enum):
    """Backend preference for native accelerators."""

    RUST = "rust"  # Force Rust (fail if unavailable)
    PYTHON = "python"  # Force Python
    AUTO = "auto"  # Use benchmark-based default


@dataclass(frozen=True)
class AcceleratorBenchmark:
    """Benchmark data for an accelerator operation."""

    name: str
    rust_ms: float
    python_ms: float
    preferred: str  # "rust" or "python"
    notes: str = ""

    @property
    def speedup(self) -> float:
        """Speedup ratio (Python time / Rust time)."""
        return self.python_ms / self.rust_ms if self.rust_ms > 0 else 0.0


# Benchmark-based defaults (measured on Apple M-series, 2025-01)
ACCELERATOR_BENCHMARKS: Dict[str, AcceleratorBenchmark] = {
    # Text processing - Rust wins
    "normalize_block": AcceleratorBenchmark(
        "normalize_block", 0.21, 1.37, "rust", "Whitespace/punctuation normalization"
    ),
    "chunk_with_overlap": AcceleratorBenchmark(
        "chunk_with_overlap", 0.03, 0.14, "rust", "Line-aware text chunking"
    ),
    "content_hashing": AcceleratorBenchmark(
        "content_hashing", 0.71, 1.76, "rust", "Hash with normalization (SHA-256 dominates)"
    ),
    "count_lines": AcceleratorBenchmark(
        "count_lines", 0.004, 0.009, "rust", "SIMD-optimized line counting"
    ),
    "type_coercion": AcceleratorBenchmark(
        "type_coercion", 0.19, 0.40, "rust", "String to bool/int/float coercion"
    ),
    # Python wins - NumPy/BLAS or simple operations
    "stdlib_detection": AcceleratorBenchmark(
        "stdlib_detection", 0.12, 0.10, "python", "frozenset O(1) lookup is optimal"
    ),
    "json_repair": AcceleratorBenchmark(
        "json_repair", 0.06, 0.04, "python", "Simple string ops faster for small inputs"
    ),
    "batch_similarity": AcceleratorBenchmark(
        "batch_similarity", 0.18, 0.03, "python", "NumPy+BLAS has hardware SIMD"
    ),
    "similarity_matrix": AcceleratorBenchmark(
        "similarity_matrix", 0.20, 0.05, "python", "NumPy matmul is highly optimized"
    ),
}

# User-configurable overrides (can be set at runtime)
_accelerator_overrides: Dict[str, str] = {}


def set_accelerator_preference(operation: str, preference: str) -> None:
    """Override the default backend for an operation.

    Args:
        operation: Operation name (e.g., "normalize_block", "batch_similarity")
        preference: "rust", "python", or "auto" (reset to benchmark default)

    Example:
        # Force Python for all similarity operations
        set_accelerator_preference("batch_similarity", "python")

        # Reset to benchmark-based default
        set_accelerator_preference("batch_similarity", "auto")
    """
    if preference == "auto":
        _accelerator_overrides.pop(operation, None)
    else:
        _accelerator_overrides[operation] = preference


def get_preferred_backend(operation: str) -> str:
    """Get the optimal backend for an operation.

    Returns "rust" or "python" based on:
    1. User override (if set via set_accelerator_preference)
    2. Benchmark data (if available)
    3. Default to "rust" if native available, else "python"

    Args:
        operation: Operation name

    Returns:
        "rust" or "python"
    """
    # Check user override first
    if operation in _accelerator_overrides:
        return _accelerator_overrides[operation]

    # Check benchmark data
    if operation in ACCELERATOR_BENCHMARKS:
        benchmark = ACCELERATOR_BENCHMARKS[operation]
        preferred = benchmark.preferred
        # Only use Rust if it's available
        if preferred == "rust" and not _NATIVE_AVAILABLE:
            return "python"
        return preferred

    # Default: use Rust if available
    return "rust" if _NATIVE_AVAILABLE else "python"


def get_all_benchmarks() -> Dict[str, Dict[str, Any]]:
    """Get all benchmark data for display/debugging.

    Returns:
        Dict mapping operation names to benchmark info
    """
    return {
        name: {
            "rust_ms": b.rust_ms,
            "python_ms": b.python_ms,
            "speedup": f"{b.speedup:.1f}x",
            "preferred": b.preferred,
            "override": _accelerator_overrides.get(name),
            "effective": get_preferred_backend(name),
            "notes": b.notes,
        }
        for name, b in ACCELERATOR_BENCHMARKS.items()
    }


# =============================================================================
# PROTOCOL-BASED DISPATCH (SOLID Design)
# =============================================================================
# These factories provide protocol-compliant implementations with automatic
# fallback from Rust to Python. See victor/native/protocols.py for the
# protocol definitions and victor/native/python/ for Python fallbacks.


def get_symbol_extractor(backend: Optional[str] = None) -> "SymbolExtractorProtocol":
    """Get a symbol extractor implementation.

    Args:
        backend: Explicit backend choice ("rust" or "python").
                 If None, uses Rust if available, otherwise Python.

    Returns:
        SymbolExtractorProtocol implementation

    Example:
        extractor = get_symbol_extractor()
        symbols = extractor.extract_functions(source, "python")
    """
    from victor.native.protocols import SymbolExtractorProtocol

    if backend == "rust" or (backend is None and _NATIVE_AVAILABLE):
        # TODO: Return Rust implementation when available
        # For now, fall through to Python
        pass

    from victor.native.python.symbol_extractor import PythonSymbolExtractor

    return PythonSymbolExtractor()


def get_argument_normalizer(backend: Optional[str] = None) -> "ArgumentNormalizerProtocol":
    """Get an argument normalizer implementation.

    Args:
        backend: Explicit backend choice ("rust" or "python").
                 If None, uses Rust if available, otherwise Python.

    Returns:
        ArgumentNormalizerProtocol implementation

    Example:
        normalizer = get_argument_normalizer()
        result, success = normalizer.normalize_json(json_str)
    """
    from victor.native.protocols import ArgumentNormalizerProtocol

    if backend == "rust" or (backend is None and _NATIVE_AVAILABLE):
        # TODO: Return Rust implementation when available
        # For now, fall through to Python
        pass

    from victor.native.python.arg_normalizer import PythonArgumentNormalizer

    return PythonArgumentNormalizer()


def get_similarity_computer(backend: Optional[str] = None) -> "SimilarityComputerProtocol":
    """Get a similarity computer implementation.

    Note: Benchmark data shows NumPy+BLAS (Python) is ~6x faster than Rust FFI
    for batch similarity operations. Default uses Python unless explicitly
    overridden or set via set_accelerator_preference("batch_similarity", "rust").

    Args:
        backend: Explicit backend choice ("rust" or "python").
                 If None, uses benchmark-based preference (Python for similarity).

    Returns:
        SimilarityComputerProtocol implementation

    Example:
        computer = get_similarity_computer()
        sim = computer.cosine(vec_a, vec_b)
    """
    from victor.native.protocols import SimilarityComputerProtocol

    effective_backend = backend or get_preferred_backend("batch_similarity")

    if effective_backend == "rust" and _NATIVE_AVAILABLE:
        # Rust implementation exists but is slower than NumPy for similarity
        # Only use if explicitly requested
        try:
            from victor.native.rust.similarity import RustSimilarityComputer

            return RustSimilarityComputer()
        except ImportError:
            pass

    from victor.native.python.similarity import PythonSimilarityComputer

    return PythonSimilarityComputer()


def get_text_chunker(backend: Optional[str] = None) -> "TextChunkerProtocol":
    """Get a text chunker implementation.

    Note: Benchmark data shows Rust is ~5x faster for text chunking.
    Default uses Rust when available.

    Args:
        backend: Explicit backend choice ("rust" or "python").
                 If None, uses benchmark-based preference (Rust for chunking).

    Returns:
        TextChunkerProtocol implementation

    Example:
        chunker = get_text_chunker()
        chunks = chunker.chunk_with_overlap(text, chunk_size=100, overlap=20)
    """
    from victor.native.protocols import TextChunkerProtocol

    effective_backend = backend or get_preferred_backend("chunk_with_overlap")

    if effective_backend == "rust" and _NATIVE_AVAILABLE:
        try:
            from victor.native.rust.chunker import RustTextChunker

            return RustTextChunker()
        except ImportError:
            # Fall through to Python if Rust wrapper not available
            pass

    from victor.native.python.chunker import PythonTextChunker

    return PythonTextChunker()


def get_ast_indexer(backend: Optional[str] = None) -> "AstIndexerProtocol":
    """Get an AST indexer implementation.

    The AST indexer accelerates hot paths in codebase indexing:
    - is_stdlib_module(): O(1) stdlib lookup with perfect hash
    - extract_identifiers(): SIMD-optimized regex extraction

    Args:
        backend: Explicit backend choice ("rust" or "python").
                 If None, uses Rust if available, otherwise Python.

    Returns:
        AstIndexerProtocol implementation

    Example:
        indexer = get_ast_indexer()
        is_stdlib = indexer.is_stdlib_module("os.path")
        identifiers = indexer.extract_identifiers(source_code)
    """
    from victor.native.protocols import AstIndexerProtocol

    if backend == "rust" or (backend is None and _NATIVE_AVAILABLE):
        try:
            from victor.native.rust.ast_indexer import RustAstIndexer

            return RustAstIndexer()
        except ImportError:
            # Fall through to Python if Rust wrapper not available
            pass

    from victor.native.python.ast_indexer import PythonAstIndexer

    return PythonAstIndexer()


# Convenience singletons for common use cases
_symbol_extractor_instance: Optional["SymbolExtractorProtocol"] = None
_argument_normalizer_instance: Optional["ArgumentNormalizerProtocol"] = None
_similarity_computer_instance: Optional["SimilarityComputerProtocol"] = None
_text_chunker_instance: Optional["TextChunkerProtocol"] = None
_ast_indexer_instance: Optional["AstIndexerProtocol"] = None


def get_default_symbol_extractor() -> "SymbolExtractorProtocol":
    """Get the default symbol extractor singleton.

    Uses lazy initialization and returns a cached instance for
    efficient repeated access.
    """
    global _symbol_extractor_instance
    if _symbol_extractor_instance is None:
        _symbol_extractor_instance = get_symbol_extractor()
    return _symbol_extractor_instance


def get_default_argument_normalizer() -> "ArgumentNormalizerProtocol":
    """Get the default argument normalizer singleton.

    Uses lazy initialization and returns a cached instance for
    efficient repeated access.
    """
    global _argument_normalizer_instance
    if _argument_normalizer_instance is None:
        _argument_normalizer_instance = get_argument_normalizer()
    return _argument_normalizer_instance


def get_default_similarity_computer() -> "SimilarityComputerProtocol":
    """Get the default similarity computer singleton.

    Uses lazy initialization and returns a cached instance for
    efficient repeated access.
    """
    global _similarity_computer_instance
    if _similarity_computer_instance is None:
        _similarity_computer_instance = get_similarity_computer()
    return _similarity_computer_instance


def get_default_text_chunker() -> "TextChunkerProtocol":
    """Get the default text chunker singleton.

    Uses lazy initialization and returns a cached instance for
    efficient repeated access.
    """
    global _text_chunker_instance
    if _text_chunker_instance is None:
        _text_chunker_instance = get_text_chunker()
    return _text_chunker_instance


def get_default_ast_indexer() -> "AstIndexerProtocol":
    """Get the default AST indexer singleton.

    Uses lazy initialization and returns a cached instance for
    efficient repeated access. This is the recommended way to access
    the AST indexer for hot path operations during codebase indexing.
    """
    global _ast_indexer_instance
    if _ast_indexer_instance is None:
        _ast_indexer_instance = get_ast_indexer()
    return _ast_indexer_instance


def get_content_hasher(
    normalize_whitespace: bool = True,
    case_insensitive: bool = False,
    hash_length: int = 16,
    remove_punctuation: bool = False,
) -> "ContentHasherProtocol":
    """Get a content hasher implementation with configured normalization.

    The ContentHasher automatically uses Rust native normalize_block()
    when available for 10-50x faster normalization.

    Args:
        normalize_whitespace: Collapse multiple whitespace to single space
        case_insensitive: Convert to lowercase before hashing
        hash_length: Number of hex chars to return (1-64)
        remove_punctuation: Remove trailing punctuation

    Returns:
        ContentHasherProtocol implementation

    Example:
        hasher = get_content_hasher(normalize_whitespace=True, case_insensitive=True)
        hash1 = hasher.hash("Hello  World")
        hash2 = hasher.hash("hello world")
        assert hash1 == hash2  # Same due to normalization
    """
    from victor.core.utils.content_hasher import ContentHasher

    return ContentHasher(
        normalize_whitespace=normalize_whitespace,
        case_insensitive=case_insensitive,
        hash_length=hash_length,
        remove_punctuation=remove_punctuation,
    )


# Content hasher preset singletons
_content_hasher_fuzzy: Optional["ContentHasherProtocol"] = None
_content_hasher_exact: Optional["ContentHasherProtocol"] = None


def get_default_content_hasher_fuzzy() -> "ContentHasherProtocol":
    """Get the default fuzzy content hasher singleton.

    Preconfigured for text deduplication with whitespace normalization,
    case insensitivity, and punctuation removal.

    Uses lazy initialization and returns a cached instance.
    """
    global _content_hasher_fuzzy
    if _content_hasher_fuzzy is None:
        from victor.core.utils.content_hasher import HasherPresets

        _content_hasher_fuzzy = HasherPresets.text_fuzzy()
    return _content_hasher_fuzzy


def get_default_content_hasher_exact() -> "ContentHasherProtocol":
    """Get the default exact content hasher singleton.

    Preconfigured for exact matching (no normalization).
    Suitable for tool call deduplication and API signature matching.

    Uses lazy initialization and returns a cached instance.
    """
    global _content_hasher_exact
    if _content_hasher_exact is None:
        from victor.core.utils.content_hasher import HasherPresets

        _content_hasher_exact = HasherPresets.exact_match()
    return _content_hasher_exact


def reset_protocol_singletons() -> None:
    """Reset all protocol singletons.

    Useful for testing to ensure clean state between tests.
    """
    global _symbol_extractor_instance, _argument_normalizer_instance
    global _similarity_computer_instance, _text_chunker_instance
    global _ast_indexer_instance, _content_hasher_fuzzy, _content_hasher_exact
    _symbol_extractor_instance = None
    _argument_normalizer_instance = None
    _similarity_computer_instance = None
    _text_chunker_instance = None
    _ast_indexer_instance = None
    _content_hasher_fuzzy = None
    _content_hasher_exact = None


# =============================================================================
# EXPORTS
# =============================================================================

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
    # Document chunking (NEW - HIGH impact)
    "chunk_by_sentences",
    "chunk_by_chars",
    "chunk_by_paragraphs",
    "detect_doc_type",
    "count_tokens_approx",
    # Secret detection (NEW - HIGH impact)
    "SecretMatch",
    "scan_secrets",
    "has_secrets",
    "get_secret_types",
    "mask_secrets",
    "list_secret_patterns",
    "scan_secrets_summary",
    # Pattern matching (NEW - HIGH impact, Aho-Corasick)
    "PatternMatcher",
    "PatternMatch",
    "contains_any_pattern",
    "find_all_patterns",
    "count_pattern_matches",
    "get_matched_pattern_indices",
    "batch_contains_any",
    "weighted_pattern_score",
    # Tool call extraction (NEW - HIGH impact)
    "extract_file_path",
    "extract_code_blocks",
    "extract_shell_commands",
    "extract_tool_call",
    "batch_extract_file_paths",
    "ExtractedToolCallResult",
    # Response sanitization (NEW - HIGH impact)
    "sanitize_response_fast",
    "is_garbage_content_fast",
    "detect_leakage_patterns",
    "strip_markup_fast",
    "validate_tool_name",
    # Type coercion (v0.4.0 - Argument Normalizer Hot Path)
    "coerce_string_type",
    # Stdlib detection (v0.4.0 - Indexer Hot Path)
    "is_stdlib_module",
    # YAML parsing (v0.4.0 - Workflow Acceleration)
    "parse_yaml",
    "parse_yaml_with_env",
    "parse_yaml_file",
    "parse_yaml_file_with_env",
    "validate_yaml",
    "extract_workflow_names",
    # Protocol-based dispatch (SOLID Design)
    "get_symbol_extractor",
    "get_argument_normalizer",
    "get_similarity_computer",
    "get_text_chunker",
    "get_ast_indexer",
    "get_default_symbol_extractor",
    "get_default_argument_normalizer",
    "get_default_similarity_computer",
    "get_default_text_chunker",
    "get_default_ast_indexer",
    "reset_protocol_singletons",
    # Content hashing (cross-vertical, uses native normalize_block)
    "get_content_hasher",
    "get_default_content_hasher_fuzzy",
    "get_default_content_hasher_exact",
    # Accelerator priority system (benchmark-based backend selection)
    "AcceleratorPreference",
    "AcceleratorBenchmark",
    "ACCELERATOR_BENCHMARKS",
    "set_accelerator_preference",
    "get_preferred_backend",
    "get_all_benchmarks",
]
