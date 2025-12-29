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
    from victor.native import (
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
    # Streaming filter (NEW)
    "StreamingFilter",
    "StreamingChunkResult",
    "strip_thinking_tokens",
    "contains_thinking_tokens",
    "find_thinking_tokens",
    "extract_thinking_content",
    # Task classifier (NEW)
    "NativeTaskClassifier",
    "NativeClassificationResult",
    "NativeTaskType",
    "classify_task_native",
    "has_action_keywords",
    "has_analysis_keywords",
    "has_generation_keywords",
    "has_negation",
    "find_all_keywords",
    # Thinking detector (NEW)
    "ThinkingDetector",
    "PatternAnalysis",
    "detect_circular_phrases",
    "count_circular_patterns",
    "find_circular_patterns",
]
