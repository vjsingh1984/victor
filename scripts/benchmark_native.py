#!/usr/bin/env python3
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

"""Benchmark script for native extensions.

Compares performance of Python fallback vs Rust native implementations.

Usage:
    python scripts/benchmark_native.py
"""

import random
import string
import time
from typing import Callable, Any


def timed(func: Callable, *args, iterations: int = 100, **kwargs) -> tuple[float, Any]:
    """Run function multiple times and return average time and result."""
    # Warm up
    result = func(*args, **kwargs)

    start = time.perf_counter()
    for _ in range(iterations):
        result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start

    return elapsed / iterations * 1000, result  # ms


def generate_content(num_blocks: int = 50, block_size: int = 200) -> str:
    """Generate test content with paragraphs."""
    blocks = []
    for _ in range(num_blocks):
        words = [
            "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
            for _ in range(block_size // 5)
        ]
        blocks.append(" ".join(words))
    return "\n\n".join(blocks)


def generate_embeddings(num_vectors: int, dim: int = 384) -> list[list[float]]:
    """Generate random embedding vectors."""
    return [[random.gauss(0, 1) for _ in range(dim)] for _ in range(num_vectors)]


def benchmark_deduplication():
    """Benchmark deduplication functions."""
    print("\n" + "=" * 60)
    print("DEDUPLICATION BENCHMARK")
    print("=" * 60)

    from victor.processing.native import (
        rolling_hash_blocks,
        normalize_block,
        find_duplicate_blocks,
        is_native_available,
    )

    print(f"Native available: {is_native_available()}")

    # Generate test content
    content_small = generate_content(10, 100)
    content_medium = generate_content(50, 200)
    content_large = generate_content(200, 300)

    print(f"\nTest data sizes:")
    print(f"  Small:  {len(content_small):,} chars, ~10 blocks")
    print(f"  Medium: {len(content_medium):,} chars, ~50 blocks")
    print(f"  Large:  {len(content_large):,} chars, ~200 blocks")

    # Benchmark normalize_block
    print("\nnormalize_block():")
    block = "  Hello   World!  This is a TEST block.  "
    time_ms, _ = timed(normalize_block, block, iterations=10000)
    print(f"  Single block: {time_ms:.4f} ms")

    # Benchmark rolling_hash_blocks
    print("\nrolling_hash_blocks():")
    time_ms, result = timed(rolling_hash_blocks, content_small, 50, iterations=100)
    print(f"  Small content:  {time_ms:.3f} ms ({len(result)} blocks)")

    time_ms, result = timed(rolling_hash_blocks, content_medium, 50, iterations=100)
    print(f"  Medium content: {time_ms:.3f} ms ({len(result)} blocks)")

    time_ms, result = timed(rolling_hash_blocks, content_large, 50, iterations=50)
    print(f"  Large content:  {time_ms:.3f} ms ({len(result)} blocks)")

    # Benchmark find_duplicate_blocks
    print("\nfind_duplicate_blocks():")
    # Create content with duplicates
    content_with_dupes = content_medium + "\n\n" + content_medium[:5000]
    time_ms, dupes = timed(find_duplicate_blocks, content_with_dupes, 50, iterations=100)
    print(f"  With duplicates: {time_ms:.3f} ms ({len(dupes)} duplicates found)")


def benchmark_similarity():
    """Benchmark similarity functions."""
    print("\n" + "=" * 60)
    print("SIMILARITY BENCHMARK")
    print("=" * 60)

    from victor.processing.native import (
        cosine_similarity,
        batch_cosine_similarity,
        top_k_similar,
        is_native_available,
    )

    print(f"Native available: {is_native_available()}")

    # Generate test vectors
    query = generate_embeddings(1, 384)[0]
    corpus_small = generate_embeddings(10, 384)
    corpus_medium = generate_embeddings(100, 384)
    corpus_large = generate_embeddings(1000, 384)

    print(f"\nTest data sizes:")
    print(f"  Query:  384 dimensions")
    print(f"  Small:  10 vectors")
    print(f"  Medium: 100 vectors")
    print(f"  Large:  1000 vectors")

    # Benchmark cosine_similarity
    print("\ncosine_similarity():")
    time_ms, _ = timed(cosine_similarity, query, corpus_small[0], iterations=10000)
    print(f"  Single pair: {time_ms:.4f} ms")

    # Benchmark batch_cosine_similarity
    print("\nbatch_cosine_similarity():")
    time_ms, result = timed(batch_cosine_similarity, query, corpus_small, iterations=1000)
    print(f"  10 vectors:   {time_ms:.4f} ms")

    time_ms, result = timed(batch_cosine_similarity, query, corpus_medium, iterations=500)
    print(f"  100 vectors:  {time_ms:.4f} ms")

    time_ms, result = timed(batch_cosine_similarity, query, corpus_large, iterations=100)
    print(f"  1000 vectors: {time_ms:.3f} ms")

    # Benchmark top_k_similar
    print("\ntop_k_similar() (k=10):")
    time_ms, result = timed(top_k_similar, query, corpus_medium, 10, iterations=500)
    print(f"  100 vectors:  {time_ms:.4f} ms")

    time_ms, result = timed(top_k_similar, query, corpus_large, 10, iterations=100)
    print(f"  1000 vectors: {time_ms:.3f} ms")


def benchmark_json_repair():
    """Benchmark JSON repair functions."""
    print("\n" + "=" * 60)
    print("JSON REPAIR BENCHMARK")
    print("=" * 60)

    from victor.processing.native import (
        repair_json,
        extract_json_objects,
        is_native_available,
    )

    print(f"Native available: {is_native_available()}")

    # Test data
    simple_python = "{'key': 'value', 'active': True}"
    complex_python = "{'outer': {'inner': [{'a': 1}, {'b': 2}], 'flag': False}, 'list': ['x', 'y', 'z'], 'null_val': None}"
    valid_json = '{"key": "value", "active": true}'

    text_with_json = f"""
    Here is some text before the JSON.

    Result: {simple_python}

    And some text after.

    Another result: {complex_python}

    End of document.
    """

    print(f"\nTest data:")
    print(f"  Simple Python dict:  {len(simple_python)} chars")
    print(f"  Complex Python dict: {len(complex_python)} chars")
    print(f"  Text with JSON:      {len(text_with_json)} chars")

    # Benchmark repair_json
    print("\nrepair_json():")
    time_ms, _ = timed(repair_json, simple_python, iterations=10000)
    print(f"  Simple:  {time_ms:.4f} ms")

    time_ms, _ = timed(repair_json, complex_python, iterations=10000)
    print(f"  Complex: {time_ms:.4f} ms")

    time_ms, _ = timed(repair_json, valid_json, iterations=10000)
    print(f"  Valid (passthrough): {time_ms:.4f} ms")

    # Benchmark extract_json_objects
    print("\nextract_json_objects():")
    time_ms, result = timed(extract_json_objects, text_with_json, iterations=1000)
    print(f"  Text with 2 objects: {time_ms:.4f} ms ({len(result)} found)")


def benchmark_hashing():
    """Benchmark hashing functions."""
    print("\n" + "=" * 60)
    print("HASHING BENCHMARK")
    print("=" * 60)

    from victor.processing.native import (
        compute_signature,
        compute_batch_signatures,
        signature_similarity,
        is_native_available,
    )

    print(f"Native available: {is_native_available()}")

    # Test data
    simple_args = {"path": "/test/file.py"}
    complex_args = {
        "path": "/test/file.py",
        "offset": 100,
        "limit": 500,
        "encoding": "utf-8",
    }

    tool_calls = [("read_file", {"path": f"/file_{i}.py"}) for i in range(100)]

    print(f"\nTest data:")
    print(f"  Simple args:  {len(simple_args)} keys")
    print(f"  Complex args: {len(complex_args)} keys")
    print(f"  Batch:        {len(tool_calls)} tool calls")

    # Benchmark compute_signature
    print("\ncompute_signature():")
    time_ms, _ = timed(compute_signature, "read_file", simple_args, iterations=10000)
    print(f"  Simple args:  {time_ms:.4f} ms")

    time_ms, _ = timed(compute_signature, "read_file", complex_args, iterations=10000)
    print(f"  Complex args: {time_ms:.4f} ms")

    # Benchmark compute_batch_signatures
    print("\ncompute_batch_signatures():")
    time_ms, result = timed(compute_batch_signatures, tool_calls, iterations=100)
    print(f"  100 tool calls: {time_ms:.3f} ms")

    # Benchmark signature_similarity
    print("\nsignature_similarity():")
    sig1 = "1234567890abcdef"
    sig2 = "1234567890fedcba"
    time_ms, _ = timed(signature_similarity, sig1, sig2, iterations=100000)
    print(f"  Single pair: {time_ms:.5f} ms")


def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("VICTOR NATIVE EXTENSIONS BENCHMARK")
    print("=" * 60)

    from victor.processing.native import is_native_available, get_native_version

    print(f"\nNative extensions available: {is_native_available()}")
    if is_native_available():
        print(f"Native version: {get_native_version()}")
    else:
        print("Using Python fallback implementations")

    benchmark_deduplication()
    benchmark_similarity()
    benchmark_json_repair()
    benchmark_hashing()

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
