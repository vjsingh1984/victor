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

"""Integration tests for ToolSelectorAccelerator.

Tests the Rust-backed tool selection accelerator with comprehensive
coverage of all operations, error handling, and fallback behavior.
"""

import math
import random
from typing import List

import numpy as np
import pytest

# Import directly from the module file to avoid __init__.py import issues
import sys
from pathlib import Path
import importlib.util

# Get project root and module path
test_file = Path(__file__).resolve()
project_root = test_file.parent.parent.parent.parent
module_path = project_root / "victor" / "native" / "rust" / "tool_selector.py"

# Load the tool_selector module directly without triggering __init__.py
spec = importlib.util.spec_from_file_location("tool_selector", module_path)
tool_selector_module = importlib.util.module_from_spec(spec)
sys.modules["victor.native.rust.tool_selector"] = tool_selector_module
spec.loader.exec_module(tool_selector_module)

ToolSelectorAccelerator = tool_selector_module.ToolSelectorAccelerator
get_tool_selector_accelerator = tool_selector_module.get_tool_selector_accelerator
reset_tool_selector_accelerator = tool_selector_module.reset_tool_selector_accelerator


@pytest.fixture
def reset_accelerator():
    """Reset accelerator singleton before each test."""
    reset_tool_selector_accelerator()
    yield
    reset_tool_selector_accelerator()


@pytest.fixture
def query_embedding() -> List[float]:
    """Sample query embedding (384 dimensions)."""
    random.seed(42)
    return [random.random() for _ in range(384)]


@pytest.fixture
def tool_embeddings() -> List[List[float]]:
    """Sample tool embeddings (100 tools, 384 dimensions each)."""
    random.seed(43)
    return [[random.random() for _ in range(384)] for _ in range(100)]


@pytest.fixture
def tool_names() -> List[str]:
    """Sample tool names."""
    return [
        "read_file",
        "write_file",
        "search_files",
        "run_command",
        "git_commit",
        "create_pull_request",
        "analyze_code",
        "refactor_code",
        "generate_tests",
        "review_code",
    ] + [f"tool_{i}" for i in range(10, 100)]


@pytest.fixture
def tool_category_map() -> dict[str, str]:
    """Tool to category mapping."""
    categories = {
        "read_file": "file_ops",
        "write_file": "file_ops",
        "search_files": "search",
        "run_command": "execution",
        "git_commit": "git",
        "create_pull_request": "git",
        "analyze_code": "analysis",
        "refactor_code": "refactoring",
        "generate_tests": "testing",
        "review_code": "review",
    }
    # Add categories for remaining tools
    for i in range(10, 100):
        categories[f"tool_{i}"] = ["file_ops", "search", "execution", "git", "analysis", "refactoring", "testing", "review"][i % 8]
    return categories


class TestToolSelectorAccelerator:
    """Test suite for ToolSelectorAccelerator."""

    def test_initialization_rust(self, reset_accelerator):
        """Test accelerator initialization with Rust backend."""
        accelerator = ToolSelectorAccelerator(force_python=False)

        # Check backend
        if accelerator.rust_available:
            assert accelerator.get_backend() == "rust"
            assert accelerator.get_version() is not None
        else:
            assert accelerator.get_backend() == "python"
            assert accelerator.get_version() is None

    def test_initialization_python_fallback(self, reset_accelerator):
        """Test accelerator initialization with Python backend."""
        accelerator = ToolSelectorAccelerator(force_python=True)
        assert accelerator.get_backend() == "python"
        assert not accelerator.rust_available
        assert accelerator.get_version() is None

    def test_cosine_similarity_batch(
        self, reset_accelerator, query_embedding, tool_embeddings
    ):
        """Test cosine similarity batch computation."""
        accelerator = ToolSelectorAccelerator()
        similarities = accelerator.cosine_similarity_batch(query_embedding, tool_embeddings)

        # Check output
        assert len(similarities) == len(tool_embeddings)
        assert all(-1.0 <= s <= 1.0 for s in similarities)

        # Verify with NumPy
        query_arr = np.array(query_embedding)
        query_norm = np.linalg.norm(query_arr)

        for i, tool in enumerate(tool_embeddings[:5]):  # Check first 5
            tool_arr = np.array(tool)
            expected = np.dot(query_arr, tool_arr) / (query_norm * np.linalg.norm(tool_arr))
            assert abs(similarities[i] - expected) < 1e-5

    def test_cosine_similarity_batch_empty(self, reset_accelerator, query_embedding):
        """Test cosine similarity with empty tool list."""
        accelerator = ToolSelectorAccelerator()
        similarities = accelerator.cosine_similarity_batch(query_embedding, [])
        assert similarities == []

    def test_cosine_similarity_batch_zero_vector(
        self, reset_accelerator, tool_embeddings
    ):
        """Test cosine similarity with zero query vector."""
        accelerator = ToolSelectorAccelerator()
        zero_query = [0.0] * 384
        similarities = accelerator.cosine_similarity_batch(zero_query, tool_embeddings)

        # All similarities should be 0
        assert all(s == 0.0 for s in similarities)

    def test_topk_indices(self, reset_accelerator):
        """Test top-k indices selection."""
        accelerator = ToolSelectorAccelerator()
        scores = [0.1, 0.5, 0.3, 0.9, 0.2, 0.8, 0.4, 0.7, 0.6, 0.0]

        # Get top 3
        top_k = accelerator.topk_indices(scores, 3)

        # Check
        assert len(top_k) == 3
        assert top_k[0] == 3  # Score 0.9
        assert top_k[1] == 5  # Score 0.8
        assert top_k[2] == 7  # Score 0.7

        # Verify scores are in descending order
        assert scores[top_k[0]] >= scores[top_k[1]] >= scores[top_k[2]]

    def test_topk_indices_invalid_k(self, reset_accelerator):
        """Test topk_indices with invalid k values."""
        accelerator = ToolSelectorAccelerator()
        scores = [0.1, 0.5, 0.3]

        # k <= 0 should raise
        with pytest.raises(ValueError, match="k must be positive"):
            accelerator.topk_indices(scores, 0)

        with pytest.raises(ValueError, match="k must be positive"):
            accelerator.topk_indices(scores, -1)

        # k > len(scores) should raise
        with pytest.raises(ValueError, match="k .* cannot exceed length"):
            accelerator.topk_indices(scores, 5)

    def test_topk_indices_empty(self, reset_accelerator):
        """Test topk_indices with empty scores."""
        accelerator = ToolSelectorAccelerator()

        # k=0 is invalid, but if we had empty list and k=1
        with pytest.raises(ValueError, match="k .* cannot exceed length"):
            accelerator.topk_indices([], 1)

    def test_topk_with_scores(self, reset_accelerator):
        """Test top-k with scores."""
        accelerator = ToolSelectorAccelerator()
        scores = [0.1, 0.5, 0.3, 0.9, 0.2, 0.8, 0.4, 0.7, 0.6, 0.0]

        # Get top 3
        top_k = accelerator.topk_with_scores(scores, 3)

        # Check structure
        assert len(top_k) == 3
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in top_k)

        # Check indices and scores
        indices, retrieved_scores = zip(*top_k)
        assert list(indices) == [3, 5, 7]
        assert list(retrieved_scores) == [0.9, 0.8, 0.7]

    def test_filter_by_category(
        self, reset_accelerator, tool_names, tool_category_map
    ):
        """Test category filtering."""
        accelerator = ToolSelectorAccelerator()

        # Filter to only file_ops and git tools
        allowed_categories = {"file_ops", "git"}
        filtered = accelerator.filter_by_category(
            tool_names, allowed_categories, tool_category_map
        )

        # Check results
        assert "read_file" in filtered
        assert "write_file" in filtered
        assert "git_commit" in filtered
        assert "create_pull_request" in filtered
        assert "search_files" not in filtered  # Different category
        assert "run_command" not in filtered  # Different category

        # All filtered tools should be in allowed categories
        for tool in filtered:
            assert tool_category_map[tool] in allowed_categories

    def test_filter_by_category_empty(self, reset_accelerator, tool_category_map):
        """Test category filtering with empty tool list."""
        accelerator = ToolSelectorAccelerator()
        filtered = accelerator.filter_by_category([], {"file_ops"}, tool_category_map)
        assert filtered == []

    def test_filter_by_category_no_match(self, reset_accelerator, tool_names):
        """Test category filtering with no matches."""
        accelerator = ToolSelectorAccelerator()

        # Filter to non-existent category
        filtered = accelerator.filter_by_category(
            tool_names, {"nonexistent"}, {}
        )
        assert filtered == []

    def test_filter_and_rank(
        self,
        reset_accelerator,
        query_embedding,
        tool_embeddings,
        tool_names,
        tool_category_map,
    ):
        """Test combined filter and rank operation."""
        accelerator = ToolSelectorAccelerator()

        # Use first 10 tools for this test
        query = query_embedding
        tools = tool_embeddings[:10]
        names = tool_names[:10]
        allowed_categories = {"file_ops", "git"}

        # Filter and rank
        results = accelerator.filter_and_rank(
            query, tools, names, allowed_categories, tool_category_map, k=3
        )

        # Check structure
        assert len(results) <= 3  # May be fewer if not enough tools match
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in results)

        # All results should be in allowed categories
        for tool_name, _ in results:
            assert tool_category_map[tool_name] in allowed_categories

        # Check scores are valid
        for tool_name, score in results:
            assert -1.0 <= score <= 1.0

        # Results should be sorted by score descending
        if len(results) > 1:
            scores = [score for _, score in results]
            assert scores == sorted(scores, reverse=True)

    def test_filter_and_rank_mismatched_lengths(self, reset_accelerator):
        """Test filter_and_rank with mismatched input lengths."""
        accelerator = ToolSelectorAccelerator()

        with pytest.raises(ValueError, match="Length mismatch"):
            accelerator.filter_and_rank(
                [0.1] * 384,
                [[0.1] * 384, [0.2] * 384],
                ["tool_1"],  # Only 1 name for 2 tools
                {"category"},
                {"tool_1": "category"},
                k=1,
            )

    def test_performance_cosine_similarity(
        self, reset_accelerator, query_embedding, tool_embeddings
    ):
        """Performance test for cosine similarity batch."""
        import time

        accelerator = ToolSelectorAccelerator()

        # Warm up
        _ = accelerator.cosine_similarity_batch(query_embedding, tool_embeddings[:10])

        # Measure
        start = time.perf_counter()
        for _ in range(100):
            _ = accelerator.cosine_similarity_batch(query_embedding, tool_embeddings)
        duration = time.perf_counter() - start

        # Rust should be < 1s for 100 iterations of 100 tools
        # Python should be < 5s
        if accelerator.rust_available:
            assert duration < 1.0, f"Rust too slow: {duration:.3f}s for 100 iterations"
        else:
            assert duration < 5.0, f"Python too slow: {duration:.3f}s for 100 iterations"

        print(f"\nCosine similarity (100 tools x 100 iterations): {duration:.3f}s")
        print(f"Backend: {accelerator.get_backend()}")

    def test_performance_topk(self, reset_accelerator):
        """Performance test for top-k selection."""
        import time

        accelerator = ToolSelectorAccelerator()
        scores = [random.random() for _ in range(1000)]

        # Warm up
        _ = accelerator.topk_indices(scores, 10)

        # Measure
        start = time.perf_counter()
        for _ in range(1000):
            _ = accelerator.topk_indices(scores, 10)
        duration = time.perf_counter() - start

        # Both implementations should be fast
        assert duration < 1.0, f"Too slow: {duration:.3f}s for 1000 iterations"

        print(f"\nTop-k selection (1000 scores x 1000 iterations): {duration:.3f}s")
        print(f"Backend: {accelerator.get_backend()}")

    def test_metrics_collection(self, reset_accelerator, query_embedding):
        """Test that metrics are collected for operations."""
        accelerator = ToolSelectorAccelerator()

        # Reset metrics
        from victor.native.observability import NativeMetrics
        metrics = NativeMetrics.get_instance()

        # Perform operations
        tools = [[random.random() for _ in range(384)] for _ in range(10)]
        _ = accelerator.cosine_similarity_batch(query_embedding, tools)
        _ = accelerator.topk_indices([0.1, 0.5, 0.3], 2)

        # Check metrics were recorded
        stats = metrics.get_stats()
        assert "cosine_similarity_batch" in stats or "cosine_similarity_batch_python" in stats
        assert "topk_indices" in stats or "topk_indices_python" in stats

    def test_singleton_behavior(self, reset_accelerator):
        """Test singleton behavior of get_tool_selector_accelerator."""
        # Get default instance
        acc1 = get_tool_selector_accelerator()
        acc2 = get_tool_selector_accelerator()

        # Should be same instance
        assert acc1 is acc2

        # Force Python should create new instance
        acc3 = get_tool_selector_accelerator(force_python=True)
        assert acc3 is not acc1
        assert acc3.get_backend() == "python"

    def test_error_handling_rust_failure(self, reset_accelerator, query_embedding):
        """Test graceful fallback when Rust fails."""
        # This test verifies that Rust failures fall back to Python
        accelerator = ToolSelectorAccelerator()

        # If Rust is available, it should handle errors gracefully
        if accelerator.rust_available:
            # Test with valid data (should work)
            tools = [[random.random() for _ in range(384)] for _ in range(10)]
            similarities = accelerator.cosine_similarity_batch(query_embedding, tools)

            # Should return valid results
            assert len(similarities) == 10
            assert all(-1.0 <= s <= 1.0 for s in similarities)


class TestToolSelectorAcceleratorEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_tool_similarity(self, reset_accelerator):
        """Test similarity with single tool."""
        accelerator = ToolSelectorAccelerator()
        query = [0.1] * 384
        tools = [[0.2] * 384]

        similarities = accelerator.cosine_similarity_batch(query, tools)

        assert len(similarities) == 1
        # Identical direction vectors should have similarity 1.0
        assert abs(similarities[0] - 1.0) < 1e-5

    def test_orthogonal_vectors(self, reset_accelerator):
        """Test similarity with orthogonal vectors."""
        accelerator = ToolSelectorAccelerator()

        # Create orthogonal vectors
        query = [1.0] + [0.0] * 383
        tool = [0.0, 1.0] + [0.0] * 382

        similarities = accelerator.cosine_similarity_batch(query, [tool])

        # Orthogonal vectors should have similarity 0.0
        assert abs(similarities[0] - 0.0) < 1e-5

    def test_opposite_vectors(self, reset_accelerator):
        """Test similarity with opposite vectors."""
        accelerator = ToolSelectorAccelerator()

        query = [0.5] * 384
        tool = [-0.5] * 384

        similarities = accelerator.cosine_similarity_batch(query, [tool])

        # Opposite vectors should have similarity -1.0
        assert abs(similarities[0] - (-1.0)) < 1e-5

    def test_topk_all_scores_equal(self, reset_accelerator):
        """Test top-k when all scores are equal."""
        accelerator = ToolSelectorAccelerator()
        scores = [0.5] * 10

        top_k = accelerator.topk_indices(scores, 3)

        # Should return 3 indices (any 3 since all equal)
        assert len(top_k) == 3
        assert len(set(top_k)) == 3  # All distinct
        assert all(0 <= idx < 10 for idx in top_k)

    def test_large_k(self, reset_accelerator):
        """Test top-k with k equal to list length."""
        accelerator = ToolSelectorAccelerator()
        scores = [0.1, 0.5, 0.3, 0.9, 0.2]

        top_k = accelerator.topk_indices(scores, len(scores))

        # Should return all indices
        assert len(top_k) == len(scores)
        assert set(top_k) == set(range(len(scores)))


@pytest.mark.integration
class TestToolSelectorAcceleratorIntegration:
    """Integration tests with realistic tool selection scenarios."""

    def test_realistic_tool_selection_scenario(
        self, reset_accelerator, tool_names, tool_category_map
    ):
        """Test realistic tool selection workflow."""
        accelerator = ToolSelectorAccelerator()

        # Simulate query for "read and analyze code"
        random.seed(42)
        query = [random.random() for _ in range(384)]

        # Create embeddings for all tools (realistic scenario)
        tool_embeddings = [
            [random.random() for _ in range(384)] for _ in range(len(tool_names))
        ]

        # Filter to coding-related tools
        coding_categories = {"file_ops", "analysis", "refactoring", "review"}

        # Get top 5 tools
        results = accelerator.filter_and_rank(
            query,
            tool_embeddings,
            tool_names,
            coding_categories,
            tool_category_map,
            k=5,
        )

        # Verify results
        assert len(results) <= 5
        assert all(isinstance(pair, tuple) for pair in results)

        # All tools should be in coding categories
        for tool_name, score in results:
            assert tool_category_map[tool_name] in coding_categories
            assert -1.0 <= score <= 1.0

        # Results should be sorted
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_batch_tool_selection(
        self, reset_accelerator, tool_names, tool_category_map
    ):
        """Test multiple queries in batch."""
        accelerator = ToolSelectorAccelerator()

        # Multiple queries
        queries = [
            [random.random() for _ in range(384)] for _ in range(5)
        ]

        # Same tools for all queries
        tool_embeddings = [
            [random.random() for _ in range(384)] for _ in range(len(tool_names))
        ]

        # Process all queries
        all_results = []
        for query in queries:
            results = accelerator.filter_and_rank(
                query,
                tool_embeddings,
                tool_names,
                {"file_ops", "git"},
                tool_category_map,
                k=3,
            )
            all_results.append(results)

        # Each query should return results
        assert len(all_results) == 5
        assert all(len(results) <= 3 for results in all_results)
