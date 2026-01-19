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

"""Integration tests for tool pipeline decision caching.

Tests the integration of ToolExecutionDecisionCache with ToolPipeline
to ensure hot path optimization works correctly.
"""
import asyncio
import pytest

from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig
from victor.agent.argument_normalizer import ArgumentNormalizer
from victor.agent.tool_executor import ToolExecutor
from victor.tools.base import ToolRegistry


class MockToolExecutor(ToolExecutor):
    """Mock tool executor for testing."""

    def __init__(self):
        self.call_count = 0
        self.calls = []

    async def execute(self, tool_name: str, arguments: dict, context: dict = None):
        self.call_count += 1
        self.calls.append((tool_name, arguments))
        return type("Result", (), {"success": True, "result": f"Executed {tool_name}", "error": None})


class TestToolPipelineDecisionCache:
    """Integration tests for decision cache in ToolPipeline."""

    @pytest.fixture
    def tool_registry(self):
        """Create a test tool registry."""
        registry = ToolRegistry()
        # In a real test, you'd register actual tools here
        return registry

    @pytest.fixture
    def tool_executor(self):
        """Create a mock tool executor."""
        return MockToolExecutor()

    @pytest.fixture
    def normalizer(self):
        """Create an argument normalizer."""
        return ArgumentNormalizer()

    @pytest.fixture
    def pipeline(self, tool_registry, tool_executor, normalizer):
        """Create a tool pipeline with decision cache."""
        config = ToolPipelineConfig(
            tool_budget=100,
            enable_caching=True,
            enable_analytics=True,
        )

        return ToolPipeline(
            tool_registry=tool_registry,
            tool_executor=tool_executor,
            config=config,
            argument_normalizer=normalizer,
        )

    def test_decision_cache_initialized(self, pipeline):
        """Test that decision cache is initialized in pipeline."""
        assert pipeline._decision_cache is not None
        assert pipeline._decision_cache._max_size == 1000

    def test_get_cache_stats_includes_decision_cache(self, pipeline):
        """Test that get_cache_stats includes decision cache stats."""
        stats = pipeline.get_cache_stats()
        assert "decision_cache" in stats
        assert "hits" in stats["decision_cache"]
        assert "misses" in stats["decision_cache"]
        assert "hit_rate" in stats["decision_cache"]
        assert "validation_cache_size" in stats["decision_cache"]
        assert "normalization_cache_size" in stats["decision_cache"]

    def test_reset_clears_decision_cache(self, pipeline):
        """Test that reset() clears the decision cache."""
        # Add some entries to the cache
        registry = pipeline.tools
        registry.is_tool_enabled = lambda x: True  # Mock
        registry.has_tool = lambda x: True

        # Use the cache
        pipeline._decision_cache.is_valid_tool("test_tool", registry)

        # Verify cache has entries
        assert pipeline._decision_cache.get_stats()["validation_cache_size"] > 0

        # Reset pipeline
        pipeline.reset()

        # Verify cache is cleared
        assert pipeline._decision_cache.get_stats()["validation_cache_size"] == 0
        assert pipeline._decision_cache.get_stats()["hits"] == 0
        assert pipeline._decision_cache.get_stats()["misses"] == 0

    @pytest.mark.asyncio
    async def test_execute_single_call_uses_cache(self, pipeline):
        """Test that _execute_single_call uses the decision cache."""
        # Mock the tool registry to return a valid tool
        pipeline.tools.is_tool_enabled = lambda x: True
        pipeline.tools.has_tool = lambda x: True

        # Create a tool call
        tool_call = {"name": "test_tool", "arguments": {"path": "/test/file.txt"}}

        # Execute the same call twice
        result1 = await pipeline._execute_single_call(tool_call, {})
        result2 = await pipeline._execute_single_call(tool_call, {})

        # Check that cache was used
        stats = pipeline._decision_cache.get_stats()
        assert stats["validation_cache_size"] > 0
        assert stats["normalization_cache_size"] > 0

    def test_cache_hit_rate_increases_with_repeated_calls(self, pipeline):
        """Test that cache hit rate improves with repeated calls."""
        # Mock the tool registry
        pipeline.tools.is_tool_enabled = lambda x: True
        pipeline.tools.has_tool = lambda x: True

        # Make multiple calls with the same tool
        for _ in range(10):
            pipeline._decision_cache.is_valid_tool("test_tool", pipeline.tools)

        stats = pipeline._decision_cache.get_stats()
        # After 10 calls, should have 1 miss + 9 hits = 90% hit rate
        assert stats["hit_rate"] == 0.9

    def test_normalization_cache_with_different_args(self, pipeline):
        """Test normalization cache with different arguments."""
        # Make calls with different args
        args1 = {"path": "/test1"}
        args2 = {"path": "/test2"}

        pipeline._decision_cache.get_normalized_args("test_tool", args1, pipeline.normalizer)
        pipeline._decision_cache.get_normalized_args("test_tool", args2, pipeline.normalizer)

        stats = pipeline._decision_cache.get_stats()
        # Both should be misses (different args)
        assert stats["misses"] >= 2
        assert stats["normalization_cache_size"] >= 2

    def test_cache_eviction(self, pipeline):
        """Test that cache evicts entries when full."""
        # Create a small cache
        from victor.agent.tool_execution_cache import ToolExecutionDecisionCache

        pipeline._decision_cache = ToolExecutionDecisionCache(max_size=2)

        pipeline.tools.is_tool_enabled = lambda x: True
        pipeline.tools.has_tool = lambda x: True

        # Add 3 tools (should evict first)
        pipeline._decision_cache.is_valid_tool("tool1", pipeline.tools)
        pipeline._decision_cache.is_valid_tool("tool2", pipeline.tools)
        pipeline._decision_cache.is_valid_tool("tool3", pipeline.tools)

        stats = pipeline._decision_cache.get_stats()
        assert stats["validation_cache_size"] == 2

    @pytest.mark.asyncio
    async def test_cache_performance_improvement(self, pipeline):
        """Benchmark: cached execution should show performance improvement."""
        # Mock the tool registry
        pipeline.tools.is_tool_enabled = lambda x: True
        pipeline.tools.has_tool = lambda x: True

        import time

        # Measure execution with cache
        tool_calls = [
            {"name": "test_tool", "arguments": {"path": "/test/file1.txt"}},
            {"name": "test_tool", "arguments": {"path": "/test/file2.txt"}},
        ] * 10  # 20 total calls, many duplicates

        start = time.time()
        for call in tool_calls:
            await pipeline._execute_single_call(call, {})
        cached_duration = time.time() - start

        # Get cache stats
        stats = pipeline._decision_cache.get_stats()

        # Should have some cache hits
        assert stats["hits"] > 0
        assert stats["hit_rate"] > 0.0

        # Performance note: In a real benchmark, we'd compare with uncached
        # For now, just verify cache is being used
        assert cached_duration < 1.0  # Should complete quickly


class TestToolPipelineCacheCorrectness:
    """Tests to ensure caching doesn't break correctness."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline for correctness testing."""
        from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig
        from victor.agent.argument_normalizer import ArgumentNormalizer

        registry = ToolRegistry()
        executor = MockToolExecutor()
        normalizer = ArgumentNormalizer()

        config = ToolPipelineConfig(
            tool_budget=100,
            enable_caching=True,
        )

        return ToolPipeline(
            tool_registry=registry,
            tool_executor=executor,
            config=config,
            argument_normalizer=normalizer,
        )

    def test_validation_result_correctness(self, pipeline):
        """Test that cached validation results are correct."""
        # Mock registry
        pipeline.tools.is_tool_enabled = lambda x: x == "valid_tool"
        pipeline.tools.has_tool = lambda x: x == "valid_tool"

        # Query valid tool
        result1 = pipeline._decision_cache.is_valid_tool("valid_tool", pipeline.tools)
        assert result1.is_valid is True

        # Query again (cache hit)
        result2 = pipeline._decision_cache.is_valid_tool("valid_tool", pipeline.tools)
        assert result2.is_valid is True

        # Query invalid tool
        result3 = pipeline._decision_cache.is_valid_tool("invalid_tool", pipeline.tools)
        assert result3.is_valid is False

    def test_normalization_result_correctness(self, pipeline):
        """Test that cached normalization results are correct."""
        args = {"path": "/test", "pattern": "*.py"}

        result1 = pipeline._decision_cache.get_normalized_args(
            "test_tool", args, pipeline.normalizer
        )
        assert result1.normalized_args == args

        # Query again (cache hit)
        result2 = pipeline._decision_cache.get_normalized_args(
            "test_tool", args, pipeline.normalizer
        )
        assert result2.normalized_args == args

    def test_signature_correctness(self, pipeline):
        """Test that signatures are computed correctly."""
        args = {"path": "/test", "pattern": "*.py"}

        result = pipeline._decision_cache.get_normalized_args(
            "test_tool", args, pipeline.normalizer
        )

        # Signature should contain tool name and args
        assert "test_tool" in result.signature
        assert "/test" in result.signature
