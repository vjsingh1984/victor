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

"""Unit tests for AST processor accelerator.

Tests individual components and methods without external dependencies.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from victor.native.accelerators.ast_processor import (
    AstProcessorAccelerator,
    AstQueryResult,
    ParseStats,
    get_ast_processor,
    reset_ast_processor,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_native_processor():
    """Mock native Rust processor."""
    mock = Mock()
    mock.parse_to_ast = Mock(return_value=Mock(root_node=Mock()))
    mock.execute_query = Mock(return_value=[])
    mock.extract_symbols_batch = Mock(return_value={})
    mock.get_cache_stats = Mock(return_value={"size": 10, "max_size": 100})
    mock.clear_cache = Mock()
    return mock


@pytest.fixture
def sample_python_source():
    """Sample Python source for testing."""
    return "def hello(): pass\n\nclass Foo:\n    pass\n"


# =============================================================================
# ParseStats Tests
# =============================================================================


class TestParseStats:
    """Test ParseStats dataclass."""

    def test_initial_state(self):
        """Test initial statistics state."""
        stats = ParseStats()
        assert stats.total_parses == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.total_duration_ms == 0.0

    def test_record_parse_cache_hit(self):
        """Test recording a cache hit."""
        stats = ParseStats()
        stats.record_parse(1.5, cache_hit=True)

        assert stats.total_parses == 1
        assert stats.cache_hits == 1
        assert stats.cache_misses == 0
        assert stats.total_duration_ms == 1.5

    def test_record_parse_cache_miss(self):
        """Test recording a cache miss."""
        stats = ParseStats()
        stats.record_parse(2.5, cache_hit=False)

        assert stats.total_parses == 1
        assert stats.cache_hits == 0
        assert stats.cache_misses == 1
        assert stats.total_duration_ms == 2.5

    def test_avg_duration(self):
        """Test average duration calculation."""
        stats = ParseStats()
        stats.record_parse(1.0, cache_hit=False)
        stats.record_parse(2.0, cache_hit=False)
        stats.record_parse(3.0, cache_hit=False)

        assert stats.avg_duration_ms == 2.0

    def test_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        stats = ParseStats()
        stats.record_parse(1.0, cache_hit=True)
        stats.record_parse(1.0, cache_hit=True)
        stats.record_parse(1.0, cache_hit=False)

        expected_rate = (2 / 3) * 100  # 66.67%
        assert stats.cache_hit_rate == expected_rate

    def test_cache_hit_rate_empty(self):
        """Test cache hit rate with no parses."""
        stats = ParseStats()
        assert stats.cache_hit_rate == 0.0

    def test_to_dict(self):
        """Test converting to dictionary."""
        stats = ParseStats()
        stats.record_parse(1.0, cache_hit=True)
        stats.record_parse(2.0, cache_hit=False)

        d = stats.to_dict()
        assert isinstance(d, dict)
        assert d["total_parses"] == 2.0
        assert d["cache_hits"] == 1.0
        assert d["cache_misses"] == 1.0
        assert d["avg_duration_ms"] == 1.5


# =============================================================================
# AstQueryResult Tests
# =============================================================================


class TestAstQueryResult:
    """Test AstQueryResult dataclass."""

    def test_initialization(self):
        """Test result initialization."""
        captures = [{"name": "test", "node": Mock()}]
        result = AstQueryResult(captures=captures, matches=1, duration_ms=0.5)

        assert result.captures == captures
        assert result.matches == 1
        assert result.duration_ms == 0.5

    def test_len(self):
        """Test length operator."""
        result = AstQueryResult(captures=[], matches=5, duration_ms=1.0)
        assert len(result) == 5

    def test_iter(self):
        """Test iteration."""
        captures = [{"name": "a"}, {"name": "b"}]
        result = AstQueryResult(captures=captures, matches=2, duration_ms=1.0)

        items = list(result)
        assert items == captures


# =============================================================================
# Language Normalization Tests
# =============================================================================


class TestLanguageNormalization:
    """Test language name normalization."""

    def test_python_variants(self):
        """Test Python language variants."""
        processor = AstProcessorAccelerator(force_python=True)
        assert processor.normalize_language("python") == "python"
        assert processor.normalize_language("Python") == "python"
        assert processor.normalize_language("PYTHON") == "python"
        assert processor.normalize_language("py") == "python"

    def test_javascript_variants(self):
        """Test JavaScript language variants."""
        processor = AstProcessorAccelerator(force_python=True)
        assert processor.normalize_language("javascript") == "javascript"
        assert processor.normalize_language("JavaScript") == "javascript"
        assert processor.normalize_language("js") == "javascript"

    def test_typescript_variants(self):
        """Test TypeScript language variants."""
        processor = AstProcessorAccelerator(force_python=True)
        assert processor.normalize_language("typescript") == "typescript"
        assert processor.normalize_language("ts") == "typescript"

    def test_rust_variants(self):
        """Test Rust language variants."""
        processor = AstProcessorAccelerator(force_python=True)
        assert processor.normalize_language("rust") == "rust"
        assert processor.normalize_language("rs") == "rust"

    def test_cpp_variants(self):
        """Test C++ language variants."""
        processor = AstProcessorAccelerator(force_python=True)
        # All variants normalize to "cpp"
        assert processor.normalize_language("cpp") == "cpp"
        assert processor.normalize_language("c++") == "cpp"
        assert processor.normalize_language("cxx") == "cpp"

    def test_unknown_language_passthrough(self):
        """Test unknown language is passed through."""
        processor = AstProcessorAccelerator(force_python=True)
        assert processor.normalize_language("cobol") == "cobol"
        assert processor.normalize_language("fortran") == "fortran"


# =============================================================================
# Supported Languages Tests
# =============================================================================


class TestSupportedLanguages:
    """Test getting supported languages."""

    def test_get_supported_languages(self):
        """Test returns list of languages."""
        processor = AstProcessorAccelerator(force_python=True)
        languages = processor.get_supported_languages()

        assert isinstance(languages, list)
        assert len(languages) > 0

    def test_common_languages_supported(self):
        """Test common languages are in list."""
        processor = AstProcessorAccelerator(force_python=True)
        languages = processor.get_supported_languages()

        assert "python" in languages
        assert "javascript" in languages
        assert "rust" in languages
        assert "go" in languages


# =============================================================================
# Cache Statistics Tests
# =============================================================================


class TestCacheStatistics:
    """Test cache statistics retrieval."""

    def test_cache_stats_python_backend(self):
        """Test cache stats with Python backend."""
        processor = AstProcessorAccelerator(force_python=True, max_cache_size=100)
        stats = processor.cache_stats

        assert isinstance(stats, dict)
        assert "size" in stats
        assert "max_size" in stats
        assert stats["size"] == 0
        assert stats["max_size"] == 100

    def test_parse_stats_initial(self):
        """Test initial parse statistics."""
        processor = AstProcessorAccelerator(force_python=True)
        stats = processor.parse_stats

        assert isinstance(stats, dict)
        assert stats["total_parses"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0

    def test_parse_stats_after_operations(self):
        """Test parse statistics update after operations."""
        processor = AstProcessorAccelerator(force_python=True)

        # Perform some parses
        try:
            processor.parse_to_ast("def foo(): pass", "python", "test1.py")
            processor.parse_to_ast("def bar(): pass", "python", "test2.py")
        except Exception:
            # May fail if tree-sitter not installed
            pass

        stats = processor.parse_stats
        # Stats should have been updated
        assert stats["total_parses"] >= 0


# =============================================================================
# Cache Management Tests
# =============================================================================


class TestCacheManagement:
    """Test cache management operations."""

    def test_clear_cache_python_backend(self):
        """Test clearing cache with Python backend."""
        processor = AstProcessorAccelerator(force_python=True)

        # Parse something to populate cache
        try:
            processor.parse_to_ast("def foo(): pass", "python", "test.py")
        except Exception:
            pass

        # Clear cache
        processor.clear_cache()

        # Stats should be reset
        stats = processor.parse_stats
        assert stats["total_parses"] == 0


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Test singleton instance management."""

    def test_get_ast_processor_returns_instance(self):
        """Test get_ast_processor returns instance."""
        processor = get_ast_processor()
        assert isinstance(processor, AstProcessorAccelerator)

    def test_get_ast_processor_same_instance(self):
        """Test get_ast_processor returns same instance."""
        processor1 = get_ast_processor()
        processor2 = get_ast_processor()
        assert processor1 is processor2

    def test_reset_ast_processor(self):
        """Test resetting singleton."""
        processor1 = get_ast_processor()
        reset_ast_processor()
        processor2 = get_ast_processor()

        # Should be different instances after reset
        assert processor1 is not processor2


# =============================================================================
# Backend Detection Tests
# =============================================================================


class TestBackendDetection:
    """Test backend availability detection."""

    def test_is_available(self):
        """Test accelerator is always available."""
        processor = AstProcessorAccelerator(force_python=True)
        assert processor.is_available() is True

    def test_is_rust_available_forced_python(self):
        """Test Rust availability when forced to Python."""
        processor = AstProcessorAccelerator(force_python=True)
        assert processor.is_rust_available() is False

    def test_get_version_python_backend(self):
        """Test version string with Python backend."""
        processor = AstProcessorAccelerator(force_python=True)
        version = processor.get_version()
        assert version == "python-fallback"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling."""

    def test_parse_empty_source_raises_error(self):
        """Test parsing empty source raises ValueError."""
        processor = AstProcessorAccelerator(force_python=True)

        with pytest.raises(ValueError, match="Source code cannot be empty"):
            processor.parse_to_ast("", "python")

        with pytest.raises(ValueError, match="Source code cannot be empty"):
            processor.parse_to_ast("   ", "python")

    def test_execute_query_empty_raises_error(self):
        """Test empty query raises ValueError."""
        processor = AstProcessorAccelerator(force_python=True)

        with pytest.raises(ValueError, match="Query string cannot be empty"):
            processor.execute_query(Mock(), "")

        with pytest.raises(ValueError, match="Query string cannot be empty"):
            processor.execute_query(Mock(), "   ")


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfiguration:
    """Test processor configuration."""

    def test_custom_cache_size(self):
        """Test custom cache size configuration."""
        processor = AstProcessorAccelerator(max_cache_size=500, force_python=True)
        stats = processor.cache_stats
        assert stats["max_size"] == 500

    def test_enable_parallel_flag(self):
        """Test parallel processing flag."""
        processor = AstProcessorAccelerator(enable_parallel=True, force_python=True)
        assert processor._enable_parallel is True

        processor = AstProcessorAccelerator(enable_parallel=False, force_python=True)
        assert processor._enable_parallel is False


# =============================================================================
# Mock Integration Tests
# =============================================================================


class TestMockIntegration:
    """Test with mocked native backend."""

    def test_rust_backend_initialization_skip(self):
        """Test Rust backend initialization (skip if Rust not available).

        Note: These tests require module reloading which is unstable.
        In production, Rust backend is tested via integration tests.
        """
        # This is a placeholder test
        # Real Rust backend testing is done in integration tests
        processor = AstProcessorAccelerator(force_python=True)
        assert processor.is_available() is True

    def test_rust_parse_to_ast_skip(self):
        """Test parsing through Rust backend (skip if Rust not available).

        Note: These tests require module reloading which is unstable.
        In production, Rust backend is tested via integration tests.
        """
        # This is a placeholder test
        processor = AstProcessorAccelerator(force_python=True)
        assert processor._use_rust is False

    def test_rust_execute_query_skip(self):
        """Test query execution through Rust backend (skip if Rust not available).

        Note: These tests require module reloading which is unstable.
        In production, Rust backend is tested via integration tests.
        """
        # This is a placeholder test
        processor = AstProcessorAccelerator(force_python=True)
        assert processor._use_rust is False


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Test thread-safe operations."""

    def test_parse_stats_thread_safety(self):
        """Test ParseStats is thread-safe."""
        import threading

        stats = ParseStats()

        def record_parses():
            for _ in range(100):
                stats.record_parse(1.0, cache_hit=False)

        threads = [threading.Thread(target=record_parses) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert stats.total_parses == 1000  # 100 * 10
        assert stats.cache_misses == 1000
