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

"""Integration tests for AST processor accelerator.

Tests the Rust-backed AST processor with Python fallback,
including caching, parallel processing, and error handling.
"""

import pytest

from victor.native.accelerators import AstProcessorAccelerator, get_ast_processor


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def processor():
    """Create a fresh AST processor instance for each test."""
    return AstProcessorAccelerator(max_cache_size=100)


@pytest.fixture
def python_source():
    """Sample Python source code for testing."""
    return '''
def hello_world():
    """Print a greeting."""
    print("Hello, world!")

class Calculator:
    """Simple calculator class."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

async def async_function():
    """An async function."""
    await asyncio.sleep(1)
    return "done"
'''


@pytest.fixture
def javascript_source():
    """Sample JavaScript source code for testing."""
    return """
function helloWorld() {
    console.log("Hello, world!");
}

class Calculator {
    add(a, b) {
        return a + b;
    }

    multiply(a, b) {
        return a * b;
    }
}

async function asyncFunction() {
    await Promise.resolve();
    return "done";
}
"""


@pytest.fixture
def multiple_sources():
    """Multiple source files for parallel processing tests."""
    return [
        (
            "python",
            """
def func1():
    return 1

class Class1:
    pass
""",
        ),
        (
            "python",
            """
def func2():
    return 2

class Class2:
    pass
""",
        ),
        (
            "javascript",
            """
function func3() {
    return 3;
}

class Class3 {}
""",
        ),
    ]


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestAstProcessorBasics:
    """Test basic AST processor functionality."""

    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor is not None
        assert processor.is_available()
        assert isinstance(processor.is_rust_available(), bool)

    def test_get_singleton(self):
        """Test singleton instance retrieval."""
        processor1 = get_ast_processor()
        processor2 = get_ast_processor()
        assert processor1 is processor2

    def test_normalize_language(self, processor):
        """Test language name normalization."""
        assert processor.normalize_language("py") == "python"
        assert processor.normalize_language("js") == "javascript"
        assert processor.normalize_language("Python") == "python"
        assert processor.normalize_language("JS") == "javascript"

    def test_get_supported_languages(self, processor):
        """Test getting supported languages."""
        languages = processor.get_supported_languages()
        assert isinstance(languages, list)
        assert "python" in languages
        assert "javascript" in languages
        assert "rust" in languages

    def test_cache_stats_initial(self, processor):
        """Test initial cache statistics."""
        stats = processor.cache_stats
        assert isinstance(stats, dict)
        assert "size" in stats or "total_parses" in stats

    def test_parse_stats_initial(self, processor):
        """Test initial parse statistics."""
        stats = processor.parse_stats
        assert isinstance(stats, dict)
        assert stats["total_parses"] == 0


# =============================================================================
# Python AST Parsing Tests
# =============================================================================


class TestPythonParsing:
    """Test Python source code parsing."""

    def test_parse_python_source(self, processor, python_source):
        """Test parsing Python source code."""
        ast = processor.parse_to_ast(python_source, "python", "test.py")
        assert ast is not None

    def test_parse_python_with_file_path(self, processor, python_source):
        """Test parsing with file path for caching."""
        ast1 = processor.parse_to_ast(python_source, "python", "cached.py")
        ast2 = processor.parse_to_ast(python_source, "python", "cached.py")

        # Both should return valid ASTs
        assert ast1 is not None
        assert ast2 is not None

        # Check cache stats (should have at least one hit)
        stats = processor.parse_stats
        assert stats["total_parses"] >= 2

    def test_parse_empty_source_raises_error(self, processor):
        """Test that empty source raises ValueError."""
        with pytest.raises(ValueError, match="Source code cannot be empty"):
            processor.parse_to_ast("", "python")

        with pytest.raises(ValueError, match="Source code cannot be empty"):
            processor.parse_to_ast("   ", "python")


# =============================================================================
# JavaScript AST Parsing Tests
# =============================================================================


class TestJavaScriptParsing:
    """Test JavaScript source code parsing."""

    def test_parse_javascript_source(self, processor, javascript_source):
        """Test parsing JavaScript source code."""
        ast = processor.parse_to_ast(javascript_source, "javascript", "test.js")
        assert ast is not None

    def test_parse_typescript_variant(self, processor):
        """Test parsing TypeScript variant."""
        ts_source = "const x: number = 42;"
        ast = processor.parse_to_ast(ts_source, "typescript", "test.ts")
        assert ast is not None


# =============================================================================
# Query Execution Tests
# =============================================================================


class TestQueryExecution:
    """Test tree-sitter query execution."""

    def test_execute_function_query(self, processor, python_source):
        """Test querying function definitions."""
        ast = processor.parse_to_ast(python_source, "python")

        results = processor.execute_query(ast, "(function_definition name: (identifier) @name)")

        assert results is not None
        assert results.matches >= 3  # hello_world, add, multiply
        assert len(results) >= 3

    def test_execute_class_query(self, processor, python_source):
        """Test querying class definitions."""
        ast = processor.parse_to_ast(python_source, "python")

        results = processor.execute_query(ast, "(class_definition name: (identifier) @name)")

        assert results is not None
        assert results.matches >= 1  # Calculator class

    def test_execute_empty_query_raises_error(self, processor, python_source):
        """Test that empty query raises ValueError."""
        ast = processor.parse_to_ast(python_source, "python")

        with pytest.raises(ValueError, match="Query string cannot be empty"):
            processor.execute_query(ast, "")


# =============================================================================
# Symbol Extraction Tests
# =============================================================================


class TestSymbolExtraction:
    """Test symbol extraction from AST."""

    def test_extract_functions(self, processor, python_source):
        """Test extracting function symbols."""
        ast = processor.parse_to_ast(python_source, "python")

        symbols = processor.extract_symbols(ast, ["function_definition"])

        assert isinstance(symbols, list)
        assert len(symbols) >= 3  # hello_world, add, multiply, async_function

    def test_extract_classes(self, processor, python_source):
        """Test extracting class symbols."""
        ast = processor.parse_to_ast(python_source, "python")

        symbols = processor.extract_symbols(ast, ["class_definition"])

        assert isinstance(symbols, list)
        assert len(symbols) >= 1  # Calculator class

    def test_extract_all_symbols_default(self, processor, python_source):
        """Test extracting all symbols with default types."""
        ast = processor.parse_to_ast(python_source, "python")

        symbols = processor.extract_symbols(ast)

        assert isinstance(symbols, list)
        # Should extract functions, classes, etc.

    def test_extract_multiple_symbol_types(self, processor, python_source):
        """Test extracting multiple symbol types."""
        ast = processor.parse_to_ast(python_source, "python")

        symbols = processor.extract_symbols(ast, ["function_definition", "class_definition"])

        assert isinstance(symbols, list)
        assert len(symbols) >= 4  # 3 functions + 1 class


# =============================================================================
# Parallel Processing Tests
# =============================================================================


class TestParallelProcessing:
    """Test parallel symbol extraction."""

    def test_extract_symbols_parallel_single(self, processor):
        """Test parallel extraction with single file."""
        files = [("python", "def foo(): pass")]

        results = processor.extract_symbols_parallel(files)

        assert isinstance(results, dict)
        assert 0 in results
        assert isinstance(results[0], list)

    def test_extract_symbols_parallel_multiple(self, processor, multiple_sources):
        """Test parallel extraction with multiple files."""
        results = processor.extract_symbols_parallel(
            multiple_sources, ["function_definition", "class_definition"]
        )

        assert isinstance(results, dict)
        assert len(results) == 3

        # Each file should have results
        for idx in range(3):
            assert idx in results
            assert isinstance(results[idx], list)

    def test_extract_symbols_parallel_empty_list(self, processor):
        """Test parallel extraction with empty list."""
        results = processor.extract_symbols_parallel([])
        assert results == {}

    def test_extract_symbols_parallel_with_mixed_languages(self, processor, multiple_sources):
        """Test parallel extraction with mixed languages."""
        results = processor.extract_symbols_parallel(multiple_sources)

        assert isinstance(results, dict)
        assert len(results) == 3


# =============================================================================
# Caching Tests
# =============================================================================


class TestCaching:
    """Test AST caching functionality."""

    def test_cache_hit_same_file(self, processor, python_source):
        """Test cache hit for same file."""
        file_path = "cache_test.py"

        # First parse (cache miss)
        ast1 = processor.parse_to_ast(python_source, "python", file_path)
        stats1 = processor.parse_stats

        # Second parse (cache hit)
        ast2 = processor.parse_to_ast(python_source, "python", file_path)
        stats2 = processor.parse_stats

        assert ast1 is not None
        assert ast2 is not None
        assert stats2["total_parses"] == stats1["total_parses"] + 1

    def test_cache_miss_different_files(self, processor, python_source):
        """Test cache miss for different files."""
        ast1 = processor.parse_to_ast(python_source, "python", "file1.py")
        ast2 = processor.parse_to_ast(python_source, "python", "file2.py")

        assert ast1 is not None
        assert ast2 is not None

        stats = processor.parse_stats
        assert stats["total_parses"] >= 2

    def test_clear_cache(self, processor, python_source):
        """Test clearing the cache."""
        # Parse some files
        processor.parse_to_ast(python_source, "python", "file1.py")
        processor.parse_to_ast(python_source, "python", "file2.py")

        stats_before = processor.parse_stats
        assert stats_before["total_parses"] >= 2

        # Clear cache
        processor.clear_cache()

        stats_after = processor.parse_stats
        assert stats_after["total_parses"] == 0

    def test_cache_statistics(self, processor, python_source):
        """Test cache statistics tracking."""
        file_path = "stats_test.py"

        # First parse
        processor.parse_to_ast(python_source, "python", file_path)
        processor.parse_to_ast(python_source, "python", file_path)
        processor.parse_to_ast(python_source, "python", file_path)

        stats = processor.cache_stats
        assert isinstance(stats, dict)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_unsupported_language(self, processor):
        """Test handling of unsupported language."""
        # Should not crash, but may return None or raise
        source = "program main; end."
        try:
            ast = processor.parse_to_ast(source, "cobol")
            # If it doesn't raise, ast should be None or a valid AST
            assert ast is None or hasattr(ast, "root_node")
        except (ValueError, NotImplementedError):
            # Expected for unsupported languages
            pass

    def test_malformed_source(self, processor):
        """Test handling of malformed source code."""
        malformed = "def foo(\n    # incomplete function"

        # Should either parse with errors or handle gracefully
        try:
            ast = processor.parse_to_ast(malformed, "python")
            assert ast is not None
        except Exception:
            # Some parsers may raise on syntax errors
            pass

    def test_invalid_query(self, processor, python_source):
        """Test handling of invalid query syntax."""
        ast = processor.parse_to_ast(python_source, "python")

        # Invalid query syntax
        results = processor.execute_query(ast, "(((invalid query)))")

        # Should return empty results, not crash
        assert results is not None
        assert results.matches == 0


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.slow
class TestPerformance:
    """Performance benchmarks for AST processor."""

    def test_parse_performance_small_file(self, processor, python_source):
        """Test parsing performance for small files."""
        import time

        start = time.perf_counter()
        for _ in range(100):
            processor.parse_to_ast(python_source, "python")
        duration = time.perf_counter() - start

        # Should complete 100 parses in reasonable time
        assert duration < 10.0  # Less than 10 seconds

    def test_parse_performance_with_cache(self, processor, python_source):
        """Test parsing performance with caching."""
        file_path = "perf_test.py"

        # Warm up cache
        processor.parse_to_ast(python_source, "python", file_path)

        # Measure cached performance
        import time

        start = time.perf_counter()
        for _ in range(100):
            processor.parse_to_ast(python_source, "python", file_path)
        duration = time.perf_counter() - start

        # Cached parses should be very fast
        assert duration < 1.0  # Less than 1 second for 100 cached parses


# =============================================================================
# Backend Tests
# =============================================================================


class TestBackendSelection:
    """Test Rust vs Python backend selection."""

    def test_force_python_backend(self):
        """Test forcing Python backend."""
        processor = AstProcessorAccelerator(force_python=True)
        assert not processor.is_rust_available()

    def test_rust_backend_available(self):
        """Test if Rust backend is available."""
        processor = AstProcessorAccelerator(force_python=False)

        # Just check it doesn't crash
        is_rust = processor.is_rust_available()
        assert isinstance(is_rust, bool)

    def test_get_version(self, processor):
        """Test getting backend version."""
        version = processor.get_version()
        assert version is None or isinstance(version, str)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_full_workflow(self, processor, python_source):
        """Test complete workflow: parse, query, extract."""
        # Parse
        ast = processor.parse_to_ast(python_source, "python", "workflow_test.py")
        assert ast is not None

        # Query
        results = processor.execute_query(ast, "(function_definition name: (identifier) @name)")
        assert results.matches >= 3

        # Extract symbols
        symbols = processor.extract_symbols(ast, ["function_definition"])
        assert len(symbols) >= 3

        # Check stats
        stats = processor.parse_stats
        assert stats["total_parses"] >= 1

    def test_multi_language_workflow(self, processor, python_source, javascript_source):
        """Test workflow with multiple languages."""
        # Parse Python
        py_ast = processor.parse_to_ast(python_source, "python", "multi.py")
        assert py_ast is not None

        # Parse JavaScript
        js_ast = processor.parse_to_ast(javascript_source, "javascript", "multi.js")
        assert js_ast is not None

        # Extract from both
        py_symbols = processor.extract_symbols(py_ast, ["function_definition"])
        js_symbols = processor.extract_symbols(js_ast, ["function_definition"])

        assert len(py_symbols) >= 3
        assert len(js_symbols) >= 3
