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

"""Tests for Rust AST processor integration with tree_sitter_manager."""

import tempfile
from pathlib import Path

import pytest

from victor.coding.codebase.tree_sitter_manager import (
    clear_ast_cache,
    extract_symbols_parallel,
    get_cache_stats,
    get_language,
    get_parser,
    parse_file_accelerated,
    parse_file_with_timing,
    run_query,
)
from victor.native.accelerators.ast_processor import (
    ASTProcessorAccelerator,
    get_ast_processor,
    is_rust_available,
    reset_ast_processor,
)


class TestASTProcessorAccelerator:
    """Test the AST processor accelerator."""

    def test_is_rust_available(self):
        """Test that we can check Rust availability."""
        # This should not raise an error
        available = is_rust_available()
        assert isinstance(available, bool)

    def test_get_ast_processor_singleton(self):
        """Test that get_ast_processor returns a singleton."""
        reset_ast_processor()

        processor1 = get_ast_processor()
        processor2 = get_ast_processor()

        assert processor1 is processor2
        assert isinstance(processor1, ASTProcessorAccelerator)

    def test_processor_has_backend(self):
        """Test that processor has a backend attribute."""
        processor = get_ast_processor()
        assert hasattr(processor, "backend")
        assert processor.backend in ("rust", "python")

    def test_processor_has_cache_stats(self):
        """Test that processor has cache stats."""
        processor = get_ast_processor()
        stats = processor.cache_stats

        assert isinstance(stats, dict)
        assert "size" in stats
        assert "max_size" in stats


class TestAcceleratedParsing:
    """Test accelerated parsing functions."""

    def setup_method(self):
        """Create temporary test files."""
        self.temp_dir = tempfile.mkdtemp()

        # Create a Python test file
        self.py_file = Path(self.temp_dir) / "test.py"
        self.py_file.write_text(
            """
def hello_world():
    '''Say hello.'''
    print("Hello, World!")

class MyClass:
    '''A test class.'''
    def method(self):
        pass
""",
            encoding="utf-8",
        )

        # Create a JavaScript test file
        self.js_file = Path(self.temp_dir) / "test.js"
        self.js_file.write_text(
            """
function helloWorld() {
    console.log("Hello, World!");
}

class MyClass {
    method() {
        return true;
    }
}
""",
            encoding="utf-8",
        )

    def teardown_method(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_parse_file_accelerated(self):
        """Test accelerated file parsing."""
        tree = parse_file_accelerated(str(self.py_file))

        assert tree is not None
        assert tree.root_node.type == "module"

    def test_parse_file_with_timing(self):
        """Test parsing with timing information."""
        tree, elapsed = parse_file_with_timing(str(self.py_file))

        assert tree is not None
        assert isinstance(elapsed, float)
        assert elapsed >= 0

    def test_parse_unsupported_language_falls_back(self):
        """Test that unsupported languages fall back gracefully."""
        # Create a file with unknown extension
        unknown_file = Path(self.temp_dir) / "test.unknown"
        unknown_file.write_text("some content", encoding="utf-8")

        # Should not crash, may return None or use default parser
        try:
            tree = parse_file_accelerated(str(unknown_file))
            # If it succeeds, tree might be None for unsupported languages
            assert tree is None or hasattr(tree, "root_node")
        except Exception as e:
            # If it raises, should be a reasonable error
            assert "Unsupported language" in str(e) or "not installed" in str(e)

    def test_run_query(self):
        """Test running queries on parsed tree."""
        tree = parse_file_accelerated(str(self.py_file))

        # Query for function definitions
        query = "(function_definition name: (identifier) @name)"
        captures = run_query(tree, query, "python")

        assert isinstance(captures, dict)
        # Should find at least the hello_world function
        assert len(captures) > 0

    def test_extract_symbols_parallel(self):
        """Test parallel symbol extraction."""
        results = extract_symbols_parallel(
            [str(self.py_file), str(self.js_file)],
            ["function", "class"],
        )

        assert isinstance(results, dict)
        assert str(self.py_file) in results
        assert str(self.js_file) in results

        # Python file should have function and class
        py_symbols = results[str(self.py_file)]
        assert len(py_symbols) >= 2  # hello_world + MyClass

        # Check symbol structure
        if py_symbols:
            symbol = py_symbols[0]
            assert "name" in symbol
            assert "type" in symbol
            assert "line" in symbol


class TestCacheManagement:
    """Test cache management functions."""

    def test_clear_ast_cache(self):
        """Test clearing the AST cache."""
        # Parse a file to populate cache
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"def test(): pass")
            temp_file = f.name

        try:
            parse_file_accelerated(temp_file)

            # Clear cache
            clear_ast_cache()

            # Should not raise
            assert True
        finally:
            Path(temp_file).unlink()

    def test_get_cache_stats(self):
        """Test getting cache statistics."""
        stats = get_cache_stats()

        assert isinstance(stats, dict)
        assert "size" in stats
        assert "max_size" in stats
        assert "hits" in stats
        assert "misses" in stats


class TestBackwardCompatibility:
    """Test that existing tree-sitter functions still work."""

    def test_get_language_still_works(self):
        """Test that get_language still works."""
        lang = get_language("python")
        assert lang is not None

    def test_get_parser_still_works(self):
        """Test that get_parser still works."""
        parser = get_parser("python")
        assert parser is not None

    def test_parser_can_parse(self):
        """Test that parser can still parse code."""
        parser = get_parser("python")
        tree = parser.parse(b"def foo(): pass")

        assert tree is not None
        assert tree.root_node.type == "module"


class TestPerformanceLogging:
    """Test performance logging integration."""

    def test_parse_logs_performance(self, caplog):
        """Test that parsing logs performance metrics."""
        import logging

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"def test(): pass")
            temp_file = f.name

        try:
            with caplog.at_level(logging.DEBUG):
                tree, elapsed = parse_file_with_timing(temp_file)

            # Check that performance was logged
            assert any(
                "Parsed" in record.message and "ms" in record.message for record in caplog.records
            )
        finally:
            Path(temp_file).unlink()


@pytest.mark.integration
class TestRealWorldIntegration:
    """Integration tests with real codebase files."""

    def test_parse_victor_module(self):
        """Test parsing actual Victor module."""
        # Try to parse this test file itself
        this_file = Path(__file__)

        if this_file.exists():
            tree = parse_file_accelerated(str(this_file))
            assert tree is not None
            assert tree.root_node.type == "module"

    def test_extract_symbols_from_victor(self):
        """Test symbol extraction from Victor codebase."""
        # Find a few Python files in victor
        victor_dir = Path(__file__).parent.parent.parent.parent.parent / "victor"

        if not victor_dir.exists():
            pytest.skip("Victor directory not found")

        py_files = list(victor_dir.rglob("*.py"))[:5]  # Test first 5 files

        if not py_files:
            pytest.skip("No Python files found in Victor")

        results = extract_symbols_parallel(
            [str(f) for f in py_files],
            ["function", "class"],
        )

        assert isinstance(results, dict)
        assert len(results) > 0
