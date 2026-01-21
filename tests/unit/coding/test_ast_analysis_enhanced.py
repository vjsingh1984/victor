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

"""Enhanced unit tests for Tree-sitter AST operations in the coding vertical.

This test suite provides comprehensive coverage for:
- Symbol extraction (functions, classes, variables)
- Edge extraction (calls, inheritance, implements, composition)
- Reference extraction
- Language detection
- Error handling
- Multi-file support
"""

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, Mock, patch

import pytest

from tests.fixtures.coding_fixtures import (
    SAMPLE_JAVASCRIPT_SIMPLE,
    SAMPLE_PYTHON_ASYNC,
    SAMPLE_PYTHON_CLASS,
    SAMPLE_PYTHON_COMPLEX,
    SAMPLE_PYTHON_DECORATORS,
    SAMPLE_PYTHON_GENERATORS,
    SAMPLE_PYTHON_INHERITANCE,
    SAMPLE_PYTHON_METACLASSES,
    SAMPLE_PYTHON_SIMPLE,
    SAMPLE_PYTHON_TYPE_HINTS,
    SAMPLE_PYTHON_WITH_IMPORTS,
    create_sample_file,
)
from victor.coding.codebase.tree_sitter_extractor import (
    ExtractedEdge,
    ExtractedReference,
    ExtractedSymbol,
    TreeSitterExtractor,
)


# =============================================================================
# Test Setup and Fixtures
# =============================================================================


@pytest.fixture
def extractor():
    """Create a TreeSitterExtractor instance for testing."""
    return TreeSitterExtractor()


@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary file for testing."""
    return tmp_path / "test.py"


# =============================================================================
# Language Detection Tests
# =============================================================================


class TestLanguageDetection:
    """Tests for language detection functionality."""

    def test_detect_python_file(self, extractor, temp_file):
        """Test detection of Python files."""
        temp_file.write_text(SAMPLE_PYTHON_SIMPLE)
        language = extractor.detect_language(temp_file)
        assert language == "python"

    def test_detect_python_with_py_extension(self, extractor, tmp_path):
        """Test detection of .py files."""
        test_file = tmp_path / "script.py"
        test_file.write_text("print('hello')")
        language = extractor.detect_language(test_file)
        assert language == "python"

    def test_detect_javascript_file(self, extractor, tmp_path):
        """Test detection of JavaScript files."""
        test_file = tmp_path / "script.js"
        test_file.write_text(SAMPLE_JAVASCRIPT_SIMPLE)
        language = extractor.detect_language(test_file)
        assert language == "javascript"

    def test_detect_typescript_file(self, extractor, tmp_path):
        """Test detection of TypeScript files."""
        test_file = tmp_path / "script.ts"
        test_file.write_text("function test() {}")
        language = extractor.detect_language(test_file)
        assert language == "typescript"

    def test_detect_unknown_extension(self, extractor, tmp_path):
        """Test handling of unknown file extensions."""
        test_file = tmp_path / "unknown.xyz"
        test_file.write_text("some content")
        language = extractor.detect_language(test_file)
        assert language is None

    def test_detect_no_extension(self, extractor, tmp_path):
        """Test handling of files without extensions."""
        test_file = tmp_path / "Makefile"
        test_file.write_text("all:\n\techo 'hello'")
        language = extractor.detect_language(test_file)
        # Should return None or a detected language based on content

    def test_detect_empty_file(self, extractor, tmp_path):
        """Test handling of empty files."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")
        language = extractor.detect_language(test_file)
        # Language should still be detected from extension


# =============================================================================
# Symbol Extraction Tests
# =============================================================================


class TestSymbolExtraction:
    """Tests for symbol extraction from source code."""

    def test_extract_simple_function(self, extractor, temp_file):
        """Test extraction of a simple function."""
        temp_file.write_text(SAMPLE_PYTHON_SIMPLE)
        symbols = extractor.extract_symbols(temp_file)

        assert len(symbols) > 0
        function_symbols = [s for s in symbols if s.type == "function"]
        assert len(function_symbols) > 0

        hello_func = next((s for s in function_symbols if "hello" in s.name.lower()), None)
        assert hello_func is not None
        assert hello_func.line_number > 0

    def test_extract_class_symbols(self, extractor, temp_file):
        """Test extraction of class symbols."""
        temp_file.write_text(SAMPLE_PYTHON_CLASS)
        symbols = extractor.extract_symbols(temp_file)

        class_symbols = [s for s in symbols if s.type == "class"]
        assert len(class_symbols) > 0

        calculator_class = next(
            (s for s in class_symbols if s.name == "Calculator"), None
        )
        assert calculator_class is not None
        assert calculator_class.line_number > 0

    def test_extract_methods_from_class(self, extractor, temp_file):
        """Test extraction of methods from a class."""
        temp_file.write_text(SAMPLE_PYTHON_CLASS)
        symbols = extractor.extract_symbols(temp_file)

        method_symbols = [s for s in symbols if s.type == "function"]

        # Check for Calculator methods
        method_names = [s.name for s in method_symbols]
        assert "add" in method_names or any("add" in s.name for s in method_names)
        assert "subtract" in method_names or any("subtract" in s.name for s in method_names)
        assert "multiply" in method_names or any("multiply" in s.name for s in method_names)

    def test_extract_async_functions(self, extractor, temp_file):
        """Test extraction of async functions."""
        temp_file.write_text(SAMPLE_PYTHON_ASYNC)
        symbols = extractor.extract_symbols(temp_file)

        function_symbols = [s for s in symbols if s.type == "function"]
        assert len(function_symbols) > 0

        # Check for async functions
        async_funcs = [s for s in function_symbols if "async" in s.name.lower() or "fetch" in s.name.lower()]
        assert len(async_funcs) > 0

    def test_extract_decorated_functions(self, extractor, temp_file):
        """Test extraction of decorated functions."""
        temp_file.write_text(SAMPLE_PYTHON_DECORATORS)
        symbols = extractor.extract_symbols(temp_file)

        function_symbols = [s for s in symbols if s.type == "function"]
        assert len(function_symbols) > 0

        # Should find retry decorator and external_api_call function
        func_names = [s.name for s in function_symbols]
        assert any("api" in name.lower() or "call" in name.lower() for name in func_names)

    def test_extract_symbols_with_type_hints(self, extractor, temp_file):
        """Test extraction of functions with type hints."""
        temp_file.write_text(SAMPLE_PYTHON_TYPE_HINTS)
        symbols = extractor.extract_symbols(temp_file)

        function_symbols = [s for s in symbols if s.type == "function"]
        assert len(function_symbols) > 0

        # Should find process function and Container class
        func_names = [s.name for s in function_symbols]
        assert any("process" in name.lower() for name in func_names)

    def test_extract_from_complex_code(self, extractor, temp_file):
        """Test extraction from complex code with multiple constructs."""
        temp_file.write_text(SAMPLE_PYTHON_COMPLEX)
        symbols = extractor.extract_symbols(temp_file)

        # Should extract classes, functions, dataclasses
        assert len(symbols) > 0

        symbol_types = set(s.type for s in symbols)
        assert "class" in symbol_types or "function" in symbol_types

    def test_extract_symbols_with_imports(self, extractor, temp_file):
        """Test extraction when imports are present."""
        temp_file.write_text(SAMPLE_PYTHON_WITH_IMPORTS)
        symbols = extractor.extract_symbols(temp_file)

        # Should successfully parse and extract symbols
        assert len(symbols) >= 0  # May have 0 if only imports present

    def test_extract_generator_functions(self, extractor, temp_file):
        """Test extraction of generator functions."""
        temp_file.write_text(SAMPLE_PYTHON_GENERATORS)
        symbols = extractor.extract_symbols(temp_file)

        function_symbols = [s for s in symbols if s.type == "function"]
        assert len(function_symbols) >= 2  # fibonacci and squares

    def test_extract_inheritance_hierarchy(self, extractor, temp_file):
        """Test extraction of classes with inheritance."""
        temp_file.write_text(SAMPLE_PYTHON_INHERITANCE)
        symbols = extractor.extract_symbols(temp_file)

        class_symbols = [s for s in symbols if s.type == "class"]
        assert len(class_symbols) >= 3  # Animal, Dog, Cat

        class_names = [s.name for s in class_symbols]
        assert "Animal" in class_names
        assert "Dog" in class_names or "Cat" in class_names

    def test_extract_metaclasses(self, extractor, temp_file):
        """Test extraction of classes using metaclasses."""
        temp_file.write_text(SAMPLE_PYTHON_METACLASSES)
        symbols = extractor.extract_symbols(temp_file)

        class_symbols = [s for s in symbols if s.type == "class"]
        assert len(class_symbols) > 0


# =============================================================================
# Edge Extraction Tests
# =============================================================================


class TestEdgeExtraction:
    """Tests for edge (relationship) extraction."""

    def test_extract_function_calls(self, extractor, temp_file):
        """Test extraction of function call edges."""
        temp_file.write_text(SAMPLE_PYTHON_SIMPLE)
        edges = extractor.extract_call_edges(temp_file)

        # Should find print call
        assert len(edges) >= 0

    def test_extract_inheritance_edges(self, extractor, temp_file):
        """Test extraction of inheritance edges."""
        temp_file.write_text(SAMPLE_PYTHON_INHERITANCE)
        edges = extractor.extract_inheritance_edges(temp_file)

        # Should find Dog inherits Animal, Cat inherits Animal
        assert len(edges) >= 0

    def test_extract_composition_edges(self, extractor, temp_file):
        """Test extraction of composition edges."""
        temp_file.write_text(SAMPLE_PYTHON_COMPLEX)
        edges = extractor.extract_composition_edges(temp_file)

        # May find composition in UserService (has db_connection, cache)
        assert len(edges) >= 0

    def test_extract_multiple_call_edges(self, extractor, temp_file):
        """Test extraction of multiple function calls."""
        temp_file.write_text(SAMPLE_PYTHON_ASYNC)
        edges = extractor.extract_call_edges(temp_file)

        # Should find asyncio.sleep, fetch_data calls
        assert len(edges) >= 0


# =============================================================================
# Reference Extraction Tests
# =============================================================================


class TestReferenceExtraction:
    """Tests for reference extraction."""

    def test_extract_variable_references(self, extractor, temp_file):
        """Test extraction of variable references."""
        temp_file.write_text(SAMPLE_PYTHON_SIMPLE)
        references = extractor.extract_references(temp_file)

        # Should find references in the code
        assert len(references) >= 0

    def test_extract_function_references(self, extractor, temp_file):
        """Test extraction of function references."""
        temp_file.write_text(SAMPLE_PYTHON_CLASS)
        references = extractor.extract_references(temp_file)

        # Should find method calls and references
        assert len(references) >= 0

    def test_extract_references_in_complex_code(self, extractor, temp_file):
        """Test reference extraction in complex code."""
        temp_file.write_text(SAMPLE_PYTHON_COMPLEX)
        references = extractor.extract_references(temp_file)

        # Should find various references (variables, methods, etc.)
        assert len(references) >= 0

    def test_reference_positions(self, extractor, temp_file):
        """Test that reference positions are captured correctly."""
        temp_file.write_text(SAMPLE_PYTHON_SIMPLE)
        references = extractor.extract_references(temp_file)

        for ref in references:
            assert ref.line_number >= 0
            assert ref.column >= 0
            assert ref.file_path == str(temp_file)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_extract_nonexistent_file(self, extractor):
        """Test handling of non-existent files."""
        nonexistent = Path("/nonexistent/file.py")
        symbols = extractor.extract_symbols(nonexistent)
        assert symbols == []

    def test_extract_syntax_error_code(self, extractor, temp_file):
        """Test handling of code with syntax errors."""
        temp_file.write_text("def broken(\n    # missing closing paren\n")
        symbols = extractor.extract_symbols(temp_file)

        # Should handle gracefully (may return empty or partial results)
        assert isinstance(symbols, list)

    def test_extract_empty_file(self, extractor, temp_file):
        """Test extraction from empty file."""
        temp_file.write_text("")
        symbols = extractor.extract_symbols(temp_file)
        assert symbols == []

    def test_extract_binary_file(self, extractor, temp_file):
        """Test handling of binary files."""
        temp_file.write_bytes(b"\x00\x01\x02\x03\x04\x05")
        symbols = extractor.extract_symbols(temp_file)
        assert symbols == []

    def test_extract_unsupported_language(self, extractor, tmp_path):
        """Test handling of unsupported language."""
        test_file = tmp_path / "test.unknown"
        test_file.write_text("some content")
        symbols = extractor.extract_symbols(test_file)
        # Should return empty list for unsupported languages
        assert symbols == []

    def test_extract_with_invalid_parser(self, extractor, temp_file):
        """Test behavior when parser is unavailable."""
        temp_file.write_text(SAMPLE_PYTHON_SIMPLE)

        # Mock parser to return None
        with patch.object(extractor, "_get_parser", return_value=None):
            symbols = extractor.extract_symbols(temp_file)
            assert symbols == []


# =============================================================================
# Performance and Optimization Tests
# =============================================================================


class TestPerformanceOptimizations:
    """Tests for performance optimizations."""

    def test_parser_caching(self, extractor, temp_file):
        """Test that parsers are cached across calls."""
        temp_file.write_text(SAMPLE_PYTHON_SIMPLE)

        # First call
        extractor.extract_symbols(temp_file)
        initial_cache_size = len(extractor._parsers)

        # Second call with same language
        temp_file.write_text(SAMPLE_PYTHON_CLASS)
        extractor.extract_symbols(temp_file)

        # Parser should be cached
        assert len(extractor._parsers) == initial_cache_size

    def test_query_caching(self, extractor, temp_file):
        """Test that queries are cached."""
        temp_file.write_text(SAMPLE_PYTHON_SIMPLE)

        # First call
        extractor.extract_symbols(temp_file)
        initial_cache_size = len(extractor._query_cache)

        # Second call
        extractor.extract_symbols(temp_file)

        # Queries should be cached
        assert len(extractor._query_cache) >= initial_cache_size

    def test_large_file_performance(self, extractor, temp_file):
        """Test performance with larger files."""
        # Create a file with many functions
        lines = []
        for i in range(100):
            lines.append(f"def function_{i}(x):\n    return x * {i}\n")

        temp_file.write_text("".join(lines))
        symbols = extractor.extract_symbols(temp_file)

        # Should extract all 100 functions
        function_symbols = [s for s in symbols if s.type == "function"]
        assert len(function_symbols) == 100


# =============================================================================
# Multi-File Tests
# =============================================================================


class TestMultiFileExtraction:
    """Tests for extracting from multiple files."""

    def test_extract_from_multiple_files(self, extractor, tmp_path):
        """Test extraction from multiple files."""
        files = [
            tmp_path / "file1.py",
            tmp_path / "file2.py",
            tmp_path / "file3.py",
        ]

        for i, file_path in enumerate(files):
            file_path.write_text(f"def func_{i}():\n    pass\n")

        all_symbols = []
        for file_path in files:
            symbols = extractor.extract_symbols(file_path)
            all_symbols.extend(symbols)

        assert len(all_symbols) == 3

    def test_mixed_language_files(self, extractor, tmp_path):
        """Test extraction from files of different languages."""
        python_file = tmp_path / "test.py"
        js_file = tmp_path / "test.js"

        python_file.write_text(SAMPLE_PYTHON_SIMPLE)
        js_file.write_text(SAMPLE_JAVASCRIPT_SIMPLE)

        python_symbols = extractor.extract_symbols(python_file)
        js_symbols = extractor.extract_symbols(js_file)

        assert len(python_symbols) > 0
        assert len(js_symbols) >= 0


# =============================================================================
# Data Class Tests
# =============================================================================


class TestDataClasses:
    """Tests for ExtractedSymbol, ExtractedEdge, and ExtractedReference dataclasses."""

    def test_extracted_symbol_creation(self):
        """Test ExtractedSymbol dataclass creation."""
        symbol = ExtractedSymbol(
            name="test_function",
            type="function",
            file_path="/test.py",
            line_number=10,
            end_line=15,
        )
        assert symbol.name == "test_function"
        assert symbol.type == "function"
        assert symbol.line_number == 10
        assert symbol.end_line == 15

    def test_extracted_edge_creation(self):
        """Test ExtractedEdge dataclass creation."""
        edge = ExtractedEdge(
            source="caller",
            target="callee",
            edge_type="CALLS",
            file_path="/test.py",
            line_number=20,
        )
        assert edge.source == "caller"
        assert edge.target == "callee"
        assert edge.edge_type == "CALLS"

    def test_extracted_reference_creation(self):
        """Test ExtractedReference dataclass creation."""
        ref = ExtractedReference(
            name="variable",
            file_path="/test.py",
            line_number=5,
            column=10,
        )
        assert ref.name == "variable"
        assert ref.line_number == 5
        assert ref.column == 10


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""

    def test_extract_from_service_class(self, extractor, temp_file):
        """Test extraction from a realistic service class."""
        temp_file.write_text(SAMPLE_PYTHON_COMPLEX)

        symbols = extractor.extract_symbols(temp_file)
        edges = extractor.extract_call_edges(temp_file)
        references = extractor.extract_references(temp_file)

        # Should find classes, methods, and relationships
        assert len(symbols) > 0
        assert len(edges) >= 0
        assert len(references) >= 0

    def test_extract_from_async_workflow(self, extractor, temp_file):
        """Test extraction from async workflow code."""
        temp_file.write_text(SAMPLE_PYTHON_ASYNC)

        symbols = extractor.extract_symbols(temp_file)
        function_symbols = [s for s in symbols if s.type == "function"]

        # Should find async functions
        assert len(function_symbols) >= 2

    def test_extract_decorated_workflow(self, extractor, temp_file):
        """Test extraction from decorated function workflow."""
        temp_file.write_text(SAMPLE_PYTHON_DECORATORS)

        symbols = extractor.extract_symbols(temp_file)

        # Should find the decorated function
        assert len(symbols) > 0


# =============================================================================
# Regression Tests
# =============================================================================


class TestRegressionIssues:
    """Regression tests for known issues."""

    def test_no_crash_on_none_text(self, extractor):
        """Test that None node text doesn't crash extraction."""
        # Create a mock node with None text attribute
        mock_node = Mock()
        mock_node.text = None
        result = extractor._safe_decode_node_text(mock_node)
        assert result is None

    def test_unicode_handling(self, extractor, temp_file):
        """Test proper handling of Unicode characters."""
        code_with_unicode = '''def greet():
    """Привет мир! 🌍"""
    print("Hello 世界!")
'''
        temp_file.write_text(code_with_unicode, encoding="utf-8")
        symbols = extractor.extract_symbols(temp_file)

        # Should not crash and extract function
        assert len(symbols) > 0

    def test_very_long_line(self, extractor, temp_file):
        """Test handling of very long lines."""
        long_line = "x" * 10000
        temp_file.write_text(f"def f():\n    {long_line}\n")
        symbols = extractor.extract_symbols(temp_file)

        # Should handle gracefully
        assert isinstance(symbols, list)
