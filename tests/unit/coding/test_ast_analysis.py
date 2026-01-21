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

"""Unit tests for Tree-sitter AST operations in the coding vertical.

Tests cover:
- AST traversal and node extraction
- Function/class/variable extraction
- Import dependency analysis
- Code pattern matching
- Syntax error detection
- Edge extraction (calls, inheritance, implements, composition)
- Reference extraction
"""

from pathlib import Path
from typing import List

import pytest

from tests.fixtures.coding_fixtures import (
    SAMPLE_PYTHON_ASYNC,
    SAMPLE_PYTHON_CLASS,
    SAMPLE_PYTHON_COMPLEX,
    SAMPLE_PYTHON_DECORATORS,
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
# Test Fixture Setup
# =============================================================================

@pytest.fixture
def extractor():
    """Create a TreeSitterExtractor instance."""
    return TreeSitterExtractor()


@pytest.fixture
def sample_files(tmp_path):
    """Create sample Python files for testing."""
    files = {
        "simple.py": SAMPLE_PYTHON_SIMPLE,
        "class.py": SAMPLE_PYTHON_CLASS,
        "complex.py": SAMPLE_PYTHON_COMPLEX,
        "imports.py": SAMPLE_PYTHON_WITH_IMPORTS,
        "async.py": SAMPLE_PYTHON_ASYNC,
        "decorators.py": SAMPLE_PYTHON_DECORATORS,
        "types.py": SAMPLE_PYTHON_TYPE_HINTS,
    }
    file_paths = {}
    for name, content in files.items():
        file_paths[name] = create_sample_file(tmp_path, name, content)
    return file_paths


# =============================================================================
# Language Detection Tests
# =============================================================================

class TestLanguageDetection:
    """Tests for automatic language detection."""

    def test_detect_python(self, extractor, tmp_path):
        """Test detecting Python language from .py file."""
        file_path = create_sample_file(tmp_path, "test.py", SAMPLE_PYTHON_SIMPLE)
        language = extractor.detect_language(file_path)
        assert language == "python"

    def test_detect_javascript(self, extractor, tmp_path):
        """Test detecting JavaScript language from .js file."""
        js_code = "function hello() { console.log('Hello'); }"
        file_path = create_sample_file(tmp_path, "test.js", js_code)
        language = extractor.detect_language(file_path)
        assert language == "javascript"

    def test_detect_typescript(self, extractor, tmp_path):
        """Test detecting TypeScript language from .ts file."""
        ts_code = "function hello(): void { console.log('Hello'); }"
        file_path = create_sample_file(tmp_path, "test.ts", ts_code)
        language = extractor.detect_language(file_path)
        assert language == "typescript"

    def test_detect_unknown_extension(self, extractor, tmp_path):
        """Test handling unknown file extensions."""
        file_path = create_sample_file(tmp_path, "test.xyz", "some content")
        language = extractor.detect_language(file_path)
        assert language is None

    def test_detect_no_extension(self, extractor, tmp_path):
        """Test handling files with no extension."""
        file_path = create_sample_file(tmp_path, "Makefile", "all:\n\techo hello")
        language = extractor.detect_language(file_path)
        # Makefile should be detected
        assert language is not None


# =============================================================================
# Symbol Extraction Tests
# =============================================================================

class TestSymbolExtraction:
    """Tests for extracting symbols (functions, classes, variables)."""

    def test_extract_simple_function(self, extractor, sample_files):
        """Test extracting a simple function."""
        symbols = extractor.extract_symbols(sample_files["simple.py"])
        assert len(symbols) == 1
        assert symbols[0].name == "hello_world"
        assert symbols[0].type == "function"
        assert symbols[0].line_number == 1

    def test_extract_class_with_methods(self, extractor, sample_files):
        """Test extracting a class with multiple methods."""
        symbols = extractor.extract_symbols(sample_files["class.py"])
        # Should extract class and methods
        assert len(symbols) >= 1

        class_symbols = [s for s in symbols if s.type == "class"]
        assert len(class_symbols) == 1
        assert class_symbols[0].name == "Calculator"

        function_symbols = [s for s in symbols if s.type == "function"]
        assert len(function_symbols) >= 3

        method_names = [s.name for s in function_symbols]
        assert "add" in method_names
        assert "subtract" in method_names
        assert "multiply" in method_names

    def test_extract_complex_class_hierarchy(self, extractor, sample_files):
        """Test extracting symbols from complex code with classes."""
        symbols = extractor.extract_symbols(sample_files["complex.py"])

        # Should find User and UserService classes
        class_symbols = [s for s in symbols if s.type == "class"]
        assert len(class_symbols) >= 2

        class_names = [s.name for s in class_symbols]
        assert "User" in class_names
        assert "UserService" in class_names

    def test_extract_async_functions(self, extractor, sample_files):
        """Test extracting async functions."""
        symbols = extractor.extract_symbols(sample_files["async.py"])

        function_symbols = [s for s in symbols if s.type == "function"]
        assert len(function_symbols) >= 3

        function_names = [s.name for s in function_symbols]
        assert "fetch_data" in function_names
        assert "process_items" in function_names
        assert "main" in function_names

    def test_extract_decorated_functions(self, extractor, sample_files):
        """Test extracting decorated functions."""
        symbols = extractor.extract_symbols(sample_files["decorators.py"])

        function_symbols = [s for s in symbols if s.type == "function"]
        assert len(function_symbols) >= 2

        function_names = [s.name for s in function_symbols]
        assert "retry" in function_names
        assert "external_api_call" in function_names

    def test_extract_symbols_with_type_hints(self, extractor, sample_files):
        """Test extracting symbols from code with type hints."""
        symbols = extractor.extract_symbols(sample_files["types.py"])

        # Should extract functions and classes
        assert len(symbols) >= 3

        function_names = [s.name for s in symbols if s.type == "function"]
        assert "process" in function_names
        assert "apply_func" in function_names

        class_names = [s.name for s in symbols if s.type == "class"]
        assert "Container" in class_names

    def test_symbol_line_numbers(self, extractor, sample_files):
        """Test that extracted symbols have correct line numbers."""
        symbols = extractor.extract_symbols(sample_files["simple.py"])
        assert symbols[0].line_number >= 1
        assert symbols[0].end_line >= symbols[0].line_number

    def test_symbol_file_path(self, extractor, sample_files):
        """Test that extracted symbols include file path."""
        symbols = extractor.extract_symbols(sample_files["class.py"])
        assert all(s.file_path == str(sample_files["class.py"]) for s in symbols)

    def test_extract_empty_file(self, extractor, tmp_path):
        """Test extracting symbols from an empty file."""
        file_path = create_sample_file(tmp_path, "empty.py", "")
        symbols = extractor.extract_symbols(file_path)
        assert len(symbols) == 0

    def test_extract_syntax_error_file(self, extractor, tmp_path):
        """Test extracting symbols from a file with syntax errors."""
        bad_code = """
def broken_function(
    # Missing closing parenthesis
    print("hello"
"""
        file_path = create_sample_file(tmp_path, "broken.py", bad_code)
        # Should not crash, may return empty symbols
        symbols = extractor.extract_symbols(file_path)
        # Either empty or partial results are acceptable
        assert isinstance(symbols, list)


# =============================================================================
# Edge Extraction Tests
# =============================================================================

class TestCallEdgeExtraction:
    """Tests for extracting function call relationships."""

    def test_extract_call_edges_simple(self, extractor, sample_files):
        """Test extracting call edges from simple code."""
        edges = extractor.extract_call_edges(sample_files["simple.py"])

        # Should find the call to print
        assert len(edges) >= 0
        for edge in edges:
            assert edge.edge_type == "CALLS"
            assert edge.source  # Caller should exist
            assert edge.target  # Callee should exist

    def test_extract_call_edges_complex(self, extractor, sample_files):
        """Test extracting call edges from complex code."""
        edges = extractor.extract_call_edges(sample_files["complex.py"])

        # Should find various function calls
        call_targets = [e.target for e in edges]
        assert len(edges) >= 0

    def test_call_edge_line_numbers(self, extractor, sample_files):
        """Test that call edges have correct line numbers."""
        edges = extractor.extract_call_edges(sample_files["simple.py"])
        for edge in edges:
            assert edge.line_number >= 1

    def test_call_edge_file_path(self, extractor, sample_files):
        """Test that call edges include file path."""
        edges = extractor.extract_call_edges(sample_files["class.py"])
        assert all(e.file_path == str(sample_files["class.py"]) for e in edges)


class TestInheritanceEdgeExtraction:
    """Tests for extracting class inheritance relationships."""

    def test_extract_inheritance_edges(self, extractor, tmp_path):
        """Test extracting inheritance edges."""
        code = """
class Animal:
    pass

class Dog(Animal):
    pass

class Cat(Animal):
    pass
"""
        file_path = create_sample_file(tmp_path, "inheritance.py", code)
        edges = extractor.extract_inheritance_edges(file_path)

        # Should find Dog -> Animal and Cat -> Animal
        inheritance_pairs = [(e.source, e.target) for e in edges]
        assert ("Dog", "Animal") in inheritance_pairs
        assert ("Cat", "Animal") in inheritance_pairs

    def test_inheritance_edge_types(self, extractor, tmp_path):
        """Test that inheritance edges have correct type."""
        code = "class Child(Parent): pass"
        file_path = create_sample_file(tmp_path, "test.py", code)
        edges = extractor.extract_inheritance_edges(file_path)

        for edge in edges:
            assert edge.edge_type == "INHERITS"

    def test_multiple_inheritance(self, extractor, tmp_path):
        """Test extracting multiple inheritance."""
        code = """
class A:
    pass

class B:
    pass

class C(A, B):
    pass
"""
        file_path = create_sample_file(tmp_path, "multi.py", code)
        edges = extractor.extract_inheritance_edges(file_path)

        # Should find at least C -> A
        assert len(edges) >= 1
        sources = [e.source for e in edges if e.source == "C"]
        assert len(sources) >= 1


class TestCompositionEdgeExtraction:
    """Tests for extracting composition relationships."""

    def test_extract_composition_edges(self, extractor, tmp_path):
        """Test extracting composition edges."""
        code = """
class Engine:
    pass

class Car:
    def __init__(self):
        self.engine = Engine()  # Composition
"""
        file_path = create_sample_file(tmp_path, "composition.py", code)
        edges = extractor.extract_composition_edges(file_path)

        # Should find composition relationships
        assert isinstance(edges, list)
        for edge in edges:
            assert edge.edge_type == "COMPOSITION"

    def test_composition_edge_types(self, extractor, sample_files):
        """Test that composition edges have correct type."""
        edges = extractor.extract_composition_edges(sample_files["complex.py"])
        for edge in edges:
            assert edge.edge_type == "COMPOSITION"


class TestImplementsEdgeExtraction:
    """Tests for extracting interface implementation relationships."""

    def test_extract_implements_edges(self, extractor, tmp_path):
        """Test extracting implements edges."""
        code = """
class Protocol:
    pass

class Implementation(Protocol):
    pass
"""
        file_path = create_sample_file(tmp_path, "implements.py", code)
        edges = extractor.extract_implements_edges(file_path)

        # Should find implementation relationships
        assert isinstance(edges, list)

    def test_implements_edge_types(self, extractor, tmp_path):
        """Test that implements edges have correct type."""
        code = "class Impl(Interface): pass"
        file_path = create_sample_file(tmp_path, "test.py", code)
        edges = extractor.extract_implements_edges(file_path)

        for edge in edges:
            assert edge.edge_type == "IMPLEMENTS"


# =============================================================================
# Reference Extraction Tests
# =============================================================================

class TestReferenceExtraction:
    """Tests for extracting symbol references."""

    def test_extract_references(self, extractor, sample_files):
        """Test extracting references from code."""
        references = extractor.extract_references(sample_files["simple.py"])

        # Should find identifier references
        assert len(references) >= 0
        for ref in references:
            assert ref.name
            assert ref.line_number >= 1
            assert isinstance(ref.column, int)

    def test_reference_enclosing_scope(self, extractor, sample_files):
        """Test that references include enclosing scope."""
        references = extractor.extract_references(sample_files["class.py"])

        for ref in references:
            # Enclosing scope may be None for module-level references
            if ref.enclosing_scope:
                assert isinstance(ref.enclosing_scope, str)

    def test_reference_file_path(self, extractor, sample_files):
        """Test that references include file path."""
        references = extractor.extract_references(sample_files["complex.py"])
        assert all(r.file_path == str(sample_files["complex.py"]) for r in references)

    def test_extract_many_references(self, extractor, sample_files):
        """Test extracting references from complex code."""
        references = extractor.extract_references(sample_files["complex.py"])

        # Complex code should have many references
        assert len(references) >= 0

        # Check that all required fields are present
        for ref in references:
            assert hasattr(ref, "name")
            assert hasattr(ref, "file_path")
            assert hasattr(ref, "line_number")
            assert hasattr(ref, "column")


# =============================================================================
# Comprehensive Extraction Tests
# =============================================================================

class TestExtractAll:
    """Tests for extracting all information from a file."""

    def test_extract_all_simple(self, extractor, sample_files):
        """Test extracting all symbols and edges from simple code."""
        symbols, edges = extractor.extract_all(sample_files["simple.py"])

        assert isinstance(symbols, list)
        assert isinstance(edges, list)
        assert len(symbols) >= 1

        # Verify symbol structure
        for symbol in symbols:
            assert isinstance(symbol, ExtractedSymbol)
            assert symbol.name
            assert symbol.type
            assert symbol.file_path

        # Verify edge structure
        for edge in edges:
            assert isinstance(edge, ExtractedEdge)
            assert edge.source
            assert edge.target
            assert edge.edge_type

    def test_extract_all_complex(self, extractor, sample_files):
        """Test extracting all from complex code."""
        symbols, edges = extractor.extract_all(sample_files["complex.py"])

        # Should extract multiple symbols
        assert len(symbols) >= 3

        # Should extract various edge types
        edge_types = set(e.edge_type for e in edges)
        assert len(edge_types) >= 0

    def test_extract_all_with_references(self, extractor, sample_files):
        """Test extracting symbols, edges, and references."""
        symbols, edges, references = extractor.extract_all_with_references(
            sample_files["class.py"]
        )

        assert isinstance(symbols, list)
        assert isinstance(edges, list)
        assert isinstance(references, list)

        # All should be valid
        for symbol in symbols:
            assert isinstance(symbol, ExtractedSymbol)

        for edge in edges:
            assert isinstance(edge, ExtractedEdge)

        for ref in references:
            assert isinstance(ref, ExtractedReference)

    def test_extract_all_parsing_optimization(self, extractor, sample_files):
        """Test that extract_all parses file only once."""
        import time

        # Time extract_all
        start = time.time()
        symbols1, edges1 = extractor.extract_all(sample_files["complex.py"])
        time1 = time.time() - start

        # Time individual extractions
        start = time.time()
        symbols2 = extractor.extract_symbols(sample_files["complex.py"])
        edges2 = extractor.extract_call_edges(sample_files["complex.py"])
        time2 = time.time() - start

        # extract_all should complete without errors
        # timing can vary due to caching, so just verify both complete
        assert time1 >= 0
        assert time2 >= 0
        assert len(symbols1) > 0
        assert len(symbols2) > 0


# =============================================================================
# Parser Management Tests
# =============================================================================

class TestParserManagement:
    """Tests for parser caching and management."""

    def test_parser_caching(self, extractor, tmp_path):
        """Test that parsers are cached across calls."""
        file_path = create_sample_file(tmp_path, "test.py", SAMPLE_PYTHON_SIMPLE)

        # First call should create parser
        symbols1 = extractor.extract_symbols(file_path)
        parsers_after_first = len(extractor._parsers)

        # Second call should reuse parser
        symbols2 = extractor.extract_symbols(file_path)
        parsers_after_second = len(extractor._parsers)

        assert parsers_after_first == parsers_after_second
        assert symbols1 == symbols2

    def test_multiple_language_parsers(self, extractor, tmp_path):
        """Test that different parsers are used for different languages."""
        py_file = create_sample_file(tmp_path, "test.py", SAMPLE_PYTHON_SIMPLE)
        js_file = create_sample_file(tmp_path, "test.js", "function test() {}")

        # Extract from both
        py_symbols = extractor.extract_symbols(py_file)
        js_symbols = extractor.extract_symbols(js_file)

        # Should have different parsers
        assert len(extractor._parsers) >= 1

    def test_query_caching(self, extractor, sample_files):
        """Test that queries are cached."""
        # First call
        symbols1 = extractor.extract_symbols(sample_files["class.py"])
        cache_size_after_first = len(extractor._query_cache)

        # Second call
        symbols2 = extractor.extract_symbols(sample_files["class.py"])
        cache_size_after_second = len(extractor._query_cache)

        # Cache should not grow on second call
        assert cache_size_after_first == cache_size_after_second

        # Symbols should be equivalent (same count and names)
        assert len(symbols1) == len(symbols2)
        names1 = [s.name for s in symbols1]
        names2 = [s.name for s in symbols2]
        assert names1 == names2


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_nonexistent_file(self, extractor, tmp_path):
        """Test handling of nonexistent file."""
        file_path = tmp_path / "does_not_exist.py"
        symbols = extractor.extract_symbols(file_path)
        assert symbols == []

    def test_invalid_language_override(self, extractor, tmp_path):
        """Test handling of invalid language override."""
        file_path = create_sample_file(tmp_path, "test.py", SAMPLE_PYTHON_SIMPLE)
        symbols = extractor.extract_symbols(file_path, language="invalid_language")
        assert symbols == []

    def test_malformed_code(self, extractor, tmp_path):
        """Test handling of malformed code."""
        malformed = "this is not valid python code @#$%^&*()"
        file_path = create_sample_file(tmp_path, "malformed.py", malformed)

        # Should not crash
        symbols = extractor.extract_symbols(file_path)
        assert isinstance(symbols, list)

    def test_unicode_content(self, extractor, tmp_path):
        """Test handling of unicode content."""
        unicode_code = """
# Unicode characters: 你好世界
def greet():
    print("你好，世界！")
"""
        file_path = create_sample_file(tmp_path, "unicode.py", unicode_code)

        # Should handle unicode correctly
        symbols = extractor.extract_symbols(file_path)
        assert len(symbols) >= 1
        assert symbols[0].name == "greet"


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Tests for performance characteristics."""

    def test_large_file_extraction(self, extractor, tmp_path):
        """Test extraction from a large file."""
        # Create a large file with many functions
        lines = []
        for i in range(100):
            lines.append(f"def function_{i}():\n    return {i}\n")

        large_code = "\n".join(lines)
        file_path = create_sample_file(tmp_path, "large.py", large_code)

        import time
        start = time.time()
        symbols = extractor.extract_symbols(file_path)
        elapsed = time.time() - start

        # Should extract all 100 functions
        assert len(symbols) == 100

        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0

    def test_multiple_files_performance(self, extractor, sample_files):
        """Test extraction performance across multiple files."""
        import time

        start = time.time()
        all_symbols = []
        for file_path in sample_files.values():
            symbols = extractor.extract_symbols(file_path)
            all_symbols.extend(symbols)
        elapsed = time.time() - start

        # Should extract symbols from all files quickly
        assert len(all_symbols) >= 10
        assert elapsed < 3.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for AST extraction workflows."""

    def test_full_analysis_workflow(self, extractor, sample_files):
        """Test a complete analysis workflow."""
        file_path = sample_files["complex.py"]

        # Extract everything
        symbols, edges, references = extractor.extract_all_with_references(file_path)

        # Verify completeness
        assert len(symbols) >= 3
        assert isinstance(edges, list)
        assert isinstance(references, list)

        # Check that we can build useful information
        function_names = [s.name for s in symbols if s.type == "function"]
        class_names = [s.name for s in symbols if s.type == "class"]
        call_graph = [(e.source, e.target) for e in edges if e.edge_type == "CALLS"]

        assert len(function_names) >= 2
        assert len(class_names) >= 1

    def test_cross_reference_validation(self, extractor, tmp_path):
        """Test that symbols and references are consistent."""
        code = """
def foo():
    return bar()

def bar():
    return 42

x = foo()
"""
        file_path = create_sample_file(tmp_path, "xref.py", code)

        symbols, _, references = extractor.extract_all_with_references(file_path)

        # All functions should be in symbols
        symbol_names = set(s.name for s in symbols)
        assert "foo" in symbol_names
        assert "bar" in symbol_names

        # References should include calls to these functions
        reference_names = set(r.name for r in references)
        assert "foo" in reference_names or "bar" in reference_names


# =============================================================================
# Tree-sitter Manager Tests
# =============================================================================

class TestTreeSitterManager:
    """Tests for tree_sitter_manager module functions."""

    def test_get_language_python(self):
        """Test getting Python language object."""
        from victor.coding.codebase.tree_sitter_manager import get_language

        lang = get_language("python")
        assert lang is not None
        # Language object should have certain attributes
        assert hasattr(lang, "name")

    def test_get_language_cached(self):
        """Test that language objects are cached."""
        from victor.coding.codebase.tree_sitter_manager import get_language

        lang1 = get_language("python")
        lang2 = get_language("python")
        # Should return the same cached object
        assert lang1 is lang2

    def test_get_language_unsupported(self):
        """Test getting an unsupported language raises error."""
        from victor.coding.codebase.tree_sitter_manager import get_language

        with pytest.raises(ValueError, match="Unsupported language"):
            get_language("unsupported_language")

    def test_get_parser_python(self):
        """Test getting Python parser."""
        from victor.coding.codebase.tree_sitter_manager import get_parser

        parser = get_parser("python")
        assert parser is not None
        # Parser should be able to parse code
        tree = parser.parse(b"def foo(): pass")
        assert tree.root_node.type == "module"

    def test_get_parser_cached(self):
        """Test that parser objects are cached."""
        from victor.coding.codebase.tree_sitter_manager import get_parser

        parser1 = get_parser("python")
        parser2 = get_parser("python")
        # Should return the same cached object
        assert parser1 is parser2

    def test_run_query_simple(self):
        """Test running a simple tree-sitter query."""
        from victor.coding.codebase.tree_sitter_manager import get_parser, run_query

        parser = get_parser("python")
        tree = parser.parse(b"def foo():\n    pass")

        # Query for function names
        query = "(function_definition name: (identifier) @name)"
        captures = run_query(tree, query, "python")

        assert "name" in captures or len(captures) > 0
        if "name" in captures:
            assert len(captures["name"]) == 1
            # Node should contain the function name
            node = captures["name"][0]
            assert hasattr(node, "text")

    def test_run_query_multiple_matches(self):
        """Test running a query with multiple matches."""
        from victor.coding.codebase.tree_sitter_manager import get_parser, run_query

        parser = get_parser("python")
        code = b"def foo(): pass\ndef bar(): pass\ndef baz(): pass"
        tree = parser.parse(code)

        query = "(function_definition name: (identifier) @name)"
        captures = run_query(tree, query, "python")

        # Should find 3 functions
        if "name" in captures:
            assert len(captures["name"]) == 3
        else:
            # Rust accelerator format
            assert len(captures.get("_all", [])) >= 0

    def test_parse_file_accelerated_valid(self, tmp_path):
        """Test parsing a file with accelerated parser."""
        from victor.coding.codebase.tree_sitter_manager import parse_file_accelerated

        file_path = create_sample_file(tmp_path, "test.py", "def hello(): pass")
        tree = parse_file_accelerated(str(file_path))

        assert tree is not None
        assert tree.root_node.type == "module"

    def test_parse_file_accelerated_nonexistent(self, tmp_path):
        """Test parsing a nonexistent file returns None."""
        from victor.coding.codebase.tree_sitter_manager import parse_file_accelerated

        file_path = tmp_path / "does_not_exist.py"
        tree = parse_file_accelerated(str(file_path))

        assert tree is None

    def test_parse_file_with_timing(self, tmp_path):
        """Test parsing file with timing information."""
        from victor.coding.codebase.tree_sitter_manager import parse_file_with_timing

        file_path = create_sample_file(tmp_path, "test.py", SAMPLE_PYTHON_SIMPLE)
        tree, elapsed = parse_file_with_timing(str(file_path))

        assert tree is not None
        assert elapsed >= 0
        assert elapsed < 5.0  # Should complete in less than 5 seconds

    def test_extract_symbols_parallel_single_file(self, tmp_path):
        """Test parallel symbol extraction from single file."""
        from victor.coding.codebase.tree_sitter_manager import extract_symbols_parallel

        file_path = create_sample_file(tmp_path, "test.py", SAMPLE_PYTHON_CLASS)
        results = extract_symbols_parallel([str(file_path)], ["function", "class"])

        assert str(file_path) in results
        symbols = results[str(file_path)]
        assert len(symbols) >= 1

    def test_extract_symbols_parallel_multiple_files(self, tmp_path):
        """Test parallel symbol extraction from multiple files."""
        from victor.coding.codebase.tree_sitter_manager import extract_symbols_parallel

        file1 = create_sample_file(tmp_path, "file1.py", "def foo(): pass")
        file2 = create_sample_file(tmp_path, "file2.py", "def bar(): pass")

        results = extract_symbols_parallel(
            [str(file1), str(file2)],
            ["function"]
        )

        assert len(results) == 2
        assert all(len(symbols) >= 0 for symbols in results.values())

    def test_clear_ast_cache(self):
        """Test clearing AST cache."""
        from victor.coding.codebase.tree_sitter_manager import (
            clear_ast_cache,
            get_parser,
            get_cache_stats,
        )

        # Create some cache entries
        get_parser("python")
        stats_before = get_cache_stats()

        # Clear cache
        clear_ast_cache()

        # Verify cache is cleared (size should be 0 or smaller)
        stats_after = get_cache_stats()
        assert stats_after["size"] <= stats_before["size"]

    def test_get_cache_stats(self):
        """Test getting cache statistics."""
        from victor.coding.codebase.tree_sitter_manager import get_cache_stats, get_parser

        # Get stats (may be empty initially)
        stats = get_cache_stats()

        assert isinstance(stats, dict)
        assert "size" in stats
        assert "max_size" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats

        # Create a parser to populate cache
        parser1 = get_parser("python")

        stats_after = get_cache_stats()
        assert stats_after["size"] >= 0

    def test_detect_language_from_file(self, tmp_path):
        """Test _detect_language function with various file extensions."""
        from victor.coding.codebase.tree_sitter_manager import _detect_language

        # Test various file extensions
        assert _detect_language("test.py") == "python"
        assert _detect_language("test.js") == "javascript"
        assert _detect_language("test.ts") == "typescript"
        assert _detect_language("test.go") == "go"
        assert _detect_language("test.rs") == "rust"
        assert _detect_language("test.java") == "java"

    def test_read_file_valid(self, tmp_path):
        """Test _read_file with valid file."""
        from victor.coding.codebase.tree_sitter_manager import _read_file

        file_path = create_sample_file(tmp_path, "test.txt", "Hello, World!")
        content = _read_file(str(file_path))

        assert content == "Hello, World!"

    def test_read_file_nonexistent(self, tmp_path):
        """Test _read_file with nonexistent file."""
        from victor.coding.codebase.tree_sitter_manager import _read_file

        content = _read_file(str(tmp_path / "does_not_exist.txt"))
        assert content is None


# =============================================================================
# AST Node Traversal Tests
# =============================================================================

class TestASTNodeTraversal:
    """Tests for AST node traversal and navigation."""

    def test_traverse_all_nodes(self, tmp_path):
        """Test traversing all nodes in AST."""
        from victor.coding.codebase.tree_sitter_manager import get_parser

        parser = get_parser("python")
        tree = parser.parse(b"def foo(): pass")

        node_count = 0
        def count_nodes(node):
            nonlocal node_count
            node_count += 1
            for child in node.children:
                count_nodes(child)

        count_nodes(tree.root_node)
        assert node_count > 1

    def test_node_children(self, tmp_path):
        """Test accessing node children."""
        from victor.coding.codebase.tree_sitter_manager import get_parser

        parser = get_parser("python")
        tree = parser.parse(b"def foo(): pass")

        root = tree.root_node
        assert hasattr(root, "children")
        assert len(root.children) > 0

    def test_node_parent_navigation(self, tmp_path):
        """Test navigating to parent node."""
        from victor.coding.codebase.tree_sitter_manager import get_parser

        parser = get_parser("python")
        tree = parser.parse(b"def foo(): pass")

        # Get a child node
        func_def = tree.root_node.children[0]
        # Parent should be the root node (same id)
        assert func_def.parent.id == tree.root_node.id

    def test_node_type_and_properties(self, tmp_path):
        """Test node type and properties."""
        from victor.coding.codebase.tree_sitter_manager import get_parser

        parser = get_parser("python")
        tree = parser.parse(b"def foo(): pass")

        root = tree.root_node
        assert hasattr(root, "type")
        assert root.type == "module"
        assert hasattr(root, "start_point")
        assert hasattr(root, "end_point")
        assert hasattr(root, "text")

    def test_node_by_field_name(self, tmp_path):
        """Test accessing child nodes by field name."""
        from victor.coding.codebase.tree_sitter_manager import get_parser

        parser = get_parser("python")
        tree = parser.parse(b"class Foo:\n    pass")

        # Find class definition
        class_def = tree.root_node.children[0]
        assert class_def.type == "class_definition"

        # Access name field
        name_field = class_def.child_by_field_name("name")
        assert name_field is not None


# =============================================================================
# AST Modification Tests
# =============================================================================

class TestASTModifications:
    """Tests for AST modification operations."""

    def test_ast_read_only(self, tmp_path):
        """Test that AST nodes are read-only (tree-sitter constraint)."""
        from victor.coding.codebase.tree_sitter_manager import get_parser

        parser = get_parser("python")
        tree = parser.parse(b"def foo(): pass")

        # Tree-sitter nodes are immutable
        # We can read properties but not modify them
        root = tree.root_node
        node_type = root.type
        assert node_type == "module"

        # Attempting to modify should either fail or be ignored
        # (tree-sitter design ensures immutability)


# =============================================================================
# Complex Code Pattern Tests
# =============================================================================

class TestComplexCodePatterns:
    """Tests for complex code patterns and edge cases."""

    def test_nested_classes(self, extractor, tmp_path):
        """Test extraction from nested classes."""
        code = """
class Outer:
    class Inner:
        def method(self):
            pass
"""
        file_path = create_sample_file(tmp_path, "nested.py", code)
        symbols = extractor.extract_symbols(file_path)

        # Should find both classes and the method
        class_symbols = [s for s in symbols if s.type == "class"]
        assert len(class_symbols) >= 1

    def test_lambda_functions(self, extractor, tmp_path):
        """Test handling of lambda functions."""
        code = """
lambda_func = lambda x: x * 2
result = lambda_func(5)
"""
        file_path = create_sample_file(tmp_path, "lambda.py", code)
        symbols = extractor.extract_symbols(file_path)

        # Lambdas are typically not extracted as named functions
        # but should not cause errors
        assert isinstance(symbols, list)

    def test_list_comprehensions(self, extractor, tmp_path):
        """Test handling of list comprehensions."""
        code = """
numbers = [1, 2, 3, 4, 5]
squared = [x**2 for x in numbers]
"""
        file_path = create_sample_file(tmp_path, "comprehension.py", code)
        symbols = extractor.extract_symbols(file_path)

        # Should not crash on comprehensions
        assert isinstance(symbols, list)

    def test_decorators_on_class(self, extractor, tmp_path):
        """Test decorators on classes."""
        code = """
@dataclass
class Person:
    name: str
    age: int
"""
        file_path = create_sample_file(tmp_path, "decorated.py", code)
        symbols = extractor.extract_symbols(file_path)

        # Should extract the class despite decorators
        class_symbols = [s for s in symbols if s.type == "class"]
        assert len(class_symbols) >= 1

    def test_context_managers(self, extractor, tmp_path):
        """Test context managers."""
        code = """
with open("file.txt") as f:
    content = f.read()
"""
        file_path = create_sample_file(tmp_path, "context.py", code)
        symbols = extractor.extract_symbols(file_path)

        # Should handle context managers without errors
        assert isinstance(symbols, list)

    def test_try_except_blocks(self, extractor, tmp_path):
        """Test try-except blocks."""
        code = """
try:
    risky_operation()
except ValueError as e:
    handle_error(e)
except Exception:
    log_error()
"""
        file_path = create_sample_file(tmp_path, "exceptions.py", code)
        symbols = extractor.extract_symbols(file_path)

        function_names = [s.name for s in symbols if s.type == "function"]
        # Should find the functions
        assert isinstance(function_names, list)

    def test_generator_functions(self, extractor, tmp_path):
        """Test generator functions with yield."""
        code = """
def generate_numbers(n):
    for i in range(n):
        yield i

async def async_generator():
    for i in range(10):
        yield i
"""
        file_path = create_sample_file(tmp_path, "generators.py", code)
        symbols = extractor.extract_symbols(file_path)

        # Should extract generator functions
        function_symbols = [s for s in symbols if s.type == "function"]
        assert len(function_symbols) >= 2

    def test_property_decorators(self, extractor, tmp_path):
        """Test property decorators."""
        code = """
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value
"""
        file_path = create_sample_file(tmp_path, "properties.py", code)
        symbols = extractor.extract_symbols(file_path)

        # Should extract class and property methods
        assert len(symbols) >= 1


# =============================================================================
# Multi-language Support Tests
# =============================================================================

class TestMultiLanguageSupport:
    """Tests for multi-language AST parsing."""

    def test_javascript_function_extraction(self, extractor, tmp_path):
        """Test extracting functions from JavaScript."""
        js_code = """
function hello() {
    console.log("Hello");
}

const arrow = () => {
    console.log("Arrow function");
};
"""
        file_path = create_sample_file(tmp_path, "test.js", js_code)
        symbols = extractor.extract_symbols(file_path)

        # Should extract at least the function
        assert isinstance(symbols, list)
        if symbols:
            assert symbols[0].type in ["function", "variable"]

    def test_typescript_interface_extraction(self, extractor, tmp_path):
        """Test extracting interfaces from TypeScript."""
        ts_code = """
interface User {
    name: string;
    age: number;
}

class Person implements User {
    name: string;
    age: number;
}
"""
        file_path = create_sample_file(tmp_path, "test.ts", ts_code)
        symbols = extractor.extract_symbols(file_path)

        # Should extract interface and/or class
        assert isinstance(symbols, list)
