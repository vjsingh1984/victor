# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for graph edge extraction — plugin-first query pattern.

Ensures that _extract_calls, _extract_inheritance, _extract_implements,
_extract_composition, and _find_enclosing_symbol_name all consult
language plugin queries before falling back to static dictionaries.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from victor.coding.codebase.graph.registry import create_graph_store
from victor.coding.codebase.indexer import CodebaseIndex, _process_file_parallel


def _skip_if_no_parser(language: str):
    """Skip test if tree-sitter parser is not available for *language*."""
    try:
        from victor.coding.codebase.tree_sitter_manager import get_parser
    except Exception:
        pytest.skip("tree-sitter not available")
    try:
        parser = get_parser(language)
    except Exception:
        pytest.skip(f"{language} parser not available")
    if parser is None:
        pytest.skip(f"{language} parser not available")


# ---------------------------------------------------------------------------
# 1. Signature tests — each method attempts plugin lookup before static dict
# ---------------------------------------------------------------------------


class TestPluginFirstLookup:
    """Verify that extraction methods consult the language plugin registry."""

    def _make_index(self, tmp_path: Path) -> CodebaseIndex:
        return CodebaseIndex(
            tmp_path,
            graph_store=create_graph_store("memory", Path(":memory:")),
            enable_watcher=False,
        )

    def test_extract_calls_uses_plugin(self, tmp_path: Path):
        """_extract_calls_with_tree_sitter should query plugin.calls first."""
        _skip_if_no_parser("javascript")
        src = tmp_path / "a.js"
        src.write_text("function f() { g(); }\nfunction g() {}\n")

        index = self._make_index(tmp_path)
        mock_plugin = MagicMock()
        mock_plugin.tree_sitter_queries.calls = None  # force fallback
        with patch.object(index._language_registry, "get", return_value=mock_plugin) as mock_get:
            index._extract_calls_with_tree_sitter(src, "javascript")
            mock_get.assert_called_with("javascript")

    def test_find_enclosing_uses_plugin(self, tmp_path: Path):
        """_find_enclosing_symbol_name should query plugin.enclosing_scopes."""
        index = self._make_index(tmp_path)
        mock_plugin = MagicMock()
        mock_plugin.tree_sitter_queries.enclosing_scopes = []
        with patch.object(index._language_registry, "get", return_value=mock_plugin) as mock_get:
            dummy_node = MagicMock()
            dummy_node.parent = None
            index._find_enclosing_symbol_name(dummy_node, "rust")
            mock_get.assert_called_with("rust")

    def test_extract_inheritance_uses_plugin(self, tmp_path: Path):
        """_extract_inheritance should query plugin.inheritance first."""
        _skip_if_no_parser("javascript")
        src = tmp_path / "b.js"
        src.write_text("class A extends B {}\n")

        index = self._make_index(tmp_path)
        mock_plugin = MagicMock()
        mock_plugin.tree_sitter_queries.inheritance = None
        with patch.object(index._language_registry, "get", return_value=mock_plugin) as mock_get:
            index._extract_inheritance(src, "javascript", [])
            mock_get.assert_called_with("javascript")

    def test_extract_implements_uses_plugin(self, tmp_path: Path):
        """_extract_implements should query plugin.implements first."""
        index = self._make_index(tmp_path)
        mock_plugin = MagicMock()
        mock_plugin.tree_sitter_queries.implements = None
        with patch.object(index._language_registry, "get", return_value=mock_plugin) as mock_get:
            src = tmp_path / "c.ts"
            src.write_text("class A implements B {}\n")
            index._extract_implements(src, "typescript", [])
            mock_get.assert_called_with("typescript")

    def test_extract_composition_uses_plugin(self, tmp_path: Path):
        """_extract_composition should query plugin.composition first."""
        index = self._make_index(tmp_path)
        mock_plugin = MagicMock()
        mock_plugin.tree_sitter_queries.composition = None
        with patch.object(index._language_registry, "get", return_value=mock_plugin) as mock_get:
            src = tmp_path / "d.go"
            src.write_text("package main\ntype Foo struct { bar Baz }\n")
            index._extract_composition(src, "go", [])
            mock_get.assert_called_with("go")


# ---------------------------------------------------------------------------
# 2. Rust edge extraction tests
# ---------------------------------------------------------------------------


class TestRustEdgeExtraction:
    """Verify that Rust code now produces CALLS, IMPLEMENTS, COMPOSITION edges."""

    def _make_index(self, tmp_path: Path) -> CodebaseIndex:
        return CodebaseIndex(
            tmp_path,
            graph_store=create_graph_store("memory", Path(":memory:")),
            enable_watcher=False,
        )

    def test_rust_call_extraction(self, tmp_path: Path):
        """Rust function calls should produce call edges via plugin queries."""
        _skip_if_no_parser("rust")
        src = tmp_path / "main.rs"
        src.write_text(
            "fn caller() {\n    callee();\n}\nfn callee() {}\n",
            encoding="utf-8",
        )

        index = self._make_index(tmp_path)
        edges = index._extract_calls_with_tree_sitter(src, "rust")
        callees = [callee for _, callee in edges]
        assert "callee" in callees, f"Expected 'callee' in call edges, got {edges}"

    def test_rust_implements_extraction(self, tmp_path: Path):
        """Rust `impl Trait for Struct` should produce implements edges."""
        _skip_if_no_parser("rust")
        src = tmp_path / "lib.rs"
        src.write_text(
            "trait Drawable {}\nstruct Circle;\nimpl Drawable for Circle {}\n",
            encoding="utf-8",
        )

        index = self._make_index(tmp_path)
        edges = index._extract_implements(src, "rust", [])
        # Rust plugin captures: @interface=Drawable, @child=Circle
        assert len(edges) > 0, f"Expected implements edges, got {edges}"
        children = [child for child, _ in edges]
        interfaces = [iface for _, iface in edges]
        assert "Circle" in children, f"Expected Circle in children, got {edges}"
        assert "Drawable" in interfaces, f"Expected Drawable in interfaces, got {edges}"

    def test_rust_composition_extraction(self, tmp_path: Path):
        """Rust struct with typed fields should produce composition edges."""
        _skip_if_no_parser("rust")
        src = tmp_path / "model.rs"
        src.write_text(
            "struct Engine;\nstruct Car {\n    engine: Engine,\n}\n",
            encoding="utf-8",
        )

        index = self._make_index(tmp_path)
        edges = index._extract_composition(src, "rust", [])
        assert len(edges) > 0, f"Expected composition edges, got {edges}"
        owners = [owner for owner, _ in edges]
        types = [t for _, t in edges]
        assert "Car" in owners, f"Expected Car in owners, got {edges}"
        assert "Engine" in types, f"Expected Engine in types, got {edges}"

    def test_rust_enclosing_scope(self, tmp_path: Path):
        """_find_enclosing_symbol_name should resolve Rust function names."""
        _skip_if_no_parser("rust")
        from victor.coding.codebase.tree_sitter_manager import get_parser
        from tree_sitter import Query, QueryCursor

        src = tmp_path / "scope.rs"
        src.write_text(
            "fn my_func() {\n    helper();\n}\n",
            encoding="utf-8",
        )

        parser = get_parser("rust")
        tree = parser.parse(src.read_bytes())
        # Find the call expression node for helper()
        call_query = Query(
            parser.language,
            "(call_expression function: (identifier) @callee)",
        )
        cursor = QueryCursor(call_query)
        captures_dict = cursor.captures(tree.root_node)
        callee_nodes = captures_dict.get("callee", [])
        assert len(callee_nodes) > 0, "Should find at least one call expression"

        call_node = callee_nodes[0]
        index = self._make_index(tmp_path)
        enclosing = index._find_enclosing_symbol_name(call_node, "rust")
        assert enclosing == "my_func", f"Expected 'my_func', got {enclosing!r}"


# ---------------------------------------------------------------------------
# 3. JS/TS import extraction tests
# ---------------------------------------------------------------------------


class TestImportExtraction:
    """Verify multi-language import extraction."""

    def _make_index(self, tmp_path: Path) -> CodebaseIndex:
        return CodebaseIndex(
            tmp_path,
            graph_store=create_graph_store("memory", Path(":memory:")),
            enable_watcher=False,
        )

    def test_js_import_statement(self, tmp_path: Path):
        """ES6 import statements should be extracted."""
        _skip_if_no_parser("javascript")
        src = tmp_path / "app.js"
        src.write_text(
            "import { foo } from 'bar';\nimport baz from './baz';\n",
            encoding="utf-8",
        )

        index = self._make_index(tmp_path)
        imports = index._extract_imports_with_tree_sitter(src, "javascript")
        assert "bar" in imports, f"Expected 'bar' in imports, got {imports}"
        assert "./baz" in imports, f"Expected './baz' in imports, got {imports}"

    def test_js_require(self, tmp_path: Path):
        """CommonJS require() calls should be extracted."""
        _skip_if_no_parser("javascript")
        src = tmp_path / "lib.js"
        src.write_text(
            "const x = require('lodash');\n",
            encoding="utf-8",
        )

        index = self._make_index(tmp_path)
        imports = index._extract_imports_with_tree_sitter(src, "javascript")
        assert "lodash" in imports, f"Expected 'lodash' in imports, got {imports}"

    def test_rust_use_declaration(self, tmp_path: Path):
        """Rust use declarations should be extracted."""
        _skip_if_no_parser("rust")
        src = tmp_path / "lib.rs"
        src.write_text(
            "use std::collections::HashMap;\nuse crate::config;\n",
            encoding="utf-8",
        )

        index = self._make_index(tmp_path)
        imports = index._extract_imports_with_tree_sitter(src, "rust")
        assert len(imports) > 0, f"Expected Rust use imports, got {imports}"
        # At least one should contain std or crate
        combined = " ".join(imports)
        assert "std" in combined or "HashMap" in combined, (
            f"Expected std import captured, got {imports}"
        )


# ---------------------------------------------------------------------------
# 4. Full indexer integration — edges produced end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rust_full_index_edges(tmp_path: Path):
    """Full indexing of a Rust file should produce non-zero relationship edges."""
    _skip_if_no_parser("rust")
    src = tmp_path / "main.rs"
    src.write_text(
        """\
trait Printable {}
struct Point {
    x: i32,
}
impl Printable for Point {}
fn render() {
    draw();
}
fn draw() {}
""",
        encoding="utf-8",
    )

    index = CodebaseIndex(
        tmp_path,
        graph_store=create_graph_store("memory", Path(":memory:")),
        enable_watcher=False,
    )
    await index.index_codebase()

    rel = str(src.relative_to(tmp_path))
    assert rel in index.files
    metadata = index.files[rel]

    # Should have call edges (render -> draw)
    callees = [callee for _, callee in metadata.call_edges]
    assert "draw" in callees, f"Expected CALLS edge to 'draw', got {metadata.call_edges}"

    # Should have implements edges (Point implements Printable)
    assert len(metadata.implements_edges) > 0, (
        f"Expected IMPLEMENTS edges, got {metadata.implements_edges}"
    )


@pytest.mark.asyncio
async def test_js_full_index_imports(tmp_path: Path):
    """Full indexing of a JS file should produce import entries."""
    _skip_if_no_parser("javascript")
    src = tmp_path / "index.js"
    src.write_text(
        "import { useState } from 'react';\nfunction App() { useState(); }\n",
        encoding="utf-8",
    )

    index = CodebaseIndex(
        tmp_path,
        graph_store=create_graph_store("memory", Path(":memory:")),
        enable_watcher=False,
    )
    await index.index_codebase()

    rel = str(src.relative_to(tmp_path))
    assert rel in index.files
    metadata = index.files[rel]
    assert "react" in metadata.imports, f"Expected 'react' in imports, got {metadata.imports}"


# ---------------------------------------------------------------------------
# 5. Parallel path tests — _process_file_parallel
# ---------------------------------------------------------------------------


class TestParallelPathExtraction:
    """Verify that _process_file_parallel extracts all edge types (not just symbols)."""

    def test_parallel_rust_call_edges(self, tmp_path: Path):
        """Parallel path should extract Rust call edges via plugin queries."""
        _skip_if_no_parser("rust")
        src = tmp_path / "main.rs"
        src.write_text("fn caller() {\n    callee();\n}\nfn callee() {}\n")

        result = _process_file_parallel(str(src), str(tmp_path), "rust")
        assert result is not None
        callees = [callee for _, callee in result["call_edges"]]
        assert "callee" in callees, f"Expected 'callee' in call_edges, got {result['call_edges']}"

    def test_parallel_rust_implements_edges(self, tmp_path: Path):
        """Parallel path should extract Rust implements edges."""
        _skip_if_no_parser("rust")
        src = tmp_path / "lib.rs"
        src.write_text("trait Drawable {}\nstruct Circle;\nimpl Drawable for Circle {}\n")

        result = _process_file_parallel(str(src), str(tmp_path), "rust")
        assert result is not None
        assert len(result["implements_edges"]) > 0, (
            f"Expected implements_edges, got {result['implements_edges']}"
        )

    def test_parallel_rust_composition_edges(self, tmp_path: Path):
        """Parallel path should extract Rust composition edges."""
        _skip_if_no_parser("rust")
        src = tmp_path / "model.rs"
        src.write_text("struct Engine;\nstruct Car {\n    engine: Engine,\n}\n")

        result = _process_file_parallel(str(src), str(tmp_path), "rust")
        assert result is not None
        assert len(result["compose_edges"]) > 0, (
            f"Expected compose_edges, got {result['compose_edges']}"
        )

    def test_parallel_js_imports(self, tmp_path: Path):
        """Parallel path should extract JS import statements."""
        _skip_if_no_parser("javascript")
        src = tmp_path / "app.js"
        src.write_text("import { foo } from 'bar';\nconst x = require('lodash');\n")

        result = _process_file_parallel(str(src), str(tmp_path), "javascript")
        assert result is not None
        assert "bar" in result["imports"], f"Expected 'bar' in imports, got {result['imports']}"
        assert "lodash" in result["imports"], (
            f"Expected 'lodash' in imports, got {result['imports']}"
        )

    def test_parallel_rust_imports(self, tmp_path: Path):
        """Parallel path should extract Rust use declarations."""
        _skip_if_no_parser("rust")
        src = tmp_path / "lib.rs"
        src.write_text("use std::collections::HashMap;\nuse crate::config;\n")

        result = _process_file_parallel(str(src), str(tmp_path), "rust")
        assert result is not None
        assert len(result["imports"]) > 0, f"Expected imports, got {result['imports']}"

    def test_parallel_rust_references(self, tmp_path: Path):
        """Parallel path should extract Rust references via plugin queries."""
        _skip_if_no_parser("rust")
        src = tmp_path / "main.rs"
        src.write_text("fn main() {\n    let x = HashMap::new();\n}\n")

        result = _process_file_parallel(str(src), str(tmp_path), "rust")
        assert result is not None
        assert len(result["references"]) > 0, (
            f"Expected references, got {result['references']}"
        )

    def test_parallel_js_inheritance(self, tmp_path: Path):
        """Parallel path should extract JS inheritance edges."""
        _skip_if_no_parser("javascript")
        src = tmp_path / "cls.js"
        src.write_text("class Animal {}\nclass Dog extends Animal {}\n")

        result = _process_file_parallel(str(src), str(tmp_path), "javascript")
        assert result is not None
        assert len(result["inherit_edges"]) > 0, (
            f"Expected inherit_edges, got {result['inherit_edges']}"
        )
