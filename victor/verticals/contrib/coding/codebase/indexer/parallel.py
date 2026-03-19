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

"""Module-level functions for parallel file indexing.

These are module-level (not methods) so they can be pickled for use
with ProcessPoolExecutor. Provides 3-8x speedup on multi-core systems.
"""

import ast
import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from victor.verticals.contrib.coding.codebase.indexer.queries import (
    CALL_QUERIES,
    COMPOSITION_QUERIES,
    IMPLEMENTS_QUERIES,
    INHERITS_QUERIES,
    REFERENCE_QUERIES,
    SYMBOL_QUERIES,
)

logger = logging.getLogger(__name__)


def _get_plugin_query(language: str, field: str) -> Optional[str]:
    """Get a tree-sitter query from the language plugin, or None if unavailable.

    Safe to call from subprocesses -- imports the registry on demand and
    discovers plugins if needed.
    """
    try:
        from victor.verticals.contrib.coding.languages.registry import get_language_registry

        registry = get_language_registry()
        if not registry._plugins:
            registry.discover_plugins()
        plugin = registry.get(language)
        if plugin:
            value = getattr(plugin.tree_sitter_queries, field, None)
            if value:
                return value
    except Exception:
        pass
    return None


def _get_plugin_enclosing_scopes(language: str) -> List[Tuple[str, str]]:
    """Get enclosing scope definitions from language plugin, or empty list."""
    try:
        from victor.verticals.contrib.coding.languages.registry import get_language_registry

        registry = get_language_registry()
        if not registry._plugins:
            registry.discover_plugins()
        plugin = registry.get(language)
        if plugin and plugin.tree_sitter_queries.enclosing_scopes:
            return plugin.tree_sitter_queries.enclosing_scopes
    except Exception:
        pass
    return []


# Tree-sitter import queries for non-Python languages (used by parallel path).
_PARALLEL_IMPORT_QUERIES: Dict[str, str] = {
    "javascript": """
        (import_statement source: (string) @source)
        (call_expression
            function: (identifier) @_fn
            arguments: (arguments (string) @source)
            (#eq? @_fn "require"))
    """,
    "typescript": """
        (import_statement source: (string) @source)
        (call_expression
            function: (identifier) @_fn
            arguments: (arguments (string) @source)
            (#eq? @_fn "require"))
    """,
    "rust": """
        (use_declaration argument: (_) @source)
    """,
    "go": """
        (import_spec path: (interpreted_string_literal) @source)
    """,
    "java": """
        (import_declaration (scoped_identifier) @source)
    """,
}


def _process_file_parallel(
    file_path_str: str,
    root_str: str,
    language: str,
) -> Optional[Dict[str, Any]]:
    """Process a single file for indexing in a subprocess.

    This is a module-level function (not a method) so it can be pickled
    for use with ProcessPoolExecutor.

    Uses plugin-first query lookup for all extraction types, falling back
    to static dictionaries for languages not yet migrated to plugins.

    Args:
        file_path_str: Absolute path to the file
        root_str: Absolute path to the codebase root
        language: Detected language for the file

    Returns:
        Dictionary with extracted file data, or None on error
    """
    import ast as py_ast
    from pathlib import Path

    file_path = Path(file_path_str)
    root = Path(root_str)

    try:
        stat = file_path.stat()
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        return None

    # Extract symbols using tree-sitter
    symbols_data: List[Dict[str, Any]] = []
    call_edges: List[Tuple[str, str]] = []
    imports: List[str] = []
    inherit_edges: List[Tuple[str, str]] = []
    implements_edges: List[Tuple[str, str]] = []
    compose_edges: List[Tuple[str, str]] = []
    references: List[str] = []

    # Tree-sitter symbol extraction
    try:
        from victor.verticals.contrib.coding.codebase.tree_sitter_manager import get_parser
        from tree_sitter import Query, QueryCursor

        parser = get_parser(language)
        if parser:
            content_bytes = file_path.read_bytes()
            tree = parser.parse(content_bytes)

            # Symbol extraction -- plugin-first, then static fallback
            query_defs = SYMBOL_QUERIES.get(language, [])
            if not query_defs:
                try:
                    from victor.verticals.contrib.coding.languages.registry import (
                        get_language_registry,
                    )

                    _reg = get_language_registry()
                    if not _reg._plugins:
                        _reg.discover_plugins()
                    _plugin = _reg.get(language)
                    if _plugin and _plugin.tree_sitter_queries.symbols:
                        query_defs = [
                            (qp.symbol_type, qp.query) for qp in _plugin.tree_sitter_queries.symbols
                        ]
                except Exception:
                    pass
            for sym_type, query_src in query_defs:
                try:
                    query = Query(parser.language, query_src)
                    cursor = QueryCursor(query)
                    captures_dict = cursor.captures(tree.root_node)

                    name_nodes = captures_dict.get("name", [])
                    def_nodes = captures_dict.get("def", [])

                    def_by_start_line = {}
                    for def_node in def_nodes:
                        def_by_start_line[def_node.start_point[0]] = def_node

                    for node in name_nodes:
                        text = node.text.decode("utf-8", errors="ignore")
                        if text:
                            name_line = node.start_point[0]
                            end_line = node.end_point[0] + 1

                            for def_start, def_node in def_by_start_line.items():
                                if def_start <= name_line <= def_node.end_point[0]:
                                    end_line = def_node.end_point[0] + 1
                                    break

                            symbols_data.append(
                                {
                                    "name": text,
                                    "type": sym_type,
                                    "file_path": str(file_path.relative_to(root)),
                                    "line_number": name_line + 1,
                                    "end_line": end_line,
                                }
                            )
                except Exception:
                    continue

            # Call edge extraction
            call_query_src = _get_plugin_query(language, "calls")
            if not call_query_src:
                call_query_src = CALL_QUERIES.get(language)
            if call_query_src:
                try:
                    query = Query(parser.language, call_query_src)
                    cursor = QueryCursor(query)
                    captures_dict = cursor.captures(tree.root_node)

                    callee_nodes = captures_dict.get("callee", [])
                    for node in callee_nodes:
                        callee = node.text.decode("utf-8", errors="ignore")
                        caller = _find_enclosing_function(node, language)
                        if caller and callee and callee not in {"function", caller}:
                            call_edges.append((caller, callee))
                except Exception:
                    pass

            # Reference extraction
            ref_query_src = _get_plugin_query(language, "references")
            if not ref_query_src:
                ref_query_src = REFERENCE_QUERIES.get(language)
            if ref_query_src:
                try:
                    query = Query(parser.language, ref_query_src)
                    cursor = QueryCursor(query)
                    captures_dict = cursor.captures(tree.root_node)
                    for _capture_name, nodes in captures_dict.items():
                        for node in nodes:
                            ref = node.text.decode("utf-8", errors="ignore")
                            if ref and len(ref) > 1:
                                references.append(ref)
                except Exception:
                    pass

            # Inheritance extraction
            inherit_query_src = _get_plugin_query(language, "inheritance")
            if not inherit_query_src:
                inherit_query_src = INHERITS_QUERIES.get(language)
            if inherit_query_src:
                try:
                    query = Query(parser.language, inherit_query_src)
                    cursor = QueryCursor(query)
                    for _pat_idx, cap_dict in cursor.matches(tree.root_node):
                        child_nodes = cap_dict.get("child", [])
                        base_nodes = cap_dict.get("base", [])
                        if child_nodes and base_nodes:
                            child_text = child_nodes[0].text.decode("utf-8", errors="ignore")
                            base_text = base_nodes[0].text.decode("utf-8", errors="ignore")
                            if child_text and base_text:
                                inherit_edges.append((child_text, base_text))
                except Exception:
                    pass

            # Implements extraction
            impl_query_src = _get_plugin_query(language, "implements")
            if not impl_query_src:
                impl_query_src = IMPLEMENTS_QUERIES.get(language)
            if impl_query_src:
                try:
                    query = Query(parser.language, impl_query_src)
                    cursor = QueryCursor(query)
                    for _pat_idx, cap_dict in cursor.matches(tree.root_node):
                        child_nodes = cap_dict.get("child", [])
                        iface_nodes = cap_dict.get("interface", []) or cap_dict.get("base", [])
                        if child_nodes and iface_nodes:
                            child_text = child_nodes[0].text.decode("utf-8", errors="ignore")
                            iface_text = iface_nodes[0].text.decode("utf-8", errors="ignore")
                            if child_text and iface_text:
                                implements_edges.append((child_text, iface_text))
                except Exception:
                    pass

            # Composition extraction
            comp_query_src = _get_plugin_query(language, "composition")
            if not comp_query_src:
                comp_query_src = COMPOSITION_QUERIES.get(language)
            if comp_query_src:
                try:
                    query = Query(parser.language, comp_query_src)
                    cursor = QueryCursor(query)
                    for _pat_idx, cap_dict in cursor.matches(tree.root_node):
                        owner_nodes = cap_dict.get("owner", [])
                        type_nodes = cap_dict.get("type", [])
                        if owner_nodes and type_nodes:
                            owner_text = owner_nodes[0].text.decode("utf-8", errors="ignore")
                            type_text = type_nodes[0].text.decode("utf-8", errors="ignore")
                            if owner_text and type_text:
                                compose_edges.append((owner_text, type_text))
                except Exception:
                    pass

            # Import extraction for non-Python (tree-sitter based)
            if language != "python":
                import_query_src = _PARALLEL_IMPORT_QUERIES.get(language)
                if import_query_src:
                    try:
                        query = Query(parser.language, import_query_src)
                        cursor = QueryCursor(query)
                        captures_dict = cursor.captures(tree.root_node)
                        for node in captures_dict.get("source", []):
                            text = node.text.decode("utf-8", errors="ignore")
                            cleaned = text.strip("'\"")
                            if cleaned:
                                imports.append(cleaned)
                    except Exception:
                        pass
    except Exception:
        pass

    # Python-specific: extract imports via ast (more reliable than tree-sitter)
    if language == "python":
        try:
            from victor.verticals.contrib.coding.codebase.utils.ast_helpers import (
                extract_base_classes as _extract_bases,
            )

            tree = py_ast.parse(content, filename=file_path_str)
            for node in py_ast.walk(tree):
                if isinstance(node, py_ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, py_ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            # Extract inheritance from AST (more complete than tree-sitter)
            for node in py_ast.walk(tree):
                if isinstance(node, py_ast.ClassDef):
                    for base_name in _extract_bases(node):
                        inherit_edges.append((node.name, base_name))
        except Exception:
            pass

    return {
        "path": str(file_path.relative_to(root)),
        "language": language,
        "symbols": symbols_data,
        "imports": imports,
        "call_edges": call_edges,
        "inherit_edges": inherit_edges,
        "implements_edges": implements_edges,
        "compose_edges": compose_edges,
        "references": list(set(references)),
        "last_modified": stat.st_mtime,
        "size": stat.st_size,
        "lines": content.count("\n") + 1,
    }


def _find_enclosing_function(node: Any, language: str) -> Optional[str]:
    """Find the enclosing function name for a node.

    Helper for parallel processing - walks up the tree to find parent function.
    Uses plugin enclosing_scopes first, then falls back to static map.
    """
    plugin_scopes = _get_plugin_enclosing_scopes(language)
    if plugin_scopes:
        current = node.parent
        while current:
            for node_type, field_name in plugin_scopes:
                if current.type == node_type:
                    field = current.child_by_field_name(field_name)
                    if field:
                        if field.type == "function_declarator":
                            inner = field.child_by_field_name("declarator")
                            if inner:
                                field = inner
                        return field.text.decode("utf-8", errors="ignore")
            current = current.parent
        return None

    enclosing_types = {
        "python": ("function_definition",),
        "javascript": ("function_declaration", "method_definition", "arrow_function"),
        "typescript": ("function_declaration", "method_definition", "arrow_function"),
        "go": ("function_declaration", "method_declaration"),
        "java": ("method_declaration",),
        "cpp": ("function_definition",),
    }

    types = enclosing_types.get(language, ("function_definition",))
    current = node.parent
    while current:
        if current.type in types:
            for child in current.children:
                if child.type in (
                    "identifier",
                    "property_identifier",
                    "name",
                    "field_identifier",
                ):
                    return child.text.decode("utf-8", errors="ignore")
        current = current.parent
    return None


def _parse_file_worker(args: Tuple[str, str]) -> Optional[Dict[str, Any]]:
    """Parse a single Python file and extract metadata.

    This is a module-level function for use with ProcessPoolExecutor.
    Returns a dict with file metadata that can be converted to FileMetadata.

    Args:
        args: Tuple of (file_path_str, root_path_str)

    Returns:
        Dict with file metadata or None if parsing failed
    """
    file_path_str, root_path_str = args
    file_path = Path(file_path_str)
    root_path = Path(root_path_str)

    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(file_path))

        stat = file_path.stat()
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        rel_path = str(file_path.relative_to(root_path))

        from victor.verticals.contrib.coding.codebase.utils.ast_helpers import (
            extract_symbols as _extract_syms,
        )

        raw_symbols = _extract_syms(tree)
        symbols = [
            {
                "name": s.name,
                "type": s.type,
                "file_path": rel_path,
                "line_number": s.line_number,
                "docstring": s.docstring,
                "signature": s.signature,
            }
            for s in raw_symbols
        ]

        imports: list = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        return {
            "path": rel_path,
            "language": "python",
            "last_modified": stat.st_mtime,
            "indexed_at": time.time(),
            "size": stat.st_size,
            "lines": content.count("\n") + 1,
            "content_hash": content_hash,
            "symbols": symbols,
            "imports": imports,
        }

    except Exception as e:
        logger.debug(f"Failed to parse {file_path}: {e}")
        return None
