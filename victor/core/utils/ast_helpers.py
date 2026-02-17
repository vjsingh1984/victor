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

"""Shared Python AST extraction helpers.

Consolidates duplicated AST extraction patterns from indexer.py,
codebase_analyzer.py, unified_extractor.py, and shared_ast_utils.py
into a single canonical module.

All functions are module-level (not class methods) to satisfy the
ProcessPoolExecutor pickling constraint in indexer.py.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List, Optional, Union

# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SymbolSummary:
    """Summary of a class or function extracted from the AST."""

    name: str
    type: str  # "class" | "function"
    line_number: int
    docstring: Optional[str] = None
    signature: Optional[str] = None
    base_classes: list = field(default_factory=list)
    # Enriched fields (populated when enrich=True)
    is_async: bool = False
    parameters: list = field(default_factory=list)
    return_type: Optional[str] = None
    decorators: list = field(default_factory=list)


# =============================================================================
# Standard Library Detection
# =============================================================================

# Merged, deduplicated superset from indexer.py (67 entries) and
# codebase_analyzer.py (31 entries).
STDLIB_MODULES: frozenset = frozenset(
    {
        # Core builtins
        "abc",
        "asyncio",
        "builtins",
        "collections",
        "contextlib",
        "copy",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "functools",
        "gc",
        "hashlib",
        "heapq",
        "importlib",
        "inspect",
        "io",
        "itertools",
        "json",
        "logging",
        "math",
        "operator",
        "os",
        "pathlib",
        "pickle",
        "platform",
        "pprint",
        "queue",
        "random",
        "re",
        "secrets",
        "shutil",
        "signal",
        "socket",
        "sqlite3",
        "ssl",
        "string",
        "struct",
        "subprocess",
        "sys",
        "tempfile",
        "threading",
        "time",
        "traceback",
        "typing",
        "unittest",
        "urllib",
        "uuid",
        "warnings",
        "weakref",
        "xml",
        "zipfile",
        "zlib",
        # Typing extensions
        "typing_extensions",
        # Common third-party (excluded from graph to avoid inflating PageRank)
        "numpy",
        "pandas",
        "requests",
        "aiohttp",
        "httpx",
        "pydantic",
        "pytest",
        "mock",
        "unittest.mock",
    }
)


def is_stdlib_module(module_name: str) -> bool:
    """Check if a module is stdlib or a common third-party module.

    Delegates to the Rust accelerator when available for 5-10x speedup.
    """
    try:
        from victor.processing.native import (
            is_native_available,
            is_stdlib_module as native_is_stdlib_module,
        )

        if is_native_available():
            return native_is_stdlib_module(module_name)
    except ImportError:
        pass

    if module_name in STDLIB_MODULES:
        return True
    top_level = module_name.split(".")[0]
    return top_level in STDLIB_MODULES


# =============================================================================
# AST Annotation / Decorator Helpers
# =============================================================================


def get_annotation_str(node: Optional[ast.expr]) -> Optional[str]:
    """Convert an AST annotation node to its string representation."""
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def get_decorator_name(node: ast.expr) -> str:
    """Get the full dotted name of a decorator AST node.

    Handles ``Name``, ``Attribute``, and ``Call`` nodes.
    """
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        parts: list = []
        current: ast.expr = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
    elif isinstance(node, ast.Call):
        return get_decorator_name(node.func)
    try:
        return ast.unparse(node)
    except Exception:
        return "<decorator>"


# =============================================================================
# Import Extraction
# =============================================================================


def extract_imports(tree: ast.AST, top_level_only: bool = True) -> List[str]:
    """Extract imported module names from an AST.

    Args:
        tree: Parsed AST (typically ``ast.Module``).
        top_level_only: If True, returns top-level module names
            (e.g. ``"os"`` from ``import os.path``).
            If False, returns fully-qualified names
            (e.g. ``"os.path.join"`` from ``from os.path import join``).
    """
    imports: list = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if top_level_only:
                    imports.append(alias.name.split(".")[0])
                else:
                    imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if top_level_only:
                if module:
                    imports.append(module.split(".")[0])
            else:
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)
    return imports


# =============================================================================
# Base Class Extraction
# =============================================================================


def extract_base_classes(node: ast.ClassDef) -> List[str]:
    """Extract base class names from a ClassDef node.

    Uses ``ast.unparse()`` (Python 3.10+) with ``Name.id`` /
    ``Attribute.attr`` fallback.
    """
    bases: list = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            bases.append(base.id)
        elif isinstance(base, ast.Attribute):
            try:
                bases.append(ast.unparse(base))
            except Exception:
                bases.append(base.attr)
        elif isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name):
            # e.g. Generic[T] → "Generic"
            bases.append(base.value.id)
        else:
            try:
                bases.append(ast.unparse(base))
            except Exception:
                name = getattr(base, "id", None)
                if name:
                    bases.append(name)
    return bases


# =============================================================================
# Signature / Parameter Extraction
# =============================================================================


def build_signature(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
    """Build a human-readable signature string like ``foo(self, x, y)``."""
    args = [arg.arg for arg in node.args.args]
    return f"{node.name}({', '.join(args)})"


def extract_parameters(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    include_annotations: bool = False,
) -> List[str]:
    """Extract parameter names, optionally with type annotations.

    With ``include_annotations=True``: ``["x: int", "y"]``.
    With ``include_annotations=False``: ``["x", "y"]``.
    """
    params: list = []
    for arg in node.args.args:
        if include_annotations and arg.annotation:
            ann = get_annotation_str(arg.annotation)
            params.append(f"{arg.arg}: {ann}" if ann else arg.arg)
        else:
            params.append(arg.arg)
    return params


# =============================================================================
# Symbol Extraction
# =============================================================================


def extract_symbols(tree: ast.AST, enrich: bool = False) -> List[SymbolSummary]:
    """Extract class and function symbols from an AST.

    Args:
        tree: Parsed AST (typically ``ast.Module``).
        enrich: If True, also populate ``parameters``, ``return_type``,
            ``decorators``, and ``is_async`` fields.
    """
    symbols: list = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            sym = SymbolSummary(
                name=node.name,
                type="class",
                line_number=node.lineno,
                docstring=ast.get_docstring(node),
                base_classes=extract_base_classes(node),
            )
            if enrich:
                sym.decorators = [get_decorator_name(d) for d in node.decorator_list]
            symbols.append(sym)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sym = SymbolSummary(
                name=node.name,
                type="function",
                line_number=node.lineno,
                docstring=ast.get_docstring(node),
                signature=build_signature(node),
            )
            if enrich:
                sym.is_async = isinstance(node, ast.AsyncFunctionDef)
                sym.parameters = extract_parameters(node, include_annotations=True)
                sym.return_type = get_annotation_str(node.returns)
                sym.decorators = [get_decorator_name(d) for d in node.decorator_list]
            symbols.append(sym)
    return symbols
