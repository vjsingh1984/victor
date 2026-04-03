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

"""Shared AST utilities for Python code analysis.

This module provides common AST parsing and analysis functions used across
multiple tools (code_review, metrics, refactor, documentation). Consolidating
these utilities reduces code duplication and ensures consistent analysis.

Features:
- Safe AST parsing with error handling
- Function and class discovery
- Cyclomatic complexity calculation
- Maintainability index calculation
- Symbol lookup
- Documentation analysis
"""

from __future__ import annotations

import ast
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

from victor.core.utils.ast_helpers import (
    extract_base_classes as _extract_base_classes,
    extract_imports as _extract_imports,
    get_decorator_name as _get_decorator_name,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FunctionInfo:
    """Information about a function in the AST."""

    name: str
    lineno: int
    end_lineno: Optional[int]
    col_offset: int
    is_async: bool
    args: List[str]
    defaults_count: int
    has_docstring: bool
    docstring: Optional[str]
    decorators: List[str]
    complexity: int
    return_count: int
    has_yield: bool


@dataclass
class ClassInfo:
    """Information about a class in the AST."""

    name: str
    lineno: int
    end_lineno: Optional[int]
    col_offset: int
    bases: List[str]
    has_docstring: bool
    docstring: Optional[str]
    decorators: List[str]
    methods: List[FunctionInfo] = field(default_factory=list)
    class_variables: List[str] = field(default_factory=list)


@dataclass
class ModuleInfo:
    """Information about a Python module."""

    path: Optional[Path]
    lines_of_code: int
    blank_lines: int
    comment_lines: int
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    imports: List[str]
    docstring: Optional[str]
    complexity: int
    maintainability_index: float


@dataclass
class ParseResult:
    """Result of parsing code to AST."""

    success: bool
    tree: Optional[ast.AST]
    error: Optional[str]
    syntax_error_line: Optional[int]


# =============================================================================
# Parsing Functions
# =============================================================================


def parse_code(code: str) -> ParseResult:
    """Safely parse Python code to AST.

    Args:
        code: Python source code string.

    Returns:
        ParseResult with tree on success, error details on failure.
    """
    try:
        tree = ast.parse(code)
        return ParseResult(success=True, tree=tree, error=None, syntax_error_line=None)
    except SyntaxError as e:
        return ParseResult(
            success=False,
            tree=None,
            error=str(e),
            syntax_error_line=e.lineno,
        )
    except Exception as e:
        return ParseResult(success=False, tree=None, error=str(e), syntax_error_line=None)


def parse_file(path: Union[str, Path], encoding: str = "utf-8") -> ParseResult:
    """Safely parse a Python file to AST.

    Args:
        path: Path to Python file.
        encoding: File encoding (default: utf-8).

    Returns:
        ParseResult with tree on success, error details on failure.
    """
    path_obj = Path(path)

    if not path_obj.exists():
        return ParseResult(
            success=False,
            tree=None,
            error=f"File not found: {path}",
            syntax_error_line=None,
        )

    if not path_obj.is_file():
        return ParseResult(
            success=False,
            tree=None,
            error=f"Not a file: {path}",
            syntax_error_line=None,
        )

    try:
        code = path_obj.read_text(encoding=encoding)
        return parse_code(code)
    except UnicodeDecodeError as e:
        return ParseResult(
            success=False,
            tree=None,
            error=f"Encoding error: {e}",
            syntax_error_line=None,
        )
    except Exception as e:
        return ParseResult(
            success=False, tree=None, error=f"Failed to read file: {e}", syntax_error_line=None
        )


# =============================================================================
# Symbol Discovery
# =============================================================================


def find_functions(
    tree: ast.AST, include_methods: bool = True, include_nested: bool = True
) -> Iterator[Union[ast.FunctionDef, ast.AsyncFunctionDef]]:
    """Find all function definitions in AST.

    Args:
        tree: AST to search.
        include_methods: Include class methods (default: True).
        include_nested: Include nested functions (default: True).

    Yields:
        FunctionDef and AsyncFunctionDef nodes.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def find_classes(tree: ast.AST) -> Iterator[ast.ClassDef]:
    """Find all class definitions in AST.

    Args:
        tree: AST to search.

    Yields:
        ClassDef nodes.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            yield node


def find_symbol(
    tree: ast.AST, name: str
) -> Optional[Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]]:
    """Find a specific function or class by name.

    Args:
        tree: AST to search.
        name: Name of the symbol to find.

    Returns:
        The matching node or None if not found.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                return node
    return None


def find_imports(tree: ast.AST) -> List[str]:
    """Find all import statements in AST.

    Args:
        tree: AST to search.

    Returns:
        List of imported module names (fully qualified).
    """
    return _extract_imports(tree, top_level_only=False)


# =============================================================================
# Complexity Metrics
# =============================================================================


def calculate_complexity(node: ast.AST) -> int:
    """Calculate cyclomatic complexity of a function or module.

    Cyclomatic complexity counts the number of independent paths through code.
    Each decision point (if, for, while, except, etc.) adds 1 to complexity.

    Args:
        node: AST node (typically FunctionDef or Module).

    Returns:
        Cyclomatic complexity score (minimum 1).
    """
    complexity = 1  # Base complexity

    for child in ast.walk(node):
        # Each decision point adds 1
        if isinstance(child, (ast.If, ast.While, ast.For)):
            complexity += 1
        elif isinstance(child, ast.ExceptHandler):
            complexity += 1
        elif isinstance(child, ast.With):
            complexity += 1
        elif isinstance(child, ast.Assert):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            # Boolean operators add (num_values - 1) complexity
            complexity += len(child.values) - 1
        elif isinstance(child, ast.comprehension):
            # List/dict/set comprehensions add complexity
            complexity += 1
            complexity += len(child.ifs)  # Additional complexity for if clauses
        elif isinstance(child, ast.IfExp):
            # Ternary expressions
            complexity += 1

    return complexity


def calculate_maintainability_index(code: str) -> float:
    """Calculate maintainability index (0-100).

    The maintainability index is a composite metric based on:
    - Halstead Volume (based on operands and operators)
    - Cyclomatic Complexity
    - Lines of Code

    Higher values indicate more maintainable code.

    Args:
        code: Python source code string.

    Returns:
        Maintainability index (0-100).
    """
    result = parse_code(code)
    if not result.success or not result.tree:
        return 0.0

    lines = code.split("\n")
    loc = len([line for line in lines if line.strip() and not line.strip().startswith("#")])

    if loc == 0:
        return 100.0  # Empty code is perfectly maintainable

    # Calculate average complexity
    total_complexity = 0
    function_count = 0
    for node in ast.walk(result.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            total_complexity += calculate_complexity(node)
            function_count += 1

    avg_complexity = total_complexity / max(function_count, 1)

    # Simplified Halstead Volume estimation (based on tokens)
    # Real implementation would count unique operators and operands
    try:
        import tokenize
        from io import StringIO

        tokens = list(tokenize.generate_tokens(StringIO(code).readline))
        n = len(tokens)  # Total tokens
        volume = n * math.log2(max(n, 1))
    except Exception:
        volume = loc * 10  # Fallback estimation

    # Maintainability Index formula (Microsoft variant)
    # MI = max(0, (171 - 5.2 * ln(V) - 0.23 * G - 16.2 * ln(LOC)) * 100 / 171)
    try:
        mi = 171 - 5.2 * math.log(max(volume, 1)) - 0.23 * avg_complexity - 16.2 * math.log(loc)
        mi = (mi * 100) / 171
        return max(0.0, min(100.0, mi))
    except (ValueError, ZeroDivisionError):
        return 0.0


def calculate_cognitive_complexity(node: ast.AST) -> int:
    """Calculate cognitive complexity of a function.

    Cognitive complexity measures how hard code is to understand.
    It penalizes nested structures more than cyclomatic complexity.

    Args:
        node: AST node (typically FunctionDef).

    Returns:
        Cognitive complexity score.
    """
    complexity = 0
    nesting_level = 0

    class CognitiveVisitor(ast.NodeVisitor):
        """AST visitor for calculating cognitive complexity.

        This visitor traverses an AST and calculates cognitive complexity
        by counting control flow structures with nesting penalties. Unlike
        cyclomatic complexity, cognitive complexity penalizes deeply nested
        code more heavily, reflecting the cognitive load on developers.

        Complexity increments:
        - If/For/While/ExceptHandler: +1 base, +nesting_level penalty
        - BoolOp (and/or): +(num_values - 1) for chained conditions
        - Lambda: +1 for inline function complexity

        Uses nonlocal to track state across the recursive visit.
        """

        nonlocal complexity, nesting_level

        def visit_If(self, node: ast.If) -> None:
            nonlocal complexity, nesting_level
            complexity += 1 + nesting_level
            nesting_level += 1
            self.generic_visit(node)
            nesting_level -= 1

        def visit_For(self, node: ast.For) -> None:
            nonlocal complexity, nesting_level
            complexity += 1 + nesting_level
            nesting_level += 1
            self.generic_visit(node)
            nesting_level -= 1

        def visit_While(self, node: ast.While) -> None:
            nonlocal complexity, nesting_level
            complexity += 1 + nesting_level
            nesting_level += 1
            self.generic_visit(node)
            nesting_level -= 1

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            nonlocal complexity, nesting_level
            complexity += 1 + nesting_level
            nesting_level += 1
            self.generic_visit(node)
            nesting_level -= 1

        def visit_BoolOp(self, node: ast.BoolOp) -> None:
            nonlocal complexity
            # Each boolean operator adds 1
            complexity += len(node.values) - 1
            self.generic_visit(node)

        def visit_Lambda(self, node: ast.Lambda) -> None:
            nonlocal complexity
            complexity += 1
            self.generic_visit(node)

    visitor = CognitiveVisitor()
    visitor.visit(node)

    return complexity


# =============================================================================
# Information Extraction
# =============================================================================


def get_function_info(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> FunctionInfo:
    """Extract detailed information about a function.

    Args:
        node: FunctionDef or AsyncFunctionDef node.

    Returns:
        FunctionInfo with extracted details.
    """
    # Get argument names
    args = []
    for arg in node.args.args:
        args.append(arg.arg)
    for arg in node.args.posonlyargs:
        args.append(arg.arg)
    for arg in node.args.kwonlyargs:
        args.append(arg.arg)
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")

    # Get decorators
    decorators = [_get_decorator_name(d) for d in node.decorator_list]

    # Check for docstring
    docstring = ast.get_docstring(node)

    # Count returns and yields
    return_count = sum(1 for n in ast.walk(node) if isinstance(n, ast.Return))
    has_yield = any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(node))

    return FunctionInfo(
        name=node.name,
        lineno=node.lineno,
        end_lineno=getattr(node, "end_lineno", None),
        col_offset=node.col_offset,
        is_async=isinstance(node, ast.AsyncFunctionDef),
        args=args,
        defaults_count=len(node.args.defaults) + len(node.args.kw_defaults),
        has_docstring=docstring is not None,
        docstring=docstring,
        decorators=decorators,
        complexity=calculate_complexity(node),
        return_count=return_count,
        has_yield=has_yield,
    )


def get_class_info(node: ast.ClassDef) -> ClassInfo:
    """Extract detailed information about a class.

    Args:
        node: ClassDef node.

    Returns:
        ClassInfo with extracted details.
    """
    # Get base classes
    bases = _extract_base_classes(node)

    # Get decorators
    decorators = [_get_decorator_name(d) for d in node.decorator_list]

    # Check for docstring
    docstring = ast.get_docstring(node)

    # Get methods
    methods = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.append(get_function_info(item))

    # Get class variables
    class_vars = []
    for item in node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    class_vars.append(target.id)
        elif isinstance(item, ast.AnnAssign):
            if isinstance(item.target, ast.Name):
                class_vars.append(item.target.id)

    return ClassInfo(
        name=node.name,
        lineno=node.lineno,
        end_lineno=getattr(node, "end_lineno", None),
        col_offset=node.col_offset,
        bases=bases,
        has_docstring=docstring is not None,
        docstring=docstring,
        decorators=decorators,
        methods=methods,
        class_variables=class_vars,
    )


def get_module_info(code: str, path: Optional[Path] = None) -> Optional[ModuleInfo]:
    """Extract comprehensive information about a Python module.

    Args:
        code: Python source code.
        path: Optional path to the source file.

    Returns:
        ModuleInfo with extracted details, or None if parsing fails.
    """
    result = parse_code(code)
    if not result.success or not result.tree:
        return None

    lines = code.split("\n")
    loc = len([line for line in lines if line.strip()])
    blank_lines = len([line for line in lines if not line.strip()])
    comment_lines = len([line for line in lines if line.strip().startswith("#")])

    functions = [get_function_info(f) for f in find_functions(result.tree)]
    classes = [get_class_info(c) for c in find_classes(result.tree)]
    imports = find_imports(result.tree)
    docstring = ast.get_docstring(result.tree)

    return ModuleInfo(
        path=path,
        lines_of_code=loc,
        blank_lines=blank_lines,
        comment_lines=comment_lines,
        functions=functions,
        classes=classes,
        imports=imports,
        docstring=docstring,
        complexity=calculate_complexity(result.tree),
        maintainability_index=calculate_maintainability_index(code),
    )


# =============================================================================
# Documentation Analysis
# =============================================================================


def has_docstring(node: ast.AST) -> bool:
    """Check if a node has a docstring.

    Args:
        node: AST node (Module, ClassDef, or FunctionDef).

    Returns:
        True if the node has a docstring.
    """
    return ast.get_docstring(node) is not None


def get_undocumented_functions(tree: ast.AST) -> List[Tuple[str, int]]:
    """Find functions without docstrings.

    Args:
        tree: AST to search.

    Returns:
        List of (function_name, line_number) tuples.
    """
    undocumented = []
    for node in find_functions(tree):
        if not has_docstring(node):
            undocumented.append((node.name, node.lineno))
    return undocumented


def get_undocumented_classes(tree: ast.AST) -> List[Tuple[str, int]]:
    """Find classes without docstrings.

    Args:
        tree: AST to search.

    Returns:
        List of (class_name, line_number) tuples.
    """
    undocumented = []
    for node in find_classes(tree):
        if not has_docstring(node):
            undocumented.append((node.name, node.lineno))
    return undocumented


# =============================================================================
# Helper Functions
# =============================================================================


def _get_attribute_name(node: ast.Attribute) -> str:
    """Get the full attribute name (e.g., 'module.submodule.attr').

    Args:
        node: Attribute node.

    Returns:
        Dot-separated attribute name.
    """
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def count_functions(tree: ast.AST) -> int:
    """Count total functions in AST.

    Args:
        tree: AST to count.

    Returns:
        Number of function definitions.
    """
    return sum(1 for _ in find_functions(tree))


def count_classes(tree: ast.AST) -> int:
    """Count total classes in AST.

    Args:
        tree: AST to count.

    Returns:
        Number of class definitions.
    """
    return sum(1 for _ in find_classes(tree))


def get_line_count(code: str) -> Dict[str, int]:
    """Count different types of lines in code.

    Args:
        code: Python source code.

    Returns:
        Dictionary with 'total', 'code', 'blank', 'comment' counts.
    """
    lines = code.split("\n")
    total = len(lines)
    blank = len([line for line in lines if not line.strip()])
    comment = len([line for line in lines if line.strip().startswith("#")])
    code_lines = total - blank - comment

    return {
        "total": total,
        "code": code_lines,
        "blank": blank,
        "comment": comment,
    }
