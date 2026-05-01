"""Lightweight AST helpers shared by core and extracted coding packages."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

STDLIB_MODULES: frozenset[str] = frozenset(getattr(sys, "stdlib_module_names", ()))


@dataclass(frozen=True)
class SymbolSummary:
    """Summary of a Python symbol extracted from the AST."""

    name: str
    type: str
    line_number: int
    end_line_number: Optional[int] = None
    parent_symbol: Optional[str] = None
    signature: Optional[str] = None
    parameters: List[tuple[str, Optional[str]]] = field(default_factory=list)
    return_type: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    base_classes: List[str] = field(default_factory=list)
    is_async: bool = False


def _to_source(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def is_stdlib_module(module_name: str) -> bool:
    """Return whether a module belongs to the Python standard library."""

    if not module_name:
        return False
    top_level = module_name.split(".", 1)[0]
    return top_level in STDLIB_MODULES


def get_annotation_str(annotation: Optional[ast.AST]) -> Optional[str]:
    """Return the source string for an annotation node."""

    return _to_source(annotation)


def get_decorator_name(node: ast.expr) -> str:
    """Return a normalized decorator string including the ``@`` prefix."""

    return f"@{_to_source(node) or '<unknown>'}"


def extract_parameters(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> List[tuple[str, Optional[str]]]:
    """Extract function parameters with annotation strings."""

    parameters: List[tuple[str, Optional[str]]] = []

    positional = list(node.args.posonlyargs) + list(node.args.args)
    for arg in positional:
        parameters.append((arg.arg, get_annotation_str(arg.annotation)))

    if node.args.vararg is not None:
        parameters.append(
            (f"*{node.args.vararg.arg}", get_annotation_str(node.args.vararg.annotation))
        )

    for arg in node.args.kwonlyargs:
        parameters.append((arg.arg, get_annotation_str(arg.annotation)))

    if node.args.kwarg is not None:
        parameters.append(
            (f"**{node.args.kwarg.arg}", get_annotation_str(node.args.kwarg.annotation))
        )

    return parameters


def _format_arg(
    arg_name: str,
    annotation: Optional[str],
    default: Optional[ast.AST] = None,
) -> str:
    formatted = arg_name
    if annotation:
        formatted = f"{formatted}: {annotation}"
    if default is not None:
        formatted = f"{formatted} = {_to_source(default) or '...'}"
    return formatted


def build_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Build a readable function signature string."""

    parts: List[str] = []
    positional = list(node.args.posonlyargs) + list(node.args.args)
    positional_defaults: List[Optional[ast.AST]] = [None] * (
        len(positional) - len(node.args.defaults)
    ) + list(node.args.defaults)

    for arg, default in zip(positional, positional_defaults, strict=False):
        parts.append(_format_arg(arg.arg, get_annotation_str(arg.annotation), default))

    if node.args.vararg is not None:
        parts.append(
            _format_arg(
                f"*{node.args.vararg.arg}",
                get_annotation_str(node.args.vararg.annotation),
            )
        )
    elif node.args.kwonlyargs:
        parts.append("*")

    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=False):
        parts.append(_format_arg(arg.arg, get_annotation_str(arg.annotation), default))

    if node.args.kwarg is not None:
        parts.append(
            _format_arg(
                f"**{node.args.kwarg.arg}",
                get_annotation_str(node.args.kwarg.annotation),
            )
        )

    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    signature = f"{prefix}{node.name}({', '.join(parts)})"
    return_annotation = get_annotation_str(node.returns)
    if return_annotation:
        signature = f"{signature} -> {return_annotation}"
    return signature


def extract_base_classes(node: ast.ClassDef) -> List[str]:
    """Extract base class expressions from a class definition."""

    return [_to_source(base) or "<unknown>" for base in node.bases]


def extract_imports(tree: ast.AST) -> List[str]:
    """Extract imported module names in sorted order."""

    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    return sorted(imports)


def _iter_symbol_nodes(body: Iterable[ast.stmt]) -> Iterable[ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef]:
    for node in body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _summarize_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    parent_symbol: Optional[str],
    enrich: bool,
) -> SymbolSummary:
    return SymbolSummary(
        name=node.name,
        type="function",
        line_number=node.lineno,
        end_line_number=getattr(node, "end_lineno", None),
        parent_symbol=parent_symbol,
        signature=build_signature(node) if enrich else None,
        parameters=extract_parameters(node) if enrich else [],
        return_type=get_annotation_str(node.returns) if enrich else None,
        decorators=[get_decorator_name(item) for item in node.decorator_list] if enrich else [],
        docstring=ast.get_docstring(node) if enrich else None,
        is_async=isinstance(node, ast.AsyncFunctionDef),
    )


def _summarize_class(
    node: ast.ClassDef,
    *,
    parent_symbol: Optional[str],
    enrich: bool,
) -> SymbolSummary:
    return SymbolSummary(
        name=node.name,
        type="class",
        line_number=node.lineno,
        end_line_number=getattr(node, "end_lineno", None),
        parent_symbol=parent_symbol,
        docstring=ast.get_docstring(node) if enrich else None,
        base_classes=extract_base_classes(node) if enrich else [],
    )


def extract_symbols(tree: ast.AST, *, enrich: bool = False) -> List[SymbolSummary]:
    """Extract class and function symbols from a Python AST."""

    if not isinstance(tree, ast.Module):
        return []

    symbols: List[SymbolSummary] = []

    def visit(body: Iterable[ast.stmt], parent_symbol: Optional[str] = None) -> None:
        for node in _iter_symbol_nodes(body):
            if isinstance(node, ast.ClassDef):
                symbols.append(_summarize_class(node, parent_symbol=parent_symbol, enrich=enrich))
                visit(node.body, parent_symbol=node.name)
            else:
                symbols.append(
                    _summarize_function(node, parent_symbol=parent_symbol, enrich=enrich)
                )
                visit(node.body, parent_symbol=node.name)

    visit(tree.body)
    return symbols


__all__ = [
    "STDLIB_MODULES",
    "SymbolSummary",
    "build_signature",
    "extract_base_classes",
    "extract_imports",
    "extract_parameters",
    "extract_symbols",
    "get_annotation_str",
    "get_decorator_name",
    "is_stdlib_module",
]
