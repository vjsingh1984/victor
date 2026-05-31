from __future__ import annotations

import ast

from victor.core.utils import ast_helpers


def _parse(source: str) -> ast.Module:
    return ast.parse(source.lstrip("\n"))


def test_stdlib_detection_and_import_extraction() -> None:
    tree = _parse("""
import os
import json as js
from typing import Optional
from project.local import helper
""")

    assert ast_helpers.is_stdlib_module("os.path") is True
    assert ast_helpers.is_stdlib_module("typing.Optional") is True
    assert ast_helpers.is_stdlib_module("project.local") is False

    assert ast_helpers.extract_imports(tree) == [
        "json",
        "os",
        "project.local",
        "typing",
    ]


def test_symbol_extraction_includes_enrichment_for_python_nodes() -> None:
    tree = _parse('''
class Base:
    pass


class Worker(Base):
    """Worker docs."""

    @decorator
    async def run(self, value: int, label: str = "x") -> str:
        """Run docs."""
        return label
''')

    symbols = ast_helpers.extract_symbols(tree, enrich=True)
    by_name = {(symbol.name, symbol.line_number): symbol for symbol in symbols}

    worker = by_name[("Worker", 5)]
    assert worker.type == "class"
    assert worker.base_classes == ["Base"]
    assert worker.docstring == "Worker docs."

    run = by_name[("run", 9)]
    assert run.type == "function"
    assert run.parent_symbol == "Worker"
    assert run.is_async is True
    assert run.return_type == "str"
    assert run.docstring == "Run docs."
    assert run.decorators == ["@decorator"]
    assert run.parameters == [("self", None), ("value", "int"), ("label", "str")]
    assert "async run(" in (run.signature or "")
    assert "value: int" in (run.signature or "")
    assert "label: str" in (run.signature or "")
    assert "-> str" in (run.signature or "")


def test_helper_primitives_cover_common_python_syntax() -> None:
    function_node = _parse("""
@retry(3)
def work(item: list[str], enabled: bool = True, *args, **kwargs) -> dict[str, int]:
    pass
""").body[0]
    assert isinstance(function_node, ast.FunctionDef)

    class_node = _parse("class Child(Parent, mixins.Base):\n    pass\n").body[0]
    assert isinstance(class_node, ast.ClassDef)

    assert ast_helpers.get_decorator_name(function_node.decorator_list[0]) == "@retry(3)"
    assert ast_helpers.get_annotation_str(function_node.args.args[0].annotation) == "list[str]"
    assert ast_helpers.extract_parameters(function_node) == [
        ("item", "list[str]"),
        ("enabled", "bool"),
        ("*args", None),
        ("**kwargs", None),
    ]
    assert ast_helpers.build_signature(function_node).startswith(
        "work(item: list[str], enabled: bool = True"
    )
    assert ast_helpers.build_signature(function_node).endswith("-> dict[str, int]")
    assert ast_helpers.extract_base_classes(class_node) == ["Parent", "mixins.Base"]
