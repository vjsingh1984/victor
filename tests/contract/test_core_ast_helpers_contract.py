from __future__ import annotations

import ast

from victor.core.utils.ast_helpers import (
    STDLIB_MODULES,
    build_signature,
    extract_imports,
    extract_symbols,
    is_stdlib_module,
)


def test_core_ast_helpers_remain_available_to_victor_coding() -> None:
    tree = ast.parse(
        """
import os

class Worker:
    async def run(self, value: int) -> str:
        return str(value)
"""
    )

    symbols = extract_symbols(tree, enrich=True)
    run_symbol = next(symbol for symbol in symbols if symbol.name == "run")

    assert "os" in STDLIB_MODULES
    assert is_stdlib_module("os.path") is True
    assert extract_imports(tree) == ["os"]
    assert build_signature(tree.body[1].body[0]).startswith("async run(")
    assert run_symbol.parent_symbol == "Worker"
    assert run_symbol.return_type == "str"
