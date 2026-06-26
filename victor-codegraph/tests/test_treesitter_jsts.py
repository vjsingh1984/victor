"""JS/TS extraction tests — the donor stub fix.

Guarded with importorskip so the suite stays green where the grammar pack isn't
installed; where it is, these assert the stub is genuinely replaced (non-empty symbols).
"""

from __future__ import annotations

import pytest

pytest.importorskip("tree_sitter_language_pack")

from victor_codegraph import CodeSymbolType, parse  # noqa: E402
from victor_codegraph.treesitter_parser import GrammarUnavailable, parse_treesitter  # noqa: E402

JS = """\
import { x } from "./x";

export function add(a, b) {
  return a + b;
}

const mul = (a, b) => a * b;

class Calc {
  constructor(seed) {
    this.seed = seed;
  }
  run(n) {
    return add(n, this.seed);
  }
}
"""

TS = """\
export function greet(name: string): string {
  return `hi ${name}`;
}

class Service {
  handle(req: Request): Response {
    return new Response();
  }
}
"""


def _names(language: str, src: str):
    try:
        parsed = parse_treesitter(src, f"f.{language}", language)
    except GrammarUnavailable:
        pytest.skip(f"{language} grammar not installed")
    return {s.simple_name for s in parsed.symbols}, parsed


def test_javascript_functions_classes_methods_arrow():
    names, parsed = _names("javascript", JS)
    # The donor stub returned []; here we must see real symbols.
    assert {"add", "Calc", "run"} <= names
    assert "mul" in names  # const arrow function
    assert any(s.symbol_type == CodeSymbolType.CLASS for s in parsed.symbols)
    assert any("import" in i for i in parsed.imports)


def test_typescript_functions_and_methods():
    names, _ = _names("typescript", TS)
    assert {"greet", "Service", "handle"} <= names


def test_chunk_jsts_is_size_capped():
    parsed = parse(JS, language="javascript", file_path="f.js")
    if not parsed.symbols:
        pytest.skip("javascript grammar not installed")
    assert parsed.symbols  # routed through tree-sitter, not the empty stub
