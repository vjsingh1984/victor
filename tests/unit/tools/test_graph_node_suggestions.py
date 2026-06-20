# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Graph `file:Symbol` resolution: same-file suggestions + unambiguous basename fallback."""

from __future__ import annotations

from victor.storage.graph.protocol import GraphNode
from victor.tools.graph_tool import (
    GraphAnalyzer,
    _find_similar_node_names,
    _symbols_in_file,
)


def _node(node_id: str, name: str, file: str, type_: str = "class") -> GraphNode:
    return GraphNode(node_id=node_id, type=type_, name=name, file=file)


def _analyzer(*nodes: GraphNode) -> GraphAnalyzer:
    a = GraphAnalyzer()
    for n in nodes:
        a.add_node(n)
    return a


def test_symbols_in_file_lists_only_that_files_symbols():
    a = _analyzer(
        _node("n1", "ToolSelector", "victor/agent/tool_selection.py"),
        _node("n2", "ToolSelectionStats", "victor/agent/tool_selection.py"),
        _node("n3", "pick_tools", "victor/agent/tool_selection.py", "function"),
        _node("n4", "Other", "victor/other.py"),
    )
    syms = set(_symbols_in_file(a, "victor/agent/tool_selection.py"))
    assert {"ToolSelector", "ToolSelectionStats", "pick_tools"} <= syms
    assert "Other" not in syms


def test_symbols_in_file_resolves_by_path_suffix():
    a = _analyzer(_node("n1", "ToolSelector", "victor/agent/tool_selection.py"))
    # A bare file name still finds the file via suffix match.
    assert "ToolSelector" in _symbols_in_file(a, "tool_selection.py")


def test_suggestions_for_hallucinated_compound_symbol():
    # The exact transcript failure: file:Symbol where the Symbol does not exist.
    a = _analyzer(
        _node("n1", "ToolSelector", "victor/agent/tool_selection.py"),
        _node("n2", "ToolSelectionStats", "victor/agent/tool_selection.py"),
    )
    names = _find_similar_node_names(a, "victor/agent/tool_selection.py:ToolSelectionService")
    # The real symbols in that file are surfaced so the caller can self-correct.
    assert "ToolSelector" in names


def test_basename_fallback_resolves_unambiguously():
    a = _analyzer(_node("n1", "ToolSelector", "victor/agent/tool_selection.py"))
    # Differently-formatted (wrong-dir) path, correct symbol -> basename fallback resolves.
    assert a.resolve_node_id("some/wrong/dir/tool_selection.py:ToolSelector") == "n1"


def test_basename_fallback_skips_when_ambiguous():
    a = _analyzer(
        _node("n1", "Dup", "a/tool_selection.py"),
        _node("n2", "Dup", "b/tool_selection.py"),
    )
    # Two same-basename files define the symbol -> ambiguous -> do NOT silently pick one.
    assert a.resolve_node_id("z/tool_selection.py:Dup") is None


def test_exact_path_still_preferred_over_basename():
    a = _analyzer(
        _node("n1", "Sym", "pkg/a/mod.py"),
        _node("n2", "Sym", "pkg/b/mod.py"),
    )
    # Exact/suffix path match wins; basename ambiguity is irrelevant here.
    assert a.resolve_node_id("pkg/a/mod.py:Sym") == "n1"
