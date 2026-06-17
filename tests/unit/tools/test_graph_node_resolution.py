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

"""Dotted-module node resolution against path-based (e.g. Rust) file nodes.

Regression for "Could not resolve graph node 'src.network.multi_server'" — the
model addressed files with Python-style dotted paths while the graph stored
``src/network/multi_server.rs`` file nodes.
"""

from victor.storage.graph.protocol import GraphNode
from victor.tools.graph_tool import GraphAnalyzer, _module_path_stem


def _analyzer() -> GraphAnalyzer:
    analyzer = GraphAnalyzer()
    for node_id, type_, name, file in [
        ("n1", "file", "multi_server.rs", "src/network/multi_server.rs"),
        ("n2", "file", "mod.rs", "src/network/mod.rs"),
        ("n3", "file", "lib.rs", "src/lib.rs"),
        ("n4", "struct", "MultiServer", "src/network/multi_server.rs"),
    ]:
        analyzer.add_node(GraphNode(node_id=node_id, type=type_, name=name, file=file))
    return analyzer


def test_module_path_stem_strips_extension_and_index_files() -> None:
    assert _module_path_stem("src/network/multi_server.rs") == "src/network/multi_server"
    assert _module_path_stem("src/network/mod.rs") == "src/network"
    assert _module_path_stem("src/lib.rs") == "src/lib"
    assert _module_path_stem("a/b/__init__.py") == "a/b"


def test_dotted_file_reference_resolves() -> None:
    analyzer = _analyzer()
    assert analyzer.resolve_node_id("src.network.multi_server") in {"n1", "n4"}


def test_dotted_module_dir_resolves_via_index_file() -> None:
    analyzer = _analyzer()
    # src.network -> src/network/mod.rs
    assert analyzer.resolve_node_id("src.network") == "n2"


def test_dotted_lib_reference_resolves() -> None:
    analyzer = _analyzer()
    assert analyzer.resolve_node_id("src.lib") == "n3"


def test_unrelated_dotted_reference_still_unresolved() -> None:
    analyzer = _analyzer()
    assert analyzer.resolve_node_id("totally.unknown.thing") is None
