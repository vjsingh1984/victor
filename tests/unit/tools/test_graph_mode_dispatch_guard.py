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

"""Guard: every advertised GraphMode has a dispatch handler.

`dead_code` / `dynamic_imports` were members of the GraphMode enum (so they appeared in
the schema and the "Supported modes" error) but had no handler in the dispatch chain — so
calling them passed validation then hit the "Unsupported graph mode" fallback. This guard
fails if any advertised mode is not referenced in the dispatch.
"""

from __future__ import annotations

from pathlib import Path

from victor.tools.graph_tool import GraphMode

_GRAPH_TOOL = Path(__file__).resolve().parents[3] / "victor" / "tools" / "graph_tool.py"


def test_removed_modes_are_not_advertised():
    values = {m.value for m in GraphMode}
    assert "dead_code" not in values
    assert "dynamic_imports" not in values


def test_every_graph_mode_has_a_dispatch_handler():
    source = _GRAPH_TOOL.read_text(encoding="utf-8")
    # Strip the enum member declarations so a mode's literal must appear in real
    # dispatch logic, not merely in its own `NAME = "value"` definition.
    enum_decls = {f'{m.name} = "{m.value}"' for m in GraphMode}
    body = "\n".join(
        line
        for line in source.splitlines()
        if line.strip().split("  #")[0].strip() not in enum_decls
    )
    missing = [
        m.value
        for m in GraphMode
        if f'"{m.value}"' not in body and f"GraphMode.{m.name}" not in body
    ]
    assert not missing, (
        "These GraphMode members are advertised but have no dispatch handler "
        f"(they would hit 'Unsupported graph mode'): {missing}"
    )
