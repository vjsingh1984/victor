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

"""Shared CLI/web markdown presenters (no Chainlit dependency)."""

from __future__ import annotations

from victor.ui.rendering.markdown_presenters import (
    preview_to_markdown,
    strip_rich_markup,
    tool_call_summary,
    tool_result_markdown,
)
from victor.ui.rendering.tool_preview import RenderedPreview


def test_strip_rich_markup():
    assert strip_rich_markup("[green]+added[/] [dim]ctx[/]") == "+added ctx"


def test_tool_call_summary_includes_args():
    summary = tool_call_summary("bash", {"command": "ls -la"})
    assert "command" in summary and "ls -la" in summary
    assert tool_call_summary("read", {}) == "read()"


def test_preview_to_markdown_diff_fence_and_header():
    preview = RenderedPreview(
        lines=["+ added", "- removed"],
        header="+1 -1 file.py",
        total_line_count=2,
        syntax_hint="diff",
    )
    md = preview_to_markdown(preview)
    assert "```diff" in md
    assert "+ added" in md and "- removed" in md
    assert "**+1 -1 file.py**" in md


def test_preview_to_markdown_strips_rich_markup_lines():
    preview = RenderedPreview(
        lines=["[green]+ a[/]", "[red]- b[/]"],
        total_line_count=2,
        syntax_hint="diff",
        contains_rich_markup=True,
    )
    md = preview_to_markdown(preview)
    assert "[green]" not in md and "[/]" not in md
    assert "+ a" in md and "- b" in md


def test_preview_to_markdown_truncation_footer():
    preview = RenderedPreview(lines=["line 1"], total_line_count=10, syntax_hint="text")
    md = preview_to_markdown(preview)
    assert "9 more lines" in md


def test_tool_result_markdown_renders_a_fence():
    md = tool_result_markdown("ls", {"path": "."}, "file1\nfile2", success=True)
    assert "```" in md
    assert "file1" in md or "file" in md


def test_tool_result_markdown_marks_failure():
    md = tool_result_markdown("bash", {"command": "false"}, "boom", success=False)
    assert "failed" in md.lower()


def test_tool_result_markdown_never_raises_on_bad_result():
    # A non-string, non-dict result must not crash the presenter.
    md = tool_result_markdown("weird", None, object(), success=True)
    assert isinstance(md, str) and md
