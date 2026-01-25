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

"""Tests for RAG assistant integration with shared defaults.

Tests that RAG assistant properly uses shared tool defaults from
victor.core.verticals.defaults to eliminate code duplication.

Note: RAG is a read-only vertical (for analysis), so it uses
COMMON_READONLY_TOOLS instead of COMMON_REQUIRED_TOOLS.
"""

import pytest

from victor.core.verticals.defaults.tool_defaults import COMMON_READONLY_TOOLS
from victor.framework.tool_naming import ToolNames
from victor.rag.assistant import RAGAssistant


class TestRAGDefaultsIntegration:
    """Tests for RAG assistant using shared defaults."""

    def test_rag_uses_common_readonly_tools(self):
        """RAG should extend COMMON_READONLY_TOOLS (read-only subset)."""
        assistant = RAGAssistant()
        tools = assistant.get_tools()

        # All read-only common tools should be present
        for tool in COMMON_READONLY_TOOLS:
            assert tool in tools, f"RAG should include readonly tool: {tool}"

    def test_rag_has_rag_specific_tools(self):
        """RAG should add its specific tools."""
        assistant = RAGAssistant()
        tools = assistant.get_tools()

        # RAG-specific tools
        assert "rag_ingest" in tools, "RAG should have rag_ingest tool"
        assert "rag_search" in tools, "RAG should have rag_search tool"
        assert "rag_query" in tools, "RAG should have rag_query tool"
        assert "rag_list" in tools, "RAG should have rag_list tool"
        assert "rag_delete" in tools, "RAG should have rag_delete tool"
        assert "rag_stats" in tools, "RAG should have rag_stats tool"

    def test_rag_has_filesystem_tools(self):
        """RAG should have filesystem tools for document access."""
        assistant = RAGAssistant()
        tools = assistant.get_tools()

        assert ToolNames.READ in tools, "RAG should have read tool"
        assert ToolNames.LS in tools, "RAG should have ls tool"

    def test_rag_has_web_tools(self):
        """RAG should have web fetch tool for fetching web content."""
        assistant = RAGAssistant()
        tools = assistant.get_tools()

        assert ToolNames.WEB_FETCH in tools, "RAG should have web_fetch tool"

    def test_rag_has_shell_tool(self):
        """RAG should have shell tool for document processing."""
        assistant = RAGAssistant()
        tools = assistant.get_tools()

        assert ToolNames.SHELL in tools, "RAG should have shell tool"

    def test_rag_no_duplicate_tools(self):
        """RAG should not have duplicate tools."""
        assistant = RAGAssistant()
        tools = assistant.get_tools()

        # Check for duplicates
        tool_set = set(tools)
        assert len(tools) == len(tool_set), "RAG should not have duplicate tools"

    def test_rag_tools_are_strings(self):
        """RAG tools should be strings (canonical names)."""
        assistant = RAGAssistant()
        tools = assistant.get_tools()

        for tool in tools:
            assert isinstance(tool, str), f"Tool should be string: {tool}"

    def test_rag_tool_count_reasonable(self):
        """RAG should have reasonable number of tools."""
        assistant = RAGAssistant()
        tools = assistant.get_tools()

        # Should have at least common tools (4) + RAG tools (6) + additional tools
        assert len(tools) >= 10, f"RAG should have at least 10 tools, got {len(tools)}"
        # Should not have excessive tools
        assert len(tools) < 25, f"RAG should have less than 25 tools, got {len(tools)}"


class TestRAGSharedDefaultsUsage:
    """Tests for RAG explicitly importing and using shared defaults."""

    def test_rag_imports_merge_required_tools(self):
        """RAG assistant should import merge_required_tools from defaults.

        This test verifies that RAG is using the shared defaults infrastructure
        rather than hardcoding tool lists.
        """
        import victor.rag.assistant as rag_assistant_module
        import inspect

        # Check if the module imports merge_required_tools
        source = inspect.getsource(rag_assistant_module)

        # Look for import statement
        has_import = (
            "from victor.core.verticals.defaults.tool_defaults import" in source
            or "from victor.core.verticals.defaults import" in source
        )

        # This test will FAIL initially, then PASS after migration
        assert has_import, "RAG should import from victor.core.verticals.defaults"

    def test_rag_uses_merge_required_tools_in_get_tools(self):
        """RAG get_tools() should use merge_required_tools()."""
        import victor.rag.assistant as rag_assistant_module
        import inspect

        # Get the source code of get_tools method
        source = inspect.getsource(RAGAssistant.get_tools)

        # Check if merge_required_tools is called
        uses_merge = "merge_required_tools" in source

        # This test will FAIL initially, then PASS after migration
        assert uses_merge, "RAG.get_tools() should call merge_required_tools()"
