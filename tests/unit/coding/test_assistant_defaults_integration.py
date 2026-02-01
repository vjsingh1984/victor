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

"""Tests for Coding assistant integration with shared defaults.

Tests that Coding assistant properly uses shared tool defaults from
victor.core.verticals.defaults to eliminate code duplication.

Note: Coding is currently using framework capabilities (FileOperationsCapability)
through the capability injector. This test suite validates migration to shared
defaults for consistency with other verticals.
"""


from victor.core.verticals.defaults.tool_defaults import COMMON_REQUIRED_TOOLS
from victor.framework.tool_naming import ToolNames
from victor.coding.assistant import CodingAssistant


class TestCodingDefaultsIntegration:
    """Tests for Coding assistant using shared defaults."""

    def test_coding_uses_common_required_tools(self):
        """Coding should extend COMMON_REQUIRED_TOOLS."""
        assistant = CodingAssistant()
        tools = assistant.get_tools()

        # All common tools should be present
        for tool in COMMON_REQUIRED_TOOLS:
            assert tool in tools, f"Coding should include common tool: {tool}"

    def test_coding_has_git_tool(self):
        """Coding should have git tool for version control."""
        assistant = CodingAssistant()
        tools = assistant.get_tools()

        assert ToolNames.GIT in tools, "Coding should have git tool"

    def test_coding_has_shell_tool(self):
        """Coding should have shell tool for command execution."""
        assistant = CodingAssistant()
        tools = assistant.get_tools()

        assert ToolNames.SHELL in tools, "Coding should have shell tool"

    def test_coding_has_search_tools(self):
        """Coding should have search tools for code exploration."""
        assistant = CodingAssistant()
        tools = assistant.get_tools()

        assert ToolNames.CODE_SEARCH in tools, "Coding should have code_search tool"
        assert ToolNames.SYMBOL in tools, "Coding should have symbol tool"

    def test_coding_has_refactoring_tools(self):
        """Coding should have refactoring tools."""
        assistant = CodingAssistant()
        tools = assistant.get_tools()

        assert ToolNames.RENAME in tools, "Coding should have rename tool"
        assert ToolNames.EXTRACT in tools, "Coding should have extract tool"

    def test_coding_has_lsp_tools(self):
        """Coding should have LSP tools for code intelligence."""
        assistant = CodingAssistant()
        tools = assistant.get_tools()

        assert ToolNames.LSP in tools, "Coding should have LSP tool"
        assert ToolNames.REFS in tools, "Coding should have refs tool"

    def test_coding_has_test_tool(self):
        """Coding should have test tool for running tests."""
        assistant = CodingAssistant()
        tools = assistant.get_tools()

        assert ToolNames.TEST in tools, "Coding should have test tool"

    def test_coding_has_docker_tool(self):
        """Coding should have docker tool for container operations."""
        assistant = CodingAssistant()
        tools = assistant.get_tools()

        assert ToolNames.DOCKER in tools, "Coding should have docker tool"

    def test_coding_has_web_tools(self):
        """Coding should have web tools for documentation lookup."""
        assistant = CodingAssistant()
        tools = assistant.get_tools()

        assert ToolNames.WEB_SEARCH in tools, "Coding should have web_search tool"
        assert ToolNames.WEB_FETCH in tools, "Coding should have web_fetch tool"

    def test_coding_no_duplicate_tools(self):
        """Coding should not have duplicate tools."""
        assistant = CodingAssistant()
        tools = assistant.get_tools()

        # Check for duplicates
        tool_set = set(tools)
        assert len(tools) == len(tool_set), "Coding should not have duplicate tools"

    def test_coding_tools_are_strings(self):
        """Coding tools should be strings (canonical names)."""
        assistant = CodingAssistant()
        tools = assistant.get_tools()

        for tool in tools:
            assert isinstance(tool, str), f"Tool should be string: {tool}"

    def test_coding_tool_count_reasonable(self):
        """Coding should have reasonable number of tools."""
        assistant = CodingAssistant()
        tools = assistant.get_tools()

        # Should have at least common tools (4) + coding tools (14+)
        assert len(tools) >= 18, f"Coding should have at least 18 tools, got {len(tools)}"
        # Should not have excessive tools
        assert len(tools) < 60, f"Coding should have less than 60 tools, got {len(tools)}"


class TestCodingSharedDefaultsUsage:
    """Tests for Coding explicitly importing and using shared defaults."""

    def test_coding_imports_merge_required_tools(self):
        """Coding assistant should import merge_required_tools from defaults.

        This test verifies that Coding is using the shared defaults infrastructure
        rather than hardcoding tool lists or using only capability injector.

        Note: Coding is currently migrating from capability injector to shared defaults.
        """
        import victor.coding.assistant as coding_assistant_module
        import inspect

        # Check if the module imports merge_required_tools
        source = inspect.getsource(coding_assistant_module)

        # Look for import statement
        has_import = (
            "from victor.core.verticals.defaults.tool_defaults import" in source
            or "from victor.core.verticals.defaults import" in source
        )

        # This test will FAIL initially, then PASS after migration
        assert has_import, "Coding should import from victor.core.verticals.defaults"

    def test_coding_uses_merge_required_tools_in_get_tools(self):
        """Coding get_tools() should use merge_required_tools().

        After migration, Coding should use merge_required_tools() to combine
        COMMON_REQUIRED_TOOLS with coding-specific tools.

        Note: Coding is currently migrating from capability injector to shared defaults.
        """
        import inspect

        # Get the source code of get_tools method
        source = inspect.getsource(CodingAssistant.get_tools)

        # Check if merge_required_tools is called
        uses_merge = "merge_required_tools" in source

        # This test will FAIL initially, then PASS after migration
        assert uses_merge, "Coding.get_tools() should call merge_required_tools()"
