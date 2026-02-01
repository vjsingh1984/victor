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

"""Tests for DevOps assistant integration with shared defaults.

Tests that DevOps assistant properly uses shared tool defaults from
victor.core.verticals.defaults to eliminate code duplication.
"""


from victor.core.verticals.defaults.tool_defaults import COMMON_REQUIRED_TOOLS
from victor.framework.tool_naming import ToolNames
from victor.devops.assistant import DevOpsAssistant


class TestDevOpsDefaultsIntegration:
    """Tests for DevOps assistant using shared defaults."""

    def test_devops_uses_common_required_tools(self):
        """DevOps should extend COMMON_REQUIRED_TOOLS."""
        assistant = DevOpsAssistant()
        tools = assistant.get_tools()

        # All common tools should be present
        for tool in COMMON_REQUIRED_TOOLS:
            assert tool in tools, f"DevOps should include common tool: {tool}"

    def test_devops_has_devops_specific_tools(self):
        """DevOps should add its specific tools."""
        assistant = DevOpsAssistant()
        tools = assistant.get_tools()

        # DevOps-specific tools
        assert ToolNames.SHELL in tools, "DevOps should have shell tool"
        assert ToolNames.GIT in tools, "DevOps should have git tool"
        assert ToolNames.DOCKER in tools, "DevOps should have docker tool"
        assert ToolNames.TEST in tools, "DevOps should have test tool"

    def test_devops_has_search_tools(self):
        """DevOps should have search tools for configuration discovery."""
        assistant = DevOpsAssistant()
        tools = assistant.get_tools()

        assert ToolNames.GREP in tools, "DevOps should have grep tool"
        assert ToolNames.CODE_SEARCH in tools, "DevOps should have code_search tool"

    def test_devops_has_web_tools(self):
        """DevOps should have web tools for documentation lookup."""
        assistant = DevOpsAssistant()
        tools = assistant.get_tools()

        assert ToolNames.WEB_SEARCH in tools, "DevOps should have web_search tool"
        assert ToolNames.WEB_FETCH in tools, "DevOps should have web_fetch tool"

    def test_devops_has_overview_tool(self):
        """DevOps should have overview tool for codebase understanding."""
        assistant = DevOpsAssistant()
        tools = assistant.get_tools()

        assert ToolNames.OVERVIEW in tools, "DevOps should have overview tool"

    def test_devops_no_duplicate_tools(self):
        """DevOps should not have duplicate tools."""
        assistant = DevOpsAssistant()
        tools = assistant.get_tools()

        # Check for duplicates
        tool_set = set(tools)
        assert len(tools) == len(tool_set), "DevOps should not have duplicate tools"

    def test_devops_tools_are_strings(self):
        """DevOps tools should be strings (canonical names)."""
        assistant = DevOpsAssistant()
        tools = assistant.get_tools()

        for tool in tools:
            assert isinstance(tool, str), f"Tool should be string: {tool}"

    def test_devops_tool_count_reasonable(self):
        """DevOps should have reasonable number of tools."""
        assistant = DevOpsAssistant()
        tools = assistant.get_tools()

        # Should have at least common tools (4) + DevOps tools (5+)
        assert len(tools) >= 9, f"DevOps should have at least 9 tools, got {len(tools)}"
        # Should not have excessive tools
        assert len(tools) < 30, f"DevOps should have less than 30 tools, got {len(tools)}"


class TestDevOpsSharedDefaultsUsage:
    """Tests for DevOps explicitly importing and using shared defaults."""

    def test_devops_imports_merge_required_tools(self):
        """DevOps assistant should import merge_required_tools from defaults.

        This test verifies that DevOps is using the shared defaults infrastructure
        rather than hardcoding tool lists.
        """
        import victor.devops.assistant as devops_assistant_module
        import inspect

        # Check if the module imports merge_required_tools
        source = inspect.getsource(devops_assistant_module)

        # Look for import statement
        has_import = (
            "from victor.core.verticals.defaults.tool_defaults import" in source
            or "from victor.core.verticals.defaults import" in source
        )

        # This test will FAIL initially, then PASS after migration
        assert has_import, "DevOps should import from victor.core.verticals.defaults"

    def test_devops_uses_merge_required_tools_in_get_tools(self):
        """DevOps get_tools() should use merge_required_tools()."""
        import inspect

        # Get the source code of get_tools method
        source = inspect.getsource(DevOpsAssistant.get_tools)

        # Check if merge_required_tools is called
        uses_merge = "merge_required_tools" in source

        # This test will FAIL initially, then PASS after migration
        assert uses_merge, "DevOps.get_tools() should call merge_required_tools()"
