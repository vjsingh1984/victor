"""Tests for tool name resolution and canonicalization."""

import pytest
from victor.tools.decorators import _resolve_tool_name


class TestToolNameResolution:
    """Tests for the _resolve_tool_name function."""

    def test_explicit_name_resolves_to_canonical(self):
        """When an explicit name is provided that has an alias, it should resolve to canonical."""
        # "search" is an alias for "code_search" in TOOL_ALIASES
        result = _resolve_tool_name("search_tool", "search")
        assert result == "code_search", f"Expected 'code_search', got '{result}'"

    def test_function_name_resolves_to_canonical(self):
        """When no explicit name is provided, function name should resolve to canonical."""
        # "execute_bash" is an alias for "shell"
        result = _resolve_tool_name("execute_bash", None)
        assert result == "shell", f"Expected 'shell', got '{result}'"

    def test_function_name_without_alias_unchanged(self):
        """When function name has no alias, it should remain unchanged."""
        result = _resolve_tool_name("custom_tool", None)
        assert result == "custom_tool", f"Expected 'custom_tool', got '{result}'"

    def test_explicit_name_without_alias_unchanged(self):
        """When explicit name has no alias, it should remain unchanged."""
        result = _resolve_tool_name("func", "custom_name")
        assert result == "custom_name", f"Expected 'custom_name', got '{result}'"

    def test_read_file_resolves_to_read(self):
        """Verify 'read_file' alias resolves to 'read' canonical name."""
        result = _resolve_tool_name("read_file", None)
        assert result == "read", f"Expected 'read', got '{result}'"

    def test_list_directory_resolves_to_ls(self):
        """Verify 'list_directory' alias resolves to 'ls' canonical name."""
        result = _resolve_tool_name("list_directory", None)
        assert result == "ls", f"Expected 'ls', got '{result}'"

    def test_write_file_resolves_to_write(self):
        """Verify 'write_file' alias resolves to 'write' canonical name."""
        result = _resolve_tool_name("write_file", None)
        assert result == "write", f"Expected 'write', got '{result}'"

    def test_bash_resolves_to_shell(self):
        """Verify 'bash' alias resolves to 'shell' canonical name."""
        result = _resolve_tool_name("bash", None)
        assert result == "shell", f"Expected 'shell', got '{result}'"

    def test_edit_files_resolves_to_edit(self):
        """Verify 'edit_files' alias resolves to 'edit' canonical name."""
        result = _resolve_tool_name("edit_files", None)
        assert result == "edit", f"Expected 'edit', got '{result}'"


class TestToolNameConsistency:
    """Tests for consistency between tool registration and LLM-facing names."""

    def test_search_tool_has_canonical_name(self):
        """Verify the search tool is registered with canonical name 'code_search'."""
        from victor.tools.unified.search_tool import search_tool

        # The tool should be registered as "code_search" (canonical)
        # not "search" (alias)
        # Access via .Tool property (the tool instance)
        assert search_tool.Tool.name == "code_search", (
            f"Tool name should be 'code_search' (canonical), "
            f"but got '{search_tool.Tool.name}'"
        )

    def test_search_tool_json_schema_has_canonical_name(self):
        """Verify the search tool's JSON schema uses the canonical name."""
        from victor.tools.unified.search_tool import search_tool

        schema = search_tool.Tool.to_json_schema()
        # Schema structure: {"type": "function", "function": {"name": "code_search", ...}}
        function_schema = schema.get("function", schema)  # Handle both formats
        assert function_schema["name"] == "code_search", (
            f"Schema name should be 'code_search' (canonical), "
            f"but got '{function_schema.get('name')}'"
        )
