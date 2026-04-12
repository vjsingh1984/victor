"""Tests that semantic selector works after dead stub removal."""
from unittest.mock import MagicMock


class TestCreateToolTextAfterCleanup:
    """Verify _create_tool_text works for tools with and without get_metadata."""

    def test_tool_with_get_metadata(self):
        """Tools with get_metadata() use it for descriptive text."""
        from victor.tools.semantic_selector import SemanticToolSelector

        tool = MagicMock()
        tool.name = "test_tool"
        tool.description = "A test tool"
        metadata = MagicMock()
        metadata.use_cases = ["search code", "find files"]
        metadata.keywords = ["search", "find"]
        metadata.examples = ["search for function"]
        tool.get_metadata.return_value = metadata

        result = SemanticToolSelector._create_tool_text(tool)
        assert "test_tool" in result
        assert "search code" in result

    def test_tool_without_get_metadata(self):
        """Tools without get_metadata() still get a description."""
        from victor.tools.semantic_selector import SemanticToolSelector

        tool = MagicMock(spec=[])  # No get_metadata attribute
        tool.name = "legacy_tool"
        tool.description = "A legacy tool"

        result = SemanticToolSelector._create_tool_text(tool)
        assert "legacy_tool" in result
        assert "A legacy tool" in result

    def test_dead_methods_removed(self):
        """_build_use_case_text and _get_tool_use_cases no longer exist."""
        from victor.tools.semantic_selector import SemanticToolSelector

        assert not hasattr(SemanticToolSelector, '_build_use_case_text')
        assert not hasattr(SemanticToolSelector, '_get_tool_use_cases')
