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
        assert "search code" in result
        assert "A test tool" in result

    def test_tool_without_get_metadata(self):
        """Tools without get_metadata() still get a description."""
        from victor.tools.semantic_selector import SemanticToolSelector

        tool = MagicMock(spec=[])  # No get_metadata attribute
        tool.name = "legacy_tool"
        tool.description = "A legacy tool"

        result = SemanticToolSelector._create_tool_text(tool)
        assert "A legacy tool" in result

    def test_dead_methods_not_in_source(self):
        """_build_use_case_text and _get_tool_use_cases are removed from source."""
        import inspect
        from victor.tools import semantic_selector

        source = inspect.getsource(semantic_selector)
        # The method definitions should no longer exist
        assert "def _build_use_case_text" not in source
        assert "def _get_tool_use_cases" not in source
