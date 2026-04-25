"""Unit tests for formatter base classes and data structures."""

import pytest

from victor.tools.formatters.base import FormattedOutput, ToolFormatter


class TestFormattedOutput:
    """Test FormattedOutput dataclass."""

    def test_default_values(self):
        """Test FormattedOutput with default values."""
        output = FormattedOutput()

        assert output.content == ""
        assert output.format_type == "rich"
        assert output.summary is None
        assert output.metadata == {}
        assert output.line_count == 0
        assert output.contains_markup is False

    def test_with_content(self):
        """Test FormattedOutput with content."""
        output = FormattedOutput(content="Test output")

        assert output.content == "Test output"
        assert output.line_count == 1

    def test_with_all_fields(self):
        """Test FormattedOutput with all fields populated."""
        output = FormattedOutput(
            content="[green]Test[/]",
            format_type="rich",
            summary="Test summary",
            metadata={"key": "value"},
            line_count=1,
            contains_markup=True,
        )

        assert output.content == "[green]Test[/]"
        assert output.format_type == "rich"
        assert output.summary == "Test summary"
        assert output.metadata == {"key": "value"}
        assert output.line_count == 1
        assert output.contains_markup is True

    def test_line_count_multiline(self):
        """Test line_count with multiline content."""
        content = "Line 1\nLine 2\nLine 3"
        output = FormattedOutput(content=content)

        assert output.line_count == 3


class TestToolFormatter:
    """Test ToolFormatter base protocol."""

    def test_cannot_instantiate_abstract(self):
        """Test that ToolFormatter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ToolFormatter()

    def test_format_is_abstract(self):
        """Test that format() must be implemented by subclasses."""

        class ConcreteFormatter(ToolFormatter):
            pass

        with pytest.raises(TypeError, match="abstract"):
            ConcreteFormatter()

    def test_validate_input_default(self):
        """Test default validate_input() accepts all data."""

        class ConcreteFormatter(ToolFormatter):
            def format(self, data, **kwargs):
                return FormattedOutput(content="test")

        formatter = ConcreteFormatter()
        assert formatter.validate_input({}) is True
        assert formatter.validate_input(None) is True
        assert formatter.validate_input("string") is True

    def test_get_fallback_default(self):
        """Test default get_fallback() returns None."""

        class ConcreteFormatter(ToolFormatter):
            def format(self, data, **kwargs):
                return FormattedOutput(content="test")

        formatter = ConcreteFormatter()
        assert formatter.get_fallback() is None

    def test_concrete_implementation(self):
        """Test that concrete implementation works correctly."""

        class ConcreteFormatter(ToolFormatter):
            def format(self, data, **kwargs):
                content = data.get("content", "")
                return FormattedOutput(
                    content=content,
                    format_type="rich",
                    summary="Concrete output",
                    contains_markup=False,
                )

        formatter = ConcreteFormatter()
        result = formatter.format({"content": "Test content"})

        assert isinstance(result, FormattedOutput)
        assert result.content == "Test content"
        assert result.summary == "Concrete output"
        assert result.contains_markup is False
