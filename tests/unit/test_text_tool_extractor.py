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

"""Tests for the Python-style tool call text extractor."""

import pytest
from victor.agent.tool_calling.text_extractor import (
    PythonCallExtractor,
    ExtractionResult,
    extract_tool_calls_from_text,
    KNOWN_TOOL_PATTERNS,
)


class TestPythonCallExtractor:
    """Tests for PythonCallExtractor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = PythonCallExtractor()

    def test_simple_single_quoted_arg(self):
        """Test extraction with single-quoted argument."""
        content = "I'll read the file: read_file(path='foo.py')"
        result = self.extractor.extract_from_text(content)

        assert result.success
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "read_file"
        assert result.tool_calls[0].arguments == {"path": "foo.py"}

    def test_simple_double_quoted_arg(self):
        """Test extraction with double-quoted argument."""
        content = 'Let me run: shell(command="ls -la")'
        result = self.extractor.extract_from_text(content)

        assert result.success
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "shell"
        assert result.tool_calls[0].arguments == {"command": "ls -la"}

    def test_multiple_arguments(self):
        """Test extraction with multiple arguments."""
        content = "edit(file_path='/path/to/file', old_string='foo', new_string='bar')"
        result = self.extractor.extract_from_text(content)

        assert result.success
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.name == "edit"
        assert tc.arguments["file_path"] == "/path/to/file"
        assert tc.arguments["old_string"] == "foo"
        assert tc.arguments["new_string"] == "bar"

    def test_numeric_argument(self):
        """Test extraction with numeric argument."""
        content = "shell(command='sleep', timeout=30)"
        result = self.extractor.extract_from_text(content)

        assert result.success
        assert result.tool_calls[0].arguments.get("timeout") == 30

    def test_boolean_argument(self):
        """Test extraction with boolean argument."""
        content = "read(path='file.py', binary=True)"
        result = self.extractor.extract_from_text(content)

        assert result.success
        assert result.tool_calls[0].arguments.get("binary") is True

    def test_multiple_tool_calls(self):
        """Test extraction of multiple tool calls in one text."""
        content = """
        First, I'll read the file: read(path='input.py')
        Then I'll write the output: write(path='output.py', content='hello')
        """
        result = self.extractor.extract_from_text(content)

        assert result.success
        assert len(result.tool_calls) == 2
        names = {tc.name for tc in result.tool_calls}
        assert names == {"read", "write"}

    def test_valid_tool_names_filter(self):
        """Test filtering by valid tool names."""
        content = "I'll call read(path='foo.py') and unknown_function(arg=1)"
        result = self.extractor.extract_from_text(content, valid_tool_names={"read", "write"})

        assert result.success
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "read"

    def test_strict_mode(self):
        """Test strict mode only extracts known tools."""
        extractor = PythonCallExtractor(strict_mode=True)
        content = "some_random_func(arg=1) and read(path='foo.py')"
        result = extractor.extract_from_text(content)

        assert result.success
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "read"

    def test_empty_content(self):
        """Test handling of empty content."""
        result = self.extractor.extract_from_text("")
        assert not result.success
        assert len(result.tool_calls) == 0

    def test_no_tool_calls(self):
        """Test content with no tool calls."""
        content = "This is just regular text without any function calls."
        result = self.extractor.extract_from_text(content)

        assert not result.success
        assert len(result.tool_calls) == 0

    def test_remaining_content(self):
        """Test that remaining content is calculated correctly."""
        content = "Before read(path='foo.py') after"
        result = self.extractor.extract_from_text(content)

        assert result.success
        # The remaining content should have the tool call removed
        assert "read(path='foo.py')" not in result.remaining_content

    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        content = "read_file(path='test.py')"
        result = self.extractor.extract_from_text(content)

        assert result.success
        assert result.confidence > 0.5
        assert result.confidence <= 1.0

    def test_known_tool_patterns(self):
        """Test that known tool patterns are defined."""
        assert "read" in KNOWN_TOOL_PATTERNS
        assert "write" in KNOWN_TOOL_PATTERNS
        assert "shell" in KNOWN_TOOL_PATTERNS
        assert "edit" in KNOWN_TOOL_PATTERNS
        assert "grep" in KNOWN_TOOL_PATTERNS

    def test_convenience_function(self):
        """Test the convenience function."""
        result = extract_tool_calls_from_text("read(path='test.py')")

        assert result.success
        assert result.tool_calls[0].name == "read"


class TestPythonCallExtractorEdgeCases:
    """Edge case tests for PythonCallExtractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = PythonCallExtractor()

    def test_nested_quotes(self):
        """Test handling of nested quotes."""
        content = """shell(command="echo 'hello world'")"""
        result = self.extractor.extract_from_text(content)

        assert result.success
        assert result.tool_calls[0].arguments["command"] == "echo 'hello world'"

    def test_path_with_spaces(self):
        """Test path with spaces."""
        content = "read(path='/path/with spaces/file.py')"
        result = self.extractor.extract_from_text(content)

        assert result.success
        assert result.tool_calls[0].arguments["path"] == "/path/with spaces/file.py"

    def test_multiline_content(self):
        """Test multiline argument content."""
        content = """write(path='test.py', content='line1
line2
line3')"""
        result = self.extractor.extract_from_text(content)

        # Should still extract, though content may be partial
        assert result.success
        assert result.tool_calls[0].name == "write"

    def test_special_characters_in_path(self):
        """Test paths with special characters."""
        content = "read(path='/home/user/.config/app.yaml')"
        result = self.extractor.extract_from_text(content)

        assert result.success
        assert ".config" in result.tool_calls[0].arguments["path"]

    def test_underscore_tool_name(self):
        """Test tool names with underscores."""
        content = "read_file(path='test.py')"
        result = self.extractor.extract_from_text(content)

        assert result.success
        assert result.tool_calls[0].name == "read_file"

    def test_no_arguments(self):
        """Test function call with no arguments."""
        content = "ls()"
        result = self.extractor.extract_from_text(content)

        assert result.success
        assert result.tool_calls[0].name == "ls"
        assert result.tool_calls[0].arguments == {}

    def test_skip_python_builtins(self):
        """Test that Python builtins are skipped."""
        content = "print('hello') and len(data) and read(path='file.py')"
        result = self.extractor.extract_from_text(content)

        # Should only extract 'read', not 'print' or 'len'
        assert result.success
        names = [tc.name for tc in result.tool_calls]
        assert "read" in names
        assert "print" not in names
        assert "len" not in names


class TestFallbackParsingMixinIntegration:
    """Tests for FallbackParsingMixin integration with text extraction."""

    def test_parse_python_call_from_content(self):
        """Test the mixin method for Python call parsing."""
        from victor.agent.tool_calling.base import FallbackParsingMixin, ToolCallParseResult

        # Create a class that uses the mixin
        class TestMixin(FallbackParsingMixin):
            pass

        mixin = TestMixin()
        result = mixin.parse_python_call_from_content(
            "read(path='test.py')",
            validate_name_fn=None,
            valid_tool_names={"read"},
        )

        assert result.tool_calls
        assert result.tool_calls[0].name == "read"
        assert result.parse_method == "python_call_fallback"

    def test_parse_from_content_includes_python_calls(self):
        """Test that parse_from_content tries Python call parsing."""
        from victor.agent.tool_calling.base import FallbackParsingMixin

        class TestMixin(FallbackParsingMixin):
            pass

        mixin = TestMixin()
        result = mixin.parse_from_content(
            "Let me read: read(path='test.py')",
            valid_tool_names={"read"},
        )

        assert result.tool_calls
        assert result.tool_calls[0].name == "read"

    def test_openai_adapter_fallback(self):
        """Test that OpenAI adapter falls back to content parsing."""
        from victor.agent.tool_calling.adapters import OpenAIToolCallingAdapter

        adapter = OpenAIToolCallingAdapter(model="gpt-4")

        # Set valid tool names for extraction
        adapter._valid_tool_names = {"read", "write", "shell"}

        # Mock valid tool name check
        adapter.is_valid_tool_name = lambda name: name in {"read", "write", "shell"}

        # Parse with no raw_tool_calls (simulates model outputting text)
        result = adapter.parse_tool_calls(
            content="I'll read the file: read(path='test.py')",
            raw_tool_calls=None,
        )

        assert result.tool_calls
        assert result.tool_calls[0].name == "read"
        assert result.tool_calls[0].arguments == {"path": "test.py"}
        assert "python_call" in result.parse_method or "json" in result.parse_method
