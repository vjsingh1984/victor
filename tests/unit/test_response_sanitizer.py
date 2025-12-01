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

"""Tests for ResponseSanitizer module."""

import pytest

from victor.agent.response_sanitizer import (
    ResponseSanitizer,
    sanitize_response,
    is_garbage_content,
    is_valid_tool_name,
    strip_markup,
)


class TestResponseSanitizer:
    """Tests for ResponseSanitizer class."""

    @pytest.fixture
    def sanitizer(self):
        """Create a ResponseSanitizer instance."""
        return ResponseSanitizer()

    def test_strip_markup_removes_html_tags(self, sanitizer):
        """Test that HTML/XML tags are removed."""
        text = "<p>Hello <b>World</b></p>"
        result = sanitizer.strip_markup(text)
        assert result == "Hello World"

    def test_strip_markup_handles_empty_string(self, sanitizer):
        """Test empty string handling."""
        assert sanitizer.strip_markup("") == ""
        assert sanitizer.strip_markup(None) is None

    def test_sanitize_removes_repeated_closing_tags(self, sanitizer):
        """Test removal of repeated closing tags."""
        text = "Hello</function></function></function></function>"
        result = sanitizer.sanitize(text)
        assert "</function>" not in result or result.count("</function>") < 3

    def test_sanitize_removes_orphaned_tags(self, sanitizer):
        """Test removal of orphaned XML-like tags."""
        text = "Hello <function>test</function> World"
        result = sanitizer.sanitize(text)
        assert "<function>" not in result
        assert "</function>" not in result

    def test_sanitize_removes_parameter_tags(self, sanitizer):
        """Test removal of parameter tags."""
        text = "Result <parameter>value</parameter> here"
        result = sanitizer.sanitize(text)
        assert "<parameter>" not in result

    def test_sanitize_removes_json_tool_calls(self, sanitizer):
        """Test removal of embedded JSON tool calls."""
        text = 'Here is the result {"name": "read_file", "arguments": {"path": "test.py"}} done'
        result = sanitizer.sanitize(text)
        assert '{"name":' not in result

    def test_sanitize_handles_empty_string(self, sanitizer):
        """Test empty string handling."""
        assert sanitizer.sanitize("") == ""
        assert sanitizer.sanitize(None) is None

    def test_sanitize_preserves_clean_text(self, sanitizer):
        """Test that clean text is preserved."""
        text = "This is a normal response with no issues."
        result = sanitizer.sanitize(text)
        assert result == text

    def test_is_garbage_content_detects_function_call_syntax(self, sanitizer):
        """Test garbage detection for raw function call syntax."""
        assert sanitizer.is_garbage_content("FUNCTION_CALL {")
        assert sanitizer.is_garbage_content("FUNCTION_CALL{name}")

    def test_is_garbage_content_detects_repeated_tags(self, sanitizer):
        """Test garbage detection for repeated closing tags."""
        assert sanitizer.is_garbage_content("</function></function>")

    def test_is_garbage_content_detects_instruction_leakage(self, sanitizer):
        """Test garbage detection for instruction leakage."""
        assert sanitizer.is_garbage_content("<IMPORTANT>Do not...")
        assert sanitizer.is_garbage_content("Do NOT mention...")
        assert sanitizer.is_garbage_content("NEVER use this")

    def test_is_garbage_content_detects_tool_request_format(self, sanitizer):
        """Test garbage detection for LMStudio format."""
        assert sanitizer.is_garbage_content("[TOOL_REQUEST]")
        assert sanitizer.is_garbage_content("[END_TOOL_REQUEST]")

    def test_is_garbage_content_returns_false_for_clean_text(self, sanitizer):
        """Test that clean text is not flagged as garbage."""
        assert not sanitizer.is_garbage_content("This is a normal response.")
        assert not sanitizer.is_garbage_content("Here are the file contents...")

    def test_is_garbage_content_handles_empty(self, sanitizer):
        """Test empty string handling."""
        assert not sanitizer.is_garbage_content("")
        assert not sanitizer.is_garbage_content(None)

    def test_is_valid_tool_name_accepts_valid_names(self, sanitizer):
        """Test valid tool names are accepted."""
        assert sanitizer.is_valid_tool_name("read_file")
        assert sanitizer.is_valid_tool_name("execute_bash")
        assert sanitizer.is_valid_tool_name("listDirectory")
        assert sanitizer.is_valid_tool_name("tool123")

    def test_is_valid_tool_name_rejects_example_names(self, sanitizer):
        """Test that example/hallucinated names are rejected."""
        assert not sanitizer.is_valid_tool_name("example_tool")
        assert not sanitizer.is_valid_tool_name("func_test")
        assert not sanitizer.is_valid_tool_name("function_name")
        assert not sanitizer.is_valid_tool_name("tool_name")
        assert not sanitizer.is_valid_tool_name("my_custom_tool")
        assert not sanitizer.is_valid_tool_name("test_tool")
        assert not sanitizer.is_valid_tool_name("sample_tool")

    def test_is_valid_tool_name_rejects_malformed_names(self, sanitizer):
        """Test that malformed names are rejected."""
        assert not sanitizer.is_valid_tool_name("<function>")
        assert not sanitizer.is_valid_tool_name("tool/path")
        assert not sanitizer.is_valid_tool_name("tool with space")
        assert not sanitizer.is_valid_tool_name("123tool")
        assert not sanitizer.is_valid_tool_name("")
        assert not sanitizer.is_valid_tool_name(None)


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_sanitize_response_function(self):
        """Test the sanitize_response convenience function."""
        text = "Hello <function>test</function> World"
        result = sanitize_response(text)
        assert "<function>" not in result

    def test_is_garbage_content_function(self):
        """Test the is_garbage_content convenience function."""
        assert is_garbage_content("FUNCTION_CALL {")
        assert not is_garbage_content("Normal text")

    def test_is_valid_tool_name_function(self):
        """Test the is_valid_tool_name convenience function."""
        assert is_valid_tool_name("read_file")
        assert not is_valid_tool_name("example_tool")

    def test_strip_markup_function(self):
        """Test the strip_markup convenience function."""
        assert strip_markup("<p>Hello</p>") == "Hello"
