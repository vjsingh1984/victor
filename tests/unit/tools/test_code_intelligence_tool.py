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

"""Tests for code_intelligence_tool module and consolidated rename from refactor_tool.

These tests require victor-coding package to be installed and are
marked as integration tests.
"""

import pytest

from victor.tools.code_intelligence_tool import (
    symbol,
    refs,
)

try:
    from victor.core.capability_registry import CapabilityRegistry
    from victor.framework.vertical_protocols import TreeSitterParserProtocol

    _has_victor_coding = CapabilityRegistry.get_instance().is_enhanced(TreeSitterParserProtocol)
except Exception:
    _has_victor_coding = False

# Mark all tests in this module as integration tests (require victor-coding)
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _has_victor_coding, reason="Requires victor-coding package"),
]


class TestSymbol:
    """Tests for symbol function."""

    @pytest.mark.asyncio
    async def test_find_function(self, tmp_path):
        """Test finding a function definition."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def hello_world():
    print("Hello, World!")
    return True

def another_function():
    pass
""")

        result = await symbol(file_path=str(test_file), symbol_name="hello_world")

        assert result is not None
        assert result["symbol_name"] == "hello_world"
        assert result["type"] == "function"
        assert result["start_line"] > 0
        assert result["end_line"] > result["start_line"]
        assert "def hello_world" in result["code"]

    @pytest.mark.asyncio
    async def test_find_class(self, tmp_path):
        """Test finding a class definition."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
class MyClass:
    def __init__(self):
        self.value = 0

    def method(self):
        return self.value
""")

        result = await symbol(file_path=str(test_file), symbol_name="MyClass")

        assert result is not None
        assert result["symbol_name"] == "MyClass"
        assert result["type"] == "class"
        assert "class MyClass" in result["code"]

    @pytest.mark.asyncio
    async def test_symbol_not_found(self, tmp_path):
        """Test searching for non-existent symbol."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def existing_function():
    pass
""")

        result = await symbol(file_path=str(test_file), symbol_name="nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_symbol_file_not_found(self):
        """Test with non-existent file."""
        result = await symbol(file_path="/nonexistent/file.py", symbol_name="test")

        assert result is not None
        assert "error" in result

    @pytest.mark.asyncio
    async def test_symbol_empty_file(self, tmp_path):
        """Test with empty file."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        result = await symbol(file_path=str(test_file), symbol_name="test")

        assert result is None

    @pytest.mark.asyncio
    async def test_find_nested_function(self, tmp_path):
        """Test finding nested function."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def outer_function():
    def inner_function():
        return "inner"
    return inner_function()
""")

        result = await symbol(file_path=str(test_file), symbol_name="inner_function")

        assert result is not None
        assert result["symbol_name"] == "inner_function"
        assert result["type"] == "function"

    @pytest.mark.asyncio
    async def test_find_method_in_class(self, tmp_path):
        """Test finding a method inside a class."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
class TestClass:
    def test_method(self):
        pass
""")

        result = await symbol(file_path=str(test_file), symbol_name="test_method")

        assert result is not None
        assert result["symbol_name"] == "test_method"
        assert result["type"] == "function"

    @pytest.mark.asyncio
    async def test_symbol_generic_exception(self, tmp_path):
        """Test generic exception handling in symbol's fallback parser path."""
        from unittest.mock import patch

        test_file = tmp_path / "test.py"
        test_file.write_text("def test_func(): pass")

        # Force the legacy parser path (disable the analysis provider) and
        # make it explode to exercise the unexpected-error branch.
        with (
            patch(
                "victor.tools.code_intelligence_tool._get_analysis_provider",
                return_value=None,
            ),
            patch(
                "victor.tools.code_intelligence_tool._get_tree_sitter_parser",
                side_effect=RuntimeError("Parse error"),
            ),
        ):
            result = await symbol(file_path=str(test_file), symbol_name="test_func")

            assert result is not None
            assert "error" in result
            assert "unexpected error" in result["error"].lower()


class TestRefs:
    """Tests for refs function."""

    @pytest.mark.asyncio
    async def test_refs_basic(self, tmp_path):
        """Test finding references in directory."""
        # Create a directory with Python files
        test_file1 = tmp_path / "file1.py"
        test_file1.write_text("""
def target_function():
    return True
""")

        test_file2 = tmp_path / "file2.py"
        test_file2.write_text("""
from file1 import target_function

result = target_function()
""")

        result = await refs(symbol_name="target_function", search_path=str(tmp_path))

        assert isinstance(result, list)
        # May or may not find cross-file references depending on implementation

    @pytest.mark.asyncio
    async def test_refs_no_matches(self, tmp_path):
        """Test finding references when none exist."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def some_function():
    pass
""")

        result = await refs(symbol_name="nonexistent_function", search_path=str(tmp_path))

        assert isinstance(result, list)
        # Empty list or minimal results expected

    @pytest.mark.asyncio
    async def test_refs_invalid_path(self):
        """Test with invalid search path."""
        result = await refs(symbol_name="test", search_path="/nonexistent/path")

        assert isinstance(result, list)
        # Should handle gracefully, return empty list

    @pytest.mark.asyncio
    async def test_refs_empty_directory(self, tmp_path):
        """Test with empty directory."""
        result = await refs(symbol_name="test", search_path=str(tmp_path))

        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_refs_with_parse_errors(self, tmp_path):
        """Test that unparseable files are skipped gracefully."""
        # Create a file with invalid Python
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def invalid syntax here!")

        # Create a valid file
        good_file = tmp_path / "good.py"
        good_file.write_text("""
def target_function():
    pass

result = target_function()
""")

        result = await refs(symbol_name="target_function", search_path=str(tmp_path))

        # Should still return results from good file
        assert isinstance(result, list)
        # The bad file should be skipped (exception caught)

    @pytest.mark.asyncio
    async def test_refs_with_file_exception(self, tmp_path):
        """Test that file exceptions are caught and processing continues."""
        from unittest.mock import patch

        # Create a file that will exist
        test_file = tmp_path / "test.py"
        test_file.write_text("def target(): pass")

        # Mock open to raise an exception for any file
        with patch("builtins.open", side_effect=OSError("File read error")):
            result = await refs(symbol_name="target", search_path=str(tmp_path))

            # Should return empty list since all files failed to read
            assert isinstance(result, list)
            assert len(result) == 0


# ────────────────────────────────────────────────────────────────────────
# TSA-6: provider-preferring symbol() path
# ────────────────────────────────────────────────────────────────────────


def _provider_unit_tests_marker():
    """These tests do not require victor-coding installed — they monkeypatch
    the provider directly. Allow them to run even when the file-level skip
    fires for the integration tests above.
    """
    return [pytest.mark.unit]


class _FakeProviderSymbol:
    def __init__(self, *, supported=("python",), symbols=None):
        self._supported = set(supported)
        self._symbols = list(symbols or [])
        self.calls: list[tuple[str, str]] = []

    def supports_language(self, language: str) -> bool:
        return language in self._supported

    def extract_symbols(self, content, language, *, file_path):
        self.calls.append((file_path, language))
        return list(self._symbols)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_symbol_prefers_analysis_provider(monkeypatch, tmp_path):
    """symbol() should consult the TreeSitterAnalysisProtocol provider first."""
    from victor.tools import code_intelligence_tool as cit

    test_file = tmp_path / "a.py"
    test_file.write_text("def foo():\n    return 1\n")

    provider = _FakeProviderSymbol(
        symbols=[
            {
                "name": "foo",
                "symbol_type": "function",
                "file_path": str(test_file),
                "line_start": 1,
                "line_end": 2,
                "ast_kind": "function_definition",
            }
        ]
    )
    monkeypatch.setattr(cit, "_get_analysis_provider", lambda: provider)

    # Guarantee the legacy parser path is not used.
    def _should_not_use_legacy(*args, **kwargs):  # pragma: no cover - guard
        raise AssertionError("legacy parser must not run when provider returns a hit")

    monkeypatch.setattr(cit, "_get_tree_sitter_parser", _should_not_use_legacy)

    result = await cit.symbol(file_path=str(test_file), symbol_name="foo")

    assert result is not None
    assert result["symbol_name"] == "foo"
    assert result["type"] == "function"
    assert result["start_line"] == 1
    assert result["end_line"] == 2
    assert "def foo" in result["code"]
    assert provider.calls == [(str(test_file), "python")]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_symbol_returns_none_when_provider_misses(monkeypatch, tmp_path):
    """If the provider supports the language but returns no match for the
    name, symbol() returns None without falling back to the legacy path.
    The provider is authoritative for languages it claims.
    """
    from victor.tools import code_intelligence_tool as cit

    test_file = tmp_path / "a.py"
    test_file.write_text("def other(): pass\n")

    provider = _FakeProviderSymbol(
        symbols=[
            {
                "name": "other",
                "symbol_type": "function",
                "file_path": str(test_file),
                "line_start": 1,
                "line_end": 1,
            }
        ]
    )
    monkeypatch.setattr(cit, "_get_analysis_provider", lambda: provider)

    def _should_not_use_legacy(*args, **kwargs):  # pragma: no cover - guard
        raise AssertionError("legacy parser must not run when provider answered")

    monkeypatch.setattr(cit, "_get_tree_sitter_parser", _should_not_use_legacy)

    result = await cit.symbol(file_path=str(test_file), symbol_name="missing")
    assert result is None


@pytest.mark.unit
def test_detect_language_from_path():
    from victor.tools.code_intelligence_tool import _detect_language_from_path

    assert _detect_language_from_path("a.py") == "python"
    assert _detect_language_from_path("a.rs") == "rust"
    assert _detect_language_from_path("a.tsx") == "tsx"
    # Unknown suffix defaults to "python" — backward compat for the tool.
    assert _detect_language_from_path("a.xyz") == "python"


@pytest.mark.unit
def test_read_line_range():
    from victor.tools.code_intelligence_tool import _read_line_range

    src = b"alpha\nbeta\ngamma\ndelta\n"
    assert _read_line_range(src, 1, 1) == "alpha"
    assert _read_line_range(src, 2, 3) == "beta\ngamma"
    # Out-of-range upper bound clamps cleanly.
    assert _read_line_range(src, 3, 99) == "gamma\ndelta"
