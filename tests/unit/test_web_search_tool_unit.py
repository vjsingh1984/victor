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

"""Tests for web_search_tool module."""

from unittest.mock import MagicMock

from victor.tools.web_search_tool import (
    set_web_tool_defaults,
    set_web_search_provider,
    _parse_ddg_results,
    _config,
)


class TestSetWebToolDefaults:
    """Tests for set_web_tool_defaults function."""

    def test_set_fetch_top(self):
        """Test setting fetch_top."""
        original = _config["fetch_top"]
        set_web_tool_defaults(fetch_top=5)
        assert _config["fetch_top"] == 5
        _config["fetch_top"] = original

    def test_set_fetch_pool(self):
        """Test setting fetch_pool."""
        original = _config["fetch_pool"]
        set_web_tool_defaults(fetch_pool=10)
        assert _config["fetch_pool"] == 10
        _config["fetch_pool"] = original

    def test_set_max_content_length(self):
        """Test setting max_content_length."""
        original = _config["max_content_length"]
        set_web_tool_defaults(max_content_length=10000)
        assert _config["max_content_length"] == 10000
        _config["max_content_length"] = original


class TestSetWebSearchProvider:
    """Tests for set_web_search_provider function."""

    def test_set_provider(self):
        """Test setting provider."""
        mock_provider = MagicMock()
        set_web_search_provider(mock_provider, model="test-model")
        # Provider is set globally - just verify no exception


class TestParseDDGResults:
    """Tests for _parse_ddg_results function."""

    def test_parse_empty_html(self):
        """Test parsing empty HTML."""
        results = _parse_ddg_results("<html></html>", max_results=10)
        assert results == []

    def test_parse_valid_results(self):
        """Test parsing valid DuckDuckGo results."""
        html = """
        <html>
        <body>
        <div class="result">
            <a class="result__a" href="https://example.com">Example Title</a>
            <a class="result__snippet">This is a test snippet</a>
        </div>
        <div class="result">
            <a class="result__a" href="https://test.com">Test Title</a>
            <a class="result__snippet">Another snippet</a>
        </div>
        </body>
        </html>
        """
        results = _parse_ddg_results(html, max_results=10)
        assert len(results) == 2
        assert results[0]["title"] == "Example Title"
        assert results[0]["url"] == "https://example.com"

    def test_parse_respects_max_results(self):
        """Test that max_results is respected."""
        html = """
        <html>
        <body>
        <div class="result">
            <a class="result__a" href="https://a.com">A</a>
        </div>
        <div class="result">
            <a class="result__a" href="https://b.com">B</a>
        </div>
        <div class="result">
            <a class="result__a" href="https://c.com">C</a>
        </div>
        </body>
        </html>
        """
        results = _parse_ddg_results(html, max_results=2)
        assert len(results) <= 2

    def test_parse_result_missing_link(self):
        """Test parsing result without link."""
        html = """
        <html>
        <body>
        <div class="result">
            <span>No link here</span>
        </div>
        </body>
        </html>
        """
        results = _parse_ddg_results(html, max_results=10)
        assert results == []
