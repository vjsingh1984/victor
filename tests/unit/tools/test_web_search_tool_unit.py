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

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.tools.base import ToolConfig
from victor.tools.web_search_tool import (
    _parse_ddg_results,
    _get_web_config,
    web_fetch,
    web_search,
)


class TestGetWebConfig:
    """Tests for _get_web_config function."""

    def test_config_from_tool_config(self):
        """Test getting config from ToolConfig context."""
        mock_provider = MagicMock()
        tool_config = ToolConfig(
            provider=mock_provider,
            model="test-model",
            web_fetch_top=5,
            web_fetch_pool=10,
            max_content_length=8000,
            generic_result_cache_enabled=True,
            generic_result_cache_ttl=120,
            http_connection_pool_enabled=True,
            http_connection_pool_max_connections=64,
        )
        context = {"tool_config": tool_config}

        config = _get_web_config(context)

        assert config["provider"] == mock_provider
        assert config["model"] == "test-model"
        assert config["fetch_top"] == 5
        assert config["fetch_pool"] == 10
        assert config["max_content_length"] == 8000
        assert config["generic_result_cache_enabled"] is True
        assert config["generic_result_cache_ttl"] == 120
        assert config["http_connection_pool_enabled"] is True
        assert config["http_connection_pool_max_connections"] == 64

    def test_config_without_context(self):
        """Test getting config without context returns defaults."""
        config = _get_web_config(None)

        assert config["provider"] is None
        assert config["model"] is None
        assert config["fetch_top"] is None
        assert config["fetch_pool"] is None
        assert config["max_content_length"] == 5000
        assert config["generic_result_cache_enabled"] is False
        assert config["http_connection_pool_enabled"] is False

    def test_config_with_empty_context(self):
        """Test getting config with empty context returns defaults."""
        config = _get_web_config({})

        assert config["provider"] is None
        assert config["model"] is None


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


class TestRuntimeInfraIntegration:
    """Tests for runtime cache/pool integrations in web tools."""

    @pytest.mark.asyncio
    async def test_web_search_passes_pool_enabled_config_to_request_layer(self):
        """web_search should forward pool-enabled config to _request_text."""
        html = """
        <html><body>
        <div class="result">
            <a class="result__a" href="https://example.com">Example</a>
            <a class="result__snippet">Snippet</a>
        </div>
        </body></html>
        """
        tool_config = ToolConfig(http_connection_pool_enabled=True)
        context = {"tool_config": tool_config}

        with patch(
            "victor.tools.web_search_tool._request_text",
            new=AsyncMock(return_value=(200, html)),
        ) as mock_request:
            result = await web_search(query="test query", context=context)

        assert result["success"] is True
        assert mock_request.await_count == 1
        web_config = mock_request.await_args.kwargs["web_config"]
        assert web_config["http_connection_pool_enabled"] is True

    @pytest.mark.asyncio
    async def test_web_fetch_uses_generic_cache_on_repeated_calls(self, monkeypatch):
        """web_fetch should serve repeated identical requests from GenericResultCache."""
        monkeypatch.setattr("victor.tools.web_search_tool._GENERIC_WEB_CACHE", None)
        html = "<html><body><main>" + ("content " * 30) + "</main></body></html>"
        url = f"https://example.com/{uuid.uuid4().hex}"
        tool_config = ToolConfig(generic_result_cache_enabled=True, generic_result_cache_ttl=120)
        context = {"tool_config": tool_config}

        with patch(
            "victor.tools.web_search_tool._request_text",
            new=AsyncMock(return_value=(200, html)),
        ) as mock_request:
            first = await web_fetch(url, context=context)

        assert first["success"] is True
        assert mock_request.await_count == 1

        with patch(
            "victor.tools.web_search_tool._request_text",
            new=AsyncMock(side_effect=AssertionError("network should not be called")),
        ):
            second = await web_fetch(url, context=context)

        assert second["success"] is True
        assert second["cached"] is True
