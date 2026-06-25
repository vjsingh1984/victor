import pytest
from unittest.mock import patch, AsyncMock
from victor.tools.unified.web_tool import web_tool


@pytest.mark.asyncio
async def test_web_tool_fetch():
    """Test `web fetch` subcommand formatting."""
    with patch("victor.tools.unified.web_tool.fetch_url", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = "<html><body>Hello</body></html>"

        result = await web_tool('web fetch "https://example.com"')

        mock_fetch.assert_called_once_with("https://example.com")
        # Ensure it returns the raw string cleanly
        assert "<html>" in result
        assert "### ❌ ERROR" not in result


@pytest.mark.asyncio
async def test_web_tool_search():
    """Test `web search` subcommand."""
    with patch("victor.tools.unified.web_tool.search_web", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [
            {"title": "Result 1", "url": "https://url1", "snippet": "Snippet 1"},
            {"title": "Result 2", "url": "https://url2", "snippet": "Snippet 2"},
        ]

        result = await web_tool('web search "python async"')

        mock_search.assert_called_once_with("python async")
        # Check markdown formatting for search results
        assert "[Result 1](https://url1)" in result
        assert "Snippet 1" in result
        assert "[Result 2](https://url2)" in result


@pytest.mark.asyncio
async def test_web_tool_render_error():
    """Test `web render` error formatting."""
    with patch("victor.tools.unified.web_tool.render_page", new_callable=AsyncMock) as mock_render:
        mock_render.side_effect = ConnectionError("Timeout")

        result = await web_tool('web render "https://badurl.com"')

        assert "### ❌ ERROR" in result
        assert "Timeout" in result
