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

"""Tests for decorator-based tools (bash, cache, http, web_search).

This module tests the newly migrated decorator-based tools to ensure
they work correctly and increase code coverage.
"""

import json
import subprocess
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, ANY
from victor.tools.bash import execute_bash
from victor.tools.cache_tool import cache_stats, cache_clear, cache_info, set_cache_manager
from victor.tools.http_tool import http_request, http_test
from victor.tools.web_search_tool import (
    web_search,
    web_fetch,
    web_summarize,
    set_web_search_provider,
)


# ============================================================================
# Bash Tool Tests
# ============================================================================


@pytest.mark.asyncio
async def test_execute_bash_simple_command():
    """Test executing a simple bash command."""
    result = await execute_bash(command="echo 'hello'")

    assert result["success"] is True
    assert "hello" in result["stdout"]
    assert result["return_code"] == 0


@pytest.mark.asyncio
async def test_execute_bash_missing_command():
    """Test bash tool with missing command."""
    result = await execute_bash(command="")

    assert result["success"] is False
    assert "Missing required parameter" in result["error"]


@pytest.mark.asyncio
async def test_execute_bash_dangerous_command():
    """Test bash tool blocks dangerous commands."""
    result = await execute_bash(command="rm -rf /", allow_dangerous=False)

    assert result["success"] is False
    assert "Dangerous command blocked" in result["error"]


@pytest.mark.asyncio
async def test_execute_bash_allow_dangerous():
    """Test bash tool allows dangerous commands when explicitly allowed."""
    # This won't actually execute, just test the allow flag works
    with patch("asyncio.create_subprocess_shell") as mock_subprocess:
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))
        mock_subprocess.return_value = mock_process

        result = await execute_bash(command="rm -rf test", allow_dangerous=True)

        # Should attempt to execute
        mock_subprocess.assert_called_once()


@pytest.mark.asyncio
async def test_execute_bash_with_working_dir():
    """Test bash command with working directory."""
    result = await execute_bash(command="pwd", working_dir="/tmp")

    assert result["success"] is True
    assert result["working_dir"] == "/tmp"


@pytest.mark.asyncio
async def test_execute_bash_timeout():
    """Test bash command timeout."""
    with patch("asyncio.create_subprocess_shell") as mock_subprocess:
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=TimeoutError())
        mock_subprocess.return_value = mock_process

        result = await execute_bash(command="sleep 100", timeout=1)

        assert result["success"] is False
        assert "timed out" in result["error"] or "Failed to execute" in result["error"]


# ============================================================================
# Cache Tool Tests
# ============================================================================


@pytest.mark.asyncio
async def test_cache_stats_no_manager():
    """Test cache_stats with no manager set."""
    # Reset manager
    set_cache_manager(None)

    result = await cache_stats()

    assert result["success"] is False
    assert "not initialized" in result["error"]


@pytest.mark.asyncio
async def test_cache_stats_with_manager():
    """Test cache_stats with manager set."""
    mock_manager = MagicMock()
    mock_manager.get_stats.return_value = {
        "memory_hit_rate": 0.75,
        "disk_hit_rate": 0.60,
        "memory_hits": 150,
        "memory_misses": 50,
        "disk_hits": 120,
        "disk_misses": 80,
        "sets": 200,
        "memory_size": 50,
        "memory_max_size": 100,
        "disk_size": 300,
        "disk_volume": 1024000,
    }

    set_cache_manager(mock_manager)

    result = await cache_stats()

    assert result["success"] is True
    assert "stats" in result
    assert "formatted_report" in result
    assert result["stats"]["memory_hit_rate"] == 0.75
    assert "Cache Statistics" in result["formatted_report"]


@pytest.mark.asyncio
async def test_cache_clear_no_manager():
    """Test cache_clear with no manager set."""
    set_cache_manager(None)

    result = await cache_clear()

    assert result["success"] is False
    assert "not initialized" in result["error"]


@pytest.mark.asyncio
async def test_cache_clear_all():
    """Test cache_clear clearing all cache."""
    mock_manager = MagicMock()
    mock_manager.clear.return_value = 250

    set_cache_manager(mock_manager)

    result = await cache_clear(namespace=None)

    assert result["success"] is True
    assert result["cleared_count"] == 250
    assert "all cache" in result["message"]
    mock_manager.clear.assert_called_once_with(None)


@pytest.mark.asyncio
async def test_cache_clear_namespace():
    """Test cache_clear with specific namespace."""
    mock_manager = MagicMock()
    mock_manager.clear.return_value = 50

    set_cache_manager(mock_manager)

    result = await cache_clear(namespace="responses")

    assert result["success"] is True
    assert result["cleared_count"] == 50
    assert "responses" in result["message"]
    mock_manager.clear.assert_called_once_with("responses")


@pytest.mark.asyncio
async def test_cache_clear_error():
    """Test cache_clear with exception."""
    mock_manager = MagicMock()
    mock_manager.clear.side_effect = Exception("Clear failed")

    set_cache_manager(mock_manager)

    result = await cache_clear()

    assert result["success"] is False
    assert "Clear failed" in result["error"]


@pytest.mark.asyncio
async def test_cache_info_no_manager():
    """Test cache_info with no manager set."""
    set_cache_manager(None)

    result = await cache_info()

    assert result["success"] is False
    assert "not initialized" in result["error"]


@pytest.mark.asyncio
async def test_cache_info_with_manager():
    """Test cache_info with manager set."""
    mock_config = MagicMock()
    mock_config.enable_memory = True
    mock_config.memory_max_size = 100
    mock_config.memory_ttl = 3600
    mock_config.enable_disk = True
    mock_config.disk_max_size = 10485760  # 10MB
    mock_config.disk_ttl = 86400
    mock_config.disk_path = "/tmp/cache"

    mock_manager = MagicMock()
    mock_manager.config = mock_config

    set_cache_manager(mock_manager)

    result = await cache_info()

    assert result["success"] is True
    assert "config" in result
    assert "formatted_report" in result
    assert result["config"]["enable_memory"] is True
    assert result["config"]["memory_max_size"] == 100
    assert "Cache Configuration" in result["formatted_report"]


# ============================================================================
# HTTP Tool Tests
# ============================================================================


@pytest.mark.asyncio
async def test_http_request_missing_url():
    """Test http_request with missing URL."""
    result = await http_request(method="GET", url="")

    assert result["success"] is False
    assert "Missing required parameter" in result["error"]


@pytest.mark.asyncio
async def test_http_request_get_success():
    """Test successful GET request."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.reason_phrase = "OK"
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"data": "test"}
        mock_response.url = "https://example.com/api"

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        result = await http_request(method="GET", url="https://example.com/api")

        assert result["success"] is True
        assert result["status_code"] == 200
        assert result["body"] == {"data": "test"}
        assert "duration_ms" in result


@pytest.mark.asyncio
async def test_http_request_with_auth():
    """Test http_request with authentication."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.reason_phrase = "OK"
        mock_response.headers = {}
        mock_response.json.side_effect = Exception()
        mock_response.text = "OK"
        mock_response.url = "https://example.com/api"

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        result = await http_request(
            method="GET", url="https://example.com/api", auth="Bearer test-token"
        )

        assert result["success"] is True
        # Verify auth header was passed
        call_args = mock_client.request.call_args
        assert call_args.kwargs["headers"]["Authorization"] == "Bearer test-token"


@pytest.mark.asyncio
async def test_http_request_timeout():
    """Test http_request timeout."""
    with patch("httpx.AsyncClient") as mock_client_class:
        import httpx

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.request = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        mock_client_class.return_value = mock_client

        result = await http_request(method="GET", url="https://example.com/slow", timeout=1)

        assert result["success"] is False
        assert "timed out" in result["error"]


@pytest.mark.asyncio
async def test_http_request_post_with_json():
    """Test POST request with JSON body."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.reason_phrase = "Created"
        mock_response.headers = {}
        mock_response.json.return_value = {"id": 123}
        mock_response.url = "https://example.com/api"

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        result = await http_request(
            method="POST", url="https://example.com/api", json={"name": "test"}
        )

        assert result["success"] is True
        assert result["status_code"] == 201
        # Verify JSON was passed
        call_args = mock_client.request.call_args
        assert call_args.kwargs["json"] == {"name": "test"}


@pytest.mark.asyncio
async def test_http_test_missing_url():
    """Test http_test with missing URL."""
    result = await http_test(method="GET", url="")

    assert result["success"] is False
    assert "Missing required parameter" in result["error"]


@pytest.mark.asyncio
async def test_http_test_success():
    """Test successful API test."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com/api"

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        result = await http_test(method="GET", url="https://example.com/api", expected_status=200)

        assert result["success"] is True
        assert result["all_passed"] is True
        assert result["status_code"] == 200
        assert len(result["validations"]) == 1
        assert result["validations"][0]["passed"] is True


@pytest.mark.asyncio
async def test_http_test_validation_failed():
    """Test API test with failed validation."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.url = "https://example.com/api"

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        result = await http_test(method="GET", url="https://example.com/api", expected_status=200)

        assert result["success"] is False
        assert result["all_passed"] is False
        assert result["status_code"] == 404
        assert result["validations"][0]["passed"] is False
        assert "validations failed" in result["error"]


# ============================================================================
# Web Search Tool Tests
# ============================================================================


@pytest.mark.asyncio
async def test_web_search_missing_query():
    """Test web_search with missing query."""
    result = await web_search(query="")

    assert result["success"] is False
    assert "Missing required parameter" in result["error"]


@pytest.mark.asyncio
async def test_web_search_success():
    """Test successful web search."""
    mock_html = """
    <div class="result">
        <a class="result__a" href="https://example.com">Example Title</a>
        <a class="result__snippet">Example snippet text</a>
    </div>
    """

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = mock_html

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        result = await web_search(query="test query", max_results=5)

        assert result["success"] is True
        assert "results" in result
        assert "result_count" in result


@pytest.mark.asyncio
async def test_web_search_no_results():
    """Test web search with no results."""
    mock_html = "<html><body>No results</body></html>"

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = mock_html

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        result = await web_search(query="test query")

        assert result["success"] is True
        assert result["result_count"] == 0
        assert "No results found" in result["results"]


@pytest.mark.asyncio
async def test_web_search_timeout():
    """Test web search timeout."""
    with patch("httpx.AsyncClient") as mock_client_class:
        import httpx

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        mock_client_class.return_value = mock_client

        result = await web_search(query="test query")

        assert result["success"] is False
        assert "timed out" in result["error"]


@pytest.mark.asyncio
async def test_web_fetch_missing_url():
    """Test web_fetch with missing URL."""
    result = await web_fetch(url="")

    assert result["success"] is False
    assert "Missing required parameter" in result["error"]


@pytest.mark.asyncio
async def test_web_fetch_success():
    """Test successful web fetch."""
    mock_html = """
    <html>
        <body>
            <main>
                <p>Main content here with lots of text that is definitely over 100 characters in length to ensure it passes the minimum content check in the extract_content function.</p>
            </main>
        </body>
    </html>
    """

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = mock_html

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        result = await web_fetch(url="https://example.com")

        assert result["success"] is True
        assert "content" in result
        assert len(result["content"]) > 0


@pytest.mark.asyncio
async def test_web_fetch_no_content():
    """Test web fetch with no extractable content."""
    mock_html = "<html><body><script>alert('test')</script></body></html>"

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = mock_html

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        result = await web_fetch(url="https://example.com")

        assert result["success"] is False
        assert "No content could be extracted" in result["error"]


@pytest.mark.asyncio
async def test_web_summarize_no_provider():
    """Test web_summarize with no provider set."""
    set_web_search_provider(None, None)

    result = await web_summarize(query="test query")

    assert result["success"] is False
    assert "No LLM provider available" in result["error"]


@pytest.mark.asyncio
async def test_web_summarize_missing_query():
    """Test web_summarize with missing query."""
    mock_provider = MagicMock()
    set_web_search_provider(mock_provider, "test-model")

    result = await web_summarize(query="")

    assert result["success"] is False
    assert "Missing required parameter" in result["error"]


@pytest.mark.asyncio
async def test_web_summarize_success():
    """Test successful web summarize."""
    mock_html = """
    <div class="result">
        <a class="result__a" href="https://example.com">Test Result</a>
        <a class="result__snippet">Test snippet</a>
    </div>
    """

    mock_provider = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Summary of search results"
    mock_provider.complete = AsyncMock(return_value=mock_response)

    set_web_search_provider(mock_provider, "test-model")

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_http_response = MagicMock()
        mock_http_response.status_code = 200
        mock_http_response.text = mock_html

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(return_value=mock_http_response)
        mock_client_class.return_value = mock_client

        result = await web_summarize(query="test query", max_results=3)

        assert result["success"] is True
        assert "summary" in result
        assert "original_results" in result
        assert "Summary of search results" in result["summary"]


@pytest.mark.asyncio
async def test_web_summarize_ai_fails():
    """Test web summarize when AI summarization fails."""
    mock_html = """
    <div class="result">
        <a class="result__a" href="https://example.com">Test Result</a>
        <a class="result__snippet">Test snippet</a>
    </div>
    """

    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(side_effect=Exception("AI failed"))

    set_web_search_provider(mock_provider, "test-model")

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_http_response = MagicMock()
        mock_http_response.status_code = 200
        mock_http_response.text = mock_html

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(return_value=mock_http_response)
        mock_client_class.return_value = mock_client

        result = await web_summarize(query="test query")

        assert result["success"] is True
        assert "AI summarization failed" in result["summary"]
        assert "original_results" in result


# ============================================================================
# File Editor Tool Tests
# ============================================================================

# Import consolidated edit_files function
from victor.tools.file_editor_tool import edit_files
import os
import tempfile


@pytest.mark.asyncio
async def test_edit_files_no_operations():
    """Test edit_files with no operations."""
    result = await edit_files(operations=[])

    assert result["success"] is False
    assert "No operations provided" in result["error"]


@pytest.mark.asyncio
async def test_edit_files_invalid_operations_type():
    """Test edit_files with invalid operations type."""
    result = await edit_files(operations=[{"path": "test.txt"}])

    assert result["success"] is False
    assert "missing required field: type" in result["error"]


@pytest.mark.asyncio
async def test_edit_files_invalid_operation_type():
    """Test edit_files with invalid operation type."""
    result = await edit_files(operations=[{"type": "invalid", "path": "test.txt"}])

    assert result["success"] is False
    assert "invalid type" in result["error"]


@pytest.mark.asyncio
async def test_edit_files_missing_path():
    """Test edit_files with missing path."""
    result = await edit_files(operations=[{"type": "create"}])

    assert result["success"] is False
    assert "missing required field: path" in result["error"]


@pytest.mark.asyncio
async def test_edit_files_create_success(tmp_path):
    """Test successful file creation."""
    test_file = tmp_path / "new_file.txt"

    result = await edit_files(
        operations=[{"type": "create", "path": str(test_file), "content": "hello world"}],
        auto_commit=True,
    )

    assert result["success"] is True
    assert result["operations_queued"] == 1
    assert test_file.exists()
    assert test_file.read_text() == "hello world"


@pytest.mark.asyncio
async def test_edit_files_modify_success(tmp_path):
    """Test successful file modification."""
    test_file = tmp_path / "existing.txt"
    test_file.write_text("original content")

    result = await edit_files(
        operations=[{"type": "modify", "path": str(test_file), "content": "modified content"}]
    )

    assert result["success"] is True
    assert test_file.read_text() == "modified content"


@pytest.mark.asyncio
async def test_edit_files_delete_success(tmp_path):
    """Test successful file deletion."""
    test_file = tmp_path / "to_delete.txt"
    test_file.write_text("delete me")

    result = await edit_files(operations=[{"type": "delete", "path": str(test_file)}])

    assert result["success"] is True
    assert not test_file.exists()


@pytest.mark.asyncio
async def test_edit_files_rename_success(tmp_path):
    """Test successful file rename."""
    old_file = tmp_path / "old_name.txt"
    new_file = tmp_path / "new_name.txt"
    old_file.write_text("rename me")

    result = await edit_files(
        operations=[{"type": "rename", "path": str(old_file), "new_path": str(new_file)}]
    )

    assert result["success"] is True
    assert not old_file.exists()
    assert new_file.exists()
    assert new_file.read_text() == "rename me"


@pytest.mark.asyncio
async def test_edit_files_multiple_operations(tmp_path):
    """Test multiple operations in single call."""
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    file2.write_text("modify me")
    file3 = tmp_path / "file3.txt"
    file3.write_text("delete me")

    result = await edit_files(
        operations=[
            {"type": "create", "path": str(file1), "content": "new file"},
            {"type": "modify", "path": str(file2), "content": "modified"},
            {"type": "delete", "path": str(file3)},
        ]
    )

    assert result["success"] is True
    assert result["operations_queued"] == 3
    assert file1.exists() and file1.read_text() == "new file"
    assert file2.read_text() == "modified"
    assert not file3.exists()


@pytest.mark.asyncio
async def test_edit_files_preview_mode(tmp_path):
    """Test preview mode without applying changes."""
    test_file = tmp_path / "preview_test.txt"
    test_file.write_text("original")

    result = await edit_files(
        operations=[{"type": "modify", "path": str(test_file), "content": "modified"}],
        preview=True,
        auto_commit=False,
    )

    assert result["success"] is True
    # File should not be modified in preview mode
    assert test_file.read_text() == "original"


@pytest.mark.asyncio
async def test_edit_files_json_string_operations(tmp_path):
    """Test operations passed as JSON string."""
    test_file = tmp_path / "json_test.txt"

    result = await edit_files(
        operations='[{"type": "create", "path": "' + str(test_file) + '", "content": "json"}]'
    )

    assert result["success"] is True
    assert test_file.exists()


@pytest.mark.asyncio
async def test_edit_files_invalid_json_string():
    """Test invalid JSON string for operations."""
    result = await edit_files(operations="not valid json")

    assert result["success"] is False
    assert "Invalid JSON" in result["error"]


@pytest.mark.asyncio
async def test_edit_files_with_description(tmp_path):
    """Test operations with description."""
    test_file = tmp_path / "desc_test.txt"

    result = await edit_files(
        operations=[{"type": "create", "path": str(test_file), "content": "test"}],
        description="Creating test file",
    )

    assert result["success"] is True


@pytest.mark.asyncio
async def test_edit_files_rename_missing_new_path(tmp_path):
    """Test rename without new_path."""
    test_file = tmp_path / "rename_test.txt"
    test_file.write_text("test")

    result = await edit_files(operations=[{"type": "rename", "path": str(test_file)}])

    assert result["success"] is False
    assert "new_path" in result["error"].lower()


@pytest.mark.asyncio
async def test_edit_files_modify_nonexistent():
    """Test modifying non-existent file."""
    result = await edit_files(
        operations=[{"type": "modify", "path": "/nonexistent/path/file.txt", "content": "test"}]
    )

    assert result["success"] is False


@pytest.mark.asyncio
async def test_edit_files_operation_not_dict():
    """Test operations list with non-dict item."""
    result = await edit_files(operations=["not a dict"])

    assert result["success"] is False
    assert "must be a dictionary" in result["error"]


# ============================================================================
# Dependency Tool Tests
# ============================================================================

from victor.tools.dependency_tool import (
    dependency_list,
    dependency_outdated,
    dependency_security,
    dependency_generate,
    dependency_update,
    dependency_tree,
    dependency_check,
)


@pytest.mark.asyncio
async def test_dependency_list_success():
    """Test successful package listing."""
    mock_packages = [
        {"name": "requests", "version": "2.28.0"},
        {"name": "pytest", "version": "7.2.0"},
        {"name": "aiohttp", "version": "3.8.0"},
    ]

    mock_result = MagicMock()
    mock_result.stdout = json.dumps(mock_packages)

    with patch("subprocess.run", return_value=mock_result):
        result = await dependency_list()

        assert result["success"] is True
        assert result["count"] == 3
        assert result["packages"] == mock_packages
        assert "formatted_report" in result
        assert "Installed Packages" in result["formatted_report"]
        assert "Total: 3 packages" in result["formatted_report"]


@pytest.mark.asyncio
async def test_dependency_list_failure():
    """Test dependency list when pip fails."""
    from subprocess import CalledProcessError

    with patch("subprocess.run", side_effect=CalledProcessError(1, "pip", stderr="error")):
        result = await dependency_list()

        assert result["success"] is False
        assert "error" in result
        assert "Failed to list packages" in result["error"]


@pytest.mark.asyncio
async def test_dependency_outdated_success():
    """Test successful outdated package check."""
    mock_outdated = [
        {"name": "requests", "version": "2.28.0", "latest_version": "3.0.0"},  # Major
        {"name": "pytest", "version": "7.2.0", "latest_version": "7.3.0"},  # Minor
        {"name": "aiohttp", "version": "3.8.0", "latest_version": "3.8.1"},  # Patch
    ]

    mock_result = MagicMock()
    mock_result.stdout = json.dumps(mock_outdated)

    with patch("subprocess.run", return_value=mock_result):
        result = await dependency_outdated()

        assert result["success"] is True
        assert result["count"] == 3
        assert "by_severity" in result
        assert len(result["by_severity"]["major"]) == 1
        assert len(result["by_severity"]["minor"]) == 1
        assert len(result["by_severity"]["patch"]) == 1
        assert "formatted_report" in result
        assert "Outdated Packages" in result["formatted_report"]


@pytest.mark.asyncio
async def test_dependency_outdated_up_to_date():
    """Test outdated check when all packages are up to date."""
    mock_result = MagicMock()
    mock_result.stdout = json.dumps([])

    with patch("subprocess.run", return_value=mock_result):
        result = await dependency_outdated()

        assert result["success"] is True
        assert result["count"] == 0
        assert "All packages are up to date" in result["message"]


@pytest.mark.asyncio
async def test_dependency_security_no_vulnerabilities():
    """Test security check with no vulnerabilities."""
    mock_packages = [
        {"name": "requests", "version": "2.28.0"},
        {"name": "pytest", "version": "7.2.0"},
    ]

    mock_result = MagicMock()
    mock_result.stdout = json.dumps(mock_packages)

    with patch("subprocess.run", return_value=mock_result):
        result = await dependency_security()

        assert result["success"] is True
        assert result["count"] == 0
        assert "formatted_report" in result
        assert "No known vulnerabilities found" in result["formatted_report"]


@pytest.mark.asyncio
async def test_dependency_security_with_vulnerabilities():
    """Test security check with known vulnerabilities."""
    mock_packages = [
        {"name": "django", "version": "2.2.20"},  # Vulnerable
        {"name": "requests", "version": "2.28.0"},  # Safe
    ]

    mock_result = MagicMock()
    mock_result.stdout = json.dumps(mock_packages)

    with patch("subprocess.run", return_value=mock_result):
        result = await dependency_security()

        assert result["success"] is True
        assert result["count"] >= 1
        assert len(result["vulnerabilities"]) >= 1

        vuln = result["vulnerabilities"][0]
        assert vuln["package"] == "django"
        assert vuln["version"] == "2.2.20"
        assert "cve" in vuln
        assert "formatted_report" in result


@pytest.mark.asyncio
async def test_dependency_generate_success():
    """Test successful requirements file generation."""
    import tempfile, os

    mock_requirements = "requests==2.28.0\npytest==7.2.0\naiohttp==3.8.0"

    mock_result = MagicMock()
    mock_result.stdout = mock_requirements

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        temp_path = f.name

    try:
        with patch("subprocess.run", return_value=mock_result):
            result = await dependency_generate(output=temp_path)

            assert result["success"] is True
            assert result["file"] == temp_path
            assert result["packages_count"] == 3
            assert "Generated" in result["message"]

            # Verify file was written
            assert os.path.exists(temp_path)
            with open(temp_path) as f:
                content = f.read()
                assert "requests==2.28.0" in content
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.asyncio
async def test_dependency_generate_failure():
    """Test requirements generation when pip fails."""
    from subprocess import CalledProcessError

    with patch("subprocess.run", side_effect=CalledProcessError(1, "pip", stderr="error")):
        result = await dependency_generate()

        assert result["success"] is False
        assert "error" in result
        assert "Failed to generate requirements" in result["error"]


@pytest.mark.asyncio
async def test_dependency_update_dry_run():
    """Test dependency update in dry-run mode."""
    result = await dependency_update(packages=["requests", "pytest"], dry_run=True)

    assert result["success"] is True
    assert "would_update" in result
    assert len(result["would_update"]) == 2
    assert "Dry run" in result["message"]


@pytest.mark.asyncio
async def test_dependency_update_no_packages():
    """Test dependency update with no packages specified."""
    result = await dependency_update(packages=[], dry_run=False)

    assert result["success"] is False
    assert "No packages specified" in result["error"]


@pytest.mark.asyncio
async def test_dependency_update_actual():
    """Test actual dependency update."""
    mock_result = MagicMock()
    mock_result.stdout = "Successfully installed requests-2.28.0"

    with patch("subprocess.run", return_value=mock_result):
        result = await dependency_update(packages=["requests"], dry_run=False)

        assert result["success"] is True
        assert "updated" in result
        assert "requests" in result["updated"]
        assert "Successfully updated" in result["message"]


@pytest.mark.asyncio
async def test_dependency_tree_not_installed():
    """Test dependency tree when pipdeptree is not installed."""
    mock_result = MagicMock()
    mock_result.returncode = 1

    with patch("subprocess.run", return_value=mock_result):
        result = await dependency_tree()

        assert result["success"] is False
        assert "pipdeptree not installed" in result["error"]


@pytest.mark.asyncio
async def test_dependency_tree_success():
    """Test successful dependency tree display."""
    mock_check = MagicMock()
    mock_check.returncode = 0

    mock_tree_result = MagicMock()
    mock_tree_result.stdout = "requests==2.28.0\n  - urllib3 [required: >=1.21.1]"

    with patch("subprocess.run", side_effect=[mock_check, mock_tree_result]):
        result = await dependency_tree(package="requests")

        assert result["success"] is True
        assert "tree" in result
        assert "requests" in result["tree"]
        assert result["package"] == "requests"


@pytest.mark.asyncio
async def test_dependency_check_file_not_found():
    """Test dependency check when requirements file doesn't exist."""
    result = await dependency_check(requirements_file="nonexistent.txt")

    assert result["success"] is False
    assert "not found" in result["error"]


@pytest.mark.asyncio
async def test_dependency_check_success():
    """Test successful dependency check."""
    import tempfile, os

    # Create temp requirements file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("requests==2.28.0\npytest==7.2.0\n")
        req_path = f.name

    mock_packages = [
        {"name": "requests", "version": "2.28.0"},
        {"name": "pytest", "version": "7.2.0"},
    ]

    mock_result = MagicMock()
    mock_result.stdout = json.dumps(mock_packages)

    try:
        with patch("subprocess.run", return_value=mock_result):
            result = await dependency_check(requirements_file=req_path)

            assert result["success"] is True
            assert len(result["missing"]) == 0
            assert len(result["mismatched"]) == 0
            assert "formatted_report" in result
            assert "All requirements satisfied" in result["formatted_report"]
    finally:
        if os.path.exists(req_path):
            os.unlink(req_path)


@pytest.mark.asyncio
async def test_dependency_check_missing_packages():
    """Test dependency check with missing packages."""
    import tempfile, os

    # Create temp requirements file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("requests==2.28.0\nmissing-package==1.0.0\n")
        req_path = f.name

    mock_packages = [
        {"name": "requests", "version": "2.28.0"},
    ]

    mock_result = MagicMock()
    mock_result.stdout = json.dumps(mock_packages)

    try:
        with patch("subprocess.run", return_value=mock_result):
            result = await dependency_check(requirements_file=req_path)

            assert result["success"] is False
            assert len(result["missing"]) == 1
            assert "missing-package==1.0.0" in result["missing"]
            assert "formatted_report" in result
    finally:
        if os.path.exists(req_path):
            os.unlink(req_path)


@pytest.mark.asyncio
async def test_dependency_check_version_mismatch():
    """Test dependency check with version mismatches."""
    import tempfile, os

    # Create temp requirements file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("requests==2.28.0\n")
        req_path = f.name

    mock_packages = [
        {"name": "requests", "version": "2.27.0"},  # Different version
    ]

    mock_result = MagicMock()
    mock_result.stdout = json.dumps(mock_packages)

    try:
        with patch("subprocess.run", return_value=mock_result):
            result = await dependency_check(requirements_file=req_path)

            assert result["success"] is False
            assert len(result["mismatched"]) == 1

            mismatch = result["mismatched"][0]
            assert mismatch["package"] == "requests"
            assert mismatch["required"] == "2.28.0"
            assert mismatch["installed"] == "2.27.0"
    finally:
        if os.path.exists(req_path):
            os.unlink(req_path)


# ============================================================================
# Database Tool Tests
# ============================================================================

from victor.tools.database_tool import (
    database_connect,
    database_query,
    database_tables,
    database_describe,
    database_schema,
    database_disconnect,
    set_database_config,
)


@pytest.mark.asyncio
async def test_database_connect_sqlite_success():
    """Test successful SQLite database connection."""
    import tempfile
    import os

    # Create temporary database file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name

    try:
        result = await database_connect(database=db_path, db_type="sqlite")

        assert result["success"] is True
        assert "connection_id" in result
        assert "sqlite_" in result["connection_id"]
        assert "Connected to SQLite" in result["message"]

        # Cleanup connection
        await database_disconnect(result["connection_id"])
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.mark.asyncio
async def test_database_connect_invalid_type():
    """Test database connection with invalid database type."""
    result = await database_connect(database="test.db", db_type="invalid_type")

    assert result["success"] is False
    assert "Unsupported database type" in result["error"]


@pytest.mark.asyncio
async def test_database_query_missing_connection():
    """Test database query with missing connection_id."""
    result = await database_query(connection_id="invalid_id", sql="SELECT 1")

    assert result["success"] is False
    assert "Invalid or missing connection_id" in result["error"]


@pytest.mark.asyncio
async def test_database_query_missing_sql():
    """Test database query with missing SQL."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name

    try:
        connect_result = await database_connect(database=db_path, db_type="sqlite")
        connection_id = connect_result["connection_id"]

        result = await database_query(connection_id=connection_id, sql="")

        assert result["success"] is False
        assert "Missing required parameter: sql" in result["error"]

        # Cleanup
        await database_disconnect(connection_id)
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.mark.asyncio
async def test_database_query_select_success():
    """Test successful SELECT query."""
    import tempfile
    import os
    import sqlite3

    # Create temporary database with test table
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name

    try:
        # Setup test table
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE test_table (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO test_table VALUES (1, 'Alice'), (2, 'Bob')")
        conn.commit()
        conn.close()

        # Connect and query
        connect_result = await database_connect(database=db_path, db_type="sqlite")
        connection_id = connect_result["connection_id"]

        result = await database_query(connection_id=connection_id, sql="SELECT * FROM test_table")

        assert result["success"] is True
        assert "columns" in result
        assert "rows" in result
        assert result["count"] == 2
        assert result["columns"] == ["id", "name"]
        assert len(result["rows"]) == 2
        assert result["rows"][0]["id"] == 1
        assert result["rows"][0]["name"] == "Alice"

        # Cleanup
        await database_disconnect(connection_id)
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.mark.asyncio
async def test_database_query_dangerous_operation_blocked():
    """Test that dangerous operations are blocked by default."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name

    try:
        connect_result = await database_connect(database=db_path, db_type="sqlite")
        connection_id = connect_result["connection_id"]

        # Try dangerous operations
        dangerous_queries = [
            "DROP TABLE test",
            "DELETE FROM test",
            "UPDATE test SET x=1",
            "INSERT INTO test VALUES (1)",
        ]

        for query in dangerous_queries:
            result = await database_query(connection_id=connection_id, sql=query)
            assert result["success"] is False
            assert "not allowed" in result["error"]

        # Cleanup
        await database_disconnect(connection_id)
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.mark.asyncio
async def test_database_query_modifications_allowed():
    """Test that modifications work when explicitly allowed."""
    import tempfile
    import os
    import sqlite3

    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name

    try:
        # Setup
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE test_table (id INTEGER)")
        conn.commit()
        conn.close()

        # Enable modifications
        set_database_config(allow_modifications=True, max_rows=100)

        connect_result = await database_connect(database=db_path, db_type="sqlite")
        connection_id = connect_result["connection_id"]

        # INSERT should work now
        result = await database_query(
            connection_id=connection_id, sql="INSERT INTO test_table VALUES (1)"
        )

        assert result["success"] is True
        assert "rows_affected" in result

        # Reset config
        set_database_config(allow_modifications=False, max_rows=100)

        # Cleanup
        await database_disconnect(connection_id)
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.mark.asyncio
async def test_database_tables_success():
    """Test listing database tables."""
    import tempfile
    import os
    import sqlite3

    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name

    try:
        # Setup test tables
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE users (id INTEGER)")
        conn.execute("CREATE TABLE posts (id INTEGER)")
        conn.commit()
        conn.close()

        connect_result = await database_connect(database=db_path, db_type="sqlite")
        connection_id = connect_result["connection_id"]

        result = await database_tables(connection_id=connection_id)

        assert result["success"] is True
        assert "tables" in result
        assert result["count"] == 2
        assert "users" in result["tables"]
        assert "posts" in result["tables"]

        # Cleanup
        await database_disconnect(connection_id)
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.mark.asyncio
async def test_database_tables_invalid_connection():
    """Test listing tables with invalid connection."""
    result = await database_tables(connection_id="invalid_id")

    assert result["success"] is False
    assert "Invalid or missing connection_id" in result["error"]


@pytest.mark.asyncio
async def test_database_describe_success():
    """Test describing table structure."""
    import tempfile
    import os
    import sqlite3

    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name

    try:
        # Setup test table
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT
            )
        """
        )
        conn.commit()
        conn.close()

        connect_result = await database_connect(database=db_path, db_type="sqlite")
        connection_id = connect_result["connection_id"]

        result = await database_describe(connection_id=connection_id, table="users")

        assert result["success"] is True
        assert result["table"] == "users"
        assert "columns" in result
        assert result["count"] == 3

        # Check column details
        columns = {col["name"]: col for col in result["columns"]}
        assert "id" in columns
        assert columns["id"]["primary_key"] is True
        assert "name" in columns
        assert columns["name"]["nullable"] is False
        assert "email" in columns
        assert columns["email"]["nullable"] is True

        # Cleanup
        await database_disconnect(connection_id)
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.mark.asyncio
async def test_database_describe_missing_table():
    """Test describing table with missing table name."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name

    try:
        connect_result = await database_connect(database=db_path, db_type="sqlite")
        connection_id = connect_result["connection_id"]

        result = await database_describe(connection_id=connection_id, table="")

        assert result["success"] is False
        assert "Missing required parameter: table" in result["error"]

        # Cleanup
        await database_disconnect(connection_id)
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.mark.asyncio
async def test_database_schema_success():
    """Test getting complete database schema."""
    import tempfile
    import os
    import sqlite3

    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name

    try:
        # Setup test tables
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("CREATE TABLE posts (id INTEGER PRIMARY KEY, title TEXT)")
        conn.commit()
        conn.close()

        connect_result = await database_connect(database=db_path, db_type="sqlite")
        connection_id = connect_result["connection_id"]

        result = await database_schema(connection_id=connection_id)

        assert result["success"] is True
        assert "tables" in result
        assert len(result["tables"]) == 2

        # Check table details
        table_names = [t["name"] for t in result["tables"]]
        assert "users" in table_names
        assert "posts" in table_names

        # Check columns are included
        users_table = [t for t in result["tables"] if t["name"] == "users"][0]
        assert "columns" in users_table
        assert len(users_table["columns"]) == 2

        # Cleanup
        await database_disconnect(connection_id)
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.mark.asyncio
async def test_database_schema_invalid_connection():
    """Test getting schema with invalid connection."""
    result = await database_schema(connection_id="invalid_id")

    assert result["success"] is False
    assert "Invalid or missing connection_id" in result["error"]


@pytest.mark.asyncio
async def test_database_disconnect_success():
    """Test successful database disconnection."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name

    try:
        connect_result = await database_connect(database=db_path, db_type="sqlite")
        connection_id = connect_result["connection_id"]

        result = await database_disconnect(connection_id=connection_id)

        assert result["success"] is True
        assert "Disconnected from database" in result["message"]
        assert connection_id in result["message"]
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.mark.asyncio
async def test_database_disconnect_invalid_connection():
    """Test disconnection with invalid connection."""
    result = await database_disconnect(connection_id="invalid_id")

    assert result["success"] is False
    assert "Invalid or missing connection_id" in result["error"]


@pytest.mark.asyncio
async def test_database_query_with_limit():
    """Test SELECT query with custom limit."""
    import tempfile
    import os
    import sqlite3

    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name

    try:
        # Setup test table with multiple rows
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE test_table (id INTEGER)")
        for i in range(10):
            conn.execute(f"INSERT INTO test_table VALUES ({i})")
        conn.commit()
        conn.close()

        connect_result = await database_connect(database=db_path, db_type="sqlite")
        connection_id = connect_result["connection_id"]

        # Query with limit
        result = await database_query(
            connection_id=connection_id, sql="SELECT * FROM test_table", limit=5
        )

        assert result["success"] is True
        assert result["count"] == 5
        assert len(result["rows"]) == 5

        # Cleanup
        await database_disconnect(connection_id)
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


# ============================================================================
# Filesystem Tool Tests
# ============================================================================

from victor.tools.filesystem import read_file, write_file, list_directory


@pytest.mark.asyncio
async def test_read_file_success():
    """Test successful file reading."""
    import tempfile
    import os

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("Hello, World!")
        temp_path = f.name

    try:
        result = await read_file(path=temp_path)

        assert result == "Hello, World!"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.asyncio
async def test_read_file_not_found():
    """Test reading non-existent file."""
    result = await read_file(path="/nonexistent/path/file.txt")

    assert "Error: File not found" in result


@pytest.mark.asyncio
async def test_read_file_not_a_file():
    """Test reading a directory path."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        result = await read_file(path=tmpdir)

        assert "Error: Path" in result
        assert "is not a file" in result


@pytest.mark.asyncio
async def test_write_file_success():
    """Test successful file writing."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test_file.txt")
        content = "Test content\nMultiple lines"

        result = await write_file(path=file_path, content=content)

        assert "Successfully wrote" in result
        assert f"{len(content)} characters" in result

        # Verify file was actually written
        assert os.path.exists(file_path)
        with open(file_path, "r") as f:
            written_content = f.read()
            assert written_content == content


@pytest.mark.asyncio
async def test_write_file_creates_directories():
    """Test that write_file creates parent directories."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested path that doesn't exist
        file_path = os.path.join(tmpdir, "subdir1", "subdir2", "test.txt")
        content = "Test content"

        result = await write_file(path=file_path, content=content)

        assert "Successfully wrote" in result

        # Verify directories and file were created
        assert os.path.exists(file_path)
        assert os.path.isfile(file_path)


@pytest.mark.asyncio
async def test_write_file_overwrites_existing():
    """Test that write_file overwrites existing files."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("Original content")
        temp_path = f.name

    try:
        new_content = "New content"
        result = await write_file(path=temp_path, content=new_content)

        assert "Successfully wrote" in result

        # Verify content was overwritten
        with open(temp_path, "r") as f:
            assert f.read() == new_content
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.asyncio
async def test_list_directory_success():
    """Test listing directory contents."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some files and directories
        file1 = os.path.join(tmpdir, "file1.txt")
        file2 = os.path.join(tmpdir, "file2.txt")
        subdir = os.path.join(tmpdir, "subdir")

        with open(file1, "w") as f:
            f.write("test")
        with open(file2, "w") as f:
            f.write("test")
        os.mkdir(subdir)

        result = await list_directory(path=tmpdir)

        assert len(result) == 3
        names = [item["name"] for item in result]
        assert "file1.txt" in names
        assert "file2.txt" in names
        assert "subdir" in names

        # Check types
        types = {item["name"]: item["type"] for item in result}
        assert types["file1.txt"] == "file"
        assert types["file2.txt"] == "file"
        assert types["subdir"] == "directory"


@pytest.mark.asyncio
async def test_list_directory_recursive():
    """Test listing directory contents recursively."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested structure
        subdir = os.path.join(tmpdir, "subdir")
        os.mkdir(subdir)
        file1 = os.path.join(tmpdir, "file1.txt")
        file2 = os.path.join(subdir, "file2.txt")

        with open(file1, "w") as f:
            f.write("test")
        with open(file2, "w") as f:
            f.write("test")

        result = await list_directory(path=tmpdir, recursive=True)

        assert len(result) == 3  # subdir, file1.txt, subdir/file2.txt
        paths = [item["path"] for item in result]
        assert "subdir" in paths
        assert "file1.txt" in paths
        assert os.path.join("subdir", "file2.txt") in paths or "subdir/file2.txt" in paths


@pytest.mark.asyncio
async def test_list_directory_not_found():
    """Test listing non-existent directory."""
    with pytest.raises(FileNotFoundError):
        await list_directory(path="/nonexistent/directory")


@pytest.mark.asyncio
async def test_list_directory_not_a_directory():
    """Test listing a file path."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        temp_path = f.name

    try:
        with pytest.raises(NotADirectoryError):
            await list_directory(path=temp_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.asyncio
async def test_list_directory_empty():
    """Test listing empty directory."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        result = await list_directory(path=tmpdir)

        assert result == []


# ============================================================================
# Git Tool Tests
# ============================================================================

from victor.tools.git_tool import (
    git,
    git_suggest_commit,
    git_create_pr,
    git_analyze_conflicts,
    set_git_provider,
    _run_git,
)


@pytest.mark.asyncio
async def test_git_missing_operation():
    """Test git with missing operation."""
    result = await git(operation="")

    assert result["success"] is False
    assert "Missing required parameter" in result["error"]


@pytest.mark.asyncio
async def test_git_invalid_operation():
    """Test git with invalid operation."""
    result = await git(operation="invalid_op")

    assert result["success"] is False
    assert "Invalid operation" in result["error"]


@pytest.mark.asyncio
async def test_git_status_success():
    """Test successful git status."""
    with patch("victor.tools.git_tool._run_git") as mock_run_git:
        mock_run_git.side_effect = [
            (True, "## main\n M file1.txt", ""),
            (True, "On branch main\nChanges not staged for commit:\n  modified: file1.txt", ""),
        ]

        result = await git(operation="status")

        assert result["success"] is True
        assert "Short status:" in result["output"]
        assert "Full status:" in result["output"]


@pytest.mark.asyncio
async def test_git_status_failure():
    """Test git status failure."""
    with patch("victor.tools.git_tool._run_git") as mock_run_git:
        mock_run_git.return_value = (False, "", "fatal: not a git repository")

        result = await git(operation="status")

        assert result["success"] is False
        assert "not a git repository" in result["error"]


@pytest.mark.asyncio
async def test_git_diff_unstaged():
    """Test git diff for unstaged changes."""
    with patch("victor.tools.git_tool._run_git") as mock_run_git:
        mock_run_git.return_value = (True, "diff --git a/file.txt\n+new line", "")

        result = await git(operation="diff", staged=False)

        assert result["success"] is True
        assert "diff --git" in result["output"]


@pytest.mark.asyncio
async def test_git_diff_staged():
    """Test git diff for staged changes."""
    with patch("victor.tools.git_tool._run_git") as mock_run_git:
        mock_run_git.return_value = (True, "diff --git a/staged.txt\n+staged line", "")

        result = await git(operation="diff", staged=True)

        assert result["success"] is True
        assert "staged line" in result["output"]


@pytest.mark.asyncio
async def test_git_diff_no_changes():
    """Test git diff with no changes."""
    with patch("victor.tools.git_tool._run_git") as mock_run_git:
        mock_run_git.return_value = (True, "", "")

        result = await git(operation="diff", staged=False)

        assert result["success"] is True
        assert "No changes to show" in result["output"]


@pytest.mark.asyncio
async def test_git_stage_all():
    """Test staging all changes."""
    with patch("victor.tools.git_tool._run_git") as mock_run_git:
        mock_run_git.side_effect = [(True, "", ""), (True, "M  file1.txt\nM  file2.txt", "")]

        result = await git(operation="stage")

        assert result["success"] is True
        assert "Files staged successfully" in result["output"]


@pytest.mark.asyncio
async def test_git_stage_specific_files():
    """Test staging specific files."""
    with patch("victor.tools.git_tool._run_git") as mock_run_git:
        mock_run_git.side_effect = [(True, "", ""), (True, "M  file1.txt", "")]

        result = await git(operation="stage", files=["file1.txt"])

        assert result["success"] is True
        assert "Files staged successfully" in result["output"]


@pytest.mark.asyncio
async def test_git_stage_failure():
    """Test git stage failure."""
    with patch("victor.tools.git_tool._run_git") as mock_run_git:
        mock_run_git.return_value = (
            False,
            "",
            "fatal: pathspec 'nonexistent' did not match any files",
        )

        result = await git(operation="stage", files=["nonexistent"])

        assert result["success"] is False
        assert "did not match" in result["error"]


@pytest.mark.asyncio
async def test_git_commit_with_message():
    """Test git commit with provided message."""
    with patch("victor.tools.git_tool._run_git") as mock_run_git:
        mock_run_git.return_value = (True, "[main abc123] test commit\n 1 file changed", "")

        result = await git(operation="commit", message="test commit")

        assert result["success"] is True
        assert "Committed successfully" in result["output"]


@pytest.mark.asyncio
async def test_git_commit_no_message():
    """Test git commit without message."""
    result = await git(operation="commit", message=None)

    assert result["success"] is False
    assert "No commit message" in result["error"] or "message" in result["error"].lower()


@pytest.mark.asyncio
async def test_git_log_success():
    """Test git log."""
    with patch("victor.tools.git_tool._run_git") as mock_run_git:
        mock_run_git.return_value = (
            True,
            "* abc123 - Initial commit (Author, 2 days ago)\n* def456 - Second commit",
            "",
        )

        result = await git(operation="log", limit=5)

        assert result["success"] is True
        assert "Initial commit" in result["output"]
        assert "abc123" in result["output"]


@pytest.mark.asyncio
async def test_git_log_failure():
    """Test git log failure."""
    with patch("victor.tools.git_tool._run_git") as mock_run_git:
        mock_run_git.return_value = (
            False,
            "",
            "fatal: your current branch does not have any commits yet",
        )

        result = await git(operation="log")

        assert result["success"] is False
        assert "does not have any commits" in result["error"]


@pytest.mark.asyncio
async def test_git_branch_list():
    """Test listing branches."""
    with patch("victor.tools.git_tool._run_git") as mock_run_git:
        mock_run_git.return_value = (True, "* main\n  feature-branch\n  develop", "")

        result = await git(operation="branch")

        assert result["success"] is True
        assert "main" in result["output"]
        assert "feature-branch" in result["output"]


@pytest.mark.asyncio
async def test_git_branch_switch():
    """Test switching branches."""
    with patch("victor.tools.git_tool._run_git") as mock_run_git:
        mock_run_git.return_value = (True, "Switched to branch 'develop'", "")

        result = await git(operation="branch", branch="develop")

        assert result["success"] is True
        assert "Switched" in result["output"]


@pytest.mark.asyncio
async def test_git_suggest_commit_no_provider():
    """Test commit suggestion without AI provider."""
    result = await git_suggest_commit()

    assert result["success"] is False
    assert "No LLM provider available" in result["error"]


@pytest.mark.asyncio
async def test_git_suggest_commit_no_changes():
    """Test commit suggestion with no staged changes."""
    mock_provider = MagicMock()
    set_git_provider(mock_provider, "test-model")

    with patch("victor.tools.git_tool._run_git") as mock_run_git:
        mock_run_git.return_value = (True, "", "")

        result = await git_suggest_commit()

        assert result["success"] is False
        assert "No staged changes" in result["error"]

    # Cleanup
    set_git_provider(None, None)


@pytest.mark.asyncio
async def test_git_suggest_commit_success():
    """Test successful AI commit message generation."""
    mock_provider = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "feat(auth): add user authentication"
    mock_provider.complete = AsyncMock(return_value=mock_response)

    set_git_provider(mock_provider, "test-model")

    with patch("victor.tools.git_tool._run_git") as mock_run_git:
        mock_run_git.side_effect = [
            (True, "diff --git a/auth.py\n+def login():", ""),
            (True, "auth.py", ""),
        ]

        result = await git_suggest_commit()

        assert result["success"] is True
        assert "feat" in result["output"] or "auth" in result["output"]

    # Cleanup
    set_git_provider(None, None)


@pytest.mark.asyncio
async def test_git_create_pr_no_gh_cli():
    """Test PR creation without GitHub CLI."""
    with patch("victor.tools.git_tool._run_git") as mock_run_git:
        mock_run_git.side_effect = [(True, "feature-branch", ""), (True, "", "")]

        with patch("subprocess.run", side_effect=FileNotFoundError()):
            result = await git_create_pr()

            assert result["success"] is False
            assert "gh" in result["error"]


@pytest.mark.asyncio
async def test_git_create_pr_push_failure():
    """Test PR creation with push failure."""
    with patch("victor.tools.git_tool._run_git") as mock_run_git:
        mock_run_git.side_effect = [
            (True, "feature-branch", ""),
            (False, "", "fatal: could not push"),
        ]

        result = await git_create_pr()

        assert result["success"] is False
        assert "Failed to push" in result["error"]


@pytest.mark.asyncio
async def test_git_analyze_conflicts_no_conflicts():
    """Test conflict analysis with no conflicts."""
    with patch("victor.tools.git_tool._run_git") as mock_run_git:
        mock_run_git.return_value = (True, "M  file1.txt\nM  file2.txt", "")

        result = await git_analyze_conflicts()

        assert result["success"] is True
        assert "No merge conflicts" in result["output"]


@pytest.mark.asyncio
async def test_run_git_success():
    """Test _run_git helper function."""
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "success output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        success, stdout, stderr = _run_git("status")

        assert success is True
        assert stdout == "success output"
        assert stderr == ""


@pytest.mark.asyncio
async def test_run_git_timeout():
    """Test _run_git with timeout."""
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 30)):
        success, stdout, stderr = _run_git("log")

        assert success is False
        assert "timed out" in stderr


# ============================================================================
# Batch Processor Tool Tests
# ============================================================================

from victor.tools.batch_processor_tool import (
    batch_search,
    batch_replace,
    batch_analyze,
    batch_list_files,
)


@pytest.mark.asyncio
async def test_batch_search_success():
    """Test batch search finds pattern in files."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("hello world\ntest pattern\n")

        result = await batch_search(path=tmpdir, pattern="pattern", file_pattern="*.txt")

        assert result["success"] is True
        assert result["total_files"] >= 1
        assert result["total_matches"] >= 1


@pytest.mark.asyncio
async def test_batch_search_missing_path():
    """Test batch search with missing path."""
    result = await batch_search(path="", pattern="test")

    assert result["success"] is False
    assert "Missing required parameter" in result["error"]


@pytest.mark.asyncio
async def test_batch_replace_dry_run():
    """Test batch replace in dry run mode."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.py")
        with open(test_file, "w") as f:
            f.write("old_name = 'value'")

        result = await batch_replace(
            path=tmpdir, find="old_name", replace="new_name", file_pattern="*.py", dry_run=True
        )

        assert result["success"] is True
        assert result["dry_run"] is True


@pytest.mark.asyncio
async def test_batch_analyze():
    """Test batch analyze."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.py")
        with open(test_file, "w") as f:
            f.write("def foo():\n    pass\n")

        result = await batch_analyze(path=tmpdir, file_pattern="*.py")

        assert result["success"] is True
        assert result["total_files"] >= 1
        assert result["total_lines"] >= 1


@pytest.mark.asyncio
async def test_batch_list_files():
    """Test batch list files."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("content")

        result = await batch_list_files(path=tmpdir, file_pattern="*.txt")

        assert result["success"] is True
        assert result["total_files"] >= 1


# ============================================================================
# CI/CD Tool Tests
# ============================================================================

from victor.tools.cicd_tool import (
    cicd_generate,
    cicd_validate,
    cicd_list_templates,
    cicd_create_workflow,
)


@pytest.mark.asyncio
async def test_cicd_generate_success():
    """Test CI/CD config generation."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "test-workflow.yml")
        result = await cicd_generate(platform="github", workflow="python-test", output=output_file)

        assert result["success"] is True
        assert os.path.exists(output_file)
        assert "config" in result


@pytest.mark.asyncio
async def test_cicd_generate_invalid_platform():
    """Test CI/CD generate with invalid platform."""
    result = await cicd_generate(platform="invalid", workflow="test")

    assert result["success"] is False
    assert "not yet supported" in result["error"]


@pytest.mark.asyncio
async def test_cicd_validate_valid_file():
    """Test CI/CD validation with valid file."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(
            "name: Test\non:\n  push:\n    branches: [main]\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n"
        )
        temp_file = f.name

    try:
        result = await cicd_validate(file=temp_file)
        # Should succeed (has required fields)
        assert result["success"] is True or "jobs_count" in result
    finally:
        os.unlink(temp_file)


@pytest.mark.asyncio
async def test_cicd_list_templates():
    """Test CI/CD list templates."""
    result = await cicd_list_templates()

    assert result["success"] is True
    assert "templates" in result
    assert len(result["templates"]) > 0


@pytest.mark.asyncio
async def test_cicd_create_workflow():
    """Test CI/CD create workflow."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "workflow.yml")
        result = await cicd_create_workflow(type="test", output=output_file)

        assert result["success"] is True
        assert os.path.exists(output_file)


# ============================================================================
# Scaffold Tool Tests
# ============================================================================

from victor.tools.scaffold_tool import (
    scaffold_create,
    scaffold_list_templates,
    scaffold_add_file,
)


@pytest.mark.asyncio
async def test_scaffold_create_success():
    """Test scaffold project creation."""
    import tempfile
    import os
    import shutil

    tmpdir = tempfile.mkdtemp()
    project_name = os.path.join(tmpdir, "test-project")

    try:
        result = await scaffold_create(template="python-cli", name=project_name, force=False)

        assert result["success"] is True
        assert os.path.exists(project_name)
        assert len(result["files_created"]) > 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_scaffold_create_missing_template():
    """Test scaffold with missing template parameter."""
    result = await scaffold_create(template="", name="test")

    assert result["success"] is False
    assert "Missing required parameter" in result["error"]


@pytest.mark.asyncio
async def test_scaffold_create_invalid_template():
    """Test scaffold with invalid template."""
    result = await scaffold_create(template="invalid", name="test")

    assert result["success"] is False
    assert "Unknown template" in result["error"]


@pytest.mark.asyncio
async def test_scaffold_list_templates():
    """Test scaffold list templates."""
    result = await scaffold_list_templates()

    assert result["success"] is True
    assert "templates" in result
    assert result["count"] > 0


@pytest.mark.asyncio
async def test_scaffold_add_file():
    """Test scaffold add file."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.txt")
        result = await scaffold_add_file(path=file_path, content="test content")

        assert result["success"] is True
        assert os.path.exists(file_path)


# ============================================================================
# Docker Tool Tests
# ============================================================================

from victor.tools.docker_tool import (
    docker_ps,
    docker_images,
    docker_pull,
    docker_stop,
    docker_logs,
)


@pytest.mark.asyncio
async def test_docker_ps_no_docker():
    """Test docker ps without Docker installed."""
    with patch("victor.tools.docker_tool._check_docker", return_value=False):
        result = await docker_ps()

        assert result["success"] is False
        assert "Docker CLI not found" in result["error"]


@pytest.mark.asyncio
async def test_docker_ps_success():
    """Test docker ps with Docker installed."""
    with patch("victor.tools.docker_tool._check_docker", return_value=True):
        with patch("victor.tools.docker_tool._run_docker_command") as mock_run:
            mock_run.return_value = (True, '{"ID":"abc123","Names":"test"}', "")

            result = await docker_ps()

            assert result["success"] is True
            assert "containers" in result


@pytest.mark.asyncio
async def test_docker_images_success():
    """Test docker images."""
    with patch("victor.tools.docker_tool._check_docker", return_value=True):
        with patch("victor.tools.docker_tool._run_docker_command") as mock_run:
            mock_run.return_value = (True, '{"Repository":"nginx","Tag":"latest"}', "")

            result = await docker_images()

            assert result["success"] is True
            assert "images" in result


@pytest.mark.asyncio
async def test_docker_pull_missing_image():
    """Test docker pull with missing image parameter."""
    result = await docker_pull(image="")

    assert result["success"] is False
    assert "Missing required parameter" in result["error"]


@pytest.mark.asyncio
async def test_docker_stop_success():
    """Test docker stop."""
    with patch("victor.tools.docker_tool._check_docker", return_value=True):
        with patch("victor.tools.docker_tool._run_docker_command") as mock_run:
            mock_run.return_value = (True, "container_id", "")

            result = await docker_stop(container="test_container")

            assert result["success"] is True
            assert "stopped" in result["message"]


@pytest.mark.asyncio
async def test_docker_logs_success():
    """Test docker logs."""
    with patch("victor.tools.docker_tool._check_docker", return_value=True):
        with patch("victor.tools.docker_tool._run_docker_command") as mock_run:
            mock_run.return_value = (True, "log output", "")

            result = await docker_logs(container="test_container", tail=50)

            assert result["success"] is True
            assert "logs" in result


# ============================================================================
# Metrics Tool Tests
# ============================================================================

from victor.tools.metrics_tool import (
    metrics_complexity,
    metrics_maintainability,
    metrics_debt,
    metrics_profile,
    metrics_analyze,
)


@pytest.mark.asyncio
async def test_metrics_complexity_success():
    """Test metrics complexity calculation."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("def foo():\n    if True:\n        pass\n")
        temp_file = f.name

    try:
        result = await metrics_complexity(file=temp_file, threshold=10)

        assert result["success"] is True
        assert "complexity" in result
        assert result["complexity"] >= 1
    finally:
        os.unlink(temp_file)


@pytest.mark.asyncio
async def test_metrics_complexity_missing_file():
    """Test metrics complexity with missing file."""
    result = await metrics_complexity(file="")

    assert result["success"] is False
    assert "Missing required parameter" in result["error"]


@pytest.mark.asyncio
async def test_metrics_maintainability():
    """Test metrics maintainability."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("def foo():\n    return 42\n")
        temp_file = f.name

    try:
        result = await metrics_maintainability(file=temp_file)

        assert result["success"] is True
        assert "maintainability_index" in result
        assert 0 <= result["maintainability_index"] <= 100
    finally:
        os.unlink(temp_file)


@pytest.mark.asyncio
async def test_metrics_debt():
    """Test metrics technical debt estimation."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("def foo():\n    pass\n")
        temp_file = f.name

    try:
        result = await metrics_debt(file=temp_file)

        assert result["success"] is True
        assert "debt_hours" in result
        assert "debt_level" in result
    finally:
        os.unlink(temp_file)


@pytest.mark.asyncio
async def test_metrics_profile():
    """Test metrics code profiling."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("class Foo:\n    def bar(self):\n        pass\n")
        temp_file = f.name

    try:
        result = await metrics_profile(file=temp_file)

        assert result["success"] is True
        assert "lines" in result
        assert "functions" in result
        assert "classes" in result
    finally:
        os.unlink(temp_file)


@pytest.mark.asyncio
async def test_metrics_analyze():
    """Test comprehensive metrics analysis."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("def foo():\n    return 42\n")
        temp_file = f.name

    try:
        result = await metrics_analyze(file=temp_file)

        assert result["success"] is True
        assert "complexity" in result
        assert "maintainability" in result
        assert "debt" in result
        assert "profile" in result
    finally:
        os.unlink(temp_file)


# ============================================================================
# Additional Docker Tool Tests (Coverage Improvement)
# ============================================================================

from victor.tools.docker_tool import (
    docker_run,
    docker_start,
    docker_restart,
    docker_rm,
    docker_rmi,
    docker_stats,
    docker_inspect,
    docker_networks,
    docker_volumes,
    docker_exec,
)


@pytest.mark.asyncio
async def test_docker_run_success():
    """Test docker run with full parameters."""
    with patch("victor.tools.docker_tool._check_docker", return_value=True):
        with patch("victor.tools.docker_tool._run_docker_command") as mock_run:
            mock_run.return_value = (True, "container_abc123", "")

            result = await docker_run(
                image="nginx:latest",
                name="test-nginx",
                ports=["80:80"],
                env=["ENV=prod"],
                volumes=["/data:/data"],
                detach=True,
            )

            assert result["success"] is True
            assert "container_id" in result
            assert "container_abc123" in result["container_id"]


@pytest.mark.asyncio
async def test_docker_run_missing_image():
    """Test docker run without image."""
    result = await docker_run(image="")

    assert result["success"] is False
    assert "Missing required parameter" in result["error"]


@pytest.mark.asyncio
async def test_docker_start_success():
    """Test docker start."""
    with patch("victor.tools.docker_tool._check_docker", return_value=True):
        with patch("victor.tools.docker_tool._run_docker_command") as mock_run:
            mock_run.return_value = (True, "test_container", "")

            result = await docker_start(container="test_container")

            assert result["success"] is True
            assert "started" in result["message"]


@pytest.mark.asyncio
async def test_docker_start_missing_container():
    """Test docker start without container."""
    result = await docker_start(container="")

    assert result["success"] is False
    assert "Missing required parameter" in result["error"]


@pytest.mark.asyncio
async def test_docker_restart_success():
    """Test docker restart."""
    with patch("victor.tools.docker_tool._check_docker", return_value=True):
        with patch("victor.tools.docker_tool._run_docker_command") as mock_run:
            mock_run.return_value = (True, "test_container", "")

            result = await docker_restart(container="test_container")

            assert result["success"] is True
            assert "restarted" in result["message"]


@pytest.mark.asyncio
async def test_docker_rm_success():
    """Test docker rm (remove container)."""
    with patch("victor.tools.docker_tool._check_docker", return_value=True):
        with patch("victor.tools.docker_tool._run_docker_command") as mock_run:
            mock_run.return_value = (True, "test_container", "")

            result = await docker_rm(container="test_container")

            assert result["success"] is True
            assert "removed" in result["message"]


@pytest.mark.asyncio
async def test_docker_rm_missing_container():
    """Test docker rm without container."""
    result = await docker_rm(container="")

    assert result["success"] is False
    assert "Missing required parameter" in result["error"]


@pytest.mark.asyncio
async def test_docker_rmi_success():
    """Test docker rmi (remove image)."""
    with patch("victor.tools.docker_tool._check_docker", return_value=True):
        with patch("victor.tools.docker_tool._run_docker_command") as mock_run:
            mock_run.return_value = (True, "nginx:latest", "")

            result = await docker_rmi(image="nginx:latest")

            assert result["success"] is True
            assert "removed" in result["message"]


@pytest.mark.asyncio
async def test_docker_rmi_missing_image():
    """Test docker rmi without image."""
    result = await docker_rmi(image="")

    assert result["success"] is False
    assert "Missing required parameter" in result["error"]


@pytest.mark.asyncio
async def test_docker_stats_success():
    """Test docker stats."""
    with patch("victor.tools.docker_tool._check_docker", return_value=True):
        with patch("victor.tools.docker_tool._run_docker_command") as mock_run:
            mock_run.return_value = (True, '{"container":"test","cpu":"0.5%"}', "")

            result = await docker_stats(container="test_container")

            assert result["success"] is True
            assert "stats" in result


@pytest.mark.asyncio
async def test_docker_stats_all_containers():
    """Test docker stats for all containers."""
    with patch("victor.tools.docker_tool._check_docker", return_value=True):
        with patch("victor.tools.docker_tool._run_docker_command") as mock_run:
            mock_run.return_value = (True, '{"container":"all"}', "")

            result = await docker_stats()

            assert result["success"] is True
            assert "stats" in result


@pytest.mark.asyncio
async def test_docker_inspect_container():
    """Test docker inspect for container."""
    with patch("victor.tools.docker_tool._check_docker", return_value=True):
        with patch("victor.tools.docker_tool._run_docker_command") as mock_run:
            mock_run.return_value = (True, '{"Id":"abc123","Name":"test"}', "")

            result = await docker_inspect(container="test_container")

            assert result["success"] is True
            assert "details" in result


@pytest.mark.asyncio
async def test_docker_inspect_image():
    """Test docker inspect for image."""
    with patch("victor.tools.docker_tool._check_docker", return_value=True):
        with patch("victor.tools.docker_tool._run_docker_command") as mock_run:
            mock_run.return_value = (True, '{"Id":"img123","RepoTags":["nginx:latest"]}', "")

            result = await docker_inspect(image="nginx:latest")

            assert result["success"] is True
            assert "details" in result


@pytest.mark.asyncio
async def test_docker_inspect_missing_target():
    """Test docker inspect without container or image."""
    result = await docker_inspect()

    assert result["success"] is False
    assert "Missing required parameter" in result["error"]


@pytest.mark.asyncio
async def test_docker_networks_success():
    """Test docker networks list."""
    with patch("victor.tools.docker_tool._check_docker", return_value=True):
        with patch("victor.tools.docker_tool._run_docker_command") as mock_run:
            mock_run.return_value = (True, '{"Name":"bridge","Driver":"bridge"}', "")

            result = await docker_networks()

            assert result["success"] is True
            assert "networks" in result


@pytest.mark.asyncio
async def test_docker_volumes_success():
    """Test docker volumes list."""
    with patch("victor.tools.docker_tool._check_docker", return_value=True):
        with patch("victor.tools.docker_tool._run_docker_command") as mock_run:
            mock_run.return_value = (True, '{"Name":"my-volume","Driver":"local"}', "")

            result = await docker_volumes()

            assert result["success"] is True
            assert "volumes" in result


@pytest.mark.asyncio
async def test_docker_exec_success():
    """Test docker exec command."""
    with patch("victor.tools.docker_tool._check_docker", return_value=True):
        with patch("victor.tools.docker_tool._run_docker_command") as mock_run:
            mock_run.return_value = (True, "command output", "")

            result = await docker_exec(container="test_container", command="ls -la")

            assert result["success"] is True
            assert "output" in result


@pytest.mark.asyncio
async def test_docker_exec_missing_container():
    """Test docker exec without container."""
    result = await docker_exec(container="", command="ls")

    assert result["success"] is False
    assert "Missing required parameter: container" in result["error"]


@pytest.mark.asyncio
async def test_docker_exec_missing_command():
    """Test docker exec without command."""
    result = await docker_exec(container="test", command="")

    assert result["success"] is False
    assert "Missing required parameter: command" in result["error"]


@pytest.mark.asyncio
async def test_docker_command_failure():
    """Test docker command failure handling."""
    with patch("victor.tools.docker_tool._check_docker", return_value=True):
        with patch("victor.tools.docker_tool._run_docker_command") as mock_run:
            mock_run.return_value = (False, "", "Error: container not found")

            result = await docker_ps()

            assert result["success"] is False
            assert "Error" in result["error"]
