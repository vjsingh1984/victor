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

"""Enhanced unit tests for LSP client integration.

This test suite provides comprehensive coverage for:
- LSP client lifecycle management
- Document operations
- Completion requests
- Hover information
- Diagnostics handling
- Definition and reference lookup
- Error handling
- Message parsing
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.fixtures.coding_fixtures import (
    MOCK_LSP_COMPLETIONS,
    MOCK_LSP_DIAGNOSTICS,
    MOCK_LSP_HOVER_RESPONSES,
)
from victor.coding.lsp.client import LSPClient
from victor.coding.lsp.config import LSPServerConfig
from victor.protocols.lsp_types import (
    CompletionItem,
    Hover,
    Location,
    Position,
    Range,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def lsp_config():
    """Create a mock LSP server configuration."""
    return LSPServerConfig(
        name="test-server",
        command=["echo"],
        language_id="python",
        file_extensions=[".py", ".pyi"],
        args=[],
        install_command="pip install test-server",
    )


@pytest.fixture
def lsp_client(lsp_config):
    """Create an LSP client instance."""
    return LSPClient(config=lsp_config, root_uri="file:///test")


@pytest.fixture
def mock_process():
    """Create a mock subprocess process."""
    process = MagicMock()
    process.poll.return_value = None  # Process is running
    process.stdin = MagicMock()
    process.stdout = MagicMock()
    process.stderr = MagicMock()
    return process


# =============================================================================
# LSP Client Lifecycle Tests
# =============================================================================


class TestLSPClientLifecycle:
    """Tests for LSP client startup and shutdown."""

    def test_client_initialization(self, lsp_client, lsp_config):
        """Test client initialization."""
        assert lsp_client.config == lsp_config
        assert lsp_client.root_uri == "file:///test"
        assert not lsp_client.is_running
        assert lsp_client._request_id == 0

    def test_is_running_false_when_no_process(self, lsp_client):
        """Test is_running returns False when no process."""
        assert not lsp_client.is_running

    def test_is_running_true_when_process_active(self, lsp_client, mock_process):
        """Test is_running returns True when process is running."""
        lsp_client._process = mock_process
        assert lsp_client.is_running

    def test_is_running_false_when_process_exited(self, lsp_client, mock_process):
        """Test is_running returns False when process has exited."""
        mock_process.poll.return_value = 0  # Exit code
        lsp_client._process = mock_process
        assert not lsp_client.is_running

    @pytest.mark.asyncio
    async def test_start_server_success(self, lsp_client, mock_process):
        """Test successful server start."""
        with patch("subprocess.Popen", return_value=mock_process):
            with patch.object(lsp_client, "_initialize", new_callable=AsyncMock):
                result = await lsp_client.start()
                assert result is True
                assert lsp_client._process == mock_process

    @pytest.mark.asyncio
    async def test_start_server_already_running(self, lsp_client, mock_process):
        """Test starting server when already running."""
        lsp_client._process = mock_process
        result = await lsp_client.start()
        assert result is True

    @pytest.mark.asyncio
    async def test_start_server_file_not_found(self, lsp_client):
        """Test server start when executable not found."""
        with patch("subprocess.Popen", side_effect=FileNotFoundError):
            result = await lsp_client.start()
            assert result is False

    @pytest.mark.asyncio
    async def test_start_server_exception(self, lsp_client):
        """Test server start with unexpected exception."""
        with patch("subprocess.Popen", side_effect=Exception("Unknown error")):
            result = await lsp_client.start()
            assert result is False

    @pytest.mark.asyncio
    async def test_stop_server(self, lsp_client, mock_process):
        """Test server stop."""
        lsp_client._process = mock_process
        mock_process.wait.return_value = None

        with patch.object(lsp_client, "_send_request", new_callable=AsyncMock):
            with patch.object(lsp_client, "_send_notification"):
                await lsp_client.stop()
                mock_process.wait.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_server_not_running(self, lsp_client):
        """Test stopping server when not running."""
        # Should not raise exception
        await lsp_client.stop()

    @pytest.mark.asyncio
    async def test_stop_server_timeout(self, lsp_client, mock_process):
        """Test server stop with timeout."""
        from subprocess import TimeoutExpired

        mock_process.wait.side_effect = TimeoutExpired("cmd", 5)
        lsp_client._process = mock_process

        with patch.object(lsp_client, "_send_request", new_callable=AsyncMock):
            with patch.object(lsp_client, "_send_notification"):
                await lsp_client.stop()
                mock_process.kill.assert_called_once()


# =============================================================================
# Request ID Management Tests
# =============================================================================


class TestRequestIdManagement:
    """Tests for request ID generation."""

    def test_get_next_id_increments(self, lsp_client):
        """Test that request IDs increment."""
        initial_id = lsp_client._get_next_id()
        next_id = lsp_client._get_next_id()
        assert next_id == initial_id + 1

    def test_get_next_id_multiple_calls(self, lsp_client):
        """Test multiple request ID generations."""
        ids = [lsp_client._get_next_id() for _ in range(10)]
        assert ids == list(range(1, 11))


# =============================================================================
# Message Writing Tests
# =============================================================================


class TestMessageWriting:
    """Tests for writing messages to LSP server."""

    def test_write_message_success(self, lsp_client, mock_process):
        """Test successful message writing."""
        message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "test",
        }
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.flush = MagicMock()

        lsp_client._process = mock_process
        lsp_client._write_message(message)

        # Verify write and flush were called
        assert mock_process.stdin.write.call_count == 2  # header + content
        mock_process.stdin.flush.assert_called_once()

    def test_write_message_no_process(self, lsp_client):
        """Test writing when no process exists."""
        message = {"test": "data"}
        # Should not raise exception
        lsp_client._write_message(message)

    def test_write_message_no_stdin(self, lsp_client, mock_process):
        """Test writing when stdin is None."""
        mock_process.stdin = None
        lsp_client._process = mock_process

        message = {"test": "data"}
        # Should not raise exception
        lsp_client._write_message(message)

    def test_write_message_broken_pipe(self, lsp_client, mock_process):
        """Test handling of broken pipe error."""
        mock_process.stdin.write.side_effect = BrokenPipeError()
        lsp_client._process = mock_process

        message = {"test": "data"}
        # Should not raise exception
        lsp_client._write_message(message)


# =============================================================================
# Message Parsing Tests
# =============================================================================


class TestMessageParsing:
    """Tests for parsing LSP messages."""

    def test_parse_complete_message(self, lsp_client):
        """Test parsing a complete message."""
        content = '{"jsonrpc": "2.0", "id": 1, "result": {}}'
        buffer = f"Content-Length: {len(content)}\r\n\r\n{content}".encode()

        message, remaining = lsp_client._parse_message(buffer)

        assert message is not None
        assert message["id"] == 1
        assert remaining == b""

    def test_parse_partial_message(self, lsp_client):
        """Test parsing incomplete message."""
        buffer = b"Content-Length: 100\r\n\r\n"
        message, remaining = lsp_client._parse_message(buffer)

        assert message is None
        assert remaining == b""

    def test_parse_no_header_end(self, lsp_client):
        """Test parsing buffer without header end."""
        buffer = b"Content-Length: 100\r\n"
        message, remaining = lsp_client._parse_message(buffer)

        assert message is None
        assert remaining == buffer

    def test_parse_zero_content_length(self, lsp_client):
        """Test parsing with zero content length."""
        buffer = b"Content-Length: 0\r\n\r\n"
        message, remaining = lsp_client._parse_message(buffer)

        assert message is None
        assert remaining == b""

    def test_parse_insufficient_buffer(self, lsp_client):
        """Test parsing when buffer is smaller than content length."""
        buffer = b"Content-Length: 100\r\n\r\nincomplete"
        message, remaining = lsp_client._parse_message(buffer)

        assert message is None
        assert remaining == buffer

    def test_parse_invalid_json(self, lsp_client):
        """Test parsing invalid JSON."""
        content = "invalid json"
        buffer = f"Content-Length: {len(content)}\r\n\r\n{content}".encode()

        message, remaining = lsp_client._parse_message(buffer)

        assert message is None
        assert remaining == b""

    def test_parse_multiple_messages(self, lsp_client):
        """Test parsing multiple messages from buffer."""
        content1 = '{"id": 1}'
        content2 = '{"id": 2}'
        buffer = f"Content-Length: {len(content1)}\r\n\r\n{content1}Content-Length: {len(content2)}\r\n\r\n{content2}".encode()

        # Parse first message
        message1, remaining1 = lsp_client._parse_message(buffer)
        assert message1 is not None
        assert message1["id"] == 1

        # Parse second message from remaining buffer
        message2, remaining2 = lsp_client._parse_message(remaining1)
        assert message2 is not None
        assert message2["id"] == 2
        assert remaining2 == b""


# =============================================================================
# Document Operations Tests
# =============================================================================


class TestDocumentOperations:
    """Tests for document operations."""

    def test_open_document(self, lsp_client):
        """Test opening a document."""
        lsp_client._send_notification = MagicMock()

        lsp_client.open_document(
            uri="file:///test.py",
            text="print('hello')",
            language_id="python",
        )

        # Verify notification was sent
        lsp_client._send_notification.assert_called_once()
        call_args = lsp_client._send_notification.call_args
        assert call_args[0][0] == "textDocument/didOpen"
        assert "test.py" in call_args[0][1]["textDocument"]["uri"]

    def test_open_document_already_open(self, lsp_client):
        """Test opening a document that's already open."""
        lsp_client._open_documents = {"file:///test.py": 1}
        lsp_client._send_notification = MagicMock()

        lsp_client.open_document(
            uri="file:///test.py",
            text="print('hello')",
        )

        # Should not send notification if already open
        lsp_client._send_notification.assert_not_called()

    def test_open_document_default_language(self, lsp_client, lsp_config):
        """Test opening document with default language from config."""
        lsp_client._send_notification = MagicMock()

        lsp_client.open_document(
            uri="file:///test.py",
            text="print('hello')",
        )

        call_args = lsp_client._send_notification.call_args
        assert call_args[0][1]["textDocument"]["languageId"] == lsp_config.language_id

    def test_close_document(self, lsp_client):
        """Test closing a document."""
        lsp_client._open_documents = {"file:///test.py": 1}
        lsp_client._send_notification = MagicMock()

        lsp_client.close_document("file:///test.py")

        # Verify notification was sent
        lsp_client._send_notification.assert_called_once()
        assert "file:///test.py" not in lsp_client._open_documents

    def test_close_document_not_open(self, lsp_client):
        """Test closing a document that's not open."""
        lsp_client._send_notification = MagicMock()

        lsp_client.close_document("file:///test.py")

        # Should not send notification
        lsp_client._send_notification.assert_not_called()

    def test_update_document(self, lsp_client):
        """Test updating a document."""
        lsp_client._open_documents = {"file:///test.py": 1}
        lsp_client._send_notification = MagicMock()

        lsp_client.update_document(
            uri="file:///test.py",
            text="new content",
        )

        # Verify notification was sent and version incremented
        lsp_client._send_notification.assert_called_once()
        assert lsp_client._open_documents["file:///test.py"] == 2

    def test_update_document_not_open(self, lsp_client):
        """Test updating a document that's not open (should open it)."""
        lsp_client._send_notification = MagicMock()

        lsp_client.update_document(
            uri="file:///test.py",
            text="new content",
        )

        # Should open the document
        assert "file:///test.py" in lsp_client._open_documents


# =============================================================================
# Completion Tests
# =============================================================================


class TestCompletions:
    """Tests for code completion."""

    @pytest.mark.asyncio
    async def test_get_completions_success(self, lsp_client):
        """Test successful completion request."""
        lsp_client._initialized = True
        lsp_client._send_request = AsyncMock(
            return_value={"items": [item.to_dict() for item in MOCK_LSP_COMPLETIONS]}
        )

        position = Position(line=0, character=0)
        completions = await lsp_client.get_completions("file:///test.py", position)

        assert len(completions) == len(MOCK_LSP_COMPLETIONS)
        assert all(isinstance(c, CompletionItem) for c in completions)

    @pytest.mark.asyncio
    async def test_get_completions_not_initialized(self, lsp_client):
        """Test completion when not initialized."""
        lsp_client._initialized = False

        position = Position(line=0, character=0)
        completions = await lsp_client.get_completions("file:///test.py", position)

        assert completions == []

    @pytest.mark.asyncio
    async def test_get_completions_null_result(self, lsp_client):
        """Test completion with null result."""
        lsp_client._initialized = True
        lsp_client._send_request = AsyncMock(return_value=None)

        position = Position(line=0, character=0)
        completions = await lsp_client.get_completions("file:///test.py", position)

        assert completions == []

    @pytest.mark.asyncio
    async def test_get_completions_list_result(self, lsp_client):
        """Test completion with list result (not dict)."""
        lsp_client._initialized = True
        lsp_client._send_request = AsyncMock(
            return_value=[item.to_dict() for item in MOCK_LSP_COMPLETIONS]
        )

        position = Position(line=0, character=0)
        completions = await lsp_client.get_completions("file:///test.py", position)

        assert len(completions) > 0

    @pytest.mark.asyncio
    async def test_get_completions_exception(self, lsp_client):
        """Test completion with exception."""
        lsp_client._initialized = True
        lsp_client._send_request = AsyncMock(side_effect=Exception("Error"))

        position = Position(line=0, character=0)
        completions = await lsp_client.get_completions("file:///test.py", position)

        assert completions == []


# =============================================================================
# Hover Tests
# =============================================================================


class TestHover:
    """Tests for hover information."""

    @pytest.mark.asyncio
    async def test_get_hover_success(self, lsp_client):
        """Test successful hover request."""
        mock_hover = MOCK_LSP_HOVER_RESPONSES[0]
        lsp_client._initialized = True
        lsp_client._send_request = AsyncMock(return_value=mock_hover)

        position = Position(line=0, character=0)
        hover = await lsp_client.get_hover("file:///test.py", position)

        assert hover is not None
        assert isinstance(hover, Hover)

    @pytest.mark.asyncio
    async def test_get_hover_not_initialized(self, lsp_client):
        """Test hover when not initialized."""
        lsp_client._initialized = False

        position = Position(line=0, character=0)
        hover = await lsp_client.get_hover("file:///test.py", position)

        assert hover is None

    @pytest.mark.asyncio
    async def test_get_hover_null_result(self, lsp_client):
        """Test hover with null result."""
        lsp_client._initialized = True
        lsp_client._send_request = AsyncMock(return_value=None)

        position = Position(line=0, character=0)
        hover = await lsp_client.get_hover("file:///test.py", position)

        assert hover is None

    @pytest.mark.asyncio
    async def test_get_hover_exception(self, lsp_client):
        """Test hover with exception."""
        lsp_client._initialized = True
        lsp_client._send_request = AsyncMock(side_effect=Exception("Error"))

        position = Position(line=0, character=0)
        hover = await lsp_client.get_hover("file:///test.py", position)

        assert hover is None


# =============================================================================
# Definition Tests
# =============================================================================


class TestDefinition:
    """Tests for go-to-definition."""

    @pytest.mark.asyncio
    async def test_get_definition_single(self, lsp_client):
        """Test getting single definition location."""
        mock_location = Location(
            uri="file:///test.py",
            range=Range(
                start=Position(line=5, character=0),
                end=Position(line=5, character=10),
            ),
        )

        lsp_client._initialized = True
        lsp_client._send_request = AsyncMock(return_value=mock_location.to_dict())

        position = Position(line=0, character=0)
        definitions = await lsp_client.get_definition("file:///test.py", position)

        assert len(definitions) == 1
        assert definitions[0].uri == "file:///test.py"

    @pytest.mark.asyncio
    async def test_get_definition_multiple(self, lsp_client):
        """Test getting multiple definition locations."""
        locations = [
            Location(
                uri="file:///test1.py",
                range=Range(start=Position(line=0, character=0), end=Position(line=0, character=5)),
            ),
            Location(
                uri="file:///test2.py",
                range=Range(start=Position(line=0, character=0), end=Position(line=0, character=5)),
            ),
        ]

        lsp_client._initialized = True
        lsp_client._send_request = AsyncMock(return_value=[loc.to_dict() for loc in locations])

        position = Position(line=0, character=0)
        definitions = await lsp_client.get_definition("file:///test.py", position)

        assert len(definitions) == 2

    @pytest.mark.asyncio
    async def test_get_definition_null_result(self, lsp_client):
        """Test definition with null result."""
        lsp_client._initialized = True
        lsp_client._send_request = AsyncMock(return_value=None)

        position = Position(line=0, character=0)
        definitions = await lsp_client.get_definition("file:///test.py", position)

        assert definitions == []

    @pytest.mark.asyncio
    async def test_get_definition_not_initialized(self, lsp_client):
        """Test definition when not initialized."""
        lsp_client._initialized = False

        position = Position(line=0, character=0)
        definitions = await lsp_client.get_definition("file:///test.py", position)

        assert definitions == []


# =============================================================================
# Diagnostics Tests
# =============================================================================


class TestDiagnostics:
    """Tests for diagnostics handling."""

    @pytest.mark.asyncio
    async def test_handle_diagnostics_notification(self, lsp_client):
        """Test handling diagnostics notification."""
        params = {
            "uri": "file:///test.py",
            "diagnostics": [d.to_dict() for d in MOCK_LSP_DIAGNOSTICS],
        }

        await lsp_client._handle_diagnostics(params)

        diagnostics = lsp_client.get_diagnostics("file:///test.py")
        assert len(diagnostics) == len(MOCK_LSP_DIAGNOSTICS)

    def test_get_diagnostics_empty(self, lsp_client):
        """Test getting diagnostics when none exist."""
        diagnostics = lsp_client.get_diagnostics("file:///test.py")
        assert diagnostics == []

    def test_get_diagnostics_existing(self, lsp_client):
        """Test getting existing diagnostics."""
        lsp_client._diagnostics = {
            "file:///test.py": MOCK_LSP_DIAGNOSTICS,
        }

        diagnostics = lsp_client.get_diagnostics("file:///test.py")
        assert len(diagnostics) == len(MOCK_LSP_DIAGNOSTICS)

    def test_close_document_clears_diagnostics(self, lsp_client):
        """Test that closing document clears diagnostics."""
        lsp_client._open_documents = {"file:///test.py": 1}
        lsp_client._diagnostics = {"file:///test.py": MOCK_LSP_DIAGNOSTICS}
        lsp_client._send_notification = MagicMock()

        lsp_client.close_document("file:///test.py")

        assert "file:///test.py" not in lsp_client._diagnostics


# =============================================================================
# Request/Response Tests
# =============================================================================


class TestRequestResponse:
    """Tests for request/response handling."""

    @pytest.mark.asyncio
    async def test_send_request_success(self, lsp_client, mock_process):
        """Test successful request."""
        lsp_client._process = mock_process

        future = MagicMock()
        asyncio_future = asyncio.Future()
        asyncio_future.set_result({"result": "success"})
        future = asyncio_future

        with patch("asyncio.Future", return_value=future):
            result = await lsp_client._send_request("test_method", {})
            assert result is not None

    @pytest.mark.asyncio
    async def test_send_request_timeout(self, lsp_client, mock_process):
        """Test request timeout."""
        lsp_client._process = mock_process

        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            with pytest.raises(TimeoutError):
                await lsp_client._send_request("test_method", {}, timeout=0.1)

    @pytest.mark.asyncio
    async def test_send_request_not_running(self, lsp_client):
        """Test sending request when server not running."""
        with pytest.raises(RuntimeError):
            await lsp_client._send_request("test_method", {})


# =============================================================================
# Notification Handler Tests
# =============================================================================


class TestNotificationHandlers:
    """Tests for notification handler registration."""

    def test_register_notification_handler(self, lsp_client):
        """Test registering a notification handler."""
        handler = MagicMock()
        lsp_client.register_notification_handler("test/notification", handler)

        assert "test/notification" in lsp_client._notification_handlers
        assert handler in lsp_client._notification_handlers["test/notification"]

    def test_register_multiple_handlers(self, lsp_client):
        """Test registering multiple handlers for same notification."""
        handler1 = MagicMock()
        handler2 = MagicMock()

        lsp_client.register_notification_handler("test/notification", handler1)
        lsp_client.register_notification_handler("test/notification", handler2)

        assert len(lsp_client._notification_handlers["test/notification"]) == 2

    @pytest.mark.asyncio
    async def test_handle_message_with_notification(self, lsp_client):
        """Test handling a notification message."""
        handler = MagicMock()
        lsp_client.register_notification_handler("test/notification", handler)

        message = {
            "method": "test/notification",
            "params": {"data": "test"},
        }

        await lsp_client._handle_message(message)

        handler.assert_called_once_with({"data": "test"})

    @pytest.mark.asyncio
    async def test_handle_notification_exception(self, lsp_client):
        """Test handling notification that raises exception."""
        handler = MagicMock(side_effect=Exception("Handler error"))
        lsp_client.register_notification_handler("test/notification", handler)

        message = {
            "method": "test/notification",
            "params": {"data": "test"},
        }

        # Should not raise exception
        await lsp_client._handle_message(message)


# =============================================================================
# Integration Tests
# =============================================================================


class TestLSPIntegration:
    """Integration tests for LSP client."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, lsp_client, mock_process):
        """Test complete lifecycle: start, use, stop."""
        with patch("subprocess.Popen", return_value=mock_process):
            with patch.object(lsp_client, "_initialize", new_callable=AsyncMock):
                # Start
                assert await lsp_client.start() is True

                # Use
                lsp_client.open_document("file:///test.py", "content")

                # Stop
                await lsp_client.stop()
                assert not lsp_client.is_running


# =============================================================================
# Helper Imports
# =============================================================================

# Import asyncio for tests that need it
import asyncio
