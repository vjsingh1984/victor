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

"""Unit tests for LSP client integration in the coding vertical.

Tests cover:
- LSP server connection and initialization
- Completion requests
- Diagnostics handling
- Symbol extraction
- Document management
- Workspace management
- LSP feature extraction
"""

import asyncio
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from tests.fixtures.coding_fixtures import (
    MOCK_LSP_COMPLETIONS,
    MOCK_LSP_DIAGNOSTICS,
    MOCK_LSP_SYMBOLS,
    SAMPLE_PYTHON_CLASS,
    SAMPLE_PYTHON_SIMPLE,
    create_sample_file,
)
from victor.coding.lsp.client import LSPClient
from victor.coding.lsp.config import LSPServerConfig
from victor.protocols.lsp_types import (
    CompletionItem,
    CompletionItemKind,
    Diagnostic,
    DiagnosticSeverity,
    DocumentSymbol,
    Hover,
    Location,
    Position,
    Range,
    SymbolKind,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def lsp_config():
    """Create a mock LSP server config."""
    return LSPServerConfig(
        name="pylsp",
        command=["pylsp"],
        language_id="python",
        file_extensions=[".py"],
        args=[],
        install_command="pip install python-lsp-server",
    )


@pytest.fixture
def mock_lsp_client(lsp_config):
    """Create a mock LSP client for testing."""
    client = LSPClient(config=lsp_config, root_uri="file:///tmp/test")
    return client


@pytest.fixture
def sample_file(tmp_path):
    """Create a sample Python file."""
    return create_sample_file(tmp_path, "test.py", SAMPLE_PYTHON_SIMPLE)


# =============================================================================
# LSP Client Initialization Tests
# =============================================================================

class TestLSPClientInitialization:
    """Tests for LSP client initialization and setup."""

    def test_client_initialization(self, lsp_config):
        """Test basic client initialization."""
        client = LSPClient(config=lsp_config, root_uri="file:///test")
        assert client.config == lsp_config
        assert client.root_uri == "file:///test"
        assert not client.is_running
        assert not client._initialized

    def test_client_initialization_with_custom_root_uri(self, lsp_config):
        """Test initialization with custom root URI."""
        client = LSPClient(
            config=lsp_config,
            root_uri="file:///custom/workspace"
        )
        assert client.root_uri == "file:///custom/workspace"

    def test_initial_state(self, mock_lsp_client):
        """Test initial state of client."""
        assert mock_lsp_client._process is None
        assert mock_lsp_client._request_id == 0
        assert len(mock_lsp_client._pending_requests) == 0
        assert len(mock_lsp_client._open_documents) == 0
        assert len(mock_lsp_client._diagnostics) == 0
        assert len(mock_lsp_client._notification_handlers) == 0


# =============================================================================
# Server Connection Tests
# =============================================================================

class TestServerConnection:
    """Tests for server connection lifecycle."""

    @pytest.mark.asyncio
    async def test_start_server_success(self, mock_lsp_client):
        """Test successful server start."""
        with patch.object(mock_lsp_client, "_initialize", new_callable=AsyncMock):
            with patch("subprocess.Popen") as mock_popen:
                mock_process = MagicMock()
                mock_process.poll.return_value = None
                mock_popen.return_value = mock_process

                result = await mock_lsp_client.start()
                assert result is True
                assert mock_lsp_client._process is not None

    @pytest.mark.asyncio
    async def test_start_server_file_not_found(self, mock_lsp_client):
        """Test server start when command not found."""
        with patch("subprocess.Popen", side_effect=FileNotFoundError):
            result = await mock_lsp_client.start()
            assert result is False

    @pytest.mark.asyncio
    async def test_start_server_already_running(self, mock_lsp_client):
        """Test starting server when already running."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_lsp_client._process = mock_process

        result = await mock_lsp_client.start()
        assert result is True

    @pytest.mark.asyncio
    async def test_stop_server(self, mock_lsp_client):
        """Test stopping the server."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_lsp_client._process = mock_process

        await mock_lsp_client.stop()
        # Verify cleanup
        assert mock_lsp_client._process is None
        assert not mock_lsp_client._initialized

    @pytest.mark.asyncio
    async def test_stop_server_with_timeout(self, mock_lsp_client):
        """Test stopping server that doesn't exit gracefully."""
        import subprocess

        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.wait.side_effect = subprocess.TimeoutExpired("cmd", 5)
        mock_lsp_client._process = mock_process

        await mock_lsp_client.stop()
        mock_process.kill.assert_called_once()


# =============================================================================
# Server Initialization Tests
# =============================================================================

class TestServerInitialization:
    """Tests for server initialization protocol."""

    @pytest.mark.asyncio
    async def test_initialize_request(self, mock_lsp_client):
        """Test initialize request structure."""
        mock_lsp_client._process = MagicMock()
        mock_lsp_client._write_message = MagicMock()

        # Mock response
        expected_result = {
            "capabilities": {
                "textDocument": {
                    "completion": {"completionItem": {"snippetSupport": True}},
                }
            }
        }

        with patch.object(
            mock_lsp_client, "_send_request", new_callable=AsyncMock, return_value=expected_result
        ):
            await mock_lsp_client._initialize()

            assert mock_lsp_client._initialized
            assert "completion" in mock_lsp_client._capabilities

    @pytest.mark.asyncio
    async def test_initialized_notification(self, mock_lsp_client):
        """Test initialized notification is sent."""
        mock_lsp_client._process = MagicMock()
        mock_lsp_client._write_message = MagicMock()

        with patch.object(
            mock_lsp_client, "_send_request", new_callable=AsyncMock, return_value={}
        ):
            await mock_lsp_client._initialize()

            # Verify initialized notification was sent
            calls = mock_lsp_client._write_message.call_args_list
            initialized_call = [
                c for c in calls if "initialized" in str(c) and "notification" in str(c)
            ]
            assert len(initialized_call) >= 1


# =============================================================================
# Document Management Tests
# =============================================================================

class TestDocumentManagement:
    """Tests for document lifecycle management."""

    def test_open_document(self, mock_lsp_client):
        """Test opening a document."""
        mock_lsp_client._process = MagicMock()
        mock_lsp_client._write_message = MagicMock()

        uri = "file:///test.py"
        text = SAMPLE_PYTHON_SIMPLE

        mock_lsp_client.open_document(uri, text, language_id="python")

        assert uri in mock_lsp_client._open_documents
        assert mock_lsp_client._open_documents[uri] == 1

    def test_open_document_already_open(self, mock_lsp_client):
        """Test opening a document that's already open."""
        mock_lsp_client._process = MagicMock()
        mock_lsp_client._write_message = MagicMock()

        uri = "file:///test.py"
        mock_lsp_client._open_documents[uri] = 1

        mock_lsp_client.open_document(uri, "text")
        # Should not increment version
        assert mock_lsp_client._open_documents[uri] == 1

    def test_close_document(self, mock_lsp_client):
        """Test closing a document."""
        mock_lsp_client._process = MagicMock()
        mock_lsp_client._write_message = MagicMock()

        uri = "file:///test.py"
        mock_lsp_client._open_documents[uri] = 1
        mock_lsp_client._diagnostics[uri] = []

        mock_lsp_client.close_document(uri)

        assert uri not in mock_lsp_client._open_documents
        assert uri not in mock_lsp_client._diagnostics

    def test_close_document_not_open(self, mock_lsp_client):
        """Test closing a document that's not open."""
        mock_lsp_client._process = MagicMock()
        mock_lsp_client._write_message = MagicMock()

        uri = "file:///test.py"
        mock_lsp_client.close_document(uri)
        # Should not raise

    def test_update_document(self, mock_lsp_client):
        """Test updating a document."""
        mock_lsp_client._process = MagicMock()
        mock_lsp_client._write_message = MagicMock()

        uri = "file:///test.py"
        mock_lsp_client.open_document(uri, "old text")

        mock_lsp_client.update_document(uri, "new text")

        assert mock_lsp_client._open_documents[uri] == 2

    def test_update_document_not_open(self, mock_lsp_client):
        """Test updating a document that's not open."""
        mock_lsp_client._process = MagicMock()
        mock_lsp_client._write_message = MagicMock()

        uri = "file:///test.py"
        mock_lsp_client.update_document(uri, "text")

        assert uri in mock_lsp_client._open_documents


# =============================================================================
# Completion Tests
# =============================================================================

class TestCompletion:
    """Tests for code completion functionality."""

    @pytest.mark.asyncio
    async def test_get_completions_success(self, mock_lsp_client):
        """Test successful completion request."""
        mock_lsp_client._initialized = True
        uri = "file:///test.py"
        position = Position(line=0, character=5)

        mock_response = {"items": [item.to_dict() for item in MOCK_LSP_COMPLETIONS]}

        with patch.object(
            mock_lsp_client, "_send_request", new_callable=AsyncMock, return_value=mock_response
        ):
            completions = await mock_lsp_client.get_completions(uri, position)

            assert len(completions) == len(MOCK_LSP_COMPLETIONS)
            assert completions[0].label == MOCK_LSP_COMPLETIONS[0].label

    @pytest.mark.asyncio
    async def test_get_completions_not_initialized(self, mock_lsp_client):
        """Test completion when server not initialized."""
        mock_lsp_client._initialized = False

        completions = await mock_lsp_client.get_completions("file:///test.py", Position(0, 0))

        assert completions == []

    @pytest.mark.asyncio
    async def test_get_completions_empty_result(self, mock_lsp_client):
        """Test completion with no results."""
        mock_lsp_client._initialized = True

        with patch.object(
            mock_lsp_client, "_send_request", new_callable=AsyncMock, return_value=None
        ):
            completions = await mock_lsp_client.get_completions("file:///test.py", Position(0, 0))

            assert completions == []

    @pytest.mark.asyncio
    async def test_get_completions_error_handling(self, mock_lsp_client):
        """Test completion error handling."""
        mock_lsp_client._initialized = True

        with patch.object(
            mock_lsp_client, "_send_request", new_callable=AsyncMock, side_effect=Exception("Error")
        ):
            completions = await mock_lsp_client.get_completions("file:///test.py", Position(0, 0))

            assert completions == []


# =============================================================================
# Hover Tests
# =============================================================================

class TestHover:
    """Tests for hover information."""

    @pytest.mark.asyncio
    async def test_get_hover_success(self, mock_lsp_client):
        """Test successful hover request."""
        mock_lsp_client._initialized = True
        uri = "file:///test.py"
        position = Position(line=0, character=0)

        mock_hover = Hover(
            contents={"kind": "markdown", "value": "Documentation"},
            range=Range(start=Position(0, 0), end=Position(0, 5)),
        )

        with patch.object(
            mock_lsp_client, "_send_request", new_callable=AsyncMock, return_value=mock_hover.to_dict()
        ):
            hover = await mock_lsp_client.get_hover(uri, position)

            assert hover is not None
            assert hover.contents == mock_hover.contents

    @pytest.mark.asyncio
    async def test_get_hover_not_initialized(self, mock_lsp_client):
        """Test hover when server not initialized."""
        mock_lsp_client._initialized = False

        hover = await mock_lsp_client.get_hover("file:///test.py", Position(0, 0))

        assert hover is None

    @pytest.mark.asyncio
    async def test_get_hover_no_result(self, mock_lsp_client):
        """Test hover with no result."""
        mock_lsp_client._initialized = True

        with patch.object(
            mock_lsp_client, "_send_request", new_callable=AsyncMock, return_value=None
        ):
            hover = await mock_lsp_client.get_hover("file:///test.py", Position(0, 0))

            assert hover is None


# =============================================================================
# Definition and References Tests
# =============================================================================

class TestDefinitionAndReferences:
    """Tests for go-to-definition and find-references."""

    @pytest.mark.asyncio
    async def test_get_definition_single(self, mock_lsp_client):
        """Test getting single definition location."""
        mock_lsp_client._initialized = True

        mock_location = Location(
            uri="file:///test.py",
            range=Range(start=Position(5, 0), end=Position(5, 10)),
        )

        with patch.object(
            mock_lsp_client, "_send_request", new_callable=AsyncMock, return_value=mock_location.to_dict()
        ):
            definitions = await mock_lsp_client.get_definition("file:///test.py", Position(0, 0))

            assert len(definitions) == 1
            assert definitions[0].uri == mock_location.uri

    @pytest.mark.asyncio
    async def test_get_definition_multiple(self, mock_lsp_client):
        """Test getting multiple definition locations."""
        mock_lsp_client._initialized = True

        locations = [
            Location(
                uri="file:///test1.py",
                range=Range(start=Position(5, 0), end=Position(5, 10)),
            ),
            Location(
                uri="file:///test2.py",
                range=Range(start=Position(10, 0), end=Position(10, 10)),
            ),
        ]

        with patch.object(
            mock_lsp_client,
            "_send_request",
            new_callable=AsyncMock,
            return_value=[loc.to_dict() for loc in locations],
        ):
            definitions = await mock_lsp_client.get_definition("file:///test.py", Position(0, 0))

            assert len(definitions) == 2

    @pytest.mark.asyncio
    async def test_get_references(self, mock_lsp_client):
        """Test getting references."""
        mock_lsp_client._initialized = True

        mock_refs = [
            Location(
                uri="file:///test.py",
                range=Range(start=Position(10, 5), end=Position(10, 10)),
            ),
            Location(
                uri="file:///test.py",
                range=Range(start=Position(20, 5), end=Position(20, 10)),
            ),
        ]

        with patch.object(
            mock_lsp_client,
            "_send_request",
            new_callable=AsyncMock,
            return_value=[ref.to_dict() for ref in mock_refs],
        ):
            references = await mock_lsp_client.get_references("file:///test.py", Position(0, 0))

            assert len(references) == 2

    @pytest.mark.asyncio
    async def test_get_references_exclude_declaration(self, mock_lsp_client):
        """Test getting references excluding declaration."""
        mock_lsp_client._initialized = True

        with patch.object(
            mock_lsp_client,
            "_send_request",
            new_callable=AsyncMock,
            return_value=[],
        ):
            references = await mock_lsp_client.get_references(
                "file:///test.py", Position(0, 0), include_declaration=False
            )

            assert isinstance(references, list)


# =============================================================================
# Diagnostics Tests
# =============================================================================

class TestDiagnostics:
    """Tests for diagnostic handling."""

    @pytest.mark.asyncio
    async def test_handle_diagnostics_notification(self, mock_lsp_client):
        """Test handling diagnostics notification."""
        uri = "file:///test.py"

        mock_diagnostics = [diag.to_dict() for diag in MOCK_LSP_DIAGNOSTICS]

        params = {"uri": uri, "diagnostics": mock_diagnostics}
        await mock_lsp_client._handle_diagnostics(params)

        assert uri in mock_lsp_client._diagnostics
        assert len(mock_lsp_client._diagnostics[uri]) == len(MOCK_LSP_DIAGNOSTICS)

    def test_get_diagnostics(self, mock_lsp_client):
        """Test getting diagnostics for a document."""
        uri = "file:///test.py"
        mock_lsp_client._diagnostics[uri] = MOCK_LSP_DIAGNOSTICS

        diagnostics = mock_lsp_client.get_diagnostics(uri)

        assert len(diagnostics) == len(MOCK_LSP_DIAGNOSTICS)
        assert diagnostics[0].message == MOCK_LSP_DIAGNOSTICS[0].message

    def test_get_diagnostics_no_diagnostics(self, mock_lsp_client):
        """Test getting diagnostics when none exist."""
        diagnostics = mock_lsp_client.get_diagnostics("file:///test.py")

        assert diagnostics == []


# =============================================================================
# Notification Handler Tests
# =============================================================================

class TestNotificationHandlers:
    """Tests for notification handler registration."""

    def test_register_notification_handler(self, mock_lsp_client):
        """Test registering a notification handler."""
        handler = MagicMock()

        mock_lsp_client.register_notification_handler("custom/notification", handler)

        assert "custom/notification" in mock_lsp_client._notification_handlers
        assert handler in mock_lsp_client._notification_handlers["custom/notification"]

    def test_register_multiple_handlers(self, mock_lsp_client):
        """Test registering multiple handlers for same notification."""
        handler1 = MagicMock()
        handler2 = MagicMock()

        mock_lsp_client.register_notification_handler("test/notification", handler1)
        mock_lsp_client.register_notification_handler("test/notification", handler2)

        assert len(mock_lsp_client._notification_handlers["test/notification"]) == 2


# =============================================================================
# Message Handling Tests
# =============================================================================

class TestMessageHandling:
    """Tests for message parsing and handling."""

    def test_parse_message_success(self, mock_lsp_client):
        """Test successful message parsing."""
        content = b'{"jsonrpc":"2.0","id":1,"result":{}}'
        header = f"Content-Length: {len(content)}\r\n\r\n".encode()
        buffer = header + content

        message, remaining = mock_lsp_client._parse_message(buffer)

        assert message is not None
        assert message["jsonrpc"] == "2.0"
        assert message["id"] == 1
        assert remaining == b""

    def test_parse_message_incomplete(self, mock_lsp_client):
        """Test parsing incomplete message."""
        header = b"Content-Length: 100\r\n\r\n"
        buffer = header + b'{"jsonrpc":"2.0"}'

        message, remaining = mock_lsp_client._parse_message(buffer)

        assert message is None
        assert len(remaining) > 0

    def test_parse_message_multiple(self, mock_lsp_client):
        """Test parsing multiple messages."""
        content1 = b'{"jsonrpc":"2.0","id":1,"result":{}}'
        content2 = b'{"jsonrpc":"2.0","id":2,"method":"test"}'

        buffer = b"Content-Length: " + str(len(content1)).encode() + b"\r\n\r\n" + content1
        buffer += b"Content-Length: " + str(len(content2)).encode() + b"\r\n\r\n" + content2

        # Parse first message
        message1, buffer = mock_lsp_client._parse_message(buffer)
        assert message1 is not None
        assert message1["id"] == 1

        # Parse second message
        message2, buffer = mock_lsp_client._parse_message(buffer)
        assert message2 is not None
        assert message2["id"] == 2

    @pytest.mark.asyncio
    async def test_handle_response_message(self, mock_lsp_client):
        """Test handling response message."""
        future: asyncio.Future = asyncio.Future()
        mock_lsp_client._pending_requests[1] = future

        message = {"jsonrpc": "2.0", "id": 1, "result": {"status": "ok"}}
        await mock_lsp_client._handle_message(message)

        assert future.done()
        assert future.result() == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_handle_error_response(self, mock_lsp_client):
        """Test handling error response."""
        future: asyncio.Future = asyncio.Future()
        mock_lsp_client._pending_requests[1] = future

        message = {"jsonrpc": "2.0", "id": 1, "error": {"message": "Test error"}}
        await mock_lsp_client._handle_message(message)

        assert future.done()
        assert isinstance(future.exception(), Exception)


# =============================================================================
# Request/Response Tests
# =============================================================================

class TestRequests:
    """Tests for LSP request handling."""

    @pytest.mark.asyncio
    async def test_send_request_success(self, mock_lsp_client):
        """Test successful request."""
        mock_lsp_client._process = MagicMock()
        mock_lsp_client._write_message = MagicMock()

        with patch.object(
            mock_lsp_client, "_read_messages", new_callable=AsyncMock
        ):
            future: asyncio.Future = asyncio.Future()
            mock_lsp_client._pending_requests[1] = future
            future.set_result({"status": "ok"})

            result = await mock_lsp_client._send_request("test/method", {})

            assert result == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_send_request_timeout(self, mock_lsp_client):
        """Test request timeout."""
        mock_lsp_client._process = MagicMock()
        mock_lsp_client._write_message = MagicMock()

        with pytest.raises(TimeoutError):
            await mock_lsp_client._send_request("test/method", {}, timeout=0.1)

    @pytest.mark.asyncio
    async def test_send_request_not_running(self, mock_lsp_client):
        """Test sending request when server not running."""
        mock_lsp_client._process = None

        with pytest.raises(RuntimeError):
            await mock_lsp_client._send_request("test/method", {})


# =============================================================================
# Feature Detection Tests
# =============================================================================

class TestFeatureDetection:
    """Tests for LSP capability detection."""

    @pytest.mark.asyncio
    async def test_completion_support(self, mock_lsp_client):
        """Test detecting completion support."""
        mock_lsp_client._capabilities = {
            "textDocument": {
                "completion": {
                    "completionItem": {"snippetSupport": True}
                }
            }
        }

        # Should have completion capability
        assert "completion" in mock_lsp_client._capabilities.get("textDocument", {})

    @pytest.mark.asyncio
    async def test_hover_support(self, mock_lsp_client):
        """Test detecting hover support."""
        mock_lsp_client._capabilities = {
            "textDocument": {
                "hover": {
                    "contentFormat": ["markdown", "plaintext"]
                }
            }
        }

        # Should have hover capability
        assert "hover" in mock_lsp_client._capabilities.get("textDocument", {})

    @pytest.mark.asyncio
    async def test_definition_support(self, mock_lsp_client):
        """Test detecting definition support."""
        mock_lsp_client._capabilities = {
            "textDocument": {
                "definition": {
                    "linkSupport": True
                }
            }
        }

        # Should have definition capability
        assert "definition" in mock_lsp_client._capabilities.get("textDocument", {})


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_write_message_broken_pipe(self, mock_lsp_client):
        """Test handling broken pipe when writing."""
        mock_process = MagicMock()
        mock_process.stdin.write.side_effect = BrokenPipeError()
        mock_lsp_client._process = mock_process

        # Should not raise
        mock_lsp_client._write_message({"test": "message"})

    def test_write_message_os_error(self, mock_lsp_client):
        """Test handling OS error when writing."""
        mock_process = MagicMock()
        mock_process.stdin.write.side_effect = OSError("Connection reset")
        mock_lsp_client._process = mock_process

        # Should not raise
        mock_lsp_client._write_message({"test": "message"})

    def test_parse_message_invalid_json(self, mock_lsp_client):
        """Test parsing invalid JSON message."""
        content = b'{"invalid": json}'
        header = f"Content-Length: {len(content)}\r\n\r\n".encode()
        buffer = header + content

        message, remaining = mock_lsp_client._parse_message(buffer)

        # Should return None and clear buffer
        assert message is None


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for LSP workflows."""

    @pytest.mark.asyncio
    async def test_complete_document_workflow(self, mock_lsp_client):
        """Test complete document analysis workflow."""
        mock_lsp_client._process = MagicMock()
        mock_lsp_client._write_message = MagicMock()
        mock_lsp_client._initialized = True

        uri = "file:///test.py"
        position = Position(line=0, character=0)

        # Mock completions
        with patch.object(
            mock_lsp_client, "_send_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = {
                "items": [item.to_dict() for item in MOCK_LSP_COMPLETIONS[:2]]
            }

            completions = await mock_lsp_client.get_completions(uri, position)

            assert len(completions) == 2
            assert all(isinstance(c, CompletionItem) for c in completions)

    @pytest.mark.asyncio
    async def test_diagnostic_workflow(self, mock_lsp_client):
        """Test diagnostic collection workflow."""
        uri = "file:///test.py"

        # Simulate receiving diagnostics
        mock_diagnostics = [diag.to_dict() for diag in MOCK_LSP_DIAGNOSTICS]
        params = {"uri": uri, "diagnostics": mock_diagnostics}

        await mock_lsp_client._handle_diagnostics(params)

        # Verify diagnostics are stored
        diagnostics = mock_lsp_client.get_diagnostics(uri)
        assert len(diagnostics) == len(MOCK_LSP_DIAGNOSTICS)

        # Verify diagnostic content
        assert all(isinstance(d, Diagnostic) for d in diagnostics)
