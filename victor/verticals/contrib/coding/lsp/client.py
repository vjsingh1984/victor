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

"""LSP client implementation for communicating with language servers."""

import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from victor.verticals.contrib.coding.lsp.config import LSPServerConfig

# Use core LSP types from victor.protocols
from victor.protocols.lsp_types import (
    Position,
    Range,
    Location,
    Diagnostic,
    DiagnosticSeverity,
    CompletionItem,
    CompletionItemKind,
    Hover,
)

logger = logging.getLogger(__name__)


class LSPClient:
    """Client for communicating with a Language Server Protocol server."""

    def __init__(self, config: LSPServerConfig, root_uri: str):
        """Initialize the LSP client.

        Args:
            config: Server configuration
            root_uri: Root URI of the workspace
        """
        self.config = config
        self.root_uri = root_uri
        self._process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._initialized = False
        self._capabilities: Dict[str, Any] = {}
        self._open_documents: Dict[str, int] = {}  # uri -> version
        self._diagnostics: Dict[str, List[Diagnostic]] = {}
        self._notification_handlers: Dict[str, List[Callable]] = {}
        self._reader_task: Optional[asyncio.Task] = None

    @property
    def is_running(self) -> bool:
        """Check if the server process is running."""
        return self._process is not None and self._process.poll() is None

    async def start(self) -> bool:
        """Start the language server.

        Returns:
            True if started successfully
        """
        if self.is_running:
            logger.warning(f"Server {self.config.name} already running")
            return True

        try:
            cmd = self.config.command + self.config.args
            logger.info(f"Starting LSP server: {' '.join(cmd)}")

            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )

            # Start reader task
            self._reader_task = asyncio.create_task(self._read_messages())

            # Initialize the server
            await self._initialize()
            return True

        except FileNotFoundError:
            logger.error(f"LSP server not found: {self.config.command[0]}")
            logger.info(f"Install with: {self.config.install_command}")
            return False
        except Exception as e:
            logger.exception(f"Failed to start LSP server: {e}")
            return False

    async def stop(self) -> None:
        """Stop the language server."""
        if not self.is_running:
            return

        try:
            # Send shutdown request
            await self._send_request("shutdown", None)
            # Send exit notification
            self._send_notification("exit", None)

            # Wait for process to exit
            if self._process:
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()

        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")
            if self._process:
                self._process.kill()

        finally:
            if self._reader_task:
                self._reader_task.cancel()
                try:
                    await self._reader_task
                except asyncio.CancelledError:
                    pass

            self._process = None
            self._initialized = False
            self._open_documents.clear()

    async def _initialize(self) -> None:
        """Initialize the language server."""
        params = {
            "processId": os.getpid(),
            "rootUri": self.root_uri,
            "capabilities": {
                "textDocument": {
                    "completion": {
                        "completionItem": {
                            "snippetSupport": True,
                            "documentationFormat": ["markdown", "plaintext"],
                        }
                    },
                    "hover": {"contentFormat": ["markdown", "plaintext"]},
                    "definition": {"linkSupport": True},
                    "references": {},
                    "documentSymbol": {},
                    "publishDiagnostics": {"relatedInformation": True},
                },
                "workspace": {
                    "workspaceFolders": True,
                    "configuration": True,
                },
            },
            "initializationOptions": self.config.initialization_options,
            "workspaceFolders": [{"uri": self.root_uri, "name": Path(self.root_uri).name}],
        }

        result = await self._send_request("initialize", params)
        self._capabilities = result.get("capabilities", {})
        self._initialized = True

        # Send initialized notification
        self._send_notification("initialized", {})

        # Apply settings if any
        if self.config.settings:
            self._send_notification(
                "workspace/didChangeConfiguration",
                {"settings": self.config.settings},
            )

        logger.info(f"LSP server {self.config.name} initialized")

    def _get_next_id(self) -> int:
        """Get next request ID."""
        self._request_id += 1
        return self._request_id

    async def _send_request(self, method: str, params: Any, timeout: float = 30.0) -> Any:
        """Send a request to the server and wait for response.

        Args:
            method: LSP method name
            params: Request parameters
            timeout: Timeout in seconds

        Returns:
            Response result
        """
        if not self.is_running:
            raise RuntimeError("Server not running")

        request_id = self._get_next_id()
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        future: asyncio.Future = asyncio.Future()
        self._pending_requests[request_id] = future

        self._write_message(message)

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            del self._pending_requests[request_id]
            raise TimeoutError(f"Request {method} timed out")

    def _send_notification(self, method: str, params: Any) -> None:
        """Send a notification to the server (no response expected)."""
        if not self.is_running:
            return

        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        self._write_message(message)

    def _write_message(self, message: Dict[str, Any]) -> None:
        """Write a message to the server."""
        if not self._process or not self._process.stdin:
            return

        content = json.dumps(message)
        header = f"Content-Length: {len(content)}\r\n\r\n"

        try:
            self._process.stdin.write(header.encode("utf-8"))
            self._process.stdin.write(content.encode("utf-8"))
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            logger.error(f"Failed to write to server: {e}")

    async def _read_messages(self) -> None:
        """Read messages from the server."""
        if not self._process or not self._process.stdout:
            return

        buffer = b""
        while self.is_running:
            try:
                # Read in a non-blocking way
                chunk = await asyncio.get_event_loop().run_in_executor(
                    None, self._process.stdout.read, 4096
                )
                if not chunk:
                    break

                buffer += chunk

                # Parse messages from buffer
                while True:
                    message, buffer = self._parse_message(buffer)
                    if message is None:
                        break
                    await self._handle_message(message)

            except Exception as e:
                logger.error(f"Error reading from server: {e}")
                break

    def _parse_message(self, buffer: bytes) -> Tuple[Optional[Dict[str, Any]], bytes]:
        """Parse a message from the buffer.

        Returns:
            Tuple of (message or None, remaining buffer)
        """
        # Find header end
        header_end = buffer.find(b"\r\n\r\n")
        if header_end == -1:
            return None, buffer

        header = buffer[:header_end].decode("utf-8")
        content_length = 0

        for line in header.split("\r\n"):
            if line.lower().startswith("content-length:"):
                content_length = int(line.split(":")[1].strip())
                break

        if content_length == 0:
            return None, buffer[header_end + 4 :]

        content_start = header_end + 4
        content_end = content_start + content_length

        if len(buffer) < content_end:
            return None, buffer

        content = buffer[content_start:content_end].decode("utf-8")
        remaining = buffer[content_end:]

        try:
            message = json.loads(content)
            return message, remaining
        except json.JSONDecodeError:
            logger.error(f"Failed to parse message: {content[:100]}")
            return None, remaining

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle a message from the server."""
        if "id" in message:
            # Response to a request
            request_id = message["id"]
            if request_id in self._pending_requests:
                future = self._pending_requests.pop(request_id)
                if "error" in message:
                    future.set_exception(
                        RuntimeError(message["error"].get("message", "Unknown error"))
                    )
                else:
                    future.set_result(message.get("result"))
        else:
            # Notification from server
            method = message.get("method", "")
            params = message.get("params", {})

            if method == "textDocument/publishDiagnostics":
                await self._handle_diagnostics(params)
            elif method in self._notification_handlers:
                for handler in self._notification_handlers[method]:
                    try:
                        handler(params)
                    except Exception as e:
                        logger.error(f"Notification handler error: {e}")

    async def _handle_diagnostics(self, params: Dict[str, Any]) -> None:
        """Handle diagnostics notification."""
        uri = params.get("uri", "")
        diagnostics = [Diagnostic.from_dict(d) for d in params.get("diagnostics", [])]
        self._diagnostics[uri] = diagnostics
        logger.debug(f"Received {len(diagnostics)} diagnostics for {uri}")

    # Public API methods

    def open_document(self, uri: str, text: str, language_id: str = None) -> None:
        """Open a document in the server.

        Args:
            uri: Document URI
            text: Document text
            language_id: Language identifier (optional)
        """
        if uri in self._open_documents:
            return

        version = 1
        self._open_documents[uri] = version

        self._send_notification(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": language_id or self.config.language_id,
                    "version": version,
                    "text": text,
                }
            },
        )

    def close_document(self, uri: str) -> None:
        """Close a document."""
        if uri not in self._open_documents:
            return

        del self._open_documents[uri]
        self._diagnostics.pop(uri, None)

        self._send_notification(
            "textDocument/didClose",
            {"textDocument": {"uri": uri}},
        )

    def update_document(self, uri: str, text: str) -> None:
        """Update a document's contents.

        Args:
            uri: Document URI
            text: New document text
        """
        if uri not in self._open_documents:
            self.open_document(uri, text)
            return

        version = self._open_documents[uri] + 1
        self._open_documents[uri] = version

        self._send_notification(
            "textDocument/didChange",
            {
                "textDocument": {"uri": uri, "version": version},
                "contentChanges": [{"text": text}],
            },
        )

    async def get_completions(self, uri: str, position: Position) -> List[CompletionItem]:
        """Get completion items at a position.

        Args:
            uri: Document URI
            position: Cursor position

        Returns:
            List of completion items
        """
        if not self._initialized:
            return []

        params = {
            "textDocument": {"uri": uri},
            "position": position.to_dict(),
        }

        try:
            result = await self._send_request("textDocument/completion", params)
            if result is None:
                return []

            items = result.get("items", result) if isinstance(result, dict) else result
            return [CompletionItem.from_dict(item) for item in items]
        except Exception as e:
            logger.error(f"Completion error: {e}")
            return []

    async def get_hover(self, uri: str, position: Position) -> Optional[Hover]:
        """Get hover information at a position.

        Args:
            uri: Document URI
            position: Cursor position

        Returns:
            Hover information or None
        """
        if not self._initialized:
            return None

        params = {
            "textDocument": {"uri": uri},
            "position": position.to_dict(),
        }

        try:
            result = await self._send_request("textDocument/hover", params)
            if result:
                return Hover.from_dict(result)
        except Exception as e:
            logger.error(f"Hover error: {e}")

        return None

    async def get_definition(self, uri: str, position: Position) -> List[Location]:
        """Get definition locations.

        Args:
            uri: Document URI
            position: Cursor position

        Returns:
            List of definition locations
        """
        if not self._initialized:
            return []

        params = {
            "textDocument": {"uri": uri},
            "position": position.to_dict(),
        }

        try:
            result = await self._send_request("textDocument/definition", params)
            if result is None:
                return []

            if isinstance(result, dict):
                return [Location.from_dict(result)]
            return [Location.from_dict(loc) for loc in result]
        except Exception as e:
            logger.error(f"Definition error: {e}")
            return []

    async def get_references(
        self, uri: str, position: Position, include_declaration: bool = True
    ) -> List[Location]:
        """Get reference locations.

        Args:
            uri: Document URI
            position: Cursor position
            include_declaration: Include the declaration

        Returns:
            List of reference locations
        """
        if not self._initialized:
            return []

        params = {
            "textDocument": {"uri": uri},
            "position": position.to_dict(),
            "context": {"includeDeclaration": include_declaration},
        }

        try:
            result = await self._send_request("textDocument/references", params)
            if result is None:
                return []
            return [Location.from_dict(loc) for loc in result]
        except Exception as e:
            logger.error(f"References error: {e}")
            return []

    def get_diagnostics(self, uri: str) -> List[Diagnostic]:
        """Get current diagnostics for a document.

        Args:
            uri: Document URI

        Returns:
            List of diagnostics
        """
        return self._diagnostics.get(uri, [])

    def register_notification_handler(self, method: str, handler: Callable) -> None:
        """Register a handler for server notifications.

        Args:
            method: Notification method
            handler: Handler function
        """
        if method not in self._notification_handlers:
            self._notification_handlers[method] = []
        self._notification_handlers[method].append(handler)
