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

"""HTTP API Server for Victor.

Provides REST API endpoints for IDE integrations (VS Code, JetBrains, etc.)
and external tool access.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from aiohttp import web
from aiohttp.web import Request, Response, StreamResponse

logger = logging.getLogger(__name__)


class VictorAPIServer:
    """HTTP API Server for Victor."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        workspace_root: Optional[str] = None,
    ):
        """Initialize the API server.

        Args:
            host: Host to bind to
            port: Port to listen on
            workspace_root: Root directory of the workspace
        """
        self.host = host
        self.port = port
        self.workspace_root = workspace_root or str(Path.cwd())
        self._app = web.Application()
        self._orchestrator = None
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Set up API routes."""
        self._app.router.add_get("/health", self._health)
        self._app.router.add_get("/status", self._status)

        # Chat endpoints
        self._app.router.add_post("/chat", self._chat)
        self._app.router.add_post("/chat/stream", self._chat_stream)

        # Completion endpoints
        self._app.router.add_post("/completions", self._completions)

        # Search endpoints
        self._app.router.add_post("/search/semantic", self._semantic_search)
        self._app.router.add_post("/search/code", self._code_search)

        # Model/Mode management
        self._app.router.add_post("/model/switch", self._switch_model)
        self._app.router.add_post("/mode/switch", self._switch_mode)
        self._app.router.add_get("/models", self._list_models)

        # Conversation management
        self._app.router.add_post("/conversation/reset", self._reset_conversation)
        self._app.router.add_get("/conversation/export", self._export_conversation)

        # Undo/Redo
        self._app.router.add_post("/undo", self._undo)
        self._app.router.add_post("/redo", self._redo)
        self._app.router.add_get("/history", self._history)

        # Patch
        self._app.router.add_post("/patch/apply", self._apply_patch)
        self._app.router.add_post("/patch/create", self._create_patch)

        # LSP
        self._app.router.add_post("/lsp/completions", self._lsp_completions)
        self._app.router.add_post("/lsp/hover", self._lsp_hover)
        self._app.router.add_post("/lsp/definition", self._lsp_definition)
        self._app.router.add_post("/lsp/references", self._lsp_references)
        self._app.router.add_post("/lsp/diagnostics", self._lsp_diagnostics)

        # Server management
        self._app.router.add_post("/shutdown", self._shutdown)

        # WebSocket
        self._app.router.add_get("/ws", self._websocket_handler)

        # CORS middleware
        self._app.middlewares.append(self._cors_middleware)

        # WebSocket clients
        self._ws_clients: List[web.WebSocketResponse] = []

    @web.middleware
    async def _cors_middleware(self, request: Request, handler) -> Response:
        """Handle CORS for browser-based clients."""
        if request.method == "OPTIONS":
            return Response(
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type",
                }
            )

        response = await handler(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

    async def _health(self, request: Request) -> Response:
        """Health check endpoint."""
        return web.json_response({"status": "healthy", "version": "0.1.0"})

    async def _status(self, request: Request) -> Response:
        """Get current status."""
        from victor.agent.mode_controller import get_mode_controller
        from victor.agent.model_switcher import get_model_switcher

        mode_manager = get_mode_controller()
        model_switcher = get_model_switcher()

        return web.json_response(
            {
                "connected": True,
                "mode": mode_manager.current_mode.value,
                "provider": model_switcher.current_provider,
                "model": model_switcher.current_model,
                "workspace": self.workspace_root,
            }
        )

    async def _chat(self, request: Request) -> Response:
        """Chat endpoint (non-streaming)."""
        try:
            data = await request.json()
            messages = data.get("messages", [])

            if not messages:
                return web.json_response({"error": "No messages provided"}, status=400)

            # Get or create orchestrator
            orchestrator = await self._get_orchestrator()

            # Process chat
            response = await orchestrator.chat(messages[-1].get("content", ""))

            return web.json_response(
                {
                    "role": "assistant",
                    "content": response.get("content", ""),
                    "tool_calls": response.get("tool_calls", []),
                }
            )

        except Exception as e:
            logger.exception("Chat error")
            return web.json_response({"error": str(e)}, status=500)

    async def _chat_stream(self, request: Request) -> StreamResponse:
        """Streaming chat endpoint (Server-Sent Events)."""
        response = StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)

        try:
            data = await request.json()
            messages = data.get("messages", [])

            if not messages:
                await response.write(b'data: {"type": "error", "message": "No messages"}\n\n')
                return response

            orchestrator = await self._get_orchestrator()

            # Stream response
            async for chunk in orchestrator.stream_chat(messages[-1].get("content", "")):
                if chunk.get("type") == "content":
                    event = {"type": "content", "content": chunk["content"]}
                elif chunk.get("type") == "tool_call":
                    event = {"type": "tool_call", "tool_call": chunk["tool_call"]}
                else:
                    event = chunk

                await response.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))

            await response.write(b"data: [DONE]\n\n")

        except Exception as e:
            logger.exception("Stream chat error")
            await response.write(
                f'data: {{"type": "error", "message": "{str(e)}"}}\n\n'.encode("utf-8")
            )

        return response

    async def _completions(self, request: Request) -> Response:
        """Get code completions."""
        try:
            data = await request.json()
            prompt = data.get("prompt", "")
            file_path = data.get("file", "")
            language = data.get("language", "")

            if not prompt:
                return web.json_response({"completions": []})

            # Use LLM for completions
            orchestrator = await self._get_orchestrator()

            # Include file context if available
            file_context = f" in file {file_path}" if file_path else ""
            completion_prompt = f"""Complete the following {language} code{file_context}. Only provide the completion, no explanation.

{prompt}"""

            response = await orchestrator.chat(completion_prompt)
            content = response.get("content", "")

            # Extract code from response
            completions = [content.strip()]

            return web.json_response({"completions": completions})

        except Exception as e:
            logger.exception("Completions error")
            return web.json_response({"completions": [], "error": str(e)})

    async def _semantic_search(self, request: Request) -> Response:
        """Semantic code search."""
        try:
            data = await request.json()
            query = data.get("query", "")
            max_results = data.get("max_results", 10)

            if not query:
                return web.json_response({"results": []})

            # Use orchestrator for semantic search if available
            orchestrator = await self._get_orchestrator()

            # Execute semantic code search tool
            tool_result = await orchestrator.execute_tool(
                "semantic_code_search",
                query=query,
                max_results=max_results,
            )

            if tool_result.success:
                results = tool_result.data.get("matches", [])
            else:
                results = []

            return web.json_response(
                {
                    "results": [
                        {
                            "file": r.get("file", ""),
                            "line": r.get("line", 0),
                            "content": r.get("content", ""),
                            "score": r.get("score", 0.0),
                        }
                        for r in results
                    ]
                }
            )

        except Exception as e:
            logger.exception("Semantic search error")
            return web.json_response({"results": [], "error": str(e)})

    async def _code_search(self, request: Request) -> Response:
        """Code search (regex/literal)."""
        try:
            data = await request.json()
            query = data.get("query", "")
            regex = data.get("regex", False)
            case_sensitive = data.get("case_sensitive", True)
            file_pattern = data.get("file_pattern", "*")

            if not query:
                return web.json_response({"results": []})

            # Use grep-like search
            import subprocess

            cmd = ["rg", "--json", "-n"]
            if not case_sensitive:
                cmd.append("-i")
            if not regex:
                cmd.append("-F")
            if file_pattern != "*":
                cmd.extend(["-g", file_pattern])
            cmd.append(query)
            cmd.append(self.workspace_root)

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            results = []
            for line in result.stdout.splitlines():
                try:
                    match = json.loads(line)
                    if match.get("type") == "match":
                        data = match.get("data", {})
                        results.append(
                            {
                                "file": data.get("path", {}).get("text", ""),
                                "line": data.get("line_number", 0),
                                "content": data.get("lines", {}).get("text", "").strip(),
                                "score": 1.0,
                            }
                        )
                except json.JSONDecodeError:
                    continue

            return web.json_response({"results": results[:50]})

        except Exception as e:
            logger.exception("Code search error")
            return web.json_response({"results": [], "error": str(e)})

    async def _switch_model(self, request: Request) -> Response:
        """Switch AI model."""
        try:
            data = await request.json()
            provider = data.get("provider")
            model = data.get("model")

            if not provider or not model:
                return web.json_response({"error": "provider and model required"}, status=400)

            from victor.agent.model_switcher import get_model_switcher

            switcher = get_model_switcher()
            switcher.switch(provider, model)

            return web.json_response(
                {
                    "success": True,
                    "provider": provider,
                    "model": model,
                }
            )

        except Exception as e:
            logger.exception("Switch model error")
            return web.json_response({"error": str(e)}, status=500)

    async def _switch_mode(self, request: Request) -> Response:
        """Switch agent mode."""
        try:
            data = await request.json()
            mode = data.get("mode")

            if not mode:
                return web.json_response({"error": "mode required"}, status=400)

            from victor.agent.mode_controller import AgentMode, get_mode_controller

            manager = get_mode_controller()
            manager.switch_mode(AgentMode(mode))

            return web.json_response(
                {
                    "success": True,
                    "mode": mode,
                }
            )

        except Exception as e:
            logger.exception("Switch mode error")
            return web.json_response({"error": str(e)}, status=500)

    async def _list_models(self, request: Request) -> Response:
        """List available models."""
        from victor.agent.model_switcher import get_model_switcher

        switcher = get_model_switcher()
        models = switcher.get_available_models()

        return web.json_response(
            {
                "models": [
                    {
                        "provider": m.provider,
                        "model_id": m.model_id,
                        "display_name": m.display_name,
                        "is_local": m.is_local,
                    }
                    for m in models
                ]
            }
        )

    async def _reset_conversation(self, request: Request) -> Response:
        """Reset conversation history."""
        try:
            if self._orchestrator:
                self._orchestrator.reset_conversation()
            return web.json_response({"success": True, "message": "Conversation reset"})
        except Exception as e:
            logger.exception("Reset conversation error")
            return web.json_response({"error": str(e)}, status=500)

    async def _export_conversation(self, request: Request) -> Response:
        """Export conversation history."""
        try:
            format_type = request.query.get("format", "json")

            if not self._orchestrator:
                return web.json_response({"messages": []})

            messages = self._orchestrator.get_messages()

            if format_type == "markdown":
                content = self._format_messages_markdown(messages)
                return Response(
                    text=content,
                    content_type="text/markdown",
                )
            else:
                return web.json_response({"messages": messages})

        except Exception as e:
            logger.exception("Export conversation error")
            return web.json_response({"error": str(e)}, status=500)

    def _format_messages_markdown(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages as markdown."""
        lines = ["# Conversation Export\n"]
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            lines.append(f"## {role}\n")
            lines.append(f"{content}\n")
        return "\n".join(lines)

    async def _undo(self, request: Request) -> Response:
        """Undo last change."""
        try:
            from victor.agent.change_tracker import get_change_tracker

            tracker = get_change_tracker()
            success, message, files = tracker.undo()

            return web.json_response(
                {
                    "success": success,
                    "message": message,
                    "files": files,
                }
            )

        except Exception as e:
            logger.exception("Undo error")
            return web.json_response({"error": str(e)}, status=500)

    async def _redo(self, request: Request) -> Response:
        """Redo last undone change."""
        try:
            from victor.agent.change_tracker import get_change_tracker

            tracker = get_change_tracker()
            success, message, files = tracker.redo()

            return web.json_response(
                {
                    "success": success,
                    "message": message,
                    "files": files,
                }
            )

        except Exception as e:
            logger.exception("Redo error")
            return web.json_response({"error": str(e)}, status=500)

    async def _history(self, request: Request) -> Response:
        """Get change history."""
        try:
            limit = int(request.query.get("limit", "10"))

            from victor.agent.change_tracker import get_change_tracker

            tracker = get_change_tracker()
            history = tracker.get_history(limit=limit)

            return web.json_response({"history": history})

        except Exception as e:
            logger.exception("History error")
            return web.json_response({"error": str(e)}, status=500)

    async def _apply_patch(self, request: Request) -> Response:
        """Apply a patch."""
        try:
            data = await request.json()
            patch_content = data.get("patch", "")
            dry_run = data.get("dry_run", False)

            if not patch_content:
                return web.json_response({"error": "patch required"}, status=400)

            from victor.tools import patch_tool

            result = await patch_tool.apply_patch(patch=patch_content, dry_run=dry_run)

            return web.json_response(result)

        except Exception as e:
            logger.exception("Apply patch error")
            return web.json_response({"error": str(e)}, status=500)

    async def _create_patch(self, request: Request) -> Response:
        """Create a patch."""
        try:
            data = await request.json()
            target_file = data.get("file_path", "")
            new_content = data.get("new_content", "")

            if not target_file or not new_content:
                return web.json_response(
                    {"error": "file_path and new_content required"}, status=400
                )

            from victor.tools import patch_tool

            result = await patch_tool.create_patch(file_path=target_file, new_content=new_content)

            return web.json_response(result)

        except Exception as e:
            logger.exception("Create patch error")
            return web.json_response({"error": str(e)}, status=500)

    async def _lsp_completions(self, request: Request) -> Response:
        """LSP completions."""
        try:
            data = await request.json()
            file_path = data.get("file", "")
            line = data.get("line", 0)
            character = data.get("character", 0)

            from victor.lsp.manager import get_lsp_manager

            manager = get_lsp_manager()
            completions = await manager.get_completions(file_path, line, character)

            return web.json_response(
                {
                    "completions": [
                        {
                            "label": c.label,
                            "kind": c.kind,
                            "detail": c.detail,
                            "insert_text": c.insert_text,
                        }
                        for c in completions
                    ]
                }
            )

        except Exception as e:
            logger.exception("LSP completions error")
            return web.json_response({"completions": [], "error": str(e)})

    async def _lsp_hover(self, request: Request) -> Response:
        """LSP hover."""
        try:
            data = await request.json()
            file_path = data.get("file", "")
            line = data.get("line", 0)
            character = data.get("character", 0)

            from victor.lsp.manager import get_lsp_manager

            manager = get_lsp_manager()
            hover = await manager.get_hover(file_path, line, character)

            return web.json_response(
                {
                    "contents": hover.contents if hover else None,
                }
            )

        except Exception as e:
            logger.exception("LSP hover error")
            return web.json_response({"contents": None, "error": str(e)})

    async def _lsp_definition(self, request: Request) -> Response:
        """LSP definition."""
        try:
            data = await request.json()
            file_path = data.get("file", "")
            line = data.get("line", 0)
            character = data.get("character", 0)

            from victor.lsp.manager import get_lsp_manager

            manager = get_lsp_manager()
            locations = await manager.get_definition(file_path, line, character)

            return web.json_response({"locations": locations})

        except Exception as e:
            logger.exception("LSP definition error")
            return web.json_response({"locations": [], "error": str(e)})

    async def _lsp_references(self, request: Request) -> Response:
        """LSP references."""
        try:
            data = await request.json()
            file_path = data.get("file", "")
            line = data.get("line", 0)
            character = data.get("character", 0)

            from victor.lsp.manager import get_lsp_manager

            manager = get_lsp_manager()
            locations = await manager.get_references(file_path, line, character)

            return web.json_response({"locations": locations})

        except Exception as e:
            logger.exception("LSP references error")
            return web.json_response({"locations": [], "error": str(e)})

    async def _lsp_diagnostics(self, request: Request) -> Response:
        """LSP diagnostics."""
        try:
            data = await request.json()
            file_path = data.get("file", "")

            from victor.lsp.manager import get_lsp_manager

            manager = get_lsp_manager()
            diagnostics = manager.get_diagnostics(file_path)

            return web.json_response({"diagnostics": diagnostics})

        except Exception as e:
            logger.exception("LSP diagnostics error")
            return web.json_response({"diagnostics": [], "error": str(e)})

    async def _websocket_handler(self, request: Request) -> web.WebSocketResponse:
        """Handle WebSocket connections for real-time updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self._ws_clients.append(ws)
        logger.info(f"WebSocket client connected. Total: {len(self._ws_clients)}")

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_ws_message(ws, data)
                    except json.JSONDecodeError:
                        await ws.send_json({"error": "Invalid JSON"})
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
        finally:
            self._ws_clients.remove(ws)
            logger.info(f"WebSocket client disconnected. Total: {len(self._ws_clients)}")

        return ws

    async def _handle_ws_message(self, ws: web.WebSocketResponse, data: Dict[str, Any]) -> None:
        """Handle incoming WebSocket messages."""
        msg_type = data.get("type", "")

        if msg_type == "chat":
            # Handle streaming chat over WebSocket
            messages = data.get("messages", [])
            if not messages:
                await ws.send_json({"type": "error", "message": "No messages"})
                return

            orchestrator = await self._get_orchestrator()

            try:
                async for chunk in orchestrator.stream_chat(messages[-1].get("content", "")):
                    if chunk.get("type") == "content":
                        await ws.send_json({"type": "content", "content": chunk["content"]})
                    elif chunk.get("type") == "tool_call":
                        await ws.send_json({"type": "tool_call", "tool_call": chunk["tool_call"]})

                await ws.send_json({"type": "done"})
            except Exception as e:
                await ws.send_json({"type": "error", "message": str(e)})

        elif msg_type == "ping":
            await ws.send_json({"type": "pong"})

        elif msg_type == "subscribe":
            # Subscribe to events (e.g., file changes, tool results)
            channel = data.get("channel", "")
            await ws.send_json({"type": "subscribed", "channel": channel})

    async def _broadcast_ws(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all connected WebSocket clients."""
        for ws in self._ws_clients:
            try:
                await ws.send_json(message)
            except Exception:
                pass  # Ignore send errors

    async def _shutdown(self, request: Request) -> Response:
        """Shutdown the server."""
        logger.info("Shutdown requested")

        # Close all WebSocket connections
        for ws in self._ws_clients:
            await ws.close()

        asyncio.get_event_loop().call_later(0.5, asyncio.get_event_loop().stop)
        return web.json_response({"status": "shutting_down"})

    async def _get_orchestrator(self) -> Any:
        """Get or create the orchestrator."""
        if self._orchestrator is None:
            from victor.agent.orchestrator import AgentOrchestrator
            from victor.config.settings import load_settings

            settings = load_settings()
            # Create orchestrator with settings
            self._orchestrator = await AgentOrchestrator.from_settings(settings)

        return self._orchestrator

    def run(self) -> None:
        """Run the server."""
        logger.info(f"Starting Victor API server on {self.host}:{self.port}")
        web.run_app(self._app, host=self.host, port=self.port)

    async def start_async(self) -> web.AppRunner:
        """Start the server asynchronously."""
        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(f"Victor API server running on {self.host}:{self.port}")
        return runner


def create_app(workspace_root: Optional[str] = None) -> web.Application:
    """Create the API application."""
    server = VictorAPIServer(workspace_root=workspace_root)
    return server._app
