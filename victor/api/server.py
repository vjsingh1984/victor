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

Features:
- REST API for chat, completions, code search
- WebSocket support for real-time updates
- Optional rate limiting and API key authentication
- CORS support for browser-based clients
"""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from aiohttp import web
from aiohttp.web import Request, Response, StreamResponse

# Import middleware stack for optional rate limiting and auth
from victor.api.middleware import APIMiddlewareStack

logger = logging.getLogger(__name__)


class VictorAPIServer:
    """HTTP API Server for Victor."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        workspace_root: Optional[str] = None,
        rate_limit_rpm: Optional[int] = None,
        api_keys: Optional[Dict[str, str]] = None,
        enable_cors: bool = True,
    ):
        """Initialize the API server.

        Args:
            host: Host to bind to
            port: Port to listen on
            workspace_root: Root directory of the workspace
            rate_limit_rpm: Optional requests per minute limit (None = no limit)
            api_keys: Optional dict of {api_key: client_id} for authentication
            enable_cors: Enable CORS headers (default: True)
        """
        self.host = host
        self.port = port
        self.workspace_root = workspace_root or str(Path.cwd())

        # Build middleware stack based on configuration
        middleware_stack = APIMiddlewareStack()
        if enable_cors:
            middleware_stack.add_cors()
        if rate_limit_rpm is not None and rate_limit_rpm > 0:
            middleware_stack.add_rate_limiting(
                requests_per_minute=rate_limit_rpm,
                burst_size=rate_limit_rpm // 4 or 5,  # Allow burst of 25%
            )
            logger.info(f"Rate limiting enabled: {rate_limit_rpm} requests/minute")
        if api_keys:
            middleware_stack.add_authentication(api_keys=api_keys)
            logger.info(f"API key authentication enabled for {len(api_keys)} client(s)")

        self._app = web.Application(middlewares=middleware_stack.build())
        self._orchestrator = None
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Set up API routes."""
        self._app.router.add_get("/health", self._health)
        self._app.router.add_get("/status", self._status)
        self._app.router.add_get("/credentials/get", self._credentials_placeholder)

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
        self._app.router.add_get("/providers", self._list_providers)
        self._app.router.add_get("/tools", self._list_tools)

        # Conversation management
        self._app.router.add_post("/conversation/reset", self._reset_conversation)
        self._app.router.add_get("/conversation/export", self._export_conversation)

        # Undo/Redo
        self._app.router.add_post("/undo", self._undo)
        self._app.router.add_post("/redo", self._redo)
        self._app.router.add_get("/history", self._history)
        self._app.router.add_get("/credentials/get", self._credentials_placeholder)
        self._app.router.add_post("/session/token", self._session_token_placeholder)

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

        # Workspace analysis endpoints
        self._app.router.add_get("/workspace/overview", self._workspace_overview)
        self._app.router.add_get("/workspace/metrics", self._workspace_metrics)
        self._app.router.add_get("/workspace/security", self._workspace_security)
        self._app.router.add_get("/workspace/dependencies", self._workspace_dependencies)

        # Tool approval endpoints
        self._app.router.add_post("/tools/approve", self._approve_tool)
        self._app.router.add_get("/tools/pending", self._pending_approvals)

        # Git integration endpoints
        self._app.router.add_get("/git/status", self._git_status)
        self._app.router.add_post("/git/commit", self._git_commit)
        self._app.router.add_get("/git/log", self._git_log)
        self._app.router.add_get("/git/diff", self._git_diff)

        # MCP endpoints
        self._app.router.add_get("/mcp/servers", self._mcp_servers)
        self._app.router.add_post("/mcp/connect", self._mcp_connect)
        self._app.router.add_post("/mcp/disconnect", self._mcp_disconnect)

        # RL Model Selector endpoints
        self._app.router.add_get("/rl/stats", self._rl_stats)
        self._app.router.add_get("/rl/recommend", self._rl_recommend)
        self._app.router.add_post("/rl/explore", self._rl_explore)
        self._app.router.add_post("/rl/strategy", self._rl_strategy)
        self._app.router.add_post("/rl/reset", self._rl_reset)

        # WebSocket
        self._app.router.add_get("/ws", self._websocket_handler)

        # Note: CORS/rate limiting/auth middleware is configured in __init__ via APIMiddlewareStack

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

            # CompletionResponse is a Pydantic model; access attributes
            content = getattr(response, "content", None) or ""
            tool_calls = getattr(response, "tool_calls", None) or []
            return web.json_response(
                {"role": "assistant", "content": content, "tool_calls": tool_calls}
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
                # Support both dict and StreamChunk objects
                if hasattr(chunk, "content") or hasattr(chunk, "tool_calls"):
                    content = getattr(chunk, "content", "")
                    tool_calls = getattr(chunk, "tool_calls", None)
                    event_type = "content" if content else "tool_call" if tool_calls else "chunk"
                    if event_type == "content":
                        event = {"type": "content", "content": content}
                    elif event_type == "tool_call":
                        event = {"type": "tool_call", "tool_call": tool_calls}
                    else:
                        event = {}
                else:
                    if chunk.get("type") == "content":
                        event = {"type": "content", "content": chunk.get("content", "")}
                    elif chunk.get("type") == "tool_call":
                        event = {"type": "tool_call", "tool_call": chunk.get("tool_call", {})}
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
            content = getattr(response, "content", None) or ""

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

    async def _list_providers(self, request: Request) -> Response:
        """List available LLM providers with their configuration status."""
        try:
            from victor.providers.registry import get_provider_registry

            registry = get_provider_registry()
            providers_info = []

            for provider_name in registry.list_providers():
                try:
                    provider = registry.get(provider_name)
                    providers_info.append(
                        {
                            "name": provider_name,
                            "display_name": provider_name.replace("_", " ").title(),
                            "is_local": provider_name in ("ollama", "lmstudio", "vllm"),
                            "configured": (
                                provider.is_configured()
                                if hasattr(provider, "is_configured")
                                else True
                            ),
                            "supports_tools": (
                                provider.supports_tools()
                                if hasattr(provider, "supports_tools")
                                else False
                            ),
                            "supports_streaming": (
                                provider.supports_streaming()
                                if hasattr(provider, "supports_streaming")
                                else True
                            ),
                        }
                    )
                except Exception as e:
                    logger.debug(f"Provider {provider_name} not available: {e}")
                    providers_info.append(
                        {
                            "name": provider_name,
                            "display_name": provider_name.replace("_", " ").title(),
                            "is_local": provider_name in ("ollama", "lmstudio", "vllm"),
                            "configured": False,
                            "supports_tools": False,
                            "supports_streaming": True,
                        }
                    )

            return web.json_response({"providers": providers_info})

        except Exception as e:
            logger.exception("List providers error")
            return web.json_response({"providers": [], "error": str(e)})

    async def _credentials_placeholder(self, request: Request) -> Response:
        """Placeholder credentials endpoint to satisfy IDE clients."""
        provider = request.query.get("provider", "")
        return web.json_response({"provider": provider, "api_key": None})

    async def _session_token_placeholder(self, request: Request) -> Response:
        """Placeholder session token endpoint for clients expecting /session/token."""
        session_id = str(uuid.uuid4())
        session_token = str(uuid.uuid4())
        return web.json_response({"session_token": session_token, "session_id": session_id})

    async def _list_tools(self, request: Request) -> Response:
        """List available tools with their metadata."""
        try:
            from victor.tools.base import CostTier, ToolRegistry

            # ToolRegistry is not a singleton; instantiate to list registered tools
            registry = ToolRegistry()
            tools_info = []

            for tool in registry.list_tools():
                # Get tool metadata
                cost_tier = (
                    tool.cost_tier.value
                    if hasattr(tool, "cost_tier") and tool.cost_tier
                    else "free"
                )

                # Determine category from tool name or module
                category = self._get_tool_category(tool.name)

                tools_info.append(
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "category": category,
                        "cost_tier": cost_tier,
                        "parameters": tool.parameters if hasattr(tool, "parameters") else {},
                        "is_dangerous": self._is_dangerous_tool(tool.name),
                        "requires_approval": cost_tier in ("medium", "high")
                        or self._is_dangerous_tool(tool.name),
                    }
                )

            # Sort by category then name
            tools_info.sort(key=lambda t: (t["category"], t["name"]))

            return web.json_response(
                {
                    "tools": tools_info,
                    "total": len(tools_info),
                    "categories": list({t["category"] for t in tools_info}),
                }
            )

        except Exception as e:
            logger.exception("List tools error")
            return web.json_response({"tools": [], "total": 0, "error": str(e)})

    def _get_tool_category(self, tool_name: str) -> str:
        """Get category for a tool based on its name."""
        categories = {
            "filesystem": ["read", "write", "ls", "edit", "glob", "overview"],
            "search": ["search", "grep", "semantic_code_search", "code_search"],
            "git": ["git", "commit_msg", "conflicts", "pr", "merge"],
            "shell": ["shell", "bash", "sandbox"],
            "refactor": ["extract", "inline", "organize_imports", "rename", "refactor"],
            "code_intelligence": ["symbol", "refs", "lsp"],
            "web": ["fetch", "http", "web", "web_search", "web_fetch"],
            "docker": ["docker"],
            "database": ["database", "db"],
            "testing": ["test", "pytest"],
            "documentation": ["docs", "documentation", "docs_coverage"],
            "analysis": ["code_review", "metrics", "scan", "audit", "iac"],
            "infrastructure": ["cicd", "pipeline", "dependency"],
            "batch": ["batch"],
            "cache": ["cache"],
            "mcp": ["mcp"],
            "workflow": ["workflow", "scaffold"],
            "patch": ["patch"],
        }

        for category, keywords in categories.items():
            if any(kw in tool_name.lower() for kw in keywords):
                return category.replace("_", " ").title()

        return "Other"

    def _is_dangerous_tool(self, tool_name: str) -> bool:
        """Check if a tool is considered dangerous."""
        dangerous_tools = {
            "shell",
            "bash",
            "sandbox",
            "docker",
            "write",
            "edit",
            "delete",
            "rm",
            "database",
        }
        return tool_name.lower() in dangerous_tools

    async def _reset_conversation(self, request: Request) -> Response:
        """Reset conversation history."""
        try:
            # Record RL feedback before resetting (marks end of session)
            await self._record_rl_feedback()

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

        # Record RL feedback before shutdown
        await self._record_rl_feedback()

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

    async def _record_rl_feedback(self) -> None:
        """Record RL feedback for the current session.

        Called when a session ends (conversation reset, server shutdown, etc.)
        to update Q-values based on session performance.
        """
        if self._orchestrator is None:
            return

        try:
            from victor.agent.rl.base import RLOutcome
            from victor.agent.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("model_selector")
            if learner is None:
                return

            provider = self._orchestrator.provider
            if provider is None:
                return

            # Check if there were actual interactions
            msg_count = 0
            if hasattr(self._orchestrator, "message_count"):
                msg_count = self._orchestrator.message_count
            elif hasattr(self._orchestrator, "get_messages"):
                messages = self._orchestrator.get_messages()
                msg_count = len(messages) if messages else 0

            if msg_count == 0:
                return  # No interactions, skip

            # Gather session metrics
            metrics = {}
            if hasattr(self._orchestrator, "get_session_metrics"):
                metrics = self._orchestrator.get_session_metrics() or {}

            import uuid

            # Compute quality score based on session metrics
            latency = metrics.get("total_latency", 0)
            token_count = metrics.get("total_tokens", 0)
            tool_calls = metrics.get("tool_calls", 0)

            # Base quality: 1.0 for successful session
            quality_score = 1.0

            # Latency penalty: -0.1 per 30s over 30s threshold (max -0.5)
            if latency > 30:
                quality_score -= min(0.1 * (latency - 30) / 30, 0.5)

            # Tool usage bonus: +0.05 per tool (max +0.2)
            if tool_calls > 0:
                quality_score += min(0.05 * tool_calls, 0.2)

            outcome = RLOutcome(
                provider=provider.name,
                model=getattr(provider, "model", "unknown"),
                task_type="chat",  # API sessions are general chat
                success=True,  # Session completed normally
                quality_score=max(0.0, min(1.0, quality_score)),
                metadata={
                    "session_id": str(uuid.uuid4())[:8],
                    "latency_seconds": latency,
                    "token_count": token_count,
                    "tool_calls_made": tool_calls,
                    "message_count": msg_count,
                },
                vertical="coding",
            )

            coordinator.record_outcome("model_selector", outcome, "coding")

            # Get updated Q-value for logging
            rankings = learner.get_provider_rankings()
            provider_ranking = next(
                (r for r in rankings if r["provider"] == provider.name), None
            )
            new_q = provider_ranking["q_value"] if provider_ranking else 0.0

            logger.info(
                f"RL API session feedback: {provider.name} "
                f"({msg_count} messages, {tool_calls} tools) → Q={new_q:.3f}"
            )

        except Exception as e:
            logger.debug(f"RL feedback recording skipped: {e}")

    # =========================================================================
    # Workspace Analysis Endpoints
    # =========================================================================

    async def _workspace_overview(self, request: Request) -> Response:
        """Get workspace structure overview."""
        try:
            import os
            from pathlib import Path

            root = Path(self.workspace_root)
            overview = {
                "root": str(root),
                "name": root.name,
                "directories": [],
                "files": [],
                "file_counts": {},
                "total_files": 0,
                "total_size": 0,
            }

            # Walk directory tree (limit depth)
            max_depth = int(request.query.get("depth", "3"))
            exclude_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", ".victor"}

            def scan_dir(path: Path, depth: int = 0) -> Dict[str, Any]:
                if depth > max_depth:
                    return {"name": path.name, "type": "directory", "truncated": True}

                result: Dict[str, Any] = {
                    "name": path.name,
                    "path": str(path.relative_to(root)),
                    "type": "directory",
                    "children": [],
                }

                try:
                    for entry in sorted(
                        path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())
                    ):
                        if entry.name.startswith(".") and entry.name not in {".github", ".vscode"}:
                            continue
                        if entry.name in exclude_dirs:
                            continue

                        if entry.is_dir():
                            result["children"].append(scan_dir(entry, depth + 1))
                        else:
                            ext = entry.suffix.lower()
                            overview["file_counts"][ext] = overview["file_counts"].get(ext, 0) + 1
                            overview["total_files"] += 1
                            try:
                                overview["total_size"] += entry.stat().st_size
                            except OSError:
                                pass

                            if depth <= 1:  # Only include files at shallow depth
                                result["children"].append(
                                    {
                                        "name": entry.name,
                                        "path": str(entry.relative_to(root)),
                                        "type": "file",
                                        "extension": ext,
                                    }
                                )
                except PermissionError:
                    result["error"] = "Permission denied"

                return result

            overview["tree"] = scan_dir(root)

            return web.json_response(overview)

        except Exception as e:
            logger.exception("Workspace overview error")
            return web.json_response({"error": str(e)}, status=500)

    async def _workspace_metrics(self, request: Request) -> Response:
        """Get code metrics for the workspace."""
        try:
            orchestrator = await self._get_orchestrator()

            # Execute metrics tool if available
            try:
                tool_result = await orchestrator.execute_tool("metrics", path=self.workspace_root)
                if tool_result.success:
                    return web.json_response(tool_result.data)
            except Exception:
                pass  # Fall back to basic metrics

            # Basic metrics fallback
            import os
            from pathlib import Path

            root = Path(self.workspace_root)
            metrics = {
                "lines_of_code": 0,
                "files_by_type": {},
                "largest_files": [],
                "recent_files": [],
            }

            code_extensions = {
                ".py",
                ".ts",
                ".js",
                ".tsx",
                ".jsx",
                ".java",
                ".go",
                ".rs",
                ".cpp",
                ".c",
                ".h",
            }
            file_sizes = []

            for path in root.rglob("*"):
                if path.is_file() and not any(
                    p.startswith(".") for p in path.parts[len(root.parts) :]
                ):
                    ext = path.suffix.lower()
                    if ext in code_extensions:
                        try:
                            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                                lines = len(f.readlines())
                                metrics["lines_of_code"] += lines
                                metrics["files_by_type"][ext] = (
                                    metrics["files_by_type"].get(ext, 0) + 1
                                )
                                file_sizes.append(
                                    {
                                        "path": str(path.relative_to(root)),
                                        "lines": lines,
                                        "size": path.stat().st_size,
                                    }
                                )
                        except Exception:
                            pass

            # Get largest files
            file_sizes.sort(key=lambda x: x["lines"], reverse=True)
            metrics["largest_files"] = file_sizes[:10]

            return web.json_response(metrics)

        except Exception as e:
            logger.exception("Workspace metrics error")
            return web.json_response({"error": str(e)}, status=500)

    async def _workspace_security(self, request: Request) -> Response:
        """Get security scan results for the workspace."""
        try:
            orchestrator = await self._get_orchestrator()

            # Try to execute security scan tool
            try:
                tool_result = await orchestrator.execute_tool(
                    "scan",
                    path=self.workspace_root,
                    scan_type="secrets",
                )
                if tool_result.success:
                    return web.json_response(
                        {
                            "scan_completed": True,
                            "results": tool_result.data,
                        }
                    )
            except Exception:
                pass

            # Fallback basic secret detection
            import re
            from pathlib import Path

            root = Path(self.workspace_root)
            findings = []

            secret_patterns = [
                (r'(?i)(api[_-]?key|apikey)\s*[:=]\s*["\']?[\w-]{20,}', "API Key"),
                (r'(?i)(secret|password|passwd|pwd)\s*[:=]\s*["\'][^"\']{8,}', "Secret/Password"),
                (r"(?i)bearer\s+[\w-]{20,}", "Bearer Token"),
                (r"sk-[a-zA-Z0-9]{20,}", "OpenAI API Key"),
                (r"ghp_[a-zA-Z0-9]{36}", "GitHub Token"),
                (r"AKIA[A-Z0-9]{16}", "AWS Access Key"),
            ]

            code_extensions = {".py", ".ts", ".js", ".json", ".yaml", ".yml", ".env", ".sh"}

            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in code_extensions:
                    if any(
                        p.startswith(".") or p in {"node_modules", "__pycache__"}
                        for p in path.parts
                    ):
                        continue

                    try:
                        content = path.read_text(encoding="utf-8", errors="ignore")
                        for pattern, finding_type in secret_patterns:
                            for match in re.finditer(pattern, content):
                                line_num = content[: match.start()].count("\n") + 1
                                findings.append(
                                    {
                                        "file": str(path.relative_to(root)),
                                        "line": line_num,
                                        "type": finding_type,
                                        "severity": "high",
                                        "snippet": match.group()[:30] + "...",
                                    }
                                )
                    except Exception:
                        pass

            return web.json_response(
                {
                    "scan_completed": True,
                    "findings": findings[:50],  # Limit to 50 findings
                    "total_findings": len(findings),
                    "severity_counts": {
                        "high": len([f for f in findings if f["severity"] == "high"]),
                        "medium": 0,
                        "low": 0,
                    },
                }
            )

        except Exception as e:
            logger.exception("Workspace security error")
            return web.json_response({"error": str(e)}, status=500)

    async def _workspace_dependencies(self, request: Request) -> Response:
        """Get dependency information for the workspace."""
        try:
            from pathlib import Path
            import json

            root = Path(self.workspace_root)
            dependencies = {
                "python": None,
                "node": None,
                "rust": None,
                "go": None,
            }

            # Python dependencies
            for req_file in ["requirements.txt", "pyproject.toml", "setup.py"]:
                req_path = root / req_file
                if req_path.exists():
                    if req_file == "requirements.txt":
                        deps = []
                        for line in req_path.read_text().splitlines():
                            line = line.strip()
                            if line and not line.startswith("#"):
                                deps.append(line.split("==")[0].split(">=")[0].split("<")[0])
                        dependencies["python"] = {
                            "file": req_file,
                            "count": len(deps),
                            "packages": deps[:20],
                        }
                    break

            # Node dependencies
            pkg_json = root / "package.json"
            if pkg_json.exists():
                try:
                    pkg_data = json.loads(pkg_json.read_text())
                    deps = list(pkg_data.get("dependencies", {}).keys())
                    dev_deps = list(pkg_data.get("devDependencies", {}).keys())
                    dependencies["node"] = {
                        "file": "package.json",
                        "dependencies": len(deps),
                        "devDependencies": len(dev_deps),
                        "packages": deps[:20],
                    }
                except json.JSONDecodeError:
                    pass

            # Rust dependencies
            cargo_toml = root / "Cargo.toml"
            if cargo_toml.exists():
                dependencies["rust"] = {
                    "file": "Cargo.toml",
                    "exists": True,
                }

            # Go dependencies
            go_mod = root / "go.mod"
            if go_mod.exists():
                dependencies["go"] = {
                    "file": "go.mod",
                    "exists": True,
                }

            return web.json_response(
                {
                    "workspace": str(root),
                    "dependencies": {k: v for k, v in dependencies.items() if v is not None},
                }
            )

        except Exception as e:
            logger.exception("Workspace dependencies error")
            return web.json_response({"error": str(e)}, status=500)

    # =========================================================================
    # Tool Approval Endpoints
    # =========================================================================

    # Pending tool approvals (in-memory for now, could be persisted)
    _pending_tool_approvals: Dict[str, Dict[str, Any]] = {}

    async def _approve_tool(self, request: Request) -> Response:
        """Approve or reject a pending tool execution."""
        try:
            data = await request.json()
            approval_id = data.get("approval_id")
            approved = data.get("approved", False)

            if not approval_id:
                return web.json_response({"error": "approval_id required"}, status=400)

            if approval_id in self._pending_tool_approvals:
                approval = self._pending_tool_approvals.pop(approval_id)
                approval["approved"] = approved
                approval["resolved"] = True

                # Broadcast approval status to WebSocket clients
                await self._broadcast_ws(
                    {
                        "type": "tool_approval_resolved",
                        "approval_id": approval_id,
                        "approved": approved,
                        "tool_name": approval.get("tool_name"),
                    }
                )

                return web.json_response(
                    {
                        "success": True,
                        "approval_id": approval_id,
                        "approved": approved,
                    }
                )
            else:
                return web.json_response({"error": "Approval not found"}, status=404)

        except Exception as e:
            logger.exception("Tool approval error")
            return web.json_response({"error": str(e)}, status=500)

    async def _pending_approvals(self, request: Request) -> Response:
        """Get list of pending tool approvals."""
        try:
            pending = [
                {
                    "approval_id": aid,
                    "tool_name": info.get("tool_name"),
                    "arguments": info.get("arguments"),
                    "danger_level": info.get("danger_level"),
                    "cost_tier": info.get("cost_tier"),
                    "created_at": info.get("created_at"),
                }
                for aid, info in self._pending_tool_approvals.items()
                if not info.get("resolved")
            ]

            return web.json_response(
                {
                    "pending": pending,
                    "count": len(pending),
                }
            )

        except Exception as e:
            logger.exception("Pending approvals error")
            return web.json_response({"error": str(e)}, status=500)

    # =========================================================================
    # Git Integration Endpoints
    # =========================================================================

    async def _git_status(self, request: Request) -> Response:
        """Get git status for the workspace."""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "status", "--porcelain", "-b"],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return web.json_response(
                    {
                        "is_git_repo": False,
                        "error": result.stderr,
                    }
                )

            lines = result.stdout.strip().split("\n")
            branch_line = lines[0] if lines else ""

            # Parse branch info
            branch = "unknown"
            tracking = None
            if branch_line.startswith("## "):
                branch_info = branch_line[3:]
                if "..." in branch_info:
                    parts = branch_info.split("...")
                    branch = parts[0]
                    tracking = parts[1].split()[0] if len(parts) > 1 else None
                else:
                    branch = branch_info.split()[0]

            # Parse file statuses
            staged = []
            unstaged = []
            untracked = []

            for line in lines[1:]:
                if not line.strip():
                    continue
                status = line[:2]
                filepath = line[3:]

                if status[0] in "MADRC":
                    staged.append({"status": status[0], "file": filepath})
                if status[1] in "MD":
                    unstaged.append({"status": status[1], "file": filepath})
                if status == "??":
                    untracked.append(filepath)

            return web.json_response(
                {
                    "is_git_repo": True,
                    "branch": branch,
                    "tracking": tracking,
                    "staged": staged,
                    "unstaged": unstaged,
                    "untracked": untracked,
                    "is_clean": len(staged) == 0 and len(unstaged) == 0 and len(untracked) == 0,
                }
            )

        except subprocess.TimeoutExpired:
            return web.json_response({"error": "Git command timed out"}, status=500)
        except FileNotFoundError:
            return web.json_response({"is_git_repo": False, "error": "Git not installed"})
        except Exception as e:
            logger.exception("Git status error")
            return web.json_response({"error": str(e)}, status=500)

    async def _git_commit(self, request: Request) -> Response:
        """Create a git commit with AI-generated message option."""
        try:
            import subprocess

            data = await request.json()
            message = data.get("message")
            use_ai = data.get("use_ai", False)
            files = data.get("files")  # Optional list of files to stage

            # Stage files if specified
            if files:
                for f in files:
                    subprocess.run(
                        ["git", "add", f],
                        cwd=self.workspace_root,
                        capture_output=True,
                        timeout=10,
                    )

            # Generate AI commit message if requested
            if use_ai and not message:
                # Get diff for staged changes
                diff_result = subprocess.run(
                    ["git", "diff", "--cached", "--stat"],
                    cwd=self.workspace_root,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if diff_result.stdout.strip():
                    orchestrator = await self._get_orchestrator()
                    prompt = f"Generate a concise git commit message for these changes:\n{diff_result.stdout[:2000]}"
                    response = await orchestrator.chat(prompt)
                    message = response.get("content", "Update files").strip()
                    # Clean up the message
                    message = message.replace("```", "").strip()
                    if message.startswith('"') and message.endswith('"'):
                        message = message[1:-1]

            if not message:
                return web.json_response({"error": "Commit message required"}, status=400)

            # Perform commit
            result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return web.json_response(
                    {
                        "success": False,
                        "error": result.stderr or "Commit failed",
                    }
                )

            return web.json_response(
                {
                    "success": True,
                    "message": message,
                    "output": result.stdout,
                }
            )

        except Exception as e:
            logger.exception("Git commit error")
            return web.json_response({"error": str(e)}, status=500)

    async def _git_log(self, request: Request) -> Response:
        """Get git commit log."""
        try:
            import subprocess

            limit = int(request.query.get("limit", "20"))

            result = subprocess.run(
                ["git", "log", f"-{limit}", "--pretty=format:%H|%an|%ae|%ar|%s"],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return web.json_response({"error": result.stderr}, status=500)

            commits = []
            for line in result.stdout.strip().split("\n"):
                if "|" in line:
                    parts = line.split("|", 4)
                    if len(parts) >= 5:
                        commits.append(
                            {
                                "hash": parts[0],
                                "author": parts[1],
                                "email": parts[2],
                                "relative_date": parts[3],
                                "message": parts[4],
                            }
                        )

            return web.json_response({"commits": commits})

        except Exception as e:
            logger.exception("Git log error")
            return web.json_response({"error": str(e)}, status=500)

    async def _git_diff(self, request: Request) -> Response:
        """Get git diff."""
        try:
            import subprocess

            staged = request.query.get("staged", "false").lower() == "true"
            file_path = request.query.get("file")

            cmd = ["git", "diff"]
            if staged:
                cmd.append("--cached")
            if file_path:
                cmd.append("--")
                cmd.append(file_path)

            result = subprocess.run(
                cmd,
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=30,
            )

            return web.json_response(
                {
                    "diff": result.stdout[:50000],  # Limit size
                    "truncated": len(result.stdout) > 50000,
                }
            )

        except Exception as e:
            logger.exception("Git diff error")
            return web.json_response({"error": str(e)}, status=500)

    # =========================================================================
    # MCP Integration Endpoints
    # =========================================================================

    async def _mcp_servers(self, request: Request) -> Response:
        """Get list of configured MCP servers."""
        try:
            from victor.mcp.registry import get_mcp_registry

            registry = get_mcp_registry()
            servers = []

            for name in registry.list_servers():
                server_info = registry.get_server_info(name)
                servers.append(
                    {
                        "name": name,
                        "connected": server_info.get("connected", False),
                        "tools": server_info.get("tools", []),
                        "endpoint": server_info.get("endpoint"),
                    }
                )

            return web.json_response({"servers": servers})

        except ImportError:
            return web.json_response({"servers": [], "error": "MCP not available"})
        except Exception as e:
            logger.exception("MCP servers error")
            return web.json_response({"error": str(e)}, status=500)

    async def _mcp_connect(self, request: Request) -> Response:
        """Connect to an MCP server."""
        try:
            from victor.mcp.registry import get_mcp_registry

            data = await request.json()
            server_name = data.get("server")
            endpoint = data.get("endpoint")

            if not server_name:
                return web.json_response({"error": "server name required"}, status=400)

            registry = get_mcp_registry()
            success = await registry.connect(server_name, endpoint=endpoint)

            return web.json_response(
                {
                    "success": success,
                    "server": server_name,
                }
            )

        except ImportError:
            return web.json_response({"error": "MCP not available"}, status=501)
        except Exception as e:
            logger.exception("MCP connect error")
            return web.json_response({"error": str(e)}, status=500)

    async def _mcp_disconnect(self, request: Request) -> Response:
        """Disconnect from an MCP server."""
        try:
            from victor.mcp.registry import get_mcp_registry

            data = await request.json()
            server_name = data.get("server")

            if not server_name:
                return web.json_response({"error": "server name required"}, status=400)

            registry = get_mcp_registry()
            await registry.disconnect(server_name)

            return web.json_response(
                {
                    "success": True,
                    "server": server_name,
                }
            )

        except ImportError:
            return web.json_response({"error": "MCP not available"}, status=501)
        except Exception as e:
            logger.exception("MCP disconnect error")
            return web.json_response({"error": str(e)}, status=500)

    # =========================================================================
    # RL Model Selector Endpoints
    # =========================================================================

    async def _rl_stats(self, request: Request) -> Response:
        """Get RL model selector statistics."""
        try:
            from victor.agent.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("model_selector")
            if learner is None:
                return web.json_response(
                    {"error": "Model selector learner not available"}, status=503
                )

            # Get provider rankings
            rankings = learner.get_provider_rankings()

            # Build task-specific Q-table summary from database
            import sqlite3
            task_q_summary = {}
            conn = sqlite3.connect(str(coordinator.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT provider, task_type, q_value FROM model_selector_task_q_values")
            for row in cursor.fetchall():
                provider, task_type, q_value = row
                if provider not in task_q_summary:
                    task_q_summary[provider] = {}
                task_q_summary[provider][task_type] = round(q_value, 3)
            conn.close()

            stats = {
                "strategy": learner.strategy.value,
                "epsilon": round(learner.epsilon, 3),
                "total_selections": learner._total_selections,
                "provider_rankings": [
                    {
                        "provider": r["provider"],
                        "q_value": round(r["q_value"], 3),
                        "selection_count": r["selection_count"],
                    }
                    for r in rankings
                ],
                "task_q_tables": task_q_summary,
                "q_table_path": str(coordinator.db_path),
            }

            return web.json_response(stats)

        except Exception as e:
            logger.exception("RL stats error")
            return web.json_response({"error": str(e)}, status=500)

    async def _rl_recommend(self, request: Request) -> Response:
        """Get model recommendation based on Q-values.

        Query params:
            task_type: Optional task type for context-aware recommendation
                       (simple, complex, action, generation, analysis)
        """
        try:
            from victor.agent.rl.coordinator import get_rl_coordinator
            import json

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("model_selector")
            if learner is None:
                return web.json_response(
                    {"error": "Model selector learner not available"}, status=503
                )

            task_type = request.query.get("task_type")

            # Get available providers from learner's Q-table
            available = list(learner._q_table.keys()) if learner._q_table else ["ollama"]

            # Get recommendation from coordinator
            recommendation = coordinator.get_recommendation(
                "model_selector",
                json.dumps(available),  # Pass as JSON string
                "",  # model param not used
                task_type or "chat",
            )

            if recommendation is None:
                return web.json_response(
                    {"error": "No recommendation available"}, status=500
                )

            # Get rankings for alternatives
            rankings = learner.get_provider_rankings()
            alternatives = [
                {"provider": r["provider"], "q_value": round(r["q_value"], 3)}
                for r in rankings
                if r["provider"] != recommendation.value
            ][:5]  # Top 5 alternatives

            return web.json_response(
                {
                    "provider": recommendation.value,
                    "model": recommendation.model or "default",
                    "q_value": round(recommendation.confidence, 3),
                    "confidence": round(recommendation.confidence, 3),
                    "reason": recommendation.reasoning,
                    "task_type": task_type,
                    "alternatives": alternatives,
                }
            )

        except Exception as e:
            logger.exception("RL recommend error")
            return web.json_response({"error": str(e)}, status=500)

    async def _rl_explore(self, request: Request) -> Response:
        """Set exploration rate for RL model selector."""
        try:
            from victor.agent.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("model_selector")
            if learner is None:
                return web.json_response(
                    {"error": "Model selector learner not available"}, status=503
                )

            data = await request.json()
            rate = data.get("rate")

            if rate is None:
                return web.json_response({"error": "rate required"}, status=400)

            try:
                rate = float(rate)
                if not 0.0 <= rate <= 1.0:
                    return web.json_response(
                        {"error": "rate must be between 0.0 and 1.0"}, status=400
                    )
            except ValueError:
                return web.json_response({"error": "rate must be a number"}, status=400)

            old_rate = learner.epsilon
            learner.epsilon = rate

            return web.json_response(
                {
                    "success": True,
                    "old_rate": round(old_rate, 3),
                    "new_rate": round(rate, 3),
                }
            )

        except Exception as e:
            logger.exception("RL explore error")
            return web.json_response({"error": str(e)}, status=500)

    async def _rl_strategy(self, request: Request) -> Response:
        """Set selection strategy for RL model selector."""
        try:
            from victor.agent.rl.coordinator import get_rl_coordinator
            from victor.agent.rl.learners.model_selector import SelectionStrategy

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("model_selector")
            if learner is None:
                return web.json_response(
                    {"error": "Model selector learner not available"}, status=503
                )

            data = await request.json()
            strategy_name = data.get("strategy")

            if not strategy_name:
                return web.json_response({"error": "strategy required"}, status=400)

            try:
                strategy = SelectionStrategy(strategy_name.lower())
            except ValueError:
                available = [s.value for s in SelectionStrategy]
                return web.json_response(
                    {
                        "error": f"Unknown strategy: {strategy_name}",
                        "available": available,
                    },
                    status=400,
                )

            old_strategy = learner.strategy.value
            learner.strategy = strategy

            return web.json_response(
                {
                    "success": True,
                    "old_strategy": old_strategy,
                    "new_strategy": strategy.value,
                }
            )

        except Exception as e:
            logger.exception("RL strategy error")
            return web.json_response({"error": str(e)}, status=500)

    async def _rl_reset(self, request: Request) -> Response:
        """Reset RL model selector Q-values to initial state."""
        try:
            from victor.agent.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("model_selector")
            if learner is None:
                return web.json_response(
                    {"error": "Model selector learner not available"}, status=503
                )

            # Reset Q-values by clearing database tables
            import sqlite3
            db_path = coordinator.db_path
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("DELETE FROM model_selector_q_values")
            cursor.execute("DELETE FROM model_selector_task_q_values")
            cursor.execute("DELETE FROM model_selector_state")
            conn.commit()
            conn.close()

            # Reload learner to pick up cleared state
            coordinator._learners.pop("model_selector", None)
            learner = coordinator.get_learner("model_selector")

            return web.json_response(
                {
                    "success": True,
                    "message": "RL model selector reset to initial state",
                }
            )

        except Exception as e:
            logger.exception("RL reset error")
            return web.json_response({"error": str(e)}, status=500)

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
