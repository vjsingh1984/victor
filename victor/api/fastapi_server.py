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

"""FastAPI-based HTTP API Server for Victor.

Provides REST API endpoints for IDE integrations (VS Code, JetBrains, etc.)
and external tool access. This is the modern replacement for the aiohttp server.

Features:
- REST API for chat, completions, code search
- WebSocket support for real-time updates
- Server-Sent Events (SSE) for streaming
- Optional rate limiting and API key authentication
- CORS support for browser-based clients
- Auto-generated OpenAPI documentation
"""

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models for Request/Response validation
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "0.1.0"


class StatusResponse(BaseModel):
    """Server status response."""

    connected: bool
    mode: str
    provider: str
    model: str
    workspace: str


class ChatMessage(BaseModel):
    """Single chat message."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Chat request payload."""

    messages: List[ChatMessage]


class ChatResponse(BaseModel):
    """Chat response payload."""

    role: str = "assistant"
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None


class CompletionRequest(BaseModel):
    """Code completion request."""

    prompt: str
    file: Optional[str] = None
    language: Optional[str] = None


class CompletionResponse(BaseModel):
    """Code completion response."""

    completions: List[str]
    error: Optional[str] = None


class SearchRequest(BaseModel):
    """Search request payload."""

    query: str
    max_results: int = Field(default=10, ge=1, le=100)


class CodeSearchRequest(BaseModel):
    """Code search request payload."""

    query: str
    regex: bool = False
    case_sensitive: bool = True
    file_pattern: str = "*"


class SearchResult(BaseModel):
    """Single search result."""

    file: str
    line: int
    content: str
    score: float


class SearchResponse(BaseModel):
    """Search response payload."""

    results: List[SearchResult]
    error: Optional[str] = None


class SwitchModelRequest(BaseModel):
    """Model switch request."""

    provider: str
    model: str


class SwitchModeRequest(BaseModel):
    """Mode switch request."""

    mode: str


class PatchApplyRequest(BaseModel):
    """Patch apply request."""

    patch: str
    dry_run: bool = False


class PatchCreateRequest(BaseModel):
    """Patch create request."""

    file_path: str
    new_content: str


class GitCommitRequest(BaseModel):
    """Git commit request."""

    message: Optional[str] = None
    use_ai: bool = False
    files: Optional[List[str]] = None


class MCPConnectRequest(BaseModel):
    """MCP connect request."""

    server: str
    endpoint: Optional[str] = None


class RLExploreRequest(BaseModel):
    """RL exploration rate request."""

    rate: float = Field(ge=0.0, le=1.0)


class RLStrategyRequest(BaseModel):
    """RL strategy request."""

    strategy: str


class ToolApprovalRequest(BaseModel):
    """Tool approval request."""

    approval_id: str
    approved: bool = False


class LSPRequest(BaseModel):
    """LSP request payload."""

    file: str
    line: int = 0
    character: int = 0


# =============================================================================
# FastAPI Server Class
# =============================================================================


class VictorFastAPIServer:
    """FastAPI-based HTTP API Server for Victor."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        workspace_root: Optional[str] = None,
        rate_limit_rpm: Optional[int] = None,
        api_keys: Optional[Dict[str, str]] = None,
        enable_cors: bool = True,
    ):
        """Initialize the FastAPI server.

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
        self.rate_limit_rpm = rate_limit_rpm
        self.api_keys = api_keys or {}
        self.enable_cors = enable_cors

        self._orchestrator = None
        self._ws_clients: List[WebSocket] = []
        self._pending_tool_approvals: Dict[str, Dict[str, Any]] = {}

        # Create FastAPI app with lifespan
        self.app = FastAPI(
            title="Victor API",
            description="AI Coding Assistant API for IDE integrations",
            version="0.1.0",
            lifespan=self._lifespan,
        )

        # Configure CORS
        if enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # Setup routes
        self._setup_routes()

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncIterator[None]:
        """Manage server lifespan."""
        logger.info(f"Starting Victor FastAPI server on {self.host}:{self.port}")
        yield
        # Cleanup
        if self._orchestrator:
            await self._orchestrator.graceful_shutdown()
        # Close WebSocket connections
        for ws in self._ws_clients:
            try:
                await ws.close()
            except Exception:
                pass
        logger.info("Victor FastAPI server shutdown complete")

    def _setup_routes(self) -> None:
        """Set up API routes."""
        app = self.app

        # Health & Status
        @app.get("/health", response_model=HealthResponse, tags=["System"])
        async def health() -> HealthResponse:
            """Health check endpoint."""
            return HealthResponse()

        @app.get("/status", response_model=StatusResponse, tags=["System"])
        async def status() -> StatusResponse:
            """Get current server status."""
            try:
                orchestrator = await self._get_orchestrator()
                provider_name = "unknown"
                model_name = "unknown"
                mode = "chat"

                if orchestrator.provider:
                    provider_name = getattr(orchestrator.provider, "name", "unknown")
                    model_name = getattr(orchestrator.provider, "model", "unknown")

                if hasattr(orchestrator, "adaptive_controller") and orchestrator.adaptive_controller:
                    mode = orchestrator.adaptive_controller.current_mode.value

                return StatusResponse(
                    connected=True,
                    mode=mode,
                    provider=provider_name,
                    model=model_name,
                    workspace=self.workspace_root,
                )
            except Exception as e:
                logger.warning(f"Status check error: {e}")
                return StatusResponse(
                    connected=False,
                    mode="unknown",
                    provider="unknown",
                    model="unknown",
                    workspace=self.workspace_root,
                )

        @app.get("/providers", tags=["System"])
        async def providers() -> JSONResponse:
            """List available providers."""
            try:
                from victor.providers.registry import ProviderRegistry

                provider_list = ProviderRegistry.list_providers()
                return JSONResponse({"providers": provider_list})
            except Exception as e:
                logger.warning(f"Providers list error: {e}")
                return JSONResponse({"providers": [], "error": str(e)})

        # Chat endpoints
        @app.post("/chat", response_model=ChatResponse, tags=["Chat"])
        async def chat(request: ChatRequest) -> ChatResponse:
            """Chat endpoint (non-streaming)."""
            if not request.messages:
                raise HTTPException(status_code=400, detail="No messages provided")

            orchestrator = await self._get_orchestrator()
            response = await orchestrator.chat(request.messages[-1].content)

            content = getattr(response, "content", None) or ""
            tool_calls = getattr(response, "tool_calls", None) or []

            return ChatResponse(role="assistant", content=content, tool_calls=tool_calls)

        @app.post("/chat/stream", tags=["Chat"])
        async def chat_stream(request: ChatRequest) -> StreamingResponse:
            """Streaming chat endpoint (Server-Sent Events)."""
            if not request.messages:
                raise HTTPException(status_code=400, detail="No messages provided")

            async def event_generator() -> AsyncIterator[str]:
                try:
                    orchestrator = await self._get_orchestrator()
                    async for chunk in orchestrator.stream_chat(
                        request.messages[-1].content
                    ):
                        if hasattr(chunk, "content") or hasattr(chunk, "tool_calls"):
                            content = getattr(chunk, "content", "")
                            tool_calls = getattr(chunk, "tool_calls", None)
                            if content:
                                event = {"type": "content", "content": content}
                            elif tool_calls:
                                event = {"type": "tool_call", "tool_call": tool_calls}
                            else:
                                continue
                        else:
                            event = chunk

                        yield f"data: {json.dumps(event)}\n\n"

                    yield "data: [DONE]\n\n"

                except Exception as e:
                    logger.exception("Stream chat error")
                    yield f'data: {{"type": "error", "message": "{str(e)}"}}\n\n'

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        # Completions
        @app.post("/completions", response_model=CompletionResponse, tags=["Completions"])
        async def completions(request: CompletionRequest) -> CompletionResponse:
            """Get code completions."""
            if not request.prompt:
                return CompletionResponse(completions=[])

            try:
                orchestrator = await self._get_orchestrator()
                file_context = f" in file {request.file}" if request.file else ""
                completion_prompt = f"""Complete the following {request.language or ''} code{file_context}. Only provide the completion, no explanation.

{request.prompt}"""

                response = await orchestrator.chat(completion_prompt)
                content = getattr(response, "content", None) or ""
                return CompletionResponse(completions=[content.strip()])

            except Exception as e:
                logger.exception("Completions error")
                return CompletionResponse(completions=[], error=str(e))

        # Search endpoints
        @app.post("/search/semantic", response_model=SearchResponse, tags=["Search"])
        async def semantic_search(request: SearchRequest) -> SearchResponse:
            """Semantic code search."""
            if not request.query:
                return SearchResponse(results=[])

            try:
                orchestrator = await self._get_orchestrator()
                tool_result = await orchestrator.execute_tool(
                    "semantic_code_search",
                    query=request.query,
                    max_results=request.max_results,
                )

                if tool_result.success:
                    matches = tool_result.data.get("matches", [])
                    results = [
                        SearchResult(
                            file=r.get("file", ""),
                            line=r.get("line", 0),
                            content=r.get("content", ""),
                            score=r.get("score", 0.0),
                        )
                        for r in matches
                    ]
                    return SearchResponse(results=results)
                return SearchResponse(results=[])

            except Exception as e:
                logger.exception("Semantic search error")
                return SearchResponse(results=[], error=str(e))

        @app.post("/search/code", response_model=SearchResponse, tags=["Search"])
        async def code_search(request: CodeSearchRequest) -> SearchResponse:
            """Code search (regex/literal)."""
            if not request.query:
                return SearchResponse(results=[])

            try:
                import subprocess

                cmd = ["rg", "--json", "-n"]
                if not request.case_sensitive:
                    cmd.append("-i")
                if not request.regex:
                    cmd.append("-F")
                if request.file_pattern != "*":
                    cmd.extend(["-g", request.file_pattern])
                cmd.append(request.query)
                cmd.append(self.workspace_root)

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                results = []
                for line in result.stdout.splitlines():
                    try:
                        match = json.loads(line)
                        if match.get("type") == "match":
                            data = match.get("data", {})
                            results.append(
                                SearchResult(
                                    file=data.get("path", {}).get("text", ""),
                                    line=data.get("line_number", 0),
                                    content=data.get("lines", {}).get("text", "").strip(),
                                    score=1.0,
                                )
                            )
                    except json.JSONDecodeError:
                        continue

                return SearchResponse(results=results[:50])

            except Exception as e:
                logger.exception("Code search error")
                return SearchResponse(results=[], error=str(e))

        # Model/Mode management
        @app.post("/model/switch", tags=["Configuration"])
        async def switch_model(request: SwitchModelRequest) -> JSONResponse:
            """Switch AI model."""
            from victor.agent.model_switcher import get_model_switcher

            switcher = get_model_switcher()
            switcher.switch(request.provider, request.model)

            return JSONResponse(
                {"success": True, "provider": request.provider, "model": request.model}
            )

        @app.post("/mode/switch", tags=["Configuration"])
        async def switch_mode(request: SwitchModeRequest) -> JSONResponse:
            """Switch agent mode."""
            from victor.agent.mode_controller import AgentMode, get_mode_controller

            manager = get_mode_controller()
            manager.switch_mode(AgentMode(request.mode))

            return JSONResponse({"success": True, "mode": request.mode})

        @app.get("/models", tags=["Configuration"])
        async def list_models() -> JSONResponse:
            """List available models."""
            from victor.agent.model_switcher import get_model_switcher

            switcher = get_model_switcher()
            models = switcher.get_available_models()

            return JSONResponse(
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

        @app.get("/providers", tags=["Configuration"])
        async def list_providers() -> JSONResponse:
            """List available LLM providers."""
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

                return JSONResponse({"providers": providers_info})

            except Exception as e:
                logger.exception("List providers error")
                return JSONResponse({"providers": [], "error": str(e)})

        # Tools
        @app.get("/tools", tags=["Tools"])
        async def list_tools() -> JSONResponse:
            """List available tools with metadata."""
            try:
                from victor.tools.base import ToolRegistry

                registry = ToolRegistry()
                tools_info = []

                for tool in registry.list_tools():
                    cost_tier = (
                        tool.cost_tier.value
                        if hasattr(tool, "cost_tier") and tool.cost_tier
                        else "free"
                    )
                    category = self._get_tool_category(tool.name)

                    tools_info.append(
                        {
                            "name": tool.name,
                            "description": tool.description or "",
                            "category": category,
                            "cost_tier": cost_tier,
                            "parameters": (
                                tool.parameters if hasattr(tool, "parameters") else {}
                            ),
                            "is_dangerous": self._is_dangerous_tool(tool.name),
                            "requires_approval": cost_tier in ("medium", "high")
                            or self._is_dangerous_tool(tool.name),
                        }
                    )

                tools_info.sort(key=lambda t: (t["category"], t["name"]))

                return JSONResponse(
                    {
                        "tools": tools_info,
                        "total": len(tools_info),
                        "categories": list({t["category"] for t in tools_info}),
                    }
                )

            except Exception as e:
                logger.exception("List tools error")
                return JSONResponse({"tools": [], "total": 0, "error": str(e)})

        @app.post("/tools/approve", tags=["Tools"])
        async def approve_tool(request: ToolApprovalRequest) -> JSONResponse:
            """Approve or reject a pending tool execution."""
            if request.approval_id in self._pending_tool_approvals:
                approval = self._pending_tool_approvals.pop(request.approval_id)
                approval["approved"] = request.approved
                approval["resolved"] = True

                await self._broadcast_ws(
                    {
                        "type": "tool_approval_resolved",
                        "approval_id": request.approval_id,
                        "approved": request.approved,
                        "tool_name": approval.get("tool_name"),
                    }
                )

                return JSONResponse(
                    {
                        "success": True,
                        "approval_id": request.approval_id,
                        "approved": request.approved,
                    }
                )
            else:
                raise HTTPException(status_code=404, detail="Approval not found")

        @app.get("/tools/pending", tags=["Tools"])
        async def pending_approvals() -> JSONResponse:
            """Get list of pending tool approvals."""
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

            return JSONResponse({"pending": pending, "count": len(pending)})

        # Conversation management
        @app.post("/conversation/reset", tags=["Conversation"])
        async def reset_conversation() -> JSONResponse:
            """Reset conversation history."""
            await self._record_rl_feedback()
            if self._orchestrator:
                self._orchestrator.reset_conversation()
            return JSONResponse({"success": True, "message": "Conversation reset"})

        @app.get("/conversation/export", tags=["Conversation"])
        async def export_conversation(format: str = Query("json")) -> Any:
            """Export conversation history."""
            if not self._orchestrator:
                return JSONResponse({"messages": []})

            messages = self._orchestrator.get_messages()

            if format == "markdown":
                content = self._format_messages_markdown(messages)
                return StreamingResponse(
                    iter([content]),
                    media_type="text/markdown",
                )
            else:
                return JSONResponse({"messages": messages})

        # Undo/Redo
        @app.post("/undo", tags=["History"])
        async def undo() -> JSONResponse:
            """Undo last change."""
            from victor.agent.change_tracker import get_change_tracker

            tracker = get_change_tracker()
            success, message, files = tracker.undo()

            return JSONResponse({"success": success, "message": message, "files": files})

        @app.post("/redo", tags=["History"])
        async def redo() -> JSONResponse:
            """Redo last undone change."""
            from victor.agent.change_tracker import get_change_tracker

            tracker = get_change_tracker()
            success, message, files = tracker.redo()

            return JSONResponse({"success": success, "message": message, "files": files})

        @app.get("/history", tags=["History"])
        async def history(limit: int = Query(10, ge=1, le=100)) -> JSONResponse:
            """Get change history."""
            from victor.agent.change_tracker import get_change_tracker

            tracker = get_change_tracker()
            hist = tracker.get_history(limit=limit)

            return JSONResponse({"history": hist})

        # Patch operations
        @app.post("/patch/apply", tags=["Patch"])
        async def apply_patch(request: PatchApplyRequest) -> JSONResponse:
            """Apply a patch."""
            from victor.tools import patch_tool

            result = await patch_tool.apply_patch(
                patch=request.patch, dry_run=request.dry_run
            )
            return JSONResponse(result)

        @app.post("/patch/create", tags=["Patch"])
        async def create_patch(request: PatchCreateRequest) -> JSONResponse:
            """Create a patch."""
            from victor.tools import patch_tool

            result = await patch_tool.create_patch(
                file_path=request.file_path, new_content=request.new_content
            )
            return JSONResponse(result)

        # LSP endpoints
        @app.post("/lsp/completions", tags=["LSP"])
        async def lsp_completions(request: LSPRequest) -> JSONResponse:
            """LSP completions."""
            try:
                from victor.lsp.manager import get_lsp_manager

                manager = get_lsp_manager()
                completions = await manager.get_completions(
                    request.file, request.line, request.character
                )

                return JSONResponse(
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
                return JSONResponse({"completions": [], "error": str(e)})

        @app.post("/lsp/hover", tags=["LSP"])
        async def lsp_hover(request: LSPRequest) -> JSONResponse:
            """LSP hover."""
            try:
                from victor.lsp.manager import get_lsp_manager

                manager = get_lsp_manager()
                hover = await manager.get_hover(
                    request.file, request.line, request.character
                )

                return JSONResponse({"contents": hover.contents if hover else None})

            except Exception as e:
                logger.exception("LSP hover error")
                return JSONResponse({"contents": None, "error": str(e)})

        @app.post("/lsp/definition", tags=["LSP"])
        async def lsp_definition(request: LSPRequest) -> JSONResponse:
            """LSP definition."""
            try:
                from victor.lsp.manager import get_lsp_manager

                manager = get_lsp_manager()
                locations = await manager.get_definition(
                    request.file, request.line, request.character
                )

                return JSONResponse({"locations": locations})

            except Exception as e:
                logger.exception("LSP definition error")
                return JSONResponse({"locations": [], "error": str(e)})

        @app.post("/lsp/references", tags=["LSP"])
        async def lsp_references(request: LSPRequest) -> JSONResponse:
            """LSP references."""
            try:
                from victor.lsp.manager import get_lsp_manager

                manager = get_lsp_manager()
                locations = await manager.get_references(
                    request.file, request.line, request.character
                )

                return JSONResponse({"locations": locations})

            except Exception as e:
                logger.exception("LSP references error")
                return JSONResponse({"locations": [], "error": str(e)})

        @app.post("/lsp/diagnostics", tags=["LSP"])
        async def lsp_diagnostics(request: LSPRequest) -> JSONResponse:
            """LSP diagnostics."""
            try:
                from victor.lsp.manager import get_lsp_manager

                manager = get_lsp_manager()
                diagnostics = manager.get_diagnostics(request.file)

                return JSONResponse({"diagnostics": diagnostics})

            except Exception as e:
                logger.exception("LSP diagnostics error")
                return JSONResponse({"diagnostics": [], "error": str(e)})

        # Git integration
        @app.get("/git/status", tags=["Git"])
        async def git_status() -> JSONResponse:
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
                    return JSONResponse({"is_git_repo": False, "error": result.stderr})

                lines = result.stdout.strip().split("\n")
                branch_line = lines[0] if lines else ""

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

                return JSONResponse(
                    {
                        "is_git_repo": True,
                        "branch": branch,
                        "tracking": tracking,
                        "staged": staged,
                        "unstaged": unstaged,
                        "untracked": untracked,
                        "is_clean": len(staged) == 0
                        and len(unstaged) == 0
                        and len(untracked) == 0,
                    }
                )

            except subprocess.TimeoutExpired:
                return JSONResponse(
                    {"error": "Git command timed out"}, status_code=500
                )
            except FileNotFoundError:
                return JSONResponse({"is_git_repo": False, "error": "Git not installed"})
            except Exception as e:
                logger.exception("Git status error")
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.post("/git/commit", tags=["Git"])
        async def git_commit(request: GitCommitRequest) -> JSONResponse:
            """Create a git commit."""
            try:
                import subprocess

                if request.files:
                    for f in request.files:
                        subprocess.run(
                            ["git", "add", f],
                            cwd=self.workspace_root,
                            capture_output=True,
                            timeout=10,
                        )

                message = request.message
                if request.use_ai and not message:
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
                        message = message.replace("```", "").strip()
                        if message.startswith('"') and message.endswith('"'):
                            message = message[1:-1]

                if not message:
                    raise HTTPException(
                        status_code=400, detail="Commit message required"
                    )

                result = subprocess.run(
                    ["git", "commit", "-m", message],
                    cwd=self.workspace_root,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode != 0:
                    return JSONResponse(
                        {"success": False, "error": result.stderr or "Commit failed"}
                    )

                return JSONResponse(
                    {"success": True, "message": message, "output": result.stdout}
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.exception("Git commit error")
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.get("/git/log", tags=["Git"])
        async def git_log(limit: int = Query(20, ge=1, le=100)) -> JSONResponse:
            """Get git commit log."""
            try:
                import subprocess

                result = subprocess.run(
                    ["git", "log", f"-{limit}", "--pretty=format:%H|%an|%ae|%ar|%s"],
                    cwd=self.workspace_root,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode != 0:
                    return JSONResponse({"error": result.stderr}, status_code=500)

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

                return JSONResponse({"commits": commits})

            except Exception as e:
                logger.exception("Git log error")
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.get("/git/diff", tags=["Git"])
        async def git_diff(
            staged: bool = Query(False),
            file: Optional[str] = Query(None),
        ) -> JSONResponse:
            """Get git diff."""
            try:
                import subprocess

                cmd = ["git", "diff"]
                if staged:
                    cmd.append("--cached")
                if file:
                    cmd.append("--")
                    cmd.append(file)

                result = subprocess.run(
                    cmd,
                    cwd=self.workspace_root,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                return JSONResponse(
                    {
                        "diff": result.stdout[:50000],
                        "truncated": len(result.stdout) > 50000,
                    }
                )

            except Exception as e:
                logger.exception("Git diff error")
                return JSONResponse({"error": str(e)}, status_code=500)

        # Workspace analysis
        @app.get("/workspace/overview", tags=["Workspace"])
        async def workspace_overview(depth: int = Query(3, ge=1, le=10)) -> JSONResponse:
            """Get workspace structure overview."""
            try:
                import os
                from pathlib import Path

                root = Path(self.workspace_root)
                overview: Dict[str, Any] = {
                    "root": str(root),
                    "name": root.name,
                    "file_counts": {},
                    "total_files": 0,
                    "total_size": 0,
                }

                exclude_dirs = {
                    ".git",
                    "node_modules",
                    "__pycache__",
                    ".venv",
                    "venv",
                    ".victor",
                }

                def scan_dir(path: Path, d: int = 0) -> Dict[str, Any]:
                    if d > depth:
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
                            if entry.name.startswith(".") and entry.name not in {
                                ".github",
                                ".vscode",
                            }:
                                continue
                            if entry.name in exclude_dirs:
                                continue

                            if entry.is_dir():
                                result["children"].append(scan_dir(entry, d + 1))
                            else:
                                ext = entry.suffix.lower()
                                overview["file_counts"][ext] = (
                                    overview["file_counts"].get(ext, 0) + 1
                                )
                                overview["total_files"] += 1
                                try:
                                    overview["total_size"] += entry.stat().st_size
                                except OSError:
                                    pass

                                if d <= 1:
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

                return JSONResponse(overview)

            except Exception as e:
                logger.exception("Workspace overview error")
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.get("/workspace/metrics", tags=["Workspace"])
        async def workspace_metrics() -> JSONResponse:
            """Get code metrics for the workspace."""
            try:
                orchestrator = await self._get_orchestrator()

                try:
                    tool_result = await orchestrator.execute_tool(
                        "metrics", path=self.workspace_root
                    )
                    if tool_result.success:
                        return JSONResponse(tool_result.data)
                except Exception:
                    pass

                # Basic metrics fallback
                from pathlib import Path

                root = Path(self.workspace_root)
                metrics: Dict[str, Any] = {
                    "lines_of_code": 0,
                    "files_by_type": {},
                    "largest_files": [],
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
                                with open(
                                    path, "r", encoding="utf-8", errors="ignore"
                                ) as f:
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

                file_sizes.sort(key=lambda x: x["lines"], reverse=True)
                metrics["largest_files"] = file_sizes[:10]

                return JSONResponse(metrics)

            except Exception as e:
                logger.exception("Workspace metrics error")
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.get("/workspace/security", tags=["Workspace"])
        async def workspace_security() -> JSONResponse:
            """Get security scan results."""
            try:
                orchestrator = await self._get_orchestrator()

                try:
                    tool_result = await orchestrator.execute_tool(
                        "scan",
                        path=self.workspace_root,
                        scan_type="secrets",
                    )
                    if tool_result.success:
                        return JSONResponse(
                            {"scan_completed": True, "results": tool_result.data}
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
                    (
                        r'(?i)(secret|password|passwd|pwd)\s*[:=]\s*["\'][^"\']{8,}',
                        "Secret/Password",
                    ),
                    (r"(?i)bearer\s+[\w-]{20,}", "Bearer Token"),
                    (r"sk-[a-zA-Z0-9]{20,}", "OpenAI API Key"),
                    (r"ghp_[a-zA-Z0-9]{36}", "GitHub Token"),
                    (r"AKIA[A-Z0-9]{16}", "AWS Access Key"),
                ]

                code_extensions = {
                    ".py",
                    ".ts",
                    ".js",
                    ".json",
                    ".yaml",
                    ".yml",
                    ".env",
                    ".sh",
                }

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

                return JSONResponse(
                    {
                        "scan_completed": True,
                        "findings": findings[:50],
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
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.get("/workspace/dependencies", tags=["Workspace"])
        async def workspace_dependencies() -> JSONResponse:
            """Get dependency information."""
            try:
                from pathlib import Path

                root = Path(self.workspace_root)
                dependencies: Dict[str, Any] = {}

                # Python dependencies
                for req_file in ["requirements.txt", "pyproject.toml", "setup.py"]:
                    req_path = root / req_file
                    if req_path.exists():
                        if req_file == "requirements.txt":
                            deps = []
                            for line in req_path.read_text().splitlines():
                                line = line.strip()
                                if line and not line.startswith("#"):
                                    deps.append(
                                        line.split("==")[0].split(">=")[0].split("<")[0]
                                    )
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
                    dependencies["rust"] = {"file": "Cargo.toml", "exists": True}

                # Go dependencies
                go_mod = root / "go.mod"
                if go_mod.exists():
                    dependencies["go"] = {"file": "go.mod", "exists": True}

                return JSONResponse(
                    {"workspace": str(root), "dependencies": dependencies}
                )

            except Exception as e:
                logger.exception("Workspace dependencies error")
                return JSONResponse({"error": str(e)}, status_code=500)

        # MCP endpoints
        @app.get("/mcp/servers", tags=["MCP"])
        async def mcp_servers() -> JSONResponse:
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

                return JSONResponse({"servers": servers})

            except ImportError:
                return JSONResponse({"servers": [], "error": "MCP not available"})
            except Exception as e:
                logger.exception("MCP servers error")
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.post("/mcp/connect", tags=["MCP"])
        async def mcp_connect(request: MCPConnectRequest) -> JSONResponse:
            """Connect to an MCP server."""
            try:
                from victor.mcp.registry import get_mcp_registry

                registry = get_mcp_registry()
                success = await registry.connect(request.server, endpoint=request.endpoint)

                return JSONResponse({"success": success, "server": request.server})

            except ImportError:
                raise HTTPException(status_code=501, detail="MCP not available")
            except Exception as e:
                logger.exception("MCP connect error")
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.post("/mcp/disconnect", tags=["MCP"])
        async def mcp_disconnect(request: MCPConnectRequest) -> JSONResponse:
            """Disconnect from an MCP server."""
            try:
                from victor.mcp.registry import get_mcp_registry

                registry = get_mcp_registry()
                await registry.disconnect(request.server)

                return JSONResponse({"success": True, "server": request.server})

            except ImportError:
                raise HTTPException(status_code=501, detail="MCP not available")
            except Exception as e:
                logger.exception("MCP disconnect error")
                return JSONResponse({"error": str(e)}, status_code=500)

        # RL Model Selector endpoints
        @app.get("/rl/stats", tags=["RL"])
        async def rl_stats() -> JSONResponse:
            """Get RL model selector statistics."""
            try:
                from victor.agent.rl.coordinator import get_rl_coordinator

                coordinator = get_rl_coordinator()
                learner = coordinator.get_learner("model_selector")

                if not learner:
                    return JSONResponse({"error": "Model selector learner not available"}, status_code=503)

                # Get rankings
                rankings = learner.get_provider_rankings()

                # Build task Q-table summary
                task_q_summary = {}
                for provider, task_q_table in learner._q_table_by_task.items():
                    task_q_summary[provider] = {
                        task_type: round(q_val, 3)
                        for task_type, q_val in task_q_table.items()
                    }

                stats = {
                    "strategy": learner.strategy.value,
                    "epsilon": round(learner.epsilon, 3),
                    "total_selections": learner._total_selections,
                    "num_providers": len(learner._q_table),
                    "top_provider": rankings[0]["provider"] if rankings else None,
                    "top_q_value": round(rankings[0]["q_value"], 3) if rankings else 0.0,
                    "learning_rate": learner.learning_rate,
                    "ucb_c": learner.ucb_c,
                    "provider_rankings": [
                        {
                            "provider": r["provider"],
                            "q_value": round(r["q_value"], 3),
                            "sessions": r["session_count"],
                            "confidence": round(r["confidence"], 3),
                        }
                        for r in rankings[:5]
                    ],
                    "task_q_tables": task_q_summary,
                    "db_path": str(coordinator.db_path),
                }

                return JSONResponse(stats)

            except Exception as e:
                logger.exception("RL stats error")
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.get("/rl/recommend", tags=["RL"])
        async def rl_recommend(task_type: Optional[str] = Query(None)) -> JSONResponse:
            """Get model recommendation based on Q-values."""
            try:
                from victor.agent.rl.coordinator import get_rl_coordinator
                import json

                coordinator = get_rl_coordinator()
                learner = coordinator.get_learner("model_selector")

                if not learner:
                    return JSONResponse({"error": "Model selector learner not available"}, status_code=503)

                available = list(learner._q_table.keys()) if learner._q_table else ["ollama"]

                # Get recommendation
                recommendation = coordinator.get_recommendation(
                    "model_selector",
                    json.dumps(available),
                    "",
                    task_type or "unknown"
                )

                if not recommendation:
                    return JSONResponse(
                        {
                            "provider": available[0] if available else "ollama",
                            "model": None,
                            "q_value": 0.5,
                            "confidence": 0.0,
                            "reason": "No recommendation available",
                            "task_type": task_type,
                            "alternatives": [],
                        }
                    )

                # Get alternatives
                alternatives = []
                for provider in available:
                    if provider != recommendation.value:
                        q_val = learner._get_q_value(provider, task_type)
                        alternatives.append({"provider": provider, "q_value": round(q_val, 3)})
                alternatives.sort(key=lambda x: x["q_value"], reverse=True)

                return JSONResponse(
                    {
                        "provider": recommendation.value,
                        "model": None,
                        "q_value": round(learner._get_q_value(recommendation.value, task_type), 3),
                        "confidence": round(recommendation.confidence, 3),
                        "reason": recommendation.reason,
                        "task_type": task_type,
                        "alternatives": alternatives[:3],
                    }
                )

            except Exception as e:
                logger.exception("RL recommend error")
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.post("/rl/explore", tags=["RL"])
        async def rl_explore(request: RLExploreRequest) -> JSONResponse:
            """Set exploration rate for RL model selector."""
            try:
                from victor.agent.rl.coordinator import get_rl_coordinator

                coordinator = get_rl_coordinator()
                learner = coordinator.get_learner("model_selector")

                if not learner:
                    return JSONResponse({"error": "Model selector learner not available"}, status_code=503)

                old_rate = learner.epsilon
                learner.epsilon = request.rate

                return JSONResponse(
                    {
                        "success": True,
                        "old_rate": round(old_rate, 3),
                        "new_rate": round(request.rate, 3),
                    }
                )

            except Exception as e:
                logger.exception("RL explore error")
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.post("/rl/strategy", tags=["RL"])
        async def rl_strategy(request: RLStrategyRequest) -> JSONResponse:
            """Set selection strategy for RL model selector."""
            try:
                from victor.agent.rl.coordinator import get_rl_coordinator
                from victor.agent.rl.learners.model_selector import SelectionStrategy

                coordinator = get_rl_coordinator()
                learner = coordinator.get_learner("model_selector")

                if not learner:
                    return JSONResponse({"error": "Model selector learner not available"}, status_code=503)

                try:
                    strategy = SelectionStrategy(request.strategy.lower())
                except ValueError:
                    available = [s.value for s in SelectionStrategy]
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown strategy: {request.strategy}. Available: {available}",
                    )

                old_strategy = learner.strategy.value
                learner.strategy = strategy

                return JSONResponse(
                    {
                        "success": True,
                        "old_strategy": old_strategy,
                        "new_strategy": strategy.value,
                    }
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.exception("RL strategy error")
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.post("/rl/reset", tags=["RL"])
        async def rl_reset() -> JSONResponse:
            """Reset RL model selector Q-values."""
            try:
                from victor.agent.rl.coordinator import get_rl_coordinator

                coordinator = get_rl_coordinator()
                learner = coordinator.get_learner("model_selector")
                if learner is None:
                    return JSONResponse(
                        {"error": "Model selector learner not available"}, status_code=503
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

                return JSONResponse(
                    {"success": True, "message": "RL model selector reset to initial state"}
                )

            except Exception as e:
                logger.exception("RL reset error")
                return JSONResponse({"error": str(e)}, status_code=500)

        # Placeholder endpoints
        @app.get("/credentials/get", tags=["System"])
        async def credentials_get(provider: str = Query("")) -> JSONResponse:
            """Placeholder credentials endpoint."""
            return JSONResponse({"provider": provider, "api_key": None})

        @app.post("/session/token", tags=["System"])
        async def session_token() -> JSONResponse:
            """Placeholder session token endpoint."""
            return JSONResponse(
                {"session_token": str(uuid.uuid4()), "session_id": str(uuid.uuid4())}
            )

        # Server shutdown
        @app.post("/shutdown", tags=["System"])
        async def shutdown() -> JSONResponse:
            """Shutdown the server."""
            logger.info("Shutdown requested")
            await self._record_rl_feedback()

            for ws in self._ws_clients:
                try:
                    await ws.close()
                except Exception:
                    pass

            asyncio.get_event_loop().call_later(0.5, asyncio.get_event_loop().stop)
            return JSONResponse({"status": "shutting_down"})

        # WebSocket endpoint
        @app.websocket("/ws")
        async def websocket_handler(websocket: WebSocket) -> None:
            """Handle WebSocket connections."""
            await websocket.accept()
            self._ws_clients.append(websocket)
            logger.info(f"WebSocket client connected. Total: {len(self._ws_clients)}")

            try:
                while True:
                    data = await websocket.receive_json()
                    await self._handle_ws_message(websocket, data)
            except WebSocketDisconnect:
                pass
            finally:
                if websocket in self._ws_clients:
                    self._ws_clients.remove(websocket)
                logger.info(
                    f"WebSocket client disconnected. Total: {len(self._ws_clients)}"
                )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _get_orchestrator(self) -> Any:
        """Get or create the orchestrator."""
        if self._orchestrator is None:
            from victor.agent.orchestrator import AgentOrchestrator
            from victor.config.settings import load_settings

            settings = load_settings()
            self._orchestrator = await AgentOrchestrator.from_settings(settings)

        return self._orchestrator

    async def _record_rl_feedback(self) -> None:
        """Record RL feedback for the current session."""
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

            msg_count = 0
            if hasattr(self._orchestrator, "message_count"):
                msg_count = self._orchestrator.message_count
            elif hasattr(self._orchestrator, "get_messages"):
                messages = self._orchestrator.get_messages()
                msg_count = len(messages) if messages else 0

            if msg_count == 0:
                return

            metrics = {}
            if hasattr(self._orchestrator, "get_session_metrics"):
                metrics = self._orchestrator.get_session_metrics() or {}

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
                success=True,
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

    async def _handle_ws_message(
        self, ws: WebSocket, data: Dict[str, Any]
    ) -> None:
        """Handle incoming WebSocket messages."""
        msg_type = data.get("type", "")

        if msg_type == "chat":
            messages = data.get("messages", [])
            if not messages:
                await ws.send_json({"type": "error", "message": "No messages"})
                return

            orchestrator = await self._get_orchestrator()

            try:
                async for chunk in orchestrator.stream_chat(
                    messages[-1].get("content", "")
                ):
                    if chunk.get("type") == "content":
                        await ws.send_json(
                            {"type": "content", "content": chunk["content"]}
                        )
                    elif chunk.get("type") == "tool_call":
                        await ws.send_json(
                            {"type": "tool_call", "tool_call": chunk["tool_call"]}
                        )

                await ws.send_json({"type": "done"})
            except Exception as e:
                await ws.send_json({"type": "error", "message": str(e)})

        elif msg_type == "ping":
            await ws.send_json({"type": "pong"})

        elif msg_type == "subscribe":
            channel = data.get("channel", "")
            await ws.send_json({"type": "subscribed", "channel": channel})

    async def _broadcast_ws(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all connected WebSocket clients."""
        for ws in self._ws_clients:
            try:
                await ws.send_json(message)
            except Exception:
                pass

    def _format_messages_markdown(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages as markdown."""
        lines = ["# Conversation Export\n"]
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            lines.append(f"## {role}\n")
            lines.append(f"{content}\n")
        return "\n".join(lines)

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

    def run(self) -> None:
        """Run the server synchronously."""
        import uvicorn

        uvicorn.run(self.app, host=self.host, port=self.port)

    async def start_async(self) -> "VictorFastAPIServer":
        """Start the server asynchronously and return self for cleanup."""
        import uvicorn

        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        self._server = uvicorn.Server(config)
        asyncio.create_task(self._server.serve())
        logger.info(f"Victor FastAPI server running on {self.host}:{self.port}")
        return self

    async def shutdown(self) -> None:
        """Shutdown the server."""
        if hasattr(self, "_server"):
            self._server.should_exit = True


def create_fastapi_app(workspace_root: Optional[str] = None) -> FastAPI:
    """Create the FastAPI application."""
    server = VictorFastAPIServer(workspace_root=workspace_root)
    return server.app
