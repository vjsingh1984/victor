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
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import (
    Body,
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from victor.integrations.search_types import CodeSearchResult
from victor.integrations.api.event_bridge import EventBridge
from victor.integrations.api.graph_export import (
    export_graph_schema,
    get_execution_state,
    WorkflowExecutionState,
)
from victor.integrations.api.workflow_event_bridge import WorkflowEventBridge
from victor.core.events import ObservabilityBus as EventBus, get_observability_bus
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models for Request/Response validation
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "0.5.0"


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


class CompletionPosition(BaseModel):
    """Cursor position for completions."""

    line: int
    character: int


class CompletionRequest(BaseModel):
    """Code completion request with FIM (Fill-in-the-Middle) support."""

    prompt: str  # Code before cursor (prefix)
    suffix: Optional[str] = None  # Code after cursor (for FIM)
    file: Optional[str] = None
    language: Optional[str] = None
    position: Optional[CompletionPosition] = None
    context: Optional[str] = None  # Additional file context (imports, etc.)
    max_tokens: int = Field(default=128, ge=1, le=512)  # Limit completion length
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)  # Deterministic by default
    stop_sequences: Optional[List[str]] = None  # Custom stop sequences


class CompletionResponse(BaseModel):
    """Code completion response."""

    completions: List[str]
    error: Optional[str] = None
    latency_ms: Optional[float] = None  # Track performance


class SearchRequest(BaseModel):
    """Search request payload."""

    query: str
    max_results: int = Field(default=10, ge=1, le=100)


class CodeSearchRequest(BaseModel):
    """Code search request payload."""

    query: str = Field(..., min_length=1, max_length=1000)
    regex: bool = False
    case_sensitive: bool = True
    file_pattern: str = Field(default="*", max_length=200)

    @field_validator("file_pattern")
    @classmethod
    def validate_file_pattern(cls, v: str) -> str:
        """Validate file pattern to prevent shell injection."""
        # Disallow shell metacharacters except glob patterns
        dangerous_chars = [";", "|", "&", "$", "`", "(", ")", "<", ">", "\\"]
        for char in dangerous_chars:
            if char in v:
                raise ValueError(f"Invalid character '{char}' in file pattern")
        return v


class APISearchResult(BaseModel):
    """Single search result for API responses.

    For HTTP API search result serialization.
    Renamed from SearchResult to be semantically distinct from other search types.
    """

    file: str
    line: int
    content: str
    score: float

    @classmethod
    def from_code_result(cls, result: CodeSearchResult) -> "APISearchResult":
        return cls(
            file=result.file,
            line=result.line,
            content=result.content,
            score=result.score,
        )


# Backward compatibility alias
class SearchResponse(BaseModel):
    """Search response payload."""

    results: List[APISearchResult]
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

    message: Optional[str] = Field(default=None, max_length=5000)
    use_ai: bool = False
    files: Optional[List[str]] = None

    @field_validator("files")
    @classmethod
    def validate_files(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate file paths to prevent path traversal."""
        if v is None:
            return v
        for file_path in v:
            # Disallow absolute paths and path traversal
            if file_path.startswith("/") or file_path.startswith("\\"):
                raise ValueError("Absolute paths not allowed")
            if ".." in file_path:
                raise ValueError("Path traversal not allowed")
            # Disallow shell metacharacters
            dangerous_chars = [";", "|", "&", "$", "`", "(", ")", "<", ">"]
            for char in dangerous_chars:
                if char in file_path:
                    raise ValueError(f"Invalid character '{char}' in file path")
        return v


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


class TerminalCommandRequest(BaseModel):
    """Terminal command execution request."""

    command: str = Field(..., min_length=1, max_length=5000)
    working_dir: Optional[str] = None
    timeout: int = Field(default=60, ge=1, le=300)
    require_approval: bool = True

    @field_validator("command")
    @classmethod
    def validate_command(cls, v: str) -> str:
        """Validate command to prevent obviously dangerous operations."""
        # Disallow command chaining with shell metacharacters
        dangerous_patterns = [
            "rm -rf /",
            "rm -rf ~",
            "> /dev/sd",
            "mkfs.",
            "dd if=",
            ":(){:|:&};:",  # Fork bomb
        ]
        for pattern in dangerous_patterns:
            if pattern in v:
                raise ValueError(f"Command contains dangerous pattern: {pattern}")
        return v


class TerminalCommandResponse(BaseModel):
    """Terminal command execution response."""

    command_id: str
    command: str
    status: str  # 'pending', 'running', 'completed', 'failed', 'cancelled'
    output: Optional[str] = None
    exit_code: Optional[int] = None
    is_dangerous: bool = False
    requires_approval: bool = True


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
        enable_hitl: bool = False,
        hitl_auth_token: Optional[str] = None,
        hitl_persistent: bool = True,
    ):
        """Initialize the FastAPI server.

        Args:
            host: Host to bind to
            port: Port to listen on
            workspace_root: Root directory of the workspace
            rate_limit_rpm: Optional requests per minute limit (None = no limit)
            api_keys: Optional dict of {api_key: client_id} for authentication
            enable_cors: Enable CORS headers (default: True)
            enable_hitl: Enable HITL (Human-in-the-Loop) endpoints (default: False)
            hitl_auth_token: Optional auth token for HITL endpoints
            hitl_persistent: Use SQLite for persistent HITL storage (default: True)
        """
        self.host = host
        self.port = port
        self.workspace_root = workspace_root or str(Path.cwd())
        self.rate_limit_rpm = rate_limit_rpm
        self.api_keys = api_keys or {}
        self.enable_cors = enable_cors
        self.enable_hitl = enable_hitl
        self.hitl_auth_token = hitl_auth_token
        self.hitl_persistent = hitl_persistent

        self._orchestrator = None
        self._ws_clients: List[WebSocket] = []
        self._pending_tool_approvals: Dict[str, Dict[str, Any]] = {}
        self._hitl_store: Optional[Any] = None
        self._event_bridge: Optional[EventBridge] = None
        self._event_clients: List[WebSocket] = []
        self._workflow_event_bridge: Optional[WorkflowEventBridge] = None
        self._workflow_executions: Dict[str, Dict[str, Any]] = {}
        self._shutting_down = False

        # Create FastAPI app with lifespan
        self.app = FastAPI(
            title="Victor API",
            description="AI Coding Assistant API for IDE integrations",
            version="0.5.0",
            lifespan=self._lifespan,
        )

        # Configure CORS with secure defaults
        # Allow localhost for development and VS Code webview origins
        if enable_cors:
            cors_origins = [
                "http://localhost:*",
                "http://127.0.0.1:*",
                "https://localhost:*",
                "https://127.0.0.1:*",
                "vscode-webview://*",
            ]
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_origins,
                allow_origin_regex=r"^(http://localhost:\d+|http://127\.0\.0\.1:\d+|vscode-webview://[a-z0-9-]+)$",
                allow_credentials=True,
                allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                allow_headers=["*"],
            )

        # Setup routes
        self._setup_routes()

        # Setup HITL routes if enabled
        if self.enable_hitl:
            self._setup_hitl_routes()

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncIterator[None]:
        """Manage server lifespan."""
        logger.info(f"Starting Victor FastAPI server on {self.host}:{self.port}")

        # Initialize EventBridge for real-time event streaming
        event_bus = get_observability_bus()
        self._event_bridge = EventBridge(event_bus)
        self._event_bridge.start()
        logger.info("EventBridge started for real-time event streaming")

        # Initialize WorkflowEventBridge for workflow visualization
        self._workflow_event_bridge = WorkflowEventBridge(event_bus)
        await self._workflow_event_bridge.start()
        logger.info("WorkflowEventBridge started for workflow visualization")

        yield

        # Cleanup
        if self._event_bridge:
            self._event_bridge.stop()
        if self._workflow_event_bridge:
            await self._workflow_event_bridge.stop()
        if self._orchestrator:
            await self._orchestrator.graceful_shutdown()
        # Close WebSocket connections
        for ws in self._ws_clients + self._event_clients:
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

                if (
                    hasattr(orchestrator, "adaptive_controller")
                    and orchestrator.adaptive_controller
                ):
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
                    async for chunk in orchestrator.stream_chat(request.messages[-1].content):
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

        # Completions with FIM (Fill-in-the-Middle) support
        @app.post("/completions", response_model=CompletionResponse, tags=["Completions"])
        async def completions(request: CompletionRequest) -> CompletionResponse:
            """Get fast code completions with FIM support.

            For inline/ghost text completions, this endpoint is optimized for:
            - Low latency (bypasses orchestrator overhead)
            - FIM format (uses both prefix and suffix context)
            - Deterministic output (temperature=0 by default)
            """
            import time

            start_time = time.perf_counter()

            if not request.prompt:
                return CompletionResponse(completions=[], latency_ms=0.0)

            try:
                orchestrator = await self._get_orchestrator()
                provider = orchestrator.provider_manager.current_provider

                # Build FIM prompt for better completions
                file_info = f" ({request.language})" if request.language else ""
                if request.file:
                    file_info = f" in {request.file}{file_info}"

                # Use FIM format if suffix is provided
                if request.suffix:
                    # FIM format: prefix <FILL> suffix
                    completion_prompt = f"""Complete the code at <FILL>. Only output the completion, nothing else.

{request.context or ''}

{request.prompt}<FILL>{request.suffix}"""
                else:
                    # Standard completion (no suffix)
                    completion_prompt = f"""Complete this {request.language or 'code'}{file_info}. Only output the completion.

{request.context or ''}

{request.prompt}"""

                # Default stop sequences for code completions
                stop_sequences = request.stop_sequences or [
                    "\n\n",  # Stop at blank line
                    "\ndef ",  # Python function
                    "\nclass ",  # Python class
                    "\nfunction ",  # JavaScript function
                    "\n//",  # Comment start
                    "\n#",  # Python/shell comment
                ]

                # Direct provider call for lower latency
                messages = [{"role": "user", "content": completion_prompt}]
                response = await provider.chat(
                    messages=messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    stop=stop_sequences,
                )

                content = getattr(response, "content", "") or ""
                # Clean up the completion
                completion = content.strip()
                # Remove any explanation text that might follow the code
                if "\n\nExplanation:" in completion:
                    completion = completion.split("\n\nExplanation:")[0]
                if "\n\nNote:" in completion:
                    completion = completion.split("\n\nNote:")[0]

                latency_ms = (time.perf_counter() - start_time) * 1000

                return CompletionResponse(
                    completions=[completion] if completion else [],
                    latency_ms=round(latency_ms, 2),
                )

            except Exception as e:
                logger.exception("Completions error")
                latency_ms = (time.perf_counter() - start_time) * 1000
                return CompletionResponse(
                    completions=[],
                    error=str(e),
                    latency_ms=round(latency_ms, 2),
                )

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
                    code_results = [
                        CodeSearchResult(
                            file=r.get("file", ""),
                            line=r.get("line", 0),
                            content=r.get("content", ""),
                            score=r.get("score", 0.0),
                            context=r.get("context", ""),
                        )
                        for r in matches
                    ]
                    results = [APISearchResult.from_code_result(r) for r in code_results]
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

                code_results = []
                for line in result.stdout.splitlines():
                    try:
                        match = json.loads(line)
                        if match.get("type") == "match":
                            data = match.get("data", {})
                            code_results.append(
                                CodeSearchResult(
                                    file=data.get("path", {}).get("text", ""),
                                    line=data.get("line_number", 0),
                                    content=data.get("lines", {}).get("text", "").strip(),
                                    score=1.0,
                                )
                            )
                    except json.JSONDecodeError:
                        continue

                results = [APISearchResult.from_code_result(r) for r in code_results[:50]]
                return SearchResponse(results=results)

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
                from victor.providers.registry import get_provider_registry, ProviderRegistry

                providers_info = []

                for provider_name in ProviderRegistry.list_providers():
                    try:
                        provider_class = ProviderRegistry.get(provider_name)
                        if provider_class is not None:
                            # Try to instantiate provider to check capabilities
                            supports_tools = False
                            supports_streaming = True
                            configured = True
                            try:
                                provider_instance = provider_class()
                                supports_tools = provider_instance.supports_tools() if hasattr(provider_instance, "supports_tools") else False
                                supports_streaming = provider_instance.supports_streaming() if hasattr(provider_instance, "supports_streaming") else True
                                configured = provider_instance.is_configured() if hasattr(provider_instance, "is_configured") else True
                            except Exception:
                                pass

                            providers_info.append(
                                {
                                    "name": provider_name,
                                    "display_name": provider_name.replace("_", " ").title(),
                                    "is_local": provider_name in ("ollama", "lmstudio", "vllm"),
                                    "configured": configured,
                                    "supports_tools": supports_tools,
                                    "supports_streaming": supports_streaming,
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

        @app.get("/capabilities", tags=["Configuration"])
        async def get_capabilities(
            vertical: Optional[str] = Query(None, description="Filter by vertical")
        ) -> JSONResponse:
            """Discover all Victor capabilities.

            Returns aggregated information about tools, verticals, personas,
            teams, workflows, and other discoverable features.
            """
            try:
                from victor.ui.commands.capabilities import get_capability_discovery

                discovery = get_capability_discovery()

                if vertical:
                    manifest = discovery.discover_by_vertical(vertical)
                    # Convert to dict if it's a dataclass
                    if hasattr(manifest, "__dict__"):
                        return JSONResponse(manifest.__dict__)
                    return JSONResponse(manifest)
                else:
                    manifest = discovery.discover_all()
                    # Convert to dict if it's a dataclass
                    if hasattr(manifest, "__dict__"):
                        return JSONResponse(manifest.__dict__)
                    return JSONResponse(manifest)

            except Exception as e:
                logger.exception("Capabilities discovery error")
                return JSONResponse({"error": str(e), "capabilities": {}}, status_code=500)

        # Tools
        @app.get("/tools", tags=["Tools"])
        async def list_tools() -> JSONResponse:
            """List available tools with metadata."""
            try:
                from victor.tools.registry import ToolRegistry

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
                            "parameters": (tool.parameters if hasattr(tool, "parameters") else {}),
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
            if self._orchestrator is not None:
                self._orchestrator.reset_conversation()  # type: ignore[unreachable]
            return JSONResponse({"success": True, "message": "Conversation reset"})

        @app.get("/conversation/export", tags=["Conversation"])
        async def export_conversation(format: str = Query("json")) -> Any:
            """Export conversation history."""
            if self._orchestrator is None:
                return JSONResponse({"messages": []})

            messages = self._orchestrator.get_messages()  # type: ignore[unreachable]

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
            from victor.tools.patch_tool import patch

            result = await patch(
                operation="apply",
                patch_content=request.patch,
                dry_run=request.dry_run,
            )
            return JSONResponse(result)

        @app.post("/patch/create", tags=["Patch"])
        async def create_patch(request: PatchCreateRequest) -> JSONResponse:
            """Create a patch."""
            from victor.tools.patch_tool import patch

            result = await patch(
                operation="create",
                file_path=request.file_path,
                new_content=request.new_content,
            )
            return JSONResponse(result)

        # LSP endpoints
        @app.post("/lsp/completions", tags=["LSP"])
        async def lsp_completions(request: LSPRequest) -> JSONResponse:
            """LSP completions."""
            try:
                from victor.coding.lsp.manager import get_lsp_manager

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
                from victor.coding.lsp.manager import get_lsp_manager

                manager = get_lsp_manager()
                hover = await manager.get_hover(request.file, request.line, request.character)

                return JSONResponse({"contents": hover.contents if hover else None})

            except Exception as e:
                logger.exception("LSP hover error")
                return JSONResponse({"contents": None, "error": str(e)})

        @app.post("/lsp/definition", tags=["LSP"])
        async def lsp_definition(request: LSPRequest) -> JSONResponse:
            """LSP definition."""
            try:
                from victor.coding.lsp.manager import get_lsp_manager

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
                from victor.coding.lsp.manager import get_lsp_manager

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
                from victor.coding.lsp.manager import get_lsp_manager

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
                        "is_clean": len(staged) == 0 and len(unstaged) == 0 and len(untracked) == 0,
                    }
                )

            except subprocess.TimeoutExpired:
                return JSONResponse({"error": "Git command timed out"}, status_code=500)
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
                    raise HTTPException(status_code=400, detail="Commit message required")

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

                return JSONResponse({"success": True, "message": message, "output": result.stdout})

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

        # Terminal Agent endpoints
        @app.post("/terminal/suggest", tags=["Terminal"])
        async def terminal_suggest(intent: str = Query(..., min_length=1)) -> JSONResponse:
            """Suggest a terminal command based on user intent."""
            try:
                orchestrator = await self._get_orchestrator()
                prompt = f"""Generate a terminal command for this task. Return ONLY the command, nothing else.

Working directory: {self.workspace_root}
OS: {__import__('sys').platform}
Task: {intent}

Respond with just the command to run."""

                response = await orchestrator.chat(prompt)
                command = response.get("content", "").strip()

                # Clean up markdown code blocks
                if command.startswith("```"):
                    lines = command.split("\n")
                    command = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
                command = command.strip()

                if not command:
                    return JSONResponse({"error": "Could not generate command"}, status_code=400)

                # Check if command is dangerous
                dangerous_patterns = [
                    "rm -rf /",
                    "rm -rf ~",
                    "> /dev/sd",
                    "mkfs.",
                    "dd if=",
                    "sudo rm",
                    ":(){",
                    "chmod -R 777",
                    "curl | sh",
                    "wget | sh",
                ]
                is_dangerous = any(p in command.lower() for p in dangerous_patterns)

                cmd_id = f"cmd-{int(time.time() * 1000)}"
                return JSONResponse(
                    {
                        "command_id": cmd_id,
                        "command": command,
                        "description": intent,
                        "is_dangerous": is_dangerous,
                        "status": "pending",
                    }
                )

            except Exception as e:
                logger.exception("Terminal suggest error")
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.post("/terminal/execute", response_model=TerminalCommandResponse, tags=["Terminal"])
        async def terminal_execute(request: TerminalCommandRequest) -> TerminalCommandResponse:
            """Execute a terminal command."""
            import asyncio.subprocess as asp

            cmd_id = f"cmd-{int(time.time() * 1000)}"
            working_dir = request.working_dir or self.workspace_root

            # Check for dangerous commands
            dangerous_patterns = [
                "rm -rf /",
                "rm -rf ~",
                "> /dev/sd",
                "mkfs.",
                "dd if=",
                "sudo rm",
                ":(){",
                "chmod -R 777",
            ]
            is_dangerous = any(p in request.command.lower() for p in dangerous_patterns)

            if is_dangerous and request.require_approval:
                return TerminalCommandResponse(
                    command_id=cmd_id,
                    command=request.command,
                    status="pending",
                    is_dangerous=True,
                    requires_approval=True,
                )

            try:
                # Execute command asynchronously
                proc = await asyncio.create_subprocess_shell(
                    request.command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=working_dir,
                )

                try:
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=request.timeout)
                    output = stdout.decode("utf-8", errors="replace")
                    exit_code = proc.returncode

                    return TerminalCommandResponse(
                        command_id=cmd_id,
                        command=request.command,
                        status="completed" if exit_code == 0 else "failed",
                        output=output[:50000],  # Limit output size
                        exit_code=exit_code,
                        is_dangerous=is_dangerous,
                        requires_approval=False,
                    )

                except asyncio.TimeoutError:
                    proc.kill()
                    return TerminalCommandResponse(
                        command_id=cmd_id,
                        command=request.command,
                        status="failed",
                        output="Command timed out",
                        exit_code=-1,
                        is_dangerous=is_dangerous,
                        requires_approval=False,
                    )

            except Exception as e:
                logger.exception("Terminal execute error")
                return TerminalCommandResponse(
                    command_id=cmd_id,
                    command=request.command,
                    status="failed",
                    output=str(e),
                    exit_code=-1,
                    is_dangerous=is_dangerous,
                    requires_approval=False,
                )

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

                file_sizes.sort(key=lambda x: int(x.get("lines", 0) or 0), reverse=True)  # type: ignore[call-overload]
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
                        return JSONResponse({"scan_completed": True, "results": tool_result.data})
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
                    dependencies["rust"] = {"file": "Cargo.toml", "exists": True}

                # Go dependencies
                go_mod = root / "go.mod"
                if go_mod.exists():
                    dependencies["go"] = {"file": "go.mod", "exists": True}

                return JSONResponse({"workspace": str(root), "dependencies": dependencies})

            except Exception as e:
                logger.exception("Workspace dependencies error")
                return JSONResponse({"error": str(e)}, status_code=500)

        # MCP endpoints
        @app.get("/mcp/servers", tags=["MCP"])
        async def mcp_servers() -> JSONResponse:
            """Get list of configured MCP servers."""
            try:
                from victor.integrations.mcp.registry import get_mcp_registry

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
                from victor.integrations.mcp.registry import get_mcp_registry

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
                from victor.integrations.mcp.registry import get_mcp_registry

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
                from victor.framework.rl.coordinator import get_rl_coordinator

                coordinator = get_rl_coordinator()
                learner = coordinator.get_learner("model_selector")

                if not learner:
                    return JSONResponse(
                        {"error": "Model selector learner not available"}, status_code=503
                    )

                # Get rankings
                rankings = learner.get_provider_rankings()

                # Build task Q-table summary
                task_q_summary = {}
                for provider, task_q_table in learner._q_table_by_task.items():
                    task_q_summary[provider] = {
                        task_type: round(q_val, 3) for task_type, q_val in task_q_table.items()
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
                from victor.framework.rl.coordinator import get_rl_coordinator
                import json

                coordinator = get_rl_coordinator()
                learner = coordinator.get_learner("model_selector")

                if not learner:
                    return JSONResponse(
                        {"error": "Model selector learner not available"}, status_code=503
                    )

                available = list(learner._q_table.keys()) if learner._q_table else ["ollama"]

                # Get recommendation
                recommendation = coordinator.get_recommendation(
                    "model_selector", json.dumps(available), "", task_type or "unknown"
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
                from victor.framework.rl.coordinator import get_rl_coordinator

                coordinator = get_rl_coordinator()
                learner = coordinator.get_learner("model_selector")

                if not learner:
                    return JSONResponse(
                        {"error": "Model selector learner not available"}, status_code=503
                    )

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
                from victor.framework.rl.coordinator import get_rl_coordinator
                from victor.framework.rl.learners.model_selector import SelectionStrategy

                coordinator = get_rl_coordinator()
                learner = coordinator.get_learner("model_selector")

                if not learner:
                    return JSONResponse(
                        {"error": "Model selector learner not available"}, status_code=503
                    )

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
                from victor.framework.rl.coordinator import get_rl_coordinator

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

        # =====================================================================
        # Background Agent Endpoints
        # =====================================================================

        @app.post("/agents/start", tags=["Agents"])
        async def start_agent(
            task: str = Body(..., description="Task for the agent to execute"),
            mode: str = Body("build", description="Agent mode: build, plan, explore"),
            name: Optional[str] = Body(None, description="Display name for the agent"),
        ) -> JSONResponse:
            """Start a new background agent.

            Agents run asynchronously and report progress via WebSocket events.
            Maximum 4 concurrent agents allowed.
            """
            try:
                from victor.agent.background_agent import get_agent_manager, init_agent_manager

                manager = get_agent_manager()
                if manager is None:
                    # Initialize with orchestrator
                    orchestrator = await self._get_orchestrator()
                    manager = init_agent_manager(
                        orchestrator=orchestrator,
                        max_concurrent=4,
                        event_callback=lambda t, d: asyncio.create_task(
                            self._broadcast_agent_event(t, d)
                        ),
                    )

                agent_id = await manager.start_agent(
                    task=task,
                    mode=mode,
                    name=name,
                )

                return JSONResponse(
                    {
                        "success": True,
                        "agent_id": agent_id,
                        "message": f"Agent started: {agent_id}",
                    }
                )

            except RuntimeError as e:
                return JSONResponse({"error": str(e)}, status_code=429)
            except Exception as e:
                logger.exception("Failed to start agent")
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.get("/agents", tags=["Agents"])
        async def list_agents(
            status: Optional[str] = Query(None, description="Filter by status"),
            limit: int = Query(20, ge=1, le=100),
        ) -> JSONResponse:
            """List background agents."""
            try:
                from victor.agent.background_agent import get_agent_manager, AgentStatus

                manager = get_agent_manager()
                if manager is None:
                    return JSONResponse({"agents": []})

                status_filter = None
                if status:
                    try:
                        status_filter = AgentStatus(status)
                    except ValueError:
                        pass

                agents = manager.list_agents(status=status_filter, limit=limit)
                return JSONResponse(
                    {
                        "agents": agents,
                        "active_count": manager.active_count,
                    }
                )

            except Exception as e:
                logger.exception("Failed to list agents")
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.get("/agents/{agent_id}", tags=["Agents"])
        async def get_agent(agent_id: str) -> JSONResponse:
            """Get a specific agent's status."""
            try:
                from victor.agent.background_agent import get_agent_manager

                manager = get_agent_manager()
                if manager is None:
                    return JSONResponse({"error": "No agents running"}, status_code=404)

                agent_data = manager.get_agent_status(agent_id)
                if agent_data is None:
                    return JSONResponse({"error": "Agent not found"}, status_code=404)

                return JSONResponse(agent_data)

            except Exception as e:
                logger.exception("Failed to get agent")
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.post("/agents/{agent_id}/cancel", tags=["Agents"])
        async def cancel_agent(agent_id: str) -> JSONResponse:
            """Cancel a running agent."""
            try:
                from victor.agent.background_agent import get_agent_manager

                manager = get_agent_manager()
                if manager is None:
                    return JSONResponse({"error": "No agents running"}, status_code=404)

                cancelled = await manager.cancel_agent(agent_id)
                if cancelled:
                    return JSONResponse(
                        {
                            "success": True,
                            "message": f"Agent {agent_id} cancelled",
                        }
                    )
                else:
                    return JSONResponse(
                        {"error": "Agent not found or not running"},
                        status_code=404,
                    )

            except Exception as e:
                logger.exception("Failed to cancel agent")
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.post("/agents/clear", tags=["Agents"])
        async def clear_agents() -> JSONResponse:
            """Clear completed/failed/cancelled agents."""
            try:
                from victor.agent.background_agent import get_agent_manager

                manager = get_agent_manager()
                if manager is None:
                    return JSONResponse({"cleared": 0})

                cleared = manager.clear_completed()
                return JSONResponse(
                    {
                        "success": True,
                        "cleared": cleared,
                        "message": f"Cleared {cleared} agents",
                    }
                )

            except Exception as e:
                logger.exception("Failed to clear agents")
                return JSONResponse({"error": str(e)}, status_code=500)

        # =====================================================================
        # Plan Management Endpoints
        # =====================================================================

        # In-memory plan storage (would use database in production)
        _plans: Dict[str, Dict[str, Any]] = {}

        @app.get("/plans", tags=["Plans"])
        async def list_plans() -> JSONResponse:
            """List all plans."""
            plans_list = []
            for plan_id, plan in _plans.items():
                plans_list.append(
                    {
                        "id": plan_id,
                        "title": plan.get("title", ""),
                        "description": plan.get("description", ""),
                        "status": plan.get("status", "draft"),
                        "created_at": plan.get("created_at"),
                        "approved_at": plan.get("approved_at"),
                        "executed_at": plan.get("executed_at"),
                        "steps": plan.get("steps", []),
                    }
                )
            return JSONResponse({"plans": plans_list})

        @app.post("/plans", tags=["Plans"])
        async def create_plan(request: Request) -> JSONResponse:
            """Create a new plan."""
            try:
                data = await request.json()
                title = data.get("title", "Untitled Plan")
                description = data.get("description", "")
                steps = data.get("steps", [])

                plan_id = str(uuid.uuid4())[:8]
                import time

                _plans[plan_id] = {
                    "id": plan_id,
                    "title": title,
                    "description": description,
                    "status": "draft",
                    "created_at": time.time(),
                    "approved_at": None,
                    "executed_at": None,
                    "completed_at": None,
                    "steps": steps,
                    "current_step": 0,
                    "output": "",
                }

                return JSONResponse(
                    {"id": plan_id, "status": "draft", "message": f"Plan created: {title}"}
                )

            except Exception as e:
                logger.exception("Failed to create plan")
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.get("/plans/{plan_id}", tags=["Plans"])
        async def get_plan(plan_id: str) -> JSONResponse:
            """Get plan details."""
            if plan_id not in _plans:
                return JSONResponse({"error": "Plan not found"}, status_code=404)
            return JSONResponse(_plans[plan_id])

        @app.post("/plans/{plan_id}/approve", tags=["Plans"])
        async def approve_plan(plan_id: str) -> JSONResponse:
            """Approve a plan for execution."""
            if plan_id not in _plans:
                return JSONResponse({"error": "Plan not found"}, status_code=404)

            plan = _plans[plan_id]
            if plan.get("status") != "draft":
                return JSONResponse(
                    {"error": f"Plan is not in draft status (status: {plan.get('status')})"},
                    status_code=400,
                )

            import time

            plan["status"] = "approved"
            plan["approved_at"] = time.time()

            return JSONResponse(
                {"success": True, "message": f"Plan {plan_id} approved", "status": "approved"}
            )

        @app.post("/plans/{plan_id}/execute", tags=["Plans"])
        async def execute_plan(plan_id: str) -> JSONResponse:
            """Execute an approved plan."""
            if plan_id not in _plans:
                return JSONResponse({"error": "Plan not found"}, status_code=404)

            plan = _plans[plan_id]
            if plan.get("status") != "approved":
                return JSONResponse(
                    {
                        "error": f"Plan must be approved before execution (status: {plan.get('status')})"
                    },
                    status_code=400,
                )

            import time

            plan["status"] = "executing"
            plan["executed_at"] = time.time()

            # Execute steps asynchronously (simplified - in production would use background task)
            async def execute_steps():
                try:
                    steps = plan.get("steps", [])
                    for i, step in enumerate(steps):
                        if plan_id not in _plans:
                            break
                        plan["current_step"] = i
                        step_desc = (
                            step.get("description", step) if isinstance(step, dict) else step
                        )
                        plan["output"] += f"\n## Step {i+1}: {step_desc}\n"
                        if isinstance(step, dict):
                            step["status"] = "completed"

                    if plan_id in _plans:
                        plan["status"] = "completed"
                        plan["completed_at"] = time.time()
                except Exception as e:
                    logger.exception(f"Plan {plan_id} execution error")
                    if plan_id in _plans:
                        plan["status"] = "failed"
                        plan["error"] = str(e)

            import asyncio

            asyncio.create_task(execute_steps())

            return JSONResponse(
                {
                    "success": True,
                    "message": f"Plan {plan_id} execution started",
                    "status": "executing",
                }
            )

        @app.delete("/plans/{plan_id}", tags=["Plans"])
        async def delete_plan(plan_id: str) -> JSONResponse:
            """Delete a plan."""
            if plan_id not in _plans:
                return JSONResponse({"error": "Plan not found"}, status_code=404)

            del _plans[plan_id]
            return JSONResponse({"success": True, "message": f"Plan {plan_id} deleted"})

        # =====================================================================
        # Teams API Endpoints
        # =====================================================================

        # In-memory team storage
        _teams: Dict[str, Dict[str, Any]] = {}
        _team_messages: Dict[str, List[Dict[str, Any]]] = {}

        @app.post("/teams", tags=["Teams"])
        async def create_team(request: Request) -> JSONResponse:
            """Create a new agent team."""
            try:
                data = await request.json()
                name = data.get("name", "Unnamed Team")
                goal = data.get("goal", "")
                formation = data.get("formation", "sequential")
                members = data.get("members", [])
                total_tool_budget = data.get("total_tool_budget", 100)

                team_id = str(uuid.uuid4())[:8]

                # Build member list with default values
                team_members = []
                for i, m in enumerate(members):
                    member = {
                        "id": m.get("id", f"member_{i+1}"),
                        "role": m.get("role", "executor"),
                        "name": m.get("name", f"Agent {i+1}"),
                        "goal": m.get("goal", ""),
                        "status": "pending",
                        "tool_budget": total_tool_budget // len(members) if members else 0,
                        "tools_used": 0,
                        "discoveries": [],
                        "is_manager": m.get("is_manager", False),
                    }
                    team_members.append(member)

                # Set first member as manager for hierarchical formation
                if formation == "hierarchical" and team_members:
                    team_members[0]["is_manager"] = True

                _teams[team_id] = {
                    "id": team_id,
                    "name": name,
                    "goal": goal,
                    "formation": formation,
                    "status": "draft",
                    "members": team_members,
                    "total_tool_budget": total_tool_budget,
                    "total_tools_used": 0,
                    "start_time": None,
                    "end_time": None,
                    "current_step": None,
                    "output": None,
                    "error": None,
                }
                _team_messages[team_id] = []

                # Broadcast team created event
                await self._broadcast_ws(
                    {
                        "type": "agent_event",
                        "event": "team_created",
                        "data": _teams[team_id],
                        "timestamp": time.time(),
                    }
                )

                return JSONResponse(
                    {
                        "success": True,
                        "team_id": team_id,
                        "message": f"Team '{name}' created",
                    }
                )

            except Exception as e:
                logger.exception("Failed to create team")
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.get("/teams", tags=["Teams"])
        async def list_teams(
            status: Optional[str] = Query(None, description="Filter by status"),
        ) -> JSONResponse:
            """List all teams."""
            teams_list = []
            for team in _teams.values():
                if status is None or team.get("status") == status:
                    teams_list.append(team)
            return JSONResponse({"teams": teams_list})

        @app.get("/teams/{team_id}", tags=["Teams"])
        async def get_team(team_id: str) -> JSONResponse:
            """Get team details."""
            if team_id not in _teams:
                return JSONResponse({"error": "Team not found"}, status_code=404)
            return JSONResponse(_teams[team_id])

        @app.post("/teams/{team_id}/start", tags=["Teams"])
        async def start_team(team_id: str) -> JSONResponse:
            """Start team execution."""
            if team_id not in _teams:
                return JSONResponse({"error": "Team not found"}, status_code=404)

            team = _teams[team_id]
            if team["status"] not in ("draft", "paused"):
                return JSONResponse(
                    {"error": f"Cannot start team in status: {team['status']}"},
                    status_code=400,
                )

            team["status"] = "running"
            team["start_time"] = time.time()

            # Set all members to pending
            for member in team["members"]:
                member["status"] = "pending"

            # Broadcast team started event
            await self._broadcast_ws(
                {
                    "type": "agent_event",
                    "event": "team_started",
                    "data": team,
                    "timestamp": time.time(),
                }
            )

            # Start team execution in background
            async def execute_team():
                try:
                    from victor.teams import (
                        TeamConfig,
                        TeamMember,
                        TeamFormation,
                        TeamCoordinator,
                    )
                    from victor.agent.subagents import SubAgentRole

                    orchestrator = await self._get_orchestrator()

                    # Build TeamConfig
                    role_map = {
                        "researcher": SubAgentRole.RESEARCHER,
                        "planner": SubAgentRole.PLANNER,
                        "executor": SubAgentRole.EXECUTOR,
                        "reviewer": SubAgentRole.REVIEWER,
                        "tester": SubAgentRole.TESTER,
                    }
                    formation_map = {
                        "sequential": TeamFormation.SEQUENTIAL,
                        "parallel": TeamFormation.PARALLEL,
                        "hierarchical": TeamFormation.HIERARCHICAL,
                        "pipeline": TeamFormation.PIPELINE,
                    }

                    members = []
                    for m in team["members"]:
                        role = role_map.get(m["role"], SubAgentRole.EXECUTOR)
                        members.append(
                            TeamMember(
                                id=m["id"],
                                role=role,
                                name=m["name"],
                                goal=m["goal"],
                                tool_budget=m["tool_budget"],
                                is_manager=m.get("is_manager", False),
                            )
                        )

                    config = TeamConfig(
                        name=team["name"],
                        goal=team["goal"],
                        members=members,
                        formation=formation_map.get(team["formation"], TeamFormation.SEQUENTIAL),
                        total_tool_budget=team["total_tool_budget"],
                    )

                    coordinator = TeamCoordinator(orchestrator)
                    result = await coordinator.execute_team(config)

                    # Update team status
                    if team_id in _teams:
                        team["status"] = "completed" if result.success else "failed"
                        team["end_time"] = time.time()
                        team["output"] = result.final_output
                        team["total_tools_used"] = result.total_tool_calls

                        # Update member results
                        for member in team["members"]:
                            if member["id"] in result.member_results:
                                mr = result.member_results[member["id"]]
                                member["status"] = "completed" if mr.success else "failed"
                                member["tools_used"] = mr.tool_calls_used
                                member["discoveries"] = mr.discoveries

                        # Broadcast completion
                        await self._broadcast_ws(
                            {
                                "type": "agent_event",
                                "event": "team_completed" if result.success else "team_failed",
                                "data": team,
                                "timestamp": time.time(),
                            }
                        )

                except Exception as e:
                    logger.exception(f"Team {team_id} execution error")
                    if team_id in _teams:
                        team["status"] = "failed"
                        team["error"] = str(e)
                        team["end_time"] = time.time()

                        await self._broadcast_ws(
                            {
                                "type": "agent_event",
                                "event": "team_failed",
                                "data": team,
                                "timestamp": time.time(),
                            }
                        )

            asyncio.create_task(execute_team())

            return JSONResponse(
                {
                    "success": True,
                    "message": f"Team {team_id} started",
                }
            )

        @app.post("/teams/{team_id}/cancel", tags=["Teams"])
        async def cancel_team(team_id: str) -> JSONResponse:
            """Cancel a running team."""
            if team_id not in _teams:
                return JSONResponse({"error": "Team not found"}, status_code=404)

            team = _teams[team_id]
            if team["status"] != "running":
                return JSONResponse(
                    {"error": f"Cannot cancel team in status: {team['status']}"},
                    status_code=400,
                )

            team["status"] = "cancelled"
            team["end_time"] = time.time()

            # Broadcast cancellation
            await self._broadcast_ws(
                {
                    "type": "agent_event",
                    "event": "team_cancelled",
                    "data": team,
                    "timestamp": time.time(),
                }
            )

            return JSONResponse(
                {
                    "success": True,
                    "message": f"Team {team_id} cancelled",
                }
            )

        @app.post("/teams/clear", tags=["Teams"])
        async def clear_teams() -> JSONResponse:
            """Clear completed/failed/cancelled teams."""
            cleared = 0
            to_delete = []
            for team_id, team in _teams.items():
                if team["status"] in ("completed", "failed", "cancelled"):
                    to_delete.append(team_id)

            for team_id in to_delete:
                del _teams[team_id]
                if team_id in _team_messages:
                    del _team_messages[team_id]
                cleared += 1

            return JSONResponse(
                {
                    "success": True,
                    "cleared": cleared,
                }
            )

        @app.get("/teams/{team_id}/messages", tags=["Teams"])
        async def get_team_messages(team_id: str) -> JSONResponse:
            """Get team inter-agent messages."""
            if team_id not in _teams:
                return JSONResponse({"error": "Team not found"}, status_code=404)

            messages = _team_messages.get(team_id, [])
            return JSONResponse({"messages": messages})

        # =====================================================================
        # Workflows API Endpoints
        # =====================================================================

        # In-memory workflow storage
        _workflow_executions: Dict[str, Dict[str, Any]] = {}

        @app.get("/workflows/templates", tags=["Workflows"])
        async def list_workflow_templates() -> JSONResponse:
            """List available workflow templates."""
            try:
                from victor.workflows.registry import get_global_registry

                registry = get_global_registry()
                templates = []

                for workflow_id, workflow_def in registry._definitions.items():
                    templates.append(
                        {
                            "id": workflow_id,
                            "name": workflow_def.name,
                            "description": workflow_def.description or "",
                            "category": workflow_def.metadata.get("category", "General"),
                            "steps": [
                                {
                                    "id": node.id,
                                    "name": node.name or node.id,
                                    "type": node.type.value,
                                    "role": getattr(node, "role", None),
                                    "goal": getattr(node, "goal", None),
                                }
                                for node in workflow_def.nodes.values()
                            ],
                            "tags": workflow_def.metadata.get("tags", []),
                            "estimated_duration": workflow_def.metadata.get("estimated_duration"),
                        }
                    )

                return JSONResponse({"templates": templates})

            except ImportError:
                # Workflows module not available - return empty list
                return JSONResponse({"templates": []})
            except Exception as e:
                logger.exception("Failed to list workflow templates")
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.get("/workflows/templates/{template_id}", tags=["Workflows"])
        async def get_workflow_template(template_id: str) -> JSONResponse:
            """Get workflow template details."""
            try:
                from victor.workflows.registry import get_global_registry

                registry = get_global_registry()
                workflow_def = registry.get(template_id)

                if workflow_def is None:
                    return JSONResponse({"error": "Template not found"}, status_code=404)

                return JSONResponse(
                    {
                        "id": template_id,
                        "name": workflow_def.name,
                        "description": workflow_def.description or "",
                        "category": workflow_def.metadata.get("category", "General"),
                        "steps": [
                            {
                                "id": node.id,
                                "name": node.name or node.id,
                                "type": node.type.value,
                                "role": getattr(node, "role", None),
                                "goal": getattr(node, "goal", None),
                            }
                            for node in workflow_def.nodes.values()
                        ],
                        "tags": workflow_def.metadata.get("tags", []),
                    }
                )

            except ImportError:
                return JSONResponse({"error": "Workflows module not available"}, status_code=404)
            except Exception as e:
                logger.exception("Failed to get workflow template")
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.post("/workflows/execute", tags=["Workflows"])
        async def execute_workflow(request: Request) -> JSONResponse:
            """Execute a workflow."""
            try:
                data = await request.json()
                template_id = data.get("template_id")
                parameters = data.get("parameters", {})

                if not template_id:
                    return JSONResponse({"error": "template_id required"}, status_code=400)

                from victor.workflows import get_workflow_registry, WorkflowExecutor

                registry = get_workflow_registry()
                workflow_def = registry.get(template_id)

                if workflow_def is None:
                    return JSONResponse({"error": "Template not found"}, status_code=404)

                execution_id = str(uuid.uuid4())[:8]

                # Initialize execution state
                _workflow_executions[execution_id] = {
                    "id": execution_id,
                    "workflow_id": template_id,
                    "workflow_name": workflow_def.name,
                    "status": "running",
                    "parameters": parameters,
                    "current_step": None,
                    "progress": 0,
                    "start_time": time.time(),
                    "end_time": None,
                    "steps": [
                        {
                            "id": node.id,
                            "name": node.name or node.id,
                            "type": node.type.value,
                            "status": "pending",
                        }
                        for node in workflow_def.nodes.values()
                    ],
                    "output": None,
                    "error": None,
                }

                # Broadcast workflow started
                await self._broadcast_ws(
                    {
                        "type": "agent_event",
                        "event": "workflow_started",
                        "data": _workflow_executions[execution_id],
                        "timestamp": time.time(),
                    }
                )

                # Execute in background
                async def run_workflow():
                    try:
                        orchestrator = await self._get_orchestrator()
                        executor = WorkflowExecutor(orchestrator)

                        result = await executor.execute(
                            workflow_def,
                            initial_context=parameters,
                        )

                        if execution_id in _workflow_executions:
                            exec_state = _workflow_executions[execution_id]
                            exec_state["status"] = "completed" if result.success else "failed"
                            exec_state["end_time"] = time.time()
                            exec_state["progress"] = 100
                            exec_state["output"] = (
                                str(result.final_output) if result.final_output else None
                            )

                            # Update step statuses
                            for step in exec_state["steps"]:
                                node_result = result.node_results.get(step["id"])
                                if node_result:
                                    step["status"] = (
                                        "completed" if node_result.success else "failed"
                                    )
                                    step["duration"] = (
                                        node_result.duration_ms / 1000
                                        if node_result.duration_ms
                                        else None
                                    )

                            # Broadcast completion
                            await self._broadcast_ws(
                                {
                                    "type": "agent_event",
                                    "event": (
                                        "workflow_completed"
                                        if result.success
                                        else "workflow_failed"
                                    ),
                                    "data": exec_state,
                                    "timestamp": time.time(),
                                }
                            )

                    except Exception as e:
                        logger.exception(f"Workflow {execution_id} error")
                        if execution_id in _workflow_executions:
                            exec_state = _workflow_executions[execution_id]
                            exec_state["status"] = "failed"
                            exec_state["error"] = str(e)
                            exec_state["end_time"] = time.time()

                            await self._broadcast_ws(
                                {
                                    "type": "agent_event",
                                    "event": "workflow_failed",
                                    "data": exec_state,
                                    "timestamp": time.time(),
                                }
                            )

                asyncio.create_task(run_workflow())

                return JSONResponse(
                    {
                        "success": True,
                        "execution_id": execution_id,
                        "message": f"Workflow '{workflow_def.name}' started",
                    }
                )

            except ImportError:
                return JSONResponse({"error": "Workflows module not available"}, status_code=500)
            except Exception as e:
                logger.exception("Failed to execute workflow")
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.get("/workflows/executions", tags=["Workflows"])
        async def list_workflow_executions(
            status: Optional[str] = Query(None, description="Filter by status"),
        ) -> JSONResponse:
            """List workflow executions."""
            executions = []
            for exec_state in _workflow_executions.values():
                if status is None or exec_state.get("status") == status:
                    executions.append(exec_state)
            return JSONResponse({"executions": executions})

        @app.get("/workflows/executions/{execution_id}", tags=["Workflows"])
        async def get_workflow_execution(execution_id: str) -> JSONResponse:
            """Get workflow execution details."""
            if execution_id not in _workflow_executions:
                return JSONResponse({"error": "Execution not found"}, status_code=404)
            return JSONResponse(_workflow_executions[execution_id])

        @app.post("/workflows/executions/{execution_id}/cancel", tags=["Workflows"])
        async def cancel_workflow_execution(execution_id: str) -> JSONResponse:
            """Cancel a workflow execution."""
            if execution_id not in _workflow_executions:
                return JSONResponse({"error": "Execution not found"}, status_code=404)

            exec_state = _workflow_executions[execution_id]
            if exec_state["status"] != "running":
                return JSONResponse(
                    {"error": f"Cannot cancel execution in status: {exec_state['status']}"},
                    status_code=400,
                )

            exec_state["status"] = "cancelled"
            exec_state["end_time"] = time.time()

            # Broadcast cancellation
            await self._broadcast_ws(
                {
                    "type": "agent_event",
                    "event": "workflow_cancelled",
                    "data": exec_state,
                    "timestamp": time.time(),
                }
            )

            return JSONResponse(
                {
                    "success": True,
                    "message": f"Workflow execution {execution_id} cancelled",
                }
            )

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
            """Shutdown the server gracefully.

            This endpoint initiates a graceful shutdown of the server without
            forcefully stopping the event loop, which prevents system-wide issues.
            """
            # Check if shutdown is already in progress
            if self._shutting_down:
                return JSONResponse({"status": "already_shutting_down"})

            self._shutting_down = True
            logger.info("Shutdown requested")
            await self._record_rl_feedback()

            # Close all WebSocket connections
            for ws in self._ws_clients:
                try:
                    await ws.close()
                except Exception:
                    pass

            # Use server's built-in shutdown mechanism instead of loop.stop()
            # This prevents event loop corruption and system-wide issues
            if hasattr(self, "_server") and self._server is not None:
                self._server.should_exit = True
                # Schedule actual shutdown after response is sent
                asyncio.create_task(self._delayed_shutdown())

            return JSONResponse({"status": "shutting_down"})

        # =============================================================================
        # Workflow Visualization Endpoints
        # =============================================================================

        @app.get("/workflows/{workflow_id}/graph", tags=["Workflows"])
        async def get_workflow_graph(workflow_id: str) -> JSONResponse:
            """Get static workflow graph structure for visualization.

            Returns the workflow DAG structure (nodes + edges) in Cytoscape.js format.
            This endpoint provides the initial graph structure for rendering.

            Args:
                workflow_id: Workflow identifier

            Returns:
                Graph schema with nodes and edges in Cytoscape.js format

            Raises:
                HTTPException 404: If workflow not found
                HTTPException 500: If graph export fails
            """
            try:
                # Try to get from execution store first
                if workflow_id in self._workflow_executions:
                    exec_data = self._workflow_executions[workflow_id]
                    if "graph" in exec_data:
                        return JSONResponse(exec_data["graph"])

                # If not in execution store, return 404
                # In production, you might load from workflow registry
                raise HTTPException(
                    status_code=404,
                    detail=f"Workflow {workflow_id} not found. Execute the workflow first.",
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get graph for workflow {workflow_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to export graph: {str(e)}")

        @app.get("/workflows/{workflow_id}/execution", tags=["Workflows"])
        async def get_workflow_execution_status(workflow_id: str) -> JSONResponse:
            """Get current workflow execution status.

            Returns detailed execution state including which nodes have executed,
            current progress, timing metrics, and tool/token usage.

            Args:
                workflow_id: Workflow identifier

            Returns:
                WorkflowExecutionState with detailed execution information

            Raises:
                HTTPException 404: If workflow execution not found
            """
            try:
                if workflow_id not in self._workflow_executions:
                    raise HTTPException(
                        status_code=404, detail=f"Workflow execution {workflow_id} not found"
                    )

                # Get execution state
                exec_state = get_execution_state(workflow_id, self._workflow_executions)

                return JSONResponse(exec_state.to_dict())

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get execution state for {workflow_id}: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to get execution state: {str(e)}"
                )

        @app.websocket("/workflows/{workflow_id}/stream")
        async def workflow_websocket_stream(websocket: WebSocket, workflow_id: str) -> None:
            """Real-time workflow execution event stream via WebSocket.

            Provides live updates for workflow execution including:
            - Node start/complete/error events
            - Progress updates
            - Workflow completion

            Args:
                websocket: WebSocket connection
                workflow_id: Workflow identifier to stream events for

            Example client usage:
                const ws = new WebSocket(`ws://localhost:8000/workflows/wf_123/stream`);
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    console.log('Event:', data);
                };
            """
            await websocket.accept()

            if not self._workflow_event_bridge:
                await websocket.close(code=1011, reason="Workflow event bridge not initialized")
                return

            try:
                # Handle connection through event bridge
                await self._workflow_event_bridge.handle_websocket_connection(
                    websocket, workflow_id, client_id=uuid.uuid4().hex[:12]
                )
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for workflow {workflow_id}")
            except Exception as e:
                logger.error(f"WebSocket error for workflow {workflow_id}: {e}")
                try:
                    await websocket.close(code=1011, reason=str(e))
                except Exception:
                    pass

        @app.get(
            "/workflows/visualize/{workflow_id}", response_class=HTMLResponse, tags=["Workflows"]
        )
        async def visualize_workflow(workflow_id: str) -> HTMLResponse:
            """Serve HTML page with interactive workflow visualization.

            Returns a single-page HTML application with Cytoscape.js for
            interactive workflow graph visualization with real-time updates.

            Args:
                workflow_id: Workflow identifier to visualize

            Returns:
                HTML page with embedded Cytoscape.js visualization

            Example:
                Open in browser: http://localhost:8000/workflows/visualize/wf_123
            """
            try:
                # Read template
                template_path = Path(__file__).parent / "templates" / "workflow_visualizer.html"

                if not template_path.exists():
                    # Fallback: Return simple HTML if template not found
                    html_content = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Victor Workflow Visualization</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 40px; }}
                            .error {{ color: red; }}
                        </style>
                    </head>
                    <body>
                        <h1>Workflow Visualization Not Available</h1>
                        <p class="error">Template file not found: {template_path}</p>
                        <p>The workflow visualization feature requires the template file to be installed.</p>
                    </body>
                    </html>
                    """
                    return HTMLResponse(content=html_content, status_code=200)

                with open(template_path, "r", encoding="utf-8") as f:
                    template_html = f.read()

                # In production, you might use Jinja2 templates for variable substitution
                # For MVP, we're using client-side JS to fetch data
                return HTMLResponse(content=template_html)

            except Exception as e:
                logger.error(f"Failed to serve visualization for {workflow_id}: {e}")
                return HTMLResponse(
                    content=f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>",
                    status_code=500,
                )

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
                logger.info(f"WebSocket client disconnected. Total: {len(self._ws_clients)}")

        # EventBridge WebSocket endpoint for real-time events
        @app.websocket("/ws/events")
        async def events_websocket_handler(websocket: WebSocket) -> None:
            """Handle EventBridge WebSocket connections for real-time events.

            This endpoint streams Victor events (tool execution, file changes,
            provider updates, etc.) to connected clients like VS Code.

            Message format:
                Incoming: {"type": "subscribe", "categories": ["all"]}
                Outgoing: {"type": "event", "event": {...}}
            """
            await websocket.accept()
            self._event_clients.append(websocket)
            client_id = uuid.uuid4().hex[:12]
            logger.info(
                f"EventBridge client {client_id} connected. Total: {len(self._event_clients)}"
            )

            # Register with EventBridge for event forwarding
            async def send_event(message: str) -> None:
                try:
                    await websocket.send_text(message)
                except Exception:
                    pass

            if self._event_bridge:
                self._event_bridge._broadcaster.add_client(client_id, send_event)

            try:
                while True:
                    data = await websocket.receive_json()
                    msg_type = data.get("type", "")

                    if msg_type == "subscribe":
                        # Client wants to subscribe to specific event categories
                        categories = data.get("categories", ["all"])
                        logger.debug(f"Client {client_id} subscribed to: {categories}")
                        # Send acknowledgment
                        await websocket.send_json({"type": "subscribed", "categories": categories})

                    elif msg_type == "ping":
                        await websocket.send_json({"type": "pong"})

            except WebSocketDisconnect:
                pass
            finally:
                if self._event_bridge:
                    self._event_bridge._broadcaster.remove_client(client_id)
                if websocket in self._event_clients:
                    self._event_clients.remove(websocket)
                logger.info(
                    f"EventBridge client {client_id} disconnected. "
                    f"Total: {len(self._event_clients)}"
                )

    # =========================================================================
    # HITL (Human-in-the-Loop) Routes
    # =========================================================================

    def _setup_hitl_routes(self) -> None:
        """Set up HITL (Human-in-the-Loop) endpoints for workflow approvals.

        This enables the unified server to handle both VS Code/IDE requests
        and workflow approval requests on the same port.

        Endpoints added:
            GET /hitl/requests - List pending approval requests
            GET /hitl/requests/{id} - Get specific request
            POST /hitl/respond/{id} - Submit approval/rejection
            GET /hitl/ui - Web-based approval UI
            GET /hitl/history - Approval history (SQLite only)
        """
        try:
            from victor.workflows.hitl_api import (
                HITLStore,
                SQLiteHITLStore,
                create_hitl_router,
            )

            # Create HITL store - SQLite for persistence, in-memory otherwise
            if self.hitl_persistent:
                self._hitl_store = SQLiteHITLStore()
                if hasattr(self._hitl_store, "db_path"):
                    logger.info(f"HITL using SQLite store: {self._hitl_store.db_path}")
                else:
                    logger.info("HITL using SQLite store")
            else:
                self._hitl_store = HITLStore()
                logger.info("HITL using in-memory store")

            # Create and include the HITL router
            hitl_router = create_hitl_router(
                store=self._hitl_store,
                require_auth=bool(self.hitl_auth_token),
                auth_token=self.hitl_auth_token,
            )
            self.app.include_router(hitl_router, prefix="/hitl")

            logger.info("HITL endpoints enabled at /hitl/*")

        except ImportError as e:
            logger.warning(f"HITL endpoints not available: {e}")

    def get_hitl_store(self) -> Optional[Any]:
        """Get the HITL store for this server instance.

        Returns:
            HITLStore if HITL is enabled, None otherwise
        """
        return self._hitl_store

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
            from victor.framework.rl.base import RLOutcome
            from victor.framework.rl.coordinator import get_rl_coordinator

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

            metrics: Dict[str, Any] = {}
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
            provider_ranking = next((r for r in rankings if r["provider"] == provider.name), None)
            new_q = provider_ranking["q_value"] if provider_ranking else 0.0

            logger.info(
                f"RL API session feedback: {provider.name} "
                f"({msg_count} messages, {tool_calls} tools) → Q={new_q:.3f}"
            )

        except Exception as e:
            logger.debug(f"RL feedback recording skipped: {e}")

    async def _handle_ws_message(self, ws: WebSocket, data: Dict[str, Any]) -> None:
        """Handle incoming WebSocket messages."""
        msg_type = data.get("type", "")

        if msg_type == "chat":
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

        elif msg_type == "auth":
            # Handle WebSocket authentication (more secure than URL query params)
            # API key is sent in first message after connection instead of URL
            api_key = data.get("api_key", "")
            if api_key:
                # Store auth state on the websocket scope for future operations
                if not hasattr(ws, "state"):
                    ws.state = type("State", (), {})()
                ws.state.authenticated = True
                ws.state.api_key = api_key
                logger.debug("WebSocket client authenticated via message")
                await ws.send_json({"type": "auth_success"})
            else:
                await ws.send_json({"type": "auth_failed", "message": "No API key provided"})

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

    async def _broadcast_agent_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast agent events to all connected WebSocket clients.

        Used by BackgroundAgentManager to send real-time updates.
        """
        message = {
            "type": "agent_event",
            "event": event_type,
            "data": data,
            "timestamp": time.time(),
        }
        await self._broadcast_ws(message)

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

        # Start the server in a task
        task = asyncio.create_task(self._server.serve())

        # Wait for the server to actually start serving
        # The server is started when the serve() task begins execution
        await asyncio.sleep(0.2)

        # Verify the task is still running (no startup errors)
        if not task.done():
            logger.info(f"Victor FastAPI server running on {self.host}:{self.port}")
        else:
            # If the task completed already, it likely failed
            try:
                task.result()  # This will raise the exception if one occurred
            except Exception as e:
                logger.error(f"Failed to start server: {e}")
                raise

        return self

    async def shutdown(self) -> None:
        """Shutdown the server gracefully.

        This uses uvicorn's built-in shutdown mechanism to properly terminate
        the server without affecting the asyncio event loop or causing system issues.
        """
        if hasattr(self, "_server") and self._server is not None:
            # Signal the server to exit gracefully
            self._server.should_exit = True

            # Use uvicorn's built-in shutdown if available
            if hasattr(self._server, "shutdown"):
                try:
                    await asyncio.wait_for(self._server.shutdown(), timeout=3.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    # Timeout is acceptable - server will exit on its own
                    pass
                except Exception as e:
                    logger.warning(f"Server shutdown encountered error: {e}")

            # Small delay to ensure port is released (not for event loop stability)
            await asyncio.sleep(0.1)

    async def _delayed_shutdown(self) -> None:
        """Delayed shutdown helper for /shutdown endpoint.

        This method is called asynchronously after the shutdown response is sent,
        allowing the HTTP response to complete before the server actually shuts down.
        """
        # Check if shutdown is already in progress
        if self._shutting_down:
            return

        # Small delay to ensure response is sent
        await asyncio.sleep(0.5)

        # Perform actual shutdown
        if hasattr(self, "_server") and self._server is not None:
            if hasattr(self._server, "shutdown"):
                try:
                    await asyncio.wait_for(self._server.shutdown(), timeout=3.0)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as e:
                    logger.warning(f"Delayed shutdown encountered error: {e}")


def create_fastapi_app(workspace_root: Optional[str] = None) -> FastAPI:
    """Create the FastAPI application."""
    server = VictorFastAPIServer(workspace_root=workspace_root)
    return server.app
