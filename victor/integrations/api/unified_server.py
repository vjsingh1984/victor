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

"""Unified Victor API Server - Simplified Version.

This is a cleaner implementation that properly integrates all three servers
by including their routers with appropriate prefixes.

Server Structure:
    /api/v1/*              - Main API (chat, completions, search)
    /api/v1/hitl/*         - HITL approval endpoints
    /api/v1/workflows/*    - Workflow editor endpoints
    /ui                    - Landing page
    /ui/hitl               - HITL approval UI
    /ui/workflow-editor    - Workflow editor frontend
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

logger = logging.getLogger(__name__)


# =============================================================================
# UI Templates
# =============================================================================


LANDING_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Victor AI - Unified UI Portal</title>
    <style>
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --primary-light: #818cf8;
            --success: #10b981;
            --bg: #0f172a;
            --bg-card: #1e293b;
            --text: #f1f5f9;
            --text-secondary: #cbd5e1;
            --border: #334155;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Inter', sans-serif;
            background: linear-gradient(135deg, var(--bg) 0%, #1a1f35 100%);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .navbar {
            background: rgba(30, 41, 59, 0.8);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border);
            padding: 1rem 2rem;
        }

        .nav-content {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .logo {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--primary), var(--primary-light));
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1.5rem;
            color: white;
        }

        .logo-text {
            font-size: 1.25rem;
            font-weight: 600;
        }

        .logo-text span {
            color: var(--text-secondary);
            font-weight: 400;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 3rem 2rem;
            flex: 1;
        }

        h1 {
            font-size: 2.5rem;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, var(--primary), var(--primary-light));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .subtitle {
            font-size: 1.125rem;
            color: var(--text-secondary);
            margin-bottom: 3rem;
        }

        .ui-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
        }

        .ui-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 2rem;
            transition: all 0.3s;
            cursor: pointer;
            text-decoration: none;
            color: inherit;
            display: block;
        }

        .ui-card:hover {
            border-color: var(--primary);
            transform: translateY(-4px);
            box-shadow: 0 10px 40px rgba(99, 102, 241, 0.2);
        }

        .ui-card-icon {
            width: 60px;
            height: 60px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            margin-bottom: 1.5rem;
        }

        .ui-card-icon.workflow {
            background: rgba(99, 102, 241, 0.2);
        }

        .ui-card-icon.hitl {
            background: rgba(16, 185, 129, 0.2);
        }

        .ui-card-icon.docs {
            background: rgba(245, 158, 11, 0.2);
        }

        .ui-card-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
        }

        .ui-card-description {
            color: var(--text-secondary);
            line-height: 1.6;
            margin-bottom: 1.5rem;
        }

        .ui-card-link {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--primary-light);
            font-weight: 500;
        }

        .ui-card-link::after {
            content: '→';
            transition: transform 0.2s;
        }

        .ui-card:hover .ui-card-link::after {
            transform: translateX(4px);
        }

        .status-bar {
            background: var(--bg-card);
            border-top: 1px solid var(--border);
            padding: 1.5rem 2rem;
            margin-top: auto;
        }

        .status-content {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 1rem;
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--success);
            font-size: 0.875rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--success);
            border-radius: 50%;
        }

        .status-links {
            display: flex;
            gap: 2rem;
            font-size: 0.875rem;
        }

        .status-links a {
            color: var(--text-secondary);
            text-decoration: none;
            transition: color 0.2s;
        }

        .status-links a:hover {
            color: var(--primary-light);
        }

        @media (max-width: 768px) {
            h1 {
                font-size: 2rem;
            }

            .ui-grid {
                grid-template-columns: 1fr;
            }

            .status-content {
                flex-direction: column;
                text-align: center;
            }
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="nav-content">
            <div class="logo">V</div>
            <div class="logo-text">Victor <span>Unified Server</span></div>
        </div>
    </nav>

    <div class="container">
        <h1>Victor AI - Unified UI Portal</h1>
        <p class="subtitle">Access all Victor AI interfaces from a single location</p>

        <div class="ui-grid">
            <a href="/ui/workflow-editor" class="ui-card">
                <div class="ui-card-icon workflow">⚙️</div>
                <h2 class="ui-card-title">Workflow Editor</h2>
                <p class="ui-card-description">
                    Visual workflow editor for creating and managing StateGraph workflows.
                    Design, validate, and execute complex multi-step workflows.
                </p>
                <span class="ui-card-link">Open Editor</span>
            </a>

            <a href="/ui/hitl" class="ui-card">
                <div class="ui-card-icon hitl">✓</div>
                <h2 class="ui-card-title">Approvals</h2>
                <p class="ui-card-description">
                    Human-in-the-Loop approval interface for workflow decisions.
                    Review, approve, or reject pending workflow requests.
                </p>
                <span class="ui-card-link">Open Approvals</span>
            </a>

            <a href="/docs" class="ui-card">
                <div class="ui-card-icon docs">📚</div>
                <h2 class="ui-card-title">API Documentation</h2>
                <p class="ui-card-description">
                    Interactive API documentation with OpenAPI/Swagger.
                    Explore all available endpoints and test them directly.
                </p>
                <span class="ui-card-link">View Docs</span>
            </a>
        </div>
    </div>

    <div class="status-bar">
        <div class="status-content">
            <div class="status-indicator">
                <span class="status-dot"></span>
                All Systems Operational
            </div>
            <div class="status-links">
                <a href="/health">Health Check</a>
                <a href="/docs">API Docs</a>
                <a href="https://github.com/vijaydsingh/victor" target="_blank">GitHub</a>
            </div>
        </div>
    </div>
</body>
</html>
"""


# =============================================================================
# Unified Server Factory
# =============================================================================


def create_unified_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workspace_root: Optional[str] = None,
    enable_hitl: bool = True,
    hitl_persistent: bool = True,
    hitl_auth_token: Optional[str] = None,
    enable_cors: bool = True,
) -> FastAPI:
    """Create the unified Victor API server.

    This server consolidates all Victor backend services into a single
    FastAPI application with proper URL routing and CORS configuration.

    Args:
        host: Host to bind to
        port: Port to listen on
        workspace_root: Root directory of the workspace
        enable_hitl: Enable HITL (Human-in-the-Loop) endpoints (default: True)
        hitl_persistent: Use SQLite for persistent HITL storage (default: True)
        hitl_auth_token: Optional auth token for HITL endpoints
        enable_cors: Enable CORS headers (default: True)

    Returns:
        Configured FastAPI application
    """
    # Create FastAPI app
    app = FastAPI(
        title="Victor Unified API",
        description="Consolidated API for all Victor services",
        version="0.5.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # For development
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Store configuration
    app.state.host = host
    app.state.port = port
    app.state.workspace_root = workspace_root or str(Path.cwd())

    # Include main API routes
    _include_main_api(app, workspace_root)

    # Include HITL routes
    if enable_hitl:
        _include_hitl_api(app, hitl_persistent, hitl_auth_token)

    # Include workflow editor routes
    _include_workflow_editor_api(app)

    # Setup frontend routes
    _setup_frontend_routes(app)

    # Setup health check
    _setup_health_check(app)

    logger.info("Unified Victor API server created successfully")
    return app


def _include_main_api(app: FastAPI, workspace_root: Optional[str]) -> None:
    """Include main API server routes.

    The main API server provides:
    - Chat and completion endpoints
    - Code search (semantic and regex)
    - WebSocket streaming
    - Git integration
    - LSP services
    - Terminal commands
    - And many more...

    All routes are prefixed with /api/v1
    """
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

    # Create the main API server instance
    main_server = VictorFastAPIServer(
        host="0.0.0.0",
        port=8000,
        workspace_root=workspace_root,
        enable_hitl=False,  # We handle HITL separately
        enable_cors=False,  # Already configured
    )

    # Store reference for cleanup
    app.state.main_server = main_server

    # Mount the main API server at /api/v1
    # This preserves all existing routes without modification
    app.mount("/api/v1", main_server.app, name="main_api")

    logger.info("Main API routes mounted at /api/v1")


def _include_hitl_api(
    app: FastAPI,
    persistent: bool = True,
    auth_token: Optional[str] = None,
) -> None:
    """Include HITL (Human-in-the-Loop) API routes.

    The HITL API provides:
    - Approval request management
    - Response submission
    - Request history
    - WebSocket notifications

    All routes are prefixed with /api/v1/hitl
    """
    try:
        from victor.workflows.hitl_api import (
            HITLStore,
            SQLiteHITLStore,
            create_hitl_router,
        )

        # Create HITL store
        if persistent:
            hitl_store = SQLiteHITLStore()  # type: ignore[assignment]
            logger.info(f"HITL using SQLite store: {hitl_store.db_path}")
        else:
            hitl_store = HITLStore()
            logger.info("HITL using in-memory store")

        app.state.hitl_store = hitl_store  # type: ignore[assignment]

        # Create HITL router
        hitl_router = create_hitl_router(
            store=hitl_store,  # type: ignore[arg-type]
            require_auth=bool(auth_token),
            auth_token=auth_token,
        )

        # Include HITL router at /api/v1/hitl
        app.include_router(hitl_router, prefix="/api/v1/hitl", tags=["HITL"])

        logger.info("HITL routes mounted at /api/v1/hitl")

    except ImportError as e:
        logger.warning(f"HITL routes not available: {e}")


def _include_workflow_editor_api(app: FastAPI) -> None:
    """Include workflow editor API routes.

    The workflow editor provides:
    - Workflow validation
    - YAML import/export
    - Node type definitions
    - Team formation configurations

    Routes are available at /api/v1/workflows
    """
    try:
        # Try to import the workflow editor
        # Note: The workflow editor may have import issues, so we handle it gracefully
        import sys
        from pathlib import Path

        # Add project root to path
        project_root = Path(__file__).parent.parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        try:
            from tools.workflow_editor.backend.api import app as workflow_app

            # Mount at a cleaner path
            app.mount("/workflow-editor", workflow_app, name="workflow_editor")

            # Create convenience proxies for common endpoints
            @app.get("/api/v1/workflows/nodes/types", tags=["Workflows"])
            async def get_node_types_proxy() -> JSONResponse:
                """Get available workflow node types."""
                from tools.workflow_editor.backend.api import get_node_types

                return await get_node_types()

            @app.get("/api/v1/workflows/formations", tags=["Workflows"])
            async def get_formations_proxy() -> JSONResponse:
                """Get available team formation types."""
                from tools.workflow_editor.backend.api import get_formations

                return await get_formations()

            logger.info("Workflow editor routes mounted at /workflow-editor and /api/v1/workflows")

        except ImportError as import_error:
            # Workflow editor has dependency issues, create simplified endpoints
            logger.warning(f"Workflow editor import failed: {import_error}")
            logger.info("Creating simplified workflow endpoints")

            @app.get("/api/v1/workflows/nodes/types", tags=["Workflows"])
            async def get_node_types_fallback() -> Dict[str, Any]:
                """Get available workflow node types (simplified)."""
                return {
                    "agent": {
                        "name": "Agent Node",
                        "description": "LLM-powered agent",
                        "color": "#E3F2FD",
                    },
                    "compute": {
                        "name": "Compute Node",
                        "description": "Execute tools without LLM",
                        "color": "#E8F5E9",
                    },
                    "team": {
                        "name": "Team Node",
                        "description": "Multi-agent team",
                        "color": "#F3E5F5",
                    },
                    "condition": {
                        "name": "Condition Node",
                        "description": "Branching logic",
                        "color": "#FFF3E0",
                    },
                    "hitl": {
                        "name": "Human-in-the-Loop",
                        "description": "Human interaction",
                        "color": "#FFEBEE",
                    },
                }

            @app.get("/api/v1/workflows/formations", tags=["Workflows"])
            async def get_formations_fallback() -> Dict[str, Any]:
                """Get available team formation types (simplified)."""
                return {
                    "parallel": {
                        "name": "Parallel",
                        "description": "All members work simultaneously",
                        "icon": "||",
                    },
                    "sequential": {
                        "name": "Sequential",
                        "description": "Members work in sequence",
                        "icon": "→",
                    },
                    "pipeline": {
                        "name": "Pipeline",
                        "description": "Output passes through stages",
                        "icon": "⇒",
                    },
                    "hierarchical": {
                        "name": "Hierarchical",
                        "description": "Manager-worker coordination",
                        "icon": "⬗",
                    },
                }

            logger.info("Simplified workflow endpoints created")

    except Exception as e:
        logger.warning(f"Workflow editor routes not available: {e}")


def _setup_frontend_routes(app: FastAPI) -> None:
    """Setup frontend UI routes."""

    # Redirect root to landing page
    @app.get("/", response_class=RedirectResponse, include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/ui")

    # Landing page
    @app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
    async def ui_landing() -> HTMLResponse:
        return HTMLResponse(content=LANDING_PAGE_HTML)

    # HITL UI
    @app.get("/ui/hitl", response_class=HTMLResponse, include_in_schema=False)
    async def hitl_ui() -> HTMLResponse:
        try:
            from victor.workflows.hitl_api import get_hitl_ui_html

            return HTMLResponse(content=get_hitl_ui_html())
        except ImportError:
            return HTMLResponse(
                content="<h1>HITL UI Not Available</h1><p>HITL module not found.</p>",
                status_code=503,
            )

    # Workflow editor UI
    @app.get("/ui/workflow-editor", response_class=HTMLResponse, include_in_schema=False)
    async def workflow_editor_ui() -> HTMLResponse:
        workflow_editor_dist = (
            Path(__file__).parent.parent.parent.parent
            / "tools"
            / "workflow_editor"
            / "frontend"
            / "dist"
        )

        if workflow_editor_dist.exists():
            index_path = workflow_editor_dist / "index.html"
            if index_path.exists():
                with open(index_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Update API paths
                content = content.replace("/api/workflows/", "/workflow-editor/api/workflows/")

                return HTMLResponse(content=content)
            else:
                return HTMLResponse(
                    content="<h1>Workflow Editor Not Found</h1><p>index.html not found.</p>",
                    status_code=404,
                )
        else:
            return HTMLResponse(
                content="""
<!DOCTYPE html>
<html>
<head>
    <title>Workflow Editor - Development Mode</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 100px auto; padding: 2rem; background: #0f172a; color: #f1f5f9; }
        h1 { color: #6366f1; }
        .code { background: #1e293b; padding: 1rem; border-radius: 8px; margin: 1rem 0; font-family: monospace; }
        a { color: #818cf8; }
    </style>
</head>
<body>
    <h1>Workflow Editor - Development Mode</h1>
    <p>The workflow editor frontend is not built. In development, run the Vite dev server:</p>
    <div class="code">
        cd tools/workflow_editor/frontend && npm install && npm run dev
    </div>
    <p>Or build for production:</p>
    <div class="code">
        cd tools/workflow_editor/frontend && npm run build
    </p>
    <p>The editor will be available at <a href="http://localhost:5173">http://localhost:5173</a></p>
</body>
</html>
"""
            )


def _setup_health_check(app: FastAPI) -> None:
    """Setup unified health check."""

    @app.get("/health", tags=["System"])
    async def unified_health_check():
        """Health check for all services."""
        status: Dict[str, Any] = {
            "status": "healthy",
            "services": {
                "main_api": "healthy",
                "hitl": "not_enabled",
                "workflow_editor": "available",
            },
        }

        # Check HITL
        if hasattr(app.state, "hitl_store") and app.state.hitl_store:
            try:
                pending = await app.state.hitl_store.list_pending()
                status["services"]["hitl"] = {
                    "status": "healthy",
                    "pending_requests": len(pending),
                }
            except Exception as e:
                status["services"]["hitl"] = {"status": "unhealthy", "error": str(e)}
                status["status"] = "degraded"

        # Check workflow editor
        workflow_editor_dist = (
            Path(__file__).parent.parent.parent.parent
            / "tools"
            / "workflow_editor"
            / "frontend"
            / "dist"
        )
        if not workflow_editor_dist.exists():
            status["services"]["workflow_editor"] = "not_built"

        return JSONResponse(content=status)


def run_unified_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workspace_root: Optional[str] = None,
    enable_hitl: bool = True,
    hitl_persistent: bool = True,
    hitl_auth_token: Optional[str] = None,
    log_level: str = "info",
) -> None:
    """Run the unified server."""
    import uvicorn

    app = create_unified_server(
        host=host,
        port=port,
        workspace_root=workspace_root,
        enable_hitl=enable_hitl,
        hitl_persistent=hitl_persistent,
        hitl_auth_token=hitl_auth_token,
    )

    uvicorn.run(app, host=host, port=port, log_level=log_level)


__all__ = ["create_unified_server", "run_unified_server"]
