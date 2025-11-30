import logging
import asyncio
import uuid
import subprocess
import tempfile
import time
import os
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from victor.config.settings import load_settings
from victor.agent.orchestrator import AgentOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Victor AI Assistant API", version="2.0.0")

# CORS configuration for cross-origin requests
# Set CORS_ORIGINS environment variable for production (comma-separated)
# Example: CORS_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"
cors_origins_env = os.getenv("CORS_ORIGINS", "")
if cors_origins_env:
    allowed_origins = [origin.strip() for origin in cors_origins_env.split(",")]
else:
    # Default to localhost for development
    allowed_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]

logger.info(f"CORS enabled for origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load settings once at startup
try:
    settings = load_settings()
    logger.info("Settings loaded successfully")
except Exception as e:
    logger.critical(f"FATAL: Failed to load settings: {e}")
    import sys

    sys.exit(1)  # Fail fast - don't start with broken config

# Session management with metadata
SESSION_AGENTS: Dict[str, Dict[str, Any]] = {}
SESSION_LOCK = asyncio.Lock()

# Configuration constants
HEARTBEAT_INTERVAL = 30  # seconds
SESSION_IDLE_TIMEOUT = 3600  # 1 hour in seconds
CLEANUP_INTERVAL = 300  # 5 minutes
MESSAGE_TIMEOUT = 300  # 5 minutes for receive timeout

# Track background tasks for graceful shutdown
_background_tasks = []


def _render_plantuml_svg(source: str) -> str:
    """Render PlantUML text to SVG using local plantuml CLI."""
    try:
        proc = subprocess.run(
            ["plantuml", "-tsvg", "-pipe"],
            input=source.encode(),
            capture_output=True,
            check=True,
        )
        return proc.stdout.decode()
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500, detail=f"PlantUML render failed: {exc.stderr.decode()}"
        ) from exc


def _render_mermaid_svg(source: str) -> str:
    """Render Mermaid text to SVG using local mmdc CLI."""
    try:
        with (
            tempfile.NamedTemporaryFile(mode="w+", suffix=".mmd", delete=True) as fin,
            tempfile.NamedTemporaryFile(mode="r", suffix=".svg", delete=True) as fout,
        ):
            fin.write(source)
            fin.flush()
            cmd = ["mmdc", "-i", fin.name, "-o", fout.name]
            subprocess.run(cmd, check=True, capture_output=True)
            fout.seek(0)
            return fout.read()
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500, detail=f"Mermaid render failed: {exc.stderr.decode()}"
        ) from exc


def _render_drawio_svg(source: str) -> str:
    """Render Draw.io (or Lucid-style XML) to SVG using local drawio CLI."""
    try:
        with (
            tempfile.NamedTemporaryFile(mode="w+", suffix=".drawio", delete=True) as fin,
            tempfile.NamedTemporaryFile(mode="r", suffix=".svg", delete=True) as fout,
        ):
            fin.write(source)
            fin.flush()
            # drawio CLI flags: -x (export), -f svg (format), -o output
            cmd = ["drawio", "-x", "-f", "svg", "-o", fout.name, fin.name]
            subprocess.run(cmd, check=True, capture_output=True)
            fout.seek(0)
            return fout.read()
    except FileNotFoundError:
        raise HTTPException(
            status_code=500, detail="drawio CLI not found. Please install draw.io desktop/CLI."
        ) from None
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500, detail=f"Draw.io render failed: {exc.stderr.decode()}"
        ) from exc


@app.post("/render/plantuml")
async def render_plantuml(payload: str = Body(..., media_type="text/plain")):
    svg = _render_plantuml_svg(payload)
    return Response(content=svg, media_type="image/svg+xml")


@app.post("/render/mermaid")
async def render_mermaid(payload: str = Body(..., media_type="text/plain")):
    svg = _render_mermaid_svg(payload)
    return Response(content=svg, media_type="image/svg+xml")


@app.post("/render/drawio")
async def render_drawio(payload: str = Body(..., media_type="text/plain")):
    svg = _render_drawio_svg(payload)
    return Response(content=svg, media_type="image/svg+xml")


def _render_graphviz_svg(source: str, engine: str = "dot") -> str:
    """Render Graphviz DOT to SVG using local graphviz CLI.

    Supported engines: dot, neato, fdp, circo, twopi, sfdp
    - dot: Hierarchical/layered graphs (default)
    - neato: Spring model layouts
    - fdp: Force-directed placement
    - circo: Circular layout
    - twopi: Radial layouts
    - sfdp: Large graph layouts (scalable force-directed)
    """
    valid_engines = {"dot", "neato", "fdp", "circo", "twopi", "sfdp"}
    if engine not in valid_engines:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid engine '{engine}'. Must be one of: {', '.join(valid_engines)}",
        )

    try:
        proc = subprocess.run(
            [engine, "-Tsvg"],
            input=source.encode(),
            capture_output=True,
            check=True,
        )
        return proc.stdout.decode()
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=f"Graphviz engine '{engine}' not found. Install: brew install graphviz (macOS) or apt-get install graphviz (Linux)",
        )
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500, detail=f"Graphviz render failed: {exc.stderr.decode()}"
        )


def _render_d2_svg(source: str) -> str:
    """Render D2 diagram to SVG using local d2 CLI.

    D2 is a modern diagram scripting language with features:
    - Auto-layout algorithms
    - Built-in themes
    - Icons and shapes
    - Connections and relationships
    """
    try:
        with (
            tempfile.NamedTemporaryFile(mode="w+", suffix=".d2", delete=True) as fin,
            tempfile.NamedTemporaryFile(mode="r", suffix=".svg", delete=True) as fout,
        ):
            fin.write(source)
            fin.flush()
            cmd = ["d2", fin.name, fout.name]
            subprocess.run(cmd, check=True, capture_output=True)
            fout.seek(0)
            return fout.read()
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="d2 CLI not found. Install: curl -fsSL https://d2lang.com/install.sh | sh -s --",
        )
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail=f"D2 render failed: {exc.stderr.decode()}")


@app.post("/render/graphviz")
async def render_graphviz(payload: str = Body(..., media_type="text/plain"), engine: str = "dot"):
    """Render Graphviz DOT diagram to SVG.

    Query param 'engine' can be: dot (default), neato, fdp, circo, twopi, sfdp
    """
    svg = _render_graphviz_svg(payload, engine)
    return Response(content=svg, media_type="image/svg+xml")


@app.post("/render/d2")
async def render_d2(payload: str = Body(..., media_type="text/plain")):
    """Render D2 diagram to SVG."""
    svg = _render_d2_svg(payload)
    return Response(content=svg, media_type="image/svg+xml")


async def heartbeat_loop(websocket: WebSocket, session_id: str):
    """Send periodic ping messages to keep connection alive and detect dead connections."""
    try:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            try:
                await websocket.send_text("[ping]")
                # Update last activity on successful ping
                async with SESSION_LOCK:
                    if session_id in SESSION_AGENTS:
                        SESSION_AGENTS[session_id]["last_activity"] = time.time()
            except Exception as e:
                logger.warning(f"Heartbeat failed for session {session_id}: {e}")
                break
    except asyncio.CancelledError:
        logger.debug(f"Heartbeat loop cancelled for session {session_id}")


async def cleanup_idle_sessions():
    """Background task to clean up idle sessions to prevent memory leaks."""
    logger.info("Starting idle session cleanup task")
    try:
        while True:
            try:
                await asyncio.sleep(CLEANUP_INTERVAL)

                current_time = time.time()
                sessions_to_remove = []

                async with SESSION_LOCK:
                    for session_id, session_data in SESSION_AGENTS.items():
                        last_activity = session_data.get("last_activity", 0)
                        idle_time = current_time - last_activity

                        if idle_time > SESSION_IDLE_TIMEOUT:
                            sessions_to_remove.append(session_id)

                    for session_id in sessions_to_remove:
                        logger.info(f"Cleaning up idle session: {session_id}")
                        # Cleanup agent resources
                        try:
                            agent = SESSION_AGENTS[session_id]["agent"]
                            if hasattr(agent, "shutdown"):
                                agent.shutdown()
                            # Also try closing provider if available
                            if hasattr(agent, "provider"):
                                await agent.provider.close()
                        except Exception as e:
                            logger.warning(
                                f"Error shutting down agent for session {session_id}: {e}"
                            )

                        del SESSION_AGENTS[session_id]

                if sessions_to_remove:
                    logger.info(f"Cleaned up {len(sessions_to_remove)} idle sessions")

            except Exception as e:
                logger.error(f"Error in cleanup task: {e}", exc_info=True)

    except asyncio.CancelledError:
        logger.info("Idle session cleanup task cancelled gracefully")
        raise  # Re-raise to properly handle task cancellation


@app.on_event("startup")
async def startup_event():
    """Start background tasks on server startup."""
    logger.info("Starting background tasks...")
    task = asyncio.create_task(cleanup_idle_sessions())
    _background_tasks.append(task)
    logger.info("Server started successfully with background tasks tracked")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up background tasks and sessions on server shutdown."""
    logger.info("Shutting down background tasks...")

    # Cancel all background tasks gracefully
    for task in _background_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Clean up all active sessions
    async with SESSION_LOCK:
        for session_id, session_data in list(SESSION_AGENTS.items()):
            try:
                agent = session_data.get("agent")
                if agent:
                    if hasattr(agent, "shutdown"):
                        agent.shutdown()
                    if hasattr(agent, "provider"):
                        await agent.provider.close()
                    logger.info(f"Closed session {session_id} during shutdown")
            except Exception as e:
                logger.error(f"Error closing session {session_id}: {e}")

        SESSION_AGENTS.clear()

    logger.info("Shutdown complete - all tasks and sessions cleaned up")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_sessions": len(SESSION_AGENTS),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket endpoint with heartbeat, timeout, and error handling."""
    session_id = None
    heartbeat_task = None
    session_initialized = False  # Track if we successfully initialized session

    try:
        await websocket.accept()
        logger.info("WebSocket connection established.")

        # Session-aware agent reuse
        session_id = websocket.query_params.get("session_id") or str(uuid.uuid4())

        try:
            await websocket.send_text(f"[session] {session_id}")
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected before session handshake.")
            return

        # Create or retrieve agent with metadata
        async with SESSION_LOCK:
            if session_id in SESSION_AGENTS:
                session_data = SESSION_AGENTS[session_id]
                agent = session_data["agent"]
                session_data["last_activity"] = time.time()
                session_data["connection_count"] = session_data.get("connection_count", 0) + 1
                logger.info(
                    f"Reusing existing agent for session {session_id} (connection #{session_data['connection_count']})"
                )
                session_initialized = True  # Mark as initialized
            else:
                try:
                    agent = await AgentOrchestrator.from_settings(settings=settings)

                    # CRITICAL FIX: Trigger background embedding preload
                    agent.start_embedding_preload()

                    SESSION_AGENTS[session_id] = {
                        "agent": agent,
                        "created_at": time.time(),
                        "last_activity": time.time(),
                        "connection_count": 1,
                    }
                    logger.info(
                        f"Created new AgentOrchestrator for session {session_id} with preload started."
                    )
                    session_initialized = True  # Mark as initialized
                except Exception as e:
                    logger.error(f"Failed to create AgentOrchestrator: {e}", exc_info=True)
                    await websocket.send_text(f"[error] Could not initialize agent: {str(e)}")
                    await websocket.close(code=1011, reason="Agent initialization failed")
                    return

        # Start heartbeat loop
        heartbeat_task = asyncio.create_task(heartbeat_loop(websocket, session_id))

        # Main message loop with timeout
        while True:
            try:
                # Receive message with timeout to prevent hanging
                user_message = await asyncio.wait_for(
                    websocket.receive_text(), timeout=MESSAGE_TIMEOUT
                )

                logger.info(f"Session {session_id}: Received message")

                # Update last activity
                async with SESSION_LOCK:
                    if session_id in SESSION_AGENTS:
                        SESSION_AGENTS[session_id]["last_activity"] = time.time()

                # Handle special commands
                if user_message.strip() == "__reset_session__":
                    agent.reset_conversation()
                    await websocket.send_text("[session] reset")
                    logger.info(f"Session {session_id}: Conversation reset")
                    continue

                # Stream the agent's response back to the client
                try:
                    async for chunk in agent.stream_chat(user_message):
                        if chunk.content:
                            await websocket.send_text(chunk.content)

                    # Send final empty chunk to signal completion
                    await websocket.send_text("")

                except Exception as e:
                    logger.error(
                        f"Session {session_id}: Error during agent response: {e}", exc_info=True
                    )
                    await websocket.send_text(
                        f"[error] An error occurred while processing your request: {str(e)}"
                    )

            except asyncio.TimeoutError:
                logger.warning(f"Session {session_id}: Timeout waiting for message")
                await websocket.send_text("[error] Connection timeout. Please refresh.")
                break

            except WebSocketDisconnect:
                logger.info(f"Session {session_id}: WebSocket disconnected by client")
                break

    except WebSocketDisconnect:
        logger.info(f"Session {session_id}: WebSocket connection disconnected.")
    except Exception as e:
        logger.error(f"Session {session_id}: Unexpected error in WebSocket: {e}", exc_info=True)
        try:
            await websocket.send_text(f"[error] Server error: {str(e)}")
        except (WebSocketDisconnect, RuntimeError) as send_error:
            logger.debug(f"Failed to send error message to client: {send_error}")
    finally:
        # Cancel heartbeat task
        if heartbeat_task and not heartbeat_task.done():
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        # Update connection count (only if we successfully initialized)
        if session_id and session_initialized:
            async with SESSION_LOCK:
                if session_id in SESSION_AGENTS:
                    current_count = SESSION_AGENTS[session_id].get("connection_count", 1)
                    new_count = max(0, current_count - 1)  # Prevent negative counts
                    SESSION_AGENTS[session_id]["connection_count"] = new_count
                    logger.debug(
                        f"Session {session_id}: Connection count decremented to {new_count}"
                    )

        logger.info(f"Session {session_id}: WebSocket connection closed and cleaned up.")
