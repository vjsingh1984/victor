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

"""System routes: /health, /status, /events/recent, /credentials, /session, /shutdown."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, List, Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from victor.integrations.api.event_bridge import EventBroadcaster
from victor.integrations.api.fastapi_server import HealthResponse, StatusResponse

if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

logger = logging.getLogger(__name__)


def create_router(server: "VictorFastAPIServer") -> APIRouter:
    """Create system routes bound to *server*."""
    router = APIRouter(tags=["System"])

    @router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse()

    @router.get("/status", response_model=StatusResponse)
    async def status() -> StatusResponse:
        """Get current server status."""
        try:
            orchestrator = await server._get_orchestrator()
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
                workspace=server.workspace_root,
                capabilities=server._detect_capabilities(),
            )
        except Exception as e:
            logger.warning(f"Status check error: {e}")
            return StatusResponse(
                connected=False,
                mode="unknown",
                provider="unknown",
                model="unknown",
                workspace=server.workspace_root,
                capabilities=server._detect_capabilities(),
            )

    @router.get("/events/recent")
    async def recent_events(
        limit: int = Query(default=12, ge=1, le=100),
        categories: Optional[List[str]] = Query(default=None),
        correlation_id: Optional[str] = Query(default=None),
    ) -> JSONResponse:
        """Return recent bridged events for websocket timeline hydration."""
        broadcaster = (
            server._event_bridge._broadcaster if server._event_bridge else EventBroadcaster()
        )
        events = broadcaster.get_recent_events(
            limit=limit,
            subscriptions=set(categories) if categories else None,
            correlation_id=correlation_id,
        )
        return JSONResponse(
            {
                "events": [event.to_dict() for event in events],
                "count": len(events),
            }
        )

    @router.get("/credentials/get")
    async def credentials_get(provider: str = Query("")) -> JSONResponse:
        """Placeholder credentials endpoint."""
        return JSONResponse({"provider": provider, "api_key": None})

    @router.post("/session/token")
    async def session_token() -> JSONResponse:
        """Placeholder session token endpoint."""
        return JSONResponse({"session_token": str(uuid.uuid4()), "session_id": str(uuid.uuid4())})

    @router.post("/shutdown")
    async def shutdown() -> JSONResponse:
        """Shutdown the server gracefully."""
        if server._shutting_down:
            return JSONResponse({"status": "already_shutting_down"})

        server._shutting_down = True
        logger.info("Shutdown requested")
        await server._record_rl_feedback()

        for ws in server._ws_clients:
            try:
                await ws.close()
            except Exception:
                pass

        if hasattr(server, "_server") and server._server is not None:
            server._server.should_exit = True
            asyncio.create_task(server._delayed_shutdown())

        return JSONResponse({"status": "shutting_down"})

    return router
