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

"""Configuration routes: /model/switch, /mode/switch, /models, /providers, /capabilities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from victor.integrations.api.fastapi_server import SwitchModelRequest, SwitchModeRequest

if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

logger = logging.getLogger(__name__)


def create_router(server: "VictorFastAPIServer") -> APIRouter:
    """Create configuration routes bound to *server*."""
    router = APIRouter(tags=["Configuration"])

    @router.post("/model/switch")
    async def switch_model(request: SwitchModelRequest) -> JSONResponse:
        """Switch AI model."""
        from victor.agent.model_switcher import get_model_switcher

        switcher = get_model_switcher()
        switcher.switch(request.provider, request.model)

        return JSONResponse(
            {"success": True, "provider": request.provider, "model": request.model}
        )

    @router.post("/mode/switch")
    async def switch_mode(request: SwitchModeRequest) -> JSONResponse:
        """Switch agent mode."""
        from victor.agent.mode_controller import AgentMode, get_mode_controller

        manager = get_mode_controller()
        manager.switch_mode(AgentMode(request.mode))

        return JSONResponse({"success": True, "mode": request.mode})

    @router.get("/models")
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

    @router.get("/providers")
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

    @router.get("/capabilities")
    async def get_capabilities(
        vertical: Optional[str] = Query(None, description="Filter by vertical"),
    ) -> JSONResponse:
        """Discover all Victor capabilities."""
        try:
            from victor.ui.commands.capabilities import get_capability_discovery

            discovery = get_capability_discovery()

            if vertical:
                manifest = discovery.discover_by_vertical(vertical)
                return JSONResponse(manifest)
            else:
                manifest = discovery.discover_all()
                return JSONResponse(manifest.to_dict())

        except Exception:
            logger.exception("Capabilities discovery error")
            return JSONResponse(
                {"error": "Internal server error", "capabilities": {}}, status_code=500
            )

    return router
