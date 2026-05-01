# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Service-owned provider management runtime helper."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ProviderManagementRuntime:
    """Bridge provider service operations back to orchestrator runtime state."""

    def __init__(self, runtime_host: Any) -> None:
        self._runtime = runtime_host

    @staticmethod
    def _get_runtime_state_host(runtime_host: Any) -> Any:
        """Return the concrete object that owns cached provider runtime state."""
        instance_dict = getattr(runtime_host, "__dict__", {})
        if isinstance(instance_dict, dict) and "_orchestrator" in instance_dict:
            return instance_dict["_orchestrator"]
        return runtime_host

    def _get_provider_service(self) -> Any:
        """Return the canonical provider service, if initialized."""
        return getattr(self._runtime, "_provider_service", None)

    def _sync_runtime_provider_state(self, provider_service: Any) -> Any:
        """Sync current provider/model state back to the concrete runtime host."""
        state_host = self._get_runtime_state_host(self._runtime)
        info = provider_service.get_current_provider_info()
        state_host.provider = provider_service.get_current_provider()
        state_host.provider_name = info.provider_name
        state_host.model = info.model_name
        return info

    async def switch_provider(
        self,
        provider_name: str,
        model: Optional[str] = None,
        on_switch: Optional[Any] = None,
    ) -> bool:
        """Switch provider and sync runtime state."""
        provider_service = self._get_provider_service()
        if provider_service is None:
            logger.error("Provider switch failed: canonical provider service is not initialized")
            return False

        try:
            await provider_service.switch_provider(provider_name, model)
            info = self._sync_runtime_provider_state(provider_service)
            if on_switch:
                on_switch(info.provider_name, info.model_name)
            return True
        except Exception as exc:
            logger.error("Provider switch failed: %s", exc)
            return False

    async def switch_model(self, model: str) -> bool:
        """Switch model and sync runtime state."""
        provider_service = self._get_provider_service()
        if provider_service is None:
            logger.error("Model switch failed: canonical provider service is not initialized")
            return False

        try:
            await provider_service.switch_model(model)
            self._sync_runtime_provider_state(provider_service)
            return True
        except Exception as exc:
            logger.error("Model switch failed: %s", exc)
            return False

    async def start_health_monitoring(self) -> bool:
        """Start provider health monitoring via the canonical service."""
        provider_service = self._get_provider_service()
        if provider_service is None:
            return False

        try:
            await provider_service.start_health_monitoring()
            return True
        except Exception as exc:
            logger.warning("Failed to start health monitoring: %s", exc)
            return False

    async def stop_health_monitoring(self) -> bool:
        """Stop provider health monitoring via the canonical service."""
        provider_service = self._get_provider_service()
        if provider_service is None:
            return False

        try:
            await provider_service.stop_health_monitoring()
            return True
        except Exception as exc:
            logger.warning("Failed to stop health monitoring: %s", exc)
            return False

    async def get_provider_health(self) -> Dict[str, Any]:
        """Return provider health details from the canonical service."""
        provider_service = self._get_provider_service()
        try:
            is_healthy = await provider_service.check_provider_health()
            info = provider_service.get_current_provider_info()
            return {
                "healthy": is_healthy,
                "provider": info.provider_name,
                "model": info.model_name,
                "api_key_configured": info.api_key_configured,
            }
        except Exception as exc:
            logger.warning("Health check failed: %s", exc)
            return {"healthy": False, "error": str(exc)}

    def get_current_provider_info(self) -> Dict[str, Any]:
        """Return provider metadata merged with orchestrator runtime state."""
        runtime = self._runtime
        provider_service = self._get_provider_service()
        info = provider_service.get_current_provider_info()
        result = {
            "provider_name": info.provider_name,
            "model_name": info.model_name,
            "api_key_configured": info.api_key_configured,
            "supports_streaming": info.supports_streaming,
            "supports_tool_calling": info.supports_tool_calling,
            "max_tokens": info.max_tokens,
            "tool_budget": runtime.tool_budget,
            "tool_calls_used": runtime.tool_calls_used,
        }

        stats = provider_service.get_rate_limit_stats()
        if stats:
            result.update(stats)

        return result
