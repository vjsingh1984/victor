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
            logger.error(
                "Provider switch failed: canonical provider service is not initialized"
            )
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
            logger.error(
                "Model switch failed: canonical provider service is not initialized"
            )
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

    def model_supports_tool_calls(self) -> bool:
        """Check the active provider/model against Victor's tool-call capability matrix."""
        runtime = self._runtime
        provider = getattr(runtime, "provider", None)
        provider_key = getattr(runtime, "provider_name", None) or getattr(
            provider, "name", ""
        )
        if not provider_key:
            return True

        model = getattr(runtime, "model", "")
        tool_capabilities = getattr(runtime, "tool_capabilities", None)

        if tool_capabilities is not None:
            supported = tool_capabilities.is_tool_call_supported(provider_key, model)
            if supported:
                return True

        if str(provider_key).lower() == "ollama":
            try:
                from victor.providers.ollama_capability_detector import (
                    get_global_detector,
                )

                settings = getattr(runtime, "settings", None)
                provider_settings = getattr(settings, "provider", None)
                base_url = getattr(
                    provider_settings,
                    "ollama_base_url",
                    "http://localhost:11434",
                )
                detector = get_global_detector(base_url)
                tool_support = detector.get_tool_support(model)
                if tool_support.supports_tools:
                    logger.info(
                        "Model '%s' supports tools (detected via Ollama API, method=%s)",
                        model,
                        tool_support.detection_method,
                    )
                    return True
            except Exception as exc:
                logger.debug("Ollama capability detection failed: %s", exc)

        if not getattr(runtime, "_tool_capability_warned", False):
            known = ""
            if tool_capabilities is not None:
                known = ", ".join(tool_capabilities.get_supported_models(provider_key))
            known = known or "none"
            logger.warning(
                "Model '%s' is not marked as tool-call-capable for provider '%s'. "
                "Known tool-capable models: %s",
                model,
                provider_key,
                known,
            )
            console = getattr(runtime, "console", None)
            presentation = getattr(runtime, "_presentation", None)
            if console is not None:
                icon = (
                    presentation.icon("warning", with_color=False)
                    if presentation is not None
                    else "!"
                )
                console.print(
                    f"[yellow]{icon} Model '{model}' is not marked as "
                    f"tool-call-capable for provider '{provider_key}'. Running without tools.[/]"
                )
            runtime._tool_capability_warned = True
        return False
