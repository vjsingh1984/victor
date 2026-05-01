# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Service-owned planning runtime helper for chat execution."""

from __future__ import annotations

from typing import Any

from victor.providers.base import CompletionResponse


class PlanningChatRuntime:
    """Execute planning-mode chat turns through the canonical runtime surface."""

    def __init__(self, runtime_host: Any) -> None:
        self._runtime = runtime_host

    @staticmethod
    def _get_runtime_state_host(runtime_host: Any) -> Any:
        """Return the concrete object that owns cached runtime state."""
        instance_dict = getattr(runtime_host, "__dict__", {})
        if isinstance(instance_dict, dict) and "_orchestrator" in instance_dict:
            return instance_dict["_orchestrator"]
        return runtime_host

    async def run(self, user_message: str) -> CompletionResponse:
        """Run one planning-enabled chat turn."""
        from victor.agent.services.planning_runtime import PlanningCoordinator

        runtime = self._runtime
        state_host = self._get_runtime_state_host(runtime)
        state_dict = getattr(state_host, "__dict__", {})
        planning_coordinator = state_dict.get("_service_planning_coordinator")
        if planning_coordinator is None:
            planning_coordinator = PlanningCoordinator(runtime)
            runtime._service_planning_coordinator = planning_coordinator

        task_analysis = runtime.task_analyzer.analyze(user_message)
        initial_message_count = runtime._get_conversation_message_count()
        response = await planning_coordinator.chat_with_planning(
            user_message,
            task_analysis=task_analysis,
        )

        if runtime._get_conversation_message_count() > initial_message_count:
            return response

        if not runtime._system_added:
            runtime.conversation.ensure_system_prompt()
            runtime._system_added = True

        runtime.add_message("user", user_message)
        if response.content:
            runtime.add_message("assistant", response.content)

        return response
