# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Interaction runtime boundaries for AgentOrchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from victor.agent.runtime.provider_runtime import LazyRuntimeProxy


@dataclass(frozen=True)
class InteractionRuntimeComponents:
    """Interaction runtime handles exposed to the orchestrator facade."""

    chat_coordinator: LazyRuntimeProxy[Any]
    tool_coordinator: LazyRuntimeProxy[Any]
    session_coordinator: LazyRuntimeProxy[Any]


def create_interaction_runtime_components(
    *,
    orchestrator: Any,
    tool_pipeline: Any,
    tool_registry: Any,
    tool_selector: Any,
    tool_access_controller: Any,
    mode_controller: Any,
    session_state_manager: Any,
    lifecycle_manager: Any,
    memory_manager: Any,
    checkpoint_manager: Any,
    cost_tracker: Any,
) -> InteractionRuntimeComponents:
    """Create lazy interaction/runtime coordinator components."""

    def _build_chat_coordinator() -> Any:
        from victor.agent.coordinators.chat_coordinator import ChatCoordinator

        return ChatCoordinator(orchestrator)

    def _build_tool_coordinator() -> Any:
        from victor.agent.coordinators.tool_coordinator import ToolCoordinator

        coordinator = ToolCoordinator(
            tool_pipeline=tool_pipeline,
            tool_registry=tool_registry,
            tool_selector=tool_selector,
            tool_access_controller=tool_access_controller,
        )
        coordinator.set_mode_controller(mode_controller)
        coordinator.set_orchestrator_reference(orchestrator)
        return coordinator

    def _build_session_coordinator() -> Any:
        from victor.agent.coordinators.session_coordinator import create_session_coordinator

        return create_session_coordinator(
            session_state_manager=session_state_manager,
            lifecycle_manager=lifecycle_manager,
            memory_manager=memory_manager,
            checkpoint_manager=checkpoint_manager,
            cost_tracker=cost_tracker,
        )

    return InteractionRuntimeComponents(
        chat_coordinator=LazyRuntimeProxy(
            factory=_build_chat_coordinator,
            name="chat_coordinator",
        ),
        tool_coordinator=LazyRuntimeProxy(
            factory=_build_tool_coordinator,
            name="tool_coordinator",
        ),
        session_coordinator=LazyRuntimeProxy(
            factory=_build_session_coordinator,
            name="session_coordinator",
        ),
    )
