# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Interaction runtime boundaries for AgentOrchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from victor.agent.runtime.provider_runtime import LazyRuntimeProxy


@dataclass(frozen=True)
class InteractionRuntimeComponents:
    """Interaction runtime handles exposed to the orchestrator facade."""

    tool_service: Any
    session_service: Any
    chat_coordinator: LazyRuntimeProxy[Any]
    tool_coordinator: LazyRuntimeProxy[Any]
    session_coordinator: LazyRuntimeProxy[Any]


def create_interaction_runtime_components(
    *,
    orchestrator: Any,
    factory: Any,
    tool_pipeline: Any,
    tool_registry: Any,
    tool_executor: Any,
    tool_budget: int,
    tool_selector: Any,
    tool_access_controller: Any,
    mode_controller: Any,
    session_state_manager: Any,
    lifecycle_manager: Any,
    memory_manager: Any,
    memory_session_id: Optional[str],
    checkpoint_manager: Any,
    cost_tracker: Any,
    tool_service: Optional[Any] = None,
    session_service: Optional[Any] = None,
) -> InteractionRuntimeComponents:
    """Create service-first interaction/runtime components for orchestrator wiring."""
    from victor.agent.services.session_service import SessionService
    from victor.agent.services.tool_service import ToolService, ToolServiceConfig

    resolved_tool_service = tool_service
    if resolved_tool_service is None:
        resolved_tool_service = ToolService(
            config=ToolServiceConfig(default_tool_budget=tool_budget),
            tool_selector=tool_selector,
            tool_executor=tool_executor,
            tool_registrar=tool_registry,
        )

    if hasattr(resolved_tool_service, "bind_runtime_components"):
        resolved_tool_service.bind_runtime_components(
            tool_registry=tool_registry,
            mode_controller=mode_controller,
        )

    orch_tools = getattr(orchestrator, "_enabled_tools", None)
    if orch_tools and hasattr(resolved_tool_service, "set_enabled_tools"):
        resolved_tool_service.set_enabled_tools(orch_tools)

    resolved_session_service = session_service
    if resolved_session_service is None:
        resolved_session_service = SessionService(
            session_state_manager=session_state_manager,
            lifecycle_manager=lifecycle_manager,
            memory_manager=memory_manager,
            checkpoint_manager=checkpoint_manager,
            cost_tracker=cost_tracker,
        )

    if hasattr(resolved_session_service, "bind_runtime_components"):
        resolved_session_service.bind_runtime_components(
            lifecycle_manager=lifecycle_manager,
            memory_manager=memory_manager,
            checkpoint_manager=checkpoint_manager,
            cost_tracker=cost_tracker,
            memory_session_id=memory_session_id,
        )

    def _build_chat_coordinator() -> Any:
        from victor.agent.coordinators.chat_coordinator import ChatCoordinator

        coordinator = ChatCoordinator(orchestrator)
        if hasattr(factory, "create_streaming_chat_pipeline"):
            pipeline = factory.create_streaming_chat_pipeline(coordinator)
            coordinator.set_streaming_pipeline(pipeline)
        return coordinator

    def _build_tool_coordinator() -> Any:
        from victor.agent.coordinators.tool_coordinator import ToolCoordinator

        coordinator = ToolCoordinator(
            tool_pipeline=tool_pipeline,
            tool_registry=tool_registry,
            tool_selector=tool_selector,
            tool_access_controller=tool_access_controller,
        )
        coordinator.set_mode_controller(mode_controller)
        coordinator.bind_tool_service(resolved_tool_service)
        if orch_tools:
            coordinator.set_enabled_tools(orch_tools)
        return coordinator

    def _build_session_coordinator() -> Any:
        from victor.agent.coordinators.session_coordinator import (
            create_session_coordinator,
        )

        coordinator = create_session_coordinator(
            session_state_manager=session_state_manager,
            lifecycle_manager=lifecycle_manager,
            memory_manager=memory_manager,
            checkpoint_manager=checkpoint_manager,
            cost_tracker=cost_tracker,
        )
        coordinator.bind_session_service(resolved_session_service)
        return coordinator

    return InteractionRuntimeComponents(
        tool_service=resolved_tool_service,
        session_service=resolved_session_service,
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
