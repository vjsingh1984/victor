# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Interaction runtime boundaries for AgentOrchestrator."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING, cast

from victor.agent.runtime.provider_runtime import LazyRuntimeProxy

if TYPE_CHECKING:
    from victor.agent.services.protocols.chat_runtime import ChatCompatRuntimeProtocol


@dataclass(frozen=True)
class InteractionRuntimeComponents:
    """Interaction runtime handles exposed to the orchestrator facade."""

    chat_service: Any
    tool_service: Any
    session_service: Any
    context_service: Any
    recovery_service: Any


def create_chat_coordinator_shim(
    *,
    runtime: "ChatCompatRuntimeProtocol",
    chat_service: Optional[Any] = None,
    get_chat_service: Optional[Any] = None,
) -> LazyRuntimeProxy[Any]:
    """Create the deprecated ChatCoordinator compatibility shim.

    This helper is intentionally separate from ``InteractionRuntimeComponents``
    so the primary interaction runtime contract stays service-first.
    """

    def _build_chat_coordinator() -> Any:
        warnings.warn(
            "ChatCoordinator shim is deprecated. Use ChatService (self._chat_service) "
            "directly. The shim will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        from victor.agent.services.chat_compat import ChatCoordinator

        # The service-first shim binds ChatService immediately, so the adapter only
        # needs the public compatibility runtime surface here.
        coordinator = ChatCoordinator(cast(Any, runtime))
        if get_chat_service is not None and hasattr(coordinator, "bind_chat_service_getter"):
            coordinator.bind_chat_service_getter(get_chat_service)
        elif chat_service is not None and hasattr(coordinator, "bind_chat_service"):
            coordinator.bind_chat_service(chat_service)
        return coordinator

    return LazyRuntimeProxy(
        factory=_build_chat_coordinator,
        name="chat_coordinator",
    )


def create_tool_coordinator_shim(
    *,
    orchestrator: Any,
    tool_pipeline: Any,
    tool_registry: Any,
    tool_selector: Any,
    tool_access_controller: Any,
    mode_controller: Any,
    tool_service: Optional[Any] = None,
) -> LazyRuntimeProxy[Any]:
    """Create the deprecated ToolCoordinator compatibility shim.

    This helper is intentionally separate from ``InteractionRuntimeComponents``
    so the main interaction runtime contract stays service-first.
    """

    orch_tools = getattr(orchestrator, "_enabled_tools", None)

    def _build_tool_coordinator() -> Any:
        warnings.warn(
            "ToolCoordinator shim is deprecated. Use ToolService (self._tool_service) "
            "directly. The shim will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        from victor.agent.services.tool_compat import ToolCoordinator

        coordinator = ToolCoordinator(
            tool_pipeline=tool_pipeline,
            tool_registry=tool_registry,
            tool_selector=tool_selector,
            tool_access_controller=tool_access_controller,
            warn_on_init=False,
        )
        coordinator.set_mode_controller(mode_controller)
        if tool_service is not None:
            coordinator.bind_tool_service(tool_service)
        if orch_tools:
            coordinator.set_enabled_tools(orch_tools)
        return coordinator

    return LazyRuntimeProxy(
        factory=_build_tool_coordinator,
        name="tool_coordinator",
    )


def create_session_coordinator_shim(
    *,
    session_state_manager: Any,
    lifecycle_manager: Any,
    memory_manager: Any,
    checkpoint_manager: Any,
    cost_tracker: Any,
    session_service: Optional[Any] = None,
) -> LazyRuntimeProxy[Any]:
    """Create the deprecated SessionCoordinator compatibility shim.

    This helper is intentionally separate from ``InteractionRuntimeComponents``
    so the primary interaction runtime contract stays service-first.
    """

    def _build_session_coordinator() -> Any:
        warnings.warn(
            "SessionCoordinator shim is deprecated. Use SessionService "
            "(self._session_service) directly. The shim will be removed in a "
            "future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        from victor.agent.services.session_compat import create_session_coordinator

        coordinator = create_session_coordinator(
            session_state_manager=session_state_manager,
            lifecycle_manager=lifecycle_manager,
            memory_manager=memory_manager,
            checkpoint_manager=checkpoint_manager,
            cost_tracker=cost_tracker,
            warn_on_init=False,
        )
        if session_service is not None and hasattr(coordinator, "bind_session_service"):
            coordinator.bind_session_service(session_service)
        return coordinator

    return LazyRuntimeProxy(
        factory=_build_session_coordinator,
        name="session_coordinator",
    )


def create_interaction_runtime_components(
    *,
    enabled_tools: Optional[Any],
    factory: Any,
    tool_pipeline: Any,
    tool_registry: Any,
    tool_executor: Any,
    tool_cache: Any,
    tool_budget: int,
    tool_selector: Any,
    tool_access_controller: Any,
    mode_controller: Any,
    argument_normalizer: Any,
    session_state_manager: Any,
    lifecycle_manager: Any,
    memory_manager: Any,
    memory_session_id: Optional[str],
    checkpoint_manager: Any,
    cost_tracker: Any,
    conversation_controller: Any,
    streaming_coordinator: Any,
    provider_service: Optional[Any] = None,
    chat_service: Optional[Any] = None,
    context_service: Optional[Any] = None,
    recovery_service: Optional[Any] = None,
    tool_service: Optional[Any] = None,
    session_service: Optional[Any] = None,
) -> InteractionRuntimeComponents:
    """Create service-first interaction/runtime components for orchestrator wiring."""
    from victor.agent.services.chat_service import ChatService, ChatServiceConfig
    from victor.agent.services.context_service import (
        ContextService,
        ContextServiceConfig,
    )
    from victor.agent.services.recovery_service import RecoveryService
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
            tool_pipeline=tool_pipeline,
            tool_cache=tool_cache,
            mode_controller=mode_controller,
            argument_normalizer=argument_normalizer,
        )

    if enabled_tools and hasattr(resolved_tool_service, "set_enabled_tools"):
        resolved_tool_service.set_enabled_tools(enabled_tools)

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

    resolved_context_service = context_service
    if resolved_context_service is None:
        resolved_context_service = ContextService(
            config=ContextServiceConfig(
                max_tokens=100000,
                overflow_threshold_percent=90.0,
            )
        )

    resolved_recovery_service = recovery_service
    if resolved_recovery_service is None:
        resolved_recovery_service = RecoveryService()

    resolved_chat_service = chat_service
    if resolved_chat_service is None:
        resolved_chat_service = ChatService(
            config=ChatServiceConfig(
                max_iterations=200,
                stream_chunk_size=100,
            ),
            provider_service=provider_service,
            tool_service=resolved_tool_service,
            context_service=resolved_context_service,
            recovery_service=resolved_recovery_service,
            conversation_controller=conversation_controller,
            streaming_coordinator=streaming_coordinator,
        )

    return InteractionRuntimeComponents(
        chat_service=resolved_chat_service,
        tool_service=resolved_tool_service,
        session_service=resolved_session_service,
        context_service=resolved_context_service,
        recovery_service=resolved_recovery_service,
    )
