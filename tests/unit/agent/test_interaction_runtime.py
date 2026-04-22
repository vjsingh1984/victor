# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from unittest.mock import MagicMock, patch

from victor.agent.runtime.interaction_runtime import (
    create_interaction_runtime_components,
)


def test_create_interaction_runtime_components_lazy_materialization():
    orchestrator = MagicMock()
    orchestrator._enabled_tools = None
    tool_pipeline = MagicMock()
    tool_registry = MagicMock()
    tool_executor = MagicMock()
    tool_selector = MagicMock()
    tool_access_controller = MagicMock()
    mode_controller = MagicMock()
    session_state_manager = MagicMock()
    lifecycle_manager = MagicMock()
    memory_manager = MagicMock()
    memory_session_id = "mem-123"
    checkpoint_manager = MagicMock()
    cost_tracker = MagicMock()
    tool_service = MagicMock()
    session_service = MagicMock()
    factory = MagicMock()
    pipeline = MagicMock()
    factory.create_streaming_chat_pipeline.return_value = pipeline

    with patch("victor.agent.coordinators.chat_coordinator.ChatCoordinator") as chat_cls:
        with patch("victor.agent.coordinators.tool_coordinator.ToolCoordinator") as tool_cls:
            with patch(
                "victor.agent.coordinators.session_coordinator.create_session_coordinator"
            ) as session_factory:
                chat_coordinator = MagicMock()
                tool_coordinator = MagicMock()
                session_coordinator = MagicMock()
                chat_cls.return_value = chat_coordinator
                tool_cls.return_value = tool_coordinator
                session_factory.return_value = session_coordinator

                chat_coordinator.set_streaming_pipeline = MagicMock()

                runtime = create_interaction_runtime_components(
                    orchestrator=orchestrator,
                    factory=factory,
                    tool_pipeline=tool_pipeline,
                    tool_registry=tool_registry,
                    tool_executor=tool_executor,
                    tool_budget=25,
                    tool_selector=tool_selector,
                    tool_access_controller=tool_access_controller,
                    mode_controller=mode_controller,
                    session_state_manager=session_state_manager,
                    lifecycle_manager=lifecycle_manager,
                    memory_manager=memory_manager,
                    memory_session_id=memory_session_id,
                    checkpoint_manager=checkpoint_manager,
                    cost_tracker=cost_tracker,
                    tool_service=tool_service,
                    session_service=session_service,
                )

                assert runtime.tool_service is tool_service
                assert runtime.session_service is session_service
                assert runtime.chat_coordinator.initialized is False
                assert runtime.tool_coordinator.initialized is False
                assert runtime.session_coordinator.initialized is False

                assert runtime.chat_coordinator.get_instance() is chat_coordinator
                assert runtime.tool_coordinator.get_instance() is tool_coordinator
                assert runtime.session_coordinator.get_instance() is session_coordinator

    chat_cls.assert_called_once_with(orchestrator)
    factory.create_streaming_chat_pipeline.assert_called_once_with(chat_coordinator)
    chat_coordinator.set_streaming_pipeline.assert_called_once_with(pipeline)
    tool_cls.assert_called_once_with(
        tool_pipeline=tool_pipeline,
        tool_registry=tool_registry,
        tool_selector=tool_selector,
        tool_access_controller=tool_access_controller,
    )
    tool_coordinator.set_mode_controller.assert_called_once_with(mode_controller)
    tool_coordinator.bind_tool_service.assert_called_once_with(tool_service)
    # set_orchestrator_reference replaced with direct set_enabled_tools
    # (protocol-based injection — no orchestrator reference passed)
    session_factory.assert_called_once_with(
        session_state_manager=session_state_manager,
        lifecycle_manager=lifecycle_manager,
        memory_manager=memory_manager,
        checkpoint_manager=checkpoint_manager,
        cost_tracker=cost_tracker,
    )
    session_coordinator.bind_session_service.assert_called_once_with(session_service)
