# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from unittest.mock import MagicMock, patch

from victor.agent.runtime.interaction_runtime import create_interaction_runtime_components


def test_create_interaction_runtime_components_lazy_materialization():
    orchestrator = MagicMock()
    tool_pipeline = MagicMock()
    tool_registry = MagicMock()
    tool_selector = MagicMock()
    tool_access_controller = MagicMock()
    mode_controller = MagicMock()
    session_state_manager = MagicMock()
    lifecycle_manager = MagicMock()
    memory_manager = MagicMock()
    checkpoint_manager = MagicMock()
    cost_tracker = MagicMock()

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

                runtime = create_interaction_runtime_components(
                    orchestrator=orchestrator,
                    tool_pipeline=tool_pipeline,
                    tool_registry=tool_registry,
                    tool_selector=tool_selector,
                    tool_access_controller=tool_access_controller,
                    mode_controller=mode_controller,
                    session_state_manager=session_state_manager,
                    lifecycle_manager=lifecycle_manager,
                    memory_manager=memory_manager,
                    checkpoint_manager=checkpoint_manager,
                    cost_tracker=cost_tracker,
                )

                assert runtime.chat_coordinator.initialized is False
                assert runtime.tool_coordinator.initialized is False
                assert runtime.session_coordinator.initialized is False

                assert runtime.chat_coordinator.get_instance() is chat_coordinator
                assert runtime.tool_coordinator.get_instance() is tool_coordinator
                assert runtime.session_coordinator.get_instance() is session_coordinator

    chat_cls.assert_called_once_with(orchestrator)
    tool_cls.assert_called_once_with(
        tool_pipeline=tool_pipeline,
        tool_registry=tool_registry,
        tool_selector=tool_selector,
        tool_access_controller=tool_access_controller,
    )
    tool_coordinator.set_mode_controller.assert_called_once_with(mode_controller)
    tool_coordinator.set_orchestrator_reference.assert_called_once_with(orchestrator)
    session_factory.assert_called_once_with(
        session_state_manager=session_state_manager,
        lifecycle_manager=lifecycle_manager,
        memory_manager=memory_manager,
        checkpoint_manager=checkpoint_manager,
        cost_tracker=cost_tracker,
    )
