# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from unittest.mock import MagicMock

from victor.agent.runtime.coordination_runtime import create_coordination_runtime_components


def test_create_coordination_runtime_components_lazy_materialization():
    factory = MagicMock()
    recovery_coordinator = MagicMock()
    chunk_generator = MagicMock()
    tool_planner = MagicMock()
    task_coordinator = MagicMock()

    factory.create_recovery_coordinator.return_value = recovery_coordinator
    factory.create_chunk_generator.return_value = chunk_generator
    factory.create_tool_planner.return_value = tool_planner
    factory.create_task_coordinator.return_value = task_coordinator

    runtime = create_coordination_runtime_components(factory=factory)

    assert runtime.recovery_coordinator.initialized is False
    assert runtime.chunk_generator.initialized is False
    assert runtime.tool_planner.initialized is False
    assert runtime.task_coordinator.initialized is False

    assert runtime.recovery_coordinator.get_instance() is recovery_coordinator
    assert runtime.chunk_generator.get_instance() is chunk_generator
    assert runtime.tool_planner.get_instance() is tool_planner
    assert runtime.task_coordinator.get_instance() is task_coordinator

    assert runtime.recovery_coordinator.initialized is True
    assert runtime.chunk_generator.initialized is True
    assert runtime.tool_planner.initialized is True
    assert runtime.task_coordinator.initialized is True

    factory.create_recovery_coordinator.assert_called_once_with()
    factory.create_chunk_generator.assert_called_once_with()
    factory.create_tool_planner.assert_called_once_with()
    factory.create_task_coordinator.assert_called_once_with()
