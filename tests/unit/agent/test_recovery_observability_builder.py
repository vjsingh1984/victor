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

"""Tests for RecoveryObservabilityBuilder."""

from unittest.mock import MagicMock

from victor.agent.builders.recovery_observability_builder import (
    RecoveryObservabilityBuilder,
)
from victor.agent.coordinators.chat_coordinator import ChatCoordinator
from victor.agent.coordinators.checkpoint_coordinator import CheckpointCoordinator
from victor.agent.coordinators.validation_coordinator import ValidationCoordinator


def test_recovery_observability_builder_wires_components():
    """RecoveryObservabilityBuilder assigns recovery and observability components."""
    settings = MagicMock()
    factory = MagicMock()
    factory.create_recovery_handler.return_value = "recovery-handler"
    factory.create_recovery_integration.return_value = "recovery-integration"
    factory.create_recovery_coordinator.return_value = "recovery-coordinator"
    factory.create_chunk_generator.return_value = "chunk-generator"
    factory.create_tool_planner.return_value = "tool-planner"
    factory.create_task_coordinator.return_value = "task-coordinator"
    factory.create_observability.return_value = "observability"
    factory.create_checkpoint_manager.return_value = "checkpoint-manager"

    orchestrator = MagicMock()
    orchestrator._context_manager = "context-manager"
    orchestrator._response_coordinator = "response-coordinator"
    orchestrator._metrics_coordinator = "metrics-coordinator"
    orchestrator._get_checkpoint_state = MagicMock(return_value={})
    orchestrator._apply_checkpoint_state = MagicMock()

    builder = RecoveryObservabilityBuilder(settings=settings, factory=factory)
    components = builder.build(orchestrator)

    assert orchestrator._recovery_handler == "recovery-handler"
    assert orchestrator._recovery_integration == "recovery-integration"
    assert orchestrator._recovery_coordinator == "recovery-coordinator"
    assert orchestrator._chunk_generator == "chunk-generator"
    assert orchestrator._tool_planner == "tool-planner"
    assert orchestrator._task_coordinator == "task-coordinator"
    assert orchestrator._observability == "observability"
    assert orchestrator._checkpoint_manager == "checkpoint-manager"
    assert isinstance(orchestrator._checkpoint_coordinator, CheckpointCoordinator)
    assert isinstance(orchestrator._chat_coordinator, ChatCoordinator)
    assert isinstance(orchestrator._validation_coordinator, ValidationCoordinator)
    assert components["observability"] == "observability"
