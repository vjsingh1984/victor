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

"""Focused sync-wrapper tests for CQRS bridge compatibility paths."""

from unittest.mock import MagicMock, patch

from victor.framework import cqrs_bridge as cqrs_bridge_module


class TestCQRSBridgeSyncWrappers:
    def test_framework_adapter_forwards_to_cqrs_via_shared_sync_bridge(self) -> None:
        dispatcher = MagicMock()
        awaitable = object()
        dispatcher.dispatch.return_value = awaitable
        adapter = cqrs_bridge_module.FrameworkEventAdapter(
            event_dispatcher=dispatcher,
            session_id="session-123",
        )

        with (
            patch.object(
                cqrs_bridge_module,
                "framework_event_to_cqrs",
                return_value={
                    "event_type": "tool_called",
                    "tool_name": "read_file",
                    "arguments": {"path": "/tmp/example.py"},
                    "metadata": {"source": "test"},
                },
            ),
            patch("victor.core.event_sourcing.ToolCalledEvent", return_value="cqrs-event") as mock_event,
            patch.object(
                cqrs_bridge_module.asyncio,
                "get_running_loop",
                side_effect=RuntimeError,
            ),
            patch.object(cqrs_bridge_module, "run_sync", return_value=None) as mock_run_sync,
        ):
            adapter._forward_to_cqrs(MagicMock())

        mock_event.assert_called_once_with(
            task_id="session-123",
            tool_name="read_file",
            arguments={"path": "/tmp/example.py"},
            metadata={"source": "test"},
        )
        dispatcher.dispatch.assert_called_once_with("cqrs-event")
        mock_run_sync.assert_called_once_with(awaitable)

    def test_framework_adapter_forwards_to_observability_via_shared_sync_bridge(self) -> None:
        bus = MagicMock()
        awaitable = object()
        bus.emit.return_value = awaitable
        adapter = cqrs_bridge_module.FrameworkEventAdapter(event_bus=MagicMock())

        with (
            patch.object(
                cqrs_bridge_module,
                "framework_event_to_observability",
                return_value={
                    "category": "tool",
                    "name": "complete",
                    "data": {"tool_name": "read_file"},
                },
            ),
            patch("victor.core.events.get_observability_bus", return_value=bus),
            patch.object(
                cqrs_bridge_module.asyncio,
                "get_running_loop",
                side_effect=RuntimeError,
            ),
            patch.object(cqrs_bridge_module, "run_sync", return_value=None) as mock_run_sync,
        ):
            adapter._forward_to_observability(MagicMock())

        bus.emit.assert_called_once_with(
            topic="tool.complete",
            data={"tool_name": "read_file", "category": "tool"},
        )
        mock_run_sync.assert_called_once_with(awaitable)

    def test_observability_bridge_dispatches_via_shared_sync_bridge(self) -> None:
        dispatcher = MagicMock()
        awaitable = object()
        dispatcher.dispatch.return_value = awaitable
        bridge = cqrs_bridge_module.ObservabilityToCQRSBridge(
            event_bus=MagicMock(),
            event_dispatcher=dispatcher,
            aggregate_id="obs-1",
        )

        with (
            patch.object(cqrs_bridge_module, "observability_event_to_framework", return_value=MagicMock()),
            patch.object(
                cqrs_bridge_module,
                "framework_event_to_cqrs",
                return_value={
                    "event_type": "tool_result",
                    "tool_name": "grep",
                    "success": True,
                    "result": "ok",
                    "metadata": {"origin": "test"},
                },
            ),
            patch("victor.core.event_sourcing.ToolResultEvent", return_value="cqrs-event") as mock_event,
            patch.object(
                cqrs_bridge_module.asyncio,
                "get_running_loop",
                side_effect=RuntimeError,
            ),
            patch.object(cqrs_bridge_module, "run_sync", return_value=None) as mock_run_sync,
        ):
            bridge._handle_event("tool.complete", {"tool_name": "grep"})

        mock_event.assert_called_once_with(
            task_id="obs-1",
            tool_name="grep",
            success=True,
            result="ok",
            metadata={
                "original_category": "tool",
                "original_name": "tool.complete",
                "origin": "test",
            },
        )
        dispatcher.dispatch.assert_called_once_with("cqrs-event")
        mock_run_sync.assert_called_once_with(awaitable)
        assert bridge.event_count == 1
