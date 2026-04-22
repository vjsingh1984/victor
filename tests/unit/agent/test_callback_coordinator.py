"""Tests for CallbackCoordinator."""

import pytest
from unittest.mock import MagicMock

from victor.agent.callback_coordinator import CallbackCoordinator


class TestCallbackCoordinator:
    def setup_method(self):
        self.metrics = MagicMock()
        self.tool_service = MagicMock()
        self.observability = MagicMock()
        self.analytics = MagicMock()
        self.rl_coordinator = MagicMock()
        self.vertical_context = MagicMock()

        self.coordinator = CallbackCoordinator(
            metrics_coordinator=self.metrics,
            get_observability=lambda: self.observability,
            get_pipeline_calls_used=lambda: 5,
            get_usage_analytics=lambda: self.analytics,
            get_rl_coordinator=lambda: self.rl_coordinator,
            get_vertical_context=lambda: self.vertical_context,
            get_tool_service=lambda: self.tool_service,
        )

    def test_on_tool_start_delegates_to_metrics(self):
        self.coordinator.on_tool_start("read", {"path": "/tmp"})
        self.metrics.on_tool_start.assert_called_once_with("read", {"path": "/tmp"}, 5)

    def test_on_tool_start_emits_observability_event(self):
        self.coordinator.on_tool_start("write", {"content": "hello"})
        self.observability.on_tool_start.assert_called_once_with(
            "write", {"content": "hello"}, "tool-5"
        )

    def test_on_tool_start_no_observability(self):
        coordinator = CallbackCoordinator(
            metrics_coordinator=self.metrics,
            get_observability=lambda: None,
            get_pipeline_calls_used=lambda: 0,
            get_usage_analytics=lambda: None,
            get_rl_coordinator=lambda: self.rl_coordinator,
            get_vertical_context=lambda: self.vertical_context,
        )
        coordinator.on_tool_start("read", {})
        # Should not raise

    def test_on_tool_complete_prefers_tool_service(self):
        result = MagicMock()
        nudge_flag = [False]
        self.coordinator.on_tool_complete(
            result,
            read_files_session=set(),
            required_files=[],
            required_outputs=[],
            nudge_sent_flag=nudge_flag,
            add_message=MagicMock(),
        )
        self.tool_service.on_tool_complete.assert_called_once()

    def test_on_tool_complete_raises_without_service(self):
        coordinator = CallbackCoordinator(
            metrics_coordinator=self.metrics,
            get_observability=lambda: self.observability,
            get_pipeline_calls_used=lambda: 5,
            get_usage_analytics=lambda: self.analytics,
            get_rl_coordinator=lambda: self.rl_coordinator,
            get_vertical_context=lambda: self.vertical_context,
            get_tool_service=lambda: None,
        )

        with pytest.raises(
            RuntimeError,
            match="ToolService is unavailable or does not expose on_tool_complete",
        ):
            coordinator.on_tool_complete(
                MagicMock(),
                read_files_session=set(),
                required_files=[],
                required_outputs=[],
                nudge_sent_flag=[False],
                add_message=MagicMock(),
            )

    def test_on_streaming_session_complete_full_flow(self):
        session = MagicMock()
        self.coordinator.on_streaming_session_complete(session)
        self.metrics.on_streaming_session_complete.assert_called_once_with(session)
        self.analytics.end_session.assert_called_once()
        self.metrics.send_rl_reward_signal.assert_called_once_with(
            session=session,
            rl_coordinator=self.rl_coordinator,
            vertical_context=self.vertical_context,
        )

    def test_on_streaming_session_complete_no_analytics(self):
        coordinator = CallbackCoordinator(
            metrics_coordinator=self.metrics,
            get_observability=lambda: None,
            get_pipeline_calls_used=lambda: 0,
            get_usage_analytics=lambda: None,
            get_rl_coordinator=lambda: self.rl_coordinator,
            get_vertical_context=lambda: self.vertical_context,
        )
        session = MagicMock()
        coordinator.on_streaming_session_complete(session)
        # Should not raise even without analytics
        self.metrics.on_streaming_session_complete.assert_called_once_with(session)
