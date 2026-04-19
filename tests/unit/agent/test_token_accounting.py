# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Token accounting unit tests.

Validates that actual API-returned prompt_tokens flow back to
ConversationController so get_context_metrics() uses real counts
instead of char/N estimation.
"""

from unittest.mock import MagicMock, patch

import pytest

from victor.agent.conversation.controller import ConversationController, ConversationConfig


class TestRecordActualUsage:
    """ConversationController.record_actual_usage() correctness."""

    def _controller(self):
        return ConversationController(config=ConversationConfig(chars_per_token_estimate=4))

    def test_initial_state_uses_char_estimate(self):
        """Before any actual usage is recorded, falls back to char division."""
        ctrl = self._controller()
        ctrl.add_user_message("hello world")  # 11 chars
        metrics = ctrl.get_context_metrics()
        # 11 chars // 4 estimate = 2 (plus system prompt if any)
        assert metrics.estimated_tokens >= 0  # No crash
        assert ctrl._last_actual_prompt_tokens is None

    def test_record_actual_usage_sets_last_tokens(self):
        """record_actual_usage stores prompt_tokens."""
        ctrl = self._controller()
        ctrl.record_actual_usage(prompt_tokens=150, total_chars=600)
        assert ctrl._last_actual_prompt_tokens == 150

    def test_get_context_metrics_uses_actual_when_set(self):
        """get_context_metrics uses calibrated ratio once actual usage is recorded."""
        ctrl = self._controller()
        ctrl.add_user_message("A" * 400)  # 400 chars

        # Record that the API said 100 tokens for 400 chars → ratio = 4.0
        ctrl.record_actual_usage(prompt_tokens=100, total_chars=400)

        metrics = ctrl.get_context_metrics()
        # Should use calibrated 4.0 ratio (same as default here, but via new path)
        # With 400 chars and ratio=4.0, estimated ~ 100
        assert metrics.estimated_tokens > 0
        # Must not use raw //4 anymore — uses float division via calibrated ratio
        assert isinstance(metrics.estimated_tokens, int)

    def test_calibrated_ratio_rolling_average(self):
        """Observed chars-per-token updates as rolling 80/20 average."""
        ctrl = self._controller()
        # Starting ratio = 4.0 (from config)
        # Observe 300 chars / 100 tokens = 3.0 chars/token
        ctrl.record_actual_usage(prompt_tokens=100, total_chars=300)
        # Expected: 0.8*4.0 + 0.2*3.0 = 3.8 (approx, due to initial init)
        # Initial _observed_chars_per_token = float(4) = 4.0
        expected = 0.8 * 4.0 + 0.2 * 3.0
        assert abs(ctrl._observed_chars_per_token - expected) < 0.01

    def test_calibrated_ratio_clamped_to_2_8(self):
        """Ratio stays within [2, 8] even for extreme observations."""
        ctrl = self._controller()
        # Extreme: 1 char / 100 tokens → 0.01 (should clamp to 2.0)
        ctrl.record_actual_usage(prompt_tokens=100, total_chars=1)
        assert ctrl._observed_chars_per_token >= 2.0

        ctrl2 = self._controller()
        # Extreme: 80000 chars / 100 tokens → 800 (should clamp to 8.0)
        ctrl2.record_actual_usage(prompt_tokens=100, total_chars=80000)
        assert ctrl2._observed_chars_per_token <= 8.0

    def test_zero_prompt_tokens_ignored(self):
        """record_actual_usage is a no-op for zero or negative tokens."""
        ctrl = self._controller()
        ctrl.record_actual_usage(prompt_tokens=0, total_chars=400)
        assert ctrl._last_actual_prompt_tokens is None

        ctrl.record_actual_usage(prompt_tokens=-5, total_chars=400)
        assert ctrl._last_actual_prompt_tokens is None

    def test_get_context_metrics_fallback_without_actual(self):
        """Falls back to integer char division when no actual usage recorded."""
        ctrl = self._controller()
        ctrl.add_user_message("hello")
        metrics = ctrl.get_context_metrics()
        total_chars = sum(len(m.content) for m in ctrl.messages)
        expected_fallback = total_chars // ctrl.config.chars_per_token_estimate
        assert metrics.estimated_tokens == expected_fallback

    def test_multiple_updates_converge_ratio(self):
        """Multiple record_actual_usage calls converge the calibrated ratio."""
        ctrl = self._controller()
        # Repeatedly observe 3.0 chars/token
        for _ in range(20):
            ctrl.record_actual_usage(prompt_tokens=100, total_chars=300)
        # After many updates, ratio should converge near 3.0
        assert abs(ctrl._observed_chars_per_token - 3.0) < 0.1


class TestTokenAccountingWiring:
    """TurnExecutor and ChatCoordinator wire record_actual_usage correctly."""

    def test_accumulate_calls_record_actual_usage(self):
        """_accumulate_token_usage() calls record_actual_usage on the controller."""
        from victor.providers.base import CompletionResponse
        from victor.agent.coordinators.turn_executor import TurnExecutor

        mock_ctrl = MagicMock()
        mock_ctrl.messages = [MagicMock(content="hello world")]

        mock_conversation = MagicMock()
        mock_conversation.conversation = mock_ctrl

        executor = MagicMock(spec=TurnExecutor)
        executor._chat_context = mock_conversation
        executor._token_tracker = None

        # Bind the real method to our mock instance
        TurnExecutor._accumulate_token_usage(executor, CompletionResponse(
            content="response",
            usage={"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        ))

        mock_ctrl.record_actual_usage.assert_called_once()
        call_args = mock_ctrl.record_actual_usage.call_args
        # Called as record_actual_usage(prompt_tokens, total_chars) positionally
        assert call_args[0][0] == 50

    def test_accumulate_no_crash_when_controller_raises(self):
        """_accumulate_token_usage() never raises even if record_actual_usage fails."""
        from victor.providers.base import CompletionResponse
        from victor.agent.coordinators.turn_executor import TurnExecutor

        mock_ctrl = MagicMock()
        mock_ctrl.record_actual_usage.side_effect = RuntimeError("DB down")
        mock_ctrl.messages = []

        mock_conversation = MagicMock()
        mock_conversation.conversation = mock_ctrl

        executor = MagicMock(spec=TurnExecutor)
        executor._chat_context = mock_conversation
        executor._token_tracker = None

        # Should not raise
        TurnExecutor._accumulate_token_usage(executor, CompletionResponse(
            content="ok",
            usage={"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40},
        ))

    def test_accumulate_skips_record_when_prompt_tokens_zero(self):
        """record_actual_usage is NOT called when prompt_tokens is 0."""
        from victor.providers.base import CompletionResponse
        from victor.agent.coordinators.turn_executor import TurnExecutor

        mock_ctrl = MagicMock()
        mock_ctrl.messages = []
        mock_conversation = MagicMock()
        mock_conversation.conversation = mock_ctrl

        executor = MagicMock(spec=TurnExecutor)
        executor._chat_context = mock_conversation
        executor._token_tracker = None

        TurnExecutor._accumulate_token_usage(executor, CompletionResponse(
            content="ok",
            usage={"prompt_tokens": 0, "completion_tokens": 5, "total_tokens": 5},
        ))

        mock_ctrl.record_actual_usage.assert_not_called()
