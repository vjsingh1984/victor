from types import SimpleNamespace
from unittest.mock import MagicMock

from victor.agent.streaming import (
    CoordinatorConfig,
    IterationCoordinator,
    StreamingChatContext,
)
from victor.agent.streaming.handler import StreamingChatHandler
from victor.agent.streaming.iteration import IterationAction, IterationResult
from victor.core.loop_thresholds import (
    DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD,
    DEFAULT_BLOCKED_TOTAL_THRESHOLD,
)


def _make_handler() -> StreamingChatHandler:
    settings = MagicMock()
    message_adder = MagicMock()
    return StreamingChatHandler(settings=settings, message_adder=message_adder)


class TestCoordinatorConfig:
    def test_defaults_match_shared_blocked_loop_thresholds(self):
        config = CoordinatorConfig()

        assert config.consecutive_blocked_limit == DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD
        assert config.total_blocked_limit == DEFAULT_BLOCKED_TOTAL_THRESHOLD

    def test_settings_seed_budget_warning_configuration(self):
        settings = SimpleNamespace(
            tool_call_budget_warning_threshold=12,
            tool_call_budget_warning_pct=0.6,
            tool_call_budget_warning_remaining=3,
        )

        coordinator = IterationCoordinator(handler=_make_handler(), settings=settings)

        assert coordinator.config.budget_warning_threshold == 12
        assert coordinator.config.budget_warning_pct == 0.6
        assert coordinator.config.budget_warning_remaining == 3


class TestIterationCoordinatorBlockedThresholdSync:
    def test_should_continue_respects_coordinator_blocked_limit_override(self):
        coordinator = IterationCoordinator(
            handler=_make_handler(),
            config=CoordinatorConfig(
                consecutive_blocked_limit=6, total_blocked_limit=9
            ),
        )
        ctx = StreamingChatContext(
            user_message="test",
            consecutive_blocked_attempts=4,
        )

        assert coordinator.should_continue(ctx) is True
        assert ctx.max_blocked_before_force == 6

    def test_pre_iteration_check_respects_coordinator_blocked_limit_override(self):
        coordinator = IterationCoordinator(
            handler=_make_handler(),
            config=CoordinatorConfig(
                consecutive_blocked_limit=6, total_blocked_limit=9
            ),
        )
        ctx = StreamingChatContext(
            user_message="test",
            consecutive_blocked_attempts=4,
        )

        result = coordinator.pre_iteration_check(ctx)

        assert result is None
        assert ctx.max_blocked_before_force == 6

    def test_post_iteration_check_passes_configured_budget_warning_policy(self):
        handler = MagicMock()
        handler.check_tool_budget.return_value = None
        handler.check_progress_and_force.return_value = False
        coordinator = IterationCoordinator(
            handler=handler,
            settings=SimpleNamespace(
                tool_call_budget_warning_threshold=12,
                tool_call_budget_warning_pct=0.6,
                tool_call_budget_warning_remaining=3,
            ),
        )
        ctx = StreamingChatContext(
            user_message="test", tool_budget=10, tool_calls_used=8
        )

        result = coordinator.post_iteration_check(
            ctx,
            IterationResult(action=IterationAction.CONTINUE),
        )

        assert result is None
        handler.check_tool_budget.assert_called_once_with(ctx, 12, 0.6, 3)
