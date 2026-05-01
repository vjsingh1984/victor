from unittest.mock import MagicMock

from victor.agent.streaming import CoordinatorConfig, IterationCoordinator, StreamingChatContext
from victor.agent.streaming.handler import StreamingChatHandler
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


class TestIterationCoordinatorBlockedThresholdSync:
    def test_should_continue_respects_coordinator_blocked_limit_override(self):
        coordinator = IterationCoordinator(
            handler=_make_handler(),
            config=CoordinatorConfig(consecutive_blocked_limit=6, total_blocked_limit=9),
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
            config=CoordinatorConfig(consecutive_blocked_limit=6, total_blocked_limit=9),
        )
        ctx = StreamingChatContext(
            user_message="test",
            consecutive_blocked_attempts=4,
        )

        result = coordinator.pre_iteration_check(ctx)

        assert result is None
        assert ctx.max_blocked_before_force == 6
