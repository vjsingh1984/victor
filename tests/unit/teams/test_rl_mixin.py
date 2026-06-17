from unittest.mock import MagicMock, patch

from victor.teams.mixins.rl import RLMixin


def test_rl_mixin_emits_team_completed_event() -> None:
    """Team RL recording should use the current hook registry emit API."""
    hooks = MagicMock()
    mixin = RLMixin()

    with patch("victor.framework.rl.hooks.get_rl_hooks", return_value=hooks):
        mixin._record_team_rl_outcome(
            team_name="UnifiedTeam",
            formation="hierarchical",
            success=True,
            quality_score=0.8,
            metadata={"member_count": 3},
        )

    hooks.emit.assert_called_once()
    event = hooks.emit.call_args.args[0]
    assert event.type.value == "team_completed"
    assert event.success is True
    assert event.quality_score == 0.8
    assert event.team_formation == "hierarchical"
    assert event.metadata["member_count"] == 3
