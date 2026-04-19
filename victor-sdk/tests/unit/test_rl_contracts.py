"""Tests for SDK-owned RL contracts."""

from victor_sdk.rl import (
    BaseRLConfig,
    DEFAULT_ACTIVE_LEARNERS,
    DEFAULT_PATIENCE_MAP,
    LearnerType,
)


def test_learner_type_values_remain_stable() -> None:
    assert LearnerType.TOOL_SELECTOR.value == "tool_selector"
    assert LearnerType.MODE_TRANSITION.value == "mode_transition"


def test_base_rl_config_uses_shared_defaults() -> None:
    config = BaseRLConfig()

    assert config.active_learners == DEFAULT_ACTIVE_LEARNERS
    assert config.default_patience == DEFAULT_PATIENCE_MAP
    assert config.get_tools_for_task("missing") == []
    assert config.get_quality_threshold("missing") == 0.80
    assert config.get_patience("unknown") == 4
