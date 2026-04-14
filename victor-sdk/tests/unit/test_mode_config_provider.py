"""Tests for static SDK mode configuration helpers."""

from victor_sdk.verticals.mode_config import (
    ModeConfig,
    ModeDefinition,
    StaticModeConfigProvider,
    VerticalModeConfig,
)


def test_static_mode_config_provider_returns_mode_configs() -> None:
    """Static providers should expose ModeConfig objects from definitions."""

    provider = StaticModeConfigProvider(
        VerticalModeConfig(
            vertical_name="coding",
            modes={
                "architect": ModeDefinition(
                    name="architect",
                    tool_budget=40,
                    max_iterations=80,
                    exploration_multiplier=2.5,
                ),
                "debug": ModeDefinition(
                    name="debug",
                    tool_budget=15,
                    max_iterations=30,
                ),
            },
            default_mode="debug",
            default_budget=12,
            task_budgets={"bugfix": 18},
        )
    )

    configs = provider.get_mode_configs()

    assert set(configs) == {"architect", "debug"}
    assert isinstance(configs["architect"], ModeConfig)
    assert configs["architect"].tool_budget == 40
    assert configs["architect"].exploration_multiplier == 2.5
    assert provider.get_default_mode() == "debug"


def test_static_mode_config_provider_uses_task_budget_overrides() -> None:
    """Task-specific budget overrides should take precedence over defaults."""

    provider = StaticModeConfigProvider(
        VerticalModeConfig(
            vertical_name="research",
            default_mode="standard",
            default_budget=15,
            task_budgets={"fact_check": 8},
        )
    )

    assert provider.get_default_tool_budget() == 15
    assert provider.get_default_tool_budget("fact_check") == 8
    assert provider.get_tool_budget_for_task("fact_check") == 8
    assert provider.get_tool_budget_for_task("literature_review") == 15
