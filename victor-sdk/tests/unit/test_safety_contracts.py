"""Tests for SDK-owned safety contracts."""

from victor_sdk.safety import (
    SafetyAction,
    SafetyCategory,
    SafetyCoordinator,
    SafetyRule,
)


def test_safety_coordinator_applies_sdk_rules() -> None:
    coordinator = SafetyCoordinator(enable_default_rules=False)
    coordinator.register_rule(
        SafetyRule(
            rule_id="demo",
            category=SafetyCategory.SHELL,
            pattern=r"danger",
            description="Dangerous command",
            action=SafetyAction.BLOCK,
            tool_names=["shell"],
        )
    )

    result = coordinator.check_safety("shell", ["danger"])

    assert result.is_safe is False
    assert result.action is SafetyAction.BLOCK
    assert result.block_reason == "Dangerous command"


def test_safety_rules_normalize_legacy_tool_aliases() -> None:
    coordinator = SafetyCoordinator(enable_default_rules=False)
    coordinator.register_rule(
        SafetyRule(
            rule_id="system_write",
            category=SafetyCategory.FILE,
            pattern=r"/etc/",
            description="Write to system directory",
            action=SafetyAction.BLOCK,
            tool_names=["write_file"],
        )
    )

    result = coordinator.check_safety("write", ["/etc/hosts", "payload"])

    assert result.is_safe is False
    assert result.action is SafetyAction.BLOCK
    assert result.block_reason == "Write to system directory"
