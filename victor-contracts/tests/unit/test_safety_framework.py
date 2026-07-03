"""Tests for SDK framework-style safety compatibility helpers."""

import pytest

from victor_contracts.safety.framework import (
    SafetyConfig,
    SafetyEnforcer,
    SafetyLevel,
    SafetyRule,
)


def test_framework_safety_enforcer_blocks_matching_high_rule() -> None:
    enforcer = SafetyEnforcer(SafetyConfig(level=SafetyLevel.HIGH))
    enforcer.add_rule(
        SafetyRule(
            name="block_delete",
            description="Block delete",
            check_fn=lambda operation: "delete" in operation,
            level=SafetyLevel.HIGH,
        )
    )

    allowed, reason = enforcer.check_operation("delete all")

    assert allowed is False
    assert reason == "Blocked by safety rule: block_delete - Block delete"


def test_framework_safety_enforcer_rejects_duplicate_rules() -> None:
    enforcer = SafetyEnforcer(SafetyConfig())
    rule = SafetyRule(
        name="duplicate",
        description="Duplicate",
        check_fn=lambda operation: False,
    )

    enforcer.add_rule(rule)

    with pytest.raises(ValueError):
        enforcer.add_rule(rule)


def test_framework_safety_config_round_trips_dict() -> None:
    config = SafetyConfig.from_dict(
        {
            "level": "low",
            "require_confirmation": True,
            "blocked_operations": ["upload"],
            "audit_log": False,
            "dry_run": True,
        }
    )

    assert config.level is SafetyLevel.LOW
    assert config.to_dict() == {
        "level": "low",
        "require_confirmation": True,
        "blocked_operations": ["upload"],
        "audit_log": False,
        "dry_run": True,
    }


def test_framework_safety_rule_coerces_to_host_level_when_available() -> None:
    pytest.importorskip("victor", reason="host-level coercion requires the victor-ai package")
    from victor.framework.config import SafetyLevel as HostSafetyLevel

    rule = SafetyRule(
        name="host",
        description="Host level",
        check_fn=lambda operation: True,
        level=SafetyLevel.HIGH,
    )

    assert rule.level == HostSafetyLevel.HIGH
