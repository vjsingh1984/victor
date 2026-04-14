"""Tests for SDK safety host adapter modules."""

from victor_sdk.safety_patterns import (
    CodePatternScanner,
    InfrastructureScanner,
    SecretScanner,
    validate_source_credibility,
)
from victor_sdk.safety_policy import (
    SafetyConfig,
    SafetyEnforcer,
    SafetyLevel,
    SafetyRule,
    create_file_safety_rules,
    create_git_safety_rules,
)


def test_safety_policy_adapter_exports_host_types() -> None:
    assert SafetyConfig.__name__ == "SafetyConfig"
    assert SafetyEnforcer.__name__ == "SafetyEnforcer"
    assert SafetyLevel.__name__ == "SafetyLevel"
    assert SafetyRule.__name__ == "SafetyRule"
    assert callable(create_file_safety_rules)
    assert callable(create_git_safety_rules)


def test_safety_patterns_adapter_exports_host_helpers() -> None:
    assert CodePatternScanner.__name__ == "CodePatternScanner"
    assert InfrastructureScanner.__name__ == "InfrastructureScanner"
    assert SecretScanner.__name__ == "SecretScanner"
    assert callable(validate_source_credibility)
