"""Interop tests for SDK safety host adapters."""

from victor.framework.config import (
    SafetyConfig as CoreSafetyConfig,
    SafetyEnforcer as CoreSafetyEnforcer,
    SafetyLevel as CoreSafetyLevel,
    SafetyRule as CoreSafetyRule,
)
from victor.framework.safety import (
    create_file_safety_rules as core_create_file_safety_rules,
    create_git_safety_rules as core_create_git_safety_rules,
)
from victor.security.safety.code_patterns import (
    BUILD_DEPLOY_PATTERNS as CoreBuildDeployPatterns,
    CodePatternScanner as CoreCodePatternScanner,
)
from victor.security.safety.infrastructure import (
    InfrastructureScanner as CoreInfrastructureScanner,
)
from victor.security.safety.secrets import SecretScanner as CoreSecretScanner
from victor.security.safety.source_credibility import (
    validate_source_credibility as core_validate_source_credibility,
)
from victor_sdk.safety_patterns import (
    BUILD_DEPLOY_PATTERNS as SdkBuildDeployPatterns,
    CodePatternScanner as SdkCodePatternScanner,
    InfrastructureScanner as SdkInfrastructureScanner,
    SecretScanner as SdkSecretScanner,
    validate_source_credibility as sdk_validate_source_credibility,
)
from victor_sdk.safety_policy import (
    SafetyConfig as SdkSafetyConfig,
    SafetyEnforcer as SdkSafetyEnforcer,
    SafetyLevel as SdkSafetyLevel,
    SafetyRule as SdkSafetyRule,
    create_file_safety_rules as sdk_create_file_safety_rules,
    create_git_safety_rules as sdk_create_git_safety_rules,
)


def test_safety_policy_identity_is_shared() -> None:
    assert CoreSafetyConfig is SdkSafetyConfig
    assert CoreSafetyEnforcer is SdkSafetyEnforcer
    assert CoreSafetyLevel is SdkSafetyLevel
    assert CoreSafetyRule is SdkSafetyRule
    assert sdk_create_file_safety_rules.__name__ == core_create_file_safety_rules.__name__
    assert sdk_create_file_safety_rules.__module__ == core_create_file_safety_rules.__module__
    assert sdk_create_git_safety_rules.__name__ == core_create_git_safety_rules.__name__
    assert sdk_create_git_safety_rules.__module__ == core_create_git_safety_rules.__module__


def test_safety_pattern_identity_is_shared() -> None:
    assert CoreBuildDeployPatterns is SdkBuildDeployPatterns
    assert CoreCodePatternScanner is SdkCodePatternScanner
    assert CoreInfrastructureScanner is SdkInfrastructureScanner
    assert CoreSecretScanner is SdkSecretScanner
    assert core_validate_source_credibility is sdk_validate_source_credibility
