"""Tests for SDK safety pattern declarations."""

import pytest

from victor_sdk.safety import SafetyPatternDeclaration, SafetyPatternType, SafetySeverity


class TestSafetyPatternDeclaration:

    def test_creation(self):
        p = SafetyPatternDeclaration(
            pattern_type=SafetyPatternType.FILE_DELETION,
            severity=SafetySeverity.BLOCK,
            description="Block recursive deletions",
        )
        assert p.pattern_type == SafetyPatternType.FILE_DELETION
        assert p.severity == SafetySeverity.BLOCK

    def test_default_severity_is_warn(self):
        p = SafetyPatternDeclaration(pattern_type=SafetyPatternType.SECRETS_DETECTION)
        assert p.severity == SafetySeverity.WARN

    def test_frozen(self):
        p = SafetyPatternDeclaration(pattern_type=SafetyPatternType.PII_DETECTION)
        with pytest.raises(AttributeError):
            p.severity = SafetySeverity.BLOCK

    def test_all_pattern_types(self):
        assert len(SafetyPatternType) >= 7
        assert SafetyPatternType.CODE_EXECUTION.value == "code_execution"
        assert SafetyPatternType.CUSTOM.value == "custom"

    def test_vertical_can_declare_patterns_without_framework(self):
        """Vertical should be able to declare patterns using only SDK imports."""
        # This test proves no victor.framework import is needed
        patterns = [
            SafetyPatternDeclaration(
                pattern_type=SafetyPatternType.FILE_DELETION,
                severity=SafetySeverity.BLOCK,
            ),
            SafetyPatternDeclaration(
                pattern_type=SafetyPatternType.SECRETS_DETECTION,
                severity=SafetySeverity.WARN,
            ),
        ]
        assert len(patterns) == 2
        assert all(isinstance(p, SafetyPatternDeclaration) for p in patterns)
