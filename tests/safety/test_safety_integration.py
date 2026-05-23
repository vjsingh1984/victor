# Tests for victor-coding safety rules
# Migrated from victor/tests/unit/framework/test_config.py


from victor.framework.config import SafetyConfig, SafetyEnforcer, SafetyLevel


class TestCodingAllSafetyRules:
    """Tests for all Coding safety rules combined."""

    def test_create_all_coding_safety_rules(self):
        """create_all_coding_safety_rules should register all coding rules."""
        from victor_coding.safety import create_all_coding_safety_rules

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_all_coding_safety_rules(enforcer)

        # Should have rules from git, file, and test categories
        assert len(enforcer.rules) > 0

        # Verify force push is blocked
        allowed, _ = enforcer.check_operation("git push --force origin main")
        assert allowed is False
