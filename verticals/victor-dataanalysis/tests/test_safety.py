# Tests for victor-dataanalysis safety rules
# Migrated from victor/tests/unit/framework/test_config.py

import pytest

from victor.framework.config import SafetyConfig, SafetyEnforcer, SafetyLevel


class TestDataAnalysisPIISafety:
    """Tests for DataAnalysis PII safety rules."""

    def test_dataanalysis_safety_pii_rules(self):
        """DataAnalysis PII safety rules should block PII exports."""
        from victor_dataanalysis.safety import create_dataanalysis_pii_safety_rules

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_dataanalysis_pii_safety_rules(
            enforcer,
            block_pii_exports=True,
            warn_on_pii_columns=True,
        )

        # Test PII export blocking
        allowed, reason = enforcer.check_operation("export data with SSN to CSV")
        assert allowed is False
        assert "pii" in reason.lower() or "ssn" in reason.lower() or "blocked" in reason.lower()

        # Test credit card export blocking (use space, not underscore)
        allowed, reason = enforcer.check_operation("to_csv credit card data")
        assert allowed is False
        assert "credit" in reason.lower() or "pii" in reason.lower() or "blocked" in reason.lower()


class TestDataAnalysisExportSafety:
    """Tests for DataAnalysis export safety rules."""

    def test_dataanalysis_safety_export_rules(self):
        """DataAnalysis export safety rules should block external uploads."""
        from victor_dataanalysis.safety import create_dataanalysis_export_safety_rules

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_dataanalysis_export_safety_rules(
            enforcer,
            block_external_uploads=True,
            block_production_db_access=True,
        )

        # Test external upload blocking (use "upload to" without word in between)
        allowed, reason = enforcer.check_operation("upload to s3 bucket")
        assert allowed is False
        assert "external" in reason.lower() or "s3" in reason.lower() or "blocked" in reason.lower()

        # Test production DB access blocking
        allowed, reason = enforcer.check_operation("query production database")
        assert allowed is False
        assert "production" in reason.lower() or "blocked" in reason.lower()


class TestDataAnalysisAllSafetyRules:
    """Tests for all DataAnalysis safety rules combined."""

    def test_create_all_dataanalysis_safety_rules(self):
        """create_all_dataanalysis_safety_rules should register all dataanalysis rules."""
        from victor_dataanalysis.safety import create_all_dataanalysis_safety_rules

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_all_dataanalysis_safety_rules(enforcer)

        # Should have rules from PII and export categories
        assert len(enforcer.rules) > 0

        # Verify PII export is blocked
        allowed, _ = enforcer.check_operation("export data with SSN")
        assert allowed is False

        # Verify external upload is blocked
        allowed, _ = enforcer.check_operation("upload to dropbox")
        assert allowed is False
