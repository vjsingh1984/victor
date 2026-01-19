# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for common middleware patterns extracted from verticals.

This test module verifies that the framework-level common middleware
provides the same safety enforcement as the vertical-specific implementations.
"""

import pytest

from victor.framework.config import SafetyEnforcer, SafetyConfig, SafetyLevel
from victor.framework.middleware.common_middleware import (
    create_git_safety_rules,
    create_file_operation_safety_rules,
    create_deployment_safety_rules,
    create_container_safety_rules,
    create_infrastructure_safety_rules,
    create_pii_safety_rules,
    create_source_credibility_safety_rules,
    create_content_quality_safety_rules,
    create_bulk_operation_safety_rules,
    create_ingestion_safety_rules,
    create_data_export_safety_rules,
    create_all_common_safety_rules,
)


class TestGitSafetyRules:
    """Tests for git safety rules."""

    def test_block_force_push_to_main(self):
        """Test that force push to main is blocked."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_git_safety_rules(enforcer, block_force_push=True)

        allowed, reason = enforcer.check_operation("git push --force origin main")
        assert not allowed
        assert "force push" in reason.lower()

    def test_allow_normal_push(self):
        """Test that normal push is allowed."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_git_safety_rules(enforcer)

        allowed, _ = enforcer.check_operation("git push origin feature-branch")
        assert allowed

    def test_block_main_push(self):
        """Test that direct push to main is blocked."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_git_safety_rules(enforcer, block_main_push=True)

        allowed, _ = enforcer.check_operation("git push origin main")
        assert not allowed

    def test_custom_protected_branches(self):
        """Test custom protected branches."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_git_safety_rules(
            enforcer, protected_branches=["production", "staging"]
        )

        allowed, _ = enforcer.check_operation("git push --force origin production")
        assert not allowed


class TestFileOperationSafetyRules:
    """Tests for file operation safety rules."""

    def test_block_destructive_commands(self):
        """Test that destructive commands are blocked."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_file_operation_safety_rules(enforcer)

        allowed, _ = enforcer.check_operation("rm -rf /")
        assert not allowed

        allowed, _ = enforcer.check_operation("git clean -fdx")
        assert not allowed

    def test_block_protected_files(self):
        """Test that protected file modification is blocked."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_file_operation_safety_rules(enforcer)

        allowed, _ = enforcer.check_operation("write_file('.env', 'content')")
        assert not allowed

    def test_allow_safe_operations(self):
        """Test that safe operations are allowed."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_file_operation_safety_rules(enforcer)

        allowed, _ = enforcer.check_operation("write_file('README.md', 'content')")
        assert allowed


class TestDeploymentSafetyRules:
    """Tests for deployment safety rules."""

    def test_require_approval_for_production(self):
        """Test that production deployments require approval."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_deployment_safety_rules(enforcer)

        allowed, _ = enforcer.check_operation("deploy to production")
        assert not allowed

    def test_require_backup(self):
        """Test that backup is required before deployment."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_deployment_safety_rules(enforcer, require_backup_before_deploy=True)

        allowed, _ = enforcer.check_operation("kubectl apply -f deployment.yaml")
        assert allowed  # No production environment mentioned

        allowed, _ = enforcer.check_operation("deploy to production")
        assert not allowed  # Production deployment without backup

    def test_custom_protected_environments(self):
        """Test custom protected environments."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_deployment_safety_rules(
            enforcer, protected_environments=["live", "prod-eu"]
        )

        allowed, _ = enforcer.check_operation("deploy to live")
        assert not allowed


class TestContainerSafetyRules:
    """Tests for container safety rules."""

    def test_block_privileged_containers(self):
        """Test that privileged containers are blocked."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_container_safety_rules(enforcer)

        allowed, _ = enforcer.check_operation("docker run --privileged ubuntu")
        assert not allowed

    def test_block_root_user(self):
        """Test that containers running as root are blocked."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_container_safety_rules(enforcer)

        allowed, _ = enforcer.check_operation("USER root")
        assert not allowed

    def test_warn_missing_healthchecks(self):
        """Test that missing health checks generate warnings."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.LOW))
        create_container_safety_rules(enforcer, require_health_checks=True)

        allowed, _ = enforcer.check_operation("docker run nginx")
        # With require_health_checks=True at LOW safety level, should warn but allow
        assert allowed  # Warning only at LOW safety level


class TestInfrastructureSafetyRules:
    """Tests for infrastructure safety rules."""

    def test_block_destructive_commands(self):
        """Test that destructive infrastructure commands are blocked."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_infrastructure_safety_rules(enforcer)

        allowed, _ = enforcer.check_operation("terraform destroy")
        assert not allowed

        allowed, _ = enforcer.check_operation("kubectl delete deployment")
        assert not allowed

    def test_block_protected_resource_deletion(self):
        """Test that protected resource deletion is blocked."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_infrastructure_safety_rules(enforcer)

        allowed, _ = enforcer.check_operation("delete database")
        assert not allowed


class TestPIISafetyRules:
    """Tests for PII safety rules."""

    def test_block_pii_exports(self):
        """Test that PII exports are blocked."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_pii_safety_rules(enforcer)

        allowed, _ = enforcer.check_operation("export df with SSN column")
        assert not allowed

        allowed, _ = enforcer.check_operation("to_csv with credit card data")
        assert not allowed

    def test_warn_on_pii_columns(self):
        """Test that PII columns generate warnings."""
        # LOW level warnings still block at HIGH safety level
        # Use MEDIUM safety level to allow LOW warnings through
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.MEDIUM))
        create_pii_safety_rules(enforcer, warn_on_pii_columns=True)

        # At MEDIUM level, LOW warnings should be allowed
        allowed, reason = enforcer.check_operation("process df with email column")
        assert allowed, f"Should be allowed at MEDIUM safety level. Reason: {reason}"


class TestSourceCredibilitySafetyRules:
    """Tests for source credibility safety rules."""

    def test_block_low_credibility_sources(self):
        """Test that low-credibility sources are blocked."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.MEDIUM))
        create_source_credibility_safety_rules(enforcer)

        allowed, _ = enforcer.check_operation("cite from blogspot.com")
        assert not allowed

    def test_require_source_verification(self):
        """Test that non-.edu/.gov sources require verification."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.LOW))
        create_source_credibility_safety_rules(
            enforcer, require_source_verification=True
        )

        allowed, _ = enforcer.check_operation("cite from randomwebsite.com")
        assert allowed  # Warning only

    def test_block_custom_domains(self):
        """Test blocking custom domains."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_source_credibility_safety_rules(
            enforcer, blocked_domains=["fake-news-site.com"]
        )

        allowed, _ = enforcer.check_operation("cite from fake-news-site.com")
        assert not allowed


class TestContentQualitySafetyRules:
    """Tests for content quality safety rules."""

    def test_block_fabricated_content(self):
        """Test that fabricated content is blocked."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_content_quality_safety_rules(enforcer)

        allowed, _ = enforcer.check_operation("fake source for citation")
        assert not allowed

    def test_warn_absolute_claims(self):
        """Test that absolute claims generate warnings."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.LOW))
        create_content_quality_safety_rules(enforcer)

        allowed, _ = enforcer.check_operation("always true statement")
        assert allowed  # Warning only

    def test_warn_plagiarism(self):
        """Test that plagiarism risks are flagged."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.LOW))
        create_content_quality_safety_rules(enforcer)

        allowed, _ = enforcer.check_operation("copy without attribution")
        assert allowed  # Warning only


class TestBulkOperationSafetyRules:
    """Tests for bulk operation safety rules."""

    def test_block_bulk_delete(self):
        """Test that bulk delete operations are blocked."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_bulk_operation_safety_rules(enforcer)

        allowed, _ = enforcer.check_operation("delete * from table")
        assert not allowed

    def test_block_delete_all(self):
        """Test that delete all operations are blocked."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_bulk_operation_safety_rules(enforcer)

        allowed, _ = enforcer.check_operation("delete --all")
        assert not allowed


class TestIngestionSafetyRules:
    """Tests for ingestion safety rules."""

    def test_block_executable_files(self):
        """Test that executable file ingestion is blocked."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_ingestion_safety_rules(enforcer)

        allowed, _ = enforcer.check_operation("ingest file.exe")
        assert not allowed

    def test_block_system_files(self):
        """Test that system file ingestion is blocked."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_ingestion_safety_rules(enforcer)

        allowed, _ = enforcer.check_operation("ingest /etc/passwd")
        assert not allowed

    def test_require_https(self):
        """Test that HTTPS is required for remote ingestion."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_ingestion_safety_rules(enforcer, require_https=True)

        allowed, _ = enforcer.check_operation("ingest http://example.com/data")
        assert not allowed

        allowed, _ = enforcer.check_operation("ingest https://example.com/data")
        assert allowed


class TestDataExportSafetyRules:
    """Tests for data export safety rules."""

    def test_block_external_uploads(self):
        """Test that external uploads are blocked."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_data_export_safety_rules(enforcer)

        allowed, _ = enforcer.check_operation("upload to s3")
        assert not allowed

    def test_block_production_db_access(self):
        """Test that production DB access is blocked."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_data_export_safety_rules(enforcer)

        allowed, _ = enforcer.check_operation("connect to production database")
        assert not allowed

    def test_require_encryption(self):
        """Test that encryption is required for sensitive data."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_data_export_safety_rules(enforcer, require_encryption=True)

        allowed, _ = enforcer.check_operation("export SSN data to file")
        assert not allowed


class TestCreateAllCommonSafetyRules:
    """Tests for create_all_common_safety_rules convenience function."""

    def test_creates_all_rules(self):
        """Test that all common rules are created."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))

        create_all_common_safety_rules(
            enforcer,
            include_git=True,
            include_file_operations=True,
            include_pii=True,
        )

        # Test that rules from each category are active
        allowed, _ = enforcer.check_operation("git push --force origin main")
        assert not allowed  # Git rule

        allowed, _ = enforcer.check_operation("rm -rf /")
        assert not allowed  # File operation rule

        allowed, _ = enforcer.check_operation("export SSN data")
        assert not allowed  # PII rule

    def test_selective_inclusion(self):
        """Test selective rule inclusion."""
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))

        create_all_common_safety_rules(
            enforcer,
            include_git=True,
            include_file_operations=False,  # Exclude
            include_pii=True,
        )

        # Git rule should be active
        allowed, _ = enforcer.check_operation("git push --force origin main")
        assert not allowed

        # File operation rule should NOT be active
        allowed, _ = enforcer.check_operation("rm -rf /")
        assert allowed  # Not blocked

        # PII rule should be active
        allowed, _ = enforcer.check_operation("export SSN data")
        assert not allowed


class TestVerticalDelegation:
    """Tests that vertical safety.py properly delegates to framework."""

    def test_coding_git_safety_delegates_to_framework(self):
        """Test that coding vertical delegates to framework git rules."""
        from victor.coding.safety import create_git_safety_rules as coding_git_rules

        framework_enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        coding_enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))

        # Create rules using both methods
        from victor.framework.middleware import create_git_safety_rules as framework_git_rules
        framework_git_rules(framework_enforcer)
        coding_git_rules(coding_enforcer)

        # Both should block force push to main
        allowed1, _ = framework_enforcer.check_operation("git push --force origin main")
        allowed2, _ = coding_enforcer.check_operation("git push --force origin main")

        assert allowed1 == allowed2 == False

    def test_devops_deployment_delegates_to_framework(self):
        """Test that devops vertical delegates to framework deployment rules."""
        from victor.devops.safety import (
            create_deployment_safety_rules as devops_deployment_rules,
        )

        framework_enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        devops_enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))

        # Create rules using both methods
        from victor.framework.middleware import (
            create_deployment_safety_rules as framework_deployment_rules,
        )
        framework_deployment_rules(framework_enforcer)
        devops_deployment_rules(devops_enforcer)

        # Both should block production deployment
        allowed1, _ = framework_enforcer.check_operation("deploy to production")
        allowed2, _ = devops_enforcer.check_operation("deploy to production")

        assert allowed1 == allowed2 == False


class TestCodeDuplicationElimination:
    """Tests to verify code duplication has been eliminated."""

    def test_git_safety_rules_consistency(self):
        """Verify that git safety rules are consistent across implementations."""
        from victor.coding.safety import create_git_safety_rules as coding_git

        # Test with same parameters
        coding_enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        framework_enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))

        from victor.framework.middleware import create_git_safety_rules as framework_git

        coding_git(coding_enforcer, block_force_push=True, block_main_push=True)
        framework_git(framework_enforcer, block_force_push=True, block_main_push=True)

        # Test operations
        test_ops = [
            "git push --force origin main",
            "git push origin main",
            "git push origin feature",
        ]

        for op in test_ops:
            allowed1, _ = coding_enforcer.check_operation(op)
            allowed2, _ = framework_enforcer.check_operation(op)
            assert (
                allowed1 == allowed2
            ), f"Inconsistent results for operation: {op}"

    def test_pii_safety_rules_consistency(self):
        """Verify that PII safety rules are consistent across implementations."""
        from victor.dataanalysis.safety import (
            create_dataanalysis_pii_safety_rules as dataanalysis_pii,
        )

        # Test with same parameters
        dataanalysis_enforcer = SafetyEnforcer(
            config=SafetyConfig(level=SafetyLevel.HIGH)
        )
        framework_enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))

        from victor.framework.middleware import create_pii_safety_rules as framework_pii

        dataanalysis_pii(dataanalysis_enforcer, block_pii_exports=True)
        framework_pii(framework_enforcer, block_pii_exports=True)

        # Test operations
        test_ops = [
            "export df with SSN column",
            "to_csv with credit card data",
            "process df with email column",
        ]

        for op in test_ops:
            allowed1, _ = dataanalysis_enforcer.check_operation(op)
            allowed2, _ = framework_enforcer.check_operation(op)
            assert (
                allowed1 == allowed2
            ), f"Inconsistent results for operation: {op}"
