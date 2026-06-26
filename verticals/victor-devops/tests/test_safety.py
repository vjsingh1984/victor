# Tests for victor-devops safety rules
# Migrated from victor/tests/unit/framework/test_config.py

import pytest

from victor.framework.config import SafetyConfig, SafetyEnforcer, SafetyLevel


class TestDevOpsDeploymentSafety:
    """Tests for DevOps deployment safety rules."""

    def test_deployment_safety_rules(self):
        """DevOps deployment safety rules should require approval for production."""
        from victor_devops.safety import create_deployment_safety_rules

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_deployment_safety_rules(enforcer, require_approval_for_production=True)

        # Test production deployment blocking
        allowed, reason = enforcer.check_operation("kubectl apply -f deployment.yaml -n production")
        assert allowed is False
        assert "approval" in reason.lower() or "production" in reason.lower()


class TestDevOpsContainerSafety:
    """Tests for DevOps container safety rules."""

    def test_container_safety_rules(self):
        """DevOps container safety rules should block privileged containers."""
        from victor_devops.safety import create_container_safety_rules

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_container_safety_rules(enforcer, block_privileged_containers=True)

        # Test privileged container blocking
        allowed, reason = enforcer.check_operation("docker run --privileged alpine")
        assert allowed is False
        assert "privileged" in reason.lower()


class TestDevOpsInfrastructureSafety:
    """Tests for DevOps infrastructure safety rules."""

    def test_infrastructure_safety_rules(self):
        """DevOps infrastructure safety rules should block destructive commands."""
        from victor_devops.safety import create_infrastructure_safety_rules

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_infrastructure_safety_rules(enforcer, block_destructive_commands=True)

        # Test terraform destroy blocking
        allowed, reason = enforcer.check_operation("terraform destroy -auto-approve")
        assert allowed is False
        assert "destructive" in reason.lower() or "destroy" in reason.lower()

        # Test kubectl delete blocking
        allowed, reason = enforcer.check_operation("kubectl delete deployment -n production app")
        assert allowed is False
        assert "destructive" in reason.lower() or "delete" in reason.lower()


class TestDevOpsAllSafetyRules:
    """Tests for all DevOps safety rules combined."""

    def test_create_all_devops_safety_rules(self):
        """create_all_devops_safety_rules should register all devops rules."""
        from victor_devops.safety import create_all_devops_safety_rules

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_all_devops_safety_rules(enforcer)

        # Should have rules from deployment, container, and infrastructure categories
        assert len(enforcer.rules) > 0

        # Verify privileged container is blocked
        allowed, _ = enforcer.check_operation("docker run --privileged alpine")
        assert allowed is False
