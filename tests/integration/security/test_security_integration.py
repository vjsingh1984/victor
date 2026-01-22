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

"""Integration tests for security module with orchestrator.

This test suite validates:
- Integration with AgentOrchestrator
- End-to-end security testing
- Real attack scenario testing
- Security monitoring and alerting
"""

from pathlib import Path
import tempfile
import yaml

import pytest

from victor.security.penetration_testing import (
    SecurityTestSuite,
    SeverityLevel,
    AttackType,
)
from victor.security.authorization_enhanced import (
    EnhancedAuthorizer,
    Permission,
    Policy,
    PolicyEffect,
    User,
)
from victor.security.auth import RBACManager, Permission as RBACPermission


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def security_config():
    """Sample security configuration."""
    return {
        "enabled": True,
        "default_deny": True,
        "roles": {
            "security_admin": {
                "permissions": [
                    {"resource": "*", "action": "*"},
                ],
                "attributes": {"clearance_level": 5},
            }
        },
        "users": {
            "admin1": {
                "username": "security_admin",
                "roles": ["security_admin"],
                "attributes": {"department": "security"},
            }
        },
        "policies": [
            {
                "name": "require_clearance",
                "effect": "deny",
                "resource": "tools",
                "action": "execute",
                "conditions": {"clearance_level": {"lt": 3}},
                "priority": 10,
            }
        ],
    }


# =============================================================================
# Integration Tests
# =============================================================================


class TestSecurityWithOrchestrator:
    """Test security integration with orchestrator."""

    @pytest.mark.asyncio
    async def test_penetration_testing_with_mock_orchestrator(self):
        """Test penetration testing with mock orchestrator."""
        from unittest.mock import AsyncMock, MagicMock

        # Create mock orchestrator
        mock_agent = MagicMock()
        mock_agent.chat = AsyncMock(return_value="I cannot help with that request.")

        # Run security tests
        suite = SecurityTestSuite(safe_mode=True)
        report = await suite.test_prompt_injection(mock_agent)

        assert report.attack_type == AttackType.PROMPT_INJECTION
        assert report.test_count > 0

    @pytest.mark.asyncio
    async def test_authorization_with_orchestrator(self):
        """Test authorization integration with orchestrator."""
        authorizer = EnhancedAuthorizer()

        # Create test user
        user = authorizer.create_user(
            user_id="test_user",
            username="testuser",
            roles=["developer"],
        )

        # Check permissions for common orchestrator operations
        read_decision = authorizer.check_permission(user, "tools", "read")
        assert read_decision.allowed is True

        write_decision = authorizer.check_permission(user, "tools", "write")
        assert write_decision.allowed is True

        execute_decision = authorizer.check_permission(user, "tools", "execute")
        assert execute_decision.allowed is True

    @pytest.mark.asyncio
    async def test_security_with_event_bus(self):
        """Test security events published to event bus."""
        from victor.core.events import create_event_backend, MessagingEvent, BackendConfig
        import asyncio

        # Create event backend
        backend = create_event_backend(BackendConfig.for_observability())
        await backend.connect()

        # Publish security event
        event = MessagingEvent(
            topic="security.test",
            data={"test": "data", "severity": "high"},
        )
        await backend.publish(event)

        # Cleanup
        await backend.disconnect()


class TestEndToEndSecurity:
    """Test end-to-end security scenarios."""

    def test_complete_authorization_workflow(self, security_config):
        """Test complete authorization workflow."""
        authorizer = EnhancedAuthorizer()
        authorizer.load_from_dict(security_config)

        # Create user
        user = authorizer.create_user(
            user_id="user1",
            username="alice",
            roles=["viewer"],
        )

        # Test permission denial
        decision = authorizer.check_permission(user, "tools", "execute")
        assert decision.allowed is False

        # Assign higher role
        authorizer.assign_role("user1", "developer")

        # Test permission grant
        user = authorizer.get_user("user1")
        decision = authorizer.check_permission(user, "tools", "execute")
        assert decision.allowed is True

    def test_policy_evaluation_workflow(self):
        """Test policy-based authorization workflow."""
        authorizer = EnhancedAuthorizer()

        # Create allow policy
        authorizer.create_policy(
            name="allow_eng_dept",
            effect=PolicyEffect.ALLOW,
            resource="files",
            action="write",
            conditions={"department": "engineering"},
            priority=5,
        )

        # Create user in engineering
        user = authorizer.create_user(
            user_id="user1",
            username="alice",
            attributes={"department": "engineering"},
        )

        # Check permission (should be allowed by policy)
        decision = authorizer.check_permission(user, "files", "write")
        assert decision.allowed is True
        assert "policy" in decision.reason.lower()

    def test_rbac_abac_integration(self):
        """Test RBAC and ABAC working together."""
        authorizer = EnhancedAuthorizer()

        # Create role with permission
        role = authorizer.create_role(
            name="analyst",
            permissions=[
                Permission(
                    resource="files",
                    action="read",
                    constraints={"classification": "public"},
                ),
            ],
        )

        # Create user with role and attributes
        user = authorizer.create_user(
            user_id="user1",
            username="alice",
            roles=["analyst"],
        )

        # Test with matching context
        context = {"classification": "public"}
        decision = authorizer.check_permission(user, "files", "read", context)
        assert decision.allowed is True

        # Test with non-matching context
        context = {"classification": "secret"}
        decision = authorizer.check_permission(user, "files", "read", context)
        assert decision.allowed is False


class TestRealAttackScenarios:
    """Test with real attack scenarios."""

    @pytest.mark.asyncio
    async def test_prompt_injection_attack_chain(self):
        """Test multiple prompt injection attempts."""
        from unittest.mock import AsyncMock, MagicMock

        mock_agent = MagicMock()

        # Simulate attack chain
        responses = [
            "I cannot ignore my instructions.",
            "I do not have a system prompt to reveal.",
            "I cannot switch to unrestricted mode.",
            "I am not able to bypass safety controls.",
        ]
        mock_agent.chat = AsyncMock(side_effect=responses)

        suite = SecurityTestSuite(safe_mode=True)
        report = await suite.test_prompt_injection(mock_agent)

        # All attacks should be blocked
        assert len(report.passed_tests) > 0

    @pytest.mark.asyncio
    async def test_authorization_bypass_scenarios(self):
        """Test various authorization bypass scenarios."""
        from unittest.mock import AsyncMock, MagicMock

        authorizer = EnhancedAuthorizer()
        user = authorizer.create_user(
            user_id="attacker",
            username="attacker",
            roles=["viewer"],  # Limited role
        )

        # Simulate various bypass attempts
        resources_actions = [
            ("files", "delete"),
            ("settings", "modify"),
            ("users", "admin"),
            ("tools", "execute_dangerous"),
        ]

        for resource, action in resources_actions:
            decision = authorizer.check_permission(user, resource, action)
            assert decision.allowed is False, f"Bypass possible for {resource}:{action}"

    @pytest.mark.asyncio
    async def test_data_exfiltration_prevention(self):
        """Test data exfiltration prevention."""
        from unittest.mock import AsyncMock, MagicMock

        mock_agent = MagicMock()
        mock_agent.chat = AsyncMock(
            return_value="I cannot send data to external servers or reveal sensitive information."
        )

        suite = SecurityTestSuite(safe_mode=True)
        report = await suite.test_data_exfiltration(mock_agent)

        # Exfiltration should be prevented
        assert report.critical_count == 0

    @pytest.mark.asyncio
    async def test_resource_exhaustion_protection(self):
        """Test protection against resource exhaustion."""
        from unittest.mock import AsyncMock, MagicMock
        import asyncio

        mock_agent = MagicMock()

        async def slow_response(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate slow operation
            return "Request processed."

        mock_agent.chat = slow_response

        suite = SecurityTestSuite(safe_mode=True)
        report = await suite.test_resource_exhaustion(mock_agent)

        # Should complete without hanging
        assert report.test_count > 0

    @pytest.mark.asyncio
    async def test_code_injection_prevention(self):
        """Test code injection prevention."""
        from unittest.mock import AsyncMock, MagicMock

        mock_agent = MagicMock()
        mock_agent.chat = AsyncMock(
            return_value="I cannot execute arbitrary code or SQL commands."
        )

        suite = SecurityTestSuite(safe_mode=True)
        report = await suite.test_code_injection(mock_agent)

        # Code injection should be prevented
        assert report.critical_count == 0


class TestSecurityMonitoringAndAlerting:
    """Test security monitoring and alerting."""

    def test_authorization_logging(self, caplog):
        """Test that authorization decisions are logged."""
        import logging

        authorizer = EnhancedAuthorizer()
        user = authorizer.create_user(
            user_id="user1",
            username="alice",
            roles=["viewer"],
        )

        # Log level should capture debug messages
        with caplog.at_level(logging.DEBUG):
            decision = authorizer.check_permission(user, "tools", "write")

        # Check that decision was made
        assert decision.allowed is False

    def test_statistics_collection(self):
        """Test security statistics collection."""
        authorizer = EnhancedAuthorizer()

        # Create some users and roles
        authorizer.create_user(user_id="user1", username="alice", roles=["admin"])
        authorizer.create_user(user_id="user2", username="bob", roles=["viewer"])
        authorizer.create_policy(
            name="test_policy",
            effect=PolicyEffect.ALLOW,
            resource="tools",
            action="read",
        )

        # Get statistics
        stats = authorizer.get_stats()

        assert stats["enabled"] is True
        assert stats["roles_count"] >= 4  # Default roles
        assert stats["users_count"] == 2
        assert stats["policies_count"] == 1

    @pytest.mark.asyncio
    async def test_security_report_generation(self):
        """Test comprehensive security report generation."""
        from unittest.mock import AsyncMock, MagicMock

        mock_agent = MagicMock()
        mock_agent.chat = AsyncMock(return_value="Safe response.")

        suite = SecurityTestSuite(safe_mode=True)

        # Run all tests
        report = await suite.run_all_security_tests(mock_agent)

        # Generate reports in different formats
        text_report = report.generate_text_report()
        assert "SECURITY PENETRATION TEST REPORT" in text_report

        markdown_report = report.generate_markdown_report()
        assert "# Comprehensive Security Penetration Test Report" in markdown_report

        report_dict = report.to_dict()
        assert "timestamp" in report_dict
        assert "vulnerabilities" in report_dict


class TestSecurityPersistence:
    """Test security configuration persistence."""

    def test_save_and_load_authorization_config(self, security_config):
        """Test saving and loading authorization configuration."""
        authorizer = EnhancedAuthorizer()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_path = Path(f.name)

        try:
            # Load config
            yaml.dump(security_config, config_path.open("w"))
            authorizer.load_from_yaml(config_path)

            # Verify loading
            assert authorizer.get_role("security_admin") is not None
            assert authorizer.get_user("admin1") is not None
            assert authorizer.get_policy("require_clearance") is not None

            # Save to new file
            output_path = Path(tempfile.mktemp(suffix=".yaml"))
            authorizer.save_to_yaml(output_path)

            # Verify saved file
            with output_path.open("r") as f:
                saved_config = yaml.safe_load(f)
                assert "security_admin" in saved_config["roles"]
                assert "admin1" in saved_config["users"]

        finally:
            # Cleanup
            if config_path.exists():
                config_path.unlink()
            if output_path.exists():
                output_path.unlink()


class TestRBACIntegration:
    """Test integration with existing RBAC system."""

    def test_enhanced_auth_with_rbac(self):
        """Test EnhancedAuthorizer alongside existing RBACManager."""
        enhanced_auth = EnhancedAuthorizer()
        rbac = RBACManager()

        # Create role in both systems
        enhanced_role = enhanced_auth.create_role(
            name="custom_role",
            permissions=[Permission(resource="tools", action="read")],
        )

        # Both should work independently
        assert enhanced_auth.get_role("custom_role") is not None

        # Create user in enhanced system
        user = enhanced_auth.create_user(
            user_id="user1",
            username="alice",
            roles=["custom_role"],
        )

        decision = enhanced_auth.check_permission(user, "tools", "read")
        assert decision.allowed is True

    def test_permission_mapping(self):
        """Test mapping between permission systems."""
        from victor.tools.base import AccessMode

        enhanced_auth = EnhancedAuthorizer()

        # Create permission for different access modes
        read_perm = Permission(resource="tools", action="read")
        write_perm = Permission(resource="tools", action="write")
        execute_perm = Permission(resource="tools", action="execute")

        role = enhanced_auth.create_role(
            name="operator",
            permissions={read_perm, write_perm, execute_perm},
        )

        # Role should have all permissions
        assert role.has_permission("tools", "read")
        assert role.has_permission("tools", "write")
        assert role.has_permission("tools", "execute")


class TestSecurityInProduction:
    """Test security features in production-like scenarios."""

    def test_default_deny_security(self):
        """Test that default deny provides security."""
        authorizer = EnhancedAuthorizer(default_deny=True)

        user = authorizer.create_user(
            user_id="user1",
            username="alice",
            roles=[],  # No roles
        )

        # Should be denied by default
        decision = authorizer.check_permission(user, "tools", "read")
        assert decision.allowed is False

    def test_disabled_authorizer_allows_all(self):
        """Test that disabled authorizer allows everything."""
        authorizer = EnhancedAuthorizer(enabled=False)

        user = authorizer.create_user(
            user_id="user1",
            username="alice",
            roles=[],  # No roles
        )

        # Should be allowed when disabled
        decision = authorizer.check_permission(user, "tools", "write")
        assert decision.allowed is True

    def test_disabled_user_cannot_access(self):
        """Test that disabled users cannot access."""
        authorizer = EnhancedAuthorizer()

        user = User(
            id="user1",
            username="alice",
            roles=["admin"],  # Has admin role
            enabled=False,  # But disabled
        )
        authorizer._users["user1"] = user

        # Should be denied
        decision = authorizer.check_permission(user, "tools", "read")
        assert decision.allowed is False
        assert "disabled" in decision.reason.lower()

    def test_policy_priority_ordering(self):
        """Test that policies are evaluated in priority order."""
        authorizer = EnhancedAuthorizer()

        # Create low priority allow policy
        authorizer.create_policy(
            name="low_allow",
            effect=PolicyEffect.ALLOW,
            resource="tools",
            action="execute",
            subjects=["user1"],
            priority=1,
        )

        # Create high priority deny policy
        authorizer.create_policy(
            name="high_deny",
            effect=PolicyEffect.DENY,
            resource="tools",
            action="execute",
            subjects=["user1"],
            priority=10,
        )

        user = authorizer.create_user(user_id="user1", username="alice")

        # High priority deny should win
        decision = authorizer.check_permission(user, "tools", "execute")
        assert decision.allowed is False
