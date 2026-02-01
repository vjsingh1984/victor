"""
Unit tests for Enhanced Authorization Module.

Tests cover:
- RBAC (Role-Based Access Control)
- ABAC (Attribute-Based Access Control)
- Policy-based authorization
- Permission management
- Thread safety
"""

import pytest
from victor.core.security.authorization import (
    EnhancedAuthorizer,
    User,
    Role,
    Permission,
    PolicyEffect,
)


class TestPermission:
    """Test Permission dataclass."""

    def test_permission_creation(self):
        """Test creating a permission."""
        perm = Permission(
            resource="tools", action="execute", constraints={"tool_category": "data_analysis"}
        )
        assert perm.resource == "tools"
        assert perm.action == "execute"
        assert perm.constraints == {"tool_category": "data_analysis"}

    def test_permission_matches(self):
        """Test permission matching."""
        perm = Permission(resource="tools", action="execute")
        assert perm.matches("tools", "execute") is True
        assert perm.matches("tools", "read") is False
        assert perm.matches("files", "execute") is False

    def test_permission_with_constraints(self):
        """Test permission with constraints."""
        perm = Permission(resource="tools", action="execute", constraints={"tool_name": "bash"})
        context = {"tool_name": "bash"}
        assert perm.matches("tools", "execute", context) is True

        context = {"tool_name": "python"}
        assert perm.matches("tools", "execute", context) is False

    def test_permission_hashable(self):
        """Test that permissions are hashable (can be stored in sets)."""
        perm1 = Permission(resource="tools", action="execute")
        perm2 = Permission(resource="tools", action="execute")
        perm3 = Permission(resource="files", action="read")

        perm_set = {perm1, perm2, perm3}
        assert len(perm_set) == 2  # perm1 and perm2 are duplicates


class TestRole:
    """Test Role dataclass."""

    def test_role_creation(self):
        """Test creating a role."""
        role = Role(
            name="developer",
            permissions={
                Permission(resource="tools", action="read"),
                Permission(resource="tools", action="execute"),
            },
            attributes={"clearance_level": 3},
            description="Developer role",
        )
        assert role.name == "developer"
        assert len(role.permissions) == 2
        assert role.attributes == {"clearance_level": 3}

    def test_add_permission(self):
        """Test adding permission to role."""
        role = Role(name="tester")
        perm = Permission(resource="tools", action="execute")
        role.add_permission(perm)
        assert perm in role.permissions

    def test_has_permission(self):
        """Test checking if role has permission."""
        role = Role(name="developer", permissions={Permission(resource="tools", action="execute")})
        assert role.has_permission("tools", "execute") is True
        assert role.has_permission("tools", "read") is False


class TestUser:
    """Test User dataclass."""

    def test_user_creation(self):
        """Test creating a user."""
        user = User(
            id="user1",
            username="alice",
            roles={"developer", "code_reviewer"},
            attributes={"department": "engineering", "level": "senior"},
        )
        assert user.id == "user1"
        assert user.username == "alice"
        assert len(user.roles) == 2
        assert user.get_attribute("department") == "engineering"

    def test_add_role(self):
        """Test adding role to user."""
        user = User(id="user1", username="alice")
        user.add_role("developer")
        assert "developer" in user.roles

    def test_check_role_membership(self):
        """Test checking if user has role."""
        user = User(id="user1", username="alice", roles={"developer"})
        assert "developer" in user.roles
        assert "admin" not in user.roles


class TestEnhancedAuthorizer:
    """Test EnhancedAuthorizer class."""

    def test_initialization(self):
        """Test authorizer initialization."""
        authorizer = EnhancedAuthorizer()
        assert authorizer.enabled is True
        stats = authorizer.get_stats()
        assert stats["enabled"] is True
        assert stats["default_deny"] is True
        assert len(authorizer.list_roles()) == 4  # Default roles

    def test_enable_disable(self):
        """Test enabling and disabling authorization."""
        authorizer = EnhancedAuthorizer()
        authorizer.disable()
        assert authorizer.enabled is False
        authorizer.enable()
        assert authorizer.enabled is True

    def test_create_role(self):
        """Test creating a role."""
        authorizer = EnhancedAuthorizer()
        role = authorizer.create_role(
            name="data_scientist",
            permissions=[
                Permission(resource="workflows", action="execute"),
            ],
            description="Data scientist role",
        )
        assert role.name == "data_scientist"
        retrieved = authorizer.get_role("data_scientist")
        assert retrieved is not None
        assert retrieved.name == "data_scientist"

    def test_create_user(self):
        """Test creating a user."""
        authorizer = EnhancedAuthorizer()
        user = authorizer.create_user(
            user_id="user1",
            username="alice",
            roles=["developer"],
            attributes={"department": "engineering"},
        )
        assert user.username == "alice"
        assert "developer" in user.roles

    def test_assign_role(self):
        """Test assigning role to user."""
        authorizer = EnhancedAuthorizer()
        authorizer.create_user("user1", "alice")
        result = authorizer.assign_role("user1", "developer")
        assert result is True

        user = authorizer.get_user("user1")
        assert "developer" in user.roles

    def test_grant_permission(self):
        """Test granting permission to role."""
        authorizer = EnhancedAuthorizer()
        perm = Permission(resource="api", action="read")
        result = authorizer.grant_permission("developer", perm)
        assert result is True

        role = authorizer.get_role("developer")
        assert perm in role.permissions

    def test_revoke_permission(self):
        """Test revoking permission from role."""
        authorizer = EnhancedAuthorizer()
        perm = Permission(resource="api", action="read")
        authorizer.grant_permission("developer", perm)
        result = authorizer.revoke_permission("developer", perm)
        assert result is True

        role = authorizer.get_role("developer")
        assert perm not in role.permissions

    def test_check_permission_rbac(self):
        """Test RBAC permission check."""
        authorizer = EnhancedAuthorizer()
        user = authorizer.create_user(user_id="user1", username="alice", roles=["developer"])

        # Developer should have tool read/execute permissions
        decision = authorizer.check_permission(user, "tools", "read")
        assert decision.allowed is True
        assert "role" in decision.reason.lower()

    def test_check_permission_disabled_user(self):
        """Test that disabled users are denied access."""
        authorizer = EnhancedAuthorizer()
        user = User(id="user1", username="alice", roles=["admin"], enabled=False)

        decision = authorizer.check_permission(user, "tools", "execute")
        assert decision.allowed is False
        assert "disabled" in decision.reason.lower()

    def test_check_permission_disabled_authorizer(self):
        """Test that disabled authorizer allows all."""
        authorizer = EnhancedAuthorizer()
        authorizer.disable()

        user = User(id="user1", username="alice", roles=[])
        decision = authorizer.check_permission(user, "tools", "execute")
        assert decision.allowed is True

    def test_policy_allow(self):
        """Test policy-based allow."""
        authorizer = EnhancedAuthorizer()
        user = authorizer.create_user(
            user_id="user1", username="alice", attributes={"department": "engineering"}
        )

        authorizer.create_policy(
            name="engineering_access",
            effect=PolicyEffect.ALLOW,
            resource="files",
            action="write",
            conditions={"department": "engineering"},
            priority=50,
        )

        decision = authorizer.check_permission(user, "files", "write")
        assert decision.allowed is True
        assert "policy" in decision.reason.lower()

    def test_policy_deny(self):
        """Test policy-based deny (deny takes precedence)."""
        authorizer = EnhancedAuthorizer()
        user = authorizer.create_user(
            user_id="user1", username="alice", roles=["admin"]  # Admin normally has all permissions
        )

        # Create high-priority deny policy
        authorizer.create_policy(
            name="block_dangerous",
            effect=PolicyEffect.DENY,
            resource="tools",
            action="execute",
            conditions={"tool_name": "dangerous_tool"},
            priority=100,
        )

        context = {"tool_name": "dangerous_tool"}
        decision = authorizer.check_permission(user, "tools", "execute", context)
        assert decision.allowed is False
        assert "denied" in decision.reason.lower()

    def test_default_deny(self):
        """Test default deny behavior."""
        authorizer = EnhancedAuthorizer(default_deny=True)
        user = authorizer.create_user(user_id="user1", username="alice", roles=[])  # No roles

        decision = authorizer.check_permission(user, "tools", "execute")
        assert decision.allowed is False
        assert "default deny" in decision.reason.lower()

    def test_default_allow(self):
        """Test default allow behavior."""
        authorizer = EnhancedAuthorizer(default_deny=False)
        user = authorizer.create_user(user_id="user1", username="alice", roles=[])

        decision = authorizer.check_permission(user, "tools", "execute")
        assert decision.allowed is True
        assert "default allow" in decision.reason.lower()

    def test_permission_with_constraints(self):
        """Test permission evaluation with constraints."""
        authorizer = EnhancedAuthorizer()
        authorizer.create_role(
            name="restricted_user",
            permissions=[
                Permission(
                    resource="tools", action="execute", constraints={"tool_category": "safe"}
                )
            ],
        )

        user = authorizer.create_user(user_id="user1", username="alice", roles=["restricted_user"])

        # Should allow with matching context
        context = {"tool_category": "safe"}
        decision = authorizer.check_permission(user, "tools", "execute", context)
        assert decision.allowed is True

        # Should deny with non-matching context
        context = {"tool_category": "dangerous"}
        decision = authorizer.check_permission(user, "tools", "execute", context)
        assert decision.allowed is False

    def test_delete_user(self):
        """Test deleting a user."""
        authorizer = EnhancedAuthorizer()
        authorizer.create_user("user1", "alice")
        result = authorizer.delete_user("user1")
        assert result is True
        assert authorizer.get_user("user1") is None

    def test_delete_role(self):
        """Test deleting a role."""
        authorizer = EnhancedAuthorizer()
        authorizer.create_role("temp_role", [])
        result = authorizer.delete_role("temp_role")
        assert result is True
        assert authorizer.get_role("temp_role") is None

    def test_delete_policy(self):
        """Test deleting a policy."""
        authorizer = EnhancedAuthorizer()
        authorizer.create_policy(
            name="temp_policy", effect=PolicyEffect.ALLOW, resource="test", action="test"
        )
        result = authorizer.delete_policy("temp_policy")
        assert result is True
        assert authorizer.get_policy("temp_policy") is None

    def test_get_stats(self):
        """Test getting authorizer statistics."""
        authorizer = EnhancedAuthorizer()
        authorizer.create_user("user1", "alice")
        authorizer.create_role("custom_role", [])

        stats = authorizer.get_stats()
        assert stats["enabled"] is True
        assert stats["default_deny"] is True
        assert stats["users_count"] == 1
        assert stats["roles_count"] == 5  # 4 default + 1 custom
        assert "admin" in stats["role_names"]


class TestPolicyEvaluation:
    """Test policy evaluation logic."""

    def test_policy_priority_ordering(self):
        """Test that higher priority policies are evaluated first."""
        authorizer = EnhancedAuthorizer()
        user = authorizer.create_user(user_id="user1", username="alice", attributes={"level": 3})

        # Low priority allow
        authorizer.create_policy(
            name="low_priority_allow",
            effect=PolicyEffect.ALLOW,
            resource="test",
            action="test",
            priority=10,
        )

        # High priority deny (should win)
        authorizer.create_policy(
            name="high_priority_deny",
            effect=PolicyEffect.DENY,
            resource="test",
            action="test",
            priority=100,
        )

        decision = authorizer.check_permission(user, "test", "test")
        assert decision.allowed is False
        assert "high_priority_deny" in decision.reason

    def test_policy_conditions_operators(self):
        """Test policy conditions with operators."""
        authorizer = EnhancedAuthorizer()
        user = authorizer.create_user(
            user_id="user1", username="alice", attributes={"clearance_level": 5}
        )

        # Policy with greater-than-or-equal condition
        authorizer.create_policy(
            name="clearance_check",
            effect=PolicyEffect.ALLOW,
            resource="secret",
            action="read",
            conditions={"clearance_level": {"gte": 3}},
            priority=50,
        )

        decision = authorizer.check_permission(user, "secret", "read")
        assert decision.allowed is True

        # Test with insufficient clearance
        user.attributes["clearance_level"] = 2
        decision = authorizer.check_permission(user, "secret", "read")
        assert decision.allowed is False


class TestThreadSafety:
    """Test thread safety of authorizer operations."""

    def test_concurrent_role_creation(self):
        """Test creating roles concurrently."""
        import threading

        authorizer = EnhancedAuthorizer()
        threads = []

        def create_role(i):
            authorizer.create_role(f"role_{i}", [])

        for i in range(10):
            thread = threading.Thread(target=create_role, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        stats = authorizer.get_stats()
        assert stats["roles_count"] == 14  # 4 default + 10 created


class TestYAMLConfiguration:
    """Test YAML configuration loading."""

    def test_load_from_dict(self):
        """Test loading configuration from dictionary."""
        authorizer = EnhancedAuthorizer()

        config = {
            "enabled": False,
            "default_deny": False,
            "roles": {
                "custom_role": {
                    "permissions": [{"resource": "test", "action": "test"}],
                    "attributes": {"test_attr": "value"},
                    "description": "Test role",
                }
            },
            "users": {"user1": {"username": "alice", "roles": ["custom_role"], "enabled": True}},
            "policies": [
                {
                    "name": "test_policy",
                    "effect": "allow",
                    "resource": "test",
                    "action": "test",
                    "priority": 50,
                }
            ],
        }

        authorizer.load_from_dict(config)

        assert authorizer.enabled is False
        assert authorizer._default_deny is False

        role = authorizer.get_role("custom_role")
        assert role is not None
        assert role.attributes["test_attr"] == "value"

        user = authorizer.get_user("user1")
        assert user is not None
        assert user.username == "alice"

        policy = authorizer.get_policy("test_policy")
        assert policy is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
