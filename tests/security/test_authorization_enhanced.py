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

"""Tests for enhanced authorization module.

This test suite validates RBAC and ABAC functionality including:
- Role-based access control (RBAC)
- Attribute-based access control (ABAC)
- Permission checking and management
- Policy evaluation
- User and role management
"""

from pathlib import Path
import tempfile
import yaml

import pytest

from victor.core.security.authorization import (
    EnhancedAuthorizer,
    Permission,
    Role,
    User,
    Policy,
    PolicyEffect,
    AuthorizationDecision,
    get_enhanced_authorizer,
    set_enhanced_authorizer,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def authorizer():
    """Create a fresh authorizer for each test."""
    return EnhancedAuthorizer()


@pytest.fixture
def sample_role():
    """Create a sample role."""
    return Role(
        name="developer",
        permissions={
            Permission(resource="tools", action="read"),
            Permission(resource="tools", action="execute"),
        },
        attributes={"clearance_level": 3},
    )


@pytest.fixture
def sample_user():
    """Create a sample user."""
    return User(
        id="user1",
        username="alice",
        roles={"developer"},
        attributes={"department": "engineering"},
    )


# =============================================================================
# Permission Tests
# =============================================================================


class TestPermission:
    """Test Permission functionality."""

    def test_permission_creation(self):
        """Test creating a permission."""
        perm = Permission(resource="tools", action="read")
        assert perm.resource == "tools"
        assert perm.action == "read"
        assert perm.constraints == {}

    def test_permission_with_constraints(self):
        """Test creating a permission with constraints."""
        perm = Permission(
            resource="tools",
            action="execute",
            constraints={"max_duration": 60},
        )
        assert perm.constraints == {"max_duration": 60}

    def test_permission_matches_exact(self):
        """Test permission matching with exact match."""
        perm = Permission(resource="tools", action="read")
        assert perm.matches("tools", "read")

    def test_permission_mismatch_resource(self):
        """Test permission matching with wrong resource."""
        perm = Permission(resource="tools", action="read")
        assert not perm.matches("files", "read")

    def test_permission_mismatch_action(self):
        """Test permission matching with wrong action."""
        perm = Permission(resource="tools", action="read")
        assert not perm.matches("tools", "write")

    def test_permission_with_constraint_pass(self):
        """Test permission with passing constraint."""
        perm = Permission(
            resource="tools",
            action="execute",
            constraints={"tool_type": "bash"},
        )
        context = {"tool_type": "bash"}
        assert perm.matches("tools", "execute", context)

    def test_permission_with_constraint_fail(self):
        """Test permission with failing constraint."""
        perm = Permission(
            resource="tools",
            action="execute",
            constraints={"tool_type": "bash"},
        )
        context = {"tool_type": "python"}
        assert not perm.matches("tools", "execute", context)

    def test_permission_to_dict(self):
        """Test converting permission to dictionary."""
        perm = Permission(resource="tools", action="read")
        perm_dict = perm.to_dict()
        assert perm_dict["resource"] == "tools"
        assert perm_dict["action"] == "read"


# =============================================================================
# Role Tests
# =============================================================================


class TestRole:
    """Test Role functionality."""

    def test_role_creation(self):
        """Test creating a role."""
        role = Role(name="admin")
        assert role.name == "admin"
        assert len(role.permissions) == 0

    def test_role_with_permissions(self):
        """Test creating a role with permissions."""
        perms = {Permission(resource="*", action="*")}
        role = Role(name="admin", permissions=perms)
        assert len(role.permissions) == 1

    def test_role_add_permission(self):
        """Test adding permission to role."""
        role = Role(name="developer")
        perm = Permission(resource="tools", action="read")
        role.add_permission(perm)
        assert len(role.permissions) == 1
        assert role.has_permission("tools", "read")

    def test_role_remove_permission(self):
        """Test removing permission from role."""
        role = Role(name="developer")
        perm = Permission(resource="tools", action="read")
        role.add_permission(perm)
        role.remove_permission(perm)
        assert not role.has_permission("tools", "read")

    def test_role_has_permission(self):
        """Test checking if role has permission."""
        role = Role(
            name="developer",
            permissions={
                Permission(resource="tools", action="read"),
                Permission(resource="tools", action="execute"),
            },
        )
        assert role.has_permission("tools", "read")
        assert role.has_permission("tools", "execute")
        assert not role.has_permission("tools", "write")

    def test_role_attributes(self):
        """Test role attributes."""
        role = Role(
            name="admin",
            attributes={"clearance_level": 5},
        )
        assert role.attributes["clearance_level"] == 5

    def test_role_to_dict(self):
        """Test converting role to dictionary."""
        role = Role(name="admin")
        role_dict = role.to_dict()
        assert role_dict["name"] == "admin"
        assert "permissions" in role_dict


# =============================================================================
# User Tests
# =============================================================================


class TestUser:
    """Test User functionality."""

    def test_user_creation(self):
        """Test creating a user."""
        user = User(id="user1", username="alice")
        assert user.id == "user1"
        assert user.username == "alice"
        assert user.enabled is True

    def test_user_with_roles(self):
        """Test creating user with roles."""
        user = User(
            id="user1",
            username="alice",
            roles={"developer", "viewer"},
        )
        assert len(user.roles) == 2

    def test_user_add_role(self):
        """Test adding role to user."""
        user = User(id="user1", username="alice")
        user.add_role("developer")
        assert "developer" in user.roles

    def test_user_remove_role(self):
        """Test removing role from user."""
        user = User(id="user1", username="alice", roles={"developer"})
        user.remove_role("developer")
        assert "developer" not in user.roles

    def test_user_attributes(self):
        """Test user attributes."""
        user = User(
            id="user1",
            username="alice",
            attributes={"department": "engineering", "clearance": 3},
        )
        assert user.get_attribute("department") == "engineering"
        assert user.get_attribute("clearance") == 3
        assert user.get_attribute("missing", "default") == "default"

    def test_user_direct_permissions(self):
        """Test user direct permissions."""
        user = User(id="user1", username="alice")
        perm = Permission(resource="tools", action="read")
        user.add_permission(perm)
        assert len(user.permissions) == 1

    def test_user_to_dict(self):
        """Test converting user to dictionary."""
        user = User(id="user1", username="alice")
        user_dict = user.to_dict()
        assert user_dict["id"] == "user1"
        assert user_dict["username"] == "alice"


# =============================================================================
# Policy Tests
# =============================================================================


class TestPolicy:
    """Test Policy functionality."""

    def test_policy_creation(self):
        """Test creating a policy."""
        policy = Policy(
            name="test_policy",
            effect=PolicyEffect.ALLOW,
            resource="tools",
            action="read",
        )
        assert policy.name == "test_policy"
        assert policy.effect == PolicyEffect.ALLOW

    def test_policy_matches_exact(self):
        """Test policy matching exact request."""
        policy = Policy(
            name="test",
            effect=PolicyEffect.ALLOW,
            resource="tools",
            action="read",
            subjects=["user1"],
        )
        assert policy.matches("tools", "read", "user1")

    def test_policy_mismatch_resource(self):
        """Test policy with wrong resource."""
        policy = Policy(
            name="test",
            effect=PolicyEffect.ALLOW,
            resource="tools",
            action="read",
        )
        assert not policy.matches("files", "read", "user1")

    def test_policy_with_subjects(self):
        """Test policy subject matching."""
        policy = Policy(
            name="test",
            effect=PolicyEffect.ALLOW,
            resource="tools",
            action="read",
            subjects=["user1", "user2"],
        )
        assert policy.matches("tools", "read", "user1")
        assert not policy.matches("tools", "read", "user3")

    def test_policy_with_conditions(self):
        """Test policy with ABAC conditions."""
        policy = Policy(
            name="test",
            effect=PolicyEffect.ALLOW,
            resource="tools",
            action="read",
            conditions={"clearance_level": 3},
        )
        context = {"clearance_level": 3}
        assert policy.matches("tools", "read", "user1", context)

    def test_policy_conditions_fail(self):
        """Test policy with failing conditions."""
        policy = Policy(
            name="test",
            effect=PolicyEffect.ALLOW,
            resource="tools",
            action="read",
            conditions={"clearance_level": 3},
        )
        context = {"clearance_level": 1}
        assert not policy.matches("tools", "read", "user1", context)

    def test_policy_operator_conditions(self):
        """Test policy with operator-based conditions."""
        policy = Policy(
            name="test",
            effect=PolicyEffect.ALLOW,
            resource="tools",
            action="read",
            conditions={"clearance_level": {"gte": 3}},
        )
        context = {"clearance_level": 5}
        assert policy.matches("tools", "read", "user1", context)

    def test_policy_to_dict(self):
        """Test converting policy to dictionary."""
        policy = Policy(
            name="test",
            effect=PolicyEffect.ALLOW,
            resource="tools",
            action="read",
        )
        policy_dict = policy.to_dict()
        assert policy_dict["name"] == "test"
        assert policy_dict["effect"] == "allow"


# =============================================================================
# EnhancedAuthorizer Tests
# =============================================================================


class TestEnhancedAuthorizer:
    """Test EnhancedAuthorizer functionality."""

    def test_authorizer_initialization(self):
        """Test authorizer initialization."""
        auth = EnhancedAuthorizer()
        assert auth.enabled is True
        assert len(auth.list_roles()) == 4  # Default roles

    def test_authorizer_disabled(self):
        """Test disabled authorizer allows all."""
        auth = EnhancedAuthorizer(enabled=False)
        user = User(id="user1", username="alice")
        decision = auth.check_permission(user, "tools", "write")
        assert decision.allowed is True

    def test_authorizer_enable_disable(self):
        """Test enabling and disabling authorizer."""
        auth = EnhancedAuthorizer()
        auth.disable()
        assert auth.enabled is False
        auth.enable()
        assert auth.enabled is True

    # -------------------------------------------------------------------------
    # Role Management
    # -------------------------------------------------------------------------

    def test_create_role(self, authorizer):
        """Test creating a new role."""
        perms = [Permission(resource="files", action="write")]
        role = authorizer.create_role(
            name="file_manager",
            permissions=perms,
            description="Manages files",
        )
        assert role.name == "file_manager"
        assert len(role.permissions) == 1

    def test_get_role(self, authorizer):
        """Test getting a role."""
        role = authorizer.get_role("admin")
        assert role is not None
        assert role.name == "admin"

    def test_get_nonexistent_role(self, authorizer):
        """Test getting a non-existent role."""
        role = authorizer.get_role("nonexistent")
        assert role is None

    def test_list_roles(self, authorizer):
        """Test listing all roles."""
        roles = authorizer.list_roles()
        assert len(roles) >= 4  # Default roles

    def test_delete_role(self, authorizer):
        """Test deleting a role."""
        authorizer.create_role(name="temp_role", permissions=[])
        assert authorizer.get_role("temp_role") is not None
        authorizer.delete_role("temp_role")
        assert authorizer.get_role("temp_role") is None

    # -------------------------------------------------------------------------
    # User Management
    # -------------------------------------------------------------------------

    def test_create_user(self, authorizer):
        """Test creating a user."""
        user = authorizer.create_user(
            user_id="user1",
            username="alice",
            roles=["developer"],
        )
        assert user.id == "user1"
        assert user.username == "alice"
        assert "developer" in user.roles

    def test_get_user(self, authorizer):
        """Test getting a user."""
        authorizer.create_user(user_id="user1", username="alice")
        user = authorizer.get_user("user1")
        assert user is not None
        assert user.username == "alice"

    def test_get_user_by_username(self, authorizer):
        """Test getting user by username."""
        authorizer.create_user(user_id="user1", username="alice")
        user = authorizer.get_user_by_username("alice")
        assert user is not None
        assert user.id == "user1"

    def test_list_users(self, authorizer):
        """Test listing users."""
        authorizer.create_user(user_id="user1", username="alice")
        authorizer.create_user(user_id="user2", username="bob")
        users = authorizer.list_users()
        assert len(users) >= 2

    def test_delete_user(self, authorizer):
        """Test deleting a user."""
        authorizer.create_user(user_id="user1", username="alice")
        assert authorizer.delete_user("user1") is True
        assert authorizer.get_user("user1") is None

    def test_assign_role(self, authorizer):
        """Test assigning role to user."""
        authorizer.create_user(user_id="user1", username="alice")
        result = authorizer.assign_role("user1", "developer")
        assert result is True

        user = authorizer.get_user("user1")
        assert "developer" in user.roles

    def test_revoke_role(self, authorizer):
        """Test revoking role from user."""
        authorizer.create_user(
            user_id="user1",
            username="alice",
            roles=["developer"],
        )
        result = authorizer.revoke_role("user1", "developer")
        assert result is True

        user = authorizer.get_user("user1")
        assert "developer" not in user.roles

    # -------------------------------------------------------------------------
    # Permission Management
    # -------------------------------------------------------------------------

    def test_grant_permission(self, authorizer):
        """Test granting permission to role."""
        perm = Permission(resource="files", action="delete")
        result = authorizer.grant_permission("admin", perm)
        assert result is True

        role = authorizer.get_role("admin")
        assert any(p.resource == "files" and p.action == "delete" for p in role.permissions)

    def test_revoke_permission(self, authorizer):
        """Test revoking permission from role."""
        perm = Permission(resource="files", action="delete")
        authorizer.grant_permission("admin", perm)
        result = authorizer.revoke_permission("admin", perm)
        assert result is True

    # -------------------------------------------------------------------------
    # Policy Management
    # -------------------------------------------------------------------------

    def test_create_policy(self, authorizer):
        """Test creating a policy."""
        policy = authorizer.create_policy(
            name="test_policy",
            effect=PolicyEffect.ALLOW,
            resource="tools",
            action="read",
            priority=10,
        )
        assert policy.name == "test_policy"
        assert policy.priority == 10

    def test_get_policy(self, authorizer):
        """Test getting a policy."""
        authorizer.create_policy(
            name="test_policy",
            effect=PolicyEffect.ALLOW,
            resource="tools",
            action="read",
        )
        policy = authorizer.get_policy("test_policy")
        assert policy is not None
        assert policy.name == "test_policy"

    def test_list_policies(self, authorizer):
        """Test listing policies."""
        authorizer.create_policy(
            name="policy1",
            effect=PolicyEffect.ALLOW,
            resource="tools",
            action="read",
        )
        authorizer.create_policy(
            name="policy2",
            effect=PolicyEffect.DENY,
            resource="files",
            action="delete",
        )
        policies = authorizer.list_policies()
        assert len(policies) >= 2

    def test_delete_policy(self, authorizer):
        """Test deleting a policy."""
        authorizer.create_policy(
            name="test_policy",
            effect=PolicyEffect.ALLOW,
            resource="tools",
            action="read",
        )
        assert authorizer.delete_policy("test_policy") is True
        assert authorizer.get_policy("test_policy") is None

    # -------------------------------------------------------------------------
    # Authorization Checks
    # -------------------------------------------------------------------------

    def test_check_permission_role_based(self, authorizer):
        """Test RBAC permission check."""
        user = authorizer.create_user(
            user_id="user1",
            username="alice",
            roles=["developer"],
        )

        decision = authorizer.check_permission(user, "tools", "read")
        assert decision.allowed is True
        assert "developer" in decision.reason

    def test_check_permission_denied(self, authorizer):
        """Test permission denial."""
        user = authorizer.create_user(
            user_id="user1",
            username="alice",
            roles=["viewer"],
        )

        decision = authorizer.check_permission(user, "tools", "write")
        assert decision.allowed is False

    def test_check_permission_disabled_user(self, authorizer):
        """Test disabled user cannot access."""
        user = User(id="user1", username="alice", enabled=False)
        authorizer._users["user1"] = user

        decision = authorizer.check_permission(user, "tools", "read")
        assert decision.allowed is False
        assert "disabled" in decision.reason

    def test_check_permission_policy_allow(self, authorizer):
        """Test policy-based allow."""
        user = authorizer.create_user(user_id="user1", username="alice")

        authorizer.create_policy(
            name="allow_alice",
            effect=PolicyEffect.ALLOW,
            resource="tools",
            action="read",
            subjects=["user1"],
        )

        decision = authorizer.check_permission(user, "tools", "read")
        assert decision.allowed is True
        assert "policy" in decision.reason.lower()

    def test_check_permission_policy_deny(self, authorizer):
        """Test policy-based deny (takes precedence)."""
        user = authorizer.create_user(
            user_id="user1",
            username="alice",
            roles=["admin"],  # Admin has all permissions
        )

        # Create DENY policy for specific action
        authorizer.create_policy(
            name="deny_delete",
            effect=PolicyEffect.DENY,
            resource="files",
            action="delete",
            subjects=["user1"],
        )

        decision = authorizer.check_permission(user, "files", "delete")
        assert decision.allowed is False
        assert "deny" in decision.reason.lower()

    def test_check_permission_by_id(self, authorizer):
        """Test checking permission by user ID."""
        authorizer.create_user(
            user_id="user1",
            username="alice",
            roles=["developer"],
        )

        decision = authorizer.check_permission_by_id("user1", "tools", "read")
        assert decision.allowed is True

    def test_check_permission_by_id_nonexistent_user(self, authorizer):
        """Test checking permission for non-existent user."""
        decision = authorizer.check_permission_by_id("nonexistent", "tools", "read")
        assert decision.allowed is False
        assert "not found" in decision.reason.lower()

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def test_load_from_dict(self, authorizer):
        """Test loading configuration from dictionary."""
        config = {
            "enabled": True,
            "default_deny": True,
            "roles": {
                "custom_role": {
                    "permissions": [
                        {"resource": "files", "action": "read"},
                    ],
                    "attributes": {"level": 1},
                }
            },
            "users": {
                "user1": {
                    "username": "alice",
                    "roles": ["custom_role"],
                }
            },
            "policies": [
                {
                    "name": "test_policy",
                    "effect": "allow",
                    "resource": "files",
                    "action": "write",
                    "priority": 5,
                }
            ],
        }

        authorizer.load_from_dict(config)

        # Check role
        role = authorizer.get_role("custom_role")
        assert role is not None

        # Check user
        user = authorizer.get_user("user1")
        assert user is not None

        # Check policy
        policy = authorizer.get_policy("test_policy")
        assert policy is not None

    def test_load_save_yaml(self, authorizer):
        """Test loading and saving YAML configuration."""
        config = {
            "enabled": True,
            "roles": {
                "test_role": {
                    "permissions": [{"resource": "tools", "action": "read"}],
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = Path(f.name)

        try:
            # Load
            authorizer.load_from_yaml(config_path)
            role = authorizer.get_role("test_role")
            assert role is not None

            # Save
            output_path = Path(tempfile.mktemp(suffix=".yaml"))
            authorizer.save_to_yaml(output_path)

            # Verify saved file
            with open(output_path, "r") as f:
                saved_config = yaml.safe_load(f)
                assert "test_role" in saved_config["roles"]

        finally:
            # Cleanup
            if config_path.exists():
                config_path.unlink()
            if output_path.exists():
                output_path.unlink()

    def test_get_stats(self, authorizer):
        """Test getting authorizer statistics."""
        stats = authorizer.get_stats()
        assert "enabled" in stats
        assert "roles_count" in stats
        assert "users_count" in stats
        assert "policies_count" in stats


# =============================================================================
# Global Instance Tests
# =============================================================================


class TestGlobalAuthorizer:
    """Test global authorizer instance."""

    def test_get_enhanced_authorizer_singleton(self):
        """Test that get_enhanced_authorizer returns singleton."""
        auth1 = get_enhanced_authorizer()
        auth2 = get_enhanced_authorizer()
        assert auth1 is auth2

    def test_set_enhanced_authorizer(self):
        """Test setting global authorizer."""
        custom_auth = EnhancedAuthorizer()
        set_enhanced_authorizer(custom_auth)

        global_auth = get_enhanced_authorizer()
        assert global_auth is custom_auth


# =============================================================================
# AuthorizationDecision Tests
# =============================================================================


class TestAuthorizationDecision:
    """Test AuthorizationDecision functionality."""

    def test_decision_creation(self):
        """Test creating an authorization decision."""
        decision = AuthorizationDecision(
            allowed=True,
            reason="Allowed by role: developer",
        )
        assert decision.allowed is True
        assert "developer" in decision.reason

    def test_decision_with_policies(self):
        """Test decision with matched policies."""
        policy = Policy(
            name="test",
            effect=PolicyEffect.ALLOW,
            resource="tools",
            action="read",
        )
        decision = AuthorizationDecision(
            allowed=True,
            reason="Allowed by policy",
            matched_policies=[policy],
        )
        assert len(decision.matched_policies) == 1

    def test_decision_to_dict(self):
        """Test converting decision to dictionary."""
        decision = AuthorizationDecision(
            allowed=True,
            reason="Test reason",
        )
        decision_dict = decision.to_dict()
        assert decision_dict["allowed"] is True
        assert decision_dict["reason"] == "Test reason"
        assert "timestamp" in decision_dict
