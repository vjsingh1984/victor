from __future__ import annotations

"""Tests for RBAC fail_open / fail-secure behaviour."""

import warnings

import pytest

from victor.security.auth.rbac import Permission, RBACManager


class TestRBACFailOpen:
    """Verify fail_open controls behaviour when RBAC is disabled."""

    def test_fail_open_true_allows_when_disabled(self):
        """Default fail_open=True should allow access when RBAC is disabled."""
        rbac = RBACManager(enabled=False, fail_open=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert rbac.check_permission("alice", Permission.WRITE) is True

    def test_fail_open_false_denies_when_disabled(self):
        """fail_open=False must deny access when RBAC is disabled."""
        rbac = RBACManager(enabled=False, fail_open=False)
        assert rbac.check_permission("alice", Permission.WRITE) is False

    def test_deprecation_warning_emitted_once(self):
        """DeprecationWarning should fire exactly once for fail_open=True."""
        rbac = RBACManager(enabled=False, fail_open=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rbac.check_permission("alice", Permission.READ)
            rbac.check_permission("bob", Permission.WRITE)

        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) == 1
        assert "fail_open=True is deprecated" in str(dep_warnings[0].message)

    def test_enabled_rbac_ignores_fail_open(self):
        """When RBAC is enabled, fail_open has no effect."""
        rbac = RBACManager(enabled=True, fail_open=False)
        # Add a role + user with WRITE permission
        from victor.security.auth.rbac import Role, User

        role = Role("dev", frozenset({Permission.READ, Permission.WRITE}))
        rbac.add_role(role)
        rbac.add_user(User("alice", roles={role}))

        assert rbac.check_permission("alice", Permission.WRITE) is True
        assert rbac.check_permission("alice", Permission.ADMIN) is False


class TestRBACFailOpenToolAccess:
    """Verify fail_open controls check_tool_access when RBAC is disabled."""

    def test_fail_open_true_allows_tool_access(self):
        rbac = RBACManager(enabled=False, fail_open=True)
        from victor.tools.base import AccessMode

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert rbac.check_tool_access("alice", "shell", "execute", AccessMode.EXECUTE) is True

    def test_fail_open_false_denies_tool_access(self):
        rbac = RBACManager(enabled=False, fail_open=False)
        from victor.tools.base import AccessMode

        assert rbac.check_tool_access("alice", "shell", "execute", AccessMode.EXECUTE) is False
