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

"""Enhanced Authorization System with RBAC and ABAC.

This module provides advanced authorization capabilities combining:
- Role-Based Access Control (RBAC): Permissions based on user roles
- Attribute-Based Access Control (ABAC): Permissions based on user/resource attributes
- Policy-based authorization: Fine-grained policy evaluation
- Integration with existing ActionAuthorizer and RBACManager

Design Principles:
- Defense in depth: Multiple layers of authorization checks
- Least privilege: Default deny, explicit allow
- Separation of concerns: RBAC for coarse-grained, ABAC for fine-grained
- Integration with existing security infrastructure
- Configurable policies via YAML
- Thread-safe for concurrent access

Example:
    authorizer = EnhancedAuthorizer()
    authorizer.load_from_yaml("config/policies.yaml")

    # Check permission
    if authorizer.check_permission(user, "tools", "execute"):
        # Allow tool execution
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union, TYPE_CHECKING
import yaml

if TYPE_CHECKING:
    from victor.agent.action_authorizer import ActionIntent

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class ResourceType(Enum):
    """Types of resources that can be protected."""

    TOOLS = "tools"
    WORKFLOWS = "workflows"
    VERTICALS = "verticals"
    SETTINGS = "settings"
    FILES = "files"
    API = "api"
    AGENTS = "agents"


class ActionType(Enum):
    """Actions that can be performed on resources."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"
    NETWORK = "network"


class PolicyEffect(Enum):
    """Effect of a policy rule."""

    ALLOW = "allow"
    DENY = "deny"


# =============================================================================
# Data Models
# =============================================================================


@dataclass(frozen=True)
class Permission:
    """A permission granted to a role or user.

    Attributes:
        resource: Type of resource (e.g., "tools", "workflows")
        action: Action that can be performed (e.g., "read", "write", "execute")
        constraints: Optional constraints on the permission
    """

    resource: str
    action: str
    constraints: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Make Permission hashable by using resource, action, and frozen constraints."""
        # Convert constraints dict to a sorted tuple of items for hashing
        constraints_tuple = tuple(sorted(self.constraints.items()))
        return hash((self.resource, self.action, constraints_tuple))

    def __eq__(self, other: object) -> bool:
        """Check equality based on resource, action, and constraints."""
        if not isinstance(other, Permission):
            return False
        return (
            self.resource == other.resource
            and self.action == other.action
            and self.constraints == other.constraints
        )

    def matches(self, resource: str, action: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if this permission matches a request.

        Args:
            resource: Resource type being accessed
            action: Action being performed
            context: Optional context for constraint evaluation

        Returns:
            True if permission matches the request
        """
        if self.resource != resource or self.action != action:
            return False

        # Check constraints if context provided
        if context and self.constraints:
            return self._evaluate_constraints(context)

        return True

    def _evaluate_constraints(self, context: Dict[str, Any]) -> bool:
        """Evaluate permission constraints against context.

        Args:
            context: Context containing user/resource attributes

        Returns:
            True if all constraints are satisfied
        """
        for key, value in self.constraints.items():
            if key not in context:
                return False

            if isinstance(value, list):
                if context[key] not in value:
                    return False
            elif context[key] != value:
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "resource": self.resource,
            "action": self.action,
            "constraints": self.constraints,
        }


@dataclass
class Role:
    """A role with permissions and optional attributes.

    Attributes:
        name: Role name (e.g., "admin", "developer", "viewer")
        permissions: Set of permissions granted to this role
        attributes: Role attributes for ABAC (e.g., clearance_level, department)
        description: Human-readable description
    """

    name: str
    permissions: Set[Permission] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def add_permission(self, permission: Permission) -> None:
        """Add a permission to this role.

        Args:
            permission: Permission to add
        """
        self.permissions.add(permission)
        logger.debug(
            f"Added permission {permission.resource}:{permission.action} to role {self.name}"
        )

    def remove_permission(self, permission: Permission) -> None:
        """Remove a permission from this role.

        Args:
            permission: Permission to remove
        """
        self.permissions.discard(permission)
        logger.debug(f"Removed permission from role {self.name}")

    def has_permission(self, resource: str, action: str) -> bool:
        """Check if role has a permission.

        Args:
            resource: Resource type
            action: Action type

        Returns:
            True if role has the permission
        """
        return any(p.resource == resource and p.action == action for p in self.permissions)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "permissions": [p.to_dict() for p in self.permissions],
            "attributes": self.attributes,
            "description": self.description,
        }


@dataclass
class User:
    """A user with roles and attributes for ABAC.

    Attributes:
        id: Unique user identifier
        username: Username
        roles: Set of role names assigned to the user
        attributes: User attributes for ABAC (e.g., department, clearance_level)
        permissions: Direct permissions assigned to user (not through roles)
        enabled: Whether the user is enabled
    """

    id: str
    username: str
    roles: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    permissions: Set[Permission] = field(default_factory=set)
    enabled: bool = True

    def add_role(self, role_name: str) -> None:
        """Add a role to the user.

        Args:
            role_name: Name of the role to add
        """
        self.roles.add(role_name)
        logger.debug(f"Added role {role_name} to user {self.username}")

    def remove_role(self, role_name: str) -> None:
        """Remove a role from the user.

        Args:
            role_name: Name of the role to remove
        """
        self.roles.discard(role_name)
        logger.debug(f"Removed role {role_name} from user {self.username}")

    def add_permission(self, permission: Permission) -> None:
        """Add a direct permission to the user.

        Args:
            permission: Permission to add
        """
        self.permissions.add(permission)
        logger.debug(f"Added direct permission to user {self.username}")

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get a user attribute.

        Args:
            key: Attribute key
            default: Default value if not found

        Returns:
            Attribute value or default
        """
        return self.attributes.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "username": self.username,
            "roles": list(self.roles),
            "attributes": self.attributes,
            "permissions": [p.to_dict() for p in self.permissions],
            "enabled": self.enabled,
        }


@dataclass
class Policy:
    """An authorization policy with rules and conditions.

    Attributes:
        name: Policy name
        effect: ALLOW or DENY
        resource: Resource type this policy applies to
        action: Action this policy applies to
        subjects: Who this policy applies to (user IDs, role names, or attributes)
        conditions: Conditions that must be met (ABAC rules)
        priority: Policy priority (higher = evaluated first)
        description: Human-readable description
    """

    name: str
    effect: PolicyEffect
    resource: str
    action: str
    subjects: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    description: str = ""

    def matches(
        self,
        resource: str,
        action: str,
        subject: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check if this policy matches a request.

        Args:
            resource: Resource type
            action: Action type
            subject: User ID or role name
            context: Optional context for condition evaluation

        Returns:
            True if policy matches the request
        """
        # Check resource and action
        if self.resource != resource or self.action != action:
            return False

        # Check subjects
        if self.subjects and subject not in self.subjects:
            # Check if subject matches via attribute conditions
            if not self._check_subject_attributes(subject, context):
                return False

        # Check conditions
        if context and self.conditions:
            return self._evaluate_conditions(context)

        return True

    def _check_subject_attributes(self, subject: str, context: Optional[Dict[str, Any]]) -> bool:
        """Check if subject matches via attribute conditions.

        Args:
            subject: Subject identifier
            context: Request context

        Returns:
            True if subject matches
        """
        if not context:
            return False

        # Check role-based conditions
        if "roles" in self.conditions:
            subject_roles = context.get("roles", set())
            required_roles = set(self.conditions["roles"])
            if not required_roles.intersection(subject_roles):
                return False

        return True

    def _evaluate_conditions(self, context: Dict[str, Any]) -> bool:
        """Evaluate policy conditions against context.

        Args:
            context: Request context with user/resource attributes

        Returns:
            True if all conditions are satisfied
        """
        for key, value in self.conditions.items():
            if key == "roles":
                # Already checked in _check_subject_attributes
                continue

            if key not in context:
                return False

            if isinstance(value, list):
                if context[key] not in value:
                    return False
            elif isinstance(value, dict):
                # Range or operator-based conditions
                if not self._evaluate_operator_condition(key, value, context):
                    return False
            elif context[key] != value:
                return False

        return True

    def _evaluate_operator_condition(
        self, key: str, condition: Dict[str, Any], context: Dict[str, Any]
    ) -> bool:
        """Evaluate operator-based condition.

        Args:
            key: Attribute key
            condition: Condition with operator
            context: Request context

        Returns:
            True if condition is satisfied
        """
        if key not in context:
            return False

        actual_value = context[key]

        for operator, expected_value in condition.items():
            if operator == "gt" and not (actual_value > expected_value):
                return False
            elif operator == "gte" and not (actual_value >= expected_value):
                return False
            elif operator == "lt" and not (actual_value < expected_value):
                return False
            elif operator == "lte" and not (actual_value <= expected_value):
                return False
            elif operator == "ne" and actual_value == expected_value:
                return False
            elif operator == "in" and actual_value not in expected_value:
                return False
            elif operator == "not_in" and actual_value in expected_value:
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "effect": self.effect.value,
            "resource": self.resource,
            "action": self.action,
            "subjects": self.subjects,
            "conditions": self.conditions,
            "priority": self.priority,
            "description": self.description,
        }


@dataclass
class AuthorizationDecision:
    """Result of an authorization check.

    Attributes:
        allowed: Whether the action is allowed
        reason: Human-readable reason for the decision
        matched_policies: Policies that matched the request
        timestamp: When the decision was made
    """

    allowed: bool
    reason: str
    matched_policies: List[Policy] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "matched_policies": [p.name for p in self.matched_policies],
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Enhanced Authorizer
# =============================================================================


class EnhancedAuthorizer:
    """Enhanced authorization system combining RBAC and ABAC.

    This authorizer provides:
    - Role-based permissions (RBAC)
    - Attribute-based permissions (ABAC)
    - Policy-based authorization with fine-grained control
    - Integration with existing ActionAuthorizer and RBACManager
    - Thread-safe operations
    - YAML configuration support

    Example:
        authorizer = EnhancedAuthorizer()
        authorizer.load_from_yaml("config/authorization.yaml")

        # Create a user
        user = User(id="user1", username="alice")
        authorizer.assign_role(user.id, "developer")

        # Check permission
        decision = authorizer.check_permission(
            user=user,
            resource="tools",
            action="execute",
            context={"tool_name": "bash"}
        )

        if decision.allowed:
            # Proceed with action
    """

    # Default roles
    DEFAULT_ROLES = {
        "admin": Role(
            name="admin",
            permissions={
                Permission(resource="*", action="*"),  # All permissions
            },
            attributes={"clearance_level": 5},
            description="Full system access",
        ),
        "developer": Role(
            name="developer",
            permissions={
                Permission(resource="tools", action="read"),
                Permission(resource="tools", action="write"),
                Permission(resource="tools", action="execute"),
                Permission(resource="workflows", action="read"),
                Permission(resource="workflows", action="execute"),
                Permission(resource="files", action="read"),
                Permission(resource="files", action="write"),
            },
            attributes={"clearance_level": 3},
            description="Developer access",
        ),
        "operator": Role(
            name="operator",
            permissions={
                Permission(resource="tools", action="read"),
                Permission(resource="tools", action="execute"),
                Permission(resource="workflows", action="read"),
                Permission(resource="workflows", action="execute"),
            },
            attributes={"clearance_level": 2},
            description="Operator access",
        ),
        "viewer": Role(
            name="viewer",
            permissions={
                Permission(resource="tools", action="read"),
                Permission(resource="workflows", action="read"),
                Permission(resource="files", action="read"),
            },
            attributes={"clearance_level": 1},
            description="Read-only access",
        ),
    }

    def __init__(
        self,
        enabled: bool = True,
        default_deny: bool = True,
    ):
        """Initialize the enhanced authorizer.

        Args:
            enabled: If False, all authorization checks pass
            default_deny: If True, deny by default (secure default)
        """
        self._enabled = enabled
        self._default_deny = default_deny
        self._lock = threading.RLock()

        # Initialize with default roles
        self._roles: Dict[str, Role] = {name: role for name, role in self.DEFAULT_ROLES.items()}

        # Users and policies
        self._users: Dict[str, User] = {}
        self._policies: List[Policy] = []

        logger.info(
            f"EnhancedAuthorizer initialized (enabled={enabled}, default_deny={default_deny})"
        )

    @property
    def enabled(self) -> bool:
        """Check if authorization is enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable authorization checks."""
        with self._lock:
            self._enabled = True
            logger.info("Enhanced authorization enabled")

    def disable(self) -> None:
        """Disable authorization checks (all actions allowed)."""
        with self._lock:
            self._enabled = False
            logger.warning("Enhanced authorization disabled - all actions allowed")

    # -------------------------------------------------------------------------
    # Role Management
    # -------------------------------------------------------------------------

    def create_role(
        self,
        name: str,
        permissions: List[Permission],
        attributes: Optional[Dict[str, Any]] = None,
        description: str = "",
    ) -> Role:
        """Create a new role.

        Args:
            name: Role name
            permissions: List of permissions for the role
            attributes: Role attributes for ABAC
            description: Human-readable description

        Returns:
            Created Role object
        """
        with self._lock:
            role = Role(
                name=name,
                permissions=set(permissions),
                attributes=attributes or {},
                description=description,
            )
            self._roles[name] = role
            logger.info(f"Created role: {name} with {len(permissions)} permissions")
            return role

    def get_role(self, name: str) -> Optional[Role]:
        """Get a role by name.

        Args:
            name: Role name

        Returns:
            Role or None if not found
        """
        with self._lock:
            return self._roles.get(name)

    def list_roles(self) -> List[Role]:
        """List all roles.

        Returns:
            List of all roles
        """
        with self._lock:
            return list(self._roles.values())

    def delete_role(self, name: str) -> bool:
        """Delete a role.

        Args:
            name: Role name

        Returns:
            True if role was deleted
        """
        with self._lock:
            if name in self._roles:
                del self._roles[name]
                logger.info(f"Deleted role: {name}")
                return True
            return False

    # -------------------------------------------------------------------------
    # User Management
    # -------------------------------------------------------------------------

    def create_user(
        self,
        user_id: str,
        username: str,
        roles: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> User:
        """Create a new user.

        Args:
            user_id: Unique user identifier
            username: Username
            roles: List of role names to assign
            attributes: User attributes for ABAC

        Returns:
            Created User object
        """
        with self._lock:
            user = User(
                id=user_id,
                username=username,
                roles=set(roles or []),
                attributes=attributes or {},
            )
            self._users[user_id] = user
            logger.info(f"Created user: {username} ({user_id})")
            return user

    def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID.

        Args:
            user_id: User ID

        Returns:
            User or None if not found
        """
        with self._lock:
            return self._users.get(user_id)

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by username.

        Args:
            username: Username

        Returns:
            User or None if not found
        """
        with self._lock:
            for user in self._users.values():
                if user.username == username:
                    return user
            return None

    def list_users(self) -> List[User]:
        """List all users.

        Returns:
            List of all users
        """
        with self._lock:
            return list(self._users.values())

    def delete_user(self, user_id: str) -> bool:
        """Delete a user.

        Args:
            user_id: User ID

        Returns:
            True if user was deleted
        """
        with self._lock:
            if user_id in self._users:
                del self._users[user_id]
                logger.info(f"Deleted user: {user_id}")
                return True
            return False

    def assign_role(self, user_id: str, role_name: str) -> bool:
        """Assign a role to a user.

        Args:
            user_id: User ID
            role_name: Role name

        Returns:
            True if role was assigned
        """
        with self._lock:
            user = self._users.get(user_id)
            role = self._roles.get(role_name)

            if not user:
                logger.warning(f"User not found: {user_id}")
                return False

            if not role:
                logger.warning(f"Role not found: {role_name}")
                return False

            user.add_role(role_name)
            logger.info(f"Assigned role {role_name} to user {user.username}")
            return True

    def revoke_role(self, user_id: str, role_name: str) -> bool:
        """Revoke a role from a user.

        Args:
            user_id: User ID
            role_name: Role name

        Returns:
            True if role was revoked
        """
        with self._lock:
            user = self._users.get(user_id)
            if not user:
                return False

            user.remove_role(role_name)
            logger.info(f"Revoked role {role_name} from user {user.username}")
            return True

    # -------------------------------------------------------------------------
    # Permission Management
    # -------------------------------------------------------------------------

    def grant_permission(self, role_name: str, permission: Permission) -> bool:
        """Grant a permission to a role.

        Args:
            role_name: Role name
            permission: Permission to grant

        Returns:
            True if permission was granted
        """
        with self._lock:
            role = self._roles.get(role_name)
            if not role:
                logger.warning(f"Role not found: {role_name}")
                return False

            role.add_permission(permission)
            logger.info(
                f"Granted permission {permission.resource}:{permission.action} to role {role_name}"
            )
            return True

    def revoke_permission(self, role_name: str, permission: Permission) -> bool:
        """Revoke a permission from a role.

        Args:
            role_name: Role name
            permission: Permission to revoke

        Returns:
            True if permission was revoked
        """
        with self._lock:
            role = self._roles.get(role_name)
            if not role:
                return False

            role.remove_permission(permission)
            logger.info(f"Revoked permission from role {role_name}")
            return True

    # -------------------------------------------------------------------------
    # Policy Management
    # -------------------------------------------------------------------------

    def create_policy(
        self,
        name: str,
        effect: PolicyEffect,
        resource: str,
        action: str,
        subjects: Optional[List[str]] = None,
        conditions: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        description: str = "",
    ) -> Policy:
        """Create an authorization policy.

        Args:
            name: Policy name
            effect: ALLOW or DENY
            resource: Resource type
            action: Action type
            subjects: List of user IDs or role names
            conditions: Conditions for ABAC
            priority: Policy priority (higher evaluated first)
            description: Human-readable description

        Returns:
            Created Policy object
        """
        with self._lock:
            policy = Policy(
                name=name,
                effect=effect,
                resource=resource,
                action=action,
                subjects=subjects or [],
                conditions=conditions or {},
                priority=priority,
                description=description,
            )
            self._policies.append(policy)
            # Sort policies by priority (highest first)
            self._policies.sort(key=lambda p: p.priority, reverse=True)
            logger.info(f"Created policy: {name} (effect={effect.value}, priority={priority})")
            return policy

    def get_policy(self, name: str) -> Optional[Policy]:
        """Get a policy by name.

        Args:
            name: Policy name

        Returns:
            Policy or None if not found
        """
        with self._lock:
            for policy in self._policies:
                if policy.name == name:
                    return policy
            return None

    def list_policies(self) -> List[Policy]:
        """List all policies.

        Returns:
            List of all policies
        """
        with self._lock:
            return list(self._policies)

    def delete_policy(self, name: str) -> bool:
        """Delete a policy.

        Args:
            name: Policy name

        Returns:
            True if policy was deleted
        """
        with self._lock:
            for i, policy in enumerate(self._policies):
                if policy.name == name:
                    del self._policies[i]
                    logger.info(f"Deleted policy: {name}")
                    return True
            return False

    # -------------------------------------------------------------------------
    # Authorization Checks
    # -------------------------------------------------------------------------

    def check_permission(
        self,
        user: User,
        resource: str,
        action: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuthorizationDecision:
        """Check if a user has permission to perform an action on a resource.

        This is the main authorization method combining RBAC and ABAC.

        Args:
            user: User to check
            resource: Resource type
            action: Action to perform
            context: Optional context for ABAC evaluation

        Returns:
            AuthorizationDecision with result and reasoning
        """
        # If authorization is disabled, allow all
        if not self._enabled:
            return AuthorizationDecision(
                allowed=True,
                reason="Authorization is disabled",
            )

        # Check if user is enabled
        if not user.enabled:
            return AuthorizationDecision(
                allowed=False,
                reason=f"User {user.username} is disabled",
            )

        # Build context with user info
        full_context = {
            "user_id": user.id,
            "username": user.username,
            "roles": list(user.roles),
            **user.attributes,
        }
        if context:
            full_context.update(context)

        # Check policies first (highest priority)
        matched_policies = []
        for policy in self._policies:
            if policy.matches(resource, action, user.id, full_context):
                matched_policies.append(policy)

                # DENY policies take precedence
                if policy.effect == PolicyEffect.DENY:
                    return AuthorizationDecision(
                        allowed=False,
                        reason=f"Denied by policy: {policy.name}",
                        matched_policies=[policy],
                    )

        # If any ALLOW policy matched, allow
        if matched_policies:
            return AuthorizationDecision(
                allowed=True,
                reason=f"Allowed by policy: {matched_policies[0].name}",
                matched_policies=matched_policies,
            )

        # Check role-based permissions (RBAC)
        for role_name in user.roles:
            role = self._roles.get(role_name)
            if role and role.has_permission(resource, action):
                # Check permission constraints
                for perm in role.permissions:
                    if perm.matches(resource, action, full_context):
                        return AuthorizationDecision(
                            allowed=True,
                            reason=f"Allowed by role: {role_name}",
                        )

        # Check direct user permissions
        for perm in user.permissions:
            if perm.matches(resource, action, full_context):
                return AuthorizationDecision(
                    allowed=True,
                    reason="Allowed by direct permission",
                )

        # Default deny
        if self._default_deny:
            return AuthorizationDecision(
                allowed=False,
                reason=f"Default deny: no permission for {resource}:{action}",
            )

        return AuthorizationDecision(
            allowed=True,
            reason="Default allow",
        )

    def check_permission_by_id(
        self,
        user_id: str,
        resource: str,
        action: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuthorizationDecision:
        """Check permission by user ID.

        Args:
            user_id: User ID
            resource: Resource type
            action: Action type
            context: Optional context

        Returns:
            AuthorizationDecision
        """
        user = self.get_user(user_id)
        if not user:
            return AuthorizationDecision(
                allowed=False,
                reason=f"User not found: {user_id}",
            )

        return self.check_permission(user, resource, action, context)

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def load_from_dict(self, config: Dict[str, Any]) -> None:
        """Load authorization configuration from dictionary.

        Args:
            config: Configuration dictionary
        """
        with self._lock:
            # Load settings
            self._enabled = config.get("enabled", True)
            self._default_deny = config.get("default_deny", True)

            # Load roles
            for role_name, role_data in config.get("roles", {}).items():
                permissions = []
                for perm_data in role_data.get("permissions", []):
                    perm = Permission(
                        resource=perm_data["resource"],
                        action=perm_data["action"],
                        constraints=perm_data.get("constraints", {}),
                    )
                    permissions.append(perm)

                role = Role(
                    name=role_name,
                    permissions=set(permissions),
                    attributes=role_data.get("attributes", {}),
                    description=role_data.get("description", ""),
                )
                self._roles[role_name] = role

            # Load users
            for user_id, user_data in config.get("users", {}).items():
                user = User(
                    id=user_id,
                    username=user_data.get("username", user_id),
                    roles=set(user_data.get("roles", [])),
                    attributes=user_data.get("attributes", {}),
                    enabled=user_data.get("enabled", True),
                )

                # Load direct permissions
                for perm_data in user_data.get("permissions", []):
                    perm = Permission(
                        resource=perm_data["resource"],
                        action=perm_data["action"],
                        constraints=perm_data.get("constraints", {}),
                    )
                    user.permissions.add(perm)

                self._users[user_id] = user

            # Load policies
            for policy_data in config.get("policies", []):
                policy = Policy(
                    name=policy_data["name"],
                    effect=PolicyEffect(policy_data["effect"]),
                    resource=policy_data["resource"],
                    action=policy_data["action"],
                    subjects=policy_data.get("subjects", []),
                    conditions=policy_data.get("conditions", {}),
                    priority=policy_data.get("priority", 0),
                    description=policy_data.get("description", ""),
                )
                self._policies.append(policy)

            # Sort policies by priority
            self._policies.sort(key=lambda p: p.priority, reverse=True)

            logger.info(
                f"Loaded authorization config: {len(self._roles)} roles, "
                f"{len(self._users)} users, {len(self._policies)} policies"
            )

    def load_from_yaml(self, path: Path) -> None:
        """Load authorization configuration from YAML file.

        Args:
            path: Path to YAML configuration file
        """
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        self.load_from_dict(config)

    def save_to_yaml(self, path: Path) -> None:
        """Save authorization configuration to YAML file.

        Args:
            path: Path to save configuration
        """
        config = {
            "enabled": self._enabled,
            "default_deny": self._default_deny,
            "roles": {name: role.to_dict() for name, role in self._roles.items()},
            "users": {user.id: user.to_dict() for user in self._users.values()},
            "policies": [p.to_dict() for p in self._policies],
        }

        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Saved authorization config to {path}")

    # -------------------------------------------------------------------------
    # Statistics and Monitoring
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get authorization statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            return {
                "enabled": self._enabled,
                "default_deny": self._default_deny,
                "roles_count": len(self._roles),
                "users_count": len(self._users),
                "policies_count": len(self._policies),
                "role_names": list(self._roles.keys()),
            }


# =============================================================================
# Global Instance
# =============================================================================

_global_authorizer: Optional[EnhancedAuthorizer] = None


def get_enhanced_authorizer() -> EnhancedAuthorizer:
    """Get or create the global enhanced authorizer.

    Returns:
        Global EnhancedAuthorizer instance
    """
    global _global_authorizer
    if _global_authorizer is None:
        _global_authorizer = EnhancedAuthorizer()
    return _global_authorizer


def set_enhanced_authorizer(authorizer: EnhancedAuthorizer) -> None:
    """Set the global enhanced authorizer.

    Args:
        authorizer: EnhancedAuthorizer to use globally
    """
    global _global_authorizer
    _global_authorizer = authorizer
