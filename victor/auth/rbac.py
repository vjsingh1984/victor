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

"""Role-Based Access Control (RBAC) system.

This module provides the core components for managing roles, users, and permissions.
"""

from enum import Enum
from typing import Dict, Optional, Set


class Permission(Enum):
    """Defines the permissions available in the system."""

    # Tool permissions
    TOOL_EXECUTE = "tool:execute"
    TOOL_EDIT = "tool:edit"
    TOOL_CREATE = "tool:create"
    TOOL_DELETE = "tool:delete"

    # Admin permissions
    ADMIN_MANAGE_USERS = "admin:manage_users"
    ADMIN_MANAGE_ROLES = "admin:manage_roles"
    ADMIN_VIEW_LOGS = "admin:view_logs"


class Role:
    """Represents a role with a set of permissions."""

    def __init__(self, name: str, permissions: Set[Permission]):
        self.name = name
        self.permissions = permissions

    def has_permission(self, permission: Permission) -> bool:
        """Checks if the role has a specific permission."""
        return permission in self.permissions


class User:
    """Represents a user with a set of roles."""

    def __init__(self, name: str, roles: Set[Role]):
        self.name = name
        self.roles = roles

    def has_permission(self, permission: Permission) -> bool:
        """Checks if the user has a specific permission through any of their roles."""
        return any(role.has_permission(permission) for role in self.roles)


class RBAC:
    """Provides the main interface for checking permissions."""

    def __init__(self) -> None:
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, User] = {}

    def add_role(self, role: Role) -> None:
        """Adds a role to the RBAC system."""
        self.roles[role.name] = role

    def get_role(self, name: str) -> Optional[Role]:
        """Gets a role by name."""
        return self.roles.get(name)

    def add_user(self, user: User) -> None:
        """Adds a user to the RBAC system."""
        self.users[user.name] = user

    def get_user(self, name: str) -> Optional[User]:
        """Gets a user by name."""
        return self.users.get(name)

    def check_permission(self, user_name: str, permission: Permission) -> bool:
        """Checks if a user has a specific permission."""
        user = self.get_user(user_name)
        if user:
            return user.has_permission(permission)
        return False
