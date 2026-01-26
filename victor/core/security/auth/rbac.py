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

"""Role-Based Access Control (RBAC) for Victor tool execution.

This module is the canonical location for RBAC implementation.
It provides enterprise-grade RBAC that integrates with Victor's
existing AccessMode system for tool security.

Design Principles:
- Permission model aligned with AccessMode enum (READONLY, WRITE, EXECUTE, NETWORK, MIXED)
- Category-based permission derivation (not string matching on tool names)
- YAML-configurable roles and users
- Decorator support for permission checks
- Thread-safe for concurrent access

Example Configuration (in profiles.yaml or rbac.yaml):
    rbac:
      enabled: true
      default_role: viewer
      roles:
        admin:
          permissions: [READ, WRITE, EXECUTE, NETWORK, ADMIN]
        developer:
          permissions: [READ, WRITE, EXECUTE]
          tool_categories: [filesystem, git, testing]
        viewer:
          permissions: [READ]
      users:
        alice:
          roles: [admin]
        bob:
          roles: [developer]

Usage:
    from victor.core.security.auth import RBACManager, Permission, Role

    rbac = RBACManager()
    rbac.add_role(Role("admin", {Permission.ADMIN}))
    rbac.add_user(User("alice", roles={rbac.get_role("admin")}))

    if rbac.check_tool_access("alice", "shell", "execute", AccessMode.EXECUTE):
        # Allow execution
"""

# Re-export everything from the original location
# This allows for a gradual migration while maintaining the canonical location
from victor.security.auth.rbac import (
    Permission,
    Role,
    User,
    RBACManager,
    get_permission_for_access_mode,
    # Utility functions
    require_permission,
    get_rbac_manager,
    set_rbac_manager,
)

__all__ = [
    "Permission",
    "Role",
    "User",
    "RBACManager",
    "get_permission_for_access_mode",
    # Utility functions
    "require_permission",
    "get_rbac_manager",
    "set_rbac_manager",
]
