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

"""Authentication and Authorization Module.

This module provides Role-Based Access Control (RBAC) for Victor tool execution.

Usage:
    from victor.core.security.auth import RBACManager, Permission, Role

    rbac = RBACManager()
    rbac.add_role(Role("admin", {Permission.ADMIN}))

    if rbac.check_permission("alice", Permission.EXECUTE):
        # Allow execution
"""

from victor.core.security.auth.rbac import (
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
