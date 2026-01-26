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

"""Role-Based Access Control (RBAC) for Victor AI.

This is the canonical location for RBAC functionality.
The old location (victor.security.auth) is deprecated.

Example usage:
    from victor.core.security.auth import RBACManager, Permission, Role, User

    # Create RBAC manager
    rbac = RBACManager()

    # Define roles with permissions
    rbac.add_role(Role("admin", {Permission.ADMIN, Permission.WRITE}))
    rbac.add_role(Role("user", {Permission.READ}))

    # Create users and assign roles
    rbac.add_user(User("alice", roles={rbac.get_role("admin")}))
    rbac.add_user(User("bob", roles={rbac.get_role("user")}))

    # Check permissions
    if rbac.check_tool_access("alice", "shell", "execute", AccessMode.EXECUTE):
        # Allow execution
"""

# Import from local implementation file
from victor.core.security.auth.rbac_impl import (
    Permission,
    Role,
    User,
    RBACManager,
    get_permission_for_access_mode,
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
    "require_permission",
    "get_rbac_manager",
    "set_rbac_manager",
]
