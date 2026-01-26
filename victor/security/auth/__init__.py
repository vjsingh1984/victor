# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Authentication and Authorization module for Victor.

.. deprecated:: 0.6.0
    This module is deprecated. Please migrate to ``victor.core.security.auth``.
    This module will be removed in v1.0.0.

Migration Guide:
    Old (deprecated):
        from victor.security.auth import Permission, RBACManager

    New (recommended):
        from victor.core.security.auth import Permission, RBACManager

This module provides:
- Role-Based Access Control (RBAC) for tool execution
- Permission system integrated with tool AccessMode
- User/Role management with YAML configuration support
"""

import warnings

warnings.warn(
    "victor.security.auth is deprecated and will be removed in v1.0.0. "
    "Use victor.core.security.auth instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical location
from victor.core.security.auth import (
    Permission,
    Role,
    User,
    RBACManager,
    require_permission,
    get_permission_for_access_mode,
)

__all__ = [
    "Permission",
    "Role",
    "User",
    "RBACManager",
    "require_permission",
    "get_permission_for_access_mode",
]
