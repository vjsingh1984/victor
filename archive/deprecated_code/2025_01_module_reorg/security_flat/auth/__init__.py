# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Authentication and Authorization module for Victor.

This module has moved to victor.security.auth.
Please update your imports to use the new location.

This stub provides backward compatibility.
"""

# Re-export from new location for backward compatibility
from victor.security.auth import (
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
