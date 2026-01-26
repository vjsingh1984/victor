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

"""Enhanced Authorization with ABAC (Attribute-Based Access Control).

This is the canonical location for authorization functionality.
The old location (victor.security.authorization_enhanced) is deprecated.

Example usage:
    from victor.core.security.authorization import EnhancedAuthorizer

    authorizer = EnhancedAuthorizer()
    authorizer.load_from_yaml("config/policies.yaml")

    # Check permission
    if authorizer.check_permission(user, "tools", "execute"):
        # Allow tool execution
"""

# Import from local implementation (canonical location)
from victor.core.security.authorization_impl import (
    ResourceType,
    ActionType,
    PolicyEffect,
    Permission,
    Role,
    User,
    Policy,
    AuthorizationDecision,
    EnhancedAuthorizer,
)

__all__ = [
    "ResourceType",
    "ActionType",
    "PolicyEffect",
    "Permission",
    "Role",
    "User",
    "Policy",
    "AuthorizationDecision",
    "EnhancedAuthorizer",
]
