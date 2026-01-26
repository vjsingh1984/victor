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

This is the canonical location for EnhancedAuthorizer.

Design Principles:
- Defense in depth: Multiple layers of authorization checks
- Least privilege: Default deny, explicit allow
- Separation of concerns: RBAC for coarse-grained, ABAC for fine-grained
- Integration with existing security infrastructure
- Configurable policies via YAML
- Thread-safe for concurrent access

Example:
    from victor.core.security.authorization import EnhancedAuthorizer

    authorizer = EnhancedAuthorizer()
    authorizer.load_from_yaml("config/policies.yaml")

    # Check permission
    if authorizer.check_permission(user, "tools", "execute"):
        # Allow tool execution
"""

# Re-export from the original location
from victor.security.authorization_enhanced import (
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
