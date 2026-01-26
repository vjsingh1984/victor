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

.. deprecated:: 0.6.0
    This module is deprecated. Please migrate to ``victor.core.security.authorization``.
    This module will be removed in v1.0.0.

Migration Guide:
    Old (deprecated):
        from victor.security.authorization_enhanced import EnhancedAuthorizer

    New (recommended):
        from victor.core.security.authorization import EnhancedAuthorizer
        # or
        from victor.core.security import EnhancedAuthorizer

This module provides advanced authorization capabilities combining:
- Role-Based Access Control (RBAC): Permissions based on user roles
- Attribute-Based Access Control (ABAC): Permissions based on user/resource attributes
- Policy-based authorization: Fine-grained policy evaluation
"""

import warnings

warnings.warn(
    "victor.security.authorization_enhanced is deprecated and will be removed in v1.0.0. "
    "Use victor.core.security.authorization instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical location
from victor.core.security.authorization import (
    ResourceType,
    ActionType,
    PolicyEffect,
    Permission,
    Role,
    User,
    Policy,
    AuthorizationDecision,
    EnhancedAuthorizer,
    get_enhanced_authorizer,
    set_enhanced_authorizer,
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
    "get_enhanced_authorizer",
    "set_enhanced_authorizer",
]
