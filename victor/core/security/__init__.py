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

"""Core Security Infrastructure Module.

This module provides security infrastructure components that are used
across the Victor framework. It separates security infrastructure
(RBAC, audit, authorization) from security analysis tools.

Structure:
- victor.core.security.auth: Role-Based Access Control (RBAC)
- victor.core.security.audit: Audit logging and compliance
- victor.core.security.authorization: Enhanced authorization (RBAC + ABAC)
- victor.core.security.protocol: CVE/vulnerability type definitions

Note: For security analysis tools (scanners, CVE databases, penetration testing),
use victor.security_analysis instead.

Usage:
    from victor.core.security import RBACManager, Permission
    from victor.core.security.audit import AuditManager
    from victor.core.security.authorization import EnhancedAuthorizer
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Import auth module for RBAC
from victor.core.security.auth import (
    Permission,
    Role,
    User,
    RBACManager,
    get_permission_for_access_mode,
)

# Import audit module for compliance and logging
from victor.core.security.audit import (
    AuditManager,
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    AuditConfig,
    AuditReport,
    ComplianceFramework,
    ComplianceRule,
    ComplianceViolation,
    RetentionPolicy,
    AuditLoggerProtocol,
    ComplianceCheckerProtocol,
    FileAuditLogger,
    DefaultComplianceChecker,
    create_event,
    get_compliance_summary,
    DEFAULT_COMPLIANCE_RULES,
)

# Import protocol for CVE/vulnerability types
from victor.core.security.protocol import (
    CVESeverity,
    VulnerabilityStatus,
    CVSSMetrics,
    CVE,
    SecurityDependency,
    Vulnerability,
    SecurityScanResult,
    SecurityPolicy,
    # Backward compatibility aliases
    Severity,
    Dependency,
)

# Import authorization for enhanced RBAC + ABAC
from victor.core.security.authorization import (
    EnhancedAuthorizer,
    ResourceType,
    ActionType,
    PolicyEffect,
    AuthorizationDecision,
    Policy,
)

# Re-export Permission and Role from authorization module with different names
# to avoid collision with auth.rbac exports
from victor.core.security.authorization import (
    Permission as AuthzPermission,
    Role as AuthzRole,
    User as AuthzUser,
)

__all__ = [
    # Auth (RBAC)
    "Permission",
    "Role",
    "User",
    "RBACManager",
    "get_permission_for_access_mode",
    # Audit
    "AuditManager",
    "AuditEvent",
    "AuditEventType",
    "AuditSeverity",
    "AuditConfig",
    "AuditReport",
    "ComplianceFramework",
    "ComplianceRule",
    "ComplianceViolation",
    "RetentionPolicy",
    "AuditLoggerProtocol",
    "ComplianceCheckerProtocol",
    "FileAuditLogger",
    "DefaultComplianceChecker",
    "create_event",
    "get_compliance_summary",
    "DEFAULT_COMPLIANCE_RULES",
    # Protocol (CVE types)
    "CVESeverity",
    "Severity",  # Alias for backward compatibility
    "VulnerabilityStatus",
    "CVSSMetrics",
    "CVE",
    "SecurityDependency",
    "Dependency",  # Alias for backward compatibility
    "Vulnerability",
    "SecurityScanResult",
    "SecurityPolicy",
    # Authorization (Enhanced)
    "EnhancedAuthorizer",
    "ResourceType",
    "ActionType",
    "PolicyEffect",
    "AuthorizationDecision",
    "Policy",
    "AuthzPermission",
    "AuthzRole",
    "AuthzUser",
]
