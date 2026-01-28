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
(RBAC, audit, authorization, patterns, extensions) from security analysis tools.

Structure:
- victor.core.security.pickle: Safe pickle utilities for cache security
- victor.core.security.auth: Role-Based Access Control (RBAC)
- victor.core.security.audit: Audit logging and compliance
- victor.core.security.authorization: Enhanced authorization (RBAC + ABAC)
- victor.core.security.protocol: CVE/vulnerability type definitions
- victor.core.security.patterns: Security patterns (secrets, PII, code safety, etc.)
- victor.core.security.safety_extensions: Safety extensions for vertical integration

Note: For security analysis tools (scanners, CVE databases, penetration testing),
use victor.security_analysis instead.

Usage:
    from victor.core.security import safe_pickle_dumps, safe_pickle_loads
    from victor.core.security import RBACManager, Permission
    from victor.core.security.audit import AuditManager
    from victor.core.security.authorization import EnhancedAuthorizer
    from victor.core.security.patterns import detect_secrets, PIIScanner
    from victor.core.security import SafetyExtensions
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Import safe pickle utilities for cache security
from victor.core.security.pickle import (
    CACHE_SIGNING_KEY_ENV,
    is_signed_pickle_data,
    safe_pickle_dumps,
    safe_pickle_loads,
)

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

# Import security patterns (cross-cutting framework services)
from victor.core.security.patterns import (
    SafetyPattern,
    ISafetyScanner,
    SafetyRegistry,
    CREDENTIAL_PATTERNS,
    SecretMatch,
    SecretScanner,
    SecretSeverity,
    detect_secrets,
    ANONYMIZATION_SUGGESTIONS,
    PII_COLUMN_PATTERNS,
    PII_CONTENT_PATTERNS,
    PII_SEVERITY,
    PIIMatch,
    PIIScanner,
    PIISeverity,
    PIIType,
    detect_pii_columns,
    detect_pii_in_content,
    get_anonymization_suggestion,
    get_pii_severity,
    get_pii_types,
    get_safety_reminders,
    has_pii,
    CodePatternCategory,
    CodePatternScanner,
    RiskLevel,
    GIT_PATTERNS,
    REFACTORING_PATTERNS,
    PACKAGE_MANAGER_PATTERNS,
    BUILD_DEPLOY_PATTERNS,
    SENSITIVE_FILE_PATTERNS,
    SafetyScanResult,
    scan_command,
    is_sensitive_file,
    get_all_patterns,
    InfraPatternCategory,
    InfrastructureScanner,
    CredibilityLevel,
    CredibilityMatch,
    SourceCredibilityScanner,
    validate_source_credibility,
    ContentWarningLevel,
    ContentWarningMatch,
    ContentPatternScanner,
)

# Import safety extensions (framework integration for safety patterns)
from victor.core.security.safety_extensions import SafetyExtensions

__all__ = [
    # Pickle utilities
    "safe_pickle_dumps",
    "safe_pickle_loads",
    "is_signed_pickle_data",
    "CACHE_SIGNING_KEY_ENV",
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
    "VulnerabilityStatus",
    "CVSSMetrics",
    "CVE",
    "SecurityDependency",
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
    # Security Patterns
    "SafetyPattern",
    "ISafetyScanner",
    "SafetyRegistry",
    "CREDENTIAL_PATTERNS",
    "SecretMatch",
    "SecretScanner",
    "SecretSeverity",
    "detect_secrets",
    "ANONYMIZATION_SUGGESTIONS",
    "PII_COLUMN_PATTERNS",
    "PII_CONTENT_PATTERNS",
    "PII_SEVERITY",
    "PIIMatch",
    "PIIScanner",
    "PIISeverity",
    "PIIType",
    "detect_pii_columns",
    "detect_pii_in_content",
    "get_anonymization_suggestion",
    "get_pii_severity",
    "get_pii_types",
    "get_safety_reminders",
    "has_pii",
    "CodePatternCategory",
    "CodePatternScanner",
    "RiskLevel",
    "GIT_PATTERNS",
    "REFACTORING_PATTERNS",
    "PACKAGE_MANAGER_PATTERNS",
    "BUILD_DEPLOY_PATTERNS",
    "SENSITIVE_FILE_PATTERNS",
    "SafetyScanResult",
    "scan_command",
    "is_sensitive_file",
    "get_all_patterns",
    "InfraPatternCategory",
    "InfrastructureScanner",
    "CredibilityLevel",
    "CredibilityMatch",
    "SourceCredibilityScanner",
    "validate_source_credibility",
    "ContentWarningLevel",
    "ContentWarningMatch",
    "ContentPatternScanner",
    # Safety Extensions
    "SafetyExtensions",
]
