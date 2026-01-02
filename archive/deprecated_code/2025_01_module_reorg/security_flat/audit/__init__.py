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

"""Compliance & Audit Module - Enterprise governance for Victor.

DEPRECATED: This module has moved to victor.security.audit.
Please update your imports to use the new location.

This stub provides backward compatibility but will be removed in a future release.
"""

import warnings

warnings.warn(
    "victor.audit is deprecated and has moved to victor.security.audit. "
    "Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new location for backward compatibility
from victor.security.audit.checker import DefaultComplianceChecker, get_compliance_summary
from victor.security.audit.logger import FileAuditLogger, create_event
from victor.security.audit.manager import AuditManager
from victor.security.audit.protocol import (
    AuditConfig,
    AuditEvent,
    AuditEventType,
    AuditLoggerProtocol,
    AuditReport,
    ComplianceCheckerProtocol,
    ComplianceFramework,
    ComplianceRule,
    ComplianceViolation,
    DEFAULT_COMPLIANCE_RULES,
    RetentionPolicy,
    Severity,
)

__all__ = [
    # Manager
    "AuditManager",
    # Protocols
    "AuditLoggerProtocol",
    "ComplianceCheckerProtocol",
    # Data classes
    "AuditEventType",
    "Severity",
    "ComplianceFramework",
    "AuditEvent",
    "ComplianceRule",
    "ComplianceViolation",
    "AuditReport",
    "AuditConfig",
    "RetentionPolicy",
    # Implementations
    "FileAuditLogger",
    "DefaultComplianceChecker",
    # Utilities
    "create_event",
    "get_compliance_summary",
    # Constants
    "DEFAULT_COMPLIANCE_RULES",
]
