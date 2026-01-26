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

"""Audit Protocol - Unified interface for compliance and audit logging.

This module defines the abstract interface and data structures for
enterprise compliance tracking, audit logging, and governance reporting.

This is the canonical location for audit protocol definitions.

Supports compliance frameworks:
- SOC 2 Type II
- GDPR
- HIPAA
- PCI DSS
- ISO 27001
- SOX
"""

# Re-export everything from the original location
from victor.security.audit.protocol import (
    AuditEventType,
    AuditSeverity,
    Severity,
    ComplianceFramework,
    AuditEvent,
    ComplianceRule,
    ComplianceViolation,
    AuditReport,
    RetentionPolicy,
    AuditConfig,
    AuditLoggerProtocol,
    ComplianceCheckerProtocol,
    DEFAULT_COMPLIANCE_RULES,
)

__all__ = [
    "AuditEventType",
    "AuditSeverity",
    "Severity",
    "ComplianceFramework",
    "AuditEvent",
    "ComplianceRule",
    "ComplianceViolation",
    "AuditReport",
    "RetentionPolicy",
    "AuditConfig",
    "AuditLoggerProtocol",
    "ComplianceCheckerProtocol",
    "DEFAULT_COMPLIANCE_RULES",
]
