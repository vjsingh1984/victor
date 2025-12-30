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

"""Unified Security Module for Victor.

This module provides comprehensive security capabilities including:

Submodules:
- victor.security.auth: Authentication and authorization (RBAC)
- victor.security.safety: Safety patterns (secrets, PII, code patterns)
- victor.security.audit: Compliance and audit logging

Also includes:
- Vulnerability scanning and CVE database integration
- Security policy enforcement and reporting

Example usage:
    from victor.security import get_security_manager, SecurityPolicy
    from victor.security.auth import RBACManager, Permission
    from victor.security.safety import detect_secrets, CodePatternScanner
    from victor.security.audit import AuditManager
    from pathlib import Path
    import asyncio

    async def scan_project():
        # Vulnerability scanning
        manager = get_security_manager(project_root=Path("."))
        manager.policy = SecurityPolicy(
            fail_on_critical=True,
            fail_on_high=True,
            max_medium=10,
        )

        # Scan project
        result = await manager.scan()

        # Generate report
        print(manager.generate_report(format="text"))

        # Check policy
        passed, failures = manager.check_policy()
        if not passed:
            for failure in failures:
                print(f"FAIL: {failure}")

        # Secret detection
        secrets = detect_secrets(code_content)

        # RBAC
        rbac = RBACManager()
        if rbac.check_permission("user", Permission.WRITE):
            # Allow write operation
            pass

        # Audit logging
        audit = AuditManager.get_instance()
        await audit.log_file_operation("read", "/path/to/file")

    asyncio.run(scan_project())
"""

# CVE/Vulnerability scanning (existing functionality)
from victor.security.protocol import (
    CVE,
    CVSSMetrics,
    Dependency,
    SecurityPolicy,
    SecurityScanResult,
    Severity,
    Vulnerability,
    VulnerabilityStatus,
)
from victor.security.cve_database import (
    BaseCVEDatabase,
    CachingCVEDatabase,
    CVEDatabase,
    LocalCVECache,
    OfflineCVEDatabase,
    OSVDatabase,
    get_cve_database,
)
from victor.security.scanner import (
    BaseDependencyParser,
    DependencyParser,
    GoDependencyParser,
    NodeDependencyParser,
    PythonDependencyParser,
    RustDependencyParser,
    SecurityScanner,
    get_scanner,
)
from victor.security.manager import (
    SecurityManager,
    get_security_manager,
    reset_security_manager,
)

# Submodule imports for convenience (not re-exporting all, just exposing submodules)
from victor.security import auth
from victor.security import safety
from victor.security import audit

__all__ = [
    # Submodules
    "auth",
    "safety",
    "audit",
    # Protocol types
    "CVE",
    "CVSSMetrics",
    "Dependency",
    "SecurityPolicy",
    "SecurityScanResult",
    "Severity",
    "Vulnerability",
    "VulnerabilityStatus",
    # CVE database
    "BaseCVEDatabase",
    "CachingCVEDatabase",
    "CVEDatabase",
    "LocalCVECache",
    "OfflineCVEDatabase",
    "OSVDatabase",
    "get_cve_database",
    # Scanners
    "BaseDependencyParser",
    "DependencyParser",
    "GoDependencyParser",
    "NodeDependencyParser",
    "PythonDependencyParser",
    "RustDependencyParser",
    "SecurityScanner",
    "get_scanner",
    # Manager
    "SecurityManager",
    "get_security_manager",
    "reset_security_manager",
]
