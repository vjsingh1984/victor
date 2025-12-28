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

"""Security scanning and CVE database integration.

This module provides comprehensive security scanning capabilities
including vulnerability detection, CVE database integration, and
security reporting.

Example usage:
    from victor_coding.security import get_security_manager, SecurityPolicy
    from pathlib import Path
    import asyncio

    async def scan_project():
        # Get manager with custom policy
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

        # Get fix commands
        fixes = manager.get_fix_commands()
        for ecosystem, commands in fixes.items():
            print(f"\\n{ecosystem} fixes:")
            for cmd in commands:
                print(f"  {cmd}")

    asyncio.run(scan_project())
"""

from victor_coding.security.protocol import (
    CVE,
    CVSSMetrics,
    Dependency,
    SecurityPolicy,
    SecurityScanResult,
    Severity,
    Vulnerability,
    VulnerabilityStatus,
)
from victor_coding.security.cve_database import (
    BaseCVEDatabase,
    CachingCVEDatabase,
    CVEDatabase,
    LocalCVECache,
    OfflineCVEDatabase,
    OSVDatabase,
    get_cve_database,
)
from victor_coding.security.scanner import (
    BaseDependencyParser,
    DependencyParser,
    GoDependencyParser,
    NodeDependencyParser,
    PythonDependencyParser,
    RustDependencyParser,
    SecurityScanner,
    get_scanner,
)
from victor_coding.security.manager import (
    SecurityManager,
    get_security_manager,
    reset_security_manager,
)

__all__ = [
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
