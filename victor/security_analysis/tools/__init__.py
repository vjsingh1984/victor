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

"""Security analysis tools.

This module provides security scanning tools including:
- Vulnerability scanning
- Dependency parsing
- CVE database integration
- Penetration testing utilities

These are the canonical locations for security analysis tools.
The tools in victor.security are deprecated in favor of these.
"""

# Re-export from victor.security for now
# These will be migrated to this location in the future
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

from victor.security.cve_database import (
    BaseCVEDatabase,
    CachingCVEDatabase,
    CVEDatabase,
    LocalCVECache,
    OfflineCVEDatabase,
    OSVDatabase,
    get_cve_database,
)

from victor.security.penetration_testing import (
    SecurityTestSuite,
    SeverityLevel,
    AttackType,
    SecurityReport,
    ComprehensiveSecurityReport,
    run_security_tests,
)

__all__ = [
    # Scanner
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
    # CVE Database
    "BaseCVEDatabase",
    "CachingCVEDatabase",
    "CVEDatabase",
    "LocalCVECache",
    "OfflineCVEDatabase",
    "OSVDatabase",
    "get_cve_database",
    # Penetration Testing
    "SecurityTestSuite",
    "SeverityLevel",
    "AttackType",
    "SecurityReport",
    "ComprehensiveSecurityReport",
    "run_security_tests",
]
