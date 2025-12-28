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

DEPRECATED: This module has been promoted to victor-core.
Please use `from victor.security import ...` instead.

This module re-exports from victor.security for backward compatibility.
"""

import warnings

warnings.warn(
    "victor_coding.security is deprecated. Use victor.security instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from victor-core for backward compatibility
from victor.security import (
    # Protocol types
    CVE,
    CVSSMetrics,
    Dependency,
    SecurityPolicy,
    SecurityScanResult,
    Severity,
    Vulnerability,
    VulnerabilityStatus,
    # CVE database
    BaseCVEDatabase,
    CachingCVEDatabase,
    CVEDatabase,
    LocalCVECache,
    OfflineCVEDatabase,
    OSVDatabase,
    get_cve_database,
    # Scanners
    BaseDependencyParser,
    DependencyParser,
    GoDependencyParser,
    NodeDependencyParser,
    PythonDependencyParser,
    RustDependencyParser,
    SecurityScanner,
    get_scanner,
    # Manager
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
