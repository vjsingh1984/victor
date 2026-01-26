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

"""Security scanning protocol types.

Defines data structures for vulnerability scanning, CVE tracking,
and security reporting.

This is the canonical location for CVE/vulnerability type definitions.
These types are used by both the security infrastructure and security
analysis tools.
"""

# Re-export from the original location
from victor.security.protocol import (
    CVESeverity,
    Severity,
    VulnerabilityStatus,
    CVSSMetrics,
    CVE,
    SecurityDependency,
    Dependency,
    Vulnerability,
    SecurityScanResult,
    SecurityPolicy,
)

__all__ = [
    "CVESeverity",
    "Severity",  # Backward compatibility alias
    "VulnerabilityStatus",
    "CVSSMetrics",
    "CVE",
    "SecurityDependency",
    "Dependency",  # Backward compatibility alias
    "Vulnerability",
    "SecurityScanResult",
    "SecurityPolicy",
]
