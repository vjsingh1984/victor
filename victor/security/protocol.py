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

.. deprecated:: 0.6.0
    This module is deprecated. Please migrate to ``victor.core.security.protocol``.
    This module will be removed in v1.0.0.

Migration Guide:
    Old (deprecated):
        from victor.security.protocol import CVE, CVESeverity, SecurityScanResult

    New (recommended):
        from victor.core.security.protocol import CVE, CVESeverity, SecurityScanResult
        # or
        from victor.core.security import CVE, CVESeverity, SecurityScanResult

Defines data structures for vulnerability scanning, CVE tracking,
and security reporting.
"""

import warnings

warnings.warn(
    "victor.security.protocol is deprecated and will be removed in v1.0.0. "
    "Use victor.core.security.protocol instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical location
from victor.core.security.protocol import (
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
