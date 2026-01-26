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

"""Penetration Testing Suite for Victor AI Agent.

.. deprecated:: 0.6.0
    This module is deprecated. Please migrate to ``victor.security_analysis.tools.penetration_testing``.
    This module will be removed in v1.0.0.

Migration Guide:
    Old (deprecated):
        from victor.security.penetration_testing import SecurityTestSuite

    New (recommended):
        from victor.security_analysis.tools import SecurityTestSuite
        # or
        from victor.security_analysis.tools.penetration_testing import SecurityTestSuite

This module provides comprehensive security testing capabilities for detecting
vulnerabilities in AI agent systems.
"""

import warnings

warnings.warn(
    "victor.security.penetration_testing is deprecated and will be removed in v1.0.0. "
    "Use victor.security_analysis.tools.penetration_testing instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical location
from victor.security_analysis.tools.penetration_testing import (
    SecurityTestSuite,
    SeverityLevel,
    AttackType,
    ExploitPattern,
    Vulnerability,
    SecurityVulnerability,
    SecurityReport,
    SecurityAuditReport,
    ComprehensiveSecurityReport,
    run_security_tests,
)

__all__ = [
    "SecurityTestSuite",
    "SeverityLevel",
    "AttackType",
    "ExploitPattern",
    "Vulnerability",
    "SecurityVulnerability",
    "SecurityReport",
    "SecurityAuditReport",
    "ComprehensiveSecurityReport",
    "run_security_tests",
]
