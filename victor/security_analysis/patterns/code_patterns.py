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

"""Code safety patterns.

This is the canonical location for code safety pattern utilities.
The old location (victor.security.safety.code_patterns) is deprecated.
"""

# Import from local implementation (canonical location)
from victor.security_analysis.patterns.code_patterns_impl import (
    CodePatternCategory,
    CodePatternScanner,
    RiskLevel,
    GIT_PATTERNS,
    REFACTORING_PATTERNS,
    PACKAGE_MANAGER_PATTERNS,
    BUILD_DEPLOY_PATTERNS,
    SENSITIVE_FILE_PATTERNS,
    ScanResult,
    scan_command,
    is_sensitive_file,
    get_all_patterns,
)

__all__ = [
    # Enums
    "CodePatternCategory",
    "RiskLevel",
    # Pattern lists
    "GIT_PATTERNS",
    "REFACTORING_PATTERNS",
    "PACKAGE_MANAGER_PATTERNS",
    "BUILD_DEPLOY_PATTERNS",
    "SENSITIVE_FILE_PATTERNS",
    # Classes
    "ScanResult",
    "CodePatternScanner",
    # Functions
    "scan_command",
    "is_sensitive_file",
    "get_all_patterns",
]
