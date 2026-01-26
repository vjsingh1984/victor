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

"""Content warning patterns.

This is the canonical location for content warning utilities.
"""

from victor.security.safety.content_patterns import (
    ADVICE_RISK_PATTERNS,
    CONTENT_WARNING_PATTERNS,
    MISINFORMATION_RISK_PATTERNS,
    ContentPatternScanner,
    ContentWarningLevel,
    ContentWarningMatch,
    detect_advice_risk,
    detect_misinformation_risk,
    get_content_safety_reminders,
    get_high_severity_warnings,
    has_content_warnings,
    scan_content_warnings,
)

__all__ = [
    "ADVICE_RISK_PATTERNS",
    "CONTENT_WARNING_PATTERNS",
    "MISINFORMATION_RISK_PATTERNS",
    "ContentPatternScanner",
    "ContentWarningLevel",
    "ContentWarningMatch",
    "detect_advice_risk",
    "detect_misinformation_risk",
    "get_content_safety_reminders",
    "get_high_severity_warnings",
    "has_content_warnings",
    "scan_content_warnings",
]
