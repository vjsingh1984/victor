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

"""Content safety patterns.

This is the canonical location for content safety pattern utilities.
The old location (victor.security.safety.content_patterns) is deprecated.
"""

# Import from local implementation (canonical location)
from victor.core.security.patterns.content_patterns_impl import (
    ContentWarningLevel,
    ContentWarningMatch,
    ContentPatternScanner,
    CONTENT_WARNING_PATTERNS,
    ADVICE_RISK_PATTERNS,
    MISINFORMATION_RISK_PATTERNS,
    detect_advice_risk,
    detect_misinformation_risk,
    get_content_safety_reminders,
    get_high_severity_warnings,
    has_content_warnings,
    scan_content_warnings,
)

__all__ = [
    "ContentWarningLevel",
    "ContentWarningMatch",
    "CONTENT_WARNING_PATTERNS",
    "MISINFORMATION_RISK_PATTERNS",
    "ADVICE_RISK_PATTERNS",
    "scan_content_warnings",
    "has_content_warnings",
    "get_high_severity_warnings",
    "detect_misinformation_risk",
    "detect_advice_risk",
    "get_content_safety_reminders",
    "ContentPatternScanner",
]
