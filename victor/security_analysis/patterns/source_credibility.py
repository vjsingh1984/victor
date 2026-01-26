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

"""Source credibility patterns.

This is the canonical location for source credibility utilities.
The old location (victor.security.safety.source_credibility) is deprecated.
"""

# Import from local implementation (canonical location)
from victor.security_analysis.patterns.source_credibility_impl import (
    CredibilityLevel,
    CredibilityMatch,
    SOURCE_CREDIBILITY_PATTERNS,
    SourceCredibilityScanner,
    validate_source_credibility,
    get_credibility_level,
    is_high_credibility,
    is_low_credibility,
    get_source_safety_reminders,
)

__all__ = [
    "CredibilityLevel",
    "CredibilityMatch",
    "SOURCE_CREDIBILITY_PATTERNS",
    "SourceCredibilityScanner",
    "validate_source_credibility",
    "get_credibility_level",
    "is_high_credibility",
    "is_low_credibility",
    "get_source_safety_reminders",
]
