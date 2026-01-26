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

"""Infrastructure safety patterns.

This is the canonical location for infrastructure safety pattern utilities.
The old location (victor.security.safety.infrastructure) is deprecated.
"""

# Import from local implementation (canonical location)
from victor.security_analysis.patterns.infrastructure_impl import (
    InfraPatternCategory,
    RiskLevel,
    DESTRUCTIVE_PATTERNS,
    KUBERNETES_PATTERNS,
    DOCKER_PATTERNS,
    TERRAFORM_PATTERNS,
    CLOUD_PATTERNS,
    InfraScanResult,
    InfrastructureScanner,
    scan_infrastructure_command,
    validate_dockerfile,
    validate_kubernetes_manifest,
    get_all_infrastructure_patterns,
    get_safety_reminders,
)

__all__ = [
    # Enums
    "InfraPatternCategory",
    "RiskLevel",
    # Pattern lists
    "DESTRUCTIVE_PATTERNS",
    "KUBERNETES_PATTERNS",
    "DOCKER_PATTERNS",
    "TERRAFORM_PATTERNS",
    "CLOUD_PATTERNS",
    # Classes
    "InfraScanResult",
    "InfrastructureScanner",
    # Functions
    "scan_infrastructure_command",
    "validate_dockerfile",
    "validate_kubernetes_manifest",
    "get_all_infrastructure_patterns",
    "get_safety_reminders",
]
