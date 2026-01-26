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
"""

from victor.security.safety.infrastructure import (
    CLOUD_PATTERNS,
    DESTRUCTIVE_PATTERNS,
    DOCKER_PATTERNS,
    InfraPatternCategory,
    InfraScanResult,
    InfrastructureScanner,
    KUBERNETES_PATTERNS,
    TERRAFORM_PATTERNS,
    get_all_infrastructure_patterns,
    get_safety_reminders,
    scan_infrastructure_command,
    validate_dockerfile,
    validate_kubernetes_manifest,
)

__all__ = [
    "CLOUD_PATTERNS",
    "DESTRUCTIVE_PATTERNS",
    "DOCKER_PATTERNS",
    "InfraPatternCategory",
    "InfraScanResult",
    "InfrastructureScanner",
    "KUBERNETES_PATTERNS",
    "TERRAFORM_PATTERNS",
    "get_all_infrastructure_patterns",
    "get_safety_reminders",
    "scan_infrastructure_command",
    "validate_dockerfile",
    "validate_kubernetes_manifest",
]
