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

"""Safety utilities for detecting secrets, PII, and dangerous patterns.

This module has moved to victor.security.safety.
Please update your imports to use the new location.

This stub provides backward compatibility.
"""

# Re-export from new location for backward compatibility

# Core types
from victor.security.safety.types import SafetyPattern

# Secret detection
from victor.security.safety.secrets import (
    # Types
    SecretSeverity,
    SecretMatch,
    # Patterns
    CREDENTIAL_PATTERNS,
    # Scanner
    SecretScanner,
    # Functions
    detect_secrets,
    has_secrets,
    get_secret_types,
    mask_secrets,
)

# PII detection
from victor.security.safety.pii import (
    # Types
    PIIType,
    PIISeverity,
    PIIMatch,
    # Patterns
    PII_COLUMN_PATTERNS,
    PII_CONTENT_PATTERNS,
    PII_SEVERITY,
    ANONYMIZATION_SUGGESTIONS,
    # Scanner
    PIIScanner,
    # Functions
    detect_pii_columns,
    detect_pii_in_content,
    get_anonymization_suggestion,
    get_pii_severity,
    has_pii,
    get_pii_types,
    get_safety_reminders,
)

# Code patterns (git, package managers, refactoring)
from victor.security.safety.code_patterns import (
    # Enums
    CodePatternCategory,
    # Pattern lists
    GIT_PATTERNS,
    REFACTORING_PATTERNS,
    PACKAGE_MANAGER_PATTERNS,
    BUILD_DEPLOY_PATTERNS,
    SENSITIVE_FILE_PATTERNS,
    # Classes
    ScanResult,
    CodePatternScanner,
    # Functions
    scan_command,
    is_sensitive_file,
    get_all_patterns,
)

# Infrastructure patterns (Kubernetes, Docker, Terraform)
from victor.security.safety.infrastructure import (
    # Enums
    InfraPatternCategory,
    # Pattern lists
    DESTRUCTIVE_PATTERNS,
    KUBERNETES_PATTERNS,
    DOCKER_PATTERNS,
    TERRAFORM_PATTERNS,
    CLOUD_PATTERNS,
    # Classes
    InfraScanResult,
    InfrastructureScanner,
    # Functions
    scan_infrastructure_command,
    validate_dockerfile,
    validate_kubernetes_manifest,
    get_all_infrastructure_patterns,
    get_safety_reminders as get_infrastructure_reminders,
)

# Registry for unified scanner management
from victor.security.safety.registry import (
    ISafetyScanner,
    SafetyRegistry,
)

__all__ = [
    # Core types
    "SafetyPattern",
    # Secret detection - Types
    "SecretSeverity",
    "SecretMatch",
    # Secret detection - Patterns
    "CREDENTIAL_PATTERNS",
    # Secret detection - Scanner
    "SecretScanner",
    # Secret detection - Functions
    "detect_secrets",
    "has_secrets",
    "get_secret_types",
    "mask_secrets",
    # PII detection - Types
    "PIIType",
    "PIISeverity",
    "PIIMatch",
    # PII detection - Patterns
    "PII_COLUMN_PATTERNS",
    "PII_CONTENT_PATTERNS",
    "PII_SEVERITY",
    "ANONYMIZATION_SUGGESTIONS",
    # PII detection - Scanner
    "PIIScanner",
    # PII detection - Functions
    "detect_pii_columns",
    "detect_pii_in_content",
    "get_anonymization_suggestion",
    "get_pii_severity",
    "has_pii",
    "get_pii_types",
    "get_safety_reminders",
    # Code patterns - Enums
    "CodePatternCategory",
    # Code patterns - Pattern lists
    "GIT_PATTERNS",
    "REFACTORING_PATTERNS",
    "PACKAGE_MANAGER_PATTERNS",
    "BUILD_DEPLOY_PATTERNS",
    "SENSITIVE_FILE_PATTERNS",
    # Code patterns - Classes
    "ScanResult",
    "CodePatternScanner",
    # Code patterns - Functions
    "scan_command",
    "is_sensitive_file",
    "get_all_patterns",
    # Infrastructure patterns - Enums
    "InfraPatternCategory",
    # Infrastructure patterns - Pattern lists
    "DESTRUCTIVE_PATTERNS",
    "KUBERNETES_PATTERNS",
    "DOCKER_PATTERNS",
    "TERRAFORM_PATTERNS",
    "CLOUD_PATTERNS",
    # Infrastructure patterns - Classes
    "InfraScanResult",
    "InfrastructureScanner",
    # Infrastructure patterns - Functions
    "scan_infrastructure_command",
    "validate_dockerfile",
    "validate_kubernetes_manifest",
    "get_all_infrastructure_patterns",
    "get_infrastructure_reminders",
    # Registry
    "ISafetyScanner",
    "SafetyRegistry",
]
