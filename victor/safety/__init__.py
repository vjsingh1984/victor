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

This module provides cross-vertical safety capabilities:
- Secret and credential detection
- PII (Personally Identifiable Information) detection and anonymization
- Code safety patterns (git, package managers, refactoring)
- Infrastructure safety patterns (Kubernetes, Docker, Terraform)

Example usage:
    from victor.safety import (
        # Secrets
        detect_secrets,
        has_secrets,
        mask_secrets,
        SecretScanner,
        # PII
        detect_pii_columns,
        detect_pii_in_content,
        get_anonymization_suggestion,
        PIIScanner,
        # Code patterns
        CodePatternScanner,
        scan_command,
        is_sensitive_file,
        # Infrastructure patterns
        InfrastructureScanner,
        validate_dockerfile,
        validate_kubernetes_manifest,
    )

    # Check code for hardcoded secrets
    secrets = detect_secrets(code_content)
    if secrets:
        print(f"Found {len(secrets)} secrets!")
        for secret in secrets:
            print(f"  - {secret.secret_type}: {secret.suggestion}")

    # Check dataframe columns for PII
    pii_cols = detect_pii_columns(df.columns.tolist())
    for col, pii_type in pii_cols:
        suggestion = get_anonymization_suggestion(pii_type)
        print(f"  - {col}: {pii_type.value} - {suggestion}")

    # Mask secrets before logging
    safe_content = mask_secrets(sensitive_content)

    # Scan git command for dangerous patterns
    scanner = CodePatternScanner()
    result = scanner.scan_command("git push --force origin main")
    if result.has_high:
        print("Dangerous git command!")

    # Validate Kubernetes manifest
    issues = validate_kubernetes_manifest(manifest_yaml)
    for issue in issues:
        print(f"- {issue}")
"""

# Core types
from victor.safety.types import SafetyPattern

# Secret detection
from victor.safety.secrets import (
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
from victor.safety.pii import (
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
from victor.safety.code_patterns import (
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
from victor.safety.infrastructure import (
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
]
