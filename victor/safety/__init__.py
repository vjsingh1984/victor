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
- Safety pattern matching for dangerous operations

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
"""

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

__all__ = [
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
]
