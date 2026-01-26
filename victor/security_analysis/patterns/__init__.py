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

"""Safety patterns and utilities for Victor.

This is the canonical location for safety pattern utilities.
The old location (victor.security.safety) is deprecated.

Migration Guide:
    Old (deprecated):
        from victor.security.safety import detect_secrets, SafetyPattern

    New (recommended):
        from victor.security_analysis.patterns import detect_secrets, SafetyPattern

This module provides consolidated safety utilities for:
- Secret detection (API keys, credentials, tokens)
- PII detection (email, SSN, credit cards, etc.)
- Code safety patterns (git, refactoring, package management)
- Infrastructure safety patterns (Kubernetes, Docker, Terraform)
- Unified safety registry for pluggable scanners
"""

# Import from local submodules (canonical location)
from victor.security_analysis.patterns.types import SafetyPattern
from victor.security_analysis.patterns.registry import (
    ISafetyScanner,
    SafetyRegistry,
)
from victor.security_analysis.patterns.secrets import (
    CREDENTIAL_PATTERNS,
    SecretMatch,
    SecretScanner,
    SecretSeverity,
    detect_secrets,
)
from victor.security_analysis.patterns.pii import (
    ANONYMIZATION_SUGGESTIONS,
    PII_COLUMN_PATTERNS,
    PII_CONTENT_PATTERNS,
    PII_SEVERITY,
    PIIMatch,
    PIIScanner,
    PIISeverity,
    PIIType,
    detect_pii_columns,
    detect_pii_in_content,
    get_anonymization_suggestion,
    get_pii_severity,
    get_pii_types,
    get_safety_reminders,
    has_pii,
)
from victor.security_analysis.patterns.code_patterns import (
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
from victor.security_analysis.patterns.infrastructure import (
    InfraPatternCategory,
    RiskLevel as InfraRiskLevel,
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
    get_safety_reminders as get_infrastructure_safety_reminders,
)
from victor.security_analysis.patterns.source_credibility import (
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
from victor.security_analysis.patterns.content_patterns import (
    ContentWarningLevel,
    ContentWarningMatch,
    CONTENT_WARNING_PATTERNS,
    MISINFORMATION_RISK_PATTERNS,
    ADVICE_RISK_PATTERNS,
    scan_content_warnings,
    has_content_warnings,
    get_high_severity_warnings,
    detect_misinformation_risk,
    detect_advice_risk,
    get_content_safety_reminders,
    ContentPatternScanner,
)

__all__ = [
    # Types
    "SafetyPattern",
    # Registry
    "ISafetyScanner",
    "SafetyRegistry",
    # Secrets
    "CREDENTIAL_PATTERNS",
    "SecretMatch",
    "SecretScanner",
    "SecretSeverity",
    "detect_secrets",
    # PII
    "ANONYMIZATION_SUGGESTIONS",
    "PII_COLUMN_PATTERNS",
    "PII_CONTENT_PATTERNS",
    "PII_SEVERITY",
    "PIIMatch",
    "PIIScanner",
    "PIISeverity",
    "PIIType",
    "detect_pii_columns",
    "detect_pii_in_content",
    "get_anonymization_suggestion",
    "get_pii_severity",
    "get_pii_types",
    "get_safety_reminders",
    "has_pii",
    # Code patterns
    "CodePatternCategory",
    "CodePatternScanner",
    "RiskLevel",
    "GIT_PATTERNS",
    "REFACTORING_PATTERNS",
    "PACKAGE_MANAGER_PATTERNS",
    "BUILD_DEPLOY_PATTERNS",
    "SENSITIVE_FILE_PATTERNS",
    "ScanResult",
    "scan_command",
    "is_sensitive_file",
    "get_all_patterns",
    # Infrastructure patterns
    "InfraPatternCategory",
    "InfraRiskLevel",
    "DESTRUCTIVE_PATTERNS",
    "KUBERNETES_PATTERNS",
    "DOCKER_PATTERNS",
    "TERRAFORM_PATTERNS",
    "CLOUD_PATTERNS",
    "InfraScanResult",
    "InfrastructureScanner",
    "scan_infrastructure_command",
    "validate_dockerfile",
    "validate_kubernetes_manifest",
    "get_all_infrastructure_patterns",
    "get_infrastructure_safety_reminders",
    # Source credibility
    "CredibilityLevel",
    "CredibilityMatch",
    "SOURCE_CREDIBILITY_PATTERNS",
    "SourceCredibilityScanner",
    "validate_source_credibility",
    "get_credibility_level",
    "is_high_credibility",
    "is_low_credibility",
    "get_source_safety_reminders",
    # Content patterns
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
