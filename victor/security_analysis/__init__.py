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

"""Security Analysis Vertical Package.

Victor's vertical for security analysis, vulnerability scanning, and
security testing. This vertical provides:
- Vulnerability scanning and CVE database integration
- Dependency security analysis
- Penetration testing tools

Note: Security patterns (secrets, PII, code safety) are now in
victor.core.security.patterns as cross-cutting framework services.
Use: from victor.core.security.patterns import detect_secrets, PIIScanner

This package separates security ANALYSIS (domain-specific orchestration)
from security INFRASTRUCTURE (RBAC, audit, patterns), which is located in
victor.core.security.

Package Structure:
    assistant.py        - SecurityAnalysisAssistant vertical class
    middleware.py       - Security analysis middleware
    safety.py           - Security-specific safety patterns
    prompts.py          - Security task hints and prompt contributions
    handlers.py         - Workflow handlers
    service_provider.py - DI service registration
    tools/              - Security scanning tools (scanner, CVE database)

Usage:
    from victor.security_analysis import SecurityAnalysisAssistant

    # Get vertical configuration
    config = SecurityAnalysisAssistant.get_config()

    # Get extensions for framework integration
    extensions = SecurityAnalysisAssistant.get_extensions()

    # Use security tools
    from victor.security_analysis.tools import SecurityScanner, CVEDatabase

    # Use security patterns (from core)
    from victor.core.security.patterns import detect_secrets, PIIScanner
"""

from victor.security_analysis.assistant import SecurityAnalysisAssistant
from victor.security_analysis.middleware import SecurityAnalysisMiddleware
from victor.security_analysis.safety import SecurityAnalysisSafetyExtension
from victor.security_analysis.prompts import SecurityAnalysisPromptContributor
from victor.security_analysis.service_provider import SecurityAnalysisServiceProvider

# Import tools
from victor.security_analysis.tools import (
    SecurityScanner,
    get_scanner,
    SecurityManager,
    get_security_manager,
)

# Import patterns from core (cross-cutting framework services)
from victor.core.security.patterns import (
    # Types
    SafetyPattern,
    # Registry
    ISafetyScanner,
    SafetyRegistry,
    # Secrets
    SecretScanner,
    detect_secrets,
    # PII
    PIIScanner,
    detect_pii_columns,
    detect_pii_in_content,
    # Code patterns
    CodePatternScanner,
    # Infrastructure patterns
    InfrastructureScanner,
    # Source credibility (use correct name from impl file)
    SourceCredibilityScanner,
    # Content patterns
    ContentPatternScanner,
)

__all__ = [
    # Main vertical
    "SecurityAnalysisAssistant",
    # Extensions
    "SecurityAnalysisMiddleware",
    "SecurityAnalysisSafetyExtension",
    "SecurityAnalysisPromptContributor",
    "SecurityAnalysisServiceProvider",
    # Tools
    "SecurityScanner",
    "get_scanner",
    "SecurityManager",
    "get_security_manager",
    # Patterns - Types
    "SafetyPattern",
    # Patterns - Registry
    "ISafetyScanner",
    "SafetyRegistry",
    # Patterns - Secrets
    "SecretScanner",
    "detect_secrets",
    # Patterns - PII
    "PIIScanner",
    "detect_pii_columns",
    "detect_pii_in_content",
    # Patterns - Code
    "CodePatternScanner",
    # Patterns - Infrastructure
    "InfrastructureScanner",
    # Patterns - Source credibility
    "SourceCredibilityScanner",
    # Patterns - Content
    "ContentPatternScanner",
]
