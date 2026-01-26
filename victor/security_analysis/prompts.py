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

"""Security analysis prompt contributions.

This module provides prompt hints and contributions specific to
security analysis tasks.
"""

from __future__ import annotations

from typing import Any, Dict, List

from victor.core.verticals.protocols import PromptContributorProtocol


class SecurityAnalysisPromptContributor(PromptContributorProtocol):
    """Prompt contributor for security analysis tasks.

    Provides context-aware prompt hints for:
    - Vulnerability scanning
    - Secret detection
    - Compliance checking
    - Security code review
    """

    def get_task_type_hints(self) -> Dict[str, str]:
        """Get hints for different security task types.

        Returns:
            Dict mapping task types to hint strings
        """
        return {
            "vulnerability_scan": """When scanning for vulnerabilities:
1. Check all dependency files (requirements.txt, package.json, go.mod, etc.)
2. Query CVE databases for known vulnerabilities
3. Prioritize by CVSS score (critical > high > medium > low)
4. Include remediation steps with specific version upgrades""",
            "secret_detection": """When detecting secrets:
1. Scan all source files, configs, and environment files
2. Check for API keys, passwords, tokens, and credentials
3. Look for hardcoded values that should be environment variables
4. Report file paths and line numbers for each finding""",
            "compliance_check": """When checking compliance:
1. Identify applicable frameworks (SOC2, GDPR, HIPAA, PCI-DSS)
2. Check data handling practices
3. Verify encryption and access controls
4. Document any gaps or violations""",
            "security_review": """When reviewing code for security:
1. Check input validation and sanitization
2. Look for injection vulnerabilities (SQL, XSS, command injection)
3. Verify authentication and authorization
4. Review error handling (no sensitive data in errors)
5. Check for secure communications (HTTPS, TLS)""",
            "penetration_test": """When conducting penetration testing:
1. Document scope and authorization
2. Start with reconnaissance (non-intrusive)
3. Progress to vulnerability scanning
4. Report all findings with severity ratings
5. Provide proof-of-concept where safe to do so""",
        }

    def get_context_hints(self, context: Dict[str, Any]) -> List[str]:
        """Get context-aware hints based on current state.

        Args:
            context: Current context dictionary

        Returns:
            List of relevant hints
        """
        hints = []

        # Add hints based on file types being analyzed
        if "file_path" in context:
            path = context["file_path"]
            if path.endswith(("requirements.txt", "package.json", "go.mod")):
                hints.append("Check dependencies for known CVEs using security databases")
            elif path.endswith((".env", ".env.example")):
                hints.append("Environment files may contain secrets - verify no real credentials")
            elif path.endswith((".yml", ".yaml")) and "docker" in path.lower():
                hints.append("Review Docker compose for security best practices")
            elif path.endswith("Dockerfile"):
                hints.append("Check for security issues: running as root, exposing secrets")

        # Add hints based on detected content
        if context.get("has_secrets"):
            hints.append("Secrets detected - recommend rotating credentials and using vault")

        if context.get("has_vulnerabilities"):
            hints.append("Vulnerabilities found - prioritize critical and high severity fixes")

        return hints

    def get_safety_reminders(self) -> List[str]:
        """Get security-specific safety reminders.

        Returns:
            List of safety reminder strings
        """
        return [
            "Never expose real credentials or secrets in reports",
            "Mask sensitive data when displaying findings",
            "Verify authorization before conducting any active testing",
            "Follow responsible disclosure practices",
            "Document all security testing activities",
        ]

    @property
    def name(self) -> str:
        """Get contributor name."""
        return "security_analysis"


__all__ = ["SecurityAnalysisPromptContributor"]
