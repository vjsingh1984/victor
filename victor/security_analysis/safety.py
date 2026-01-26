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

"""Security analysis-specific safety patterns.

This module defines dangerous operation patterns specific to
security analysis, particularly around secret handling and
security scanning operations.
"""

from __future__ import annotations

from typing import Any, Dict, List

from victor.core.security.patterns.types import SafetyPattern
from victor.core.security.patterns.secrets import SecretScanner
from victor.core.verticals.protocols import SafetyExtensionProtocol


class SecurityAnalysisSafetyExtension(SafetyExtensionProtocol):
    """Safety extension for security analysis vertical.

    Provides security-specific dangerous operation patterns including
    secret handling, credential scanning, and security tool operations.
    """

    def __init__(self) -> None:
        """Initialize the safety extension."""
        self._secret_scanner = SecretScanner()
        self._custom_patterns: List[SafetyPattern] = []

    def add_dangerous_pattern(self, pattern: str) -> None:
        """Add a custom dangerous pattern.

        Args:
            pattern: Regex pattern to add as dangerous
        """
        custom_pattern = SafetyPattern(
            pattern=pattern,
            description="Custom security pattern",
            risk_level="HIGH",
            category="custom",
        )
        self._custom_patterns.append(custom_pattern)

    def get_bash_patterns(self) -> List[SafetyPattern]:
        """Get security-specific bash command patterns.

        Returns:
            List of safety patterns for dangerous bash commands
        """
        # Security-specific dangerous patterns
        security_patterns = [
            SafetyPattern(
                pattern=r"curl.*\|.*bash",
                description="Piping curl to bash (potential code injection)",
                risk_level="CRITICAL",
                category="remote_execution",
            ),
            SafetyPattern(
                pattern=r"wget.*-O.*\|.*sh",
                description="Piping wget output to shell",
                risk_level="CRITICAL",
                category="remote_execution",
            ),
            SafetyPattern(
                pattern=r"echo.*\$\{?[A-Z_]+KEY[A-Z_]*\}?",
                description="Echoing potential API key",
                risk_level="HIGH",
                category="secret_exposure",
            ),
            SafetyPattern(
                pattern=r"cat.*\.env",
                description="Reading .env file (may contain secrets)",
                risk_level="MEDIUM",
                category="secret_exposure",
            ),
            SafetyPattern(
                pattern=r"export.*PASSWORD",
                description="Exporting password to environment",
                risk_level="HIGH",
                category="secret_exposure",
            ),
        ]
        return security_patterns + self._custom_patterns

    def get_file_patterns(self) -> List[SafetyPattern]:
        """Get security-specific file operation patterns.

        Returns:
            List of safety patterns for file operations
        """
        return [
            SafetyPattern(
                pattern=r"\.env(\.\w+)?$",
                description="Environment file (may contain secrets)",
                risk_level="HIGH",
                category="secret_file",
            ),
            SafetyPattern(
                pattern=r"credentials?\.(json|yaml|yml|xml)$",
                description="Credentials file",
                risk_level="CRITICAL",
                category="secret_file",
            ),
            SafetyPattern(
                pattern=r"\.pem$|\.key$|\.p12$|\.pfx$",
                description="Private key or certificate file",
                risk_level="CRITICAL",
                category="secret_file",
            ),
            SafetyPattern(
                pattern=r"id_rsa|id_dsa|id_ecdsa|id_ed25519",
                description="SSH private key",
                risk_level="CRITICAL",
                category="secret_file",
            ),
            SafetyPattern(
                pattern=r"\.htpasswd$",
                description="Apache password file",
                risk_level="HIGH",
                category="secret_file",
            ),
        ]

    def get_tool_restrictions(self) -> Dict[str, List[str]]:
        """Get tool-specific argument restrictions.

        Returns:
            Dict mapping tool names to restricted argument patterns
        """
        return {
            "write_file": [
                r"\.env$",  # Don't overwrite .env files
                r"credentials",  # Don't write to credential files
                r"\.pem$",  # Don't write to key files
            ],
            "execute_bash": [
                r"curl.*\|.*bash",  # Don't pipe curl to bash
            ],
        }

    def get_category(self) -> str:
        """Get the category name for these patterns.

        Returns:
            Category identifier
        """
        return "security_analysis"

    def scan_for_secrets(self, content: str) -> List[Dict[str, Any]]:
        """Scan content for potential secrets.

        Args:
            content: Content to scan

        Returns:
            List of detected secrets with metadata
        """
        secrets = self._secret_scanner.scan(content)
        # Convert SecretMatch objects to dicts for compatibility
        return [  # type: ignore[return-value]
            {
                "secret_type": secret.secret_type,
                "matched_text": secret.matched_text,
                "severity": secret.severity.value,
                "start": secret.start,
                "end": secret.end,
            }
            for secret in secrets
        ]


__all__ = ["SecurityAnalysisSafetyExtension"]
