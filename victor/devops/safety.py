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

"""DevOps Safety Extension - Security patterns for infrastructure work.

This module provides DevOps-specific safety patterns for infrastructure
operations including Kubernetes, Docker, Terraform, and cloud providers.

This module now delegates to the core safety infrastructure at
victor.security.safety.infrastructure for pattern scanning, while maintaining
backward compatibility for existing interfaces.
"""

from typing import Dict, List, Tuple

from victor.security.safety.infrastructure import (
    InfrastructureScanner,
    InfraScanResult,
    DESTRUCTIVE_PATTERNS,
    KUBERNETES_PATTERNS,
    DOCKER_PATTERNS,
    TERRAFORM_PATTERNS,
    CLOUD_PATTERNS,
    validate_dockerfile as core_validate_dockerfile,
    validate_kubernetes_manifest as core_validate_kubernetes_manifest,
    get_safety_reminders as core_get_safety_reminders,
)
from victor.security.safety.secrets import CREDENTIAL_PATTERNS, SecretScanner
from victor.core.verticals.protocols import SafetyExtensionProtocol, SafetyPattern


# Risk levels (kept for backward compatibility)
HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"


class DevOpsSafetyExtension(SafetyExtensionProtocol):
    """Safety extension for DevOps tasks.

    Provides DevOps-specific dangerous operation patterns including
    Kubernetes, Docker, Terraform, and cloud provider operations.

    This class delegates to the core InfrastructureScanner for pattern
    matching while providing the SafetyExtensionProtocol interface.
    """

    def __init__(
        self,
        include_destructive: bool = True,
        include_kubernetes: bool = True,
        include_docker: bool = True,
        include_terraform: bool = True,
        include_cloud: bool = True,
    ):
        """Initialize the safety extension.

        Args:
            include_destructive: Include destructive patterns
            include_kubernetes: Include Kubernetes patterns
            include_docker: Include Docker patterns
            include_terraform: Include Terraform patterns
            include_cloud: Include cloud provider patterns
        """
        self._include_destructive = include_destructive
        self._include_kubernetes = include_kubernetes
        self._include_docker = include_docker
        self._include_terraform = include_terraform
        self._include_cloud = include_cloud

        # Create an InfrastructureScanner with matching configuration
        self._scanner = InfrastructureScanner(
            include_destructive=include_destructive,
            include_kubernetes=include_kubernetes,
            include_docker=include_docker,
            include_terraform=include_terraform,
            include_cloud=include_cloud,
        )
        self._secret_scanner = SecretScanner()

    def get_bash_patterns(self) -> List[SafetyPattern]:
        """Return DevOps-specific bash patterns.

        Returns:
            List of SafetyPattern for dangerous bash commands.
        """
        return self._scanner.all_patterns

    def get_danger_patterns(self) -> List[Tuple[str, str, str]]:
        """Return DevOps-specific danger patterns (legacy format).

        Returns:
            List of (regex_pattern, description, risk_level) tuples.
        """
        return [(p.pattern, p.description, p.risk_level) for p in self._scanner.all_patterns]

    def get_blocked_operations(self) -> List[str]:
        """Return operations that should be blocked in DevOps context."""
        return [
            "delete_production_database",
            "destroy_production_infrastructure",
            "expose_secrets_to_logs",
            "disable_security_features",
            "create_public_s3_bucket",
        ]

    def get_credential_patterns(self) -> Dict[str, str]:
        """Return patterns for detecting credentials.

        Uses patterns from victor.security.safety.secrets for comprehensive detection.

        Returns:
            Dict of credential_type -> regex_pattern.
        """
        # Return simplified dict format for backward compatibility
        return {name: pattern for name, (pattern, _, _) in CREDENTIAL_PATTERNS.items()}

    def scan_for_secrets(self, content: str) -> List[Dict]:
        """Scan content for secrets using the core SecretScanner.

        Args:
            content: Text content to scan

        Returns:
            List of secret match dictionaries
        """
        matches = self._secret_scanner.scan(content)
        return [
            {
                "type": m.secret_type,
                "severity": m.severity.value,
                "line": m.line_number,
                "suggestion": m.suggestion,
            }
            for m in matches
        ]

    def scan_command(self, command: str) -> InfraScanResult:
        """Scan a command for dangerous patterns.

        Args:
            command: The command to scan

        Returns:
            InfraScanResult with matched patterns
        """
        return self._scanner.scan_command(command)

    def validate_dockerfile(self, content: str) -> List[str]:
        """Validate Dockerfile security best practices.

        Returns:
            List of security warnings found.
        """
        return core_validate_dockerfile(content)

    def validate_kubernetes_manifest(self, content: str) -> List[str]:
        """Validate Kubernetes manifest security.

        Returns:
            List of security warnings found.
        """
        return core_validate_kubernetes_manifest(content)

    def get_safety_reminders(self) -> List[str]:
        """Return safety reminders for DevOps output."""
        return core_get_safety_reminders()

    def get_category(self) -> str:
        """Get the category name for these patterns.

        Returns:
            Category identifier
        """
        return "devops"


__all__ = [
    "DevOpsSafetyExtension",
    # Re-exported from core for convenience
    "InfrastructureScanner",
    "InfraScanResult",
    "DESTRUCTIVE_PATTERNS",
    "KUBERNETES_PATTERNS",
    "DOCKER_PATTERNS",
    "TERRAFORM_PATTERNS",
    "CLOUD_PATTERNS",
    # Legacy constants
    "HIGH",
    "MEDIUM",
    "LOW",
]
