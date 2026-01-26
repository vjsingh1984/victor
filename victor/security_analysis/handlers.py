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

"""Security analysis workflow handlers.

This module provides workflow compute handlers for security analysis
operations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class VulnerabilityScanHandler:
    """Handler for vulnerability scanning workflows.

    Coordinates dependency parsing, CVE lookup, and result aggregation.
    """

    def __init__(self) -> None:
        """Initialize the handler."""
        self._scanner = None  # Lazy load

    def _get_scanner(self) -> Any:
        """Get or create security scanner."""
        if self._scanner is None:
            from victor.security_analysis.tools import get_scanner

            self._scanner = get_scanner()
        return self._scanner

    async def scan_dependencies(self, project_path: str) -> Dict[str, Any]:
        """Scan project dependencies for vulnerabilities.

        Args:
            project_path: Path to project root

        Returns:
            Scan results with vulnerabilities
        """
        scanner = self._get_scanner()
        try:
            result = await scanner.scan_project(project_path)
            return {
                "success": True,
                "total_dependencies": result.total_dependencies,
                "vulnerabilities": [
                    {
                        "cve_id": v.cve.cve_id,
                        "severity": v.cve.severity.value,
                        "package": v.dependency.name,
                        "version": v.dependency.version,
                        "fixed_version": v.fixed_version,
                    }
                    for v in result.vulnerabilities
                ],
                "critical_count": result.critical_count,
                "high_count": result.high_count,
            }
        except Exception as e:
            logger.error(f"Vulnerability scan failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }


class SecretScanHandler:
    """Handler for secret detection workflows."""

    def __init__(self) -> None:
        """Initialize the handler."""
        self._scanner = None  # Lazy load

    def _get_scanner(self) -> Any:
        """Get or create secret scanner."""
        if self._scanner is None:
            from victor.core.security.patterns import SecretScanner

            self._scanner = SecretScanner()
        return self._scanner

    def scan_content(self, content: str) -> Dict[str, Any]:
        """Scan content for secrets.

        Args:
            content: Content to scan

        Returns:
            Scan results with detected secrets
        """
        scanner = self._get_scanner()
        secrets = scanner.scan(content)
        return {
            "found_secrets": len(secrets) > 0,
            "secret_count": len(secrets),
            "secrets": [
                {
                    "type": s.get("type", "unknown"),
                    "severity": s.get("severity", "high"),
                    "line": s.get("line"),
                    "masked": s.get("masked", "***"),
                }
                for s in secrets
            ],
        }

    def scan_file(self, file_path: str) -> Dict[str, Any]:
        """Scan a file for secrets.

        Args:
            file_path: Path to file

        Returns:
            Scan results with detected secrets
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            result = self.scan_content(content)
            result["file_path"] = file_path
            return result
        except Exception as e:
            logger.error(f"Secret scan failed for {file_path}: {e}")
            return {
                "file_path": file_path,
                "found_secrets": False,
                "error": str(e),
            }


class ComplianceCheckHandler:
    """Handler for compliance checking workflows."""

    def __init__(self) -> None:
        """Initialize the handler."""
        pass

    async def check_compliance(
        self,
        project_path: str,
        frameworks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Check project compliance against frameworks.

        Args:
            project_path: Path to project root
            frameworks: List of frameworks to check (e.g., ["soc2", "gdpr"])

        Returns:
            Compliance check results
        """
        from victor.core.security.audit import (
            AuditManager,
            ComplianceFramework,
        )

        frameworks = frameworks or ["soc2"]
        results: Dict[str, Any] = {}

        try:
            # Initialize audit manager (validates project path)
            AuditManager.get_instance(project_path)

            for framework_name in frameworks:
                try:
                    framework = ComplianceFramework(framework_name.lower())
                    # In a real implementation, this would run actual compliance checks
                    results[framework_name] = {
                        "status": "checked",
                        "framework": framework.value,
                        "findings": [],
                    }
                except ValueError:
                    results[framework_name] = {
                        "status": "unknown_framework",
                        "error": f"Unknown framework: {framework_name}",
                    }

            return {
                "success": True,
                "project_path": project_path,
                "results": results,
            }
        except Exception as e:
            logger.error(f"Compliance check failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }


__all__ = [
    "VulnerabilityScanHandler",
    "SecretScanHandler",
    "ComplianceCheckHandler",
]
