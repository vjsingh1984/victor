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

"""Security analysis service provider.

This module provides dependency injection registration for security
analysis services.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from victor.core.container import ServiceLifetime

if TYPE_CHECKING:
    from victor.core.container import ServiceContainer
    from victor.config.settings import Settings
    from victor.security_analysis.tools import SecurityScanner, SecurityManager
    from victor.core.security.patterns import SecretScanner
    from victor.security_analysis.handlers import (
        VulnerabilityScanHandler,
        SecretScanHandler,
        ComplianceCheckHandler,
    )

from victor.core.verticals.protocols import ServiceProviderProtocol

logger = logging.getLogger(__name__)


class SecurityAnalysisServiceProvider(ServiceProviderProtocol):
    """Service provider for security analysis vertical.

    Registers security-specific services with the DI container:
    - SecurityScanner: Vulnerability scanning
    - SecretScanner: Secret detection
    - SecurityManager: High-level security orchestration
    """

    def register_services(self, container: "ServiceContainer", settings: "Settings") -> None:
        """Register security analysis services.

        Args:
            container: DI container to register services with
        """
        # Import classes needed for registration
        from victor.security_analysis.tools import SecurityScanner, SecurityManager
        from victor.core.security.patterns import SecretScanner
        from victor.security_analysis.handlers import (
            VulnerabilityScanHandler,
            SecretScanHandler,
            ComplianceCheckHandler,
        )

        # Register security scanner as singleton
        container.register(
            SecurityScanner,
            self._create_security_scanner,
            ServiceLifetime.SINGLETON,
        )

        # Register secret scanner as singleton
        container.register(
            SecretScanner,
            self._create_secret_scanner,
            ServiceLifetime.SINGLETON,
        )

        # Register security manager as singleton
        container.register(
            SecurityManager,
            self._create_security_manager,
            ServiceLifetime.SINGLETON,
        )

        # Register workflow handlers
        container.register(
            VulnerabilityScanHandler,
            self._create_vulnerability_handler,
            ServiceLifetime.TRANSIENT,
        )

        container.register(
            SecretScanHandler,
            self._create_secret_handler,
            ServiceLifetime.TRANSIENT,
        )

        container.register(
            ComplianceCheckHandler,
            self._create_compliance_handler,
            ServiceLifetime.TRANSIENT,
        )

        logger.debug("Registered security analysis services")

    def _create_security_scanner(self, container: ServiceContainer) -> "SecurityScanner":
        """Create security scanner instance."""
        from victor.security_analysis.tools import get_scanner

        return get_scanner()

    def _create_secret_scanner(self, container: ServiceContainer) -> "SecretScanner":
        """Create secret scanner instance."""
        from victor.core.security.patterns import SecretScanner

        return SecretScanner()

    def _create_security_manager(self, container: ServiceContainer) -> "SecurityManager":
        """Create security manager instance."""
        from victor.security_analysis.tools import get_security_manager

        return get_security_manager()

    def _create_vulnerability_handler(
        self, container: ServiceContainer
    ) -> "VulnerabilityScanHandler":
        """Create vulnerability scan handler."""
        from victor.security_analysis.handlers import VulnerabilityScanHandler

        return VulnerabilityScanHandler()

    def _create_secret_handler(self, container: ServiceContainer) -> "SecretScanHandler":
        """Create secret scan handler."""
        from victor.security_analysis.handlers import SecretScanHandler

        return SecretScanHandler()

    def _create_compliance_handler(self, container: ServiceContainer) -> "ComplianceCheckHandler":
        """Create compliance check handler."""
        from victor.security_analysis.handlers import ComplianceCheckHandler

        return ComplianceCheckHandler()

    def get_service_names(self) -> list[str]:
        """Get list of registered service names.

        Returns:
            List of service names this provider registers
        """
        return [
            "SecurityScanner",
            "SecretScanner",
            "SecurityManager",
            "VulnerabilityScanHandler",
            "SecretScanHandler",
            "ComplianceCheckHandler",
        ]

    @property
    def name(self) -> str:
        """Get provider name."""
        return "security_analysis"


__all__ = ["SecurityAnalysisServiceProvider"]
