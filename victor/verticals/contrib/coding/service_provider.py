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

"""Coding vertical service provider for DI container registration.

This module registers coding-specific services with the DI container,
enabling the framework to resolve coding vertical dependencies.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Type

from victor.core.verticals.protocols import ServiceProviderProtocol

if TYPE_CHECKING:
    from victor.core.container import ServiceContainer
    from victor.config.settings import Settings

logger = logging.getLogger(__name__)


# Protocol definitions for coding services (for type checking)
class CodingMiddlewareProtocol:
    """Protocol for coding middleware."""

    pass


class CodingSafetyProtocol:
    """Protocol for coding safety extension."""

    pass


class CodingPromptProtocol:
    """Protocol for coding prompt contributor."""

    pass


class CodingServiceProvider(ServiceProviderProtocol):
    """Service provider for coding vertical.

    Registers coding-specific services with the DI container:
    - CodeCorrectionMiddleware for code validation
    - CodingSafetyExtension for git/refactoring patterns
    - CodingPromptContributor for task hints
    - CodingModeConfigProvider for mode configurations
    """

    def register_services(
        self,
        container: "ServiceContainer",
        settings: "Settings",
    ) -> None:
        """Register coding-specific services.

        Args:
            container: DI container to register services in
            settings: Application settings
        """
        from victor.core.container import ServiceLifetime

        # Register CodeCorrectionMiddleware
        self._register_middleware(container, settings)

        # Register CodingSafetyExtension
        self._register_safety(container, settings)

        # Register CodingPromptContributor
        self._register_prompts(container, settings)

        # Register CodingModeConfigProvider
        self._register_mode_config(container, settings)

        # Register CodingToolDependencyProvider
        self._register_tool_dependencies(container, settings)

        logger.info("Registered coding vertical services")

    def _register_middleware(
        self,
        container: "ServiceContainer",
        settings: "Settings",
    ) -> None:
        """Register coding middleware services."""
        from victor.core.container import ServiceLifetime

        def create_middleware(_):
            from victor_coding.middleware import CodeCorrectionMiddleware

            enabled = getattr(settings, "enable_code_correction", True)
            auto_fix = getattr(settings, "code_correction_auto_fix", True)
            return CodeCorrectionMiddleware(enabled=enabled, auto_fix=auto_fix)

        container.register_or_replace(
            CodingMiddlewareProtocol,
            create_middleware,
            ServiceLifetime.SINGLETON,
        )

    def _register_safety(
        self,
        container: "ServiceContainer",
        settings: "Settings",
    ) -> None:
        """Register coding safety extension."""
        from victor.core.container import ServiceLifetime

        def create_safety(_):
            from victor_coding.safety import CodingSafetyExtension

            return CodingSafetyExtension()

        container.register_or_replace(
            CodingSafetyProtocol,
            create_safety,
            ServiceLifetime.SINGLETON,
        )

    def _register_prompts(
        self,
        container: "ServiceContainer",
        settings: "Settings",
    ) -> None:
        """Register coding prompt contributor."""
        from victor.core.container import ServiceLifetime

        def create_prompts(_):
            from victor_coding.prompts import CodingPromptContributor

            # Use extended grounding for local providers
            use_extended = getattr(settings, "use_extended_grounding", False)
            return CodingPromptContributor(use_extended_grounding=use_extended)

        container.register_or_replace(
            CodingPromptProtocol,
            create_prompts,
            ServiceLifetime.SINGLETON,
        )

    def _register_mode_config(
        self,
        container: "ServiceContainer",
        settings: "Settings",
    ) -> None:
        """Register mode configuration provider."""
        from victor.core.container import ServiceLifetime
        from victor.core.verticals.protocols import ModeConfigProviderProtocol

        def create_mode_config(_):
            from victor_coding.mode_config import CodingModeConfigProvider

            default_mode = getattr(settings, "default_mode", "default")
            return CodingModeConfigProvider(default_mode=default_mode)

        container.register_or_replace(
            ModeConfigProviderProtocol,
            create_mode_config,
            ServiceLifetime.SINGLETON,
        )

    def _register_tool_dependencies(
        self,
        container: "ServiceContainer",
        settings: "Settings",
    ) -> None:
        """Register tool dependency provider."""
        from victor.core.container import ServiceLifetime
        from victor.core.verticals.protocols import ToolDependencyProviderProtocol

        def create_tool_deps(_):
            from victor.core.tool_dependency_loader import (
                create_vertical_tool_dependency_provider,
            )

            return create_vertical_tool_dependency_provider("coding")

        container.register_or_replace(
            ToolDependencyProviderProtocol,
            create_tool_deps,
            ServiceLifetime.SINGLETON,
        )

    def get_required_services(self) -> List[Type]:
        """Get list of required service types.

        Returns:
            List of protocol types this vertical requires
        """
        return []  # No hard requirements

    def get_optional_services(self) -> List[Type]:
        """Get list of optional service types.

        Returns:
            List of optional protocol types
        """
        from victor.core.verticals.protocols import (
            ModeConfigProviderProtocol,
            ToolDependencyProviderProtocol,
        )

        return [
            CodingMiddlewareProtocol,
            CodingSafetyProtocol,
            CodingPromptProtocol,
            ModeConfigProviderProtocol,
            ToolDependencyProviderProtocol,
        ]


__all__ = [
    "CodingServiceProvider",
    "CodingMiddlewareProtocol",
    "CodingSafetyProtocol",
    "CodingPromptProtocol",
]
