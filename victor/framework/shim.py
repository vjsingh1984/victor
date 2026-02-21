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

"""Backward-compatible shim bridging legacy CLI to Framework API.

This module provides a transition layer that allows existing code using
AgentOrchestrator.from_settings() to benefit from framework features
(observability, verticals, unified events) without modification.

Design Pattern: Adapter
- Wraps legacy creation with framework wiring
- Adds framework features without breaking existing callers

Migration Path:
    1. Phase 1: FrameworkShim wraps from_settings (current)
    2. Phase 2: CLI uses FrameworkShim instead of direct calls
    3. Phase 3: Observability wired by default
    4. Phase 4: Clean migration to Agent.create() path

Example:
    # Before:
    orchestrator = await AgentOrchestrator.from_settings(settings, profile)

    # After (drop-in replacement):
    shim = FrameworkShim(settings, profile)
    orchestrator = await shim.create_orchestrator()

    # With vertical:
    shim = FrameworkShim(settings, profile, vertical=CodingAssistant)
    orchestrator = await shim.create_orchestrator()

    # Access observability
    if shim.observability:
        shim.observability.on_session_start({"mode": "cli"})
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Optional, Type, Union

from victor.framework.vertical_service import apply_vertical_configuration

if TYPE_CHECKING:
    from victor.core.protocols import OrchestratorProtocol as AgentOrchestrator
    from victor.config.settings import Settings
    from victor.observability.integration import ObservabilityIntegration
    from victor.core.verticals.base import VerticalBase, VerticalConfig

logger = logging.getLogger(__name__)


class FrameworkShim:
    """Shim layer for framework-enhanced orchestrator creation.

    This class bridges the gap between the legacy CLI path (which uses
    AgentOrchestrator.from_settings()) and the new Framework API (which
    uses Agent.create()). It allows the CLI to gain framework features
    like observability and vertical configuration without breaking changes.

    Features:
        - Wraps AgentOrchestrator.from_settings() transparently
        - Wires ObservabilityIntegration automatically
        - Applies vertical configuration (tools, stages, system prompt)
        - Provides session ID generation and propagation
        - Maintains 100% backward compatibility

    Attributes:
        orchestrator: Created orchestrator instance (after create_orchestrator())
        observability: ObservabilityIntegration instance (if enabled)
        session_id: Session ID for event correlation

    Usage:
        # Basic usage (same as from_settings)
        shim = FrameworkShim(settings, "default")
        orchestrator = await shim.create_orchestrator()

        # With vertical
        shim = FrameworkShim(settings, "default", vertical=CodingAssistant)
        orchestrator = await shim.create_orchestrator()

        # With explicit session ID
        shim = FrameworkShim(settings, session_id="user-session-123")
        orchestrator = await shim.create_orchestrator()
    """

    def __init__(
        self,
        settings: "Settings",
        profile_name: str = "default",
        thinking: bool = False,
        vertical: Optional[Union[Type["VerticalBase"], str]] = None,
        enable_observability: bool = True,
        session_id: Optional[str] = None,
        enable_cqrs_bridge: bool = False,
    ) -> None:
        """Initialize the FrameworkShim.

        Args:
            settings: Victor settings object.
            profile_name: Profile name from profiles.yaml.
            thinking: Enable extended thinking mode.
            vertical: Vertical class or name to apply. If a string,
                will be looked up from VerticalRegistry.
            enable_observability: Whether to wire ObservabilityIntegration.
            session_id: Optional session ID for event correlation.
                If not provided, a new UUID will be generated.
            enable_cqrs_bridge: Whether to enable CQRS event bridging.
        """
        self._settings = settings
        self._profile_name = profile_name
        self._thinking = thinking
        self._vertical = self._resolve_vertical(vertical)
        self._enable_observability = enable_observability
        self._session_id = session_id or str(uuid.uuid4())
        self._enable_cqrs_bridge = enable_cqrs_bridge

        # Set after create_orchestrator()
        self._orchestrator: Optional["AgentOrchestrator"] = None
        self._observability: Optional["ObservabilityIntegration"] = None
        self._vertical_config: Optional["VerticalConfig"] = None

    def _resolve_vertical(
        self, vertical: Optional[Union[Type["VerticalBase"], str]]
    ) -> Optional[Type["VerticalBase"]]:
        """Resolve vertical from name or class.

        Args:
            vertical: Vertical class or string name.

        Returns:
            Vertical class or None.
        """
        if vertical is None:
            return None

        if isinstance(vertical, str):
            from victor.core.verticals.base import VerticalRegistry

            resolved = VerticalRegistry.get(vertical)
            if resolved is None:
                logger.warning(
                    f"Vertical '{vertical}' not found in registry. "
                    f"Available: {VerticalRegistry.list_names()}"
                )
            return resolved

        return vertical

    async def create_orchestrator(self) -> "AgentOrchestrator":
        """Create orchestrator with framework features wired.

        This is the main entry point. It:
        1. Bootstraps the DI container with the correct vertical
        2. Creates the base orchestrator via from_settings()
        3. Applies vertical configuration if specified
        4. Wires observability if enabled
        5. Returns the enhanced orchestrator

        Returns:
            Configured AgentOrchestrator instance.

        Raises:
            RuntimeError: If orchestrator creation fails.
        """
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.core.bootstrap import ensure_bootstrapped

        logger.debug(
            f"FrameworkShim creating orchestrator: profile={self._profile_name}, "
            f"thinking={self._thinking}, vertical={self._vertical}, "
            f"observability={self._enable_observability}"
        )

        # Step 0: Ensure bootstrap with correct vertical BEFORE orchestrator creation
        # This ensures vertical services are registered with the correct vertical name
        vertical_name = self._vertical.name if self._vertical else None
        ensure_bootstrapped(self._settings, vertical=vertical_name)

        # Step 1: Create base orchestrator
        self._orchestrator = await AgentOrchestrator.from_settings(
            self._settings,
            profile_name=self._profile_name,
            thinking=self._thinking,
        )

        # Step 2: Apply vertical configuration
        if self._vertical:
            self._apply_vertical(self._vertical)

        # Step 3: Wire observability
        if self._enable_observability:
            self._wire_observability()

        logger.debug(f"FrameworkShim created orchestrator: session_id={self._session_id}")

        return self._orchestrator

    def _apply_vertical(self, vertical: Type["VerticalBase"]) -> None:
        """Apply vertical configuration to orchestrator.

        This delegates to VerticalIntegrationPipeline for unified vertical
        application across both CLI (FrameworkShim) and SDK (Agent.create()) paths.

        The pipeline applies:
        - System prompt (via prompt_builder)
        - Tool filter (via set_enabled_tools protocol)
        - Stage configuration (via vertical context)
        - Middleware chain (via apply_vertical_middleware protocol)
        - Safety extensions (via apply_vertical_safety_patterns protocol)
        - Prompt contributions (for task type hints)
        - Mode configuration and tool dependencies

        Args:
            vertical: Vertical class to apply.
        """
        logger.debug(f"Applying vertical via pipeline: {vertical.name}")

        # Use shared framework service for vertical application
        result = apply_vertical_configuration(self._orchestrator, vertical, source="cli")

        # Store result for access
        self._vertical_config = result.context.config if result.context else None
        self._integration_result = result

        if result.success:
            logger.info(
                f"Applied vertical '{result.vertical_name}' via CLI path: "
                f"tools={len(result.tools_applied)}, "
                f"middleware={result.middleware_count}, "
                f"safety={result.safety_patterns_count}, "
                f"hints={result.prompt_hints_count}"
            )
        else:
            for error in result.errors:
                logger.error(f"Vertical integration error: {error}")
            for warning in result.warnings:
                logger.warning(f"Vertical integration warning: {warning}")

    def _wire_observability(self) -> None:
        """Wire ObservabilityIntegration to orchestrator.

        Delegates to the shared setup_observability_integration() helper
        used by both SDK and CLI paths.
        """
        from victor.framework._internal import setup_observability_integration

        self._observability = setup_observability_integration(
            self._orchestrator,
            session_id=self._session_id,
            enable_cqrs_bridge=self._enable_cqrs_bridge,
        )

        logger.debug(
            f"Wired observability: session_id={self._session_id}, "
            f"cqrs_bridge={self._enable_cqrs_bridge}"
        )

    # =========================================================================
    # Public Properties
    # =========================================================================

    @property
    def orchestrator(self) -> Optional["AgentOrchestrator"]:
        """Get the created orchestrator.

        Returns:
            Orchestrator instance, or None if not yet created.
        """
        return self._orchestrator

    @property
    def observability(self) -> Optional["ObservabilityIntegration"]:
        """Get the ObservabilityIntegration if enabled.

        Returns:
            ObservabilityIntegration instance, or None if disabled.
        """
        return self._observability

    @property
    def session_id(self) -> str:
        """Get the session ID for event correlation.

        Returns:
            Session ID string.
        """
        return self._session_id

    @property
    def vertical(self) -> Optional[Type["VerticalBase"]]:
        """Get the applied vertical class.

        Returns:
            Vertical class, or None if no vertical applied.
        """
        return self._vertical

    @property
    def vertical_config(self) -> Optional["VerticalConfig"]:
        """Get the applied vertical configuration.

        Returns:
            VerticalConfig, or None if no vertical applied.
        """
        return self._vertical_config

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    def emit_session_start(self, metadata: Optional[dict] = None) -> None:
        """Emit a session start event if observability is enabled.

        Args:
            metadata: Optional session metadata.
        """
        if self._observability:
            self._observability.on_session_start(metadata or {})

    def emit_session_end(
        self,
        tool_calls: int = 0,
        duration_seconds: Optional[float] = None,
        success: bool = True,
    ) -> None:
        """Emit a session end event if observability is enabled.

        Args:
            tool_calls: Total tool calls in session.
            duration_seconds: Session duration.
            success: Whether session completed successfully.
        """
        if self._observability:
            self._observability.on_session_end(
                tool_calls=tool_calls,
                duration_seconds=duration_seconds,
                success=success,
            )


def get_vertical(name: Optional[str]) -> Optional[Type["VerticalBase"]]:
    """Convenience function to look up a vertical by name.

    This is a thin wrapper around VerticalRegistry.get() for CLI usage.

    Args:
        name: Vertical name (case-insensitive).

    Returns:
        Vertical class or None if not found.

    Example:
        vertical = get_vertical("coding")
        if vertical:
            shim = FrameworkShim(settings, vertical=vertical)
    """
    if name is None:
        return None

    # Import package root and re-run built-in registration once module init settles.
    # This self-heals early-import circular ordering where a built-in (notably coding)
    # can be skipped during the package's first import.
    import victor.core.verticals as core_verticals

    core_verticals._register_builtin_verticals()
    from victor.core.verticals.base import VerticalRegistry

    # Try exact match first
    result = VerticalRegistry.get(name)
    if result:
        return result

    # Try case-insensitive match
    name_lower = name.lower()
    for registered_name in VerticalRegistry.list_names():
        if registered_name.lower() == name_lower:
            return VerticalRegistry.get(registered_name)

    return None


def list_verticals() -> list[str]:
    """List all available vertical names.

    Returns:
        List of registered vertical names.

    Example:
        print(f"Available verticals: {list_verticals()}")
    """
    # Import package root and re-run built-in registration once module init settles.
    import victor.core.verticals as core_verticals

    core_verticals._register_builtin_verticals()
    from victor.core.verticals.base import VerticalRegistry

    return VerticalRegistry.list_names()
