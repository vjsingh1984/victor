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

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.config.settings import Settings
    from victor.observability.integration import ObservabilityIntegration
    from victor.verticals.base import VerticalBase, VerticalConfig

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
            from victor.verticals.base import VerticalRegistry

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

        This applies:
        - System prompt (via prompt_builder)
        - Tool filter (stored for selection-time filtering)
        - Stage configuration (stored for state machine hints)
        - Middleware chain (for tool execution processing)
        - Safety extensions (for dangerous operation patterns)
        - Prompt contributions (for task type hints)

        Args:
            vertical: Vertical class to apply.
        """
        logger.debug(f"Applying vertical: {vertical.name}")

        # Get the full config from vertical
        self._vertical_config = vertical.get_config()

        # Apply system prompt
        system_prompt = vertical.get_system_prompt()
        if system_prompt:
            self._apply_system_prompt(system_prompt)

        # Apply tool filter using ToolsProtocol method
        # These tools will be enabled/prioritized during tool selection
        tools = vertical.get_tools()
        if tools:
            # Use protocol method if available (proper API)
            if hasattr(self._orchestrator, "set_enabled_tools") and callable(
                self._orchestrator.set_enabled_tools
            ):
                self._orchestrator.set_enabled_tools(set(tools))
            else:
                logger.warning(
                    "Orchestrator does not implement set_enabled_tools(); "
                    "vertical tools may not be applied properly"
                )
            logger.debug(f"Applied vertical tools: {len(tools)} tools")

        # Store stage config for state machine hints
        stages = vertical.get_stages()
        if stages:
            self._orchestrator._vertical_stages = stages
            logger.debug(f"Applied vertical stages: {list(stages.keys())}")

        # Store vertical metadata on orchestrator
        self._orchestrator._vertical_name = vertical.name
        self._orchestrator._vertical_config = self._vertical_config

        # Apply vertical extensions (new protocol-based system)
        self._apply_vertical_extensions(vertical)

    def _apply_vertical_extensions(self, vertical: Type["VerticalBase"]) -> None:
        """Apply vertical extensions to orchestrator.

        This applies the protocol-based extension system:
        - Middleware chain for tool execution
        - Safety extensions for danger pattern detection
        - Prompt contributors for task hints
        - Mode configurations for tool budgets

        Args:
            vertical: Vertical class to get extensions from.
        """
        extensions = vertical.get_extensions()
        if extensions is None:
            logger.debug("No extensions available for vertical")
            return

        # Apply middleware chain
        if extensions.middleware:
            self._apply_middleware(extensions.middleware)

        # Apply safety extensions
        if extensions.safety_extensions:
            self._apply_safety_extensions(extensions.safety_extensions)

        # Apply prompt contributors
        if extensions.prompt_contributors:
            self._apply_prompt_contributors(extensions.prompt_contributors)

        # Apply mode config provider
        if extensions.mode_config_provider:
            self._apply_mode_config(extensions.mode_config_provider)

        # Apply tool dependency provider
        if extensions.tool_dependency_provider:
            self._apply_tool_dependencies(extensions.tool_dependency_provider)

        logger.debug(
            f"Applied vertical extensions: "
            f"middleware={len(extensions.middleware)}, "
            f"safety={len(extensions.safety_extensions)}, "
            f"prompts={len(extensions.prompt_contributors)}"
        )

    def _apply_middleware(self, middleware_list: list) -> None:
        """Apply middleware implementations to orchestrator.

        Args:
            middleware_list: List of MiddlewareProtocol implementations.
        """
        # Use orchestrator's apply method if available (preferred)
        if hasattr(self._orchestrator, "apply_vertical_middleware"):
            self._orchestrator.apply_vertical_middleware(middleware_list)
            return

        # Fallback: Check if orchestrator has middleware chain support
        chain = getattr(self._orchestrator, "middleware_chain", None)
        if chain is not None and hasattr(chain, "add"):
            for middleware in middleware_list:
                chain.add(middleware)
            logger.debug(f"Added {len(middleware_list)} middleware to chain")
        else:
            chain = getattr(self._orchestrator, "_middleware_chain", None)
            if chain is not None and hasattr(chain, "add"):
                for middleware in middleware_list:
                    chain.add(middleware)
                logger.debug(f"Added {len(middleware_list)} middleware to chain")
            else:
                # Store for later use by orchestrator
                self._orchestrator._vertical_middleware = middleware_list
                logger.debug(f"Stored {len(middleware_list)} middleware for orchestrator")

    def _apply_safety_extensions(self, safety_extensions: list) -> None:
        """Apply safety extensions to orchestrator.

        Args:
            safety_extensions: List of SafetyExtensionProtocol implementations.
        """
        # Collect all patterns from extensions
        all_patterns = []
        for ext in safety_extensions:
            all_patterns.extend(ext.get_bash_patterns())
            all_patterns.extend(ext.get_file_patterns())

        # Use orchestrator's apply method if available (preferred)
        if hasattr(self._orchestrator, "apply_vertical_safety_patterns"):
            self._orchestrator.apply_vertical_safety_patterns(all_patterns)
            return

        # Fallback: Store patterns on orchestrator for safety checker
        safety_checker = getattr(self._orchestrator, "safety_checker", None)
        if safety_checker is not None:
            # If safety checker exists, try to add patterns
            if hasattr(safety_checker, "add_patterns"):
                safety_checker.add_patterns(all_patterns)
            elif hasattr(safety_checker, "_custom_patterns"):
                safety_checker._custom_patterns.extend(all_patterns)
            logger.debug(f"Applied {len(all_patterns)} safety patterns to checker")
        else:
            # Store for later initialization
            self._orchestrator._vertical_safety_patterns = all_patterns
            logger.debug(f"Stored {len(all_patterns)} safety patterns for orchestrator")

    def _apply_prompt_contributors(self, contributors: list) -> None:
        """Apply prompt contributors to orchestrator.

        Args:
            contributors: List of PromptContributorProtocol implementations.
        """
        # Merge task hints from all contributors
        merged_hints = {}
        for contributor in sorted(contributors, key=lambda c: c.get_priority()):
            merged_hints.update(contributor.get_task_type_hints())

        # Apply to prompt builder if available
        if hasattr(self._orchestrator, "prompt_builder"):
            prompt_builder = self._orchestrator.prompt_builder
            if hasattr(prompt_builder, "set_task_type_hints"):
                prompt_builder.set_task_type_hints(merged_hints)
            elif hasattr(prompt_builder, "_task_type_hints"):
                prompt_builder._task_type_hints = merged_hints

            # Apply additional system prompt sections
            for contributor in sorted(contributors, key=lambda c: c.get_priority()):
                section = contributor.get_system_prompt_section()
                if section:
                    if hasattr(prompt_builder, "add_prompt_section"):
                        prompt_builder.add_prompt_section(section)
                    elif hasattr(prompt_builder, "_prompt_sections"):
                        prompt_builder._prompt_sections.append(section)

        # Store for reference
        self._orchestrator._vertical_task_hints = merged_hints
        logger.debug(f"Applied {len(merged_hints)} task type hints")

    def _apply_mode_config(self, mode_provider: Any) -> None:
        """Apply mode configuration provider.

        Args:
            mode_provider: ModeConfigProviderProtocol implementation.
        """
        mode_configs = mode_provider.get_mode_configs()
        default_mode = mode_provider.get_default_mode()
        default_budget = mode_provider.get_default_tool_budget()

        # Apply to adaptive mode controller if available
        if hasattr(self._orchestrator, "adaptive_mode_controller"):
            controller = self._orchestrator.adaptive_mode_controller
            if hasattr(controller, "set_mode_configs"):
                controller.set_mode_configs(mode_configs)
            if hasattr(controller, "set_default_budget"):
                controller.set_default_budget(default_budget)

        # Store for reference
        self._orchestrator._vertical_mode_configs = mode_configs
        self._orchestrator._vertical_default_mode = default_mode
        self._orchestrator._vertical_default_budget = default_budget
        logger.debug(f"Applied {len(mode_configs)} mode configurations")

    def _apply_tool_dependencies(self, dep_provider: Any) -> None:
        """Apply tool dependency provider.

        Args:
            dep_provider: ToolDependencyProviderProtocol implementation.
        """
        dependencies = dep_provider.get_dependencies()
        sequences = dep_provider.get_tool_sequences()

        # Apply to tool sequence tracker if available
        if hasattr(self._orchestrator, "tool_sequence_tracker"):
            tracker = self._orchestrator.tool_sequence_tracker
            if hasattr(tracker, "set_dependencies"):
                tracker.set_dependencies(dependencies)
            if hasattr(tracker, "set_sequences"):
                tracker.set_sequences(sequences)

        # Store for reference
        self._orchestrator._vertical_tool_dependencies = dependencies
        self._orchestrator._vertical_tool_sequences = sequences
        logger.debug(
            f"Applied {len(dependencies)} tool dependencies, " f"{len(sequences)} sequences"
        )

    def _apply_system_prompt(self, system_prompt: str) -> None:
        """Apply a custom system prompt to the orchestrator.

        Args:
            system_prompt: System prompt text.
        """
        # Store the custom system prompt for use in conversation
        self._orchestrator._framework_system_prompt = system_prompt

        # If orchestrator has a prompt_builder, configure it
        if hasattr(self._orchestrator, "prompt_builder"):
            prompt_builder = self._orchestrator.prompt_builder
            # Try set_custom_prompt method first
            if hasattr(prompt_builder, "set_custom_prompt"):
                prompt_builder.set_custom_prompt(system_prompt)
                logger.debug("Applied system prompt via set_custom_prompt")
            # Fall back to direct attribute
            elif hasattr(prompt_builder, "_custom_prompt"):
                prompt_builder._custom_prompt = system_prompt
                logger.debug("Applied system prompt via _custom_prompt attribute")

    def _wire_observability(self) -> None:
        """Wire ObservabilityIntegration to orchestrator.

        This enables:
        - Tool execution events
        - State transition events
        - Model response events
        - Session lifecycle events
        """
        from victor.observability.integration import ObservabilityIntegration

        self._observability = ObservabilityIntegration(
            session_id=self._session_id,
            enable_cqrs_bridge=self._enable_cqrs_bridge,
        )
        self._observability.wire_orchestrator(self._orchestrator)

        # Store reference on orchestrator for access
        self._orchestrator.observability = self._observability

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

    from victor.verticals.base import VerticalRegistry

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
    from victor.verticals.base import VerticalRegistry

    return VerticalRegistry.list_names()
