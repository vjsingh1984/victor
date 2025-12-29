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

"""Capability Registry Mixin for orchestrator capability discovery.

Provides explicit capability declaration and invocation, replacing hasattr
duck-typing with type-safe protocol conformance.

Design Pattern: Mixin + Registry
================================
- CapabilityRegistryMixin provides capability discovery methods
- Capabilities are registered at initialization, not discovered at runtime
- Invocation goes through the registry, not getattr

Usage:
    class AgentOrchestrator(ModeAwareMixin, CapabilityRegistryMixin):
        def __init__(self, ...):
            ...
            self._register_capabilities()  # At end of __init__

Example (caller):
    # Instead of:
    if hasattr(orch, "set_enabled_tools") and callable(orch.set_enabled_tools):
        orch.set_enabled_tools(tools)

    # Use:
    if orch.has_capability("enabled_tools"):
        orch.invoke_capability("enabled_tools", tools)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Set

from victor.framework.protocols import (
    CapabilityType,
    OrchestratorCapability,
    CapabilityRegistryProtocol,
    IncompatibleVersionError,
)

logger = logging.getLogger(__name__)


class CapabilityRegistryMixin:
    """Mixin providing capability registry functionality.

    Add this mixin to AgentOrchestrator to enable explicit capability
    discovery and invocation.

    The mixin maintains a registry of capabilities that can be:
    - Queried by name or type
    - Invoked via setter methods
    - Read via getter methods or attribute access
    """

    def __init_capability_registry__(self) -> None:
        """Initialize the capability registry.

        Call this at the end of __init__ after all components are initialized.
        """
        self._capabilities: Dict[str, OrchestratorCapability] = {}
        self._capability_methods: Dict[str, Callable] = {}
        self._register_default_capabilities()

    def _register_capability(
        self,
        capability: OrchestratorCapability,
        setter_method: Optional[Callable] = None,
        getter_method: Optional[Callable] = None,
    ) -> None:
        """Register a capability with its methods.

        Args:
            capability: Capability declaration (includes version)
            setter_method: Method to call for setting (if capability.setter)
            getter_method: Method to call for getting (if capability.getter)

        Note:
            Capabilities default to version "1.0" for backward compatibility.
            Use the version field in OrchestratorCapability to specify a
            different version when registering updated capabilities.
        """
        # Warn if registering deprecated capability
        if capability.deprecated:
            logger.warning(
                f"Registering deprecated capability: {capability.name} "
                f"(v{capability.version}). {capability.deprecated_message}"
            )

        self._capabilities[capability.name] = capability

        if capability.setter and setter_method:
            self._capability_methods[f"{capability.name}:set"] = setter_method
        if capability.getter and getter_method:
            self._capability_methods[f"{capability.name}:get"] = getter_method

        logger.debug(
            f"Registered capability: {capability.name} "
            f"(type={capability.capability_type.value}, version={capability.version})"
        )

    def _register_default_capabilities(self) -> None:
        """Register all orchestrator capabilities.

        This method should be called at the end of __init__ after all
        components are initialized. It registers capabilities for all
        the methods that vertical_integration.py probes with hasattr.
        """
        # Tool capabilities
        self._register_capability(
            OrchestratorCapability(
                name="enabled_tools",
                capability_type=CapabilityType.TOOL,
                setter="set_enabled_tools",
                description="Set enabled tools from vertical",
            ),
            setter_method=getattr(self, "set_enabled_tools", None),
        )

        # Prompt capabilities
        self._register_capability(
            OrchestratorCapability(
                name="prompt_builder",
                capability_type=CapabilityType.PROMPT,
                attribute="prompt_builder",
                description="System prompt builder component",
            ),
        )

        self._register_capability(
            OrchestratorCapability(
                name="custom_prompt",
                capability_type=CapabilityType.PROMPT,
                setter="set_custom_prompt",
                getter="get_custom_prompt",
                description="Custom prompt setter/getter via prompt_builder",
            ),
            setter_method=self._set_custom_prompt_via_builder,
            getter_method=self._get_custom_prompt_via_builder,
        )

        self._register_capability(
            OrchestratorCapability(
                name="task_type_hints",
                capability_type=CapabilityType.PROMPT,
                setter="set_task_type_hints",
                description="Set task type hints via prompt_builder",
            ),
            setter_method=self._set_task_type_hints_via_builder,
        )

        self._register_capability(
            OrchestratorCapability(
                name="prompt_section",
                capability_type=CapabilityType.PROMPT,
                setter="add_prompt_section",
                description="Add prompt section via prompt_builder",
            ),
            setter_method=self._add_prompt_section_via_builder,
        )

        # Middleware capabilities
        self._register_capability(
            OrchestratorCapability(
                name="vertical_middleware",
                capability_type=CapabilityType.SAFETY,
                setter="apply_vertical_middleware",
                description="Apply vertical middleware chain",
            ),
            setter_method=getattr(self, "apply_vertical_middleware", None),
        )

        self._register_capability(
            OrchestratorCapability(
                name="middleware_chain",
                capability_type=CapabilityType.SAFETY,
                attribute="_middleware_chain",
                description="Middleware chain component",
            ),
        )

        # Safety capabilities
        self._register_capability(
            OrchestratorCapability(
                name="vertical_safety_patterns",
                capability_type=CapabilityType.SAFETY,
                setter="apply_vertical_safety_patterns",
                description="Apply vertical safety patterns",
            ),
            setter_method=getattr(self, "apply_vertical_safety_patterns", None),
        )

        self._register_capability(
            OrchestratorCapability(
                name="safety_patterns",
                capability_type=CapabilityType.SAFETY,
                setter="add_safety_patterns",
                description="Add safety patterns to checker",
            ),
            setter_method=self._add_safety_patterns,
        )

        # Mode capabilities
        self._register_capability(
            OrchestratorCapability(
                name="adaptive_mode_controller",
                capability_type=CapabilityType.MODE,
                attribute="adaptive_mode_controller",
                description="Adaptive mode controller component",
            ),
        )

        self._register_capability(
            OrchestratorCapability(
                name="mode_configs",
                capability_type=CapabilityType.MODE,
                setter="set_mode_configs",
                description="Set mode configurations",
            ),
            setter_method=self._set_mode_configs,
        )

        self._register_capability(
            OrchestratorCapability(
                name="default_budget",
                capability_type=CapabilityType.MODE,
                setter="set_default_budget",
                description="Set default tool budget",
            ),
            setter_method=self._set_default_budget,
        )

        # Workflow capabilities
        self._register_capability(
            OrchestratorCapability(
                name="tool_sequence_tracker",
                capability_type=CapabilityType.WORKFLOW,
                attribute="_sequence_tracker",
                description="Tool sequence tracker component",
            ),
        )

        self._register_capability(
            OrchestratorCapability(
                name="tool_dependencies",
                capability_type=CapabilityType.WORKFLOW,
                setter="set_tool_dependencies",
                description="Set tool dependencies",
            ),
            setter_method=self._set_tool_dependencies,
        )

        self._register_capability(
            OrchestratorCapability(
                name="tool_sequences",
                capability_type=CapabilityType.WORKFLOW,
                setter="set_tool_sequences",
                description="Set tool sequences",
            ),
            setter_method=self._set_tool_sequences,
        )

        # Vertical capabilities
        self._register_capability(
            OrchestratorCapability(
                name="vertical_context",
                capability_type=CapabilityType.VERTICAL,
                setter="set_vertical_context",
                getter="get_vertical_context",
                description="Vertical context management",
            ),
            setter_method=getattr(self, "set_vertical_context", None),
            getter_method=lambda: getattr(self, "_vertical_context", None),
        )

        # RL capabilities
        self._register_capability(
            OrchestratorCapability(
                name="rl_hooks",
                capability_type=CapabilityType.RL,
                setter="set_rl_hooks",
                description="Set RL hooks for outcome recording",
            ),
            setter_method=self._set_rl_hooks,
        )

        # Team capabilities
        self._register_capability(
            OrchestratorCapability(
                name="team_specs",
                capability_type=CapabilityType.TEAM,
                setter="set_team_specs",
                description="Set team specifications",
            ),
            setter_method=self._set_team_specs,
        )

        logger.info(
            f"Capability registry initialized with {len(self._capabilities)} capabilities"
        )

    # =========================================================================
    # CapabilityRegistryProtocol implementation
    # =========================================================================

    def get_capabilities(self) -> Dict[str, OrchestratorCapability]:
        """Get all registered capabilities.

        Returns:
            Dict mapping capability names to their declarations
        """
        return dict(self._capabilities)

    def has_capability(
        self,
        name: str,
        min_version: Optional[str] = None,
    ) -> bool:
        """Check if a capability is available and meets version requirements.

        Args:
            name: Capability name to check
            min_version: Minimum required version (default: None = any version)

        Returns:
            True if capability is registered, functional, and meets version requirement

        Example:
            # Check for any version
            if orch.has_capability("enabled_tools"):
                ...

            # Check for minimum version
            if orch.has_capability("enabled_tools", min_version="1.1"):
                # Use v1.1+ features
                ...
        """
        if name not in self._capabilities:
            return False

        cap = self._capabilities[name]

        # Check version compatibility if min_version specified
        if min_version is not None:
            if not cap.is_compatible_with(min_version):
                logger.debug(
                    f"Capability '{name}' v{cap.version} does not meet "
                    f"required version {min_version}"
                )
                return False

        # Check if the underlying component exists
        if cap.attribute:
            return hasattr(self, cap.attribute) and getattr(self, cap.attribute) is not None
        if cap.setter:
            method_key = f"{name}:set"
            return method_key in self._capability_methods
        if cap.getter:
            method_key = f"{name}:get"
            return method_key in self._capability_methods

        return False

    def get_capability(self, name: str) -> Optional[OrchestratorCapability]:
        """Get a specific capability declaration.

        Args:
            name: Capability name

        Returns:
            Capability declaration or None if not found
        """
        return self._capabilities.get(name)

    def get_capability_version(self, name: str) -> Optional[str]:
        """Get the version of a registered capability.

        Args:
            name: Capability name

        Returns:
            Version string or None if capability not found

        Example:
            version = orch.get_capability_version("enabled_tools")
            if version:
                print(f"enabled_tools is at version {version}")
        """
        cap = self._capabilities.get(name)
        return cap.version if cap else None

    def invoke_capability(
        self,
        name: str,
        *args: Any,
        min_version: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke a capability's setter method with optional version check.

        Args:
            name: Capability name
            *args: Positional arguments for setter
            min_version: Minimum required version (default: None = no check)
            **kwargs: Keyword arguments for setter

        Returns:
            Result from setter method

        Raises:
            KeyError: If capability not found
            TypeError: If capability has no setter
            IncompatibleVersionError: If capability version is incompatible

        Example:
            # Invoke without version check (backward compatible)
            orch.invoke_capability("enabled_tools", {"read", "write"})

            # Invoke with version requirement
            orch.invoke_capability(
                "enabled_tools",
                {"read", "write"},
                min_version="1.1"
            )
        """
        if name not in self._capabilities:
            raise KeyError(f"Capability '{name}' not found")

        cap = self._capabilities[name]

        # Check version compatibility if required
        if min_version is not None:
            if not cap.is_compatible_with(min_version):
                raise IncompatibleVersionError(
                    capability_name=name,
                    required_version=min_version,
                    actual_version=cap.version,
                )

        # Warn if invoking deprecated capability
        if cap.deprecated:
            import warnings
            warnings.warn(
                f"Capability '{name}' (v{cap.version}) is deprecated. "
                f"{cap.deprecated_message}",
                DeprecationWarning,
                stacklevel=2,
            )

        if not cap.setter:
            raise TypeError(f"Capability '{name}' has no setter method")

        method_key = f"{name}:set"
        if method_key not in self._capability_methods:
            raise TypeError(f"Capability '{name}' setter not registered")

        method = self._capability_methods[method_key]
        if method is None:
            raise TypeError(f"Capability '{name}' setter method is None")

        return method(*args, **kwargs)

    def get_capability_value(self, name: str) -> Any:
        """Get a capability's current value via getter or attribute.

        Args:
            name: Capability name

        Returns:
            Current value

        Raises:
            KeyError: If capability not found
            TypeError: If capability has no getter/attribute
        """
        if name not in self._capabilities:
            raise KeyError(f"Capability '{name}' not found")

        cap = self._capabilities[name]

        # Try getter first
        if cap.getter:
            method_key = f"{name}:get"
            if method_key in self._capability_methods:
                method = self._capability_methods[method_key]
                if method:
                    return method()

        # Fall back to attribute
        if cap.attribute:
            return getattr(self, cap.attribute, None)

        raise TypeError(f"Capability '{name}' has no getter or attribute")

    def get_capabilities_by_type(
        self, capability_type: CapabilityType
    ) -> Dict[str, OrchestratorCapability]:
        """Get all capabilities of a specific type.

        Args:
            capability_type: Type to filter by

        Returns:
            Dict of matching capabilities
        """
        return {
            name: cap
            for name, cap in self._capabilities.items()
            if cap.capability_type == capability_type
        }

    # =========================================================================
    # Internal capability setter implementations
    # These bridge the capability system to underlying components
    # =========================================================================

    def _set_custom_prompt_via_builder(self, prompt: str) -> None:
        """Set custom prompt via prompt_builder."""
        if hasattr(self, "prompt_builder") and self.prompt_builder:
            if hasattr(self.prompt_builder, "set_custom_prompt"):
                self.prompt_builder.set_custom_prompt(prompt)
            elif hasattr(self.prompt_builder, "_custom_prompt"):
                self.prompt_builder._custom_prompt = prompt

    def _get_custom_prompt_via_builder(self) -> Optional[str]:
        """Get custom prompt via prompt_builder."""
        if hasattr(self, "prompt_builder") and self.prompt_builder:
            if hasattr(self.prompt_builder, "get_custom_prompt"):
                return self.prompt_builder.get_custom_prompt()
            elif hasattr(self.prompt_builder, "_custom_prompt"):
                return self.prompt_builder._custom_prompt
        return None

    def _set_task_type_hints_via_builder(self, hints: Dict[str, Any]) -> None:
        """Set task type hints via prompt_builder."""
        if hasattr(self, "prompt_builder") and self.prompt_builder:
            if hasattr(self.prompt_builder, "set_task_type_hints"):
                self.prompt_builder.set_task_type_hints(hints)
            elif hasattr(self.prompt_builder, "_task_type_hints"):
                self.prompt_builder._task_type_hints = hints

    def _add_prompt_section_via_builder(self, section: str) -> None:
        """Add prompt section via prompt_builder."""
        if hasattr(self, "prompt_builder") and self.prompt_builder:
            if hasattr(self.prompt_builder, "add_prompt_section"):
                self.prompt_builder.add_prompt_section(section)

    def _add_safety_patterns(self, patterns: list) -> None:
        """Add safety patterns to checker."""
        if hasattr(self, "_safety_checker") and self._safety_checker:
            if hasattr(self._safety_checker, "add_patterns"):
                self._safety_checker.add_patterns(patterns)
            elif hasattr(self._safety_checker, "_custom_patterns"):
                self._safety_checker._custom_patterns.extend(patterns)

    def _set_mode_configs(self, configs: Dict[str, Any]) -> None:
        """Set mode configurations."""
        if hasattr(self, "adaptive_mode_controller"):
            controller = getattr(self, "adaptive_mode_controller", None)
            if controller and hasattr(controller, "set_mode_configs"):
                controller.set_mode_configs(configs)

    def _set_default_budget(self, budget: int) -> None:
        """Set default tool budget."""
        if hasattr(self, "adaptive_mode_controller"):
            controller = getattr(self, "adaptive_mode_controller", None)
            if controller and hasattr(controller, "set_default_budget"):
                controller.set_default_budget(budget)

    def _set_tool_dependencies(self, dependencies: Dict[str, Set[str]]) -> None:
        """Set tool dependencies."""
        if hasattr(self, "_sequence_tracker") and self._sequence_tracker:
            if hasattr(self._sequence_tracker, "set_dependencies"):
                self._sequence_tracker.set_dependencies(dependencies)

    def _set_tool_sequences(self, sequences: list) -> None:
        """Set tool sequences."""
        if hasattr(self, "_sequence_tracker") and self._sequence_tracker:
            if hasattr(self._sequence_tracker, "set_sequences"):
                self._sequence_tracker.set_sequences(sequences)

    def _set_rl_hooks(self, hooks: Any) -> None:
        """Set RL hooks."""
        if hasattr(self, "_rl_hooks"):
            self._rl_hooks = hooks
        else:
            setattr(self, "_rl_hooks", hooks)

    def _set_team_specs(self, specs: Dict[str, Any]) -> None:
        """Set team specifications."""
        if hasattr(self, "_team_specs"):
            self._team_specs = specs
        else:
            setattr(self, "_team_specs", specs)


__all__ = ["CapabilityRegistryMixin"]
