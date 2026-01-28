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

"""Step handlers for the vertical integration pipeline.

**PRIMARY EXTENSION SURFACE**: This module provides the primary extension
mechanism for vertical development in Victor. Custom integration logic should
be implemented as step handlers (BaseStepHandler subclasses), not by modifying
VerticalIntegrationPipeline or its methods.

**Creating Custom Handlers:**
```python
from victor.framework.step_handlers import BaseStepHandler

class CustomToolsHandler(BaseStepHandler):
    @property
    def name(self) -> str:
        return "custom_tools"

    @property
    def order(self) -> int:
        return 15  # After default tools (10)

    def _do_apply(self, orchestrator, vertical, context, result):
        tools = vertical.get_tools()
        context.apply_enabled_tools(tools)
        result.add_info(f"Applied {len(tools)} tools")

# Register and use
from victor.framework.step_handlers import StepHandlerRegistry
registry = StepHandlerRegistry.default()
registry.add_handler(CustomToolsHandler())
```

**Documentation:**
- Complete Guide: docs/extensions/step_handler_guide.md
- Migration Guide: docs/extensions/step_handler_migration.md
- Examples: docs/extensions/step_handler_examples.md
- Quick Reference: docs/extensions/step_handler_quick_reference.md

This module provides focused step handler classes that implement
individual integration steps. Each handler follows the Single
Responsibility Principle, handling one specific concern.

Design Philosophy:
- Single Responsibility: Each handler handles one concern
- Open/Closed: New steps can be added by creating new handlers
- Liskov Substitution: All handlers implement StepHandlerProtocol
- Interface Segregation: Minimal protocol for step handlers
- Dependency Inversion: Handlers depend on protocols, not implementations

Step Handler Execution Order:
    The handlers execute in a specific order to ensure dependencies are
    satisfied. Lower order numbers execute first.

    Order | Handler             | Class                    | Purpose                                    | Dependencies
    ------|---------------------|--------------------------|--------------------------------------------|-------------
    5     | capability_config   | CapabilityConfigStepHandler | Centralized capability config storage    | None
    10    | tools               | ToolStepHandler          | Tool filter application                    | None
    15    | tiered_config       | TieredConfigStepHandler  | Tiered tool config (mandatory/core/pool)   | Tool
    20    | prompt              | PromptStepHandler        | System prompt and prompt contributors      | None
    30    | safety              | SafetyStepHandler        | Safety patterns and extensions             | None
    40    | config              | ConfigStepHandler        | Stages, mode configs, tool dependencies    | None
    45    | extensions          | ExtensionsStepHandler    | Coordinated extension application          | Config
    50    | middleware          | MiddlewareStepHandler    | Middleware chain application               | Extensions
    60    | framework           | FrameworkStepHandler     | Workflows, RL, teams, chains, personas     | All prior
    100   | context             | ContextStepHandler       | Attach context to orchestrator             | All prior

    Dependencies Explained:
    - TieredConfig (15) runs after Tool (10) because it applies access
      rules to the tools that were just registered
    - Extensions (45) runs after Config (40) because it needs the stages
      and mode configuration to be available for extension handlers
    - Middleware (50) runs after Extensions (45) as it's part of the
      extensions system but needs to apply after core extensions
    - Framework (60) runs last because it registers workflows and teams
      that may depend on tools, prompts, and configurations being
      fully initialized
    - Context (100) runs last as a finalization step to attach the
      complete context to the orchestrator

Architecture:
    StepHandlerProtocol
    ├── CapabilityConfigStepHandler - Centralized capability config storage
    ├── ToolStepHandler - Tool filter application
    ├── TieredConfigStepHandler - Tiered tool config (Phase 1 fix)
    ├── PromptStepHandler - System prompt and prompt contributors
    ├── SafetyStepHandler - Safety patterns and extensions
    ├── ConfigStepHandler - Stages, modes, and tool dependencies
    ├── ExtensionsStepHandler - Coordinated extension application
    ├── MiddlewareStepHandler - Middleware chain application
    ├── FrameworkStepHandler - Workflows, RL config, and team specs
    └── ContextStepHandler - Final context attachment

Usage:
    from victor.framework.step_handlers import (
        CapabilityConfigStepHandler,
        ToolStepHandler,
        TieredConfigStepHandler,
        PromptStepHandler,
        SafetyStepHandler,
        ConfigStepHandler,
        ExtensionsStepHandler,
        MiddlewareStepHandler,
        FrameworkStepHandler,
        ContextStepHandler,
    )

    # Create handlers (or use StepHandlerRegistry.default())
    handlers = [
        CapabilityConfigStepHandler(),
        ToolStepHandler(),
        TieredConfigStepHandler(),
        PromptStepHandler(),
        SafetyStepHandler(),
        ConfigStepHandler(),
        ExtensionsStepHandler(),
        MiddlewareStepHandler(),
        FrameworkStepHandler(),
        ContextStepHandler(),
    ]

    # Apply in pipeline
    for handler in handlers:
        handler.apply(orchestrator, vertical, context, result)
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Type,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.core.verticals import VerticalContext
    from victor.framework.vertical_integration import IntegrationResult
    from victor.core.verticals.base import VerticalBase
    from victor.core.verticals.protocols import (
        RLConfigProviderProtocol,
        TeamSpecProviderProtocol,
    )

# Import protocols for runtime isinstance checks
from victor.core.verticals.protocols import (
    RLConfigProviderProtocol,
    TaskTypeHint,
    TeamSpecProviderProtocol,
)

# Import PromptContributorAdapter for hint normalization
from victor.core.verticals.prompt_adapter import PromptContributorAdapter

from victor.framework.protocols import CapabilityRegistryProtocol

# Import ISP compliance protocols (Phase 1 SOLID Remediation)
from victor.protocols import (
    CapabilityContainerProtocol,
    WorkflowProviderProtocol,
    TieredConfigProviderProtocol,
    ExtensionProviderProtocol,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Capability Helpers (moved from vertical_integration.py for reuse)
# =============================================================================


def _check_capability(obj: Any, capability_name: str, strict_mode: bool = False) -> bool:
    """Check if object has capability via registry.

    .. deprecated:: 0.6.0
        Use :class:`victor.agent.capability_registry.CapabilityHelper.check_capability`
        instead. This function will be removed in v0.7.0.

    Uses protocol-based capability discovery. Orchestrators must implement
    CapabilityRegistryProtocol for proper capability checking.

    SOLID Compliance:
    - Uses protocol, not hasattr (DIP - Dependency Inversion)
    - No private attribute access (SRP - Single Responsibility)

    Args:
        obj: Object to check (should implement CapabilityRegistryProtocol)
        capability_name: Name of capability
        strict_mode: If True, raise TypeError when obj doesn't implement protocol

    Returns:
        True if capability is available via the registry

    Raises:
        TypeError: If strict_mode=True and obj doesn't implement CapabilityRegistryProtocol
    """
    # Delegate to consolidated CapabilityHelper for capability_registry (capability discovery)
    from victor.agent.capability_registry import CapabilityHelper

    return CapabilityHelper.check_capability(obj, capability_name, strict=strict_mode)


def _invoke_capability(
    obj: Any,
    capability_name: str,
    *args: Any,
    strict_mode: bool = False,
    **kwargs: Any,
) -> Any:
    """Invoke capability via registry or public method call.

    .. deprecated:: 0.6.0
        Use :class:`victor.agent.capability_registry.CapabilityHelper.invoke_capability`
        instead. This function will be removed in v0.7.0.

    Uses protocol-based capability invocation. Orchestrators must implement
    CapabilityRegistryProtocol for proper capability invocation.

    SOLID Compliance:
    - Uses protocol, not private attribute assignment (DIP)
    - No direct private attribute writes (SRP)

    Args:
        obj: Object to invoke capability on (should implement CapabilityRegistryProtocol)
        capability_name: Name of capability
        *args: Arguments for capability
        strict_mode: If True, raise TypeError when obj doesn't implement protocol
        **kwargs: Keyword arguments for capability

    Returns:
        Result of capability invocation

    Raises:
        TypeError: If strict_mode=True and obj doesn't implement CapabilityRegistryProtocol
        AttributeError: If capability cannot be invoked
    """
    # Delegate to consolidated CapabilityHelper for capability_registry (capability invocation)
    from victor.agent.capability_registry import CapabilityHelper

    return CapabilityHelper.invoke_capability(
        obj, capability_name, *args, strict=strict_mode, **kwargs
    )


# =============================================================================
# Tiered Tool Config Helper (Workstream D: API Mismatch Fix)
# =============================================================================


def get_tiered_config(vertical: Any) -> Optional[Any]:
    """Get tiered tool config with fallback chain.

    Some verticals implement get_tiered_tool_config() while others
    implement get_tiered_tools(). This helper provides a unified
    interface with a fallback chain:

    1. Try get_tiered_tool_config() first (preferred)
    2. Fall back to get_tiered_tools() if config method missing or returns None
    3. Return None if vertical has neither method

    SOLID Compliance:
    - Uses callable() check instead of just hasattr (ISP)
    - Validates return type is TieredToolConfig-like (LSP)
    - Provides single unified interface (DIP)

    Args:
        vertical: Vertical class to get config from

    Returns:
        TieredToolConfig or None if not available
    """

    def _is_tiered_config(obj: Any) -> bool:
        """Check if object is a valid TieredToolConfig-like object.

        SOLID Compliance (ISP):
        - Uses TieredConfigProviderProtocol for type safety
        - Replaces hasattr() duck typing with protocol-based check
        """
        # Use protocol-based check (ISP compliance)
        if isinstance(obj, TieredConfigProviderProtocol):
            return True

        # Fallback to hasattr for legacy code (deprecated)
        return obj is not None and hasattr(obj, "mandatory")

    def _try_get_config(vertical: Any, method_name: str) -> Optional[Any]:
        """Try to get config from a method, handling various cases."""
        attr = getattr(vertical, method_name, None)
        if attr is None:
            return None

        # Only proceed if it's callable (method/classmethod)
        if not callable(attr):
            return None

        try:
            config = attr()
            if _is_tiered_config(config):
                return config
        except (TypeError, AttributeError):
            # Method exists but is not properly callable or returns error
            pass

        return None

    # Try get_tiered_tool_config() first (preferred method)
    config = _try_get_config(vertical, "get_tiered_tool_config")
    if config is not None:
        return config

    # Fallback to get_tiered_tools()
    config = _try_get_config(vertical, "get_tiered_tools")
    if config is not None:
        return config

    return None


# =============================================================================
# Step Handler Protocol
# =============================================================================


@runtime_checkable
class StepHandlerProtocol(Protocol):
    """Protocol for pipeline step handlers.

    Each step handler implements a single integration step,
    following the Single Responsibility Principle.

    Attributes:
        name: Unique identifier for this step
        order: Execution order (lower numbers run first)
    """

    @property
    def name(self) -> str:
        """Get the unique name of this step handler."""
        ...

    @property
    def order(self) -> int:
        """Get the execution order (lower runs first)."""
        ...

    def apply(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
        strict_mode: bool = False,
    ) -> None:
        """Apply this step to the orchestrator.

        Args:
            orchestrator: Orchestrator instance to configure
            vertical: Vertical class providing configuration
            context: Vertical context to update
            result: Integration result to update with outcomes
            strict_mode: If True, fail on any error
        """
        ...


# =============================================================================
# Base Step Handler
# =============================================================================


class BaseStepHandler(ABC):
    """Abstract base class for step handlers.

    Provides common functionality for all step handlers including
    error handling, logging, and per-step status tracking.

    Subclasses must implement:
    - name property
    - order property
    - _do_apply() method

    Optionally override:
    - is_independent property (default: False)
    - _get_step_details() to provide additional status details

    Handler Independence:
        Independent handlers can run in parallel with each other.
        Dependent handlers must run sequentially after all independent handlers.
        By default, handlers are dependent (is_independent=False).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the unique name of this step handler."""
        ...

    @property
    @abstractmethod
    def order(self) -> int:
        """Get the execution order (lower runs first)."""
        ...

    @property
    def is_independent(self) -> bool:
        """Check if this handler can run independently in parallel.

        Independent handlers don't depend on state modifications from other
        handlers and can safely execute in parallel. By default, handlers
        are dependent to ensure safety.

        Returns:
            True if handler is independent, False otherwise
        """
        return False

    def apply(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
        strict_mode: bool = False,
    ) -> None:
        """Apply this step with error handling and status tracking.

        Automatically records step execution status including:
        - Success/error/warning status
        - Execution duration in milliseconds
        - Optional step-specific details

        Args:
            orchestrator: Orchestrator instance
            vertical: Vertical class
            context: Vertical context
            result: Integration result
            strict_mode: If True, fail on any error
        """
        import time

        start_time = time.perf_counter()
        status = "success"
        details: Optional[Dict[str, Any]] = None

        try:
            self._do_apply(orchestrator, vertical, context, result)
            # Get step-specific details after successful apply
            details = self._get_step_details(result)
        except Exception as e:
            if strict_mode:
                result.add_error(f"{self.name} failed: {e}")
                status = "error"
            else:
                result.add_warning(f"{self.name} error: {e}")
                status = "warning"
            details = {"error": str(e)}
            logger.debug(f"Step {self.name} failed: {e}", exc_info=True)

        # Record step status with timing
        duration_ms = (time.perf_counter() - start_time) * 1000
        result.record_step_status(
            self.name,
            status,
            details=details,
            duration_ms=round(duration_ms, 2),
        )

    def _get_step_details(self, result: "IntegrationResult") -> Optional[Dict[str, Any]]:
        """Get step-specific details for status tracking.

        Override in subclasses to provide meaningful details about
        what was applied during this step.

        Args:
            result: Integration result with updated counts

        Returns:
            Dictionary with step-specific details, or None
        """
        return None

    @abstractmethod
    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Implement the actual step logic.

        Args:
            orchestrator: Orchestrator instance
            vertical: Vertical class
            context: Vertical context
            result: Integration result
        """
        ...


# =============================================================================
# Tool Step Handler
# =============================================================================


class ToolStepHandler(BaseStepHandler):
    """Handler for tool filter application.

    Applies the tool filter from the vertical, canonicalizing
    tool names and enabling them on the orchestrator.

    This handler is independent because it only modifies the tool list
    in context and doesn't depend on state from other handlers.
    """

    @property
    def name(self) -> str:
        return "tools"

    @property
    def order(self) -> int:
        return 10

    @property
    def is_independent(self) -> bool:
        """Tool application can run in parallel with other independent handlers."""
        return True

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Apply tools filter from vertical."""
        tools = vertical.get_tools()
        if not tools:
            return

        # Canonicalize tool names
        canonical_tools = self._canonicalize_tool_names(set(tools))
        context.apply_enabled_tools(canonical_tools)
        result.tools_applied = canonical_tools

        # Use capability-based approach
        if _check_capability(orchestrator, "enabled_tools"):
            _invoke_capability(orchestrator, "enabled_tools", canonical_tools)
            logger.debug(f"Applied {len(canonical_tools)} tools via capability")
        else:
            result.add_warning(
                "Orchestrator does not implement enabled_tools capability; "
                "tools stored in context only"
            )

    def _canonicalize_tool_names(self, tools: Set[str]) -> Set[str]:
        """Canonicalize tool names to ensure consistency.

        Converts legacy tool names (e.g., 'read_file', 'edit_files') to
        canonical short names (e.g., 'read', 'edit').

        Args:
            tools: Set of tool names (may include legacy names)

        Returns:
            Set of canonical tool names
        """
        try:
            from victor.tools.tool_names import get_canonical_name

            return {get_canonical_name(tool) for tool in tools}
        except ImportError:
            logger.debug("Tool name canonicalization not available")
            return tools

    def _get_step_details(self, result: "IntegrationResult") -> Optional[Dict[str, Any]]:
        """Return tool-specific details."""
        if result.tools_applied:
            return {"tools_count": len(result.tools_applied)}
        return {"tools_count": 0, "skipped": True}


# =============================================================================
# Prompt Step Handler
# =============================================================================


class PromptStepHandler(BaseStepHandler):
    """Handler for system prompt and prompt contributors.

    Applies the system prompt from the vertical and merges
    task hints from prompt contributors.

    This handler is independent because it only modifies the prompt
    in context and doesn't depend on state from other handlers.
    """

    @property
    def name(self) -> str:
        return "prompt"

    @property
    def order(self) -> int:
        return 20

    @property
    def is_independent(self) -> bool:
        """Prompt application can run in parallel with other independent handlers."""
        return True

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Apply system prompt from vertical."""
        system_prompt = vertical.get_system_prompt()
        if not system_prompt:
            return

        context.apply_system_prompt(system_prompt)

        # Apply via capability (protocol-first)
        if _check_capability(orchestrator, "custom_prompt"):
            _invoke_capability(orchestrator, "custom_prompt", system_prompt)
            logger.debug("Applied system prompt via capability")
        elif _check_capability(orchestrator, "prompt_builder"):
            # Fallback: use public method only (SOLID compliant)
            prompt_builder = getattr(orchestrator, "prompt_builder", None)
            if prompt_builder and hasattr(prompt_builder, "set_custom_prompt"):
                prompt_builder.set_custom_prompt(system_prompt)
                logger.debug("Applied system prompt via prompt_builder")
            else:
                result.add_warning(
                    "prompt_builder lacks set_custom_prompt method; "
                    "prompt stored in context only"
                )

    def apply_contributors(
        self,
        orchestrator: Any,
        contributors: List[Any],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Apply prompt contributors to orchestrator.

        Uses PromptContributorAdapter to normalize different hint formats
        (dict, string, TaskTypeHint) into a consistent TaskTypeHint interface.

        Args:
            orchestrator: Orchestrator instance
            contributors: List of prompt contributors
            context: Vertical context
            result: Result to update
        """
        # Merge and normalize task hints from all contributors using adapter
        merged_hints: Dict[str, TaskTypeHint] = {}
        for contributor in sorted(contributors, key=lambda c: c.get_priority()):
            raw_hints = contributor.get_task_type_hints()
            # Use PromptContributorAdapter to normalize hints to TaskTypeHint
            normalized_hints = self._normalize_hints(raw_hints)
            merged_hints.update(normalized_hints)

        context.apply_task_hints(merged_hints)
        result.prompt_hints_count = len(merged_hints)

        # Apply task hints via capability (SOLID compliant)
        if _check_capability(orchestrator, "task_type_hints"):
            _invoke_capability(orchestrator, "task_type_hints", merged_hints)
            logger.debug(f"Applied {len(merged_hints)} task hints via capability")
        elif _check_capability(orchestrator, "prompt_builder"):
            prompt_builder = getattr(orchestrator, "prompt_builder", None)
            if prompt_builder and hasattr(prompt_builder, "set_task_type_hints"):
                prompt_builder.set_task_type_hints(merged_hints)
                logger.debug(f"Applied {len(merged_hints)} task hints via prompt_builder")
            else:
                result.add_warning(
                    f"Could not apply {len(merged_hints)} task hints; "
                    "hints stored in context only"
                )

        # Apply prompt sections
        for contributor in sorted(contributors, key=lambda c: c.get_priority()):
            section = contributor.get_system_prompt_section()
            if section:
                context.add_prompt_section(section)
                if _check_capability(orchestrator, "prompt_section"):
                    _invoke_capability(orchestrator, "prompt_section", section)
                elif _check_capability(orchestrator, "prompt_builder"):
                    prompt_builder = getattr(orchestrator, "prompt_builder", None)
                    if prompt_builder and hasattr(prompt_builder, "add_prompt_section"):
                        prompt_builder.add_prompt_section(section)

    def _normalize_hints(self, hints: Dict[str, Any]) -> Dict[str, TaskTypeHint]:
        """Normalize hints to TaskTypeHint using PromptContributorAdapter.

        Supports multiple hint formats:
        - TaskTypeHint instances (preserved as-is)
        - Dict with hint/tool_budget/priority_tools keys
        - String (just the hint text)

        Args:
            hints: Raw hints in various formats

        Returns:
            Dict mapping task types to normalized TaskTypeHint objects
        """
        # Use PromptContributorAdapter.from_dict for normalization
        adapter = PromptContributorAdapter.from_dict(task_hints=hints)
        return adapter.get_task_type_hints()


# =============================================================================
# Safety Step Handler
# =============================================================================


class SafetyStepHandler(BaseStepHandler):
    """Handler for safety patterns and extensions.

    Collects safety patterns from all safety extensions and
    applies them to the orchestrator.

    This handler is independent because it only modifies safety patterns
    in context and doesn't depend on state from other handlers.
    """

    @property
    def name(self) -> str:
        return "safety"

    @property
    def order(self) -> int:
        return 30

    @property
    def is_independent(self) -> bool:
        """Safety pattern application can run in parallel with other independent handlers."""
        return True

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Apply safety patterns (called from extensions handler)."""
        # Safety patterns are applied through the extensions handler
        # This _do_apply is a no-op; use apply_safety_extensions instead
        pass

    def apply_safety_extensions(
        self,
        orchestrator: Any,
        safety_extensions: List[Any],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Apply safety extensions to orchestrator.

        Args:
            orchestrator: Orchestrator instance
            safety_extensions: List of safety extensions
            context: Vertical context
            result: Result to update
        """
        all_patterns: List[Any] = []
        for ext in safety_extensions:
            all_patterns.extend(ext.get_bash_patterns())
            all_patterns.extend(ext.get_file_patterns())

        context.apply_safety_patterns(all_patterns)
        result.safety_patterns_count = len(all_patterns)

        # Use capability-based approach (protocol-first, SOLID compliant)
        if _check_capability(orchestrator, "vertical_safety_patterns"):
            _invoke_capability(orchestrator, "vertical_safety_patterns", all_patterns)
            logger.debug(f"Applied {len(all_patterns)} safety patterns via capability")
        elif _check_capability(orchestrator, "safety_patterns"):
            _invoke_capability(orchestrator, "safety_patterns", all_patterns)
            logger.debug(
                f"Applied {len(all_patterns)} safety patterns via safety_patterns capability"
            )
        else:
            # Use public method only (SOLID compliant - no private attributes)
            safety_checker = getattr(orchestrator, "safety_checker", None)
            if safety_checker and hasattr(safety_checker, "add_patterns"):
                safety_checker.add_patterns(all_patterns)
                logger.debug(f"Applied {len(all_patterns)} safety patterns via public method")
            else:
                result.add_warning(
                    f"Could not apply {len(all_patterns)} safety patterns; "
                    "patterns stored in context only"
                )


# =============================================================================
# Config Step Handler
# =============================================================================


class ConfigStepHandler(BaseStepHandler):
    """Handler for stages, mode configs, and tool dependencies.

    Applies configuration-related settings from the vertical
    to the orchestrator.
    """

    @property
    def name(self) -> str:
        return "config"

    @property
    def order(self) -> int:
        return 40

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Apply stages configuration from vertical."""
        stages = vertical.get_stages()
        if stages:
            context.apply_stages(stages)
            logger.debug(f"Applied stages: {list(stages.keys())}")

    def apply_mode_config(
        self,
        orchestrator: Any,
        mode_provider: Any,
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Apply mode configuration to orchestrator.

        Args:
            orchestrator: Orchestrator instance
            mode_provider: Mode config provider
            context: Vertical context
            result: Result to update
        """
        mode_configs = mode_provider.get_mode_configs()
        default_mode = mode_provider.get_default_mode()
        default_budget = mode_provider.get_default_tool_budget()

        context.apply_mode_configs(mode_configs, default_mode, default_budget)
        result.mode_configs_count = len(mode_configs)

        # Apply via capability (protocol-first, fallback to direct access)
        if _check_capability(orchestrator, "mode_configs"):
            _invoke_capability(orchestrator, "mode_configs", mode_configs)
            logger.debug(f"Applied {len(mode_configs)} mode configs via capability")
        if _check_capability(orchestrator, "default_budget"):
            _invoke_capability(orchestrator, "default_budget", default_budget)
            logger.debug(f"Applied default budget {default_budget} via capability")

        # Fallback: try adaptive mode controller directly
        if not _check_capability(orchestrator, "mode_configs"):
            controller = getattr(orchestrator, "adaptive_mode_controller", None)
            if controller:
                if hasattr(controller, "set_mode_configs"):
                    controller.set_mode_configs(mode_configs)
                if hasattr(controller, "set_default_budget"):
                    controller.set_default_budget(default_budget)
                logger.debug("Applied mode config via fallback")

    def apply_tool_dependencies(
        self,
        orchestrator: Any,
        dep_provider: Any,
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Apply tool dependencies to orchestrator.

        Args:
            orchestrator: Orchestrator instance
            dep_provider: Tool dependency provider
            context: Vertical context
            result: Result to update
        """
        dependencies = dep_provider.get_dependencies()
        sequences = dep_provider.get_tool_sequences()

        context.apply_tool_dependencies(dependencies, sequences)

        # Apply via capability (protocol-first, fallback to direct access)
        if _check_capability(orchestrator, "tool_dependencies"):
            _invoke_capability(orchestrator, "tool_dependencies", dependencies)
            logger.debug("Applied tool dependencies via capability")
        if _check_capability(orchestrator, "tool_sequences"):
            _invoke_capability(orchestrator, "tool_sequences", sequences)
            logger.debug("Applied tool sequences via capability")

        # If capability not available, log warning (no private attribute fallback)
        if not _check_capability(orchestrator, "tool_dependencies"):
            logger.warning(
                "Orchestrator does not implement tool_dependencies capability; "
                "dependencies stored in context only"
            )


# =============================================================================
# Capability Config Step Handler (SOLID: Centralized Config Storage)
# =============================================================================


class CapabilityConfigStepHandler(BaseStepHandler):
    """Handler for applying capability configs to VerticalContext.

    Replaces direct orchestrator attribute assignment pattern:
    - OLD: orchestrator.rag_config = {...}
    - NEW: context.set_capability_config("rag_config", {...})

    This handler centralizes all vertical capability configurations
    in VerticalContext instead of scattered orchestrator attributes.
    """

    @property
    def name(self) -> str:
        return "capability_config"

    @property
    def order(self) -> int:
        return 5  # Early, before tools (10)

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Apply capability configs from vertical to VerticalContext."""
        configs = self._get_capability_configs(vertical)
        if not configs:
            return

        # Store in VerticalContext
        context.apply_capability_configs(configs)
        result.add_info(f"Applied {len(configs)} capability configs")
        logger.debug(f"Applied capability configs: {list(configs.keys())}")

    def _get_capability_configs(self, vertical: Type["VerticalBase"]) -> Dict[str, Any]:
        """Get capability configs from vertical.

        Args:
            vertical: Vertical class

        Returns:
            Dict of config names to configuration values
        """
        # Try get_capability_configs() method on vertical
        if hasattr(vertical, "get_capability_configs"):
            return vertical.get_capability_configs()

        # Fallback: try importing from capabilities module
        try:
            module = __import__(
                f"victor.{vertical.name}.capabilities", fromlist=["get_capability_configs"]
            )
            if hasattr(module, "get_capability_configs"):
                configs = module.get_capability_configs()
                return configs if isinstance(configs, dict) else {}
        except (ImportError, AttributeError):
            pass

        return {}


# =============================================================================
# Middleware Step Handler
# =============================================================================


class MiddlewareStepHandler(BaseStepHandler):
    """Handler for middleware chain application.

    Applies middleware from the vertical to the orchestrator's
    middleware chain.
    """

    @property
    def name(self) -> str:
        return "middleware"

    @property
    def order(self) -> int:
        return 50

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Apply middleware (called from extensions handler)."""
        # Middleware is applied through the extensions handler
        # This _do_apply is a no-op; use apply_middleware instead
        pass

    def apply_middleware(
        self,
        orchestrator: Any,
        middleware_list: List[Any],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Apply middleware to orchestrator.

        Args:
            orchestrator: Orchestrator instance
            middleware_list: List of middleware
            context: Vertical context
            result: Result to update
        """
        context.apply_middleware(middleware_list)
        result.middleware_count = len(middleware_list)

        # Use capability-based approach (protocol-first, fallback to hasattr)
        if _check_capability(orchestrator, "vertical_middleware"):
            _invoke_capability(orchestrator, "vertical_middleware", middleware_list)
            logger.debug(f"Applied {len(middleware_list)} middleware via capability")
        elif _check_capability(orchestrator, "middleware_chain"):
            # Fallback: try middleware chain
            chain = getattr(orchestrator, "middleware_chain", None) or getattr(
                orchestrator, "_middleware_chain", None
            )
            if chain is not None and hasattr(chain, "add"):
                for mw in middleware_list:
                    chain.add(mw)
                logger.debug(f"Applied {len(middleware_list)} middleware to chain")


# =============================================================================
# Framework Step Handler
# =============================================================================


class FrameworkStepHandler(BaseStepHandler):
    """Handler for workflows, RL config, and team specs.

    Applies framework-level integrations from the vertical
    including workflows, reinforcement learning, and multi-agent teams.

    Phase 2 DI Support:
    - Registries can be injected via constructor for DIP compliance
    - Falls back to hard-coded imports for backward compatibility
    - Allows testing with mock registries without import side effects
    """

    def __init__(
        self,
        workflow_registry: Optional[Any] = None,
        trigger_registry: Optional[Any] = None,
        team_registry: Optional[Any] = None,
        chain_registry: Optional[Any] = None,
        persona_registry: Optional[Any] = None,
        handler_registry: Optional[Any] = None,
    ):
        """Initialize handler with optional registries (DI support).

        Args:
            workflow_registry: Optional workflow registry for DIP compliance
            trigger_registry: Optional trigger registry for DIP compliance
            team_registry: Optional team registry for DIP compliance
            chain_registry: Optional chain registry for DIP compliance
            persona_registry: Optional persona registry for DIP compliance
            handler_registry: Optional handler registry for DIP compliance

        When None, handlers will fall back to hard-coded imports (backward compatible).
        """
        self._workflow_registry = workflow_registry
        self._trigger_registry = trigger_registry
        self._team_registry = team_registry
        self._chain_registry = chain_registry
        self._persona_registry = persona_registry
        self._handler_registry = handler_registry

    @property
    def name(self) -> str:
        return "framework"

    @property
    def order(self) -> int:
        return 60

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Apply framework integrations (workflows, RL, teams, chains, personas, capabilities)."""
        self.apply_workflows(orchestrator, vertical, context, result)
        self.apply_rl_config(orchestrator, vertical, context, result)
        self.apply_team_specs(orchestrator, vertical, context, result)
        # Phase 1: Gap fix - Register chains and personas
        self.apply_chains(orchestrator, vertical, context, result)
        self.apply_personas(orchestrator, vertical, context, result)
        # Phase 1: Gap fix - Wire capability provider to framework
        self.apply_capability_provider(orchestrator, vertical, context, result)
        # Phase 1: Gap fix - Register tool graphs with global registry
        self.apply_tool_graphs(orchestrator, vertical, context, result)
        # Phase 2: Gap fix - Register handlers explicitly
        self.apply_handlers(orchestrator, vertical, context, result)

    def apply_workflows(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Apply workflow definitions from vertical.

        Args:
            orchestrator: Orchestrator instance
            vertical: Vertical class
            context: Vertical context
            result: Result to update
        """
        # Check if vertical has workflow provider using protocol (ISP compliance)
        workflow_provider = None

        # First check if vertical implements WorkflowProviderProtocol
        if isinstance(vertical, WorkflowProviderProtocol):
            workflow_provider = vertical  # type: ignore[unreachable]
        # Fallback to hasattr() for legacy code (deprecated)
        elif hasattr(vertical, "get_workflow_provider"):
            workflow_provider = vertical.get_workflow_provider()
        elif isinstance(vertical, type) and hasattr(vertical, "get_workflows"):
            # Vertical itself is a workflow provider (class-level check)
            workflow_provider = vertical

        if workflow_provider is not None:
            # Get workflows from provider
            workflows = workflow_provider.get_workflows()
            if workflows:
                workflow_count = len(workflows)
                result.workflows_count = workflow_count

                # Store in context
                context.apply_workflows(workflows)

                # Register with workflow registry if available
                try:
                    # Phase 2 DI: Use injected registry or fall back to import
                    if self._workflow_registry is not None:
                        registry = self._workflow_registry
                    else:
                        from victor.workflows.registry import get_global_registry

                        registry = get_global_registry()

                    for name, workflow in workflows.items():
                        registry.register(
                            f"{vertical.name}:{name}",
                            workflow,
                            replace=True,
                        )
                    result.add_info(
                        f"Registered {workflow_count} workflows: " f"{', '.join(workflows.keys())}"
                    )
                except ImportError:
                    result.add_warning("Workflow registry not available")
                except Exception as e:
                    result.add_warning(f"Could not register workflows: {e}")

                # Register workflow triggers with global WorkflowTriggerRegistry
                try:
                    # Phase 2 DI: Use injected registry or fall back to import
                    if self._trigger_registry is not None:
                        trigger_registry = self._trigger_registry
                    else:
                        from victor.workflows.trigger_registry import get_trigger_registry

                        trigger_registry = get_trigger_registry()

                    # Get auto_workflows from provider if available
                    if hasattr(workflow_provider, "get_auto_workflows"):
                        auto_workflows = workflow_provider.get_auto_workflows()
                        if auto_workflows:
                            trigger_registry.register_from_vertical(vertical.name, auto_workflows)
                            logger.debug(
                                f"Registered {len(auto_workflows)} workflow triggers "
                                f"for {vertical.name}"
                            )
                            result.add_info(f"Registered {len(auto_workflows)} workflow triggers")
                except ImportError:
                    # Trigger registry not available (optional dependency)
                    pass
                except Exception as e:
                    result.add_warning(f"Could not register workflow triggers: {e}")

                logger.debug(f"Applied {workflow_count} workflows from vertical")

    def apply_rl_config(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Apply RL configuration from vertical.

        Args:
            orchestrator: Orchestrator instance
            vertical: Vertical class
            context: Vertical context
            result: Result to update
        """
        # Check if vertical has RL config using protocol
        rl_config = None
        rl_provider = None

        # Check for RL config provider method
        if hasattr(vertical, "get_rl_config_provider"):
            rl_provider = vertical.get_rl_config_provider()
            if rl_provider and isinstance(rl_provider, RLConfigProviderProtocol):
                rl_config = rl_provider.get_rl_config()

        # Fallback: check if vertical implements RLConfigProviderProtocol directly
        if rl_config is None and hasattr(vertical, "get_rl_config"):
            rl_config = vertical.get_rl_config()

        if rl_config is None:
            return

        # Get learner count
        learner_count = 0
        if hasattr(rl_config, "active_learners"):
            learner_count = len(rl_config.active_learners)
        result.rl_learners_count = learner_count

        # Store in context
        context.apply_rl_config(rl_config)

        # Apply to RL hooks if vertical provides them
        if hasattr(vertical, "get_rl_hooks"):
            rl_hooks = vertical.get_rl_hooks()
            if rl_hooks:
                context.apply_rl_hooks(rl_hooks)

                # Attach hooks via capability (SOLID compliant)
                if _check_capability(orchestrator, "rl_hooks"):
                    _invoke_capability(orchestrator, "rl_hooks", rl_hooks)
                    logger.debug("Applied RL hooks via capability")
                elif hasattr(orchestrator, "set_rl_hooks"):
                    orchestrator.set_rl_hooks(rl_hooks)
                    logger.debug("Applied RL hooks via set_rl_hooks")
                else:
                    result.add_warning(
                        "Orchestrator lacks set_rl_hooks method; " "hooks stored in context only"
                    )

        result.add_info(f"Configured {learner_count} RL learners")
        logger.debug(f"Applied RL config with {learner_count} learners")

    def apply_team_specs(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Apply team specifications from vertical.

        Args:
            orchestrator: Orchestrator instance
            vertical: Vertical class
            context: Vertical context
            result: Result to update
        """
        # Check if vertical has team specs using protocol
        team_specs = None
        team_provider = None

        # Check for team spec provider method
        if hasattr(vertical, "get_team_spec_provider"):
            team_provider = vertical.get_team_spec_provider()
            if team_provider and isinstance(team_provider, TeamSpecProviderProtocol):
                team_specs = team_provider.get_team_specs()

        # Fallback: check if vertical implements TeamSpecProviderProtocol directly
        if team_specs is None and hasattr(vertical, "get_team_specs"):
            team_specs = vertical.get_team_specs()

        if not team_specs:
            return

        team_count = len(team_specs)
        result.team_specs_count = team_count

        # Store in context
        context.apply_team_specs(team_specs)

        result.add_info(f"Registered {team_count} team specs: " f"{', '.join(team_specs.keys())}")
        # Log each team spec registration (matching workflow DEBUG logging pattern)
        for team_name in team_specs.keys():
            logger.debug(f"Registered team_spec: {team_name}")

        # NEW: Register with global TeamSpecRegistry for cross-vertical discovery
        try:
            # Phase 2 DI: Use injected registry or fall back to import
            if self._team_registry is not None:
                team_registry = self._team_registry
            else:
                from victor.framework.team_registry import get_team_registry

                team_registry = get_team_registry()

            team_registry.register_from_vertical(vertical.name, team_specs, replace=True)
            logger.debug(f"Registered {team_count} teams with global registry for {vertical.name}")
        except ImportError:
            # Team registry not available (optional dependency)
            pass
        except Exception as e:
            result.add_warning(f"Could not register with team registry: {e}")

        logger.debug(f"Applied {team_count} team specifications from vertical")

    def _get_step_details(self, result: "IntegrationResult") -> Optional[Dict[str, Any]]:
        """Return framework integration details."""
        details = {}
        if result.workflows_count > 0:
            details["workflows_count"] = result.workflows_count
        if result.rl_learners_count > 0:
            details["rl_learners_count"] = result.rl_learners_count
        if result.team_specs_count > 0:
            details["team_specs_count"] = result.team_specs_count
        return details if details else None

    def apply_chains(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Register composed chains from vertical with ChainRegistry.

        Phase 1: Gap fix - Chains were defined but never registered.

        Args:
            orchestrator: Orchestrator instance
            vertical: Vertical class
            context: Vertical context
            result: Result to update
        """
        # Check if vertical provides chains (supports both get_chains and get_composed_chains)
        chains = None
        if hasattr(vertical, "get_chains"):
            chains = vertical.get_chains()
        elif hasattr(vertical, "get_composed_chains"):
            chains = vertical.get_composed_chains()
        if not chains:
            return

        chain_count = len(chains)

        # Register with ChainRegistry
        try:
            # Phase 2 DI: Use injected registry or fall back to import
            if self._chain_registry is not None:
                registry = self._chain_registry
            else:
                from victor.framework.chain_registry import get_chain_registry

                registry = get_chain_registry()

            for name, chain in chains.items():
                full_name = f"{vertical.name}:{name}"
                registry.register(
                    name,
                    chain,
                    vertical=vertical.name,
                    replace=True,
                )
                logger.debug(f"Registered chain: {full_name}")

            result.add_info(f"Registered {chain_count} chains: {', '.join(chains.keys())}")
            logger.debug(f"Applied {chain_count} chains from vertical")
        except ImportError:
            result.add_warning("ChainRegistry not available")
        except Exception as e:
            result.add_warning(f"Could not register chains: {e}")

    def apply_personas(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Register personas from vertical with PersonaRegistry.

        Phase 1: Gap fix - Personas were defined but never registered.

        Args:
            orchestrator: Orchestrator instance
            vertical: Vertical class
            context: Vertical context
            result: Result to update
        """
        # Check if vertical provides personas
        if not hasattr(vertical, "get_personas"):
            return

        personas = vertical.get_personas()
        if not personas:
            return

        persona_count = len(personas)

        # Register with PersonaRegistry
        try:
            # Phase 2 DI: Use injected registry or fall back to import
            if self._persona_registry is not None:
                registry = self._persona_registry
            else:
                from victor.framework.persona_registry import get_persona_registry

                registry = get_persona_registry()

            for name, persona in personas.items():
                # Convert to PersonaSpec if needed
                if hasattr(persona, "to_persona_spec"):
                    spec = persona.to_persona_spec()
                elif hasattr(persona, "to_dict"):
                    # Create from dict
                    from victor.framework.persona_registry import PersonaSpec

                    data = persona.to_dict()
                    spec = PersonaSpec(
                        name=data.get("name", name),
                        role=data.get("role", ""),
                        expertise=data.get("expertise", []),
                        communication_style=data.get("communication_style", ""),
                        behavioral_traits=data.get("behavioral_traits", []),
                        vertical=vertical.name,
                    )
                else:
                    # Use as-is if it's already a PersonaSpec
                    spec = persona

                full_name = f"{vertical.name}:{name}"
                registry.register(name, spec, vertical=vertical.name, replace=True)
                logger.debug(f"Registered persona: {full_name}")

            result.add_info(f"Registered {persona_count} personas: {', '.join(personas.keys())}")
            logger.debug(f"Applied {persona_count} personas from vertical")
        except ImportError:
            result.add_warning("PersonaRegistry not available")
        except Exception as e:
            result.add_warning(f"Could not register personas: {e}")

    def apply_capability_provider(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Wire capability provider from vertical to framework.

        Phase 1: Gap fix - Capability providers were defined but never wired.

        The capability provider enables dynamic capability loading for
        runtime extension of vertical-specific functionality.

        Args:
            orchestrator: Orchestrator instance
            vertical: Vertical class
            context: Vertical context
            result: Result to update
        """
        # Check if vertical provides a capability provider
        if not hasattr(vertical, "get_capability_provider"):
            return

        provider = vertical.get_capability_provider()
        if provider is None:
            return

        # Count capabilities from provider
        cap_count = 0
        if hasattr(provider, "get_capabilities"):
            caps = provider.get_capabilities()
            cap_count = len(caps) if caps else 0
        elif hasattr(provider, "CAPABILITIES"):
            cap_count = len(provider.CAPABILITIES)

        # Wire to CapabilityLoader if available
        try:
            from victor.framework.capability_loader import CapabilityLoader

            # Get or create loader
            loader = None
            if hasattr(orchestrator, "_capability_loader"):
                loader = orchestrator._capability_loader
            else:
                loader = CapabilityLoader()
                # Store for reuse
                if hasattr(orchestrator, "__dict__"):
                    orchestrator._capability_loader = loader

            # Load capabilities from provider
            if hasattr(provider, "get_capabilities"):
                for cap in provider.get_capabilities():
                    if hasattr(cap, "name") and hasattr(cap, "handler"):
                        cap_type = getattr(cap, "capability_type", None)
                        loader.register_capability(
                            name=cap.name,
                            handler=cap.handler,
                            capability_type=cap_type if cap_type is not None else "custom",  # type: ignore[arg-type]
                            version=getattr(cap, "version", "1.0"),
                        )

            # Apply loaded capabilities to orchestrator
            loader.apply_to(orchestrator)

            result.add_info(f"Wired capability provider with {cap_count} capabilities")
            logger.debug(
                f"Applied capability provider from vertical={vertical.name}, "
                f"capabilities={cap_count}"
            )
        except ImportError:
            result.add_warning("CapabilityLoader not available")
        except Exception as e:
            result.add_warning(f"Could not wire capability provider: {e}")
            logger.debug(f"Capability provider error: {e}", exc_info=True)

    def apply_tool_graphs(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Register vertical tool graphs with global ToolGraphRegistry.

        Phase 1: Gap fix - Tool graphs were defined but never registered
        globally, preventing cross-vertical tool planning.

        Args:
            orchestrator: Orchestrator instance
            vertical: Vertical class
            context: Vertical context
            result: Result to update
        """
        try:
            from victor.tools.tool_graph import ToolGraphRegistry
        except ImportError:
            # ToolGraphRegistry not available (optional dependency)
            return

        graph_registry = ToolGraphRegistry.get_instance()
        graph = None

        # Check for direct tool graph method
        if hasattr(vertical, "get_tool_graph"):
            graph = vertical.get_tool_graph()

        # Fallback: try to build from tool dependency provider
        if graph is None and hasattr(vertical, "get_tool_dependency_provider"):
            provider = vertical.get_tool_dependency_provider()
            if provider and hasattr(provider, "get_tool_graph"):
                graph = provider.get_tool_graph()

        if graph is None:
            return

        try:
            graph_registry.register_graph(vertical.name, graph)
            result.add_info(f"Registered tool graph for {vertical.name}")
            logger.debug(f"Registered tool graph for vertical={vertical.name}")
        except Exception as e:
            result.add_warning(f"Could not register tool graph: {e}")

    def apply_handlers(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Explicitly register vertical compute handlers.

        SOLID: Replaces import-side-effect registration pattern with explicit
        registration via vertical.get_handlers(). This makes registration
        traceable and testable.

        Args:
            orchestrator: Orchestrator instance
            vertical: Vertical class
            context: Vertical context
            result: Result to update
        """
        # Check if vertical provides handlers (NEW: use get_handlers())
        if not hasattr(vertical, "get_handlers"):
            return

        handlers = vertical.get_handlers()
        if not handlers:
            return

        handler_count = len(handlers)

        try:
            # Phase 2 DI: Use injected registry or fall back to import
            if self._handler_registry is not None:
                registry = self._handler_registry
            else:
                from victor.framework.handler_registry import get_handler_registry

                registry = get_handler_registry()

            for name, handler in handlers.items():
                registry.register(name, handler, vertical=vertical.name, replace=True)
                logger.debug(f"Registered handler: {vertical.name}:{name}")

            # Sync to executor for backward compatibility
            try:
                registry.sync_with_executor(direction="to_executor", replace=True)
            except Exception:
                pass  # Sync is optional

            result.add_info(f"Registered {handler_count} handlers: {', '.join(handlers.keys())}")
            logger.debug(f"Applied {handler_count} handlers from vertical")
        except ImportError:
            # Handler registry not available - fall back to executor registration
            try:
                from victor.workflows.executor import register_compute_handler

                for name, handler in handlers.items():
                    register_compute_handler(name, handler)
                    logger.debug(f"Registered handler via executor: {name}")

                result.add_info(
                    f"Registered {handler_count} handlers (fallback): "
                    f"{', '.join(handlers.keys())}"
                )
            except ImportError:
                result.add_warning("Handler registration not available")
        except Exception as e:
            result.add_warning(f"Could not register handlers: {e}")


# =============================================================================
# Tiered Config Step Handler (Phase 1: Gap Fix)
# =============================================================================


class TieredConfigStepHandler(BaseStepHandler):
    """Handler for tiered tool configuration application.

    Extracts TieredToolConfig from the vertical and applies it to:
    1. VerticalContext for storage
    2. ToolAccessController.VerticalLayer for access filtering

    This handler closes the gap where TieredToolConfig was defined
    in verticals but never applied to the tool access system.
    """

    @property
    def name(self) -> str:
        return "tiered_config"

    @property
    def order(self) -> int:
        return 15  # After tools (10), before prompt (20)

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Apply tiered tool config from vertical.

        Uses get_tiered_config() helper with fallback chain:
        1. Try get_tiered_tool_config() first
        2. Fall back to get_tiered_tools() if config missing or None
        3. Return early if neither method provides config
        """
        # Use fallback chain helper (Workstream D fix)
        tiered_config = get_tiered_config(vertical)

        if tiered_config is None:
            logger.debug(f"Vertical {vertical.name} does not provide tiered tool config")
            return

        # Store in context
        context.apply_tiered_config(tiered_config)

        # Get tool counts for logging
        mandatory_count = len(tiered_config.mandatory) if tiered_config.mandatory else 0
        core_count = len(tiered_config.vertical_core) if tiered_config.vertical_core else 0
        pool_count = len(tiered_config.semantic_pool) if tiered_config.semantic_pool else 0

        logger.debug(
            f"Applied tiered config: mandatory={mandatory_count}, "
            f"core={core_count}, pool={pool_count}"
        )

        # Apply via capability (protocol-first, SOLID compliant)
        if _check_capability(orchestrator, "tiered_tool_config"):
            _invoke_capability(orchestrator, "tiered_tool_config", tiered_config)
            logger.debug("Applied tiered config via capability")
            return

        # Fallback: try to set on ToolAccessController directly (ISP-compliant)
        tool_access_controller = None

        # Use capability-first approach for getting tool_access_controller
        if _check_capability(orchestrator, "tool_access_controller"):
            try:
                if isinstance(orchestrator, CapabilityRegistryProtocol):
                    tool_access_controller = orchestrator.get_capability_value(
                        "tool_access_controller"
                    )
            except (KeyError, TypeError):
                # Fall through to hasattr
                pass

        # Fallback to hasattr with deprecation warning
        if tool_access_controller is None and hasattr(orchestrator, "tool_access_controller"):
            warnings.warn(
                f"Orchestrator {type(orchestrator).__name__} exposes tool_access_controller "
                f"as an attribute but not via CapabilityRegistryProtocol. "
                f"Please register tool_access_controller as a capability for ISP compliance.",
                DeprecationWarning,
                stacklevel=2,
            )
            tool_access_controller = orchestrator.tool_access_controller

        if tool_access_controller is not None:
            # Use callable() check instead of hasattr (better practice)
            set_tiered_config = getattr(tool_access_controller, "set_tiered_config", None)
            if callable(set_tiered_config):
                tool_access_controller.set_tiered_config(tiered_config)
                logger.debug("Applied tiered config to ToolAccessController")
            else:
                result.add_warning(
                    "ToolAccessController lacks set_tiered_config method; "
                    "config stored in context only"
                )
        else:
            result.add_warning(
                "Orchestrator lacks tool_access_controller; " "tiered config stored in context only"
            )


# =============================================================================
# Context Step Handler
# =============================================================================


class ContextStepHandler(BaseStepHandler):
    """Handler for attaching the vertical context to the orchestrator.

    This is typically the final step in the pipeline.
    """

    @property
    def name(self) -> str:
        return "context"

    @property
    def order(self) -> int:
        return 100  # Last step

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Attach the vertical context to orchestrator."""
        # Use capability-based approach (SOLID compliant)
        if _check_capability(orchestrator, "vertical_context"):
            _invoke_capability(orchestrator, "vertical_context", context)
            logger.debug("Attached context via capability")
        else:
            result.add_warning(
                "Orchestrator lacks vertical_context capability; "
                "context not attached. Ensure CapabilityRegistry is available."
            )


# =============================================================================
# Extension Handler Registry (Phase 2: OCP Compliance)
# =============================================================================


@dataclass
class ExtensionHandler:
    """Handler for a specific extension type.

    Provides OCP-compliant extension handling where new extension types
    can be added without modifying the ExtensionsStepHandler.

    Attributes:
        name: Extension type name (matches VerticalExtensions field)
        handler: Callable that applies the extension
        priority: Execution order (lower runs first)
    """

    name: str
    handler: Callable[[Any, Any, Any, "VerticalContext", "IntegrationResult"], None]
    priority: int = 50


class ExtensionHandlerRegistry:
    """Registry for extension handlers (OCP pattern).

    Allows new extension types to be registered without modifying
    existing code, following the Open/Closed Principle.

    Usage:
        registry = ExtensionHandlerRegistry.default()
        registry.register(ExtensionHandler(
            name="custom_extension",
            handler=my_handler,
            priority=60,
        ))
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, ExtensionHandler] = {}

    def register(self, handler: ExtensionHandler, replace: bool = False) -> None:
        """Register an extension handler.

        Args:
            handler: ExtensionHandler to register
            replace: If True, replace existing handler
        """
        if handler.name in self._handlers and not replace:
            logger.warning(f"Extension handler '{handler.name}' already registered")
            return
        self._handlers[handler.name] = handler
        logger.debug(f"Registered extension handler: {handler.name}")

    def unregister(self, name: str) -> bool:
        """Unregister an extension handler.

        Args:
            name: Handler name to remove

        Returns:
            True if handler was removed
        """
        if name in self._handlers:
            del self._handlers[name]
            return True
        return False

    def get_ordered_handlers(self) -> List[ExtensionHandler]:
        """Get handlers in priority order."""
        return sorted(self._handlers.values(), key=lambda h: h.priority)

    def apply_all(
        self,
        orchestrator: Any,
        extensions: Any,
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Apply all registered extension handlers.

        Args:
            orchestrator: Orchestrator instance
            extensions: VerticalExtensions instance
            context: Vertical context
            result: Integration result
        """
        for handler in self.get_ordered_handlers():
            # Get extension value by name
            ext_value = getattr(extensions, handler.name, None)
            if ext_value is not None:
                try:
                    handler.handler(orchestrator, ext_value, extensions, context, result)
                except Exception as e:
                    result.add_warning(f"Extension handler '{handler.name}' failed: {e}")
                    logger.debug(f"Extension handler error: {e}", exc_info=True)

    @classmethod
    def default(cls) -> "ExtensionHandlerRegistry":
        """Create registry with default handlers.

        Returns:
            Registry with standard extension handlers
        """
        registry = cls()
        # Handlers are registered by ExtensionsStepHandler
        return registry


# =============================================================================
# Extensions Step Handler
# =============================================================================


class ExtensionsStepHandler(BaseStepHandler):
    """Handler for all vertical extensions.

    Coordinates the application of middleware, safety, prompts,
    mode config, and tool dependencies from the vertical extensions.

    OCP Compliance (Phase 2):
    Uses ExtensionHandlerRegistry for extensible handler registration.
    New extension types can be added via registry.register() without
    modifying this class.
    """

    def __init__(self) -> None:
        """Initialize with sub-handlers and registry."""
        self._middleware_handler = MiddlewareStepHandler()
        self._safety_handler = SafetyStepHandler()
        self._prompt_handler = PromptStepHandler()
        self._config_handler = ConfigStepHandler()
        # OCP: Extension handler registry for pluggable handlers
        self._extension_registry = ExtensionHandlerRegistry()
        self._register_default_extension_handlers()

    @property
    def name(self) -> str:
        return "extensions"

    @property
    def order(self) -> int:
        return 45  # After config, before middleware

    @property
    def extension_registry(self) -> ExtensionHandlerRegistry:
        """Get the extension handler registry for OCP extension."""
        return self._extension_registry

    def _register_default_extension_handlers(self) -> None:
        """Register default extension handlers (OCP pattern).

        These handlers implement the core extension processing. Additional
        handlers can be registered via self.extension_registry.register().
        """
        # Note: Default handlers use the existing sub-handler methods
        # This provides backward compatibility while enabling OCP extension

        def handle_middleware(
            orchestrator: Any,
            middleware: List[Any],
            extensions: Any,
            context: "VerticalContext",
            result: "IntegrationResult",
        ) -> None:
            # Check for middleware_profile first (profile-based configuration)
            profile_name = getattr(extensions, "middleware_profile", None)
            middleware_overrides = getattr(extensions, "middleware_overrides", None)

            if profile_name is not None:
                # Load middleware from profile
                from victor.framework.middleware_profiles.profiles import MiddlewareProfiles

                profile_method = getattr(MiddlewareProfiles, f"{profile_name}_profile", None)
                if profile_method is None:
                    result.add_warning(f"Unknown middleware profile: {profile_name}")
                    # Fall back to direct middleware list
                    self._middleware_handler.apply_middleware(orchestrator, middleware, context, result)
                    return

                profile = profile_method()
                combined_middleware = list(profile.middlewares)

                # Apply overrides if present
                if middleware_overrides:
                    # For now, just append overrides (future: implement merge logic)
                    combined_middleware.extend(middleware_overrides)

                self._middleware_handler.apply_middleware(orchestrator, combined_middleware, context, result)
                logger.debug(f"Applied middleware profile '{profile_name}' with {len(combined_middleware)} middleware")
            else:
                # Use direct middleware list (backward compatible)
                self._middleware_handler.apply_middleware(orchestrator, middleware, context, result)

        def handle_safety(
            orchestrator: Any,
            safety_extensions: List[Any],
            extensions: Any,
            context: "VerticalContext",
            result: "IntegrationResult",
        ) -> None:
            self._safety_handler.apply_safety_extensions(
                orchestrator, safety_extensions, context, result
            )

        def handle_prompts(
            orchestrator: Any,
            contributors: List[Any],
            extensions: Any,
            context: "VerticalContext",
            result: "IntegrationResult",
        ) -> None:
            self._prompt_handler.apply_contributors(orchestrator, contributors, context, result)

        def handle_mode_config(
            orchestrator: Any,
            provider: Any,
            extensions: Any,
            context: "VerticalContext",
            result: "IntegrationResult",
        ) -> None:
            self._config_handler.apply_mode_config(orchestrator, provider, context, result)

        def handle_tool_deps(
            orchestrator: Any,
            provider: Any,
            extensions: Any,
            context: "VerticalContext",
            result: "IntegrationResult",
        ) -> None:
            self._config_handler.apply_tool_dependencies(orchestrator, provider, context, result)

        def handle_enrichment(
            orchestrator: Any,
            strategy: Any,
            extensions: Any,
            context: "VerticalContext",
            result: "IntegrationResult",
        ) -> None:
            self._apply_enrichment_strategy(orchestrator, strategy, context, result)

        def handle_tool_selection(
            orchestrator: Any,
            strategy: Any,
            extensions: Any,
            context: "VerticalContext",
            result: "IntegrationResult",
        ) -> None:
            self._apply_tool_selection_strategy(orchestrator, strategy, context, result)

        def handle_service_provider(
            orchestrator: Any,
            provider: Any,
            extensions: Any,
            context: "VerticalContext",
            result: "IntegrationResult",
        ) -> None:
            """Register vertical-specific services with the DI container."""
            # Get container and settings from orchestrator
            container = getattr(orchestrator, "_container", None)
            settings = getattr(orchestrator, "settings", None)

            if container is None:
                result.add_warning(
                    "Cannot register vertical services: orchestrator lacks container"
                )
                return

            try:
                provider.register_services(container, settings)
                # Count registered services if method available
                required = (
                    provider.get_required_services()
                    if hasattr(provider, "get_required_services")
                    else []
                )
                optional = (
                    provider.get_optional_services()
                    if hasattr(provider, "get_optional_services")
                    else []
                )
                total = len(required) + len(optional)
                result.add_info(
                    f"Registered {total} vertical services ({len(required)} required, {len(optional)} optional)"
                )
                logger.debug(f"Registered vertical services: {total} total")
            except Exception as e:
                result.add_warning(f"Failed to register vertical services: {e}")
                logger.debug(f"Service registration error: {e}", exc_info=True)

        # Register default handlers with priorities
        # Service provider first (priority=5) so services are available to other handlers
        self._extension_registry.register(
            ExtensionHandler("service_provider", handle_service_provider, priority=5)
        )
        self._extension_registry.register(
            ExtensionHandler("middleware", handle_middleware, priority=10)
        )
        self._extension_registry.register(
            ExtensionHandler("safety_extensions", handle_safety, priority=20)
        )
        self._extension_registry.register(
            ExtensionHandler("prompt_contributors", handle_prompts, priority=30)
        )
        self._extension_registry.register(
            ExtensionHandler("mode_config_provider", handle_mode_config, priority=40)
        )
        self._extension_registry.register(
            ExtensionHandler("tool_dependency_provider", handle_tool_deps, priority=50)
        )
        self._extension_registry.register(
            ExtensionHandler("enrichment_strategy", handle_enrichment, priority=60)
        )
        self._extension_registry.register(
            ExtensionHandler("tool_selection_strategy", handle_tool_selection, priority=15)
        )

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Apply all vertical extensions via registry (OCP pattern).

        Uses ExtensionHandlerRegistry to apply extensions, allowing
        new extension types to be added without modifying this method.
        """
        extensions = vertical.get_extensions()
        if extensions is not None:
            # OCP: Apply all registered extension handlers
            self._extension_registry.apply_all(orchestrator, extensions, context, result)
        else:
            logger.debug("No extensions available for vertical")  # type: ignore[unreachable]

    def _get_step_details(self, result: "IntegrationResult") -> Optional[Dict[str, Any]]:
        """Return extension application details."""
        details = {}
        if result.middleware_count > 0:
            details["middleware_count"] = result.middleware_count
        if result.safety_patterns_count > 0:
            details["safety_patterns_count"] = result.safety_patterns_count
        if result.prompt_hints_count > 0:
            details["prompt_hints_count"] = result.prompt_hints_count
        if result.mode_configs_count > 0:
            details["mode_configs_count"] = result.mode_configs_count
        return details if details else None

    def _apply_tool_selection_strategy(
        self,
        orchestrator: Any,
        strategy: Any,
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Apply tool selection strategy from vertical extensions.

        Registers the vertical-specific tool selection strategy with the
        tool selection system, enabling domain-aware tool prioritization.

        Args:
            orchestrator: Orchestrator instance
            strategy: ToolSelectionStrategyProtocol implementation
            context: Vertical context
            result: Result to update
        """
        # Store in context for later retrieval
        context.apply_tool_selection_strategy(strategy)

        # Try to register with tool selector via capability
        if _check_capability(orchestrator, "tool_selection_strategy"):
            _invoke_capability(orchestrator, "tool_selection_strategy", strategy)
            logger.debug("Applied tool selection strategy via capability")
            return

        # Fallback: try to set on tool selector directly
        tool_selector = getattr(orchestrator, "tool_selector", None)
        if tool_selector is not None:
            if hasattr(tool_selector, "set_vertical_strategy"):
                tool_selector.set_vertical_strategy(strategy)
                logger.debug("Applied tool selection strategy to tool_selector")
                return
            elif hasattr(tool_selector, "register_strategy"):
                vertical_name = context.vertical_name or "default"
                tool_selector.register_strategy(vertical_name, strategy)
                logger.debug(f"Registered tool selection strategy for vertical={vertical_name}")
                return

        # Log that strategy is stored in context only
        logger.debug(
            "Tool selection strategy stored in context only; "
            "orchestrator lacks tool_selector with strategy support"
        )

    def _apply_enrichment_strategy(
        self,
        orchestrator: Any,
        strategy: Any,
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Apply enrichment strategy from vertical extensions.

        Stores the strategy in context and registers it with the
        PromptEnrichmentService if available.

        Args:
            orchestrator: Orchestrator instance
            strategy: EnrichmentStrategyProtocol implementation
            context: Vertical context
            result: Result to update
        """
        # Store in context
        context.apply_enrichment_strategy(strategy)

        # Try to register with enrichment service via capability
        if _check_capability(orchestrator, "enrichment_service"):
            try:
                service = getattr(orchestrator, "enrichment_service", None)
                if service and hasattr(service, "register_strategy"):
                    vertical_name = context.vertical_name or "default"
                    service.register_strategy(vertical_name, strategy)
                    logger.debug(f"Registered enrichment strategy for vertical={vertical_name}")
                    return
            except Exception as e:
                logger.debug(f"Failed to register enrichment strategy: {e}")

        # Fallback: try to invoke via capability
        if _check_capability(orchestrator, "enrichment_strategy"):
            _invoke_capability(orchestrator, "enrichment_strategy", strategy)
            logger.debug("Applied enrichment strategy via capability")
        else:
            result.add_warning(
                "Enrichment strategy stored in context only; "
                "orchestrator lacks enrichment_service capability"
            )


# =============================================================================
# Step Handler Registry
# =============================================================================


@dataclass
class StepHandlerRegistry:
    """Registry of step handlers for the pipeline.

    Maintains an ordered collection of step handlers that can be
    extended with custom handlers.

    Attributes:
        handlers: List of registered step handlers
    """

    handlers: List[StepHandlerProtocol]

    @classmethod
    def default(cls) -> "StepHandlerRegistry":
        """Create a registry with default handlers.

        Returns:
            StepHandlerRegistry with standard handlers
        """
        return cls(
            handlers=[
                CapabilityConfigStepHandler(),  # SOLID: Centralized config storage
                ToolStepHandler(),
                TieredConfigStepHandler(),  # Phase 1: Gap fix
                PromptStepHandler(),
                ConfigStepHandler(),
                ExtensionsStepHandler(),
                FrameworkStepHandler(),
                ContextStepHandler(),
            ]
        )

    def add_handler(self, handler: StepHandlerProtocol) -> None:
        """Add a handler to the registry.

        Args:
            handler: Step handler to add
        """
        self.handlers.append(handler)
        self._sort_handlers()

    def remove_handler(self, name: str) -> bool:
        """Remove a handler by name.

        Args:
            name: Name of handler to remove

        Returns:
            True if handler was found and removed
        """
        for i, handler in enumerate(self.handlers):
            if handler.name == name:
                self.handlers.pop(i)
                return True
        return False

    def get_handler(self, name: str) -> Optional[StepHandlerProtocol]:
        """Get a handler by name.

        Args:
            name: Name of handler to find

        Returns:
            Handler or None if not found
        """
        for handler in self.handlers:
            if handler.name == name:
                return handler
        return None

    def get_ordered_handlers(self) -> List[StepHandlerProtocol]:
        """Get handlers in execution order.

        Returns:
            List of handlers sorted by order
        """
        return sorted(self.handlers, key=lambda h: h.order)

    def _sort_handlers(self) -> None:
        """Sort handlers by order."""
        self.handlers.sort(key=lambda h: h.order)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Protocol
    "StepHandlerProtocol",
    # Base class
    "BaseStepHandler",
    # Concrete handlers
    "ToolStepHandler",
    "TieredConfigStepHandler",  # Phase 1: Gap fix
    "CapabilityConfigStepHandler",  # SOLID: Centralized config storage
    "PromptStepHandler",
    "SafetyStepHandler",
    "ConfigStepHandler",
    "MiddlewareStepHandler",
    "FrameworkStepHandler",
    "ContextStepHandler",
    "ExtensionsStepHandler",
    # Registries
    "StepHandlerRegistry",
    "ExtensionHandler",  # Phase 2: OCP
    "ExtensionHandlerRegistry",  # Phase 2: OCP
    # Capability helpers
    "_check_capability",
    "_invoke_capability",
    # Tiered config helper (Workstream D)
    "get_tiered_config",
]
