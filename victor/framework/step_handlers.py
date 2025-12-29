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

This module provides focused step handler classes that implement
individual integration steps. Each handler follows the Single
Responsibility Principle, handling one specific concern.

Design Philosophy:
- Single Responsibility: Each handler handles one concern
- Open/Closed: New steps can be added by creating new handlers
- Liskov Substitution: All handlers implement StepHandlerProtocol
- Interface Segregation: Minimal protocol for step handlers
- Dependency Inversion: Handlers depend on protocols, not implementations

Architecture:
    StepHandlerProtocol
    ├── ToolStepHandler - Tool filter application
    ├── PromptStepHandler - System prompt and prompt contributors
    ├── SafetyStepHandler - Safety patterns and extensions
    ├── ConfigStepHandler - Stages, modes, and tool dependencies
    ├── MiddlewareStepHandler - Middleware chain application
    └── FrameworkStepHandler - Workflows, RL config, and team specs

Usage:
    from victor.framework.step_handlers import (
        ToolStepHandler,
        PromptStepHandler,
        SafetyStepHandler,
        ConfigStepHandler,
        MiddlewareStepHandler,
        FrameworkStepHandler,
    )

    # Create handlers
    handlers = [
        ToolStepHandler(),
        PromptStepHandler(),
        SafetyStepHandler(),
        ConfigStepHandler(),
        MiddlewareStepHandler(),
        FrameworkStepHandler(),
    ]

    # Apply in pipeline
    for handler in handlers:
        handler.apply(orchestrator, vertical, context, result)
"""

from __future__ import annotations

import logging
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
    from victor.agent.vertical_context import VerticalContext
    from victor.framework.vertical_integration import IntegrationResult
    from victor.verticals.base import VerticalBase
    from victor.verticals.protocols import (
        MiddlewareProtocol,
        ModeConfigProviderProtocol,
        PromptContributorProtocol,
        RLConfigProviderProtocol,
        SafetyExtensionProtocol,
        TeamSpecProviderProtocol,
        ToolDependencyProviderProtocol,
        VerticalExtensions,
        WorkflowProviderProtocol,
    )

# Import protocols for runtime isinstance checks
from victor.verticals.protocols import (
    RLConfigProviderProtocol,
    TeamSpecProviderProtocol,
    WorkflowProviderProtocol,
)

from victor.framework.protocols import CapabilityRegistryProtocol


logger = logging.getLogger(__name__)


# =============================================================================
# Capability Helpers (moved from vertical_integration.py for reuse)
# =============================================================================


def _check_capability(obj: Any, capability_name: str) -> bool:
    """Check if object has capability via registry.

    Uses protocol-based capability discovery. Orchestrators must implement
    CapabilityRegistryProtocol for proper capability checking.

    SOLID Compliance:
    - Uses protocol, not hasattr (DIP - Dependency Inversion)
    - No private attribute access (SRP - Single Responsibility)

    Args:
        obj: Object to check (should implement CapabilityRegistryProtocol)
        capability_name: Name of capability

    Returns:
        True if capability is available via the registry
    """
    # Check capability registry (protocol-based only)
    if isinstance(obj, CapabilityRegistryProtocol):
        return obj.has_capability(capability_name)

    # For objects not implementing protocol, check for public method
    # Note: No private attribute fallbacks (SOLID compliant)
    public_methods = {
        "enabled_tools": "set_enabled_tools",
        "prompt_builder": "prompt_builder",
        "vertical_middleware": "apply_vertical_middleware",
        "vertical_safety_patterns": "apply_vertical_safety_patterns",
        "vertical_context": "set_vertical_context",
        "adaptive_mode_controller": "adaptive_mode_controller",
    }

    method_name = public_methods.get(capability_name, capability_name)
    return hasattr(obj, method_name) and (
        callable(getattr(obj, method_name, None))
        or not callable(getattr(obj, method_name, None))  # Allow properties
    )


def _invoke_capability(obj: Any, capability_name: str, *args: Any, **kwargs: Any) -> Any:
    """Invoke capability via registry or public method call.

    Uses protocol-based capability invocation. Orchestrators must implement
    CapabilityRegistryProtocol for proper capability invocation.

    SOLID Compliance:
    - Uses protocol, not private attribute assignment (DIP)
    - No direct private attribute writes (SRP)

    Args:
        obj: Object to invoke capability on (should implement CapabilityRegistryProtocol)
        capability_name: Name of capability
        *args: Arguments for capability
        **kwargs: Keyword arguments for capability

    Returns:
        Result of capability invocation

    Raises:
        AttributeError: If capability cannot be invoked
    """
    # Use capability registry if available (preferred)
    if isinstance(obj, CapabilityRegistryProtocol):
        try:
            return obj.invoke_capability(capability_name, *args, **kwargs)
        except (KeyError, TypeError) as e:
            logger.debug(f"Registry invoke failed for {capability_name}: {e}")
            # Fall through to public method fallback

    # Fallback: use public method mappings only (no private attributes)
    public_method_mappings = {
        "enabled_tools": "set_enabled_tools",
        "vertical_middleware": "apply_vertical_middleware",
        "vertical_safety_patterns": "apply_vertical_safety_patterns",
        "vertical_context": "set_vertical_context",
        "rl_hooks": "set_rl_hooks",
        "team_specs": "set_team_specs",
        "mode_configs": "set_mode_configs",
        "default_budget": "set_default_budget",
        "tool_dependencies": "set_tool_dependencies",
        "tool_sequences": "set_tool_sequences",
        "custom_prompt": "set_custom_prompt",
        "task_type_hints": "set_task_type_hints",
        "prompt_section": "add_prompt_section",
        "safety_patterns": "add_safety_patterns",
    }

    method_name = public_method_mappings.get(capability_name, f"set_{capability_name}")
    method = getattr(obj, method_name, None)
    if callable(method):
        return method(*args, **kwargs)

    # No private attribute fallback - raise clear error instead
    raise AttributeError(
        f"Cannot invoke capability '{capability_name}' on {type(obj).__name__}. "
        f"Expected method '{method_name}' not found. "
        f"Object should implement CapabilityRegistryProtocol."
    )


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
    error handling and logging.

    Subclasses must implement:
    - name property
    - order property
    - _do_apply() method
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

    def apply(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
        strict_mode: bool = False,
    ) -> None:
        """Apply this step with error handling.

        Args:
            orchestrator: Orchestrator instance
            vertical: Vertical class
            context: Vertical context
            result: Integration result
            strict_mode: If True, fail on any error
        """
        try:
            self._do_apply(orchestrator, vertical, context, result)
        except Exception as e:
            if strict_mode:
                result.add_error(f"{self.name} failed: {e}")
            else:
                result.add_warning(f"{self.name} error: {e}")
            logger.debug(f"Step {self.name} failed: {e}", exc_info=True)

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
    """

    @property
    def name(self) -> str:
        return "tools"

    @property
    def order(self) -> int:
        return 10

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


# =============================================================================
# Prompt Step Handler
# =============================================================================


class PromptStepHandler(BaseStepHandler):
    """Handler for system prompt and prompt contributors.

    Applies the system prompt from the vertical and merges
    task hints from prompt contributors.
    """

    @property
    def name(self) -> str:
        return "prompt"

    @property
    def order(self) -> int:
        return 20

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

        Args:
            orchestrator: Orchestrator instance
            contributors: List of prompt contributors
            context: Vertical context
            result: Result to update
        """
        # Merge task hints from all contributors
        merged_hints: Dict[str, Any] = {}
        for contributor in sorted(contributors, key=lambda c: c.get_priority()):
            merged_hints.update(contributor.get_task_type_hints())

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


# =============================================================================
# Safety Step Handler
# =============================================================================


class SafetyStepHandler(BaseStepHandler):
    """Handler for safety patterns and extensions.

    Collects safety patterns from all safety extensions and
    applies them to the orchestrator.
    """

    @property
    def name(self) -> str:
        return "safety"

    @property
    def order(self) -> int:
        return 30

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

        # Fallback: try tool sequence tracker directly
        if not _check_capability(orchestrator, "tool_dependencies"):
            tracker = getattr(orchestrator, "tool_sequence_tracker", None) or getattr(
                orchestrator, "_sequence_tracker", None
            )
            if tracker:
                if hasattr(tracker, "set_dependencies"):
                    tracker.set_dependencies(dependencies)
                if hasattr(tracker, "set_sequences"):
                    tracker.set_sequences(sequences)
                logger.debug("Applied tool deps via fallback")


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
    """

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
        """Apply framework integrations (workflows, RL, teams)."""
        self.apply_workflows(orchestrator, vertical, context, result)
        self.apply_rl_config(orchestrator, vertical, context, result)
        self.apply_team_specs(orchestrator, vertical, context, result)

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
        # Check if vertical has workflow provider using protocol
        workflow_provider = None
        if hasattr(vertical, "get_workflow_provider"):
            workflow_provider = vertical.get_workflow_provider()

        # Also check if vertical itself is a WorkflowProviderProtocol
        if workflow_provider is None and isinstance(vertical, type):
            if hasattr(vertical, "get_workflows"):
                workflow_provider = vertical

        if workflow_provider is None:
            return

        # Get workflows from provider
        workflows = workflow_provider.get_workflows()
        if not workflows:
            return

        workflow_count = len(workflows)
        result.workflows_count = workflow_count

        # Store in context
        context.apply_workflows(workflows)

        # Register with workflow registry if available
        try:
            from victor.workflows.registry import get_workflow_registry

            registry = get_workflow_registry()
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
                        "Orchestrator lacks set_rl_hooks method; "
                        "hooks stored in context only"
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

        # Attach via capability (SOLID compliant)
        if _check_capability(orchestrator, "team_specs"):
            _invoke_capability(orchestrator, "team_specs", team_specs)
            logger.debug("Applied team specs via capability")
        elif hasattr(orchestrator, "set_team_specs"):
            orchestrator.set_team_specs(team_specs)
            logger.debug("Applied team specs via set_team_specs")
        else:
            result.add_warning(
                "Orchestrator lacks set_team_specs method; " "specs stored in context only"
            )

        result.add_info(
            f"Registered {team_count} team specs: " f"{', '.join(team_specs.keys())}"
        )
        # Log each team spec registration (matching workflow DEBUG logging pattern)
        for team_name in team_specs.keys():
            logger.debug(f"Registered team_spec: {team_name}")
        logger.debug(f"Applied {team_count} team specifications from vertical")


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
        elif hasattr(orchestrator, "set_vertical_context"):
            orchestrator.set_vertical_context(context)
            logger.debug("Attached context via set_vertical_context")
        else:
            result.add_warning(
                "Orchestrator lacks set_vertical_context method; " "context not attached"
            )


# =============================================================================
# Extensions Step Handler
# =============================================================================


class ExtensionsStepHandler(BaseStepHandler):
    """Handler for all vertical extensions.

    Coordinates the application of middleware, safety, prompts,
    mode config, and tool dependencies from the vertical extensions.
    """

    def __init__(self):
        """Initialize with sub-handlers."""
        self._middleware_handler = MiddlewareStepHandler()
        self._safety_handler = SafetyStepHandler()
        self._prompt_handler = PromptStepHandler()
        self._config_handler = ConfigStepHandler()

    @property
    def name(self) -> str:
        return "extensions"

    @property
    def order(self) -> int:
        return 45  # After config, before middleware

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Apply all vertical extensions."""
        extensions = vertical.get_extensions()
        if extensions is None:
            logger.debug("No extensions available for vertical")
            return

        # Apply middleware
        if extensions.middleware:
            self._middleware_handler.apply_middleware(
                orchestrator, extensions.middleware, context, result
            )

        # Apply safety extensions
        if extensions.safety_extensions:
            self._safety_handler.apply_safety_extensions(
                orchestrator, extensions.safety_extensions, context, result
            )

        # Apply prompt contributors
        if extensions.prompt_contributors:
            self._prompt_handler.apply_contributors(
                orchestrator, extensions.prompt_contributors, context, result
            )

        # Apply mode config
        if extensions.mode_config_provider:
            self._config_handler.apply_mode_config(
                orchestrator, extensions.mode_config_provider, context, result
            )

        # Apply tool dependencies
        if extensions.tool_dependency_provider:
            self._config_handler.apply_tool_dependencies(
                orchestrator, extensions.tool_dependency_provider, context, result
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
                ToolStepHandler(),
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
    "PromptStepHandler",
    "SafetyStepHandler",
    "ConfigStepHandler",
    "MiddlewareStepHandler",
    "FrameworkStepHandler",
    "ContextStepHandler",
    "ExtensionsStepHandler",
    # Registry
    "StepHandlerRegistry",
    # Capability helpers
    "_check_capability",
    "_invoke_capability",
]
