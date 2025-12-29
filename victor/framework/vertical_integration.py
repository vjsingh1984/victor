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

"""Reusable pipeline for vertical extension application.

This module provides a unified pipeline for applying vertical extensions
to orchestrators, ensuring both CLI (FrameworkShim) and SDK (Agent.create())
paths apply identical vertical configurations.

Design Philosophy:
- Single implementation for all vertical integration
- Protocol-based access (no private attribute writes)
- Type-safe configuration through VerticalContext
- SOLID-compliant extension points

Architecture:
    VerticalIntegrationPipeline
    ├── Step 1: Create VerticalContext
    ├── Step 2: Apply tools filter
    ├── Step 3: Apply system prompt
    ├── Step 4: Apply stages
    ├── Step 5: Apply middleware
    ├── Step 6: Apply safety extensions
    ├── Step 7: Apply prompt contributors
    ├── Step 8: Apply mode configuration
    └── Step 9: Apply tool dependencies

Usage:
    from victor.framework.vertical_integration import (
        VerticalIntegrationPipeline,
        IntegrationResult,
    )

    # Create pipeline
    pipeline = VerticalIntegrationPipeline()

    # Apply vertical
    result = pipeline.apply(orchestrator, CodingAssistant)

    # Check result
    if result.success:
        print(f"Applied {result.vertical_name}")
        print(f"Tools: {result.tools_applied}")
    else:
        print(f"Errors: {result.errors}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
    Union,
    runtime_checkable,
)

from victor.agent.vertical_context import VerticalContext, create_vertical_context

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.verticals.base import VerticalBase, VerticalConfig
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

# Import capability registry protocol for type-safe capability access
from victor.framework.protocols import CapabilityRegistryProtocol


def _check_capability(obj: Any, capability_name: str) -> bool:
    """Check if object has capability via registry or fallback to hasattr.

    This function provides a migration path from hasattr to protocol-based
    capability discovery. It first checks the capability registry (if available),
    then falls back to hasattr for backward compatibility.

    Args:
        obj: Object to check
        capability_name: Name of capability

    Returns:
        True if capability is available
    """
    # Check capability registry first (protocol-based)
    if isinstance(obj, CapabilityRegistryProtocol):
        return obj.has_capability(capability_name)

    # Fallback mappings for hasattr compatibility
    capability_to_attr = {
        "enabled_tools": "set_enabled_tools",
        "prompt_builder": "prompt_builder",
        "custom_prompt": "prompt_builder",
        "task_type_hints": "prompt_builder",
        "prompt_section": "prompt_builder",
        "vertical_middleware": "apply_vertical_middleware",
        "middleware_chain": "_middleware_chain",
        "vertical_safety_patterns": "apply_vertical_safety_patterns",
        "safety_patterns": "_safety_checker",
        "adaptive_mode_controller": "adaptive_mode_controller",
        "mode_configs": "adaptive_mode_controller",
        "default_budget": "adaptive_mode_controller",
        "tool_sequence_tracker": "_sequence_tracker",
        "tool_dependencies": "_sequence_tracker",
        "tool_sequences": "_sequence_tracker",
        "vertical_context": "set_vertical_context",
        "rl_hooks": "_rl_hooks",
        "team_specs": "_team_specs",
    }

    attr = capability_to_attr.get(capability_name, capability_name)
    return hasattr(obj, attr)


def _invoke_capability(obj: Any, capability_name: str, *args: Any, **kwargs: Any) -> Any:
    """Invoke capability via registry or fallback to direct method call.

    This function provides a migration path from getattr to protocol-based
    capability invocation. It first tries the capability registry (if available),
    then falls back to direct method call for backward compatibility.

    Args:
        obj: Object to invoke capability on
        capability_name: Name of capability
        *args: Arguments for capability
        **kwargs: Keyword arguments for capability

    Returns:
        Result of capability invocation
    """
    # Use capability registry if available
    if isinstance(obj, CapabilityRegistryProtocol):
        try:
            return obj.invoke_capability(capability_name, *args, **kwargs)
        except (KeyError, TypeError):
            pass  # Fall through to fallback

    # Fallback mappings
    capability_to_method = {
        "enabled_tools": "set_enabled_tools",
        "vertical_middleware": "apply_vertical_middleware",
        "vertical_safety_patterns": "apply_vertical_safety_patterns",
        "vertical_context": "set_vertical_context",
        "rl_hooks": "set_rl_hooks",
        "team_specs": "set_team_specs",
    }

    method_name = capability_to_method.get(capability_name, f"set_{capability_name}")
    method = getattr(obj, method_name, None)
    if callable(method):
        return method(*args, **kwargs)

    # Fallback: try direct attribute assignment
    attr_name = f"_{capability_name}"
    if hasattr(obj, attr_name):
        setattr(obj, attr_name, args[0] if args else kwargs.get("value"))
        return None

    raise AttributeError(f"Cannot invoke capability '{capability_name}' on {type(obj)}")

logger = logging.getLogger(__name__)


# =============================================================================
# Integration Result
# =============================================================================


@dataclass
class IntegrationResult:
    """Result from applying a vertical to an orchestrator.

    Attributes:
        success: Whether integration succeeded
        vertical_name: Name of the applied vertical
        context: The created VerticalContext
        tools_applied: Set of tools enabled
        middleware_count: Number of middleware applied
        safety_patterns_count: Number of safety patterns applied
        prompt_hints_count: Number of prompt hints applied
        mode_configs_count: Number of mode configs applied
        workflows_count: Number of workflows registered
        rl_learners_count: Number of RL learners configured
        team_specs_count: Number of team specifications registered
        errors: List of errors encountered
        warnings: List of warnings
        info: List of informational messages
    """

    success: bool = True
    vertical_name: Optional[str] = None
    context: Optional[VerticalContext] = None
    tools_applied: Set[str] = field(default_factory=set)
    middleware_count: int = 0
    safety_patterns_count: int = 0
    prompt_hints_count: int = 0
    mode_configs_count: int = 0
    workflows_count: int = 0
    rl_learners_count: int = 0
    team_specs_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)

    def add_error(self, error: str) -> None:
        """Add an error."""
        self.errors.append(error)
        self.success = False

    def add_warning(self, warning: str) -> None:
        """Add a warning."""
        self.warnings.append(warning)

    def add_info(self, message: str) -> None:
        """Add an informational message."""
        self.info.append(message)


# =============================================================================
# Integration Protocol
# =============================================================================


@runtime_checkable
class OrchestratorVerticalProtocol(Protocol):
    """Protocol for orchestrators that support vertical integration.

    This defines the methods an orchestrator must implement to
    properly receive vertical configuration. Using a protocol
    ensures SOLID compliance (no private attribute access).
    """

    def set_vertical_context(self, context: VerticalContext) -> None:
        """Set the vertical context."""
        ...

    def set_enabled_tools(self, tools: Set[str]) -> None:
        """Set enabled tools from vertical."""
        ...

    def apply_vertical_middleware(self, middleware: List[Any]) -> None:
        """Apply middleware from vertical."""
        ...

    def apply_vertical_safety_patterns(self, patterns: List[Any]) -> None:
        """Apply safety patterns from vertical."""
        ...


# =============================================================================
# Integration Pipeline
# =============================================================================


class VerticalIntegrationPipeline:
    """Reusable pipeline for vertical extension application.

    This pipeline encapsulates all vertical integration logic that was
    previously spread across FrameworkShim._apply_vertical() and
    related methods. It provides:

    1. **Unified Integration**: Same logic for CLI and SDK paths
    2. **Protocol Compliance**: Uses proper methods, not private attrs
    3. **Error Handling**: Graceful degradation with detailed errors
    4. **Extensibility**: Hook points for custom integration steps

    Example:
        pipeline = VerticalIntegrationPipeline()

        # Apply with full vertical class
        result = pipeline.apply(orchestrator, CodingAssistant)

        # Apply with vertical name (from registry)
        result = pipeline.apply(orchestrator, "coding")

        # Apply with custom configuration
        result = pipeline.apply(
            orchestrator,
            CodingAssistant,
            config_overrides={"tool_budget": 30},
        )
    """

    def __init__(
        self,
        strict_mode: bool = False,
        pre_hooks: Optional[List[Callable[[Any, Type], None]]] = None,
        post_hooks: Optional[List[Callable[[Any, IntegrationResult], None]]] = None,
    ):
        """Initialize the pipeline.

        Args:
            strict_mode: If True, fail on any integration error
            pre_hooks: Callables to run before integration
            post_hooks: Callables to run after integration
        """
        self._strict_mode = strict_mode
        self._pre_hooks = pre_hooks or []
        self._post_hooks = post_hooks or []

    def apply(
        self,
        orchestrator: Any,
        vertical: Union[Type["VerticalBase"], str],
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> IntegrationResult:
        """Apply a vertical to an orchestrator.

        This is the main entry point for vertical integration. It:
        1. Resolves the vertical (if string name)
        2. Creates a VerticalContext
        3. Applies all vertical extensions
        4. Attaches context to orchestrator

        Args:
            orchestrator: AgentOrchestrator instance
            vertical: Vertical class or name string
            config_overrides: Optional configuration overrides

        Returns:
            IntegrationResult with details of what was applied
        """
        result = IntegrationResult()

        # Resolve vertical from name if needed
        vertical_class = self._resolve_vertical(vertical)
        if vertical_class is None:
            result.add_error(f"Vertical not found: {vertical}")
            return result

        result.vertical_name = vertical_class.name

        # Run pre-hooks
        for hook in self._pre_hooks:
            try:
                hook(orchestrator, vertical_class)
            except Exception as e:
                result.add_warning(f"Pre-hook error: {e}")

        # Create context
        context = self._create_context(vertical_class, result)
        if context is None:
            return result
        result.context = context

        # Apply all integration steps
        self._apply_tools(orchestrator, vertical_class, context, result)
        self._apply_system_prompt(orchestrator, vertical_class, context, result)
        self._apply_stages(orchestrator, vertical_class, context, result)
        self._apply_extensions(orchestrator, vertical_class, context, result)

        # Apply new framework integrations (workflows, RL, teams)
        self._apply_workflows(orchestrator, vertical_class, context, result)
        self._apply_rl_config(orchestrator, vertical_class, context, result)
        self._apply_team_specs(orchestrator, vertical_class, context, result)

        # Attach context to orchestrator
        self._attach_context(orchestrator, context, result)

        # Run post-hooks
        for hook in self._post_hooks:
            try:
                hook(orchestrator, result)
            except Exception as e:
                result.add_warning(f"Post-hook error: {e}")

        logger.debug(
            f"Vertical integration complete: {result.vertical_name}, "
            f"tools={len(result.tools_applied)}, "
            f"middleware={result.middleware_count}, "
            f"safety_patterns={result.safety_patterns_count}"
        )

        return result

    def _resolve_vertical(
        self, vertical: Union[Type["VerticalBase"], str]
    ) -> Optional[Type["VerticalBase"]]:
        """Resolve vertical from name or return class.

        Args:
            vertical: Vertical class or name string

        Returns:
            Vertical class or None if not found
        """
        if isinstance(vertical, str):
            from victor.verticals.base import VerticalRegistry

            resolved = VerticalRegistry.get(vertical)
            if resolved is None:
                # Try case-insensitive match
                for name in VerticalRegistry.list_names():
                    if name.lower() == vertical.lower():
                        return VerticalRegistry.get(name)
            return resolved
        return vertical

    def _create_context(
        self, vertical: Type["VerticalBase"], result: IntegrationResult
    ) -> Optional[VerticalContext]:
        """Create the VerticalContext for this vertical.

        Args:
            vertical: Vertical class
            result: Result to update

        Returns:
            VerticalContext or None on error
        """
        try:
            config = vertical.get_config()
            context = create_vertical_context(
                name=vertical.name,
                config=config,
            )
            return context
        except Exception as e:
            result.add_error(f"Failed to create context: {e}")
            return None

    def _apply_tools(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply tools filter from vertical.

        Args:
            orchestrator: Orchestrator instance
            vertical: Vertical class
            context: Vertical context
            result: Result to update
        """
        try:
            tools = vertical.get_tools()
            if tools:
                # Canonicalize tool names to ensure consistency
                canonical_tools = self._canonicalize_tool_names(set(tools))
                context.apply_enabled_tools(canonical_tools)
                result.tools_applied = canonical_tools

                # Use capability-based approach (protocol-first, fallback to hasattr)
                if _check_capability(orchestrator, "enabled_tools"):
                    _invoke_capability(orchestrator, "enabled_tools", canonical_tools)
                    logger.debug(f"Applied {len(canonical_tools)} tools via capability")
                else:
                    result.add_warning(
                        "Orchestrator does not implement enabled_tools capability; "
                        "tools stored in context only"
                    )
        except Exception as e:
            if self._strict_mode:
                result.add_error(f"Failed to apply tools: {e}")
            else:
                result.add_warning(f"Tools application error: {e}")

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
            # Fallback if tool_names module not available
            logger.debug("Tool name canonicalization not available")
            return tools

    def _apply_system_prompt(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply system prompt from vertical.

        Args:
            orchestrator: Orchestrator instance
            vertical: Vertical class
            context: Vertical context
            result: Result to update
        """
        try:
            system_prompt = vertical.get_system_prompt()
            if system_prompt:
                context.apply_system_prompt(system_prompt)

                # Apply via capability (protocol-first with fallback)
                if _check_capability(orchestrator, "custom_prompt"):
                    _invoke_capability(orchestrator, "custom_prompt", system_prompt)
                    logger.debug("Applied system prompt via capability")
                elif _check_capability(orchestrator, "prompt_builder"):
                    # Fallback: try direct prompt builder access
                    prompt_builder = getattr(orchestrator, "prompt_builder", None)
                    if prompt_builder:
                        if hasattr(prompt_builder, "set_custom_prompt"):
                            prompt_builder.set_custom_prompt(system_prompt)
                        elif hasattr(prompt_builder, "_custom_prompt"):
                            prompt_builder._custom_prompt = system_prompt
                        logger.debug("Applied system prompt via prompt_builder fallback")
        except Exception as e:
            if self._strict_mode:
                result.add_error(f"Failed to apply system prompt: {e}")
            else:
                result.add_warning(f"System prompt error: {e}")

    def _apply_stages(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply stages configuration from vertical.

        Args:
            orchestrator: Orchestrator instance
            vertical: Vertical class
            context: Vertical context
            result: Result to update
        """
        try:
            stages = vertical.get_stages()
            if stages:
                context.apply_stages(stages)
                logger.debug(f"Applied stages: {list(stages.keys())}")
        except Exception as e:
            if self._strict_mode:
                result.add_error(f"Failed to apply stages: {e}")
            else:
                result.add_warning(f"Stages application error: {e}")

    def _apply_extensions(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply all vertical extensions.

        Args:
            orchestrator: Orchestrator instance
            vertical: Vertical class
            context: Vertical context
            result: Result to update
        """
        try:
            extensions = vertical.get_extensions()
            if extensions is None:
                logger.debug("No extensions available for vertical")
                return

            # Apply middleware
            if extensions.middleware:
                self._apply_middleware(orchestrator, extensions.middleware, context, result)

            # Apply safety extensions
            if extensions.safety_extensions:
                self._apply_safety(orchestrator, extensions.safety_extensions, context, result)

            # Apply prompt contributors
            if extensions.prompt_contributors:
                self._apply_prompts(orchestrator, extensions.prompt_contributors, context, result)

            # Apply mode config
            if extensions.mode_config_provider:
                self._apply_mode_config(
                    orchestrator, extensions.mode_config_provider, context, result
                )

            # Apply tool dependencies
            if extensions.tool_dependency_provider:
                self._apply_tool_deps(
                    orchestrator, extensions.tool_dependency_provider, context, result
                )

        except Exception as e:
            if self._strict_mode:
                result.add_error(f"Failed to apply extensions: {e}")
            else:
                result.add_warning(f"Extensions application error: {e}")

    def _apply_middleware(
        self,
        orchestrator: Any,
        middleware_list: List[Any],
        context: VerticalContext,
        result: IntegrationResult,
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

    def _apply_safety(
        self,
        orchestrator: Any,
        safety_extensions: List[Any],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply safety extensions to orchestrator.

        Args:
            orchestrator: Orchestrator instance
            safety_extensions: List of safety extensions
            context: Vertical context
            result: Result to update
        """
        all_patterns = []
        for ext in safety_extensions:
            all_patterns.extend(ext.get_bash_patterns())
            all_patterns.extend(ext.get_file_patterns())

        context.apply_safety_patterns(all_patterns)
        result.safety_patterns_count = len(all_patterns)

        # Use capability-based approach (protocol-first, fallback to hasattr)
        if _check_capability(orchestrator, "vertical_safety_patterns"):
            _invoke_capability(orchestrator, "vertical_safety_patterns", all_patterns)
            logger.debug(f"Applied {len(all_patterns)} safety patterns via capability")
        elif _check_capability(orchestrator, "safety_patterns"):
            _invoke_capability(orchestrator, "safety_patterns", all_patterns)
            logger.debug(f"Applied {len(all_patterns)} safety patterns via safety_patterns capability")
        else:
            # Fallback: try safety checker directly
            safety_checker = getattr(orchestrator, "safety_checker", None) or getattr(
                orchestrator, "_safety_checker", None
            )
            if safety_checker:
                if hasattr(safety_checker, "add_patterns"):
                    safety_checker.add_patterns(all_patterns)
                elif hasattr(safety_checker, "_custom_patterns"):
                    safety_checker._custom_patterns.extend(all_patterns)
                logger.debug(f"Applied {len(all_patterns)} safety patterns via fallback")

    def _apply_prompts(
        self,
        orchestrator: Any,
        contributors: List[Any],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply prompt contributors to orchestrator.

        Args:
            orchestrator: Orchestrator instance
            contributors: List of prompt contributors
            context: Vertical context
            result: Result to update
        """
        # Merge task hints from all contributors
        merged_hints = {}
        for contributor in sorted(contributors, key=lambda c: c.get_priority()):
            merged_hints.update(contributor.get_task_type_hints())

        context.apply_task_hints(merged_hints)
        result.prompt_hints_count = len(merged_hints)

        # Apply task hints via capability
        if _check_capability(orchestrator, "task_type_hints"):
            _invoke_capability(orchestrator, "task_type_hints", merged_hints)
            logger.debug(f"Applied {len(merged_hints)} task hints via capability")
        elif _check_capability(orchestrator, "prompt_builder"):
            # Fallback: try direct prompt builder access
            prompt_builder = getattr(orchestrator, "prompt_builder", None)
            if prompt_builder:
                if hasattr(prompt_builder, "set_task_type_hints"):
                    prompt_builder.set_task_type_hints(merged_hints)
                elif hasattr(prompt_builder, "_task_type_hints"):
                    prompt_builder._task_type_hints = merged_hints

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

    def _apply_mode_config(
        self,
        orchestrator: Any,
        mode_provider: Any,
        context: VerticalContext,
        result: IntegrationResult,
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
                logger.debug(f"Applied mode config via fallback")

    def _apply_tool_deps(
        self,
        orchestrator: Any,
        dep_provider: Any,
        context: VerticalContext,
        result: IntegrationResult,
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
            logger.debug(f"Applied tool dependencies via capability")
        if _check_capability(orchestrator, "tool_sequences"):
            _invoke_capability(orchestrator, "tool_sequences", sequences)
            logger.debug(f"Applied tool sequences via capability")

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
                logger.debug(f"Applied tool deps via fallback")

    def _attach_context(
        self,
        orchestrator: Any,
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Attach the vertical context to orchestrator.

        Args:
            orchestrator: Orchestrator instance
            context: Vertical context to attach
            result: Result to update
        """
        # Use capability-based approach (protocol-first, fallback to direct)
        if _check_capability(orchestrator, "vertical_context"):
            _invoke_capability(orchestrator, "vertical_context", context)
            logger.debug("Attached context via capability")
        else:
            # Fallback: direct attribute assignment
            orchestrator._vertical_context = context
            logger.debug("Attached context via _vertical_context attribute")

    # =========================================================================
    # New Framework Integrations (Workflows, RL, Teams)
    # =========================================================================

    def _apply_workflows(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply workflow definitions from vertical.

        Registers vertical-specific workflows with the workflow registry
        for use during task execution.

        Args:
            orchestrator: Orchestrator instance
            vertical: Vertical class
            context: Vertical context
            result: Result to update
        """
        try:
            # Check if vertical has workflow provider using protocol
            workflow_provider = None
            if hasattr(vertical, "get_workflow_provider"):
                workflow_provider = vertical.get_workflow_provider()

            # Also check if vertical itself is a WorkflowProviderProtocol
            if workflow_provider is None and isinstance(vertical, type):
                if hasattr(vertical, "get_workflows"):
                    # Vertical class implements protocol directly
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
                    f"Registered {workflow_count} workflows: "
                    f"{', '.join(workflows.keys())}"
                )
            except ImportError:
                result.add_warning("Workflow registry not available")
            except Exception as e:
                result.add_warning(f"Could not register workflows: {e}")

            logger.debug(f"Applied {workflow_count} workflows from vertical")

        except Exception as e:
            if self._strict_mode:
                result.add_error(f"Failed to apply workflows: {e}")
            else:
                result.add_warning(f"Workflow application error: {e}")

    def _apply_rl_config(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply RL configuration from vertical.

        Configures the RL system with vertical-specific learners,
        task type mappings, and quality thresholds.

        Args:
            orchestrator: Orchestrator instance
            vertical: Vertical class
            context: Vertical context
            result: Result to update
        """
        try:
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

                    # Attach hooks via capability (protocol-first, fallback to direct)
                    if _check_capability(orchestrator, "rl_hooks"):
                        _invoke_capability(orchestrator, "rl_hooks", rl_hooks)
                        logger.debug("Applied RL hooks via capability")
                    else:
                        # Fallback: direct attribute assignment
                        orchestrator._rl_hooks = rl_hooks
                        logger.debug("Applied RL hooks via fallback")

            result.add_info(f"Configured {learner_count} RL learners")
            logger.debug(f"Applied RL config with {learner_count} learners")

        except Exception as e:
            if self._strict_mode:
                result.add_error(f"Failed to apply RL config: {e}")
            else:
                result.add_warning(f"RL config application error: {e}")

    def _apply_team_specs(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply team specifications from vertical.

        Registers vertical-specific team configurations for
        multi-agent task execution.

        Args:
            orchestrator: Orchestrator instance
            vertical: Vertical class
            context: Vertical context
            result: Result to update
        """
        try:
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

            # Attach via capability (protocol-first, fallback to direct)
            if _check_capability(orchestrator, "team_specs"):
                _invoke_capability(orchestrator, "team_specs", team_specs)
                logger.debug(f"Applied team specs via capability")
            else:
                # Fallback: direct attribute assignment
                orchestrator._team_specs = team_specs
                logger.debug(f"Applied team specs via fallback")

            result.add_info(
                f"Registered {team_count} team specs: "
                f"{', '.join(team_specs.keys())}"
            )
            # Log each team spec registration (matching workflow DEBUG logging pattern)
            for team_name in team_specs.keys():
                logger.debug(f"Registered team_spec: {team_name}")
            logger.debug(f"Applied {team_count} team specifications from vertical")

        except Exception as e:
            if self._strict_mode:
                result.add_error(f"Failed to apply team specs: {e}")
            else:
                result.add_warning(f"Team specs application error: {e}")


# =============================================================================
# Factory Functions
# =============================================================================


def create_integration_pipeline(
    strict: bool = False,
) -> VerticalIntegrationPipeline:
    """Create a vertical integration pipeline.

    Args:
        strict: Whether to use strict mode

    Returns:
        Configured VerticalIntegrationPipeline
    """
    return VerticalIntegrationPipeline(strict_mode=strict)


def apply_vertical(
    orchestrator: Any,
    vertical: Union[Type["VerticalBase"], str],
) -> IntegrationResult:
    """Apply a vertical to an orchestrator.

    Convenience function for one-shot vertical application.

    Args:
        orchestrator: Orchestrator instance
        vertical: Vertical class or name

    Returns:
        IntegrationResult
    """
    pipeline = VerticalIntegrationPipeline()
    return pipeline.apply(orchestrator, vertical)


__all__ = [
    "IntegrationResult",
    "VerticalIntegrationPipeline",
    "OrchestratorVerticalProtocol",
    "create_integration_pipeline",
    "apply_vertical",
]
