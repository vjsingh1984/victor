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

"""Decomposed step handlers for vertical integration (Phase 10.1).

This module provides single-responsibility step handlers that replace the
monolithic FrameworkStepHandler. Each handler focuses on one integration
concern, following SOLID principles.

Handler Order:
    60 - WorkflowStepHandler: Register workflows from vertical
    61 - RLConfigStepHandler: Apply RL configuration
    62 - TeamSpecStepHandler: Register team specifications
    63 - ChainStepHandler: Register chains
    64 - PersonaStepHandler: Register personas
    65 - CapabilityProviderStepHandler: Apply capability providers
    66 - ToolGraphStepHandler: Register tool graphs
    67 - HandlerRegistrationStepHandler: Register custom handlers

Design Philosophy:
- Single Responsibility: Each handler handles one concern
- Open/Closed: New handlers can be added without modifying existing ones
- Liskov Substitution: All handlers implement BaseStepHandler protocol
- Interface Segregation: Minimal protocol for step handlers
- Dependency Inversion: Handlers depend on protocols, not implementations

Usage:
    from victor.framework.decomposed_handlers import get_decomposed_handlers

    # Get all decomposed handlers
    handlers = get_decomposed_handlers()

    # Apply each handler in order
    for handler in handlers:
        handler.apply(orchestrator, vertical, context, result)
"""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
)

from victor.framework.step_handlers import BaseStepHandler

if TYPE_CHECKING:
    from victor.core.verticals.base import VerticalBase
    from victor.framework.vertical_integration import (
        IntegrationResult,
        VerticalContext,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# Helper functions for capability checking
# =============================================================================


def _check_capability(orchestrator: Any, capability_name: str) -> bool:
    """Check if orchestrator has a capability."""
    if hasattr(orchestrator, "has_capability"):
        return orchestrator.has_capability(capability_name)  # type: ignore[no-any-return]
    return False


def _invoke_capability(orchestrator: Any, capability_name: str, *args: Any, **kwargs: Any) -> Any:
    """Invoke a capability on orchestrator."""
    if hasattr(orchestrator, "invoke_capability"):
        return orchestrator.invoke_capability(capability_name, *args, **kwargs)
    return None


# =============================================================================
# Lazy imports for registries
# =============================================================================


def get_workflow_registry():
    """Lazy import workflow registry."""
    from victor.workflows.registry import get_global_registry

    return get_global_registry()


def get_trigger_registry():
    """Lazy import trigger registry."""
    from victor.workflows.trigger_registry import get_trigger_registry

    return get_trigger_registry()


def get_team_registry():
    """Lazy import team registry."""
    from victor.framework.team_registry import get_team_registry

    return get_team_registry()


def get_chain_registry():
    """Lazy import chain registry."""
    from victor.framework.chain_registry import get_chain_registry

    return get_chain_registry()


def get_persona_registry():
    """Lazy import persona registry."""
    from victor.framework.persona_registry import get_persona_registry

    return get_persona_registry()


def get_handler_registry():
    """Lazy import handler registry."""
    from victor.framework.handler_registry import get_handler_registry

    return get_handler_registry()


# =============================================================================
# WorkflowStepHandler (Order 60)
# =============================================================================


class WorkflowStepHandler(BaseStepHandler):
    """Handler for registering workflows from vertical.

    Single responsibility: Workflow registration with workflow registry
    and trigger registry.
    """

    def __init__(
        self,
        workflow_registry: Optional[Any] = None,
        trigger_registry: Optional[Any] = None,
    ):
        """Initialize with optional injected registries.

        Args:
            workflow_registry: Optional workflow registry for DIP compliance
            trigger_registry: Optional trigger registry for DIP compliance
        """
        self._workflow_registry = workflow_registry
        self._trigger_registry = trigger_registry

    @property
    def name(self) -> str:
        return "workflow"

    @property
    def order(self) -> int:
        return 60

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Register workflows from vertical."""
        from victor.core.verticals.protocols import WorkflowProviderProtocol

        # Check if vertical has workflow provider
        workflow_provider = None

        if isinstance(vertical, WorkflowProviderProtocol):
            workflow_provider = vertical
        elif hasattr(vertical, "get_workflow_provider"):
            workflow_provider = vertical.get_workflow_provider()
        elif isinstance(vertical, type) and hasattr(vertical, "get_workflows"):
            workflow_provider = vertical

        if workflow_provider is not None:
            # Get workflows from provider
            workflows = workflow_provider.get_workflows()
            if workflows:
                workflow_count = len(workflows)
                result.workflows_count = workflow_count

                # Store in context
                context.apply_workflows(workflows)

        # Register with workflow registry
        try:
            registry = self._workflow_registry or get_workflow_registry()
            for name, workflow in workflows.items():
                registry.register(
                    f"{vertical.name}:{name}",
                    workflow,
                    replace=True,
                )
            result.add_info(f"Registered {workflow_count} workflows: {', '.join(workflows.keys())}")
        except ImportError:
            result.add_warning("Workflow registry not available")
        except Exception as e:
            result.add_warning(f"Could not register workflows: {e}")

        # Register workflow triggers
        try:
            trigger_registry = self._trigger_registry or get_trigger_registry()
            if workflow_provider is not None and hasattr(workflow_provider, "get_auto_workflows"):
                auto_workflows = workflow_provider.get_auto_workflows()
                if auto_workflows:
                    trigger_registry.register_from_vertical(vertical.name, auto_workflows)
                    result.add_info(f"Registered {len(auto_workflows)} workflow triggers")
        except ImportError:
            pass  # Optional dependency
        except Exception as e:
            result.add_warning(f"Could not register workflow triggers: {e}")

        logger.debug(f"Applied {workflow_count} workflows from vertical")


# =============================================================================
# RLConfigStepHandler (Order 61)
# =============================================================================


class RLConfigStepHandler(BaseStepHandler):
    """Handler for applying RL configuration.

    Single responsibility: RL config and hooks application.
    """

    @property
    def name(self) -> str:
        return "rl_config"

    @property
    def order(self) -> int:
        return 61

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Apply RL configuration from vertical."""
        from victor.core.verticals.protocols import RLConfigProviderProtocol

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

        # Apply RL hooks if available
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


# =============================================================================
# TeamSpecStepHandler (Order 62)
# =============================================================================


class TeamSpecStepHandler(BaseStepHandler):
    """Handler for registering team specifications.

    Single responsibility: Team spec registration.
    """

    def __init__(self, team_registry: Optional[Any] = None):
        """Initialize with optional injected registry.

        Args:
            team_registry: Optional team registry for DIP compliance
        """
        self._team_registry = team_registry

    @property
    def name(self) -> str:
        return "team_spec"

    @property
    def order(self) -> int:
        return 62

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Register team specifications from vertical."""
        from victor.core.verticals.protocols import TeamSpecProviderProtocol

        team_specs = None
        team_provider = None

        # Check for team spec provider method
        if hasattr(vertical, "get_team_spec_provider"):
            team_provider = vertical.get_team_spec_provider()
            if team_provider and isinstance(team_provider, TeamSpecProviderProtocol):
                team_specs = team_provider.get_team_specs()

        # Fallback: check if vertical has get_team_specs directly
        if team_specs is None and hasattr(vertical, "get_team_specs"):
            team_specs = vertical.get_team_specs()

        if not team_specs:
            return

        team_count = len(team_specs)
        result.team_specs_count = team_count

        # Store in context
        context.apply_team_specs(team_specs)

        result.add_info(f"Registered {team_count} team specs: {', '.join(team_specs.keys())}")

        # Register with global registry
        try:
            team_registry = self._team_registry or get_team_registry()
            team_registry.register_from_vertical(vertical.name, team_specs, replace=True)
            logger.debug(f"Registered {team_count} teams with global registry")
        except ImportError:
            pass  # Optional dependency
        except Exception as e:
            result.add_warning(f"Could not register with team registry: {e}")

        logger.debug(f"Applied {team_count} team specifications from vertical")


# =============================================================================
# ChainStepHandler (Order 63)
# =============================================================================


class ChainStepHandler(BaseStepHandler):
    """Handler for registering chains.

    Single responsibility: Chain registration with ChainRegistry.
    """

    def __init__(self, chain_registry: Optional[Any] = None):
        """Initialize with optional injected registry.

        Args:
            chain_registry: Optional chain registry for DIP compliance
        """
        self._chain_registry = chain_registry

    @property
    def name(self) -> str:
        return "chain"

    @property
    def order(self) -> int:
        return 63

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Register chains from vertical."""
        # Check if vertical provides chains
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
            registry = self._chain_registry or get_chain_registry()
            for name, chain in chains.items():
                registry.register(
                    name,
                    chain,
                    vertical=vertical.name,
                    replace=True,
                )
                logger.debug(f"Registered chain: {vertical.name}:{name}")

            result.add_info(f"Registered {chain_count} chains: {', '.join(chains.keys())}")
            logger.debug(f"Applied {chain_count} chains from vertical")
        except ImportError:
            result.add_warning("ChainRegistry not available")
        except Exception as e:
            result.add_warning(f"Could not register chains: {e}")


# =============================================================================
# PersonaStepHandler (Order 64)
# =============================================================================


class PersonaStepHandler(BaseStepHandler):
    """Handler for registering personas.

    Single responsibility: Persona registration with PersonaRegistry.
    """

    def __init__(self, persona_registry: Optional[Any] = None):
        """Initialize with optional injected registry.

        Args:
            persona_registry: Optional persona registry for DIP compliance
        """
        self._persona_registry = persona_registry

    @property
    def name(self) -> str:
        return "persona"

    @property
    def order(self) -> int:
        return 64

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Register personas from vertical."""
        if not hasattr(vertical, "get_personas"):
            return

        personas = vertical.get_personas()
        if not personas:
            return

        persona_count = len(personas)

        # Register with PersonaRegistry
        try:
            registry = self._persona_registry or get_persona_registry()

            for name, persona in personas.items():
                # Convert to PersonaSpec if needed
                if hasattr(persona, "to_persona_spec"):
                    spec = persona.to_persona_spec()
                elif hasattr(persona, "to_dict"):
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
                    spec = persona

                registry.register(name, spec, vertical=vertical.name, replace=True)
                logger.debug(f"Registered persona: {vertical.name}:{name}")

            result.add_info(f"Registered {persona_count} personas: {', '.join(personas.keys())}")
            logger.debug(f"Applied {persona_count} personas from vertical")
        except ImportError:
            result.add_warning("PersonaRegistry not available")
        except Exception as e:
            result.add_warning(f"Could not register personas: {e}")


# =============================================================================
# CapabilityProviderStepHandler (Order 65)
# =============================================================================


class CapabilityProviderStepHandler(BaseStepHandler):
    """Handler for wiring capability providers.

    Single responsibility: Wire capability providers to CapabilityLoader.
    """

    @property
    def name(self) -> str:
        return "capability_provider"

    @property
    def order(self) -> int:
        return 65

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Wire capability provider from vertical to framework."""
        if not hasattr(vertical, "get_capability_provider"):
            return

        provider = vertical.get_capability_provider()
        if provider is None:
            return

        # Count capabilities
        cap_count = 0
        if hasattr(provider, "get_capabilities"):
            caps = provider.get_capabilities()
            cap_count = len(caps) if caps else 0
        elif hasattr(provider, "CAPABILITIES"):
            cap_count = len(provider.CAPABILITIES)

        # Wire to CapabilityLoader
        try:
            from victor.framework.capability_loader import CapabilityLoader

            loader = None
            if hasattr(orchestrator, "_capability_loader"):
                loader = orchestrator._capability_loader
            else:
                loader = CapabilityLoader()
                if hasattr(orchestrator, "__dict__"):
                    orchestrator._capability_loader = loader

            # Load capabilities from provider
            if hasattr(provider, "get_capabilities"):
                for cap in provider.get_capabilities():
                    if hasattr(cap, "name") and hasattr(cap, "handler"):
                        cap_type = getattr(cap, "capability_type", None)
                        from victor.framework.protocols import CapabilityType

                        # Ensure capability_type is a CapabilityType enum
                        if cap_type is None or not isinstance(cap_type, CapabilityType):
                            cap_type = CapabilityType.TOOL

                        loader.register_capability(
                            name=cap.name,
                            handler=cap.handler,
                            capability_type=cap_type,
                            version=getattr(cap, "version", "1.0"),
                        )

            loader.apply_to(orchestrator)

            result.add_info(f"Wired capability provider with {cap_count} capabilities")
            logger.debug(f"Applied capability provider from vertical={vertical.name}")
        except ImportError:
            result.add_warning("CapabilityLoader not available")
        except Exception as e:
            result.add_warning(f"Could not wire capability provider: {e}")
            logger.debug(f"Capability provider error: {e}", exc_info=True)


# =============================================================================
# ToolGraphStepHandler (Order 66)
# =============================================================================


class ToolGraphStepHandler(BaseStepHandler):
    """Handler for registering tool graphs.

    Single responsibility: Tool graph registration with ToolGraphRegistry.
    """

    @property
    def name(self) -> str:
        return "tool_graph"

    @property
    def order(self) -> int:
        return 66

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Register tool graphs from vertical."""
        try:
            from victor.tools.tool_graph import ToolGraphRegistry
        except ImportError:
            return  # Optional dependency

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


# =============================================================================
# HandlerRegistrationStepHandler (Order 67)
# =============================================================================


class HandlerRegistrationStepHandler(BaseStepHandler):
    """Handler for registering custom handlers.

    Single responsibility: Handler registration with HandlerRegistry.
    """

    def __init__(self, handler_registry: Optional[Any] = None):
        """Initialize with optional injected registry.

        Args:
            handler_registry: Optional handler registry for DIP compliance
        """
        self._handler_registry = handler_registry

    @property
    def name(self) -> str:
        return "handler_registration"

    @property
    def order(self) -> int:
        return 67

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: type["VerticalBase"],
        context: "VerticalContext",
        result: "IntegrationResult",
    ) -> None:
        """Register custom handlers from vertical."""
        if not hasattr(vertical, "get_handlers"):
            return

        handlers = vertical.get_handlers()
        if not handlers:
            return

        handler_count = len(handlers)

        try:
            registry = self._handler_registry or get_handler_registry()

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
            # Fall back to executor registration
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
# Handler Factory
# =============================================================================


def get_decomposed_handlers() -> list[BaseStepHandler]:
    """Get all decomposed handlers in order.

    Returns:
        List of step handlers sorted by order.
    """
    handlers = [
        WorkflowStepHandler(),
        RLConfigStepHandler(),
        TeamSpecStepHandler(),
        ChainStepHandler(),
        PersonaStepHandler(),
        CapabilityProviderStepHandler(),
        ToolGraphStepHandler(),
        HandlerRegistrationStepHandler(),
    ]
    return sorted(handlers, key=lambda h: h.order)


def create_decomposed_handlers(
    workflow_registry: Optional[Any] = None,
    trigger_registry: Optional[Any] = None,
    team_registry: Optional[Any] = None,
    chain_registry: Optional[Any] = None,
    persona_registry: Optional[Any] = None,
    handler_registry: Optional[Any] = None,
) -> list[BaseStepHandler]:
    """Create decomposed handlers with injected registries.

    Args:
        workflow_registry: Optional workflow registry
        trigger_registry: Optional trigger registry
        team_registry: Optional team registry
        chain_registry: Optional chain registry
        persona_registry: Optional persona registry
        handler_registry: Optional handler registry

    Returns:
        List of step handlers sorted by order.
    """
    handlers = [
        WorkflowStepHandler(workflow_registry, trigger_registry),
        RLConfigStepHandler(),
        TeamSpecStepHandler(team_registry),
        ChainStepHandler(chain_registry),
        PersonaStepHandler(persona_registry),
        CapabilityProviderStepHandler(),
        ToolGraphStepHandler(),
        HandlerRegistrationStepHandler(handler_registry),
    ]
    return sorted(handlers, key=lambda h: h.order)


__all__ = [
    # Individual handlers
    "WorkflowStepHandler",
    "RLConfigStepHandler",
    "TeamSpecStepHandler",
    "ChainStepHandler",
    "PersonaStepHandler",
    "CapabilityProviderStepHandler",
    "ToolGraphStepHandler",
    "HandlerRegistrationStepHandler",
    # Factory functions
    "get_decomposed_handlers",
    "create_decomposed_handlers",
]
