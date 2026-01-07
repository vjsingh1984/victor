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

"""Integration helpers to wire observability into Victor components.

This module provides integration points for connecting the EventBus
to various Victor components (orchestrator, tool pipeline, state machine).

Design Pattern: Mediator
The ObservabilityIntegration class acts as a mediator between
Victor components and the observability subsystem.

Example:
    from victor.observability.integration import ObservabilityIntegration

    # Wire up observability for an orchestrator
    integration = ObservabilityIntegration()
    integration.wire_orchestrator(orchestrator)

    # Or use the convenience function
    from victor.observability.integration import setup_observability
    setup_observability(orchestrator)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from victor.core.events import ObservabilityBus, get_observability_bus
from victor.observability.hooks import StateHookManager, TransitionHistory

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.observability.cqrs_adapter import CQRSEventAdapter, UnifiedEventBridge

logger = logging.getLogger(__name__)


class ObservabilityIntegration:
    """Integrates EventBus with Victor components.

    This class wires the observability subsystem into various
    Victor components, enabling automatic event emission.

    Features:
    - Tool execution events
    - State transition events
    - Model response events
    - Error events
    - Session lifecycle events

    Example:
        integration = ObservabilityIntegration()
        integration.wire_orchestrator(orchestrator)

        # Events are now automatically emitted
    """

    def __init__(
        self,
        event_bus: Optional["ObservabilityBus"] = None,
        session_id: Optional[str] = None,
        enable_cqrs_bridge: bool = False,
    ) -> None:
        """Initialize the integration.

        Args:
            event_bus: ObservabilityBus to use (default: singleton).
            session_id: Optional session ID for correlation.
            enable_cqrs_bridge: Whether to enable CQRS event bridging.
                When enabled, events are automatically forwarded between
                the observability ObservabilityBus and CQRS EventDispatcher.
        """
        self._bus = event_bus or get_observability_bus()
        self._session_id = session_id
        self._wired_components: List[str] = []
        self._tool_start_times: Dict[str, float] = {}
        self._cqrs_bridge: Optional["UnifiedEventBridge"] = None
        self._state_hook_manager: Optional[StateHookManager] = None

        # Note: session_id is stored in self._session_id for use in event emissions
        # The new ObservabilityBus doesn't have set_session_id() method

        if enable_cqrs_bridge:
            self._setup_cqrs_bridge()

    @property
    def event_bus(self) -> "ObservabilityBus":
        """Get the event bus."""
        return self._bus

    @property
    def cqrs_bridge(self) -> Optional["UnifiedEventBridge"]:
        """Get the CQRS bridge if enabled."""
        return self._cqrs_bridge

    @property
    def state_hook_manager(self) -> Optional[StateHookManager]:
        """Get the StateHookManager if wired to a state machine.

        Returns:
            StateHookManager instance or None.
        """
        return self._state_hook_manager

    @property
    def state_transition_history(self) -> Optional[TransitionHistory]:
        """Get the transition history from the state hook manager.

        Returns:
            TransitionHistory if available, None otherwise.
        """
        if self._state_hook_manager:
            return self._state_hook_manager.history
        return None

    def get_state_transition_metrics(self) -> Dict[str, Any]:
        """Get metrics about state transitions.

        Returns a summary of transition patterns including:
        - Total transitions
        - Unique transition paths
        - Stage visit counts
        - Cycle detection status
        - Average time per stage

        Returns:
            Dictionary with transition metrics.
        """
        if not self._state_hook_manager or not self._state_hook_manager.history:
            return {
                "total_transitions": 0,
                "unique_stages_visited": 0,
                "has_cycles": False,
                "current_stage": None,
                "stage_sequence": [],
            }

        history = self._state_hook_manager.history
        stage_sequence = history.get_stage_sequence()
        unique_stages = set(stage_sequence)

        # Calculate per-stage average durations
        stage_avg_durations = {}
        all_durations: List[float] = []
        for stage in unique_stages:
            avg = history.get_average_duration(stage)
            if avg is not None:
                stage_avg_durations[stage] = avg
                all_durations.append(avg)

        # Calculate overall average from stage averages
        overall_avg = sum(all_durations) / len(all_durations) if all_durations else None

        return {
            "total_transitions": len(history),
            "unique_stages_visited": len(unique_stages),
            "has_cycles": history.has_cycle(),
            "current_stage": history.current_stage,
            "stage_sequence": stage_sequence,
            "stage_average_durations_ms": stage_avg_durations,
            "overall_average_duration_ms": overall_avg,
            "stage_visit_counts": {
                stage: history.get_stage_visit_count(stage) for stage in unique_stages
            },
        }

    def _setup_cqrs_bridge(self) -> None:
        """Set up the CQRS event bridge.

        Creates a UnifiedEventBridge that connects the observability
        EventBus to the CQRS EventDispatcher for bidirectional event flow.
        """
        from victor.observability.cqrs_adapter import create_unified_bridge

        self._cqrs_bridge = create_unified_bridge(
            event_bus=self._bus,
            auto_start=True,
        )
        self._wired_components.append("cqrs_bridge")
        logger.debug("CQRS event bridge enabled")

    def enable_cqrs_bridge(self) -> "UnifiedEventBridge":
        """Enable CQRS event bridging.

        Creates and starts a UnifiedEventBridge if not already enabled.

        Returns:
            The UnifiedEventBridge instance.

        Example:
            integration = ObservabilityIntegration()
            bridge = integration.enable_cqrs_bridge()

            # Events now flow between EventBus and CQRS EventDispatcher
        """
        if self._cqrs_bridge is None:
            self._setup_cqrs_bridge()
        return self._cqrs_bridge

    def disable_cqrs_bridge(self) -> None:
        """Disable CQRS event bridging.

        Stops the bridge if it was enabled.
        """
        if self._cqrs_bridge is not None:
            self._cqrs_bridge.stop()
            self._cqrs_bridge = None
            if "cqrs_bridge" in self._wired_components:
                self._wired_components.remove("cqrs_bridge")
            logger.debug("CQRS event bridge disabled")

    def set_session_id(self, session_id: str) -> None:
        """Set the session ID for event correlation.

        Args:
            session_id: Session identifier.
        """
        self._session_id = session_id
        # Note: The new ObservabilityBus doesn't have set_session_id()
        # Session ID is stored in self._session_id for use in event emissions

    def wire_orchestrator(self, orchestrator: "AgentOrchestrator") -> None:
        """Wire EventBus into an AgentOrchestrator.

        This connects:
        - Tool execution events
        - State machine transitions
        - Session lifecycle events

        Args:
            orchestrator: Orchestrator to wire.
        """
        # Wire state machine hooks
        if hasattr(orchestrator, "conversation_state"):
            self.wire_state_machine(orchestrator.conversation_state)

        # Store reference for event emission
        orchestrator._observability = self
        self._wired_components.append("orchestrator")

        logger.debug("Wired observability into orchestrator")

    def wire_state_machine(self, state_machine: Any) -> None:
        """Wire EventBus into a state machine with history-aware hooks.

        Uses history-aware hooks to emit rich state events that include:
        - Transition analytics (cycle detection, visit counts)
        - Stage sequence tracking
        - Duration measurements

        Args:
            state_machine: ConversationStateMachine or compatible.
        """
        hook_manager = StateHookManager(enable_history=True, history_max_size=100)

        @hook_manager.on_transition_with_history
        def emit_transition_with_analytics(
            old_stage: str,
            new_stage: str,
            context: Dict[str, Any],
            history: TransitionHistory,
        ) -> None:
            """Emit state change with rich analytics from history."""
            # Build enhanced context with history analytics
            enhanced_context = {
                **context,
                "has_cycle": history.has_cycle(),
                "visit_count": history.get_stage_visit_count(new_stage),
                "stage_sequence": history.get_stage_sequence(),
                "transition_count": len(history),
            }

            # Get last record for duration info if available
            last_records = history.get_last(1)
            if last_records and last_records[0].duration_ms is not None:
                enhanced_context["stage_duration_ms"] = last_records[0].duration_ms

            self._bus.emit_state_change(
                old_stage=old_stage,
                new_stage=new_stage,
                confidence=context.get("confidence", 1.0),
                context=enhanced_context,
            )

            # Emit warning event if cycle detected (potential infinite loop)
            if history.has_cycle():
                cycle_count = history.get_stage_visit_count(new_stage)
                if cycle_count >= 3:
                    # Emit cycle warning event
                    self._bus.emit(
                        topic="error.cycle_warning",
                        data={
                            "stage": new_stage,
                            "visit_count": cycle_count,
                            "sequence": history.get_stage_sequence()[-5:],
                            "severity": "warning",
                            "category": "error",
                        },
                    )

        if hasattr(state_machine, "set_hooks"):
            state_machine.set_hooks(hook_manager)

        self._state_hook_manager = hook_manager
        self._wired_components.append("state_machine")
        logger.debug("Wired observability into state machine with history-aware hooks")

    # =========================================================================
    # Tool Events
    # =========================================================================

    def on_tool_start(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_id: Optional[str] = None,
    ) -> None:
        """Emit a tool start event.

        Args:
            tool_name: Name of the tool.
            arguments: Tool arguments.
            tool_id: Optional tool call ID.
        """
        self._tool_start_times[tool_id or tool_name] = time.time()
        self._bus.emit(
            topic="tool.start",
            data={
                "tool_name": tool_name,
                "arguments": arguments,
                "tool_id": tool_id,
                "category": "tool",
            },
        )

    def on_tool_end(
        self,
        tool_name: str,
        result: Any,
        success: bool = True,
        tool_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Emit a tool end event.

        Args:
            tool_name: Name of the tool.
            result: Tool result.
            success: Whether tool succeeded.
            tool_id: Optional tool call ID.
            error: Optional error message.
        """
        key = tool_id or tool_name
        start_time = self._tool_start_times.pop(key, None)
        duration_ms = (time.time() - start_time) * 1000 if start_time else None

        # Emit tool complete/end event
        self._bus.emit(
            topic="tool.end",
            data={
                "tool_name": tool_name,
                "result": result,
                "success": success,
                "tool_id": tool_id,
                "duration_ms": duration_ms,
                "category": "tool",
            },
        )

        if not success and error:
            # Emit tool error event
            self._bus.emit(
                topic=f"error.{tool_name}",
                data={
                    "tool_name": tool_name,
                    "error": error,
                    "tool_id": tool_id,
                    "category": "error",
                },
            )

    # =========================================================================
    # Model Events
    # =========================================================================

    def on_model_request(
        self,
        provider: str,
        model: str,
        message_count: int,
        tool_count: int,
    ) -> None:
        """Emit a model request event.

        Args:
            provider: LLM provider name.
            model: Model identifier.
            message_count: Number of messages in request.
            tool_count: Number of tools available.
        """
        # Emit model request event
        self._bus.emit(
            topic="model.request",
            data={
                "provider": provider,
                "model": model,
                "message_count": message_count,
                "tool_count": tool_count,
                "category": "model",
            },
        )

    def on_model_response(
        self,
        provider: str,
        model: str,
        tokens_used: Optional[int] = None,
        tool_calls: int = 0,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Emit a model response event.

        Args:
            provider: LLM provider name.
            model: Model identifier.
            tokens_used: Optional token count.
            tool_calls: Number of tool calls in response.
            latency_ms: Optional latency in milliseconds.
        """
        # Emit model response event
        self._bus.emit(
            topic="model.response",
            data={
                "provider": provider,
                "model": model,
                "tokens_used": tokens_used,
                "tool_calls": tool_calls,
                "latency_ms": latency_ms,
                "category": "model",
            },
        )

    # =========================================================================
    # Lifecycle Events
    # =========================================================================

    def on_session_start(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Emit a session start event.

        Args:
            metadata: Optional session metadata.
        """
        # Emit session start event
        data = metadata or {}
        data["category"] = "lifecycle"
        self._bus.emit(
            topic="lifecycle.session.start",
            data=data,
        )

    def on_session_end(
        self,
        tool_calls: int = 0,
        duration_seconds: Optional[float] = None,
        success: bool = True,
    ) -> None:
        """Emit a session end event.

        Args:
            tool_calls: Total tool calls in session.
            duration_seconds: Session duration.
            success: Whether session completed successfully.
        """
        # Emit session end event
        self._bus.emit(
            topic="lifecycle.session.end",
            data={
                "tool_calls": tool_calls,
                "duration_seconds": duration_seconds,
                "success": success,
                "category": "lifecycle",
            },
        )

    # =========================================================================
    # Error Events
    # =========================================================================

    def on_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
    ) -> None:
        """Emit an error event.

        Args:
            error: The exception.
            context: Optional error context.
            recoverable: Whether error is recoverable.
        """
        self._bus.emit_error(error, context, recoverable)


def setup_observability(
    orchestrator: "AgentOrchestrator",
    session_id: Optional[str] = None,
    enable_cqrs_bridge: bool = False,
) -> ObservabilityIntegration:
    """Convenience function to set up observability for an orchestrator.

    Args:
        orchestrator: Orchestrator to wire.
        session_id: Optional session ID.
        enable_cqrs_bridge: Whether to enable CQRS event bridging.
            When enabled, events are automatically forwarded between
            the observability EventBus and CQRS EventDispatcher.

    Returns:
        Configured ObservabilityIntegration instance.

    Example:
        # Basic usage
        integration = setup_observability(orchestrator)

        # With CQRS bridge for event sourcing integration
        integration = setup_observability(orchestrator, enable_cqrs_bridge=True)

        # Subscribe to CQRS events
        if integration.cqrs_bridge:
            from victor.core import EventDispatcher
            dispatcher = integration.cqrs_bridge.adapter.event_dispatcher
            dispatcher.subscribe_all(lambda e: print(f"CQRS event: {e}"))
    """
    integration = ObservabilityIntegration(
        session_id=session_id,
        enable_cqrs_bridge=enable_cqrs_bridge,
    )
    integration.wire_orchestrator(orchestrator)
    return integration


class ToolEventMiddleware:
    """Middleware for automatic tool event emission.

    Can be injected into the tool pipeline to automatically
    emit tool events without modifying individual tools.

    Example:
        middleware = ToolEventMiddleware(integration)
        tool_pipeline.add_middleware(middleware)
    """

    def __init__(self, integration: ObservabilityIntegration) -> None:
        """Initialize middleware.

        Args:
            integration: ObservabilityIntegration instance.
        """
        self._integration = integration

    def before_execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_id: Optional[str] = None,
    ) -> None:
        """Called before tool execution.

        Args:
            tool_name: Name of the tool.
            arguments: Tool arguments.
            tool_id: Optional tool call ID.
        """
        self._integration.on_tool_start(tool_name, arguments, tool_id)

    def after_execute(
        self,
        tool_name: str,
        result: Any,
        success: bool = True,
        tool_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Called after tool execution.

        Args:
            tool_name: Name of the tool.
            result: Tool result.
            success: Whether tool succeeded.
            tool_id: Optional tool call ID.
            error: Optional error message.
        """
        self._integration.on_tool_end(tool_name, result, success, tool_id, error)
