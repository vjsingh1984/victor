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

"""ServiceBuilder for building core services.

This module provides a builder for core service components used by
AgentOrchestrator, including:
- DI container
- ConversationController
- StreamingController
- TaskAnalyzer
- ContextCompactor
- UsageAnalytics
- And other core services

Part of HIGH-005: Initialization Complexity reduction.
"""

from typing import Any, Dict, Optional
from victor.agent.builders.base import FactoryAwareBuilder
from victor.core.bootstrap import ensure_bootstrapped


class ServiceBuilder(FactoryAwareBuilder):
    """Build core services (DI container, controllers, analytics, etc.).

    This builder creates all core service components that the orchestrator
    needs for its operation. It delegates to OrchestratorFactory for the
    actual component creation while providing a cleaner, more focused API.

    Components built:
        - service_provider: DI container
        - conversation_controller: Message history and state management
        - streaming_controller: Streaming session management
        - streaming_handler: Streaming chat loop logic
        - task_analyzer: Task classification and analysis
        - context_compactor: Context management and truncation
        - usage_analytics: Tool and provider usage tracking
        - sequence_tracker: Tool sequence pattern tracking
        - recovery_handler: Error recovery and retry logic
        - recovery_integration: Recovery delegation submodule
        - recovery_coordinator: Centralized recovery coordination
        - chunk_generator: Centralized chunk generation
        - tool_planner: Centralized tool planning
        - task_coordinator: Task coordination and guidance
        - observability: Event bus and metrics integration
        - rl_coordinator: Reinforcement learning coordination
        - sanitizer: Response sanitization
        - complexity_classifier: Task complexity classification
        - action_authorizer: Action intent detection
        - search_router: Intelligent search routing
        - project_context: Project-specific context loading
        - conversation_state: Conversation state machine
        - intent_classifier: Semantic intent classification
        - memory_manager: Persistent conversation memory
        - memory_session_id: Session ID for memory
        - reminder_manager: Context-aware reminder injection
        - metrics_collector: Performance metrics collection
        - streaming_metrics_collector: Streaming-specific metrics
    """

    def __init__(self, settings, factory: Optional["OrchestratorFactory"] = None):
        """Initialize the ServiceBuilder.

        Args:
            settings: Application settings
            factory: Optional OrchestratorFactory instance (created if not provided)
        """
        super().__init__(settings, factory)
        self._container = None

    def build(
        self,
        provider: Any = None,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        provider_name: Optional[str] = None,
        profile_name: Optional[str] = None,
        tool_selection: Optional[Dict[str, Any]] = None,
        thinking: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Build all core services.

        Args:
            provider: LLM provider instance (required for some services)
            model: Model identifier (required for some services)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            provider_name: Optional provider label from profile
            profile_name: Optional profile name for session tracking
            tool_selection: Optional tool selection configuration
            thinking: Enable extended thinking mode
            **kwargs: Additional dependencies from other builders

        Returns:
            Dictionary of built services with keys:
            - service_provider: DI container
            - conversation_controller: ConversationController
            - streaming_controller: StreamingController
            - streaming_handler: StreamingChatHandler
            - task_analyzer: TaskAnalyzer
            - context_compactor: ContextCompactor
            - usage_analytics: UsageAnalytics
            - sequence_tracker: ToolSequenceTracker
            - recovery_handler: RecoveryHandler
            - recovery_integration: OrchestratorRecoveryIntegration
            - recovery_coordinator: RecoveryCoordinator
            - chunk_generator: ChunkGenerator
            - tool_planner: ToolPlanner
            - task_coordinator: TaskCoordinator
            - observability: ObservabilityIntegration
            - rl_coordinator: RLCoordinator
            - sanitizer: ResponseSanitizer
            - complexity_classifier: ComplexityClassifier
            - action_authorizer: ActionAuthorizer
            - search_router: SearchRouter
            - project_context: ProjectContext
            - conversation_state: ConversationStateMachine
            - intent_classifier: IntentClassifier
            - memory_manager: ConversationStore
            - memory_session_id: str
            - reminder_manager: ReminderManager
            - metrics_collector: MetricsCollector
            - streaming_metrics_collector: StreamingMetricsCollector
        """
        services = {}

        # Bootstrap DI container
        self._container = ensure_bootstrapped(self.settings)
        services["service_provider"] = self._container

        # Create or reuse factory
        self._ensure_factory(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            provider_name=provider_name,
            profile_name=profile_name,
            tool_selection=tool_selection,
            thinking=thinking,
        )
        self._factory._container = self._container

        # Build core services via factory
        services["sanitizer"] = self._factory.create_sanitizer()
        services["project_context"] = self._factory.create_project_context()
        services["complexity_classifier"] = self._factory.create_complexity_classifier()
        services["action_authorizer"] = self._factory.create_action_authorizer()
        services["search_router"] = self._factory.create_search_router()

        # Build conversation-related services
        services["conversation_state"] = self._factory.create_conversation_state_machine()
        services["intent_classifier"] = self._factory.create_intent_classifier()

        # Build memory components if needed
        if provider and model:
            # Get tool calling capabilities to determine native tool support
            from victor.agent.tool_calling import get_adapter_for_provider

            # provider may be an object or string - extract name if needed
            actual_provider_name = provider_name or (
                provider.name if hasattr(provider, "name") else str(provider)
            )
            tool_adapter = get_adapter_for_provider(actual_provider_name, model)
            tool_calling_caps = tool_adapter.get_capabilities()

            memory_manager, memory_session_id = self._factory.create_memory_components(
                provider_name or "unknown", tool_calling_caps.native_tool_calls
            )
            services["memory_manager"] = memory_manager
            services["memory_session_id"] = memory_session_id
        else:
            services["memory_manager"] = None
            services["memory_session_id"] = None

        # Build metrics and analytics
        services["streaming_metrics_collector"] = self._factory.create_streaming_metrics_collector()
        services["usage_analytics"] = self._factory.create_usage_analytics()
        services["sequence_tracker"] = self._factory.create_sequence_tracker()

        # Build recovery components
        services["recovery_handler"] = self._factory.create_recovery_handler()
        services["recovery_integration"] = self._factory.create_recovery_integration(
            services["recovery_handler"]
        )
        services["recovery_coordinator"] = self._factory.create_recovery_coordinator()

        # Build chunk and task coordination
        services["chunk_generator"] = self._factory.create_chunk_generator()
        services["tool_planner"] = self._factory.create_tool_planner()
        services["task_coordinator"] = self._factory.create_task_coordinator()

        # Build observability
        services["observability"] = self._factory.create_observability()

        # Build RL coordinator
        services["rl_coordinator"] = self._factory.create_rl_coordinator()

        # Build reminder manager (requires provider info)
        if provider_name:
            services["reminder_manager"] = self._factory.create_reminder_manager(
                provider=provider_name,
                task_complexity="medium",  # Default, updated later
                tool_budget=kwargs.get("tool_budget", 10),
            )
        else:
            services["reminder_manager"] = None

        # Build metrics collector
        services["metrics_collector"] = self._factory.create_metrics_collector(
            streaming_metrics_collector=services["streaming_metrics_collector"],
            usage_logger=kwargs.get("usage_logger"),
            debug_logger=kwargs.get("debug_logger"),
            tool_cost_lookup=kwargs.get("tool_cost_lookup"),
        )

        # Build TaskAnalyzer (singleton)
        from victor.agent.task_analyzer import get_task_analyzer

        services["task_analyzer"] = get_task_analyzer()

        # Register all built services
        self._register_components(services)

        self._logger.info(
            f"ServiceBuilder built {len(services)} core services: " f"{', '.join(services.keys())}"
        )

        return services

    def build_conversation_controller(
        self,
        provider: Any,
        model: str,
        conversation: Any,
        conversation_state: Any,
        memory_manager: Any,
        memory_session_id: str,
        system_prompt: str,
    ) -> Any:
        """Build ConversationController after dependencies are ready.

        Args:
            provider: LLM provider instance
            model: Model identifier
            conversation: MessageHistory instance
            conversation_state: ConversationStateMachine instance
            memory_manager: ConversationStore instance
            memory_session_id: Session ID for memory
            system_prompt: System prompt text

        Returns:
            ConversationController instance
        """
        controller = self._factory.create_conversation_controller(
            provider=provider,
            model=model,
            conversation=conversation,
            conversation_state=conversation_state,
            memory_manager=memory_manager,
            memory_session_id=memory_session_id,
            system_prompt=system_prompt,
        )
        self.register_component("conversation_controller", controller)
        return controller

    def build_streaming_controller(
        self, streaming_metrics_collector: Any, on_session_complete: Any
    ) -> Any:
        """Build StreamingController.

        Args:
            streaming_metrics_collector: Metrics collector for streaming
            on_session_complete: Callback for session completion

        Returns:
            StreamingController instance
        """
        controller = self._factory.create_streaming_controller(
            streaming_metrics_collector=streaming_metrics_collector,
            on_session_complete=on_session_complete,
        )
        self.register_component("streaming_controller", controller)
        return controller

    def build_streaming_handler(self, message_adder: Any) -> Any:
        """Build StreamingChatHandler.

        Args:
            message_adder: Object with add_assistant_message method

        Returns:
            StreamingChatHandler instance
        """
        handler = self._factory.create_streaming_chat_handler(message_adder=message_adder)
        self.register_component("streaming_handler", handler)
        return handler

    def build_context_compactor(self, conversation_controller: Any) -> Any:
        """Build ContextCompactor.

        Args:
            conversation_controller: ConversationController instance

        Returns:
            ContextCompactor instance
        """
        compactor = self._factory.create_context_compactor(
            conversation_controller=conversation_controller
        )
        self.register_component("context_compactor", compactor)
        return compactor
