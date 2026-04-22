"""Post-initialization bootstrapper for AgentOrchestrator.

Extracts domain facade creation, lifecycle wiring, protocol conformance
assertions, and session context setup from the orchestrator's __init__.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)


class AgentRuntimeBootstrapper:
    """Assembles domain facades and wires post-initialization dependencies.

    All methods are static and operate on a fully-initialized orchestrator
    instance. Called at the end of AgentOrchestrator.__init__ after all
    components have been created by the factory and runtime boundaries.
    """

    @staticmethod
    def create_facades(orchestrator: AgentOrchestrator) -> None:
        """Create all 8 domain facades and set them on the orchestrator."""
        from victor.agent.facades import (
            ChatFacade,
            MetricsFacade,
            OrchestrationFacade,
            ProviderFacade,
            ResilienceFacade,
            SessionFacade,
            ToolFacade,
            WorkflowFacade,
        )

        orchestrator._chat_facade = ChatFacade(
            conversation=orchestrator.conversation,
            conversation_controller=orchestrator._conversation_controller,
            conversation_state=orchestrator.conversation_state,
            memory_manager=orchestrator.memory_manager,
            memory_session_id=orchestrator._memory_session_id,
            embedding_store=getattr(orchestrator, "_conversation_embedding_store", None),
            intent_classifier=orchestrator.intent_classifier,
            intent_detector=orchestrator.intent_detector,
            reminder_manager=orchestrator.reminder_manager,
            system_prompt=orchestrator._system_prompt,
            response_completer=orchestrator.response_completer,
            context_compactor=orchestrator._context_compactor,
            task_completion_detector=orchestrator._task_completion_detector,
        )

        orchestrator._tool_facade = ToolFacade(
            tools=orchestrator.tools,
            tool_pipeline=orchestrator._tool_pipeline,
            tool_executor=orchestrator.tool_executor,
            tool_selector=orchestrator.tool_selector,
            tool_cache=orchestrator.tool_cache,
            tool_graph=orchestrator.tool_graph,
            tool_registrar=orchestrator.tool_registrar,
            tool_budget=orchestrator.tool_budget,
            tool_output_formatter=orchestrator._tool_output_formatter,
            deduplication_tracker=orchestrator._deduplication_tracker,
            argument_normalizer=orchestrator.argument_normalizer,
            parallel_executor=orchestrator.parallel_executor,
            safety_checker=orchestrator._safety_checker,
            auto_committer=orchestrator._auto_committer,
            middleware_chain=orchestrator._middleware_chain,
            code_correction_middleware=(orchestrator._code_correction_middleware),
            tool_access_controller=orchestrator._tool_access_controller,
            budget_manager=orchestrator._budget_manager,
            search_router=orchestrator.search_router,
            semantic_selector=orchestrator.semantic_selector,
            task_classifier=orchestrator.task_classifier,
            sequence_tracker=orchestrator._sequence_tracker,
            unified_tracker=orchestrator.unified_tracker,
            plugin_manager=orchestrator.plugin_manager,
        )

        orchestrator._provider_facade = ProviderFacade(
            provider=orchestrator.provider,
            model=orchestrator.model,
            provider_name=orchestrator.provider_name,
            temperature=orchestrator.temperature,
            max_tokens=orchestrator.max_tokens,
            thinking=orchestrator.thinking,
            provider_manager=orchestrator._provider_manager,
            provider_runtime=orchestrator._provider_runtime,
            provider_coordinator=orchestrator._provider_coordinator,
            provider_switch_coordinator=(orchestrator._provider_switch_coordinator),
        )

        orchestrator._session_facade = SessionFacade(
            session_state=orchestrator._session_state,
            session_accessor=orchestrator._session_accessor,
            session_ledger=orchestrator._session_ledger,
            lifecycle_manager=orchestrator._lifecycle_manager,
            active_session_id=orchestrator.active_session_id,
            memory_session_id=orchestrator._memory_session_id,
            profile_name=orchestrator._profile_name,
            checkpoint_manager=orchestrator._checkpoint_manager,
        )

        orchestrator._metrics_facade = MetricsFacade(
            metrics_runtime=orchestrator._metrics_runtime,
            metrics_collector=orchestrator._metrics_collector,
            usage_analytics=orchestrator._usage_analytics,
            usage_logger=orchestrator.usage_logger,
            streaming_metrics_collector=(orchestrator.streaming_metrics_collector),
            session_cost_tracker=orchestrator._session_cost_tracker,
            metrics_coordinator=orchestrator._metrics_coordinator,
            debug_logger=orchestrator.debug_logger,
            callback_coordinator=orchestrator._callback_coordinator,
        )

        orchestrator._resilience_facade = ResilienceFacade(
            resilience_runtime=orchestrator._resilience_runtime,
            recovery_handler=orchestrator._recovery_handler,
            recovery_integration=orchestrator._recovery_integration,
            recovery_coordinator=orchestrator._recovery_coordinator,
            chunk_generator=orchestrator._chunk_generator,
            context_manager=orchestrator._context_manager,
            rl_coordinator=orchestrator._rl_coordinator,
            code_manager=orchestrator.code_manager,
            background_tasks=orchestrator._background_tasks,
            cancel_event=orchestrator._cancel_event,
            is_streaming=orchestrator._is_streaming,
        )

        orchestrator._workflow_facade = WorkflowFacade(
            workflow_registry=orchestrator._workflow_registry,
            workflow_runtime=orchestrator._workflow_runtime,
            workflow_optimization=orchestrator._workflow_optimization,
            mode_workflow_team_coordinator=(orchestrator._mode_workflow_team_coordinator),
        )

        orchestrator._orchestration_facade = OrchestrationFacade(
            interaction_runtime=orchestrator._interaction_runtime,
            chat_service=getattr(orchestrator, "_chat_service", None),
            tool_service=getattr(orchestrator, "_tool_service", None),
            session_service=getattr(orchestrator, "_session_service", None),
            context_service=getattr(orchestrator, "_context_service", None),
            provider_service=getattr(orchestrator, "_provider_service", None),
            recovery_service=getattr(orchestrator, "_recovery_service", None),
            chat_coordinator=orchestrator._chat_coordinator,
            get_tool_coordinator=orchestrator._get_deprecated_tool_coordinator,
            deprecated_session_coordinator=orchestrator._session_coordinator,
            turn_executor=orchestrator._turn_executor,
            sync_chat_coordinator=orchestrator._sync_chat_coordinator,
            streaming_chat_coordinator=(orchestrator._streaming_chat_coordinator),
            unified_chat_coordinator=(orchestrator._unified_chat_coordinator),
            protocol_adapter=orchestrator._protocol_adapter,
            streaming_handler=orchestrator._streaming_handler,
            streaming_controller=orchestrator._streaming_controller,
            streaming_coordinator=orchestrator._streaming_coordinator,
            iteration_coordinator=getattr(orchestrator, "_iteration_coordinator", None),
            task_analyzer=orchestrator._task_analyzer,
            presentation=orchestrator._presentation,
            vertical_integration_adapter=(orchestrator._vertical_integration_adapter),
            vertical_context=orchestrator._vertical_context,
            observability=orchestrator._observability,
            execution_tracer=getattr(orchestrator, "_execution_tracer", None),
            tool_call_tracer=getattr(orchestrator, "_tool_call_tracer", None),
            intelligent_integration=orchestrator._intelligent_integration,
            subagent_orchestrator=orchestrator._subagent_orchestrator,
        )

        logger.debug(
            "Domain facades created: ChatFacade, ToolFacade, ProviderFacade, "
            "SessionFacade, MetricsFacade, ResilienceFacade, WorkflowFacade, "
            "OrchestrationFacade"
        )

    @staticmethod
    def wire_lifecycle(orchestrator: AgentOrchestrator) -> None:
        """Wire LifecycleManager with component dependencies for shutdown."""
        lm = orchestrator._lifecycle_manager
        lm.set_provider(orchestrator.provider)
        lm.set_code_manager(
            orchestrator.code_manager if hasattr(orchestrator, "code_manager") else None
        )
        lm.set_semantic_selector(
            orchestrator.semantic_selector if hasattr(orchestrator, "semantic_selector") else None
        )
        lm.set_usage_logger(
            orchestrator.usage_logger if hasattr(orchestrator, "usage_logger") else None
        )
        lm.set_background_tasks(list(orchestrator._background_tasks))
        lm.set_flush_analytics_callback(orchestrator.flush_analytics)
        lm.set_stop_health_monitoring_callback(orchestrator.stop_health_monitoring)

    @staticmethod
    def assert_protocol_conformance(orchestrator: AgentOrchestrator) -> None:
        """Debug-mode assertion for ChatOrchestratorProtocol conformance."""
        if __debug__:
            from victor.agent.coordinators.chat_protocols import (
                ChatOrchestratorProtocol,
            )
            from victor.framework.protocols import (
                verify_protocol_conformance,
            )

            _conforms, _missing = verify_protocol_conformance(
                orchestrator, ChatOrchestratorProtocol
            )
            assert _conforms, (
                "AgentOrchestrator missing " f"ChatOrchestratorProtocol members: {_missing}"
            )

    @staticmethod
    def setup_session_context(orchestrator: AgentOrchestrator) -> None:
        """Initialize session context for tracing/propagation."""
        from victor.core.context import set_session_id

        set_session_id(orchestrator.active_session_id or "")

    @staticmethod
    def prepare_components(orchestrator: AgentOrchestrator, settings: Any) -> None:
        """Create post-factory components and wire dependencies.

        Handles checkpoint manager, workflow optimization, vertical context,
        runtime boundaries, and coordinator placeholders. Called after factory
        component creation but before facade assembly.
        """
        from victor.agent.vertical_context import (
            VerticalContext,
            create_vertical_context,
        )
        from victor.agent.vertical_integration_adapter import (
            VerticalIntegrationAdapter,
        )

        # Checkpoint manager for time-travel debugging
        orchestrator._checkpoint_manager = orchestrator._factory.create_checkpoint_manager()

        # Workflow optimization components
        orchestrator._workflow_optimization = (
            orchestrator._factory.create_workflow_optimization_components(
                timeout_seconds=getattr(settings, "execution_timeout", None)
            )
        )

        # Wire component dependencies
        orchestrator._factory.wire_component_dependencies(
            recovery_handler=None,
            context_compactor=orchestrator._context_compactor,
            observability=orchestrator._observability,
            conversation_state=orchestrator.conversation_state,
        )

        # Vertical context and integration adapter
        orchestrator._vertical_context: VerticalContext = create_vertical_context()
        orchestrator._vertical_integration_adapter = VerticalIntegrationAdapter(orchestrator)

        # Lazy coordinator placeholders
        orchestrator._mode_workflow_team_coordinator = None

        # Interaction runtime boundary
        orchestrator._initialize_interaction_runtime()

        # Service layer delegation (Strangler Fig pattern)
        orchestrator._initialize_services()

        # Sync/Streaming coordinators (lazy initialization)
        orchestrator._turn_executor = None
        orchestrator._sync_chat_coordinator = None
        orchestrator._streaming_chat_coordinator = None
        orchestrator._unified_chat_coordinator = None
        orchestrator._protocol_adapter = None

        # Capability registry
        orchestrator.__init_capability_registry__()

    @classmethod
    def finalize(cls, orchestrator: AgentOrchestrator) -> None:
        """Run all post-initialization steps in order.

        Called at the end of AgentOrchestrator.__init__ after all
        components and runtime boundaries are initialized.
        """
        cls.create_facades(orchestrator)
        cls.wire_lifecycle(orchestrator)
        cls.assert_protocol_conformance(orchestrator)
        cls.setup_session_context(orchestrator)

        logger.info(
            "Orchestrator initialized with decomposed components: "
            "ConversationController, ToolPipeline, StreamingController, "
            "StreamingChatHandler, TaskAnalyzer, ContextCompactor, "
            "UsageAnalytics, ToolSequenceTracker, ToolOutputFormatter, "
            "RecoveryCoordinator, ChunkGenerator, ToolPlanner, "
            "TaskCoordinator, ObservabilityIntegration, "
            "WorkflowOptimization, VerticalContext, "
            "ModeWorkflowTeamCoordinator, CapabilityRegistry"
        )
