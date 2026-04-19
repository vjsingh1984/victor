"""Component assembly phases for AgentOrchestrator.

Groups sequential factory component creation into logical phases,
reducing orchestrator __init__ from ~400 lines to method calls.
Each phase creates a related set of components and assigns them
to the orchestrator. Phases must run in order due to inter-dependencies.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class ComponentAssembler:
    """Groups factory component creation into logical assembly phases."""

    @staticmethod
    def assemble_tools(
        orchestrator: AgentOrchestrator,
        provider: BaseProvider,
        model: str,
    ) -> None:
        """Create tool infrastructure: cache, graph, registry, execution, selection.

        Must run after provider setup, conversation, and memory runtime.
        """
        factory = orchestrator._factory

        # Tool cache and dependency graph
        orchestrator.tool_cache = factory.create_tool_cache()
        orchestrator.tool_graph = factory.create_tool_dependency_graph()

        # Code execution manager (Docker-based)
        orchestrator.code_manager = factory.create_code_execution_manager()

        # Tool registry and registrar
        orchestrator.tools = factory.create_tool_registry()
        orchestrator.tool_registry = orchestrator.tools  # backward compat alias

        orchestrator.tool_registrar = factory.create_tool_registrar(
            orchestrator.tools, orchestrator.tool_graph, provider, model
        )
        orchestrator.tool_registrar.set_background_task_callback(
            orchestrator._create_background_task
        )
        orchestrator.tool_registrar._register_tool_dependencies()

        # Tool registration and plugins
        orchestrator._register_default_tools()
        orchestrator.tool_registrar._load_tool_configurations()
        orchestrator.tools.register_before_hook(orchestrator._log_tool_call)

        orchestrator.plugin_manager = factory.initialize_plugin_system(orchestrator.tool_registrar)

        # Argument normalizer, middleware, safety, auto-committer
        orchestrator.argument_normalizer = factory.create_argument_normalizer(provider)
        orchestrator._middleware_chain, orchestrator._code_correction_middleware = (
            factory.create_middleware_chain()
        )
        orchestrator._safety_checker = factory.create_safety_checker()
        orchestrator._auto_committer = factory.create_auto_committer()

        # Tool executor and parallel executor
        orchestrator.tool_executor = factory.create_tool_executor(
            tools=orchestrator.tools,
            argument_normalizer=orchestrator.argument_normalizer,
            tool_cache=orchestrator.tool_cache,
            safety_checker=orchestrator._safety_checker,
            code_correction_middleware=orchestrator._code_correction_middleware,
        )
        orchestrator.parallel_executor = factory.create_parallel_executor(
            orchestrator.tool_executor
        )

        # Response completer
        orchestrator.response_completer = factory.create_response_completer()

        # Semantic selection and tool selector
        orchestrator.use_semantic_selection, orchestrator._embedding_preload_task = (
            factory.setup_semantic_selection()
        )
        orchestrator._runtime_preload_task: Optional[asyncio.Task] = None
        orchestrator.semantic_selector = factory.create_semantic_selector()

        orchestrator.unified_tracker = factory.create_unified_tracker(
            orchestrator.tool_calling_caps
        )
        orchestrator.tool_selector = factory.create_tool_selector(
            tools=orchestrator.tools,
            semantic_selector=orchestrator.semantic_selector,
            conversation_state=orchestrator.conversation_state,
            unified_tracker=orchestrator.unified_tracker,
            model=orchestrator.model,
            provider_name=orchestrator.provider_name,
            tool_selection=orchestrator.tool_selection,
            on_selection_recorded=orchestrator._record_tool_selection,
        )

        # Access control and budget
        orchestrator._tool_access_controller = factory.create_tool_access_controller(
            registry=orchestrator.tools
        )
        orchestrator._budget_manager = factory.create_budget_manager()

    @staticmethod
    def assemble_conversation(
        orchestrator: AgentOrchestrator,
        provider: BaseProvider,
        model: str,
    ) -> None:
        """Create conversation management: controller, lifecycle, pipeline, streaming.

        Must run after assemble_tools.
        """
        factory = orchestrator._factory

        # Conversation controller
        orchestrator._conversation_controller = factory.create_conversation_controller(
            provider=provider,
            model=model,
            conversation=orchestrator.conversation,
            conversation_state=orchestrator.conversation_state,
            memory_manager=orchestrator.memory_manager,
            memory_session_id=orchestrator._memory_session_id,
            system_prompt=orchestrator._system_prompt,
            context_reminder_manager=(
                orchestrator.reminder_manager if hasattr(orchestrator, "reminder_manager") else None
            ),
            hierarchical_manager=(factory.create_hierarchical_compaction_manager()),
        )

        # Session ledger and lifecycle manager
        orchestrator._session_ledger = factory.create_session_ledger()
        orchestrator._lifecycle_manager = factory.create_lifecycle_manager(
            conversation_controller=orchestrator._conversation_controller,
            metrics_collector=(
                orchestrator._metrics_coordinator.metrics_collector
                if hasattr(orchestrator, "_metrics_coordinator")
                else None
            ),
            context_compactor=(
                orchestrator._context_compactor
                if hasattr(orchestrator, "_context_compactor")
                else None
            ),
            sequence_tracker=(
                orchestrator._sequence_tracker
                if hasattr(orchestrator, "_sequence_tracker")
                else None
            ),
            usage_analytics=(
                orchestrator._usage_analytics if hasattr(orchestrator, "_usage_analytics") else None
            ),
            reminder_manager=(
                orchestrator._reminder_manager
                if hasattr(orchestrator, "_reminder_manager")
                else None
            ),
        )

        # [CANONICAL] Update SessionService with lifecycle_manager after it's created
        if hasattr(orchestrator, "_session_service") and orchestrator._session_service:
            orchestrator._session_service.set_lifecycle_manager(orchestrator._lifecycle_manager)

        # Tool deduplication and pipeline
        orchestrator._deduplication_tracker = factory.create_tool_deduplication_tracker()
        orchestrator._tool_pipeline = factory.create_tool_pipeline(
            tools=orchestrator.tools,
            tool_executor=orchestrator.tool_executor,
            tool_budget=orchestrator.tool_budget,
            tool_cache=orchestrator.tool_cache,
            argument_normalizer=orchestrator.argument_normalizer,
            on_tool_start=orchestrator._on_tool_start_callback,
            on_tool_complete=orchestrator._on_tool_complete_callback,
            deduplication_tracker=orchestrator._deduplication_tracker,
            middleware_chain=orchestrator._middleware_chain,
            search_router=orchestrator.search_router,
        )

        # Wire pending semantic cache
        if (
            hasattr(orchestrator, "_pending_semantic_cache")
            and orchestrator._pending_semantic_cache is not None
        ):
            orchestrator._tool_pipeline.set_semantic_cache(orchestrator._pending_semantic_cache)
            logger.info("[AgentOrchestrator] Semantic tool result cache enabled")
            orchestrator._pending_semantic_cache = None

        # Streaming infrastructure
        orchestrator._streaming_controller = factory.create_streaming_controller(
            streaming_metrics_collector=(orchestrator.streaming_metrics_collector),
            on_session_complete=(orchestrator._on_streaming_session_complete),
        )
        orchestrator._streaming_coordinator = factory.create_streaming_coordinator(
            streaming_controller=orchestrator._streaming_controller,
        )
        orchestrator._streaming_handler = factory.create_streaming_chat_handler(
            message_adder=orchestrator
        )

        from victor.agent.streaming import IterationCoordinator

        orchestrator._iteration_coordinator: Optional[IterationCoordinator] = None

        # Task analyzer and system prompt coordinator
        from victor.agent.task_analyzer import get_task_analyzer
        from victor.agent.coordinators.system_prompt_coordinator import (
            SystemPromptCoordinator,
        )

        orchestrator._task_analyzer = get_task_analyzer()
        orchestrator._system_prompt_coordinator = SystemPromptCoordinator(
            prompt_builder=orchestrator.prompt_builder,
            get_context_window=orchestrator._get_model_context_window,
            provider_name=orchestrator.provider_name,
            model_name=orchestrator.model,
            get_tools=lambda: orchestrator.tools,
            get_mode_controller=lambda: orchestrator.mode_controller,
            task_analyzer=orchestrator._task_analyzer,
            session_id=getattr(orchestrator, "_session_id", ""),
        )

    @staticmethod
    def assemble_intelligence(
        orchestrator: AgentOrchestrator,
    ) -> None:
        """Create intelligence/resilience: RL, context, analytics, resilience, observability.

        Must run after assemble_conversation.
        """
        factory = orchestrator._factory

        # RL coordinator
        orchestrator._rl_coordinator = factory.create_rl_coordinator()

        # Context pruning learner
        pruning_learner = None
        if orchestrator._rl_coordinator is not None:
            try:
                pruning_learner = orchestrator._rl_coordinator.get_learner("context_pruning")
            except (KeyError, AttributeError, TypeError) as e:
                logger.debug("context_pruning learner unavailable: %s", e)

        # Context compactor and manager
        orchestrator._context_compactor = factory.create_context_compactor(
            conversation_controller=orchestrator._conversation_controller,
            pruning_learner=pruning_learner,
        )

        # Context assembler (turn-boundary assembly for provider calls)
        try:
            session_ledger = getattr(orchestrator, "_session_ledger", None)
            orchestrator._context_assembler = factory.create_context_assembler(
                ledger=session_ledger,
                controller=orchestrator._conversation_controller,
            )
        except Exception as e:
            logger.debug("Context assembler creation failed (graceful): %s", e)
            orchestrator._context_assembler = None

        from victor.agent.context_manager import create_context_manager

        orchestrator._context_manager = create_context_manager(
            provider_name=orchestrator.provider_name,
            model=orchestrator.model,
            conversation_controller=orchestrator._conversation_controller,
            context_compactor=orchestrator._context_compactor,
            debug_logger=orchestrator.debug_logger,
            settings=orchestrator.settings,
        )

        # Usage analytics and sequence tracker
        orchestrator._usage_analytics = factory.create_usage_analytics()
        orchestrator._sequence_tracker = factory.create_sequence_tracker()

        # Tool output formatter
        orchestrator._tool_output_formatter = factory.create_tool_output_formatter(
            orchestrator._context_compactor
        )

        # Resilience and coordination runtime boundaries
        orchestrator._initialize_resilience_runtime(
            context_compactor=orchestrator._context_compactor
        )
        orchestrator._initialize_coordination_runtime()

        # Observability
        orchestrator._observability = factory.create_observability()
