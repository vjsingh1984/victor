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

"""Coordination builder methods for OrchestratorFactory.

Provides creation methods for recovery, workflow, team coordination,
safety, middleware, context management, and workflow optimization components.

Part of CRITICAL-001: Monolithic Orchestrator decomposition.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING

from victor.agent.coordinators.factory_support import (
    create_exploration_coordinator as build_exploration_coordinator,
    create_exploration_state_passed_coordinator as build_exploration_state_passed_coordinator,
    create_safety_state_passed_coordinator as build_safety_state_passed_coordinator,
    create_system_prompt_coordinator as build_system_prompt_coordinator,
    create_system_prompt_state_passed_coordinator as build_system_prompt_state_passed_coordinator,
)

if TYPE_CHECKING:
    from pathlib import Path
    from victor.config.settings import Settings
    from victor.providers.base import BaseProvider
    from victor.agent.tool_calling import (
        BaseToolCallingAdapter,
        ToolCallingCapabilities,
    )
    from victor.agent.conversation.controller import ConversationController
    from victor.agent.context_compactor import ContextCompactor
    from victor.agent.recovery import RecoveryHandler
    from victor.agent.orchestrator_recovery import OrchestratorRecoveryIntegration
    from victor.agent.middleware_chain import MiddlewareChain
    from victor.agent.conversation.state_machine import ConversationStateMachine
    from victor.agent.task_completion import TaskCompletionDetector
    from victor.agent.read_cache import ReadResultCache
    from victor.agent.time_aware_executor import TimeAwareExecutor
    from victor.agent.thinking_detector import ThinkingPatternDetector
    from victor.agent.resource_manager import ResourceManager
    from victor.agent.budget_manager import ModeCompletionCriteria
    from victor.agent.conversation.assembler import TurnBoundaryContextAssembler
    from victor.agent.referential_intent_resolver import ReferentialIntentResolver
    from victor.agent.session_ledger import SessionLedger
    from victor.agent.mode_workflow_team_coordinator import ModeWorkflowTeamCoordinator
    from victor.observability.integration import ObservabilityIntegration
    from victor.storage.embeddings.intent_classifier import IntentClassifier
    from victor.agent.protocols.infrastructure_protocols import (
        SafetyCheckerProtocol,
        AutoCommitterProtocol,
        ReminderManagerProtocol,
        CodeExecutionManagerProtocol,
        WorkflowRegistryProtocol,
    )
    from victor.agent.services.protocols import (
        ChunkRuntimeProtocol as ChunkGeneratorProto,
        RLLearningRuntimeProtocol as RLCoordinatorProtocol,
        StreamingRecoveryRuntimeProtocol as StreamingRecoveryCoordinatorProto,
        TaskRuntimeProtocol as TaskCoordinatorProtocol,
    )

logger = logging.getLogger(__name__)


class CoordinationBuildersMixin:
    """Mixin providing coordination-related factory methods.

    Requires the host class to provide:
        - self.settings: Settings
        - self.provider: BaseProvider
        - self.model: str
        - self.provider_name: Optional[str]
        - self.container: DI container
    """

    def create_exploration_coordinator(self) -> Any:
        """Create the canonical read-only exploration runtime."""
        coordinator = build_exploration_coordinator()
        logger.debug("ExplorationCoordinator created")
        return coordinator

    def create_exploration_state_passed_coordinator(
        self,
        project_root: Optional["Path"] = None,
        max_results: int = 5,
    ) -> Any:
        """Create the state-passed exploration wrapper."""
        coordinator = build_exploration_state_passed_coordinator(
            settings=self.settings,
            project_root=project_root,
            max_results=max_results,
        )
        logger.debug("ExplorationStatePassedCoordinator created")
        return coordinator

    def create_system_prompt_coordinator(
        self,
        *,
        prompt_builder: Any = None,
        get_context_window: Optional[Callable[[], int]] = None,
        provider_name: str = "",
        model_name: str = "",
        get_tools: Optional[Callable[[], Optional[Any]]] = None,
        get_mode_controller: Optional[Callable[[], Optional[object]]] = None,
        task_analyzer: Optional[Any] = None,
        session_id: str = "",
    ) -> Any:
        """Create the compatibility system prompt runtime."""
        coordinator = build_system_prompt_coordinator(
            container=self.container,
            prompt_builder=prompt_builder,
            get_context_window=get_context_window,
            provider_name=provider_name,
            model_name=model_name,
            get_tools=get_tools,
            get_mode_controller=get_mode_controller,
            task_analyzer=task_analyzer,
            session_id=session_id,
        )
        logger.debug("SystemPromptCoordinator created")
        return coordinator

    def create_system_prompt_state_passed_coordinator(
        self,
        task_analyzer: Optional[Any] = None,
    ) -> Any:
        """Create the canonical state-passed system prompt coordinator."""
        coordinator = build_system_prompt_state_passed_coordinator(
            container=self.container,
            task_analyzer=task_analyzer,
        )
        logger.debug("SystemPromptStatePassedCoordinator created")
        return coordinator

    def create_safety_state_passed_coordinator(self) -> Any:
        """Create the canonical state-passed safety wrapper."""
        coordinator = build_safety_state_passed_coordinator()
        logger.debug("SafetyStatePassedCoordinator created")
        return coordinator

    def create_recovery_handler(self) -> Optional["RecoveryHandler"]:
        """Create recovery handler (from DI container)."""
        from victor.agent.protocols import RecoveryHandlerProtocol

        return self.container.get(RecoveryHandlerProtocol)

    def create_recovery_integration(
        self, recovery_handler: Optional["RecoveryHandler"]
    ) -> "OrchestratorRecoveryIntegration":
        """Create recovery integration submodule for clean delegation.

        Args:
            recovery_handler: RecoveryHandler instance (may be None if disabled)

        Returns:
            RecoveryIntegration instance
        """
        from victor.agent.orchestrator_recovery import create_recovery_integration

        integration = create_recovery_integration(
            recovery_handler=recovery_handler,
            settings=self.settings,
        )
        logger.debug("RecoveryIntegration created")
        return integration

    def create_recovery_coordinator(self) -> "StreamingRecoveryCoordinatorProto":
        """Create RecoveryCoordinator via DI container.

        Returns:
            StreamingRecoveryCoordinator instance for recovery coordination
        """
        from victor.agent.services.protocols import StreamingRecoveryRuntimeProtocol

        recovery_coordinator = self.container.get(StreamingRecoveryRuntimeProtocol)
        logger.debug("StreamingRecoveryCoordinator created via DI")
        return recovery_coordinator

    def create_chunk_generator(self) -> "ChunkGeneratorProto":
        """Create ChunkGenerator via DI container.

        Returns:
            ChunkGenerator instance for chunk generation
        """
        from victor.agent.services.protocols import ChunkRuntimeProtocol

        chunk_generator = self.container.get(ChunkRuntimeProtocol)
        logger.debug("ChunkGenerator created via DI")
        return chunk_generator

    def create_context_compactor(
        self,
        conversation_controller: "ConversationController",
        pruning_learner: Optional[Any] = None,
    ) -> "ContextCompactor":
        """Create context compactor for proactive context management.

        Args:
            conversation_controller: ConversationController instance for context tracking
            pruning_learner: Optional RL learner for adaptive pruning decisions

        Returns:
            ContextCompactor instance configured with settings
        """
        from victor.agent.context_compactor import (
            create_context_compactor,
            TruncationStrategy,
        )

        truncation_strategy_str = getattr(
            self.settings, "tool_truncation_strategy", "smart"
        ).lower()
        truncation_strategy_map = {
            "head": TruncationStrategy.HEAD,
            "tail": TruncationStrategy.TAIL,
            "both": TruncationStrategy.BOTH,
            "smart": TruncationStrategy.SMART,
        }
        truncation_strategy = truncation_strategy_map.get(
            truncation_strategy_str, TruncationStrategy.SMART
        )

        provider_name = str(getattr(self.settings, "provider", "")).lower()
        local_providers = {"ollama", "lmstudio", "vllm", "llamacpp", "local"}
        provider_type = "local" if any(p in provider_name for p in local_providers) else "cloud"

        compactor = create_context_compactor(
            controller=conversation_controller,
            proactive_threshold=getattr(self.settings, "context_proactive_threshold", 0.90),
            min_messages_after_compact=getattr(
                self.settings, "context_min_messages_after_compact", 8
            ),
            tool_result_max_chars=getattr(self.settings, "max_tool_output_chars", 8192),
            tool_result_max_lines=getattr(self.settings, "max_tool_output_lines", 200),
            truncation_strategy=truncation_strategy,
            preserve_code_blocks=True,
            enable_proactive=getattr(self.settings, "context_proactive_compaction", True),
            enable_tool_truncation=getattr(self.settings, "tool_result_truncation", True),
            pruning_learner=pruning_learner,
            provider_type=provider_type,
        )

        rl_status = "with RL learner" if pruning_learner else "without RL learner"
        logger.debug(
            f"ContextCompactor created {rl_status}, "
            f"truncation_strategy={truncation_strategy}, provider_type={provider_type}"
        )
        return compactor

    def create_middleware_chain(
        self,
    ) -> Tuple[Optional["MiddlewareChain"], Optional[Any]]:
        """Create middleware chain with vertical extensions.

        Returns:
            Tuple of (MiddlewareChain or None, CodeCorrectionMiddleware or None)
        """
        middleware_chain: Optional["MiddlewareChain"] = None
        code_correction_middleware: Optional[Any] = None
        code_correction_enabled = getattr(self.settings, "code_correction_enabled", True)

        try:
            from victor.agent.middleware_chain import MiddlewareChain
            from victor.core.verticals.protocols import VerticalExtensions

            middleware_chain = MiddlewareChain()

            extensions = self.container.get_optional(VerticalExtensions)
            if extensions and extensions.middleware:
                for middleware in extensions.middleware:
                    middleware_chain.add(middleware)
                    logger.debug(f"Added middleware from vertical: {type(middleware).__name__}")

                for mw in extensions.middleware:
                    if "CodeCorrection" in type(mw).__name__:
                        code_correction_middleware = mw
                        break
            else:
                if code_correction_enabled:
                    try:
                        from victor.agent.code_correction_middleware import (
                            CodeCorrectionMiddleware,
                            CodeCorrectionConfig,
                        )

                        code_correction_auto_fix = getattr(
                            self.settings, "code_correction_auto_fix", True
                        )
                        code_correction_max_iterations = getattr(
                            self.settings, "code_correction_max_iterations", 1
                        )

                        code_correction_middleware = CodeCorrectionMiddleware(
                            config=CodeCorrectionConfig(
                                enabled=True,
                                auto_fix=code_correction_auto_fix,
                                max_iterations=code_correction_max_iterations,
                            )
                        )
                        logger.debug("Using fallback CodeCorrectionMiddleware (no vertical loaded)")
                    except ImportError as e:
                        logger.warning(f"CodeCorrectionMiddleware unavailable: {e}")
        except ImportError as e:
            logger.warning(f"Middleware chain unavailable: {e}")

        return middleware_chain, code_correction_middleware

    def create_safety_checker(self) -> "SafetyCheckerProtocol":
        """Create safety checker with vertical safety patterns.

        Returns:
            SafetyChecker instance with registered patterns
        """
        from victor.agent.protocols import SafetyCheckerProtocol

        safety_checker = self.container.get(SafetyCheckerProtocol)

        try:
            from victor.core.verticals.protocols import VerticalExtensions

            extensions = self.container.get_optional(VerticalExtensions)
            if extensions and extensions.safety_extensions:
                for safety_ext in extensions.safety_extensions:
                    for pattern in safety_ext.get_bash_patterns():
                        safety_checker.add_custom_pattern(
                            pattern.pattern,
                            pattern.description,
                            pattern.risk_level,
                            pattern.category,
                        )
                    logger.debug(
                        f"Added safety patterns from vertical: {safety_ext.get_category()}"
                    )
        except Exception as e:
            logger.debug(f"Could not load vertical safety extensions: {e}")

        return safety_checker

    def create_auto_committer(self) -> Optional["AutoCommitterProtocol"]:
        """Create auto committer for AI-assisted commits.

        Returns:
            AutoCommitter instance or None
        """
        from victor.agent.protocols import AutoCommitterProtocol
        from victor.agent.auto_commit import get_auto_committer

        auto_commit_enabled = getattr(self.settings, "auto_commit_enabled", False)

        if auto_commit_enabled:
            auto_committer = self.container.get(AutoCommitterProtocol)
            logger.debug("AutoCommitter enabled for AI-assisted commits")
            return auto_committer
        else:
            return get_auto_committer()

    def create_code_execution_manager(self) -> "CodeExecutionManagerProtocol":
        """Create code execution manager for Docker-based code execution.

        Returns:
            CodeExecutionManager instance (started)
        """
        from victor.agent.protocols import CodeExecutionManagerProtocol

        return self.container.get(CodeExecutionManagerProtocol)

    def create_workflow_registry(self) -> "WorkflowRegistryProtocol":
        """Create workflow registry for managing workflow patterns.

        Returns:
            WorkflowRegistry instance
        """
        from victor.agent.protocols import WorkflowRegistryProtocol

        return self.container.get(WorkflowRegistryProtocol)

    def create_rl_coordinator(self) -> Optional["RLCoordinatorProtocol"]:
        """Create RL coordinator for reinforcement learning framework.

        Returns:
            RL coordinator instance if enabled, None otherwise
        """
        if not getattr(self.settings, "enable_continuation_rl_learning", False):
            logger.debug("RL learning disabled")
            return None

        try:
            from victor.agent.services.protocols import RLLearningRuntimeProtocol

            coordinator = self.container.get(RLLearningRuntimeProtocol)
            logger.info("RL: Coordinator initialized with unified database")
            return coordinator
        except Exception as e:
            logger.warning(f"RL: Failed to initialize RL coordinator: {e}")
            return None

    def create_reminder_manager(
        self, provider: str, task_complexity: str, tool_budget: int
    ) -> "ReminderManagerProtocol":
        """Create context reminder manager from DI container.

        Args:
            provider: Provider name
            task_complexity: Task complexity level
            tool_budget: Tool call budget

        Returns:
            ReminderManager instance
        """
        from victor.agent.protocols import ReminderManagerProtocol

        with self.container.create_scope() as scope:
            reminder_manager = scope.get(ReminderManagerProtocol)

        logger.debug(f"ReminderManager created for {provider} with complexity {task_complexity}")
        return reminder_manager

    def create_task_coordinator(self) -> "TaskCoordinatorProtocol":
        """Create TaskCoordinator via DI container.

        Returns:
            TaskCoordinator instance for task coordination
        """
        from victor.agent.services.protocols import TaskRuntimeProtocol

        task_coordinator = self.container.get(TaskRuntimeProtocol)
        logger.debug("TaskCoordinator created via DI")
        return task_coordinator

    def create_intent_classifier(self) -> "IntentClassifier":
        """Create intent classifier for semantic continuation/completion detection.

        Returns:
            IntentClassifier singleton instance
        """
        from victor.storage.embeddings.intent_classifier import IntentClassifier

        classifier = IntentClassifier.get_instance()
        logger.debug("IntentClassifier singleton retrieved")
        return classifier

    def create_mode_workflow_team_coordinator(
        self,
        vertical_context: Any,
    ) -> "ModeWorkflowTeamCoordinator":
        """Create ModeWorkflowTeamCoordinator for intelligent task coordination.

        Args:
            vertical_context: VerticalContext with team specs and workflows

        Returns:
            ModeWorkflowTeamCoordinator instance
        """
        from victor.agent.mode_workflow_team_coordinator import create_coordinator

        team_learner = None
        try:
            from victor.agent.services.protocols import RLLearningRuntimeProtocol

            rl_coordinator = self.container.get_optional(RLLearningRuntimeProtocol)
            if rl_coordinator:
                team_learner = rl_coordinator.get_learner("team_composition")
        except Exception as e:
            logger.debug(f"Could not get team composition learner: {e}")

        selection_strategy = getattr(self.settings, "team_selection_strategy", "hybrid")

        coordinator = create_coordinator(
            vertical_context=vertical_context,
            team_learner=team_learner,
            selection_strategy=selection_strategy,
        )

        logger.debug(f"ModeWorkflowTeamCoordinator created with strategy={selection_strategy}")
        return coordinator

    def setup_subagent_orchestration(self) -> tuple[Optional[Any], bool]:
        """Setup sub-agent orchestration with lazy initialization.

        Returns:
            Tuple of (None, enabled_flag) for lazy initialization pattern
        """
        enabled = getattr(self.settings, "subagent_orchestration_enabled", True)
        logger.debug(f"Sub-agent orchestration setup: enabled={enabled}")
        return (None, enabled)

    def setup_semantic_selection(self) -> tuple[bool, Optional[Any]]:
        """Setup semantic tool selection and background embedding preload.

        Returns:
            Tuple of (use_semantic_selection, embedding_preload_task_placeholder)
        """
        use_semantic = getattr(self.settings, "use_semantic_tool_selection", False)
        logger.debug(f"Semantic selection setup: enabled={use_semantic}")
        return (use_semantic, None)

    def wire_component_dependencies(
        self,
        recovery_handler: Optional["RecoveryHandler"],
        context_compactor: "ContextCompactor",
        observability: Optional["ObservabilityIntegration"],
        conversation_state: Optional["ConversationStateMachine"],
    ) -> None:
        """Wire component dependencies after initialization.

        Args:
            recovery_handler: RecoveryHandler instance
            context_compactor: ContextCompactor instance
            observability: ObservabilityIntegration instance
            conversation_state: ConversationStateMachine instance
        """
        if recovery_handler and hasattr(recovery_handler, "set_context_compactor"):
            recovery_handler.set_context_compactor(context_compactor)
            logger.debug("RecoveryHandler wired with ContextCompactor")

        if observability and conversation_state:
            observability.wire_state_machine(conversation_state)
            logger.debug("Observability integration wired with ConversationStateMachine")

    # =========================================================================
    # Workflow Optimization Components
    # =========================================================================

    def create_task_completion_detector(self) -> "TaskCompletionDetector":
        """Create TaskCompletionDetector for detecting task completion.

        Returns:
            TaskCompletionDetector instance
        """
        from victor.agent.task_completion import create_task_completion_detector

        detector = create_task_completion_detector()
        logger.debug("TaskCompletionDetector created")
        return detector

    def create_read_cache(self) -> "ReadResultCache":
        """Create ReadResultCache for file read deduplication.

        Returns:
            ReadResultCache instance with settings-derived configuration
        """
        from victor.agent.read_cache import create_read_cache

        ttl_seconds = getattr(self.settings, "read_cache_ttl", 300.0)
        max_entries = getattr(self.settings, "read_cache_max_entries", 100)

        cache = create_read_cache(ttl_seconds=ttl_seconds, max_entries=max_entries)
        logger.debug(f"ReadResultCache created (ttl={ttl_seconds}s, max={max_entries})")
        return cache

    def create_time_aware_executor(
        self, timeout_seconds: Optional[float] = None
    ) -> "TimeAwareExecutor":
        """Create TimeAwareExecutor for time-aware execution management.

        Args:
            timeout_seconds: Execution time budget (None for unlimited)

        Returns:
            TimeAwareExecutor instance
        """
        from victor.agent.time_aware_executor import create_time_aware_executor

        if timeout_seconds is None:
            timeout_seconds = getattr(self.settings, "execution_timeout", None)

        executor = create_time_aware_executor(timeout_seconds=timeout_seconds)
        if timeout_seconds:
            logger.debug(f"TimeAwareExecutor created with {timeout_seconds}s budget")
        else:
            logger.debug("TimeAwareExecutor created (no timeout)")
        return executor

    def create_thinking_detector(self) -> "ThinkingPatternDetector":
        """Create ThinkingPatternDetector for detecting thinking loops.

        Returns:
            ThinkingPatternDetector instance
        """
        from victor.agent.thinking_detector import create_thinking_detector

        repetition_threshold = getattr(self.settings, "thinking_repetition_threshold", 3)
        similarity_threshold = getattr(self.settings, "thinking_similarity_threshold", 0.65)

        detector = create_thinking_detector(
            repetition_threshold=repetition_threshold,
            similarity_threshold=similarity_threshold,
        )
        logger.debug(
            f"ThinkingPatternDetector created "
            f"(repetition={repetition_threshold}, similarity={similarity_threshold})"
        )
        return detector

    def create_resource_manager(self) -> "ResourceManager":
        """Get ResourceManager for resource lifecycle management.

        Returns:
            ResourceManager singleton instance
        """
        from victor.agent.resource_manager import get_resource_manager

        manager = get_resource_manager()
        logger.debug("ResourceManager retrieved (singleton)")
        return manager

    def create_mode_completion_criteria(self) -> "ModeCompletionCriteria":
        """Create ModeCompletionCriteria for mode-specific early exit.

        Returns:
            ModeCompletionCriteria instance
        """
        from victor.agent.budget_manager import create_mode_completion_criteria

        criteria = create_mode_completion_criteria()
        logger.debug("ModeCompletionCriteria created")
        return criteria

    def create_context_assembler(
        self,
        ledger: Optional["SessionLedger"] = None,
        controller: Optional["ConversationController"] = None,
    ) -> "TurnBoundaryContextAssembler":
        """Create TurnBoundaryContextAssembler for context selection.

        Args:
            ledger: SessionLedger instance
            controller: ConversationController (for semantic retrieval)
        """
        from victor.agent.conversation.assembler import TurnBoundaryContextAssembler
        from victor.agent.conversation.scoring import score_messages, CONTROLLER_WEIGHTS
        from victor.agent.conversation.types import ConversationMessage

        def _canonical_score_fn(messages, current_query=None):
            """Bridge provider Messages to canonical scorer."""
            conv_msgs = [ConversationMessage.from_provider_message(m) for m in messages]
            scored = score_messages(conv_msgs, current_query, weights=CONTROLLER_WEIGHTS)
            # Map back to original Message objects
            conv_to_orig = {id(cm): msg for cm, msg in zip(conv_msgs, messages)}
            return [(conv_to_orig[id(cm)], s) for cm, s in scored]

        assembler = TurnBoundaryContextAssembler(
            session_ledger=ledger,
            score_fn=_canonical_score_fn,
            conversation_controller=controller,
        )
        logger.debug("TurnBoundaryContextAssembler created")
        return assembler

    def create_referential_intent_resolver(
        self, ledger: Optional["SessionLedger"] = None
    ) -> "ReferentialIntentResolver":
        """Create ReferentialIntentResolver for anaphoric reference resolution."""
        from victor.agent.referential_intent_resolver import ReferentialIntentResolver

        resolver = ReferentialIntentResolver(session_ledger=ledger)
        logger.debug("ReferentialIntentResolver created")
        return resolver

    def create_streaming_loop_coordinator(
        self,
        termination_handler: Any,
        tool_call_handler: Any,
        recovery_handler: Any,
        chunk_generator: Any,
        intent_classifier: Any,
        continuation_strategy: Any,
    ) -> None:
        """ARCHIVED: StreamingLoopCoordinator was extracted but never integrated.

        Raises:
            NotImplementedError: This component was never integrated.
        """
        raise NotImplementedError(
            "StreamingLoopCoordinator was archived. "
            "Streaming loop remains in AgentOrchestrator. "
            "See: archive/obsolete/2024_12_cleanup/streaming_loop_coordinator.py"
        )
