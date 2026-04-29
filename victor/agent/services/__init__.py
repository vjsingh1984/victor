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

"""Service implementations for SOLID refactoring.

This package contains the service implementations that extract
functionality from the monolithic AgentOrchestrator into
focused, single-responsibility services.

Services:
    ChatService: Chat flow coordination and streaming
    ToolService: Tool selection and execution
    ContextService: Context management and metrics
    ProviderService: Provider management and switching
    RecoveryService: Error recovery and resilience
    RL runtime helpers: Service-first access to the global RL coordinator and
        prompt rollout helpers for benchmark-gated prompt optimization
    SessionService: Session lifecycle management
    LLMDecisionService: LLM-assisted decision making fallback

Deprecated compatibility exports retained here:
    PromptRuntimeAdapter: Canonical DI/runtime adapter for PromptRuntimeProtocol
    PromptRuntimeSupport: Canonical internal fallback support for orchestrator
        prompt helper seams when UnifiedPromptPipeline is unavailable
    SystemPromptCoordinator: Deprecated wrapper over UnifiedPromptPipeline
    StreamingRecoveryContext: Moved to RecoveryService for canonical service ownership
"""

from victor.agent.services.chat_service import ChatService, ChatServiceConfig
from victor.agent.services.chat_stream_runtime import ServiceStreamingRuntime
from victor.agent.services.chunk_runtime import ChunkGenerator
from victor.agent.services.context_service import ContextService, ContextServiceConfig
from victor.agent.services.decision_service import (
    LLMDecisionService,
    LLMDecisionServiceConfig,
)
from victor.agent.services.exploration_runtime import ExplorationCoordinator, ExplorationResult
from victor.agent.services.metrics_service import MetricsCoordinator, create_metrics_coordinator
from victor.agent.services.provider_service import ProviderService
from victor.agent.services.runtime_intelligence import (
    PromptOptimizationBundle,
    RuntimeIntelligenceService,
    RuntimeIntelligenceSnapshot,
)
from victor.agent.services.rl_runtime import (
    AsyncWriterQueue,
    RLCoordinator,
    analyze_prompt_rollout_experiment,
    analyze_prompt_rollout_experiment_async,
    apply_prompt_rollout_recommendation,
    apply_prompt_rollout_recommendation_async,
    process_prompt_candidate_evaluation_suite,
    process_prompt_candidate_evaluation_suite_async,
    create_prompt_rollout_experiment,
    create_prompt_rollout_experiment_async,
    get_rl_coordinator,
    get_rl_coordinator_async,
    reset_rl_coordinator,
)
from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter
from victor.agent.services.prompt_runtime import (
    PromptRuntimeAdapter,
    PromptRuntimeConfig,
    PromptRuntimeContext,
)
from victor.agent.services.prompt_runtime_support import PromptRuntimeSupport
from victor.agent.services.recovery_service import (
    RecoveryService,
    RecoveryContextImpl,
    StreamingRecoveryContext,
)
from victor.agent.services.session_service import SessionService, SessionInfoImpl
from victor.agent.services.system_prompt_runtime import SystemPromptCoordinator
from victor.agent.services.task_runtime import TaskCoordinator
from victor.agent.services.planning_runtime import (
    PlanningConfig,
    PlanningCoordinator,
    PlanningMode,
    PlanningResult,
)
from victor.agent.services.tool_service import (
    ToolBudgetExceededError,
    ToolResultContext,
    ToolService,
    ToolServiceConfig,
)
from victor.agent.services.tool_contracts import NormalizedArgs, ToolCallValidation
from victor.agent.services.tool_observability import ToolObservabilityHandler
from victor.agent.services.tool_planning_runtime import ToolPlanner
from victor.agent.services.tool_retry import ToolRetryExecutor
from victor.agent.services.turn_execution_runtime import TurnExecutor, TurnResult

__all__ = [
    "ChatService",
    "ChatServiceConfig",
    "ServiceStreamingRuntime",
    "ChunkGenerator",
    "ContextService",
    "ContextServiceConfig",
    "ExplorationCoordinator",
    "ExplorationResult",
    "LLMDecisionService",
    "LLMDecisionServiceConfig",
    "MetricsCoordinator",
    "PlanningConfig",
    "PlanningCoordinator",
    "PlanningMode",
    "PlanningResult",
    "PromptRuntimeAdapter",
    "PromptRuntimeConfig",
    "PromptRuntimeContext",
    "PromptRuntimeSupport",
    "OrchestratorProtocolAdapter",
    "AsyncWriterQueue",
    "ProviderService",
    "PromptOptimizationBundle",
    "RLCoordinator",
    "RecoveryService",
    "RecoveryContextImpl",
    "RuntimeIntelligenceService",
    "RuntimeIntelligenceSnapshot",
    "StreamingRecoveryContext",
    "SessionService",
    "SessionInfoImpl",
    "SystemPromptCoordinator",
    "TaskCoordinator",
    "ToolBudgetExceededError",
    "ToolCallValidation",
    "ToolPlanner",
    "ToolResultContext",
    "ToolService",
    "ToolServiceConfig",
    "NormalizedArgs",
    "ToolObservabilityHandler",
    "ToolRetryExecutor",
    "TurnExecutor",
    "TurnResult",
    "create_metrics_coordinator",
    "analyze_prompt_rollout_experiment",
    "analyze_prompt_rollout_experiment_async",
    "apply_prompt_rollout_recommendation",
    "apply_prompt_rollout_recommendation_async",
    "process_prompt_candidate_evaluation_suite",
    "process_prompt_candidate_evaluation_suite_async",
    "create_prompt_rollout_experiment",
    "create_prompt_rollout_experiment_async",
    "get_rl_coordinator",
    "get_rl_coordinator_async",
    "reset_rl_coordinator",
]
