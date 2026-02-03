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

"""Consolidated imports for AgentOrchestrator.

This module consolidates all imports used by the orchestrator into logical groups,
making it easier to manage dependencies and reduce import statement clutter.

Import Categories:
1. Standard library and third-party
2. Coordinators (runtime imports)
3. Type-checking only imports
4. Protocols
5. Runtime component imports
6. Configuration and enums
7. Provider and tool imports
8. Observability and analytics
9. Streaming submodule

Usage:
    from victor.agent.orchestrator_imports import (
        # Coordinators
        CheckpointCoordinator,
        StateCoordinator,

        # Runtime components
        ArgumentNormalizer,
        MessageHistory,

        # Protocols
        ResponseSanitizerProtocol,

        # etc.
    )
"""

import logging
from typing import TYPE_CHECKING

from rich.console import Console

# =============================================================================
# COORDINATORS (Runtime imports - used directly by orchestrator)
# =============================================================================

from victor.agent.coordinators.checkpoint_coordinator import (
    CheckpointCoordinator,
)
from victor.agent.coordinators.config_coordinator import (
    ToolAccessConfigCoordinator,
)
from victor.agent.coordinators.evaluation_coordinator import (
    EvaluationCoordinator,
)
from victor.agent.coordinators.metrics_coordinator import (
    MetricsCoordinator,
)
from victor.agent.coordinators.response_coordinator import (
    ResponseCoordinator,
)
from victor.agent.coordinators.middleware_coordinator import (
    MiddlewareCoordinator,
)
from victor.agent.coordinators.state_coordinator import (
    StateCoordinator,
    StateScope,
)
from victor.agent.coordinators.workflow_coordinator import (
    WorkflowCoordinator,
)
from victor.agent.coordinators.conversation_coordinator import (
    ConversationCoordinator,
)
from victor.agent.coordinators.search_coordinator import (
    SearchCoordinator,
)
from victor.agent.coordinators.team_coordinator import (
    TeamCoordinator,
)

# Export all public symbols
__all__ = [
    # Third-party
    "Console",
    "logger",
    # Coordinators
    "CheckpointCoordinator",
    "ToolAccessConfigCoordinator",
    "EvaluationCoordinator",
    "MetricsCoordinator",
    "ResponseCoordinator",
    "MiddlewareCoordinator",
    "StateCoordinator",
    "StateScope",
    "WorkflowCoordinator",
    "ConversationCoordinator",
    "SearchCoordinator",
    "TeamCoordinator",
    # Agent components
    "ArgumentNormalizer",
    "NormalizationStrategy",
    "MessageHistory",
    "ConversationStore",
    "MessageRole",
    # Mixins
    "ModeAwareMixin",
    "CapabilityRegistryMixin",
    "ComponentAccessorMixin",
    "StateDelegationMixin",
    # DI container
    "ensure_bootstrapped",
    "get_service_optional",
    "MetricsServiceProtocol",
    "LoggerServiceProtocol",
    # Protocols
    "ResponseSanitizerProtocol",
    "ComplexityClassifierProtocol",
    "ActionAuthorizerProtocol",
    "SearchRouterProtocol",
    "ProjectContextProtocol",
    "ArgumentNormalizerProtocol",
    "ConversationStateMachineProtocol",
    "TaskTrackerProtocol",
    "CodeExecutionManagerProtocol",
    "WorkflowRegistryProtocol",
    "UsageAnalyticsProtocol",
    "ToolSequenceTrackerProtocol",
    "ContextCompactorProtocol",
    "RecoveryHandlerProtocol",
    # Configuration and enums
    "get_provider_limits",
    "Settings",
    "ToolCallingMatrix",
    "ConversationStateMachine",
    "ConversationStage",
    "ActionIntent",
    "get_task_type_hint",
    "SystemPromptBuilder",
    "SearchRoute",
    "SearchType",
    "TaskComplexity",
    "DEFAULT_BUDGETS",
    "StreamMetrics",
    "MetricsCollectorConfig",
    "TrackerTaskType",
    "UnifiedTaskTracker",
    "extract_prompt_requirements",
    # Decomposed components
    "ConversationConfig",
    "ContextMetrics",
    "CompactionStrategy",
    "TruncationStrategy",
    "create_context_compactor",
    "calculate_parallel_read_budget",
    "ContextManager",
    "ContextManagerConfig",
    "create_context_manager",
    "ContinuationStrategy",
    "ExtractedToolCall",
    "get_rl_coordinator",
    "AnalyticsConfig",
    "create_sequence_tracker",
    "SessionStateManager",
    "create_session_state_manager",
    # Phase 1 extractions
    "ConfigurationManager",
    "create_configuration_manager",
    "MemoryManager",
    "SessionRecoveryManager",
    "create_memory_manager",
    "create_session_recovery_manager",
    # Recovery
    "RecoveryOutcome",
    "FailureType",
    "StrategyRecoveryAction",
    "VerticalContext",
    "create_vertical_context",
    "VerticalIntegrationAdapter",
    "create_recovery_integration",
    "OrchestratorRecoveryAction",
    # Tool output formatting
    "ToolOutputFormatterConfig",
    "FormattingContext",
    "create_tool_output_formatter",
    # Pipeline
    "ToolPipelineConfig",
    "ToolCallResult",
    "StreamingControllerConfig",
    "StreamingSession",
    "get_task_analyzer",
    "ToolRegistrarConfig",
    "ProviderManagerConfig",
    "ProviderState",
    # Observability
    "ObservabilityIntegration",
    "IntegrationConfig",
    # Tool execution
    "get_critical_tools",
    "ToolCallParseResult",
    "ValidationMode",
    "calculate_max_context_chars",
    "infer_git_operation",
    "get_tool_status_message",
    "OrchestratorFactory",
    "create_parallel_executor",
    "ToolFailureContext",
    "create_response_completer",
    # Analytics and logging
    "UsageLogger",
    "StreamingMetricsCollector",
    # Storage and caching
    "ToolCache",
    # Providers
    "BaseProvider",
    "CompletionResponse",
    "Message",
    "StreamChunk",
    "ToolDefinition",
    "ProviderRegistry",
    "ProviderAuthError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "ToolNotFoundError",
    "ToolValidationError",
    # Tools
    "CostTier",
    "ToolRegistry",
    "CodeSandbox",
    "get_mcp_tool_definitions",
    "ToolPluginRegistry",
    "SemanticToolSelector",
    "ToolNames",
    "TOOL_ALIASES",
    "get_alias_resolver",
    "get_progressive_registry",
    # Workflows and embeddings
    "IntentClassifier",
    "IntentType",
    "WorkflowRegistry",
    "register_builtin_workflows",
    # Project context
    "ProjectContext",
    # Streaming submodule
    "CoordinatorConfig",
    "ContinuationHandler",
    "ContinuationResult",
    "IntentClassificationHandler",
    "IntentClassificationResult",
    "IterationCoordinator",
    "ProgressMetrics",
    "StreamingChatContext",
    "StreamingChatHandler",
    "ToolExecutionHandler",
    "ToolExecutionResult",
    "TrackingState",
    "apply_tracking_state_updates",
    "create_continuation_handler",
    "create_coordinator",
    "create_intent_classification_handler",
    "create_stream_context",
    "create_tool_execution_handler",
    "create_tracking_state",
    # Adapters
    "CoordinatorAdapter",
    "IntelligentPipelineAdapter",
    "ResultConverters",
    "OrchestratorProtocolAdapter",
    "create_orchestrator_protocol_adapter",
]

# =============================================================================
# TYPE-CHECKING ONLY IMPORTS
# These are only used for type hints and are not needed at runtime
# =============================================================================

if TYPE_CHECKING:

    # Factory-created components (type hints only)
    pass

# =============================================================================
# RUNTIME AGENT COMPONENTS
# =============================================================================

from victor.agent.argument_normalizer import ArgumentNormalizer, NormalizationStrategy
from victor.agent.message_history import MessageHistory
from victor.agent.conversation_memory import ConversationStore, MessageRole

# Mixins (used at class definition time)
from victor.protocols.mode_aware import ModeAwareMixin
from victor.agent.capability_registry import CapabilityRegistryMixin
from victor.agent.mixins import ComponentAccessorMixin, StateDelegationMixin

# =============================================================================
# DI CONTAINER AND BOOTSTRAP
# =============================================================================

from victor.core.bootstrap import ensure_bootstrapped, get_service_optional
from victor.core.container import MetricsServiceProtocol, LoggerServiceProtocol

# =============================================================================
# PROTOCOLS
# =============================================================================

from victor.agent.protocols import (
    ResponseSanitizerProtocol,
    ComplexityClassifierProtocol,
    ActionAuthorizerProtocol,
    SearchRouterProtocol,
    ProjectContextProtocol,
    ArgumentNormalizerProtocol,
    ConversationStateMachineProtocol,
    TaskTrackerProtocol,
    CodeExecutionManagerProtocol,
    WorkflowRegistryProtocol,
    UsageAnalyticsProtocol,
    ToolSequenceTrackerProtocol,
    ContextCompactorProtocol,
    RecoveryHandlerProtocol,
)

# =============================================================================
# CONFIGURATION AND ENUMS
# =============================================================================

from victor.config.config_loaders import get_provider_limits
from victor.config.settings import Settings
from victor.config.model_capabilities import ToolCallingMatrix

from victor.agent.conversation_state import ConversationStateMachine
from victor.agent.conversation_state import ConversationStage
from victor.agent.action_authorizer import ActionIntent
from victor.agent.prompt_builder import get_task_type_hint, SystemPromptBuilder
from victor.agent.search_router import SearchRoute, SearchType
from victor.framework.task import TaskComplexity, DEFAULT_BUDGETS
from victor.agent.stream_handler import StreamMetrics
from victor.agent.metrics_collector import MetricsCollectorConfig
from victor.agent.unified_task_tracker import TrackerTaskType, UnifiedTaskTracker
from victor.agent.prompt_requirement_extractor import extract_prompt_requirements

# =============================================================================
# DECOMPOSED COMPONENTS - Configs, Strategies, Functions
# =============================================================================

from victor.agent.conversation_controller import (
    ConversationConfig,
    ContextMetrics,
    CompactionStrategy,
)
from victor.agent.context_compactor import (
    TruncationStrategy,
    create_context_compactor,
    calculate_parallel_read_budget,
)
from victor.agent.context_manager import (
    ContextManager,
    ContextManagerConfig,
    create_context_manager,
)
from victor.agent.continuation_strategy import ContinuationStrategy
from victor.agent.tool_call_extractor import ExtractedToolCall
from victor.framework.rl.coordinator import get_rl_coordinator
from victor.agent.usage_analytics import AnalyticsConfig
from victor.agent.tool_sequence_tracker import create_sequence_tracker
from victor.agent.session_state_manager import SessionStateManager, create_session_state_manager

# New Phase 1 extractions (Task 1)
from victor.agent.configuration_manager import ConfigurationManager, create_configuration_manager
from victor.agent.memory_manager import (
    MemoryManager,
    SessionRecoveryManager,
    create_memory_manager,
    create_session_recovery_manager,
)

# =============================================================================
# RECOVERY - Enums and Functions
# =============================================================================

from victor.agent.recovery import RecoveryOutcome, FailureType, StrategyRecoveryAction
from victor.agent.vertical_context import VerticalContext, create_vertical_context
from victor.agent.vertical_integration_adapter import VerticalIntegrationAdapter
from victor.agent.orchestrator_recovery import (
    create_recovery_integration,
    RecoveryAction as OrchestratorRecoveryAction,
)

# =============================================================================
# TOOL OUTPUT FORMATTING
# =============================================================================

from victor.agent.tool_output_formatter import (
    ToolOutputFormatterConfig,
    FormattingContext,
    create_tool_output_formatter,
)

# =============================================================================
# PIPELINE - Configs and Results
# =============================================================================

from victor.agent.tool_pipeline import ToolPipelineConfig, ToolCallResult
from victor.agent.streaming_controller import StreamingControllerConfig, StreamingSession
from victor.agent.task_analyzer import get_task_analyzer
from victor.agent.tool_registrar import ToolRegistrarConfig
from victor.agent.provider_manager import ProviderManagerConfig, ProviderState

# =============================================================================
# OBSERVABILITY AND INTEGRATION
# =============================================================================

from victor.observability.integration import ObservabilityIntegration
from victor.agent.orchestrator_integration import IntegrationConfig

# =============================================================================
# TOOL EXECUTION - Functions and Enums
# =============================================================================

from victor.agent.tool_selection import get_critical_tools
from victor.agent.tool_calling import ToolCallParseResult
from victor.agent.tool_executor import ValidationMode
from victor.agent.orchestrator_utils import (
    calculate_max_context_chars,
    infer_git_operation,
    get_tool_status_message,
)
from victor.agent.orchestrator_factory import OrchestratorFactory
from victor.agent.parallel_executor import create_parallel_executor
from victor.agent.response_completer import ToolFailureContext, create_response_completer

# =============================================================================
# ANALYTICS AND LOGGING
# =============================================================================

from victor.observability.analytics.logger import UsageLogger
from victor.observability.analytics.streaming_metrics import StreamingMetricsCollector

# =============================================================================
# STORAGE AND CACHING
# =============================================================================

from victor.storage.cache.tool_cache import ToolCache

# =============================================================================
# PROVIDER IMPORTS
# =============================================================================

from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    StreamChunk,
    ToolDefinition,
)
from victor.providers.registry import ProviderRegistry
from victor.core.errors import (
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ToolNotFoundError,
    ToolValidationError,
)

# =============================================================================
# TOOL IMPORTS
# =============================================================================

from victor.tools.enums import CostTier
from victor.tools.registry import ToolRegistry
from victor.tools.code_executor_tool import CodeSandbox
from victor.tools.mcp_bridge_tool import get_mcp_tool_definitions
from victor.tools.plugin_registry import ToolPluginRegistry
from victor.tools.semantic_selector import SemanticToolSelector
from victor.tools.tool_names import ToolNames, TOOL_ALIASES
from victor.tools.alias_resolver import get_alias_resolver
from victor.tools.progressive_registry import get_progressive_registry

# =============================================================================
# WORKFLOW AND EMBEDDING IMPORTS
# =============================================================================

from victor.storage.embeddings.intent_classifier import IntentClassifier, IntentType
from victor.workflows.registry import WorkflowRegistry
from victor.workflows.discovery import register_builtin_workflows

# =============================================================================
# PROJECT CONTEXT
# =============================================================================

from victor.context.project_context import ProjectContext

# =============================================================================
# STREAMING SUBMODULE
# Extracted for testability
# =============================================================================

from victor.agent.streaming import (
    CoordinatorConfig,
    ContinuationHandler,
    ContinuationResult,
    IntentClassificationHandler,
    IntentClassificationResult,
    IterationCoordinator,
    ProgressMetrics,
    StreamingChatContext,
    StreamingChatHandler,
    ToolExecutionHandler,
    ToolExecutionResult,
    TrackingState,
    apply_tracking_state_updates,
    create_continuation_handler,
    create_coordinator,
    create_intent_classification_handler,
    create_stream_context,
    create_tool_execution_handler,
    create_tracking_state,
)

# =============================================================================
# ADAPTERS
# Extracted for testability and modularity
# =============================================================================

from victor.agent.adapters import (
    CoordinatorAdapter,
    IntelligentPipelineAdapter,
    ResultConverters,
)
from victor.agent.coordinators.protocol_adapter import (
    OrchestratorProtocolAdapter,
    create_orchestrator_protocol_adapter,
)

# =============================================================================
# LOGGER
# =============================================================================

logger = logging.getLogger(__name__)
