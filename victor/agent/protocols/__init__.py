"""Protocols for orchestrator services.

All protocols are organized in domain-grouped submodules but remain importable
from this package for full backward compatibility via lazy loading.

Submodules:
    agent_factory        — IAgentFactory, IAgent
    tool_protocols       — Tool registry, pipeline, executor, selector, access control
    conversation_protocols — Conversation controller, state machine, message history
    streaming_protocols  — Streaming controller, recovery, chunk generator
    provider_protocols   — Provider manager, registry, health monitor, switcher
    analysis_protocols   — Task analyzer, complexity classifier, intent classifier
    coordination_protocols — Tool planner, task coordinator, state coordinator
    infrastructure_protocols — Observability, metrics, recovery, sanitizer, etc.
    budget_protocols     — Mode controller, budget manager, budget tracking
"""

from __future__ import annotations

import importlib
from typing import Any

# Map every exported name to its submodule
_SUBMODULE_MAP: dict[str, str] = {}

_MODULE_MEMBERS = {
    "agent_factory": [
        "IAgentFactory",
        "IAgent",
    ],
    "tool_protocols": [
        "AgentToolSelectionContext",
        "ToolSelectorFeatures",
        "ToolRegistryProtocol",
        "ToolSelectorProtocol",
        "ToolPipelineProtocol",
        "ToolExecutorProtocol",
        "ToolCacheProtocol",
        "ToolRegistrarProtocol",
        "ToolSequenceTrackerProtocol",
        "ToolOutputFormatterProtocol",
        "ToolDeduplicationTrackerProtocol",
        "ToolDependencyGraphProtocol",
        "ToolPluginRegistryProtocol",
        "SemanticToolSelectorProtocol",
        "IToolAdapterCoordinator",
        "AccessPrecedence",
        "ToolAccessDecision",
        "ToolAccessContext",
        "IToolAccessController",
    ],
    "conversation_protocols": [
        "ConversationControllerProtocol",
        "ConversationStateMachineProtocol",
        "MessageHistoryProtocol",
        "ConversationEmbeddingStoreProtocol",
        "IMessageStore",
        "IContextOverflowHandler",
        "ISessionManager",
        "IEmbeddingManager",
    ],
    "streaming_protocols": [
        "StreamingToolChunk",
        "StreamingToolAdapterProtocol",
        "StreamingControllerProtocol",
        "StreamingCoordinatorProtocol",
        "StreamingRecoveryCoordinatorProtocol",
        "ChunkGeneratorProtocol",
        "StreamingHandlerProtocol",
        "StreamingMetricsCollectorProtocol",
    ],
    "provider_protocols": [
        "ProviderManagerProtocol",
        "ProviderRegistryProtocol",
        "IProviderHealthMonitor",
        "IProviderSwitcher",
        "IProviderEventEmitter",
        "IProviderClassificationStrategy",
    ],
    "analysis_protocols": [
        "TaskAnalyzerProtocol",
        "ComplexityClassifierProtocol",
        "ActionAuthorizerProtocol",
        "SearchRouterProtocol",
        "IntentClassifierProtocol",
        "TaskTypeHinterProtocol",
        "IToolCallClassifier",
    ],
    "coordination_protocols": [
        "ToolPlannerProtocol",
        "TaskCoordinatorProtocol",
        "ToolCoordinatorProtocol",
        "StateCoordinatorProtocol",
        "PromptCoordinatorProtocol",
        "UnifiedMemoryCoordinatorProtocol",
    ],
    "infrastructure_protocols": [
        "ObservabilityProtocol",
        "MetricsCollectorProtocol",
        "FailureDetectorProtocol",
        "RecoveryExecutorProtocol",
        "RecoveryDiagnosticsProtocol",
        "RecoveryHandlerProtocol",
        "ResponseSanitizerProtocol",
        "ArgumentNormalizerProtocol",
        "ProjectContextProtocol",
        "CodeExecutionManagerProtocol",
        "WorkflowRegistryProtocol",
        "UsageAnalyticsProtocol",
        "ContextCompactorProtocol",
        "DebugLoggerProtocol",
        "ReminderManagerProtocol",
        "RLCoordinatorProtocol",
        "SafetyCheckerProtocol",
        "AutoCommitterProtocol",
        "MCPBridgeProtocol",
        "UsageLoggerProtocol",
        "SystemPromptBuilderProtocol",
        "ParallelExecutorProtocol",
        "ResponseCompleterProtocol",
        "VerticalMiddlewareStorageProtocol",
        "SafetyPatternStorageProtocol",
        "TeamSpecStorageProtocol",
        "VerticalStorageProtocol",
        "TaskTrackerProtocol",
        "CompactionSummarizerProtocol",
        "HierarchicalCompactionProtocol",
        "SessionContextLinkerProtocol",
    ],
    "budget_protocols": [
        "ModeControllerProtocol",
        "BudgetType",
        "BudgetStatus",
        "BudgetConfig",
        "IBudgetManager",
        "IBudgetTracker",
        "IMultiplierCalculator",
        "IModeCompletionChecker",
    ],
}

# Build the reverse map
for _mod, _names in _MODULE_MEMBERS.items():
    for _name in _names:
        _SUBMODULE_MAP[_name] = _mod

__all__ = list(_SUBMODULE_MAP.keys())


def __getattr__(name: str) -> Any:
    """Lazy import: resolve protocol names on first access."""
    if name in _SUBMODULE_MAP:
        mod = importlib.import_module(f"victor.agent.protocols.{_SUBMODULE_MAP[name]}")
        value = getattr(mod, name)
        # Cache on the module for subsequent access
        globals()[name] = value
        return value
    raise AttributeError(f"module 'victor.agent.protocols' has no attribute {name!r}")
