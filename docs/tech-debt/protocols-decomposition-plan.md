# Protocols.py Decomposition Plan

**Status**: Completed
**Canonical target**: `victor/agent/protocols/`
**Historical source**: legacy `victor/agent/protocols.py` monolith (removed)
**Priority**: D-01 (Tier 3 Design)

## Proposed Split

| Module | Classes | Est. LOC | Domain |
|--------|---------|----------|--------|
| `agent_protocols.py` | IAgentFactory, IAgent | ~110 | Agent lifecycle |
| `tool_protocols.py` | ToolRegistryProtocol, ToolSelectorProtocol, ToolPipelineProtocol, ToolExecutorProtocol, ToolCacheProtocol, ToolRegistrarProtocol, ToolSequenceTrackerProtocol, ToolOutputFormatterProtocol, ToolDeduplicationTrackerProtocol, ToolDependencyGraphProtocol, ToolPluginRegistryProtocol, SemanticToolSelectorProtocol, AgentToolSelectionContext, ToolSelectorFeatures, ToolAccessContext, IToolAccessController, AccessPrecedence, ToolAccessDecision, IToolAdapterCoordinator | ~800 | Tool system |
| `conversation_protocols.py` | ConversationControllerProtocol, ConversationStateMachineProtocol, MessageHistoryProtocol, ConversationEmbeddingStoreProtocol, IMessageStore, IContextOverflowHandler, ISessionManager, IEmbeddingManager | ~400 | Conversation |
| `streaming_protocols.py` | StreamingToolChunk, StreamingToolAdapterProtocol, StreamingControllerProtocol, StreamingRecoveryCoordinatorProtocol, ChunkGeneratorProtocol, StreamingHandlerProtocol, StreamingMetricsCollectorProtocol | ~500 | Streaming |
| `provider_protocols.py` | ProviderManagerProtocol, ProviderRegistryProtocol, IProviderHealthMonitor, IProviderSwitcher, IProviderEventEmitter, IProviderClassificationStrategy | ~350 | Providers |
| `analysis_protocols.py` | TaskAnalyzerProtocol, ComplexityClassifierProtocol, ActionAuthorizerProtocol, SearchRouterProtocol, IntentClassifierProtocol, TaskTypeHinterProtocol, IToolCallClassifier | ~250 | Analysis |
| `coordination_protocols.py` | ToolPlannerProtocol, TaskCoordinatorProtocol, ToolCoordinatorProtocol, StateCoordinatorProtocol, PromptCoordinatorProtocol, UnifiedMemoryCoordinatorProtocol | ~450 | Coordination |
| `infrastructure_protocols.py` | ObservabilityProtocol, MetricsCollectorProtocol, RecoveryHandlerProtocol, ResponseSanitizerProtocol, ArgumentNormalizerProtocol, ProjectContextProtocol, CodeExecutionManagerProtocol, WorkflowRegistryProtocol, UsageAnalyticsProtocol, ContextCompactorProtocol, DebugLoggerProtocol, ReminderManagerProtocol, RLCoordinatorProtocol, SafetyCheckerProtocol, AutoCommitterProtocol, MCPBridgeProtocol, UsageLoggerProtocol, SystemPromptBuilderProtocol, ParallelExecutorProtocol, ResponseCompleterProtocol, VerticalStorageProtocol | ~600 | Infrastructure |
| `budget_protocols.py` | ModeControllerProtocol, BudgetType, BudgetStatus, BudgetConfig, IBudgetManager, IBudgetTracker, IMultiplierCalculator, IModeCompletionChecker | ~250 | Budget/Mode |

## Outcome

1. `victor/agent/protocols/` is the canonical import surface.
2. Domain-grouped modules exist for tool, conversation, streaming, provider, analysis, coordination, infrastructure, and budget protocols.
3. `victor.agent.protocols` resolves to the package `__init__.py`, which re-exports the grouped modules for backward compatibility.
4. The legacy monolithic module is removed, eliminating the duplicate source-of-truth risk.

## Remaining Follow-On

- Gradually migrate internal imports to narrower submodules where that improves clarity.
- Keep the package import path stable for external consumers.
- Prevent reintroduction of `victor/agent/protocols.py` via repo-hygiene checks.

## Risk

- **Low**: Re-export shim ensures full backward compatibility
- **Testing**: Run full test suite after each group extraction
- **Import time**: May slightly increase due to more files; offset by lazy loading
