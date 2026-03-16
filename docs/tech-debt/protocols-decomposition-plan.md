# Protocols.py Decomposition Plan

**Target**: `victor/agent/protocols.py` (3,703 LOC, 85 protocol classes)
**Goal**: Split into domain-grouped modules under `victor/agent/protocols/`
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

## Migration Strategy

1. Create `victor/agent/protocols/` package
2. Move classes into domain modules
3. Keep `victor/agent/protocols.py` as a re-export shim:
   ```python
   from victor.agent.protocols.tool_protocols import *
   from victor.agent.protocols.conversation_protocols import *
   # ... etc
   ```
4. All existing imports continue to work via the shim
5. Gradually update imports to use specific modules
6. Remove shim when all imports migrated

## Risk

- **Low**: Re-export shim ensures full backward compatibility
- **Testing**: Run full test suite after each group extraction
- **Import time**: May slightly increase due to more files; offset by lazy loading
