# Dependency Audit: CRITICAL-004 - Direct Dependency Imports

**Date**: 2025-12-20
**File**: victor/agent/orchestrator.py
**Status**: Phase 1 - Audit Complete
**Total Imports**: 69 (7 standard/third-party + 62 Victor modules)

## Executive Summary

The orchestrator has **62 direct Victor module imports**, creating tight coupling and cascading changes when dependencies are modified. This audit categorizes these imports and identifies candidates for dependency injection to achieve the target of **<10 direct imports**.

### Current State
- **Direct Imports**: 62 Victor modules
- **Standard Library**: 6 imports (ast, asyncio, json, logging, time, pathlib)
- **Third Party**: 1 import (rich.console)
- **DI Adoption**: ~15% (ServiceProvider with 9 services registered)

### Target State
- **Direct Imports**: <10 (only core utilities and protocols)
- **DI Injection**: ~90% (all major components via container)
- **Protocol-Based**: Code depends on abstractions, not concretions

---

## Import Categories

### Category 1: Standard Library (KEEP AS DIRECT) ✅
**Count**: 6 imports
**Rationale**: Standard library imports are stable and don't create coupling

```python
import ast              # Line 48 - AST parsing for tool detection
import asyncio          # Line 49 - Async runtime
import json             # Line 50 - JSON serialization
import logging          # Line 51 - Logging framework
import time             # Line 52 - Timing operations
from pathlib import Path  # Line 53 - Path operations
```

**Action**: Keep as direct imports

---

### Category 2: Third Party (KEEP AS DIRECT) ✅
**Count**: 1 import
**Rationale**: External dependencies, stable interface

```python
from rich.console import Console  # Line 56 - Rich console for formatted output
```

**Action**: Keep as direct import

---

### Category 3: Type Checking Only (KEEP AS DIRECT) ✅
**Count**: 1 import
**Rationale**: TYPE_CHECKING imports are compile-time only, no runtime coupling

```python
if TYPE_CHECKING:
    from victor.agent.orchestrator_integration import OrchestratorIntegration  # Line 59
```

**Action**: Keep as TYPE_CHECKING import (already using best practice)

---

### Category 4: Typing Utilities (KEEP AS DIRECT) ✅
**Count**: 1 import
**Rationale**: Python typing module, stable interface

```python
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple, TYPE_CHECKING  # Line 54
```

**Action**: Keep as direct import

---

## Victor Module Imports (62 total)

### Category 5: DI Infrastructure (KEEP AS DIRECT) ✅
**Count**: 2 imports
**Rationale**: Bootstrap and container are the DI foundation - must be direct

**Lines 69-73**:
```python
from victor.core.bootstrap import ensure_bootstrapped, get_service_optional
from victor.core.container import (
    MetricsServiceProtocol,
    LoggerServiceProtocol,
)
```

**Action**: Keep as direct imports (DI infrastructure itself)

---

### Category 6: Protocols (KEEP AS DIRECT) ✅
**Count**: 2 imports
**Rationale**: Protocol definitions are abstractions - keeping imports doesn't create coupling

**Lines 76-90, 147**:
```python
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
)

from victor.agent.protocols import RecoveryHandlerProtocol  # Line 147
```

**Action**: Keep as direct imports (protocols = abstractions)

---

### Category 7: Configuration (CONVERT TO DI) 🔄
**Count**: 3 imports
**Rationale**: Configuration should be injected, not directly imported

**Lines 93, 209-210**:
```python
from victor.config.config_loaders import get_provider_limits  # Line 93
from victor.config.model_capabilities import ToolCallingMatrix  # Line 209
from victor.config.settings import Settings  # Line 210
```

**DI Strategy**:
- Register `Settings` as singleton in container
- Register `ToolCallingMatrix` as factory (depends on settings)
- Register `ProviderLimitsService` wrapping `get_provider_limits`

**Impact**: 3 imports → 0 (100% reduction)

---

### Category 8: Already DI-Injected Components (CONVERT TO PROTOCOL-ONLY) 🔄
**Count**: 14 imports
**Rationale**: These are already injected via ServiceProvider but still have direct imports for fallback

**Current Pattern** (Phase 10 migration):
```python
# Import concrete class
from victor.agent.response_sanitizer import ResponseSanitizer

# DI resolution with fallback
sanitizer = self._container.get_optional(ResponseSanitizerProtocol) or ResponseSanitizer()
```

**Target Pattern**:
```python
# Import only protocol
from victor.agent.protocols import ResponseSanitizerProtocol

# DI resolution (no fallback needed - bootstrap ensures registration)
sanitizer = self._container.get(ResponseSanitizerProtocol)
```

**List of Already-Injected Components**:

1. **ResponseSanitizer** (Line 105)
   - Protocol: `ResponseSanitizerProtocol`
   - Currently: `get_optional(...) or ResponseSanitizer()`
   - Target: `get(ResponseSanitizerProtocol)`

2. **ComplexityClassifier** (Line 107)
   - Protocol: `ComplexityClassifierProtocol`
   - Currently: `get_optional(...) or ComplexityClassifier()`
   - Target: `get(ComplexityClassifierProtocol)`

3. **ActionAuthorizer** (Lines 99-103)
   - Protocol: `ActionAuthorizerProtocol`
   - Currently: `get_optional(...) or ActionAuthorizer()`
   - Target: `get(ActionAuthorizerProtocol)`

4. **SearchRouter** (Line 106)
   - Protocol: `SearchRouterProtocol`
   - Currently: `get_optional(...) or SearchRouter()`
   - Target: `get(SearchRouterProtocol)`

5. **ProjectContext** (Line 211)
   - Protocol: `ProjectContextProtocol`
   - Currently: `get_optional(...) or ProjectContext()`
   - Target: `get(ProjectContextProtocol)`

6. **ArgumentNormalizer** (Line 61)
   - Protocol: `ArgumentNormalizerProtocol`
   - Currently: `get_optional(...) or ArgumentNormalizer()`
   - Target: `get(ArgumentNormalizerProtocol)`

7. **ConversationStateMachine** (Line 97)
   - Protocol: `ConversationStateMachineProtocol`
   - Currently: `get_optional(...) or ConversationStateMachine()`
   - Target: `get(ConversationStateMachineProtocol)`

8. **UnifiedTaskTracker** (Lines 114-117)
   - Protocol: `TaskTrackerProtocol`
   - Currently: `get_optional(...) or UnifiedTaskTracker()`
   - Target: `get(TaskTrackerProtocol)`

9. **CodeExecutionManager** (Line 221)
   - Protocol: `CodeExecutionManagerProtocol`
   - Currently: `get_optional(...) or CodeExecutionManager()`
   - Target: `get(CodeExecutionManagerProtocol)`

10. **WorkflowRegistry** (Line 228)
    - Protocol: `WorkflowRegistryProtocol`
    - Currently: `get_optional(...) or WorkflowRegistry()`
    - Target: `get(WorkflowRegistryProtocol)`

11. **UsageAnalytics** (Lines 133-136)
    - Protocol: `UsageAnalyticsProtocol`
    - Currently: `get_optional(...) or UsageAnalytics()`
    - Target: `get(UsageAnalyticsProtocol)`

12. **ToolSequenceTracker** (Lines 137-140)
    - Protocol: `ToolSequenceTrackerProtocol`
    - Currently: `get_optional(...) or create_sequence_tracker()`
    - Target: `get(ToolSequenceTrackerProtocol)`

13. **ContextCompactor** (Lines 126-131)
    - Protocol: `ContextCompactorProtocol`
    - Currently: `get_optional(...) or create_context_compactor()`
    - Target: `get(ContextCompactorProtocol)`

14. **RecoveryHandler** (Lines 141-146)
    - Protocol: `RecoveryHandlerProtocol`
    - Currently: `get_optional(...) or RecoveryHandler()`
    - Target: `get(RecoveryHandlerProtocol)`

**DI Strategy**:
- Remove all concrete class imports
- Keep only protocol imports
- Ensure bootstrap.py registers all services
- Remove fallback instantiation (`or ConcreteClass()` pattern)

**Impact**: 14 imports → 0 (100% reduction, protocols already imported)

---

### Category 9: Core Orchestrator Components (ALREADY INJECTED VIA FACTORY) ✅
**Count**: 8 imports
**Rationale**: These are created by OrchestratorFactory, not directly instantiated

**Lines 120-174**:
```python
from victor.agent.conversation_controller import ConversationController, ConversationConfig, ContextMetrics, CompactionStrategy
from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig, ToolCallResult
from victor.agent.streaming_controller import StreamingController, StreamingControllerConfig, StreamingSession
from victor.agent.task_analyzer import TaskAnalyzer, get_task_analyzer
from victor.agent.tool_registrar import ToolRegistrar, ToolRegistrarConfig
from victor.agent.provider_manager import ProviderManager, ProviderManagerConfig, ProviderState
from victor.agent.orchestrator_integration import IntegrationConfig
from victor.agent.orchestrator_factory import OrchestratorFactory
```

**Current Status**: Factory pattern already implemented (Sessions 1-40)
- These components are created by `OrchestratorFactory`
- Orchestrator gets instances via `self._factory.create_*()` methods
- Imports needed for type hints and config classes

**Action**: Keep as direct imports (needed for typing, already using factory pattern)

**Note**: Could convert to protocol-based DI in future, but factory pattern is already a significant improvement over direct instantiation.

---

### Category 10: Data Classes & Enums (KEEP AS DIRECT) ✅
**Count**: 8 imports
**Rationale**: Data classes and enums are value objects, not dependencies

**Lines 63-66, 97, 107, 109, 114-117, 144-145**:
```python
from victor.agent.conversation_memory import ConversationStore, MessageRole
from victor.agent.conversation_state import ConversationStage
from victor.agent.complexity_classifier import TaskComplexity, DEFAULT_BUDGETS
from victor.agent.stream_handler import StreamMetrics
from victor.agent.unified_task_tracker import TaskType
from victor.agent.recovery import RecoveryOutcome, FailureType, RecoveryAction
from victor.agent.action_authorizer import ActionIntent, INTENT_BLOCKED_TOOLS
from victor.tools.base import CostTier
```

**Action**: Keep as direct imports (data classes, enums, constants)

---

### Category 11: Utility Functions (CONVERT TO SERVICE) 🔄
**Count**: 7 imports
**Rationale**: Factory functions and utilities should be services

**Lines 98, 104, 108, 132, 191, 198, 223**:
```python
from victor.agent.debug_logger import get_debug_logger  # Line 98
from victor.agent.prompt_builder import get_task_type_hint  # Line 104
from victor.agent.context_reminder import create_reminder_manager  # Line 108
from victor.agent.rl.coordinator import get_rl_coordinator  # Line 132
from victor.agent.safety import get_safety_checker  # Line 191
from victor.agent.auto_commit import get_auto_committer  # Line 198
from victor.tools.mcp_bridge_tool import configure_mcp_client, get_mcp_tool_definitions  # Line 223
```

**DI Strategy**:
- Register singletons for these in bootstrap
- Access via container: `self._container.get(DebugLoggerService)`
- Convert `get_*` functions to service registrations

**Impact**: 7 imports → 0 (100% reduction)

---

### Category 12: Provider Infrastructure (CONVERT TO DI) 🔄
**Count**: 2 imports
**Rationale**: Provider system should be fully injected

**Lines 212-219**:
```python
from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    StreamChunk,
    ToolDefinition,
)
from victor.providers.registry import ProviderRegistry
```

**DI Strategy**:
- Keep `BaseProvider`, `CompletionResponse`, `Message`, `StreamChunk`, `ToolDefinition` as direct (base types)
- Convert `ProviderRegistry` to DI service

**Impact**: 1 import eliminated (ProviderRegistry), 1 import kept (base types)

---

### Category 13: Tool Infrastructure (CONVERT TO DI) 🔄
**Count**: 6 imports
**Rationale**: Tool system should be injected

**Lines 220-226**:
```python
from victor.tools.base import ToolRegistry  # Line 220 (+ CostTier - data class)
from victor.tools.dependency_graph import ToolDependencyGraph  # Line 222
from victor.tools.plugin_registry import ToolPluginRegistry  # Line 224
from victor.tools.semantic_selector import SemanticToolSelector  # Line 225
from victor.tools.tool_names import ToolNames, TOOL_ALIASES  # Line 226
```

**DI Strategy**:
- Register `ToolRegistry` as singleton
- Register `ToolDependencyGraph` as singleton
- Register `ToolPluginRegistry` as singleton
- Register `SemanticToolSelector` as singleton (or via ToolRegistrar)
- Keep `ToolNames`, `TOOL_ALIASES` as direct (constants)

**Impact**: 4 imports → 1 (ToolNames/TOOL_ALIASES constants)

---

### Category 14: Specialized Services (CONVERT TO DI) 🔄
**Count**: 8 imports
**Rationale**: Specialized services should be injected for testability

**Lines 62, 94-96, 110-113, 177, 206-208, 227-229**:
```python
from victor.agent.message_history import MessageHistory  # Line 62
from victor.agent.conversation_embedding_store import ConversationEmbeddingStore  # Lines 94-96
from victor.agent.metrics_collector import MetricsCollector, MetricsCollectorConfig  # Lines 110-113
from victor.observability.integration import ObservabilityIntegration  # Line 177
from victor.analytics.logger import UsageLogger  # Line 206
from victor.analytics.streaming_metrics import StreamingMetricsCollector  # Line 207
from victor.cache.tool_cache import ToolCache  # Line 208
from victor.embeddings.intent_classifier import IntentClassifier, IntentType  # Line 227
from victor.workflows.new_feature_workflow import NewFeatureWorkflow  # Lines 228-229 (WorkflowRegistry already covered)
```

**DI Strategy**:
- Register all as services in bootstrap
- Create protocols if needed
- Access via container

**Impact**: 8 imports → 0 (100% reduction)

---

### Category 15: Helper/Adapter Classes (CONVERT TO SERVICE/FACTORY) 🔄
**Count**: 9 imports
**Rationale**: Helper classes create coupling, should be injected

**Lines 104, 148-152, 153-158, 183-186, 187-189, 190, 192-196, 199-201, 202-205, 232-236**:
```python
from victor.agent.prompt_builder import SystemPromptBuilder  # Line 104 (get_task_type_hint covered in utilities)
from victor.agent.orchestrator_recovery import (  # Lines 148-152
    OrchestratorRecoveryIntegration,
    create_recovery_integration,
    RecoveryAction as OrchestratorRecoveryAction,
)
from victor.agent.tool_output_formatter import (  # Lines 153-158
    ToolOutputFormatter,
    ToolOutputFormatterConfig,
    FormattingContext,
    create_tool_output_formatter,
)
from victor.agent.tool_selection import (  # Lines 183-186
    get_critical_tools,
    ToolSelector,
)
from victor.agent.tool_calling import ToolCallParseResult  # Lines 187-189 (data class)
from victor.agent.tool_executor import ToolExecutor, ValidationMode  # Line 190
from victor.agent.orchestrator_utils import (  # Lines 192-196
    calculate_max_context_chars,
    infer_git_operation,
    get_tool_status_message,
)
from victor.agent.parallel_executor import create_parallel_executor  # Lines 199-201
from victor.agent.response_completer import (  # Lines 202-205
    ToolFailureContext,
    create_response_completer,
)
from victor.agent.streaming import (  # Lines 232-236
    StreamingChatContext,
    StreamingChatHandler,
    create_stream_context,
)
```

**DI Strategy**:
- `SystemPromptBuilder`: Register as factory (created per-provider)
- `OrchestratorRecoveryIntegration`: Register via `create_recovery_integration` factory
- `ToolOutputFormatter`: Register via `create_tool_output_formatter` factory
- `ToolSelector`: Register as singleton
- `ToolExecutor`: Register as singleton
- `orchestrator_utils`: Keep as direct (utility functions module)
- `ParallelExecutor`: Register via `create_parallel_executor` factory
- `ResponseCompleter`: Register via `create_response_completer` factory
- `StreamingChatHandler`: Register as factory (created per-session)

**Data Classes** (keep as direct):
- `ToolCallParseResult`, `ToolFailureContext`, `StreamingChatContext`, `ValidationMode`, `FormattingContext`, `OrchestratorRecoveryAction`

**Impact**: 7 imports → 1 (orchestrator_utils)

---

## Summary by Action

### Keep as Direct Imports (Total: 9)
1. Standard library (6): ast, asyncio, json, logging, time, pathlib
2. Third party (1): rich.console
3. DI infrastructure (2): bootstrap, container

### Convert to DI Injection (Total: 53)
1. Configuration (3): Settings, ToolCallingMatrix, ProviderLimits
2. Already-injected with fallback (14): ResponseSanitizer, ComplexityClassifier, etc.
3. Utility functions (7): get_debug_logger, get_task_type_hint, etc.
4. Provider infrastructure (1): ProviderRegistry
5. Tool infrastructure (4): ToolRegistry, ToolDependencyGraph, etc.
6. Specialized services (8): MessageHistory, ConversationEmbeddingStore, etc.
7. Helper/adapter classes (7): SystemPromptBuilder, ToolSelector, etc.
8. Core orchestrator components (8): **Note - already using factory, could be DI in future**
9. Streaming/async helpers (1): StreamingChatHandler

---

## Reduction Calculation

**Current State**: 62 Victor module imports

**Target State**: <10 direct imports

**Breakdown**:
- Keep as direct: 9 (standard lib, third party, DI infrastructure)
- Protocols (keep): Already imported (Category 6)
- Data classes/enums (keep): 8 (value objects)
- Utility modules (keep): 1 (orchestrator_utils)
- Factory-created (keep): 8 (ConversationController, ToolPipeline, etc.)
- Convert to DI: 36 remaining

**Result**: 62 → ~26 imports (58% reduction)

**Further optimization possible**:
- Convert factory pattern to DI protocols: 8 more eliminations
- Extract orchestrator_utils to service: 1 more elimination
- **Aggressive target**: ~17 imports (73% reduction)
- **Ultra-aggressive (DI protocols for factory)**: ~9 imports (85% reduction, hitting <10 target!)

---

## Implementation Roadmap

### Quick Wins (Phase 2A: 4 hours)
**Impact**: 14 imports → 0

1. **Remove fallback pattern for already-injected services** (14 imports)
   - Update bootstrap to ensure all services registered
   - Change `get_optional(...) or Concrete()` to `get(Protocol)`
   - Remove 14 concrete class imports
   - Test: Verify all services resolve correctly

### Medium Effort (Phase 2B: 8 hours)
**Impact**: 22 imports → 0

2. **Register utility function services** (7 imports)
   - Convert `get_*` functions to service registrations
   - Register in bootstrap: DebugLogger, TaskTypeHinter, etc.

3. **Register tool/provider infrastructure** (5 imports)
   - ToolRegistry, ToolDependencyGraph, ToolPluginRegistry, SemanticToolSelector
   - ProviderRegistry

4. **Register specialized services** (8 imports)
   - MessageHistory, ConversationEmbeddingStore, MetricsCollector
   - ObservabilityIntegration, UsageLogger, StreamingMetricsCollector
   - ToolCache, IntentClassifier

5. **Register configuration services** (3 imports)
   - Settings as singleton
   - ToolCallingMatrix as factory
   - ProviderLimitsService

### Advanced Refactoring (Phase 2C: 12 hours)
**Impact**: 7 imports → 0

6. **Convert helper/adapter classes to services** (7 imports)
   - SystemPromptBuilder, ToolSelector, ToolExecutor
   - OrchestratorRecoveryIntegration, ToolOutputFormatter
   - ParallelExecutor, ResponseCompleter, StreamingChatHandler

### Optional: Factory to DI (Phase 3: 16 hours)
**Impact**: 8 imports → 0

7. **Convert factory pattern to DI protocols** (8 imports)
   - Create protocols for: ConversationController, ToolPipeline, StreamingController
   - TaskAnalyzer, ToolRegistrar, ProviderManager, IntegrationConfig
   - Migrate from `factory.create_X()` to `container.get(XProtocol)`

---

## Risk Assessment

### Low Risk
- Removing fallback pattern (already have protocols and services)
- Registering utility functions as services
- Registering infrastructure services

### Medium Risk
- Converting helper classes to services (may affect initialization order)
- Configuration service registration (settings used in bootstrap)

### High Risk
- Factory to DI migration (large refactoring, affects many components)
- Circular dependency potential (bootstrap needs services, services need bootstrap)

### Mitigation
- Implement in phases with testing after each
- Use lazy initialization for circular dependencies
- Maintain backward compatibility shims during migration
- Comprehensive integration tests before/after each phase

---

## Success Metrics

### Phase 2A (Quick Wins)
- ✅ Import count: 62 → 48 (23% reduction)
- ✅ Test pass rate: ≥99%
- ✅ Zero fallback instantiation in orchestrator

### Phase 2B (Medium Effort)
- ✅ Import count: 48 → 26 (58% total reduction)
- ✅ All services resolvable via DI
- ✅ Bootstrap coverage: 100% of converted services

### Phase 2C (Advanced)
- ✅ Import count: 26 → 19 (69% total reduction)
- ✅ Helper/adapter pattern eliminated
- ✅ Service lifecycle managed by container

### Phase 3 (Optional - Factory to DI)
- ✅ Import count: 19 → 11 or <10 (82-84% total reduction) ⭐ **TARGET MET**
- ✅ Factory pattern eliminated
- ✅ 100% protocol-based dependencies

---

## Next Steps

1. ✅ **COMPLETED**: Audit complete - 62 imports categorized
2. ⏭️ **NEXT**: Start Phase 2A - Remove fallback pattern (Quick Win - 4 hours)
3. **THEN**: Phase 2B - Register infrastructure services (8 hours)
4. **THEN**: Phase 2C - Convert helper classes (12 hours)
5. **OPTIONAL**: Phase 3 - Factory to DI protocols (16 hours)

**Recommended**: Execute Phases 2A + 2B first (12 hours total) for 58% reduction, then evaluate if Phase 2C/3 are needed.
