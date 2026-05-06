# Agent Facade And Service Migration Audit

Date: 2026-04-26

## Scope

This note records a verified audit of the `AgentOrchestrator` facade and the
remaining compatibility shims involved in the facade/coordinator to
service/state-passed migration.

Verification notes:
- file sizes and line counts were taken from the local workspace
- symbol references were verified with literal `rg` searches
- no completion percentage is claimed because the repo does not currently
  expose a canonical migration denominator

## Verified Baseline

- `victor/agent/orchestrator.py`: 5,367 lines, 218,588 bytes
- service-owned compatibility shims audited: 9 files, 206,281 bytes total
- the orchestrator still declares itself a facade in its module docstring

## Compatibility Shim Inventory

| Module | Bytes | Verified runtime replacement |
|---|---:|---|
| `victor/agent/services/tool_compat.py` | 56,032 | `ToolService` |
| `victor/agent/services/recovery_compat.py` | 35,044 | `RecoveryService` |
| `victor/agent/services/session_compat.py` | 34,112 | `SessionService` |
| `victor/agent/services/chat_compat.py` | 20,548 | `ChatService` |
| `victor/agent/services/prompt_compat.py` | 16,398 | `PromptRuntimeAdapter` for `PromptRuntimeProtocol`; `UnifiedPromptPipeline` for orchestrator-owned live prompt assembly |
| `victor/agent/services/state_compat.py` | 14,294 | Historical shim removed; canonical live state is `StateRuntimeAdapter` over `ConversationController` + `ConversationStateMachine`, while `victor.agent.state_service.StateService` remains persisted vertical state only |
| `victor/agent/services/streaming_chat_compat.py` | 11,165 | `ChatService.stream_chat()` / service-owned streaming runtime |
| `victor/agent/services/sync_chat_compat.py` | 10,964 | `ChatService` |
| `victor/agent/services/unified_chat_compat.py` | 7,724 | `ChatService` |

## Clarifications

### State surfaces

`StateCoordinator` and `StateService` are not equivalent.

- `StateCoordinator` is a deprecated live conversation-stage wrapper around
  `ConversationController` and optional `ConversationStateMachine`
- `StateService` persists vertical state to storage and is not a drop-in
  replacement for conversation-stage coordination
- `StateRuntimeProtocol` now resolves through a canonical
  `StateRuntimeAdapter` in DI; no concrete `StateCoordinator` shim remains in
  the repo, and only the deprecated `StateCoordinatorProtocol` alias remains
  for compatibility

### Prompt surfaces

Prompt migration has two distinct canonical surfaces:

- `PromptRuntimeProtocol` now resolves through `PromptRuntimeAdapter` in DI for
  the narrower mutable prompt-runtime contract
- `UnifiedPromptPipeline` remains the canonical owner of orchestrator-managed
  live prompt assembly
- `SystemPromptCoordinator` was removed on 2026-05-05; prompt classification
  now stays on `SystemPromptStatePassedCoordinator` and live prompt assembly
  stays on `UnifiedPromptPipeline`

## Verified Orchestrator Bridge Sites

### `_handle_tool_calls`

Verified references remain in these files:

- `victor/agent/services/orchestrator_protocol_adapter.py:167`
- `victor/agent/services/orchestrator_protocol_adapter.py:359`
- `victor/agent/streaming/tool_execution.py:196`
- `victor/agent/streaming/tool_execution.py:368`
- `victor/agent/streaming/tool_execution.py:455`
- `victor/agent/services/turn_execution_runtime.py:287`
- `victor/agent/services/turn_execution_runtime.py:507`

These references are not all equivalent:
- some are bridge definitions or dependency wiring
- some are live invocation sites

Future audits should keep those categories separate.

### Backward-compat wrappers on the orchestrator

These wrappers remain present and verified:

- `_build_system_prompt_with_adapter`
- `_register_default_tools`
- `_initialize_plugins`
- `_execute_tool_with_retry`
- `_get_provider_category`
- `_emit_tool_strategy_metrics`
- `_create_recovery_context`

## Changes Applied In This Batch

### Runtime deprecation signaling

Public compatibility construction now emits explicit `DeprecationWarning`
messages with canonical replacement targets for:

- `ToolCoordinator`
- `SessionCoordinator`
- `PromptCoordinator`
- `StateCoordinator`
- `StreamingRecoveryCoordinator`

Internal runtime wiring now suppresses those constructor warnings where the
shim is being materialized only as a compatibility dependency.

### Documentation

Canonical replacements are now described directly in:

- `victor/agent/services/__init__.py`
- the touched `*_compat.py` module docstrings
- this audit note

## TDD Update: Canonical Tool-Call Execution Surface

A follow-up TDD slice introduced `AgentOrchestrator.execute_tool_calls(...)` as
the canonical façade-owned entry point for tool-call execution.

Applied migration steps:

- `AgentOrchestrator._handle_tool_calls(...)` now delegates to
  `execute_tool_calls(...)`
- `OrchestratorProtocolAdapter.execute_tool_calls(...)` now targets the
  canonical orchestrator method instead of the private bridge
- `TurnExecutor` now prefers `tool_context.execute_tool_calls(...)` and falls
  back to `_handle_tool_calls(...)` only for compatibility
- the streaming tool-execution factory now wires `orchestrator.execute_tool_calls`
  into the handler instead of the private bridge
- `ToolContextProtocol` now declares `execute_tool_calls(...)` explicitly

### Remaining `_handle_tool_calls` references after this slice

The remaining references are intentionally compatibility-only or fallback-only:

- `victor/agent/orchestrator.py`: legacy wrapper
- `victor/agent/services/orchestrator_protocol_adapter.py`: legacy wrapper
- `victor/agent/services/protocols/chat_runtime.py`: compatibility protocol declaration
- `victor/agent/services/turn_execution_runtime.py`: compatibility fallback when
  a tool context has not yet implemented `execute_tool_calls(...)`
- `victor/agent/services/streaming_chat_compat.py`: compatibility fallback for
  older tool contexts
- `victor/agent/coordinators/protocol_based_injection_example.py`: example code

## TDD Update: Canonical Recovery-Context Creation Surface

A second TDD slice introduced
`AgentOrchestrator.create_recovery_context(...)` as the canonical façade-owned
entry point for streaming recovery context creation.

Applied migration steps:

- `AgentOrchestrator._create_recovery_context(...)` now delegates to
  `create_recovery_context(...)`
- orchestrator-owned recovery helpers now build recovery state via the
  canonical surface
- the canonical streaming executor now prefers
  `orchestrator.create_recovery_context(...)` and falls back to
  `_create_recovery_context(...)` only for compatibility
- the streaming tool-execution factory now wires the canonical recovery-context
  builder into `ToolExecutionHandler`
- `ChatOrchestratorProtocol` now declares `create_recovery_context(...)`
  explicitly, and the compatibility examples/tests now mirror the real
  synchronous surface

### Remaining `_create_recovery_context` references after this slice

The remaining same-name references are now either wrappers, protocol
compatibility, or unrelated service-local helpers:

- `victor/agent/orchestrator.py`: legacy wrapper delegating to the canonical
  method
- `victor/agent/services/protocols/chat_runtime.py`: compatibility protocol
  declaration
- `victor/agent/coordinators/protocol_based_injection_example.py`: example code
- `victor/agent/services/chat_service.py`: unrelated chat-service error
  recovery helper, not the streaming recovery-context bridge

## TDD Update: Canonical System-Prompt Build Surface

A third TDD slice introduced `AgentOrchestrator.build_system_prompt()` as the
canonical façade-owned entry point for system-prompt construction.

Applied migration steps:

- `AgentOrchestrator._build_system_prompt_with_adapter(...)` now delegates to
  `build_system_prompt()`
- orchestrator-owned prompt rebuild paths now call `build_system_prompt()`
  directly during initialization, workspace switching, and query-specific
  prompt refresh
- the unified prompt-runtime logic remains centralized in a single method, with
  the legacy private name retained only as a compatibility bridge

### Remaining `_build_system_prompt_with_adapter` references after this slice

The remaining same-name references are now the intentional compatibility
wrapper plus compatibility-focused tests:

- `victor/agent/orchestrator.py`: legacy wrapper delegating to the canonical
  method
- `tests/unit/agent/test_orchestrator_core.py`: bridge coverage
- `tests/unit/agent/test_kv_cache_optimization.py`: compatibility regression
  coverage

## TDD Update: Canonical Tool-Registration Surfaces

A fourth TDD slice introduced registrar-owned public entry points for tool
bootstrap work: `ToolRegistrar.register_default_tools()` and
`ToolRegistrar.initialize_plugins()`.

Applied migration steps:

- `ComponentAssembler.assemble_tools(...)` now calls
  `tool_registrar.register_default_tools()` instead of reaching through
  `AgentOrchestrator._register_default_tools(...)`
- `ToolBuildersMixin.initialize_plugin_system(...)` now calls
  `tool_registrar.initialize_plugins()` instead of the private registrar
  implementation
- `AgentOrchestrator._register_default_tools(...)` and
  `AgentOrchestrator._initialize_plugins(...)` remain as compatibility wrappers,
  but they now delegate to public registrar-owned surfaces

### Remaining `_register_default_tools` and `_initialize_plugins` references after this slice

The remaining same-name references are now the intentional compatibility
wrappers plus private implementation coverage:

- `victor/agent/orchestrator.py`: legacy wrapper methods delegating to
  `ToolRegistrar`
- `victor/agent/tool_registrar.py`: private implementation method
  `_initialize_plugins(...)` retained behind the new public
  `initialize_plugins()`
- `tests/unit/tools/test_tool_registrar.py`: private implementation coverage
  for plugin-loading internals

## TDD Update: Canonical Tool-Retry Surface

A fifth TDD slice introduced `AgentOrchestrator.execute_tool_with_retry()` as
the canonical façade-owned entry point for retry-aware single-tool execution.

Applied migration steps:

- `AgentOrchestrator._execute_tool_with_retry(...)` now delegates to
  `execute_tool_with_retry()`
- direct orchestrator retry callers in cache-related tests now target the
  public surface instead of the private bridge
- the slice also fixed `ToolBuildersMixin.create_tool_cache()` to prefer the
  canonical nested `settings.tools.*` cache configuration and the
  `tool_cache_dir_override` path override instead of stale top-level cache
  fields

### Remaining `_execute_tool_with_retry` references after this slice

The remaining same-name reference is now the intentional compatibility wrapper:

- `victor/agent/orchestrator.py`: legacy wrapper delegating to the canonical
  retry surface

## TDD Update: Metrics-Owned Tool-Strategy Observability

A sixth TDD slice moved live tool-strategy observability off the orchestrator
and into `MetricsCoordinator`.

Applied migration steps:

- `MetricsCoordinator.emit_tool_strategy_event(...)` is now the canonical
  owner of tool-strategy summary logging, provider-category labeling, tier
  distribution reporting, and Prometheus-style metric emission
- `MetricsCoordinator.get_provider_category(...)` now reuses provider
  capability detection first and falls back to name-based classification only
  when a provider string is all that is available
- `AgentOrchestrator._emit_tool_strategy_event(...)` is now a thin bridge that
  delegates to `MetricsCoordinator.emit_tool_strategy_event(...)`
- `AgentOrchestrator._get_provider_category(...)` and
  `AgentOrchestrator._emit_tool_strategy_metrics(...)` remain only as legacy
  wrappers over metrics-owned behavior
- the context-aware KV strategy path now passes the provider object through to
  the metrics surface and no longer sends the stale `provider_category=...`
  keyword that did not match the bridge signature

### Remaining `_get_provider_category` and `_emit_tool_strategy_metrics` references after this slice

The remaining same-name references are now the intentional compatibility
wrappers:

- `victor/agent/orchestrator.py`: legacy wrapper methods delegating to
  `MetricsCoordinator`

## TDD Update: Canonical Tool-Call Surface in TurnExecutor

A seventh TDD slice tightened the live turn-execution runtime so it now relies
on the canonical `execute_tool_calls(...)` contract rather than silently
falling back to the private `_handle_tool_calls(...)` bridge.

Applied migration steps:

- `TurnExecutor._execute_tool_calls(...)` now calls
  `tool_context.execute_tool_calls(...)` directly
- `ToolContextProtocol` no longer declares `_handle_tool_calls(...)` as part
  of the expected internal runtime surface
- `OrchestratorProtocolAdapter` no longer exposes `_handle_tool_calls(...)`
  as an internal adapter helper
- the protocol-based chat-coordinator test doubles now implement
  `execute_tool_calls(...)` as the canonical method
- the protocol-based injection example was updated to mirror the same
  canonical-first contract without a duplicate legacy bridge

### Remaining live `_handle_tool_calls` usage after this slice

The remaining same-name references are now limited to compatibility-focused
tests and fixture-level bridge-avoidance assertions.

## TDD Update: Canonical Tool-Call Surface in StreamingChatCoordinator

An eighth TDD slice removed the remaining live tool-call fallback from the
streaming chat compatibility shim.

Applied migration steps:

- `StreamingChatCoordinator._execute_tool_calls_during_stream(...)` now calls
  `tool_context.execute_tool_calls(...)` directly
- compatibility tests now assert that streaming shim tool execution uses the
  canonical tool-call surface and does not reach through
  `tool_context._handle_tool_calls(...)`

### Remaining live `_handle_tool_calls` usage after this slice

The remaining same-name references are now limited to compatibility-focused
tests.

## TDD Update: Canonical Recovery-Context Surface Completion

A ninth TDD slice finished the streaming recovery-context migration.

Applied migration steps:

- the canonical streaming executor now requires
  `create_recovery_context(...)` directly and no longer falls back to
  `orchestrator._create_recovery_context(...)`
- `create_tool_execution_handler(...)` now binds the canonical
  `orchestrator.create_recovery_context(...)` surface directly
- `ChatOrchestratorProtocol` no longer declares `_create_recovery_context(...)`
  as part of the internal runtime contract
- protocol examples, stubs, and tests now model only the canonical
  recovery-context surface on the orchestrator boundary
- `AgentOrchestrator._create_recovery_context(...)` has been removed

### Remaining `_create_recovery_context` references after this slice

The remaining same-name references are now implementation-local helpers, not
orchestrator compatibility bridges:

- `victor/agent/services/chat_service.py`: service-local error recovery helper
- `victor/agent/streaming/tool_execution.py`: internal handler field name for
  the injected recovery-context factory

## TDD Update: Private Wrapper Removal

A tenth cleanup slice removed several private orchestrator wrappers that had
become test-only after the canonical migration completed.

Removed private wrappers:

- `AgentOrchestrator._build_system_prompt_with_adapter(...)`
- `AgentOrchestrator._execute_tool_with_retry(...)`
- `AgentOrchestrator._register_default_tools(...)`
- `AgentOrchestrator._initialize_plugins(...)`
- `AgentOrchestrator._get_provider_category(...)`
- `AgentOrchestrator._emit_tool_strategy_metrics(...)`

Applied migration steps:

- prompt/retry tests now target `build_system_prompt(...)` and
  `execute_tool_with_retry(...)` directly
- integration and unit registration tests now assert registrar-owned public
  surfaces without carrying dead orchestrator wrappers in their stubs
- metrics/runtime cleanup removed unused orchestrator passthrough methods once
  service-owned metrics paths were fully established

### Remaining private alias/delegation surfaces after this slice

The notable remaining private delegation surface in this migration area is now
the registrar’s own implementation helper:

- `victor/agent/tool_registrar.py:_load_plugin_tools(...)`
  retained behind `initialize_plugins()` as an implementation detail

## TDD Update: Internal Naming Cleanup for Plugin and Recovery Seams

An eleventh cleanup slice removed the last production names that still read
like deprecated bridges even though the underlying behavior was already
canonical.

Applied cleanup steps:

- `ToolRegistrar.initialize_plugins()` now delegates to
  `_load_plugin_tools(...)` instead of `_initialize_plugins(...)`, making the
  private helper describe its implementation role rather than resemble a
  public-facing legacy API
- `ToolExecutionHandler` now stores its injected recovery-context dependency as
  `_recovery_context_factory` instead of `_create_recovery_context`
- `ChatService` now uses `_build_recovery_context(...)` for its local
  error-recovery helper, removing the last production `_create_recovery_context`
  method name outside the canonical orchestrator surface
- registrar and streaming-handler tests were updated to target the public
  plugin API and the renamed internal factory seam

### Remaining legacy-name references after this slice

The remaining old-name references in this migration area are now limited to
compatibility-focused tests that explicitly assert the legacy bridges are not
used.

## TDD Update: Canonical State Runtime DI Surface

A twelfth TDD slice moved the live `StateRuntimeProtocol` DI path off the
deprecated `StateCoordinator` shim and onto a canonical adapter over
conversation-owned state.

Applied migration steps:

- added `victor.agent.services.state_runtime.StateRuntimeAdapter` as the
  canonical runtime adapter for live conversation-stage state
- `OrchestratorServiceProvider._create_state_runtime(...)` now resolves
  `ConversationControllerProtocol` and optional
  `ConversationStateMachineProtocol` from DI and returns `StateRuntimeAdapter`
- `victor.core.service_specs.RUNTIME_COORDINATOR_SPECS` now binds
  `StateRuntimeProtocol` to `_create_state_runtime(...)` with
  `pass_container=True`, so runtime resolution no longer instantiates the
  deprecated `StateCoordinator` shim internally
- new state-runtime tests cover both structural protocol conformance and DI
  resolution to the canonical adapter

### Remaining state compatibility after this slice

The concrete state shims have since been removed. The remaining compatibility
surface is now limited to deprecated protocol aliases, primarily
`StateCoordinatorProtocol`, while the canonical live runtime remains
`StateRuntimeAdapter` over conversation-owned state.

## TDD Update: Canonical Prompt Runtime DI Surface

A thirteenth TDD slice moved the `PromptRuntimeProtocol` DI path off the
deprecated `PromptCoordinator` shim and onto a canonical adapter while keeping
the broader prompt-assembly end state explicit.

Applied migration steps:

- added `victor.agent.services.prompt_runtime.PromptRuntimeAdapter` as the
  canonical `PromptRuntimeProtocol` implementation, along with
  `PromptRuntimeContext` and `PromptRuntimeConfig`
- refactored `PromptCoordinator` into a deprecated subclass over
  `PromptRuntimeAdapter`, so the public compatibility surface now reuses the
  canonical implementation instead of carrying its own duplicate logic
- `OrchestratorServiceProvider._create_prompt_runtime(...)` now returns
  `PromptRuntimeAdapter`
- `victor.core.service_specs.RUNTIME_COORDINATOR_SPECS` now binds
  `PromptRuntimeProtocol` to `_create_prompt_runtime(...)`
- new prompt-runtime tests cover structural protocol conformance, prompt
  assembly behavior, and DI resolution to the canonical adapter

### Remaining prompt compatibility after this slice

The public compatibility shim remains intentionally available, but it is no
longer the canonical runtime implementation behind DI:

- `victor/agent/services/prompt_compat.py`: deprecated public shim
- `victor/agent/prompt_coordinator.py`: deprecated import path shim
- compatibility tests that assert those shims still import and warn correctly
- `UnifiedPromptPipeline` remains the real owner of orchestrator-managed live
  prompt assembly; `PromptRuntimeAdapter` only covers the narrower protocol
  contract

## Historical TDD Update: PromptRuntimeSupport Fallback Surface

This slice is preserved as historical context only. It was superseded later on
2026-05-04 when internal orchestrator runtime assembly stopped wiring
`PromptRuntimeSupport`, leaving it and `SystemPromptCoordinator` as
compatibility-only prompt helper surfaces.

A fourteenth TDD slice removed the live internal dependency on the deprecated
`SystemPromptCoordinator` runtime object while keeping the public compatibility
class intact.

Applied migration steps:

- added `victor.agent.services.prompt_runtime_support.PromptRuntimeSupport` as
  the canonical internal fallback object for prompt helper seams when
  `UnifiedPromptPipeline` is unavailable
- refactored `SystemPromptCoordinator` into a compatibility subclass over
  `PromptRuntimeSupport`, so the public shim now reuses the canonical
  implementation instead of carrying its own duplicate logic
- `ComponentAssembler.assemble_conversation(...)` now binds
  `orchestrator._prompt_runtime_support` via
  `factory.create_prompt_runtime_support(...)` instead of materializing
  `_system_prompt_coordinator` on the live orchestrator path
- `AgentOrchestrator` prompt helper fallbacks now prefer
  `_prompt_runtime_support` and retain `_system_prompt_coordinator` only as a
  backward-safety fallback
- new tests cover the canonical support class, assembler wiring, and the
  factory surface used by the live runtime

### Remaining system-prompt compatibility after this slice

The deprecated compatibility class remains intentionally available, but it is
no longer the live internal fallback object created by the orchestrator:

- `victor/agent/services/system_prompt_runtime.py`: deprecated public shim
  subclass
- `victor/agent/coordinators/system_prompt_coordinator.py`: deprecated import
  path shim
- coordinator factory helpers that intentionally preserve public compat access
- compatibility tests that assert those shims still import and behave the same

## Migration Update: Provider Coordinators Removal (2026-05-01)

**Seam consolidated:** Both ProviderCoordinator and ProviderSwitchCoordinator lazy instantiation

**Canonical owner:** ProviderService (`victor/agent/services/provider_service.py`)

**Changes applied:**

1. **Removed both coordinators from ProviderRuntimeComponents** (`victor/agent/runtime/provider_runtime.py`)
   - Deleted `provider_coordinator` field from `ProviderRuntimeComponents` dataclass
   - Deleted `provider_switch_coordinator` field from `ProviderRuntimeComponents` dataclass
   - Removed `_build_provider_coordinator()` factory function
   - Removed `_build_provider_switch_coordinator()` factory function
   - Removed `get_provider_service` parameter from `create_provider_runtime_components()`
   - Added migration notes to module docstring

2. **Updated orchestrator initialization** (`victor/agent/orchestrator.py`)
   - Removed `get_provider_service=lambda: getattr(self, "_provider_service", None)` parameter
   - Updated `_initialize_provider_runtime()` docstring with migration note
   - Simplified runtime initialization call

3. **Updated tests** (`tests/unit/agent/test_provider_runtime.py`)
   - Replaced `test_create_provider_runtime_components_provider_coordinator_lazy` with `test_create_provider_runtime_components_provider_coordinator_removed`
   - Replaced `test_create_provider_runtime_components_switch_coordinator_lazy` with `test_create_provider_runtime_components_switch_coordinator_removed`
   - Tests now verify both coordinators are NOT in runtime components

4. **Updated lazy initialization tests** (`tests/unit/agent/test_runtime_lazy_init.py`)
   - Modified `test_provider_runtime_components_are_lazy` to reflect that only pool remains

5. **Enhanced import-boundary guard tests** (`tests/unit/agent/test_provider_root_shim_guardrails.py`)
   - AST-based test prevents new internal imports of the removed root shim modules
   - Explicitly blocks `from victor.agent.provider import ProviderSwitchCoordinator`
   - Verifies `victor.agent.provider_coordinator` and
     `victor.agent.provider_switch_coordinator` are not importable

7. **Updated initialization contract**
   - `InitializationPhaseManager` now reports `provider_runtime` as the created component for the provider runtime phase
   - stale references to `provider_coordinator` / `provider_switch_coordinator` as phase outputs were removed

8. **Completed breaking cleanup for removed provider coordinator shims** (2026-05-05)
   - Removed stale `ProviderSwitchCoordinator` type imports from factory modules
   - Tightened facade-boundary guardrails so dead provider shim helpers cannot drift back in
   - Updated runtime/docs to state the provider coordinator shims are removed, not compatibility-owned

**What remains:**

- ProviderService / ProviderManagementRuntime remain the only canonical provider runtime seam
- Historical proposal docs still describe the earlier compatibility phase for audit context

**Benefits:**

- Eliminated dead code (both coordinators were never used internally)
- Clearer ownership: ProviderService is the single authority for all provider operations
- Guard tests prevent regression
- Significantly reduced complexity in provider_runtime.py
- Simplified orchestrator initialization

**Breaking changes:** Yes. The removed root shim modules
`victor.agent.provider_coordinator` and
`victor.agent.provider_switch_coordinator` must remain unavailable in v1.0.0+.

## Migration Update: Chat/Tool Subservices Removal (2026-05-01)

**Seam consolidated:** Parallel subservice architectures under `victor/agent/services/chat/*` and `victor/agent/services/tools/*`

**Canonical owners:** ChatService (`victor/agent/services/chat_service.py`) and ToolService (`victor/agent/services/tool_service.py`)

**Changes applied:**

1. **Removed chat subservices** (`victor/agent/services/chat/`)
   - Deleted `__init__.py` (43 LOC)
   - Deleted `ChatFlowService` via `chat_flow_service.py` (410 LOC)
   - Deleted `StreamingService` via `streaming_service.py` (275 LOC)
   - Deleted `ContinuationService` via `continuation_service.py` (252 LOC)
   - Deleted `ResponseAggregationService` via `response_aggregation_service.py` (255 LOC)
   - Deleted `protocols.py` (229 LOC)
   - Deleted `chat_service_facade.py` (27 LOC)
   - Total: 1,491 LOC removed

2. **Removed tools subservices** (`victor/agent/services/tools/`)
   - Deleted `__init__.py` (55 LOC)
   - Deleted `ToolSelectorService` via `tool_selector_service.py` (305 LOC)
   - Deleted `ToolExecutorService` via `tool_executor_service.py` (339 LOC)
   - Deleted `ToolTrackerService` via `tool_tracker_service.py` (260 LOC)
   - Deleted `ToolPlannerService` via `tool_planner_service.py` (337 LOC)
   - Deleted `ToolResultProcessor` via `tool_result_processor.py` (301 LOC)
   - Deleted `protocols.py` (329 LOC)
   - Deleted `tool_service_facade.py` (27 LOC)
   - Total: 1,953 LOC removed

3. **Created import-boundary guard test** (`tests/unit/agent/test_subservices_import_guard.py`)
   - AST-based test to prevent new internal imports of removed subservices
   - Scans production code across `victor/`
   - Rejects exact package imports, submodule imports, and `from victor.agent.services import chat/tools` patterns
   - Allows canonical imports (`chat_service.py`, `tool_service.py`)
   - NOTE: Does not allow `protocols` imports (those modules were deleted)

4. **Removed test files for deleted subservices**
   - Deleted `tests/unit/agent/services/chat/test_decomposed_services.py` (442 LOC)
   - Deleted `tests/unit/agent/services/tools/test_decomposed_tools.py` (481 LOC)
   - All 629 services tests passing

**Why these were unused:**

- Subservices were only referenced in their own package `__init__.py` files for re-export
- Only instantiated in docstring examples
- Never used in production code
- Created confusion about canonical ownership (parallel vs. canonical architecture)

**What was deleted:**

- All files under `victor/agent/services/chat/` directory (complete removal)
- All files under `victor/agent/services/tools/` directory (complete removal)
- Test files: `tests/unit/agent/services/chat/test_decomposed_services.py` (442 LOC)
- Test files: `tests/unit/agent/services/tools/test_decomposed_tools.py` (481 LOC)
- Total deleted: 3,444 LOC from subservices + 923 LOC from tests = 4,367 LOC

**What remains:**

- **ChatService** (`victor/agent/services/chat_service.py`) - Canonical chat service
- **ToolService** (`victor/agent/services/tool_service.py`) - Canonical tool service

**Benefits:**

- Eliminated 4,367 LOC of completely unused code (subservices + tests)
- Clearer ownership: Single canonical service for each domain
- No parallel architecture confusion
- Guard tests prevent regression
- Simplified package structure

**Breaking changes:**
- `import victor.agent.services.chat` now raises `ModuleNotFoundError` (package directory deleted)
- `import victor.agent.services.tools` now raises `ModuleNotFoundError` (package directory deleted)
- The removed subservices were never used in production code, so no production code needs updating
- All existing production code uses the canonical ChatService and ToolService
- **External packages are NOT affected** - External verticals are already forbidden from importing from `victor.agent.*` by the import boundary guard test (`test_external_vertical_import_boundaries.py`)

**External package compatibility validation (2026-05-04):**
- ✅ All external vertical import boundary tests pass (13/13)
- ✅ External verticals (victor-coding, victor-research, victor-invest) verified to not import from `victor.agent.*`
- ✅ External verticals only import from allowed public APIs: `victor.framework.*`, `victor.tools.*`, `victor_sdk.*`
- ✅ No KNOWN_VIOLATIONS baseline entries needed
- ✅ Framework extension modules (extensions, processing, lsp) export all documented symbols
- **Conclusion:** Subservices removal has zero impact on external packages - they were never allowed to use those modules

## Migration Update: ToolRegistrar Plugin Loading Delegation (2026-05-04)

**Seam consolidated:** Plugin loading logic in ToolRegistrar

**Canonical owner:** PluginLoader component (`victor/agent/plugin_loader.py`)

**Changes applied:**

1. **Refactored ToolRegistrar._load_plugin_tools()** to delegate to PluginLoader
   - Removed ~40 lines of ToolPluginRegistry management code
   - Changed to call PluginLoader.load() for plugin lifecycle management
   - Maintained backward compatibility by storing plugin_manager reference
   - ToolRegistrar is now a thinner facade coordinating components

2. **Improved SRP compliance**
   - ToolRegistrar focuses on coordination, not implementation
   - PluginLoader is the single authority for plugin lifecycle
   - Clearer separation of concerns between facade and component

3. **All tests pass**
   - 54 tool_registrar tests pass
   - Plugin initialization tests verify delegation
   - Backward compatibility maintained

**Benefits:**

- Cleaner architecture: facade delegates to specialized components
- Easier to test: plugin loading tested in isolation in PluginLoader
- Easier to extend: new plugin sources can be added to PluginLoader
- Reduced complexity in ToolRegistrar

**Breaking changes:** None. Backward compatibility maintained through plugin_manager reference.

**Follow-up item #2** from migration audit: ToolRegistrar plugin-loading abstraction now properly delegates to PluginLoader component.

## Migration Update: Facade Boundary Guard (2026-05-04)

**Seam consolidated:** Facade behavior ownership across the agent facade layer

**Canonical owners:** Runtime behavior remains service-owned
(`ChatService`, `ToolService`, `SessionService`, `ContextService`,
`ProviderService`, `RecoveryService`) or state-passed where explicitly
modeled. Facades are grouping and compatibility surfaces only.

**Changes applied:**

1. Updated facade module/package docstrings to state the intended boundary
   directly: property access, grouping, and compatibility only.
2. Added AST guard test
   (`tests/unit/agent/facades/test_facade_boundary_guard.py`) that prevents the
   facade layer from growing new behavior-owning methods.
3. Verified the current facade shape:
   - `ChatFacade`: constructor + property accessors only
   - `ToolFacade`: constructor + property accessors only
   - `OrchestrationFacade`: property accessors plus deprecated compatibility
     warnings/lazy shim materialization only
   - `ProviderFacade`: property accessors plus explicit private compatibility
     shim materialization helpers only
   - `SessionFacade`, `WorkflowFacade`, `ResilienceFacade`, `MetricsFacade`:
     constructor + property accessors only, with deprecation warnings only
     where compatibility aliases remain

**Breaking changes:** None. This batch adds guardrails and documentation only.

## Migration Update: Provider Facade State Read-Through (2026-05-04)

**Seam consolidated:** Mutable provider configuration cached in both the
orchestrator runtime and `ProviderFacade`

**Canonical owner:** The concrete orchestrator runtime remains the canonical
owner for mutable provider state (`provider`, `model`, `provider_name`,
`temperature`, `max_tokens`, `thinking`). `ProviderFacade` is now an explicit
compatibility view over that state, not a second source of truth.

**Changes applied:**

1. Updated `victor/agent/facades/provider_facade.py` so provider-config
   properties read through to an injected `runtime_state_host` when present.
2. Updated compatibility setters on `ProviderFacade` to write through to the
   same canonical runtime host while preserving local fallback values for tests
   and isolated construction.
3. Updated `victor/agent/runtime/bootstrapper.py` to pass the orchestrator as
   the provider facade's `runtime_state_host`.
4. Added regression coverage proving that:
   - a materialized provider facade reflects later orchestrator provider-state
     changes
   - compatibility setters update the canonical runtime host instead of only
     mutating facade-local copies

**Benefits:**

- Removed a live duplication seam for provider configuration
- Prevented stale facade snapshots after runtime provider/model changes
- Kept `ProviderFacade` aligned with its intended compatibility-only role
- Added TDD coverage around the canonical ownership boundary

**Breaking changes:** None. Public compatibility access remains available; the
change narrows ownership internally.

## Migration Update: Tool And Session Facade Runtime-State Read-Through (2026-05-04)

**Seam consolidated:** Mutable tool/session compatibility state cached in
facades instead of reflecting the canonical orchestrator runtime.

**Canonical owners:**

- `tool_budget` remains owned by the concrete orchestrator runtime and the
  canonical tool-service path.
- session runtime state (`_session_ledger`, `active_session_id`,
  `_memory_session_id`) remains owned by the concrete orchestrator runtime and
  the canonical session-service path.
- `ToolFacade` and `SessionFacade` remain compatibility/grouping views only.

**Changes applied:**

1. Updated `victor/agent/facades/tool_facade.py` so `tool_budget` reads through
   an injected `runtime_state_host` when present and writes back through the
   same host via the compatibility setter.
2. Updated `victor/agent/facades/session_facade.py` so `session_ledger`,
   `active_session_id`, and `memory_session_id` read through the canonical
   runtime host and update that host via compatibility setters.
3. Updated `victor/agent/runtime/bootstrapper.py` to pass the orchestrator as
   the runtime-state host for both `ToolFacade` and `SessionFacade`.
4. Added regression coverage proving that materialized tool/session facades
   reflect later orchestrator state changes instead of holding stale snapshots.
5. Added repo-level reminders in `AGENTS.md` and `CLAUDE.md` that the target
   architecture is service-first with selective state-passed seams, and that
   facades/coordinator shims are compatibility-only.

**Benefits:**

- Removed two more live duplication seams from the facade layer
- Prevented stale compatibility views after runtime tool/session mutations
- Kept ownership aligned with the determined service-first target state
- Added guard-level guidance to reduce future architectural drift

**Breaking changes:** None. Compatibility access remains available; ownership
is simply narrower and more explicit.

## Migration Update: Chat Facade Runtime-State Read-Through (2026-05-04)

**Seam consolidated:** Mutable chat compatibility state cached in `ChatFacade`
instead of reflecting the canonical orchestrator runtime.

**Canonical owners:**

- `_system_prompt` remains owned by the concrete orchestrator runtime and the
  canonical prompt/chat runtime path.
- `_conversation_embedding_store`, `_context_compactor`, and
  `_memory_session_id` remain owned by the concrete orchestrator runtime and
  service-owned runtime collaborators.
- `ChatFacade` remains a grouping/compatibility view only.

**Changes applied:**

1. Updated `victor/agent/facades/chat_facade.py` so `memory_session_id`,
   `embedding_store`, `system_prompt`, and `context_compactor` read through an
   injected `runtime_state_host` when present.
2. Updated compatibility setters on `ChatFacade` for `embedding_store`,
   `system_prompt`, and `context_compactor` to write back to that canonical
   runtime host.
3. Updated `victor/agent/runtime/bootstrapper.py` to pass the orchestrator as
   the chat facade's runtime-state host.
4. Added regression coverage proving that a materialized chat facade reflects
   later orchestrator prompt/embedding/context/session-id changes instead of
   holding stale snapshots.

**Benefits:**

- Removed another live duplication seam from the facade layer
- Prevented stale chat compatibility views after runtime prompt/context changes
- Kept chat ownership aligned with the determined service-first target state
- Extended the same explicit runtime-state pattern already applied to provider,
  tool, and session compatibility facades

**Breaking changes:** None. Compatibility access remains available; ownership
is simply narrower and more explicit.

## Migration Update: Workflow And Resilience Facade Runtime-State Read-Through (2026-05-04)

**Seam consolidated:** Mutable workflow/resilience compatibility state cached
in facades instead of reflecting the canonical orchestrator runtime.

**Canonical owners:**

- `_workflow_registry` and `_coordination_advisor` remain owned by the
  concrete orchestrator runtime and the canonical workflow/service-owned path.
- `_recovery_handler`, `_recovery_integration`, `_cancel_event`, and
  `_is_streaming` remain owned by the concrete orchestrator runtime and the
  canonical recovery/resilience service path.
- `WorkflowFacade` and `ResilienceFacade` remain grouping/compatibility views
  only.

**Changes applied:**

1. Updated `victor/agent/facades/workflow_facade.py` so `workflow_registry`
   and `coordination_advisor` read through an injected `runtime_state_host`
   when present and write back through compatibility setters.
2. Updated `victor/agent/facades/resilience_facade.py` so `recovery_handler`,
   `recovery_integration`, `cancel_event`, and `is_streaming` read through the
   canonical runtime host and write back through compatibility setters.
3. Updated `victor/agent/runtime/bootstrapper.py` to pass the orchestrator as
   the runtime-state host for both `WorkflowFacade` and `ResilienceFacade`.
4. Added regression coverage proving that materialized workflow/resilience
   facades reflect later orchestrator state changes instead of holding stale
   snapshots.

**Benefits:**

- Removed two more live duplication seams from the facade layer
- Prevented stale compatibility views after workflow/recovery runtime mutations
- Kept workflow and resilience ownership aligned with the service-first target
  state

## Migration Update: ModeWorkflowTeamCoordinator Shim Removed (2026-05-05)

**Seam consolidated:** `ModeWorkflowTeamCoordinator` and its related
`victor.agent` compatibility aliases still existed even though the live
workflow recommendation path had already converged on
`coordination_advisor` / `VerticalCoordinationAdvisor`.

**Canonical owner:** `coordination_advisor` backed by
`victor.framework.coordination_runtime.VerticalCoordinationAdvisor`.

**Changes applied:**

1. Deleted `victor.agent.mode_workflow_team_coordinator`.
2. Removed `WorkflowFacade.mode_workflow_team_coordinator`.
3. Removed `AgentOrchestrator._mode_workflow_team_coordinator`.
4. Removed `OrchestratorFactory.create_mode_workflow_team_coordinator(...)`.
5. Replaced unit and integration coverage with removal assertions while
   preserving canonical `coordination_advisor` behavior.

**Benefits:**

- Eliminated a dead workflow-coordination wrapper that no longer owned runtime
  behavior
- Removed one more facade/orchestrator alias path that could have reintroduced
  architectural drift
- Left `coordination_advisor` as the only workflow recommendation surface
  inside `victor.agent`

**Breaking changes:** Yes. `victor.agent.mode_workflow_team_coordinator`,
`WorkflowFacade.mode_workflow_team_coordinator`,
`AgentOrchestrator._mode_workflow_team_coordinator`, and
`OrchestratorFactory.create_mode_workflow_team_coordinator(...)` have been
removed.

## Migration Update: ModeWorkflowTeamCoordinatorProtocol Alias Removed (2026-05-05)

**Seam consolidated:** `victor.protocols.coordination` still exported
`ModeWorkflowTeamCoordinatorProtocol` as a public alias even after all
corresponding `victor.agent` workflow-coordination shim surfaces had been
removed.

**Canonical owner:** `victor.protocols.coordination.CoordinationAdvisorProtocol`.

**Changes applied:**

1. Removed `ModeWorkflowTeamCoordinatorProtocol` from
   `victor.protocols.coordination`.
2. Removed the alias from `victor.protocols.coordination.__all__`.
3. Added unit and consolidation integration coverage proving the alias no
   longer imports while `CoordinationAdvisorProtocol` remains canonical.

**Benefits:**

- Aligns the public protocol surface with the already-removed workflow shim
  implementation path
- Prevents new external or internal callers from rebuilding the deleted
  mode-workflow naming seam around the canonical coordination protocol
- Leaves one clear public protocol name for workflow/team coordination

**Breaking changes:** Yes. `from victor.protocols.coordination import
ModeWorkflowTeamCoordinatorProtocol` no longer works.
- Extended the same explicit runtime-state pattern already applied to provider,
  tool, session, and chat compatibility facades

**Breaking changes:** None. Compatibility access remains available; ownership
is simply narrower and more explicit.

## Migration Update: Orchestration Facade Runtime-State Read-Through (2026-05-04)

**Seam consolidated:** Mutable orchestration compatibility state cached in
`OrchestrationFacade` instead of reflecting the canonical orchestrator runtime.

**Canonical owners:**

- `_chat_stream_adapter`, `_turn_executor`, `_protocol_adapter`,
  `_iteration_coordinator`, and `_observability` remain owned by the concrete
  orchestrator runtime.
- `_runtime_intelligence_integration` and `_subagent_orchestrator` remain
  orchestrator-owned lazy runtime integrations.
- deprecated coordinator shim slots remain orchestrator-owned compatibility
  state, not facade-owned behavior.
- `OrchestrationFacade` remains a grouping/compatibility view only.

**Changes applied:**

1. Updated `victor/agent/facades/orchestration_facade.py` so live runtime
   surfaces (`chat_stream_adapter`, `turn_executor`, `protocol_adapter`,
   `iteration_coordinator`, `observability`,
   `runtime_intelligence_integration`, and `subagent_orchestrator`) read
   through an injected `runtime_state_host` when present.
2. Updated orchestration compatibility setters to write back through that same
   canonical runtime host instead of mutating facade-local snapshots.
3. Updated deprecated coordinator shim accessors in `OrchestrationFacade` so
   materialized facades reflect later orchestrator shim-slot changes and
   deprecated shim setters update the canonical host state.
4. Updated `victor/agent/runtime/bootstrapper.py` to pass the orchestrator as
   the runtime-state host for `OrchestrationFacade`.
5. Added regression coverage proving that a materialized orchestration facade
   reflects later orchestrator runtime/shim mutations instead of holding stale
   copies.

**Benefits:**

- Removed the last major live duplication seam from the facade layer
- Prevented stale orchestration compatibility views after runtime mutations
- Kept orchestration ownership aligned with the service-first target state
- Narrowed facades further toward grouping/compatibility only

**Breaking changes:** None. Compatibility access remains available; ownership
is simply narrower and more explicit.

## Migration Update: Stage-Transition Runtime Relocated To Services (2026-05-04)

**Seam consolidated:** Active stage-transition batching runtime lived under
`victor/agent/coordinators/` even though it is effectful runtime behavior, not
one of the selective state-passed policy seams.

**Canonical owners:**

- `victor.agent.services.stage_transition_runtime.StageTransitionCoordinator`
  is now the canonical runtime owner for batched conversation-stage
  transitions.
- `victor.agent.services.stage_transition_strategies` is now the canonical
  home for the stage-transition strategy implementations.
- `victor.agent.coordinators.stage_transition_coordinator` and
  `victor.agent.coordinators.transition_strategies` remain compatibility
  re-export modules only.

**Changes applied:**

1. Added canonical service-owned modules:
   - `victor/agent/services/stage_transition_runtime.py`
   - `victor/agent/services/stage_transition_strategies.py`
2. Reduced the legacy coordinator modules to thin compatibility re-exports.
3. Updated `victor/agent/runtime/component_assembler.py` so the feature-flagged
   production runtime imports the canonical service-owned stage-transition
   implementation rather than importing from `victor.agent.coordinators`.
4. Exported the canonical stage-transition runtime surfaces from
   `victor.agent.services`.
5. Added an AST-based guard test preventing new internal production imports of
   stage-transition runtime helpers from coordinator paths.

**Benefits:**

- Removed another active runtime ownership seam from `victor/agent/coordinators`
- Kept effectful runtime behavior aligned with the service-first target state
- Preserved old coordinator import paths as explicit compatibility-only shims
- Added regression guardrails against future drift back to coordinator-owned
  runtime behavior

**Breaking changes:** None. Legacy coordinator module imports continue to work
as compatibility re-export paths.

## Migration Update: CoordinatorFactory Demoted To Explicit SDK Compatibility (2026-05-04)

**Seam consolidated:** `CoordinatorFactory` still presented conversation and
safety coordinator creation as if those were active `victor/agent` runtime
surfaces, even though the repo already treats them as SDK-owned compatibility
or extension contracts.

**Canonical owners:**

- `create_safety_state_passed_coordinator()` remains the canonical
  `victor/agent` runtime path for safety policy evaluation.
- `victor_sdk.safety.SafetyCoordinator` remains the SDK-owned extension
  contract for external safety extensions.
- `victor_sdk.conversation.ConversationCoordinator` remains the SDK-owned
  extension contract for conversation-oriented helpers.
- `CoordinatorFactory.create_safety_coordinator()` and
  `CoordinatorFactory.create_conversation_coordinator()` now exist only as
  deprecated compatibility helpers.

**Changes applied:**

1. Updated `victor/agent/coordinators/coordinator_factory.py` so the legacy
   safety and conversation factory methods instantiate the SDK coordinator
   classes directly instead of importing removed local modules.
2. Added explicit `DeprecationWarning` messages that direct agent-runtime code
   toward the canonical state-passed safety path and direct extension code
   toward `victor_sdk`.
3. Removed dead DI adapter plumbing from those compatibility methods.
4. Added regression coverage proving those methods return SDK surfaces and that
   `CoordinatorFactory` no longer imports the removed local coordinator
   modules.

**Benefits:**

- Fixed two broken legacy factory methods that referenced non-existent local
  coordinator modules
- Made the compatibility boundary explicit instead of leaving a fake ownership
  story in place
- Reduced dead protocol-adapter plumbing in `CoordinatorFactory`
- Added guardrails against drift back to local coordinator ownership for
  SDK-based seams

**Breaking changes:** None for working call sites. The factory methods remain
available, but they now warn explicitly and behave as thin SDK compatibility
helpers.

## Migration Update: Coordinator Package SDK Re-exports Marked Compatibility-Only (2026-05-04)

**Seam consolidated:** `victor.agent.coordinators` still exposed SDK-owned
safety and conversation symbols silently, which made it easy for internal code
to keep treating them like active `victor/agent` runtime architecture.

**Canonical owners:**

- `victor_sdk.safety.*` is the canonical import surface for SDK-owned safety
  types.
- `victor_sdk.conversation.*` is the canonical import surface for SDK-owned
  conversation types.
- `victor.agent.coordinators.SafetyStatePassedCoordinator` remains the
  canonical agent-runtime safety wrapper.
- `victor.agent.coordinators.*` exports for SDK-owned safety/conversation types
  now exist only as deprecated compatibility shims.

**Changes applied:**

1. Updated `victor/agent/coordinators/__init__.py` to warn explicitly when
   callers resolve SDK-owned safety or conversation exports through
   `victor.agent.coordinators`.
2. Added regression coverage proving the compatibility exports still resolve to
   the same SDK types while warning.
3. Added an AST-based import-boundary guard to prevent internal production code
   from importing SDK-owned safety/conversation symbols through
   `victor.agent.coordinators`.

**Benefits:**

- Made the coordinator package compatibility boundary explicit at access time
- Preserved backward compatibility for existing import paths that still need to
  resolve
- Added guardrails against new internal drift toward the wrong package-level
  ownership surface

**Breaking changes:** None. Compatibility exports still resolve, but they now
emit `DeprecationWarning` and point callers to the canonical SDK packages.

## Migration Update: State Runtime Compatibility Story Corrected (2026-05-04)

**Seam consolidated:** The runtime had already moved to
`StateRuntimeAdapter` over `ConversationController`, but the migration audit
and a few migration-focused tests still described deleted concrete state shims
as if they remained available.

**Canonical owners:**

- `victor.agent.services.state_runtime.StateRuntimeAdapter` is the canonical
  live conversation-stage runtime adapter.
- `victor.agent.conversation.controller.ConversationController` and
  `victor.agent.conversation.state_machine.ConversationStateMachine` own the
  underlying live conversation state.
- `victor.agent.state_service.StateService` remains the persisted vertical
  state surface.
- `victor.agent.protocols.StateCoordinatorProtocol` remains only as a
  deprecated protocol alias; there is no concrete `StateCoordinator` module.

**Changes applied:**

1. Corrected the migration audit and current-state docs to say the concrete
   `state_compat.py` and `state_coordinator.py` shims are gone.
2. Added regression coverage proving those deleted modules remain absent and
   cannot be re-imported by internal production code.
3. Updated migration-focused test language so it no longer implies a concrete
   `StateCoordinator` class still exists.

**Benefits:**

- Removed a stale architecture claim from the canonical migration evidence
- Made the actual state-runtime end state explicit: conversation-owned live
  state with a service-owned adapter, not a lingering coordinator shim
- Added guardrails against accidentally reintroducing deleted state shim
  modules

**Breaking changes:** None. The concrete state shims were already absent; this
batch aligns docs and guardrails with the actual repo state.

## Migration Update: Runtime Support Protocols Rehosted Under Services (2026-05-04)

**Seam consolidated:** `victor.agent.services.protocols.runtime_support` still
implemented its core coordination/runtime protocol names as aliases over
coordinator-era protocol definitions, which left the wrong module as the real
host for service-owned runtime abstractions.

**Canonical owners:**

- `victor.agent.services.protocols.runtime_support` now owns:
  - `CoordinationAdvisorRuntimeProtocol`
  - `ToolPlanningRuntimeProtocol`
  - `TaskRuntimeProtocol`
  - `StateRuntimeProtocol`
  - `PromptRuntimeProtocol`
- `victor.agent.protocols` compatibility names now alias back to those
  service-owned protocol definitions and remain deprecated.

**Changes applied:**

1. Moved the core coordination/runtime protocol definitions into
   `victor.agent.services.protocols.runtime_support`.
2. Reduced `victor.agent.protocols.coordination_protocols` to a compatibility
   alias layer for those runtime protocol names while keeping truly legacy-only
   coordination protocols in place.
3. Added regression coverage proving the canonical runtime protocol objects are
   hosted by `runtime_support.py` while preserving identity with deprecated
   compatibility imports.
4. Added an AST-based import guard preventing internal production code from
   importing the deprecated coordinator-era protocol names.

**Benefits:**

- Made the service layer the actual host of service-owned runtime protocols
- Preserved backward compatibility for deprecated protocol imports without
  leaving coordinator-era modules as the source of truth
- Added guardrails against future drift back to `victor.agent.protocols` for
  active runtime protocol surfaces

**Breaking changes:** None. Legacy protocol imports still resolve and warn, but
the canonical definitions now live under `victor.agent.services.protocols`.

## Migration Update: Remaining Runtime Support Protocols Rehosted Under Services (2026-05-04)

**Seam consolidated:** After the coordination/runtime protocol move,
`runtime_support.py` still aliased the remaining active runtime contracts for
chunk generation, streaming recovery, and RL back to legacy protocol modules.

**Canonical owners:**

- `victor.agent.services.protocols.runtime_support` now also owns:
  - `ChunkRuntimeProtocol`
  - `StreamingRecoveryRuntimeProtocol`
  - `RLLearningRuntimeProtocol`
- `victor.agent.protocols.streaming_protocols.ChunkGeneratorProtocol`
- `victor.agent.protocols.streaming_protocols.StreamingRecoveryCoordinatorProtocol`
- `victor.agent.protocols.infrastructure_protocols.RLCoordinatorProtocol`
  now exist only as deprecated compatibility aliases back to the service-owned
  protocol definitions.

**Changes applied:**

1. Moved the chunk-generation, streaming-recovery, and RL runtime protocol
   definitions into `victor.agent.services.protocols.runtime_support`.
2. Reduced the legacy streaming and infrastructure protocol modules to
   compatibility alias hosts for those names.
3. Extended regression coverage to assert those protocol objects are hosted by
   `runtime_support.py` while preserving identity with deprecated imports.
4. Expanded the AST-based import guard so internal production code cannot
   import the legacy chunk/recovery/RL protocol names.

**Benefits:**

- Completed the service-owned hosting move for the active runtime protocol
  surface in `runtime_support.py`
- Preserved backward compatibility for deprecated protocol imports while making
  the service layer the real source of truth
- Added guardrails against drift back to legacy protocol modules for active
  runtime seams

**Breaking changes:** None. Legacy protocol imports still resolve and warn, but
the canonical definitions now live under `victor.agent.services.protocols`.

## Migration Update: Runtime Infrastructure Protocols Rehosted Under Services (2026-05-04)

**Seam consolidated:** `victor.agent.services.protocols.infrastructure_runtime`
still aliased the remaining runtime-facing infrastructure contracts back to
legacy analysis, infrastructure, and streaming protocol modules. That left the
service layer presenting canonical names while the old protocol package still
owned the actual definitions.

**Canonical owners:**

- `victor.agent.services.protocols.infrastructure_runtime` now owns:
  - `IntentClassifierProtocol`
  - `ReminderManagerProtocol`
  - `ResponseSanitizerProtocol`
  - `StreamingHandlerProtocol`
  - `StreamingMetricsCollectorProtocol`
  - `StreamingConfidenceMonitorProtocol`
- Legacy names in:
  - `victor.agent.protocols.analysis_protocols`
  - `victor.agent.protocols.infrastructure_protocols`
  - `victor.agent.protocols.streaming_protocols`
  now exist only as compatibility aliases back to the service-owned protocol
  definitions.

**Changes applied:**

1. Moved the remaining runtime-facing infrastructure protocol definitions into
   `victor.agent.services.protocols.infrastructure_runtime`.
2. Reduced the legacy analysis, infrastructure, and streaming protocol modules
   to compatibility alias layers for those names.
3. Updated `victor.agent.services.protocols.streaming_runtime` and
   `victor.agent.orchestrator` to import the canonical service-owned protocol
   surfaces instead of the legacy protocol package.
4. Extended regression coverage and the AST import guard so internal
   production code cannot drift back to the deprecated runtime protocol names.

**Benefits:**

- Completed service-owned hosting for the active runtime-facing infrastructure
  protocol surface
- Preserved backward compatibility for deprecated protocol imports while making
  the service layer the actual source of truth
- Removed the last production runtime import of these protocol names from
  `victor.agent.protocols`

**Breaking changes:** None. Legacy protocol imports still resolve and warn, but
the canonical definitions now live under `victor.agent.services.protocols`.

## Migration Update: PromptRuntimeSupport Demoted To Compatibility-Only (2026-05-04)

**Seam consolidated:** `ComponentAssembler` and prompt-builder fallback helpers
still wired `PromptRuntimeSupport` into the live orchestrator runtime even
though `UnifiedPromptPipeline` already owned the active prompt assembly path.

**Canonical owners:**

- `victor.agent.prompt_pipeline.UnifiedPromptPipeline` remains the canonical
  owner of orchestrator-managed live prompt assembly
- `victor.agent.services.prompt_runtime.PromptRuntimeAdapter` remains the DI
  surface for `PromptRuntimeProtocol`
- `victor.agent.services.prompt_runtime_support.PromptRuntimeSupport` and
  `victor.agent.services.system_prompt_runtime.SystemPromptCoordinator` are now
  compatibility-only prompt helper surfaces

**Changes applied:**

1. Removed live runtime assembly of `PromptRuntimeSupport` from
   `ComponentAssembler.assemble_conversation(...)`.
2. Updated `PromptBuilderRuntime.build_system_prompt_fallback()` to use the
   orchestrator's own prompt-used hook instead of reaching into a support
   object.
3. Updated orchestrator prompt helper fallbacks (`_resolve_shell_variant`,
   `_classify_task_keywords`, `_classify_task_with_context`) to use direct
   runtime dependencies when the pipeline is unavailable.
4. Added a guard preventing new internal runtime imports of
   `PromptRuntimeSupport` or `create_prompt_runtime_support(...)` outside the
   explicit compatibility files.

**Benefits:**

- Removed the last internal runtime assembly dependency on
  `PromptRuntimeSupport`
- Kept deprecated coordinator/support paths available for compatibility
  without leaving them as production runtime owners
- Clarified the prompt architecture boundary: pipeline for live orchestration,
  protocol adapter for DI, coordinator/support only for compatibility

**Breaking changes:** None. `PromptRuntimeSupport` and `SystemPromptCoordinator`
still exist for compatibility, but the internal orchestrator runtime no longer
depends on them.

## Migration Update: Prompt Optimization Wiring Preserved Across Provider Tiers (2026-05-04)

**Seam consolidated:** `AgentOrchestrator.get_assembled_messages()` was still
gating prompt-pipeline turn-prefix injection on `_kv_optimization_enabled`,
which meant Tier C / non-KV providers could bypass live prompt optimization
content even though `UnifiedPromptPipeline` already owned the canonical
GEPA/MiPROv2/CoT/failure/credit path.

**Canonical owners:**

- `victor.agent.prompt_pipeline.UnifiedPromptPipeline` remains the canonical
  live prompt optimization and per-turn prefix owner
- `victor.agent.services.runtime_intelligence.RuntimeIntelligenceService`
  remains the canonical bundle provider for GEPA/MiPROv2/experiment-memory
  prompt guidance
- KV-specific flags remain cache-strategy controls only; they are not the
  ownership boundary for whether prompt optimization executes

**Changes applied:**

1. Updated `AgentOrchestrator.get_assembled_messages()` so prompt-pipeline
   turn-prefix composition runs whenever the canonical pipeline is present,
   not only when KV prefix caching is enabled.
2. Kept KV-only observability (`_kv_prefix_fingerprint()` logging) scoped to
   KV-capable providers.
3. Added regression coverage proving non-KV/Tier C providers still route
   prompt optimizations and one-shot failure hints through the live pipeline.
4. Added repo guidance in `AGENTS.md` and `CLAUDE.md` so future migration work
   does not reintroduce a KV-only optimization assumption.

## Migration Update: Prompt Invalidation and Core-vs-Dynamic Tool Prompting (2026-05-04)

**Seam consolidated:** frozen prompt management and tool-prompt placement were
still too implicit. The runtime could detect some state changes only through
manual refresh paths, query classification on frozen tiers could be dropped
instead of moved to the per-turn path, and the system prompt still treated the
entire enabled tool set as one undifferentiated guidance surface.

**Canonical owners:**

- `victor.agent.services.prompt_builder_runtime.PromptBuilderRuntime` now owns
  frozen-prompt invalidation and dynamic prompt-side tool/task guidance
  derivation.
- `victor.agent.prompt_pipeline.UnifiedPromptPipeline` remains the canonical
  per-turn user-prefix owner for query guidance, prompt optimization, and
  long-tail tool hints.
- `victor.agent.prompt_builder.SystemPromptBuilder` remains the canonical
  system-prompt builder, but now distinguishes stable core-tool guidance from
  dynamic long-tail tool hints.

**Changes applied:**

1. Added explicit prompt-runtime invalidation based on provider/model,
   mode-guidance, stable core-tool set, and `.victor/init.md` changes.
2. Force-reload project context when invalidation has already detected an
   `init.md` change, so prompt refreshes do not reuse stale TTL-cached content.
3. Preserved query classification on frozen tiers by moving task guidance into
   the pipeline-backed per-turn prefix instead of forcing a system-prompt
   rebuild.
4. Split tool prompting into stable core-tool guidance for the system prompt
   and per-turn dynamic hints for long-tail tools, reusing provider-aware tool
   tier configuration rather than introducing a second tool-core taxonomy.
5. Switched the default KV-only tool strategy toward `context_aware` so local
   KV providers favor stable core tools and economy-first locking by default.

**Benefits:**

- Stable prompts now stay fresh based on explicit invalidators instead of only
  workspace resets or manual refresh calls.
- Prompt quality no longer regresses on frozen tiers when query classification
  changes mid-session.
- Tool prompting is cheaper and more stable: core tools remain in the stable
  prefix while less-common tool hints move to the per-turn path only when
  relevant.
- KV-only providers now default closer to the desired latency/accuracy balance
  without disabling live prompt optimization.

**Breaking changes:** None intended. The public runtime surfaces are unchanged,
but the default `ContextSettings.kv_tool_strategy` is now `context_aware`
instead of `per_turn`.

## Migration Update: Dynamic Tool Hints Prefer Planned Tools (2026-05-04)

**Seam consolidated:** dynamic long-tail tool hints were improved in the
prompt/runtime layer, but session-locked providers still leaned too heavily on
generic keyword hints because the per-turn prompt path did not receive the
current planned tool sequence.

**Canonical owners:**

- `victor.agent.services.tool_planning_runtime.ToolPlanner` remains the
  canonical source of current-turn planned tools.
- `victor.agent.services.prompt_builder_runtime.PromptBuilderRuntime` now
  prefers planned tools when choosing long-tail prompt hints.
- `victor.agent.services.chat_stream_executor.StreamingChatExecutor` and
  `victor.agent.services.chat_stream_helpers` explicitly carry that planned
  sequence through the streaming provider-call path.

**Changes applied:**

1. Added `planned_tools` to the streaming context and passed it into message
   assembly for both normal and retry provider calls.
2. Updated tool-selection runtime to accept precomputed planned tools so the
   tool plan does not need to be recomputed when the caller already has it.
3. Updated dynamic tool-hint selection to prefer planned tools first, then
   selector-derived keyword slices, then selected-tool/message heuristics.

**Benefits:**

- Session-locked providers now get per-turn long-tail hints that align with the
  active execution plan instead of generic keyword matches.
- The stable system prefix remains unchanged; only the user-prefix hint block
  reflects current planning state.
- Tool planning remains singular; prompt hinting reuses it instead of adding a
  second planning abstraction.

**Breaking changes:** None intended.

**Benefits:**

- Preserved live prompt optimization wiring for all providers, including Tier C
  providers without KV caching
- Kept GEPA, MiPROv2, CoT distillation, failure hints, experiment-memory
  guidance, and credit-driven prompt guidance on the canonical runtime path
- Clarified the architectural boundary: KV support affects prefix-stability and
  cache economics, not whether prompt optimization logic executes

**Breaking changes:** None. This restores the intended canonical runtime path
without removing any compatibility surfaces.

## Migration Update: Dynamic Tool Hints Carry Planner Rationale (2026-05-04)

**Seam consolidated:** dynamic long-tail tool hints were already plan-aware,
but the rendered prompt text still looked generic. Cached/session-locked
providers were therefore getting the right long-tail tools without the compact
goal/intent rationale that made those hints actionable.

**Canonical owners:**

- `victor.agent.services.prompt_builder_runtime.PromptBuilderRuntime` remains
  the canonical runtime surface for selecting dynamic long-tail prompt hints.
- `victor.agent.prompt_builder.SystemPromptBuilder` owns the final rendering of
  those hints for the per-turn user-prefix path.
- `victor.agent.orchestrator.AgentOrchestrator` and
  `victor.agent.services.chat_stream_helpers` carry the current turn's planner
  goals into that runtime surface without introducing a second planning layer.

**Changes applied:**

1. `get_assembled_messages()` now forwards current planner goals alongside
   selected and planned tools into `PromptBuilderRuntime`.
2. `PromptBuilderRuntime` now tags the dynamic hint source
   (`planned_tools`, `keyword_selector`, `message_keywords`, or
   `selected_tools`) and passes compact goal/intent context when available.
3. `SystemPromptBuilder.get_dynamic_tool_guidance_text()` now renders that
   context as plan-focus / intent-guard rationale instead of only listing tool
   names.

**Benefits:**

- Dynamic long-tail hints remain cheap and per-turn, but now explain *why* a
  given less-common tool is relevant.
- The planner stays singular: prompt hinting reuses existing goals and intent
  state instead of recomputing planning logic inside the prompt layer.
- Stable system-prefix economics remain unchanged; only the user-prefix hint
  block becomes more specific.

**Breaking changes:** None intended.

## Migration Update: Dynamic Tool Hints Reuse Tool Metadata (2026-05-04)

**Seam consolidated:** dynamic long-tail tool hints already carried current
plan focus and intent guard context, but the per-turn hint text still lacked
compact per-tool rationale even when planned tools or registry metadata already
contained that information.

**Canonical owners:**

- `victor.agent.services.prompt_builder_runtime.PromptBuilderRuntime` remains
  the runtime owner for selecting dynamic long-tail hints and now extracts
  concise per-tool rationale from existing planned/selected tool objects or the
  tool registry.
- `victor.agent.prompt_builder.SystemPromptBuilder` remains the rendering owner
  for the user-prefix hint block.

**Changes applied:**

1. Added compact tool-rationale extraction from planned tools, selected tools,
   and the registry, preferring explicit metadata use-cases/priority hints and
   falling back to tool descriptions.
2. Passed that rationale into `SystemPromptBuilder.get_dynamic_tool_guidance_text()`
   alongside the existing plan-focus and intent-guard context.
3. Rendered the rationale as a short `tool (reason)` list inside the existing
   dynamic user-prefix hint block.

**Benefits:**

- Dynamic long-tail hints now explain why a specific less-common tool is
  relevant using information the runtime already has.
- Prompt specificity improves without creating a second planning or selection
  abstraction.
- Stable-prefix economics remain unchanged; only the per-turn user-prefix hint
  block becomes more informative.

**Breaking changes:** None intended.

## Migration Update: TurnExecutor Exploration Now Prefers State-Passed Runtime (2026-05-05)

**Seam consolidated:** `TurnExecutor` still bypassed the bootstrapped
`exploration_state_passed` surface and lazily imported the direct exploration
helper path inside `turn_execution_runtime.py`.

**Canonical owners:**

- `victor.agent.coordinators.ExplorationStatePassedCoordinator` is the
  canonical internal runtime surface for orchestrator-backed exploration
  decisions
- `victor.agent.services.exploration_runtime.ExplorationCoordinator` remains a
  direct service runtime only for fallback contexts that do not have an
  orchestrator snapshot

**Changes applied:**

1. Refactored `TurnExecutor` to prefer the shared
   `orchestration_facade.exploration_state_passed` coordinator whenever an
   orchestrator-backed runtime is available.
2. Added state-passed execution wiring that:
   - creates a `ContextSnapshot`
   - injects the current task complexity into snapshot capabilities
   - applies returned exploration transitions back onto orchestrator state
3. Extended `ExplorationStatePassedCoordinator.explore(...)` so callers can
   override `project_root` and `max_results` per turn while preserving the
   coordinator’s default behavior.
4. Removed the remaining production import of
   `create_exploration_coordinator()` from `turn_execution_runtime.py`.
5. Added import-boundary guard coverage to prevent new internal runtime code
   from importing coordinator-owned direct exploration helpers.
6. Added unit and integration regression coverage proving `TurnExecutor` now:
   - prefers the shared state-passed surface
   - updates orchestrator conversation state through returned transitions
   - preserves the direct service-runtime fallback when no orchestrator is
     available

**Benefits:**

- Internal production runtime now matches the documented target shape for the
  exploration seam
- Exploration findings are applied through explicit transitions instead of
  ad hoc runtime mutation
- The direct exploration runtime remains available without regressing the
  state-passed architecture inside orchestrator-owned flows

**Breaking changes:** None intended. Public factory access to direct
`ExplorationCoordinator` remains available; only the internal `TurnExecutor`
selection path changed.

## Migration Update: PromptRuntimeAdapter Delegates To UnifiedPromptPipeline (2026-05-05)

**Seam consolidated:** `PromptRuntimeAdapter` still owned a separate
`PromptBuilder`-driven system-prompt assembly path even though
`UnifiedPromptPipeline` was already the canonical prompt owner for the
orchestrator runtime.

**Canonical owners:**

- `victor.agent.prompt_pipeline.UnifiedPromptPipeline` remains the single
  owner of prompt assembly decisions
- `victor.agent.services.prompt_runtime.PromptRuntimeAdapter` remains the DI
  surface for `PromptRuntimeProtocol`, but now forwards its build path through
  the canonical pipeline
- `victor.agent.services.prompt_runtime_support.PromptRuntimeSupport` and
  `victor.agent.services.system_prompt_runtime.SystemPromptCoordinator` remain
  compatibility-only prompt helper surfaces

**Changes applied:**

1. Refactored `PromptRuntimeAdapter.build_system_prompt(...)` to prepare its
   mutable runtime state on a fresh builder, then delegate final system-prompt
   assembly through `UnifiedPromptPipeline`.
2. Kept fallback policy behavior in the adapter so protocol consumers still
   get a safe prompt if pipeline-driven assembly fails.
3. Added regression coverage proving `PromptRuntimeAdapter` now invokes
   `UnifiedPromptPipeline` while preserving task hints, additional sections,
   safety rules, grounding mode, and extra context.
4. Updated current-state and design docs to mark prompt-runtime convergence
   Phase 1 as implemented.

**Benefits:**

- Removed the last separate system-prompt assembly path from the canonical
  prompt runtime adapter
- Preserved the narrow mutable prompt protocol without creating a new prompt
  owner
- Kept prompt-runtime convergence incremental: adapter delegation now matches
  the designed end state while compatibility shims stay untouched

**Breaking changes:** None. `PromptRuntimeProtocol` and
`PromptRuntimeAdapter` remain stable; only the internal build path changed.

## Migration Update: PromptRuntimeSupport Reduced To A Thin Compatibility Wrapper (2026-05-05)

**Seam consolidated:** `PromptRuntimeSupport` still carried its own
pipeline-bootstrap and prompt helper implementation even though
`SystemPromptCoordinator` already existed as the canonical compatibility shim
over `UnifiedPromptPipeline`.

**Canonical owners:**

- `victor.agent.prompt_pipeline.UnifiedPromptPipeline` remains the single
  owner of prompt assembly decisions
- `victor.agent.services.system_prompt_runtime.SystemPromptCoordinator` remains
  the compatibility wrapper for deprecated system-prompt coordinator imports
- `victor.agent.services.prompt_runtime_support.PromptRuntimeSupport` now
  exists only as a thin deprecated wrapper over `SystemPromptCoordinator`

**Changes applied:**

1. Refactored `PromptRuntimeSupport` to inherit shared compatibility behavior
   from `SystemPromptCoordinator` instead of bootstrapping its own pipeline.
2. Kept the `PromptRuntimeSupport`-specific deprecation warning and the legacy
   analyzer-method fallbacks (`classify_task_keywords`,
   `classify_task_with_context`) so old callers still behave correctly.
3. Added regression coverage proving `PromptRuntimeSupport.build_system_prompt`
   now routes through the coordinator compatibility layer rather than a
   separate local implementation.
4. Updated the current-state and prompt-architecture docs to mark prompt
   convergence Phase 2 as implemented.

**Benefits:**

- Removed duplicate compatibility logic from `PromptRuntimeSupport`
- Made the compatibility stack singular and easier to retire later:
  `PromptRuntimeSupport` -> `SystemPromptCoordinator` -> `UnifiedPromptPipeline`
- Preserved deprecated import behavior without reintroducing a parallel prompt
  owner

**Breaking changes:** None. Deprecated compatibility imports still resolve and
warn; only their internal delegation path changed.

## Migration Update: Prompt Compatibility Guardrails Tightened (2026-05-05)

**Seam consolidated:** The prompt compatibility import guard only blocked
direct `from ... import ...` usage, leaving easy drift paths through package
re-exports and plain module imports.

**Changes applied:**

1. Extended the AST import guard to detect
   `from victor.agent.services import SystemPromptCoordinator` and
   `from victor.agent.services import PromptRuntimeSupport`.
2. Extended the same guard to detect direct module imports such as
   `import victor.agent.services.prompt_runtime_support`.
3. Added synthetic regression cases so future changes to the guard must keep
   covering those bypass forms.

**Benefits:**

- Reduces the chance that internal production code reattaches itself to the
  deprecated prompt compatibility layer through alternate import styles
- Makes the remaining compatibility seam easier to retire in a later breaking
  release because new internal dependencies are harder to introduce

**Breaking changes:** None. This only strengthens internal regression
guardrails.

## Migration Update: PromptRuntimeSupport Removed (2026-05-05)

**Seam consolidated:** `PromptRuntimeSupport` remained as an extra deprecated
module and `OrchestratorFactory.create_prompt_runtime_support(...)` remained as
an extra dead factory path even after the prompt runtime and compatibility
wrappers had converged on `UnifiedPromptPipeline`.

**Canonical owners:**

- `victor.agent.prompt_pipeline.UnifiedPromptPipeline` remains the single
  prompt assembly owner
- `victor.agent.services.prompt_runtime.PromptRuntimeAdapter` remains the DI
  implementation of `PromptRuntimeProtocol`
- no prompt compatibility wrapper remains on the live runtime path

**Changes applied:**

1. Deleted `victor.agent.services.prompt_runtime_support`.
2. Removed `create_prompt_runtime_support(...)` from the coordinator helper and
   `OrchestratorFactory` compatibility builder surface.
3. Replaced direct module tests with removal assertions and added integration
   coverage proving the module is no longer importable while canonical prompt
   surfaces still work.
4. Updated the prompt-runtime import guard and current-state docs to reflect
   the new breaking state.

**Benefits:**

- Removed the last extra compatibility module on the prompt path
- Removed a dead factory seam that could have reintroduced prompt-surface
  proliferation
- Simplified the prompt migration story ahead of the final wrapper removal

**Breaking changes:** Yes. `victor.agent.services.prompt_runtime_support` and
`OrchestratorFactory.create_prompt_runtime_support(...)` have been removed.

## Migration Update: SystemPromptCoordinator Removed (2026-05-05)

**Seam consolidated:** `SystemPromptCoordinator` remained as the last dead
prompt compatibility wrapper, and both `OrchestratorFactory` and
`CoordinatorFactory` still exposed matching creation methods even though the
live runtime had already converged on `UnifiedPromptPipeline` and the
state-passed prompt-classification seam.

**Canonical owners:**

- `victor.agent.prompt_pipeline.UnifiedPromptPipeline` remains the single live
  prompt assembly owner
- `victor.agent.services.prompt_runtime.PromptRuntimeAdapter` remains the DI
  implementation of `PromptRuntimeProtocol`
- `victor.agent.coordinators.SystemPromptStatePassedCoordinator` remains the
  canonical state-passed prompt-classification seam

**Changes applied:**

1. Deleted `victor.agent.services.system_prompt_runtime`.
2. Removed `SystemPromptCoordinator` re-exports from
   `victor.agent.services` and `victor.agent.coordinators`.
3. Removed `create_system_prompt_coordinator(...)` from the shared factory
   helper, `OrchestratorFactory`, and `CoordinatorFactory`.
4. Removed the dead `_system_prompt_coordinator` fallback branch from the
   orchestrator prompt event hook.
5. Added unit and integration coverage proving the removed module and factory
   methods no longer resolve while canonical prompt assembly and
   `SystemPromptStatePassedCoordinator` continue to work.

**Benefits:**

- Eliminated the last deprecated prompt wrapper from the runtime layer
- Removed dead factory surfaces that could have reintroduced prompt ownership
  ambiguity
- Left the prompt architecture with a single live assembly owner and one clear
  state-passed classification seam

**Breaking changes:** Yes. `victor.agent.services.system_prompt_runtime`,
`from victor.agent.services import SystemPromptCoordinator`,
`from victor.agent.coordinators import SystemPromptCoordinator`,
`OrchestratorFactory.create_system_prompt_coordinator(...)`, and
`CoordinatorFactory.create_system_prompt_coordinator(...)` have been removed.

## Migration Update: OrchestrationFacade Deprecated Coordinator Shims Removed (2026-05-05)

**Seam consolidated:** `OrchestrationFacade` still exposed deprecated
`chat_coordinator`, `tool_coordinator`, `session_coordinator`,
`sync_chat_coordinator`, `streaming_chat_coordinator`, and
`unified_chat_coordinator` properties even though the live runtime had already
converged on service-owned chat/tool/session surfaces plus state-passed
coordinators.

**Canonical owners:**

- `chat_service`, `tool_service`, and `session_service` remain the canonical
  orchestration runtime surfaces
- `exploration_state_passed`, `system_prompt_state_passed`,
  `safety_state_passed`, and `coordination_state_passed` remain the selective
  state-passed orchestration seams

**Changes applied:**

1. Removed the deprecated coordinator constructor inputs and properties from
   `victor.agent.facades.orchestration_facade.OrchestrationFacade`.
2. Removed the matching deprecated coordinator wiring from
   `AgentRuntimeBootstrapper.create_facades(...)`.
3. Removed bootstrapper placeholder initialization for the old sync/streaming/
   unified chat shim slots.
4. Replaced facade/bootstrapper tests with canonical-surface coverage and
   added integration coverage proving the removed properties stay absent.
5. Tightened the architecture boundary test so bootstrap wiring cannot
   reintroduce any deprecated coordinator inputs into `OrchestrationFacade`.

**Benefits:**

- Eliminated the last large deprecated coordinator surface inside the facade
  layer
- Made `OrchestrationFacade` match the target architecture directly: services
  plus selective state-passed seams only
- Reduced the risk of new internal code drifting back onto facade-owned
  coordinator compatibility names

**Breaking changes:** Yes. `OrchestrationFacade` no longer exposes
`chat_coordinator`, `tool_coordinator`, `session_coordinator`,
`sync_chat_coordinator`, `streaming_chat_coordinator`, or
`unified_chat_coordinator`.

## Migration Update: Deprecated Chat Shim Telemetry Removed (2026-05-05)

**Seam consolidated:** The runtime no longer had any live deprecated chat shim
surfaces, but `chat_compat_telemetry.py` and two orchestrator diagnostics
methods still remained as dead observability baggage for already-removed
compatibility names.

**Canonical owner:** None. This telemetry surface is removed rather than
replaced because the deprecated chat shim runtime path is already gone.

**Changes applied:**

1. Deleted `victor.agent.services.chat_compat_telemetry`.
2. Removed `AgentOrchestrator.get_deprecated_chat_compat_report()` and
   `AgentOrchestrator.has_deprecated_chat_compat_usage()`.
3. Replaced telemetry-focused unit tests with removal assertions.
4. Added a removal assertion to the shared compatibility-shim regression file.

**Benefits:**

- Eliminated dead post-migration reporting code
- Reduced the chance of future code reattaching itself to removed chat shim
  concepts through diagnostics-only APIs
- Tightened the architecture story: removed shims no longer keep leftover
  observability surfaces alive

**Breaking changes:** Yes. `victor.agent.services.chat_compat_telemetry`,
`AgentOrchestrator.get_deprecated_chat_compat_report()`, and
`AgentOrchestrator.has_deprecated_chat_compat_usage()` have been removed.

## Follow-up Work

1. ~~**Bridge-avoidance test naming**~~ - **DECIDED**: No renaming needed. Tests already use canonical method names. Old private wrappers have been removed. Test names accurately describe what they test (canonical API usage, compatibility alias behavior).
2. ~~**ToolRegistrar plugin-loading abstraction**~~ - **COMPLETED** (2026-05-04): Refactored `_load_plugin_tools()` to delegate to PluginLoader component. Improved SRP compliance by removing ~40 lines of implementation detail. ToolRegistrar is now a thinner facade.
3. ~~**Design long-term prompt end state**~~ - **IMPLEMENTED** (2026-05-05): Completed the convergence plan for aligning PromptRuntimeProtocol, prompt compatibility shims, and UnifiedPromptPipeline. See `docs/development/prompt-architecture-end-state-design.md`. Completed: PromptRuntimeAdapter now delegates its system-prompt build path through UnifiedPromptPipeline, `PromptRuntimeSupport` was removed, and `SystemPromptCoordinator` was removed. The canonical prompt surfaces are now `UnifiedPromptPipeline`, `PromptRuntimeAdapter`, and `SystemPromptStatePassedCoordinator`.
4. ~~**StateCoordinator retirement**~~ - **COMPLETED** (2026-05-04): Already retired to ConversationController ownership. See `docs/development/state-coordinator-retirement-analysis.md` for details. StateCoordinator class removed, only protocol alias remains for type checking. ConversationController + ConversationStateMachine + StateRuntimeAdapter provide canonical functionality.
   Note: StateRuntimeProtocol already uses StateRuntimeAdapter; protocol definition is service-native and does not alias StateCoordinatorProtocol.
5. ~~**External package compatibility validation**~~ - **COMPLETED**: Verified external verticals (victor-coding, victor-research, victor-invest) do not import from removed subservices (13/13 tests pass, zero impact on external packages)
6. ~~**Provider coordinator cleanup**~~ - **DECIDED** (2026-05-04): Remove in breaking release v1.0.0. See `docs/development/provider-coordinator-cleanup-proposal.md` for details. External packages do NOT use ProviderCoordinator or ProviderSwitchCoordinator. Safe to remove 1,184 LOC of dead code. ProviderService is canonical authority.
7. ~~**Chat/tool subservices removal**~~ - **COMPLETED**: Eliminated parallel architecture (4,367 LOC unused code: 3,444 from subservices + 923 from tests)
8. Keep future migration reporting evidence-based: exact counts, exact files,
   and explicit distinction between canonical services and compatibility wrappers.
