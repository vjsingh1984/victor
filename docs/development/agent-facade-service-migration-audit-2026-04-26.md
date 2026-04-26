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
| `victor/agent/services/state_compat.py` | 14,294 | `ConversationController` + `ConversationStateMachine` for live state; `victor.agent.state_service.StateService` for persisted vertical state only |
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
  `StateRuntimeAdapter` in DI; the deprecated `StateCoordinator` shim remains
  importable for public compatibility only

### Prompt surfaces

Prompt migration has two distinct canonical surfaces:

- `PromptRuntimeProtocol` now resolves through `PromptRuntimeAdapter` in DI for
  the narrower mutable prompt-runtime contract
- `UnifiedPromptPipeline` remains the canonical owner of orchestrator-managed
  live prompt assembly
- `SystemPromptCoordinator` is a compatibility wrapper around the live
  orchestrator prompt path, not the end-state prompt owner

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
- `StreamingChatPipeline` now prefers `orchestrator.create_recovery_context(...)`
  and falls back to `_create_recovery_context(...)` only for compatibility
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

- `StreamingChatPipeline` now requires `create_recovery_context(...)`
  directly and no longer falls back to `orchestrator._create_recovery_context(...)`
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

The public compatibility shim remains intentionally available, but it is no
longer the canonical live runtime implementation behind DI:

- `victor/agent/services/state_compat.py`: deprecated public shim
- `victor/agent/state_coordinator.py`: deprecated import path shim
- compatibility tests that assert those shims still import and warn correctly

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

## TDD Update: Canonical Prompt Runtime Support Fallback Surface

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

## Follow-up Work

1. Decide whether the remaining bridge-avoidance tests that mention old
   wrapper names should be renamed for clarity now that the production
   wrappers are gone.
2. Decide whether `ToolRegistrar._load_plugin_tools(...)` should remain as a
   private implementation detail or be refactored further behind a narrower
   plugin-loading abstraction.
3. Design the long-term prompt end state now that both the prompt DI path and
   the live fallback surface are canonicalized; the remaining gap is aligning
   `PromptRuntimeProtocol`, `PromptRuntimeSupport`, and `UnifiedPromptPipeline`
   around clearly separated responsibilities.
4. Decide whether `StateCoordinator` should eventually be retired into pure
   `ConversationController` ownership or replaced by a narrower state-passed
   boundary for conversation-stage transitions.
5. Keep future migration reporting evidence-based: exact counts, exact files,
   and explicit distinction between canonical services and compatibility
   wrappers.
