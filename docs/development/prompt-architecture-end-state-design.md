# Prompt Architecture End State Design

**Date:** 2026-05-04
**Status:** Implemented (2026-05-05)
**Related:** Agent Facade Service Migration Audit

## Problem Statement

Before the 2026-05-05 convergence cleanup, the prompt architecture had three
overlapping components that created confusion:

1. **PromptRuntimeAdapter** (`victor/agent/services/prompt_runtime.py`)
   - Implements `PromptRuntimeProtocol` for DI consumers
   - Provides narrow mutable prompt coordination contract
   - Handles task hints, grounding mode, safety rules
   - Now delegates system-prompt assembly through `UnifiedPromptPipeline`

2. **PromptRuntimeSupport** (`victor/agent/services/prompt_runtime_support.py`)
   - Removed on 2026-05-05
   - Previously acted as an internal fallback when UnifiedPromptPipeline was unavailable
   - Previously used in compatibility shims (SystemPromptCoordinator)
   - Overlapped with PromptRuntimeAdapter functionality
   - Added task classification, shell variant resolution

3. **UnifiedPromptPipeline** (`victor/agent/prompt_pipeline.py`)
   - Canonical owner of orchestrator-managed live prompt assembly
   - Handles provider tier detection, routing, freezing
   - GEPA optimization injection
   - Per-turn prefix composition

**Issues:**
- Unclear ownership boundaries between the three components
- PromptRuntimeAdapter and PromptRuntimeSupport have overlapping responsibilities
- Compatibility shims (SystemPromptCoordinator) create confusion
- No clear path to deprecation

## Proposed End State

### Single Canonical Owner: UnifiedPromptPipeline

**UnifiedPromptPipeline** becomes the single authority for all prompt assembly:

```
┌─────────────────────────────────────────────────────────────┐
│                  UnifiedPromptPipeline                       │
│                  (Single Authority)                          │
├─────────────────────────────────────────────────────────────┤
│ Responsibilities:                                            │
│ 1. Build system prompts (full assembly)                     │
│ 2. Compose turn prefixes (dynamic per-turn content)          │
│ 3. Manage provider tiers (A/B/C freezing strategies)         │
│ 4. Inject optimization (GEPA, credit, skills)                │
│ 5. Handle workspace switches (unfreeze)                       │
│ 6. Task classification (if task_analyzer available)           │
│ 7. Tool integration (if tool registry available)             │
└─────────────────────────────────────────────────────────────┘
```

### Protocol Layer: PromptRuntimeProtocol

**Purpose:** Narrow protocol for DI/runtime seams that need mutable prompt access

**When to use:** Service layer components that need to:
- Add task hints at runtime
- Modify grounding mode
- Add/remove safety rules
- Build prompts programmatically

**Implementation:** `PromptRuntimeAdapter` implements this protocol

```
Service Layer (ChatService, ToolService)
    ↓ (depends on)
PromptRuntimeProtocol (interface)
    ↓ (implemented by)
PromptRuntimeAdapter (delegates to)
    ↓
UnifiedPromptPipeline.build_system_prompt()
```

### Remove: PromptRuntimeSupport

**Rationale:**
- PromptRuntimeSupport was created as a fallback for compatibility shims
- With UnifiedPromptPipeline as single authority, no need for fallback
- SystemPromptCoordinator compatibility shim can directly use UnifiedPromptPipeline
- Removes ~200 LOC of duplicate/overlapping code

**Migration path:**
1. Update SystemPromptCoordinator to use UnifiedPromptPipeline directly
2. Remove PromptRuntimeSupport class
3. Update ComponentAssembler to always create UnifiedPromptPipeline
4. Remove _prompt_runtime_support from orchestrator

## Separation of Concerns

### PromptRuntimeProtocol (Interface)

**Responsibility:** Mutable prompt coordination for service layer

**Methods:**
- `build_system_prompt(context: PromptRuntimeContext) -> str`
- `add_task_hint(task_type: str, hint: str) -> None`
- `remove_task_hint(task_type: str) -> None`
- `add_section(name: str, content: str) -> None`
- `remove_section(name: str) -> None`
- `add_safety_rule(rule: str) -> None`
- `clear_safety_rules() -> None`
- `set_grounding_mode(mode: str) -> None`
- `set_base_identity(identity: str) -> None`

**Implementation:** PromptRuntimeAdapter (delegates to UnifiedPromptPipeline internally)

### UnifiedPromptPipeline (Implementation)

**Responsibility:** All prompt assembly logic

**Methods:**
- `build_system_prompt() -> str` (full assembly with freezing)
- `compose_turn_prefix() -> str` (dynamic per-turn content)
- `unfreeze() -> None` (workspace switch)
- `update_system_prompt() -> None` (query-specific refresh)

**Internal helpers:**
- Provider tier detection
- GEPA optimization injection
- Tool integration
- Task classification (delegates to TaskAnalyzer)

## Implementation Plan

### Phase 1: Consolidate PromptRuntimeAdapter (Backward Compatible) - Completed 2026-05-05

1. **Refactor PromptRuntimeAdapter** to delegate to UnifiedPromptPipeline
   - PromptRuntimeAdapter becomes a thin wrapper around UnifiedPromptPipeline
   - Implements PromptRuntimeProtocol by calling UnifiedPromptPipeline methods
   - No behavior changes, just delegation

2. **Update PromptRuntimeAdapter tests** to verify delegation
   - Test that PromptRuntimeAdapter calls UnifiedPromptPipeline
   - Verify protocol conformance maintained

### Phase 2: Migrate SystemPromptCoordinator - Completed 2026-05-05

1. **Update SystemPromptCoordinator** to use UnifiedPromptPipeline directly
   - Remove inheritance from PromptRuntimeSupport
   - Delegate all methods to UnifiedPromptPipeline
   - Maintain deprecation warnings

2. **Update ComponentAssembler**
   - Always create UnifiedPromptPipeline (no conditional fallback)
   - Remove PromptRuntimeSupport creation

3. **Reduce PromptRuntimeSupport to a thin compatibility wrapper**
   - Preserve deprecated import path only
   - Delegate shared behavior through `SystemPromptCoordinator`
   - Keep legacy task-analyzer method-name compatibility where still needed

### Phase 3: Remove PromptRuntimeSupport - Completed 2026-05-05

1. **Delete PromptRuntimeSupport class**
   - Delete `victor/agent/services/prompt_runtime_support.py`
   - Remove all imports

2. **Update tests**
   - Remove tests that directly test PromptRuntimeSupport
   - Add tests verifying UnifiedPromptPipeline is used

3. **Update documentation**
   - Clarify that UnifiedPromptPipeline is single authority
   - Document PromptRuntimeProtocol as service layer interface
   - Remove PromptRuntimeSupport references

## Benefits

1. **Clearer ownership:** UnifiedPromptPipeline is unambiguous single authority
2. **Reduced complexity:** Remove ~200 LOC of duplicate/overlapping code
3. **Better separation:** Protocol vs implementation is clearer
4. **Easier testing:** Single component to test for prompt assembly
5. **Clearer migration path:** Straightforward deprecation of compatibility shims

## Risks and Mitigations

### Risk: Breaking changes for external consumers

**Outcome:** Phase 3 intentionally introduced a breaking change by removing
`victor.agent.services.prompt_runtime_support` and
`OrchestratorFactory.create_prompt_runtime_support(...)`. The stable
service-owned runtime surface remains `PromptRuntimeProtocol` via
`PromptRuntimeAdapter`, and the only remaining deprecated prompt compatibility
wrapper is `SystemPromptCoordinator`.

### Risk: Performance regression from delegation layers

**Mitigation:** Delegation is thin (method forwarding). No significant overhead. Can measure with benchmarks if concerned.

### Risk: Compatibility shims break during migration

**Mitigation:** Incremental migration with tests at each phase. Keep SystemPromptCoordinator working throughout.

## Testing Strategy

1. **Phase 1:** Test PromptRuntimeAdapter delegation to UnifiedPromptPipeline
2. **Phase 2:** Test SystemPromptCoordinator uses UnifiedPromptPipeline correctly
3. **Phase 3:** Test PromptRuntimeSupport removal doesn't break functionality
4. **Integration tests:** Verify end-to-end prompt building works
5. **Performance tests:** Benchmark to ensure no regression

## Follow-up Work

After this design is implemented:

1. Keep current-state and migration-audit docs aligned if the remaining
   `SystemPromptCoordinator` compatibility wrapper changes.
2. Add or tighten guardrails before removing `SystemPromptCoordinator` in a
   later breaking-release batch.

## Open Questions

1. Should PromptRuntimeProtocol be moved to `victor.framework.*` for clearer public API?
2. Should we add a `PromptBuilder` protocol to abstract prompt building further?
3. Is task classification better placed in a separate service?

## Related Documentation

- Agent Facade Service Migration Audit (prompt sections)
- CLAUDE.md (System Prompt Scopes section)
- Prompt optimization architecture docs
